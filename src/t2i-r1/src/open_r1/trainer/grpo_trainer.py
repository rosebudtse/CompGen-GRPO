# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
Two Forward Passes
'''

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from PIL import Image

import numpy as np
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (

    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,

    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from janus.models import MultiModalityCausalLM, VLChatProcessor
from utils.reward_hps import HPSv2
from utils.reward_gdino_enhanced import GDinoEnhanced
from utils.reward_vlm import VLMAttr, VLMOrm
import shutil

import copy
import re


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

GC_KWARGS = {"use_reentrant": False}  # ZeRO-2 compatible gradient checkpointing


class JanusT2IR1Trainer(Trainer):

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        attn_implementation: str = "flash_attention_2",
        script_args = None,
    ):
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
                **model_init_kwargs,
            )
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # 把 attn_implementation 同步到 language_model（Janus 的 vendored LlamaForCausalLM 在
        # __init__ 时已构造完，from_pretrained 的 attn_implementation 不会自动透传到这里）
        model.language_model.config._attn_implementation = attn_implementation
        # freeze all vision encoders
        for name, param in model.named_parameters():
            if name.startswith("vision_model") or name.startswith("aligner") or name.startswith("gen"):
                param.requires_grad = False

        # gradient checkpointing (use_reentrant=False avoids ZeRO-2 double-reduce assertion)
        model.language_model.config.use_cache = False
        model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=GC_KWARGS)

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled() and args.beta != 0:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True
            )
        elif peft_config is None and args.beta != 0:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class: VLChatProcessor = VLChatProcessor.from_pretrained(model_id)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str) and 'hps' in reward_func:
                reward_funcs[i] = HPSv2(args)
            elif isinstance(reward_func, str) and 'vlm_attr' in reward_func:
                reward_funcs[i] = VLMAttr(args)
            elif isinstance(reward_func, str) and 'vlm_orm' in reward_func:
                reward_funcs[i] = VLMOrm(args)
            elif isinstance(reward_func, str) and 'gdino' in reward_func:
                reward_funcs[i] = GDinoEnhanced(args)
            else:
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.new_generations_image = args.new_generations_image
        self.beta = args.beta

        model.warnings_issued["estimate_tokens"] = True

        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        if self.beta != 0:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        else:
            self.ref_model = None

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
            elif isinstance(reward_func, (HPSv2, GDinoEnhanced, VLMAttr, VLMOrm)):
                reward_func.load_to_device(self.accelerator.device)

        # load cot prompt
        with open(args.reasoning_prompt_path, 'r') as f:
            self.cot_prompt = f.read()

        self.user_end_token_id = self.processing_class.tokenizer.encode('\n')[1]
        self.image_start_token_id = self.processing_class.tokenizer.encode(self.processing_class.image_start_tag)[1]
        self.image_token_num_per_image = args.image_token_num_per_image
        self.cfg_weight = args.cfg_weight
        self.image_gen_temperature = 1
        self.img_size = args.img_size
        self.patch_size = args.patch_size
        self.max_textcot_length = args.max_textcot_length


    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    def _get_per_token_logps(self, model, input_embeds, text_ids, img_ids, attention_mask):
        def _get_per_token_logps_part(logits, input_ids):
            logits = logits[:, :-1, :]
            input_ids = input_ids[:, 1:]
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        if img_ids is not None:
            hidden_states = model.language_model(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True).hidden_states
            last_hidden_states = hidden_states[-1]
            image_logits = model.gen_head(last_hidden_states[:, -(img_ids.size(1)+1):, :])
            img_input_ids = torch.cat([img_ids.new_zeros(img_ids.size(0), 1), img_ids], dim=1)
            per_token_logps_img = _get_per_token_logps_part(image_logits, img_input_ids)
            return torch.cat([
                per_token_logps_img.new_zeros(
                    (per_token_logps_img.size(0), input_embeds.size(1) - per_token_logps_img.size(1) - 1)
                ),
                per_token_logps_img
            ], dim=1)
        else:
            hidden_states = model.language_model(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True).hidden_states
            last_hidden_states = hidden_states[-1]
            text_logits = model.language_model.lm_head(last_hidden_states)
            per_token_logps_text = _get_per_token_logps_part(text_logits, text_ids)
            return per_token_logps_text


    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            self.processing_class.apply_sft_template_for_multi_turn_prompts(
            conversations=prompt,
            sft_format=self.processing_class.sft_format,
            system_prompt="You are a helpful assistant that receives an image prompt and generate a visualization of the prompt.",
        ) for prompt in prompts]
        prompt_inputs = self.processing_class.tokenizer(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=True,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        torch.set_grad_enabled(False)
        # Generate completions for text cot
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            unwrapped_model.language_model.config.use_cache = False
            unwrapped_model.language_model.gradient_checkpointing_disable()

            prompt_ids = prompt_ids.repeat_interleave(self.num_generations, dim=0)
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            input_embeds = unwrapped_model.language_model.get_input_embeddings()(prompt_ids)

            if self.num_generations > 100:
                total_generations = []
                for i in range(prompt_ids.shape[0] // self.num_generations):
                    current_input_embeds = input_embeds[i*self.num_generations: (i+1)*self.num_generations]
                    current_attn_mask = prompt_mask[i*self.num_generations: (i+1)*self.num_generations]
                    prompt_completion_ids = unwrapped_model.language_model.generate(
                        inputs_embeds=current_input_embeds,
                        attention_mask=current_attn_mask,
                        pad_token_id=self.processing_class.tokenizer.eos_token_id,
                        bos_token_id=self.processing_class.tokenizer.bos_token_id,
                        eos_token_id=self.processing_class.tokenizer.eos_token_id,
                        max_new_tokens=self.max_completion_length,
                        do_sample=True,
                        use_cache=True,
                    )
                    total_generations.append(prompt_completion_ids)
                prompt_completion_ids = torch.cat(total_generations, dim=0)
            else:
                prompt_completion_ids = unwrapped_model.language_model.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=prompt_mask,
                    pad_token_id=self.processing_class.tokenizer.eos_token_id,
                    bos_token_id=self.processing_class.tokenizer.bos_token_id,
                    eos_token_id=self.processing_class.tokenizer.eos_token_id,
                    max_new_tokens=self.max_completion_length,
                    do_sample=True,
                    use_cache=True,
                )

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_ids

            if self.max_textcot_length is not None:
                prompt_completion_ids = prompt_completion_ids[:, -self.max_textcot_length :]

            completion_ids = prompt_completion_ids

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Calculate semantic-cot loss
        loss_dict = {}
        model.module.language_model.config.use_cache = False
        model.module.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=GC_KWARGS)
        torch.set_grad_enabled(True)
        prompt_all_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        input_embeds = model.module.language_model.get_input_embeddings()(prompt_all_ids)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        per_token_logps = self._get_per_token_logps(
            model=model.module,
            input_embeds=input_embeds,
            text_ids=prompt_all_ids,
            img_ids=None,
            attention_mask=attention_mask)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]
        with torch.inference_mode():
            if self.ref_model is not None:
                self.ref_model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=GC_KWARGS)
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    input_embeds,
                    prompt_all_ids,
                    None,
                    attention_mask)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
            else:
                ref_per_token_logps = torch.zeros_like(per_token_logps)
        loss_dict['semantic-cot'] = {
            'per_token_logps': per_token_logps,
            'ref_per_token_logps': ref_per_token_logps,
            'completion_mask': completion_mask,
        }
        torch.set_grad_enabled(False)

        image_gen_prompt_list = []
        for i in range(completion_ids.shape[0]):
            answer = self.processing_class.tokenizer.decode(completion_ids[i].cpu().tolist(), skip_special_tokens=True)
            raw_prompt = inputs[i // self.num_generations]['raw_prompt']
            image_gen_prompt = f"{raw_prompt}. {answer}"

            conversation = [
                {
                    "role": "User",
                    "content": image_gen_prompt,
                },
                {"role": "Assistant", "content": ""},
            ]
            sft_format = self.processing_class.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.processing_class.sft_format,
                system_prompt="",
            )
            image_gen_prompt_list.append(sft_format)

        prompt_inputs = self.processing_class.tokenizer(
            text=image_gen_prompt_list,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=True,
        )

        prompt_ids, attention_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_ids = prompt_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        prompt_ids = torch.cat([prompt_ids, prompt_ids.new_full((prompt_ids.size(0), 1), self.image_start_token_id)], dim=1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.size(0), 1))], dim=1)

        prompt_ids = prompt_ids.repeat_interleave(self.new_generations_image, dim=0)
        attention_mask = attention_mask.repeat_interleave(self.new_generations_image, dim=0)

        # Generate the image tokens
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            unwrapped_model.language_model.config.use_cache = False
            unwrapped_model.language_model.gradient_checkpointing_disable()

            inputs_embeds = unwrapped_model.language_model.get_input_embeddings()(prompt_ids)
            pad_input_embeds = unwrapped_model.language_model.get_input_embeddings()(prompt_ids.new_full((1, 1), self.processing_class.pad_id))
            total_generated_tokens_img = []

            cond_inputs_embeds = inputs_embeds
            cond_attention_mask = attention_mask
            uncond_inputs_embeds = cond_inputs_embeds.clone()
            uncond_inputs_embeds[:, 1:-1] = pad_input_embeds

            inputs_embeds_img = torch.repeat_interleave(cond_inputs_embeds, 2, dim=0)
            inputs_embeds_img[1::2] = uncond_inputs_embeds
            attention_mask_img = torch.repeat_interleave(cond_attention_mask, 2, dim=0)
            attention_mask_img[1::2] = torch.ones_like(attention_mask_img[1::2])

            split_size = 32
            for jj in range(0, inputs_embeds_img.shape[0], split_size):
                print(f"Generating image {jj}")
                start = jj
                end = min(jj + split_size, inputs_embeds_img.shape[0])
                generated_tokens = torch.zeros(((end-start)//2, self.image_token_num_per_image), dtype=torch.int64).cuda()
                cur_inputs_embeds_img = inputs_embeds_img[start: end]
                cur_attention_mask_img = attention_mask_img[start: end]

                for k in range(self.image_token_num_per_image):
                    outputs = unwrapped_model.language_model.model(
                        inputs_embeds=cur_inputs_embeds_img,
                        use_cache=True,
                        past_key_values=outputs.past_key_values if k != 0 else None,
                        attention_mask=cur_attention_mask_img
                    )

                    hidden_states = outputs.last_hidden_state
                    logits = unwrapped_model.gen_head(hidden_states[:, -1, :])
                    logit_cond = logits[0::2, :]
                    logit_uncond = logits[1::2, :]

                    logits = logit_uncond + self.cfg_weight * (logit_cond-logit_uncond)
                    probs = torch.softmax(logits / self.image_gen_temperature, dim=-1)

                    next_token = torch.multinomial(probs, num_samples=1)
                    generated_tokens[:, k] = next_token.squeeze(dim=-1)

                    next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                    img_embeds = unwrapped_model.prepare_gen_img_embeds(next_token)
                    cur_inputs_embeds_img = img_embeds.unsqueeze(dim=1)
                    cur_attention_mask_img = torch.cat([cur_attention_mask_img, cur_attention_mask_img.new_ones((cur_attention_mask_img.shape[0], 1), dtype=torch.int)], dim=1)

                    del logits, probs, logit_cond, logit_uncond, hidden_states, next_token, img_embeds

                total_generated_tokens_img.append(generated_tokens)
        total_generated_tokens_img = torch.cat(total_generated_tokens_img, dim=0)

        # Calculate token-cot loss
        model.module.language_model.config.use_cache = False
        model.module.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=GC_KWARGS)
        torch.set_grad_enabled(True)
        input_embeds = torch.cat(
            [
                model.module.language_model.get_input_embeddings()(prompt_ids),
                model.module.prepare_gen_img_embeds(total_generated_tokens_img)
            ],
            dim=1
        )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones_like(total_generated_tokens_img)
            ],
            dim=1
        )

        per_token_logps = self._get_per_token_logps(
            model=model.module,
            input_embeds=input_embeds,
            text_ids=None,
            img_ids=total_generated_tokens_img,
            attention_mask=attention_mask
        )
        prompt_length = prompt_ids.size(1)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]
        completion_mask = torch.ones_like(total_generated_tokens_img)

        with torch.inference_mode():
            if self.ref_model is not None:
                self.ref_model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=GC_KWARGS)
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    input_embeds=input_embeds,
                    text_ids=None,
                    img_ids=total_generated_tokens_img,
                    attention_mask=attention_mask
                )
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
            else:
                ref_per_token_logps = torch.zeros_like(per_token_logps)
        loss_dict['token-cot'] = {
            'per_token_logps': per_token_logps,
            'ref_per_token_logps': ref_per_token_logps,
            'completion_mask': completion_mask,
        }
        torch.set_grad_enabled(False)

        total_generated_tokens_img = total_generated_tokens_img.detach()

        # Generate the image
        with unwrap_model_for_generation(model.module.gen_vision_model, self.accelerator) as unwrapped_model:
            dec = unwrapped_model.decode_code(total_generated_tokens_img.to(dtype=torch.int), shape=[total_generated_tokens_img.shape[0], 8, self.img_size//self.patch_size, self.img_size//self.patch_size])
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            dec = np.clip((dec + 1) / 2 * 255, 0, 255)
            visual_img = np.zeros((total_generated_tokens_img.shape[0], self.img_size, self.img_size, 3), dtype=np.uint8)
            visual_img[:, :, :] = dec
            images = [Image.fromarray(visual_img[idx]) for idx in range(visual_img.shape[0])]

        # Compute the rewards
        prompts = [input["raw_prompt"] for input in inputs for _ in range(self.num_generations) for __ in range(self.new_generations_image)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations * self.new_generations_image)
                output_reward_func = reward_func(prompts=prompts, images=images, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        torch.cuda.empty_cache()
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations * self.new_generations_image).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations * self.new_generations_image).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations * self.new_generations_image, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations * self.new_generations_image, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        torch.set_grad_enabled(True)

        # Calculate the loss together for semantic-cot and token-cot
        for key in loss_dict['semantic-cot']:
            loss_dict['semantic-cot'][key] = loss_dict['semantic-cot'][key].repeat_interleave(self.new_generations_image, dim=0)
        per_token_logps, ref_per_token_logps, completion_mask = [], [], []
        for key in ['semantic-cot', 'token-cot']:
            if loss_dict[key]['per_token_logps'] is None:
                loss_dict[key]['loss'] = None
                continue
            per_token_logps.append(loss_dict[key]['per_token_logps'])
            ref_per_token_logps.append(loss_dict[key]['ref_per_token_logps'])
            completion_mask.append(loss_dict[key]['completion_mask'])
        per_token_logps = torch.cat(per_token_logps, dim=1)
        ref_per_token_logps = torch.cat(ref_per_token_logps, dim=1)
        completion_mask = torch.cat(completion_mask, dim=1)

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
        self._metrics[f"kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics[f"loss"].append(self.accelerator.gather_for_metrics(loss.detach()).mean().item())

        completion_length = self.accelerator.gather_for_metrics(loss_dict['semantic-cot']['completion_mask'].sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))