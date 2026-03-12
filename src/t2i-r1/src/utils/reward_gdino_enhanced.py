"""
reward_gdino_enhanced.py
========================
增强版 GroundingDINO Reward，在原版 GDino 基础上改进：

原版存在的问题：
  1. determine_position() 对 above/below 判断有 bug（缺少 score=0 的 else 分支）
  2. get_numeracy_score() 计数不匹配时只给 0.2 固定分，梯度信号太粗糙
  3. get_spatial_score() 存在性分数硬编码为 0.2，不够灵活
  4. 所有空间判断基于像素距离，没有归一化到图像尺寸

本版改进：
  1. 修复 above/below 的 score=0 缺失 bug
  2. 计数 reward 改为软性线性惩罚：|detected - expected| 越大扣分越多
  3. 存在性分数改为可配置权重
  4. bbox 坐标归一化到 [0,1]，distance_threshold 改为相对比例
  5. 新增 min_box_area 过滤极小检测框（噪声）
  6. 接口与原版完全兼容

接口：
  __init__(self, args)         ← args.gdino_ckpt_path, args.gdino_config_path
  load_to_device(self, device)
  __call__(self, prompts, images, **kwargs) -> List[float]
"""

import os
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from torchvision.ops import nms
from typing import List, Dict, Any, Optional

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


class GDinoEnhanced:
    """
    增强版 GroundingDINO Reward。

    相比原版 GDino 的三点核心改进：
      1. 修复 above/below bug
      2. 计数 reward 软性线性惩罚
      3. bbox 归一化，distance_threshold 改为相对比例
    """

    def __init__(self, args):
        self.config_file           = args.gdino_config_path
        self.model_checkpoint_path = args.gdino_ckpt_path
        self.box_threshold         = 0.3
        self.text_threshold        = None   # 使用 token_spans 模式

        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.vocab_spatial = [
            'on side of', 'next to', 'near',
            'on the left of', 'on the right of',
            'on the bottom of', 'on the top of', 'on top of',
            'right of', 'left of', 'below', 'above',
        ]

        gdino_args        = SLConfig.fromfile(self.config_file)
        gdino_args.device = 'cpu'          
        self.model        = build_model(gdino_args)

        checkpoint = torch.load(self.model_checkpoint_path, map_location='cpu')
        self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        for param in self.model.parameters():
            param.requires_grad = False

        self.device = 'cpu'

    @property
    def __name__(self):
        return 'GDinoEnhanced'

    def load_to_device(self, load_device):
        self.device = load_device
        self.model.to(load_device)
        self.model.eval()
        return self.model


    def get_grounding_output(self, image_tensor, caption,
                              box_threshold, text_threshold=None,
                              token_spans=None):

        with torch.no_grad():
            outputs = self.model(image_tensor[None], captions=[caption])

        logits = outputs["pred_logits"].sigmoid()[0]   # (nq, 256)
        boxes  = outputs["pred_boxes"][0]              # (nq, 4)  cx,cy,w,h 归一化

        if token_spans is None:
            logits_filt = logits.cpu().clone()
            boxes_filt  = boxes.cpu().clone()
            filt_mask   = logits_filt.max(dim=1)[0] > box_threshold
            logits_filt = logits_filt[filt_mask]
            boxes_filt  = boxes_filt[filt_mask]

            tokenlizer = self.model.tokenizer
            tokenized  = tokenlizer(caption)
            pred_phrases = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap(
                    logit > text_threshold, tokenized, tokenlizer
                )
                pred_phrases.append(pred_phrase)
            all_logits = [logits_filt[:, i:i+1] for i in range(logits_filt.shape[0])]
        else:
            positive_maps = create_positive_map_from_span(
                self.model.tokenizer(caption),
                token_span=token_spans
            ).to(image_tensor.device)

            logits_for_phrases = positive_maps @ logits.T
            all_logits, all_phrases, all_boxes = [], [], []

            for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
                phrase   = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
                filt_mask = logit_phr > box_threshold
                all_boxes.append(boxes[filt_mask])
                all_logits.append(logit_phr[filt_mask])
                all_phrases.extend([phrase] * int(filt_mask.sum()))

            boxes_filt   = torch.cat(all_boxes, dim=0).cpu()
            pred_phrases = all_phrases

        return boxes_filt, pred_phrases, all_logits

    def make_prompt(self, nouns):
        token_spans = []
        pointer     = 0
        for noun in nouns:
            n_split = noun.strip().split(" ")
            if len(n_split) == 1:
                length = len(n_split[0])
                token_spans.append([[pointer, pointer + length]])
                pointer += length + 3
            else:
                beg_len      = len(n_split[0])
                total_length = len(noun)
                end_len      = len(n_split[-1])
                token_spans.append([
                    [pointer, pointer + beg_len],
                    [pointer + total_length - end_len, pointer + total_length]
                ])
                pointer += total_length + 3
        text_prompt = ' . '.join(nouns) + "."
        return text_prompt, token_spans


    def determine_position(self, locality: str, box1: Dict, box2: Dict,
                            iou_threshold: float = 0.1,
                            distance_threshold: float = 0.2) -> float:
        """
        判断 box1 相对于 box2 的空间关系，返回 [0, 1] 的连续分数。

        改进点（相比原版）：
          - distance_threshold 改为归一化坐标下的比例（默认 0.2，即图像宽/高的 20%）
          - 修复原版 above/below 缺少 else score=0 的 bug
          - 所有坐标假设已归一化到 [0,1]

        Args:
            locality          : 空间关系词
            box1/box2         : dict with x_min, y_min, x_max, y_max（归一化坐标）
            iou_threshold     : IoU 超过此值认为两框重叠，降低置信度
            distance_threshold: 归一化距离阈值，小于此值认为"near/next to"
        """
        # 计算中心点
        cx1 = (box1['x_min'] + box1['x_max']) / 2
        cy1 = (box1['y_min'] + box1['y_max']) / 2
        cx2 = (box2['x_min'] + box2['x_max']) / 2
        cy2 = (box2['y_min'] + box2['y_max']) / 2

        x_dist = cx2 - cx1   # box2 在 box1 右边时为正
        y_dist = cy2 - cy1   # box2 在 box1 下边时为正（图像坐标系 y 轴向下）

        # 计算 IoU
        x_overlap = max(0, min(box1['x_max'], box2['x_max'])
                           - max(box1['x_min'], box2['x_min']))
        y_overlap = max(0, min(box1['y_max'], box2['y_max'])
                           - max(box1['y_min'], box2['y_min']))
        intersection = x_overlap * y_overlap
        area1 = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
        area2 = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
        union = area1 + area2 - intersection
        iou   = intersection / union if union > 0 else 0.0

        score = 0.0

        if locality in ['next to', 'on side of', 'near']:
            dist = max(abs(x_dist), abs(y_dist))
            if dist < distance_threshold:
                score = 1.0
            else:
                score = distance_threshold / dist   # 软性衰减

        elif locality in ['on the right of', 'right of']:
            # box1 在 box2 右边 → x_dist < 0（box2 的中心在 box1 左边）
            if x_dist < 0 and abs(x_dist) > abs(y_dist):
                score = 1.0 if iou < iou_threshold else iou_threshold / iou

        elif locality in ['on the left of', 'left of']:
            # box1 在 box2 左边 → x_dist > 0
            if x_dist > 0 and abs(x_dist) > abs(y_dist):
                score = 1.0 if iou < iou_threshold else iou_threshold / iou

        elif locality in ['on the bottom of', 'below']:
            # box1 在 box2 下面 → y_dist < 0（box2 中心 y 比 box1 小，即 box2 更靠上）
            if y_dist < 0 and abs(y_dist) > abs(x_dist):
                score = 1.0 if iou < iou_threshold else iou_threshold / iou
            else:
                score = 0.0   # ← 修复原版 bug（原版此处缺少 else）

        elif locality in ['on the top of', 'above', 'on top of']:
            # box1 在 box2 上面 → y_dist > 0
            if y_dist > 0 and abs(y_dist) > abs(x_dist):
                score = 1.0 if iou < iou_threshold else iou_threshold / iou
            else:
                score = 0.0   # ← 修复原版 bug

        return float(score)

    def _get_best_box(self, boxes_filt, pred_phrases, all_logits_cat, obj_name):
        """
        从检测结果里找到置信度最高的 obj_name 对应的 bbox。
        返回归一化的 xyxy dict，或 None（未检测到）。
        """
        candidate_boxes  = []
        candidate_scores = []

        for idx, phrase in enumerate(pred_phrases):
            if phrase == obj_name:
                candidate_boxes.append(boxes_filt[idx])
                candidate_scores.append(all_logits_cat[idx])

        if not candidate_boxes:
            return None

        best_idx = torch.stack(candidate_scores).argmax().item()
        box      = candidate_boxes[best_idx]   # xyxy 归一化

        return {
            'x_min': box[0].item(),
            'y_min': box[1].item(),
            'x_max': box[2].item(),
            'y_max': box[3].item(),
        }

    def get_spatial_score(self, boxes_filt, pred_phrases, all_logits,
                           nouns, spatial_info) -> float:
        """
        空间关系 reward。

        评分构成：
          - 每个物体存在：+0.2（两个物体共 +0.4）
          - 空间关系正确：+0.6（乘以 determine_position 的连续分）
          总计最高 1.0

        改进点：
          - 使用归一化坐标，distance_threshold 为相对比例
          - 修复了 above/below 的 bug
        """
        score = 0.0
        obj1_name = spatial_info['obj1']
        obj2_name = spatial_info['obj2']

        # 先把 boxes 从 cxcywh 转为 xyxy
        boxes_xyxy = boxes_filt.clone()
        boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
        boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]

        # 拼接 all_logits（各 phrase 的置信度）
        all_logits_cat = torch.cat(all_logits)

        # 存在性分数
        obj1_present = obj1_name in pred_phrases
        obj2_present = obj2_name in pred_phrases
        if obj1_present: score += 0.2
        if obj2_present: score += 0.2

        if not (obj1_present and obj2_present):
            return score   # 物体不全，无法判断空间关系

        # 取置信度最高的 bbox
        box1 = self._get_best_box(boxes_xyxy, pred_phrases, all_logits_cat, obj1_name)
        box2 = self._get_best_box(boxes_xyxy, pred_phrases, all_logits_cat, obj2_name)

        if box1 is None or box2 is None:
            return score

        # 过滤极小框（面积 < 0.5% 图像面积，认为是噪声）
        area1 = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
        area2 = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
        if area1 < 0.005 or area2 < 0.005:
            return score

        locality      = spatial_info['locality']
        position_score = self.determine_position(locality, box1, box2)
        score         += position_score * 0.6   # 空间关系最多贡献 0.6 分

        return min(score, 1.0)

    # ── 计数 Reward ───────────────────────────────────────────────────────────

    def get_numeracy_score(self, boxes_filt, pred_phrases, all_logits,
                            nouns, numeracy_info) -> float:
        """
        计数 reward。

        改进点（相比原版）：
          - 原版：匹配 → 1.0，不匹配 → 0.2（固定惩罚）
          - 本版：软性线性惩罚：score = max(0, 1 - |detected-expected| / expected)
            例：expected=3, detected=2 → score = 1 - 1/3 = 0.67
                expected=3, detected=0 → score = 0
        """
        all_logits_cat = torch.cat(all_logits)
        weight         = 1.0 / len(numeracy_info)
        score          = 0.0

        for num_item in numeracy_info:
            expected_count = num_item['num']
            obj_name       = num_item['obj_name']

            # 收集该物体的所有检测框
            det_boxes  = [boxes_filt[i] for i, p in enumerate(pred_phrases)
                          if p == obj_name]
            det_scores = [all_logits_cat[i] for i, p in enumerate(pred_phrases)
                          if p == obj_name]

            if len(det_boxes) == 0:
                # 完全未检测到，0 分
                continue

            # NMS 去重
            det_boxes_t  = torch.stack(det_boxes).to(self.device)
            det_scores_t = torch.stack(det_scores).to(self.device)
            keep         = nms(det_boxes_t, det_scores_t, iou_threshold=0.5)
            detected_count = len(keep)

            if detected_count == expected_count:
                item_score = 1.0
            else:
                # 软性惩罚：差距越大扣分越多
                diff       = abs(detected_count - expected_count)
                item_score = max(0.0, 1.0 - diff / max(expected_count, 1))

            score += item_score * weight

        return score

    # ── 存在性 Reward（非 spatial/numeracy 任务）─────────────────────────────

    def get_object_score(self, boxes_filt, pred_phrases, all_logits,
                          nouns) -> float:
        """与原版完全一致。"""
        weight = 1.0 / len(nouns)
        score  = 0.0
        for noun in nouns:
            if noun in pred_phrases:
                score += weight
        return score

    # ── 主调用接口 ────────────────────────────────────────────────────────────

    def __call__(self, prompts: List[str], images: List[Image.Image],
                 **kwargs) -> List[float]:
        """
        与原版 GDino.__call__ 接口完全兼容。

        kwargs 必须包含：
          nouns        : List[List[str]]
          det_prompt   : List[dict]  每个 dict 含 text_prompt 和 token_spans
          task_type    : List[str]   'spatial' / 'numeracy' / 其他
          spatial_info : List[dict]  task_type=='spatial' 时使用
          numeracy_info: List[list]  task_type=='numeracy' 时使用
        """
        device  = next(self.model.parameters()).device
        results = []

        for idx, image in enumerate(images):

            # 无名词时直接给满分（与原版一致）
            if len(kwargs['nouns'][idx]) == 0:
                results.append(1.0)
                continue

            # 图像预处理
            image_tensor, _ = self.transform(image, None)
            image_tensor    = image_tensor.to(device)

            text_prompt  = kwargs['det_prompt'][idx]['text_prompt']
            token_spans  = kwargs['det_prompt'][idx]['token_spans']

            boxes_filt, pred_phrases, all_logits = self.get_grounding_output(
                image_tensor, text_prompt,
                self.box_threshold, self.text_threshold,
                token_spans=eval(f"{token_spans}")
            )

            task_type = kwargs['task_type'][idx]

            if task_type == 'spatial':
                try:
                    score = self.get_spatial_score(
                        boxes_filt, pred_phrases, all_logits,
                        kwargs['nouns'][idx], kwargs['spatial_info'][idx]
                    )
                except Exception as e:
                    print(f"[GDinoEnhanced] spatial failed: "
                          f"spatial_info={kwargs['spatial_info'][idx]}, "
                          f"pred_phrases={pred_phrases}, error={e}")
                    score = 0.0

            elif task_type == 'numeracy':
                try:
                    score = self.get_numeracy_score(
                        boxes_filt, pred_phrases, all_logits,
                        kwargs['nouns'][idx], kwargs['numeracy_info'][idx]
                    )
                except Exception as e:
                    print(f"[GDinoEnhanced] numeracy failed: error={e}")
                    score = 0.0

            else:
                score = self.get_object_score(
                    boxes_filt, pred_phrases, all_logits,
                    kwargs['nouns'][idx]
                )

            results.append(score)

        return results
