"""
验证 monkey-patch 是否真的能让 detectron2 的 batched_nms 在 GPU 上跑通。

1. 不 patch 时，torchvision.ops.nms 在 CUDA tensor 上会 NotImplementedError
2. patch 后，同样调用应该返回 CUDA 上的 keep tensor
3. detectron2 的 batched_nms 也应该跟着通
"""
import sys
import traceback
import torch

print(f"[env] torch={torch.__version__}  cuda_avail={torch.cuda.is_available()}  n_gpu={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    print("[env] no CUDA, aborting")
    sys.exit(1)

# ── 1) unpatch: 复现原始报错 ──────────────────────────────────────────
print("\n=== Step 1: unpatched nms on CUDA (expected to FAIL) ===")
import torchvision
boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [50., 50., 60., 60.]], device='cuda')
scores = torch.tensor([0.9, 0.8, 0.7], device='cuda')
try:
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    print(f"  UNEXPECTED SUCCESS: keep={keep}")
except NotImplementedError as e:
    print(f"  FAIL as expected: NotImplementedError (first 100 chars): {str(e)[:100]}")
except Exception as e:
    print(f"  FAIL with different error: {type(e).__name__}: {str(e)[:200]}")

# ── 2) apply monkey-patch (跟 2D_spatial_eval.py 里的写法一致) ───────
print("\n=== Step 2: apply monkey-patch ===")
import torchvision.ops.boxes as _tvbox
_orig_nms = _tvbox.nms
def _nms_cpu(boxes, scores, iou_threshold):
    dev = boxes.device
    keep = _orig_nms(boxes.detach().cpu(), scores.detach().cpu(), iou_threshold)
    return keep.to(dev)
_tvbox.nms = _nms_cpu
torchvision.ops.nms = _nms_cpu
print("  monkey-patch applied")

# ── 3) patched: torchvision.ops.nms ──────────────────────────────────
print("\n=== Step 3: patched nms on CUDA (expected to PASS) ===")
try:
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    print(f"  OK: keep={keep}  device={keep.device}")
    assert keep.is_cuda, "keep should be on CUDA"
    print("  device assertion PASS")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
    sys.exit(2)

# ── 4) patched: torchvision.ops.batched_nms ──────────────────────────
print("\n=== Step 4: patched batched_nms on CUDA (expected to PASS) ===")
idxs = torch.tensor([0, 0, 1], device='cuda')
try:
    keep = torchvision.ops.batched_nms(boxes, scores, idxs, 0.5)
    print(f"  OK: keep={keep}  device={keep.device}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
    sys.exit(3)

# ── 5) detectron2.layers.batched_nms（detectron2 用的就是这个包装） ────
print("\n=== Step 5: detectron2.layers.batched_nms on CUDA (real path) ===")
try:
    from detectron2.layers import batched_nms as d2_batched_nms
    keep = d2_batched_nms(boxes, scores, idxs, 0.5)
    print(f"  OK: keep={keep}  device={keep.device}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:300]}")
    traceback.print_exc()
    sys.exit(4)

# ── 6) detectron2 完整前向：直接调 RPN 的 find_top_rpn_proposals ──────
# 这里跑真正报错的调用链，最能证明修好了
print("\n=== Step 6: detectron2 find_top_rpn_proposals (真实调用链) ===")
try:
    from detectron2.modeling.proposal_generator.proposal_utils import find_top_rpn_proposals
    from detectron2.structures import Boxes
    # 构造 minimal input，模拟 RPN 输出
    N, A, H, W = 1, 3, 4, 4  # batch=1, 3 anchors, 4x4 feature map
    proposals = [torch.rand(N, A * H * W, 4, device='cuda') * 100]
    pred_objectness_logits = [torch.rand(N, A * H * W, device='cuda')]
    image_sizes = [(100, 100)]
    keep = find_top_rpn_proposals(
        proposals, pred_objectness_logits, image_sizes,
        nms_thresh=0.7, pre_nms_topk=100, post_nms_topk=50, min_box_size=0.0, training=False,
    )
    print(f"  OK: got {len(keep)} instance objects")
    print(f"  keep[0].proposal_boxes.tensor.device = {keep[0].proposal_boxes.tensor.device}")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:400]}")
    traceback.print_exc()
    sys.exit(5)

print("\n============================================")
print("ALL PASS — monkey-patch works end-to-end")
print("============================================")
