"""SAM 3.1 concept segmentation, in process: the engine ``serving.sam_server`` drives.

Uses the FAIR ``sam3`` package image model with detector weights from the
SAM 3.1 checkpoint. Every call is one ``model(batch)`` forward: the vision
encoder runs once per image and every concept decoder runs inside that same
forward (the ``examples/sam3_image_batched_inference.ipynb`` pattern).
"""

from __future__ import annotations

import os
import gc
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

#: The serve's fixed options: the score floor every detection meets, and the
#: square every image is resized to, SAM 3.1's encoder input (the client's
#: tiles are this size, so they go in 1:1).
SCORE_THRESHOLD = 0.3
RESOLUTION = 1008


@dataclass
class SamDetection:
    label: str
    score: float
    pixel_mask: np.ndarray
    pixel_bbox_rcrc: tuple[int, int, int, int]


@dataclass
class _LoadedSam:
    """Meta's sam3 image model and its Datapoint pipeline (transform,
    collate, postprocessor): every concept is a ``FindQueryLoaded`` in one
    Datapoint, so all of them run in one ``model(batch)`` forward."""

    model: Any
    transform: Any
    collate: Any
    postprocess: Any  # callable that takes (output, find_metadatas) -> list of dicts
    copy_to_device: Any
    device: str


_SAM_SINGLETON: _LoadedSam | None = None


def release() -> None:
    """Drop the resident model and hand its device memory back to the driver
    (on unified memory, a referenced model's allocations read as host memory
    gone)."""
    global _SAM_SINGLETON
    _SAM_SINGLETON = None
    gc.collect()
    import torch as _torch

    if _torch.cuda.is_available():
        _torch.cuda.synchronize()
        _torch.cuda.empty_cache()

# Distinct concept lists the text-encoder cache holds before it is cleared: a
# bound against a caller that never stops minting new lists.
_TEXT_MEMO_MAX = 32


def _ensure_sam(device: str) -> _LoadedSam:
    global _SAM_SINGLETON
    if _SAM_SINGLETON is not None:
        return _SAM_SINGLETON
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    import torch as _torch
    from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
    from sam3.train.transforms.basic_for_api import (
        ComposeAPI,
        NormalizeAPI,
        RandomResizeAPI,
        ToTensorAPI,
    )
    from sam3.train.data.collator import collate_fn_api
    from sam3.eval.postprocessors import PostProcessImage
    from sam3.model.utils.misc import copy_data_to_device

    print(
        f"[sam3_engine] building sam3 image model on {device}",
        flush=True,
    )
    checkpoint_path = os.environ.get("SAM3_CHECKPOINT") or None
    if checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf(version="sam3.1")
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
    )
    _dev = _torch.device(device)
    if _dev.type == "cuda" and any(p.device.type == "cpu" for p in model.parameters()):
        model.to(_dev)
    model.eval()

    # Disable the segmentation-head activation checkpoint at inference:
    # `SegmentationHead._embed_pixels` runs it on every forward (no
    # `self.training` guard), and with no backward pass it only adds an
    # RNG-state stash and a sync. The pixel decoder has nothing random, so
    # the output is identical. The class is named exactly, so a sam3 version
    # that renames it says so in a warning.
    try:
        from sam3.model.maskformer_segmentation import SegmentationHead

        n_off = 0
        for _m in model.modules():
            if isinstance(_m, SegmentationHead) and getattr(_m, "act_ckpt", False):
                _m.act_ckpt = False
                n_off += 1
        print(f"[sam3_engine] seg-head act_ckpt disabled on {n_off} module(s)", flush=True)
    except ImportError as exc:
        print(
            f"[sam3_engine] WARNING: could not import SegmentationHead "
            f"to disable inference act_ckpt ({exc}); leaving it on",
            flush=True,
        )

    _memoize_text_encoder(model)

    # Precision is bf16 autocast over fp32 master weights, the pattern the
    # model's internal `.to(torch.bfloat16)` calls expect.
    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=RESOLUTION,
                max_size=RESOLUTION,
                square=True,
                consistent_transform=False,
            ),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    postprocess = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        # The postprocessor filters by score BEFORE the full-resolution mask
        # upscale, so it never upsamples masks the engine would discard. Its
        # cut is strict ``>`` against ``SCORE_THRESHOLD - 1e-6`` where the
        # engine's own is ``>=``: a detection at the floor itself can drop;
        # every retained one is unchanged.
        detection_threshold=max(0.0, SCORE_THRESHOLD - 1e-6),
        to_cpu=False,
    )
    _SAM_SINGLETON = _LoadedSam(
        model=model,
        transform=transform,
        collate=collate_fn_api,
        postprocess=postprocess,
        copy_to_device=copy_data_to_device,
        device=device,
    )
    print("[sam3_engine] sam3 image model ready", flush=True)
    return _SAM_SINGLETON


def _memoize_text_encoder(model: Any) -> None:
    """Cache the text encoder's output on the concept phrases that produced it.

    ``VETextEncoder.forward(text: list[str], ...)`` is a pure function of the
    phrase list under ``eval`` and ``inference_mode``, and a mission asks for
    the same few lists over and over, so the same tensors come back every
    time. The tensors are cloned in and out so a downstream in-place write
    never reaches another call's copy. Bounded by ``_TEXT_MEMO_MAX`` and
    cleared wholesale. The class is named exactly, so a sam3 version that
    renames it says so in a warning.
    """
    try:
        from sam3.model.text_encoder_ve import VETextEncoder
    except ImportError as exc:
        print(
            f"[sam3_engine] WARNING: could not import VETextEncoder to memoize "
            f"the text encoder ({exc}); re-encoding concepts on every call",
            flush=True,
        )
        return

    n_wrapped = 0
    for module in model.modules():
        if not isinstance(module, VETextEncoder) or hasattr(module, "_maps_memo"):
            continue
        cache: dict[tuple, tuple] = {}
        real = module.forward

        def cached(text, input_boxes=None, device=None, _real=real, _cache=cache):
            # Only the plain-string path is a pure function of its arguments.
            # The other branch takes ALREADY-ENCODED tensors, which is what
            # this cache produces, so passing it through untouched is what
            # keeps the two from feeding each other.
            if input_boxes or not text or not isinstance(text[0], str):
                return _real(text, input_boxes, device)
            key = (tuple(text), str(device))
            hit = _cache.get(key)
            if hit is None:
                out = _real(text, input_boxes, device)
                if len(_cache) >= _TEXT_MEMO_MAX:
                    _cache.clear()
                _cache[key] = tuple(t.clone() for t in out)
                hit = _cache[key]
            return tuple(t.clone() for t in hit)

        module.forward = cached
        module._maps_memo = cache
        n_wrapped += 1

    print(
        f"[sam3_engine] text encoder memoized on {n_wrapped} module(s) "
        "(pure in the concept phrases)",
        flush=True,
    )


def _build_datapoint(rgb: np.ndarray, concepts: list[str]):
    """One Datapoint with one image and a text query per concept, and the
    query ids (1, 2, ...): each query's ``coco_image_id`` aligns its results
    back to its concept."""
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        FindQueryLoaded,
        Image as SAMImage,
        InferenceMetadata,
    )

    from PIL import Image

    h, w = rgb.shape[:2]
    qids = list(range(1, len(concepts) + 1))
    dp = Datapoint(find_queries=[], images=[SAMImage(data=Image.fromarray(rgb), objects=[], size=[h, w])])
    for concept, cid in zip(concepts, qids):
        dp.find_queries.append(
            FindQueryLoaded(
                query_text=concept,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=cid,
                    original_image_id=cid,
                    original_category_id=1,
                    # (H, W), the order the postprocessor unbinds it in;
                    # reversed, a non-square frame comes back transposed.
                    original_size=[h, w],
                    object_id=0,
                    frame_index=0,
                ),
            )
        )
    return dp, qids


def segment(
    rgb: np.ndarray,
    concepts: list[str],
    *,
    device: str = "cuda",
    min_area_px: int,
    max_per_image: int,
) -> list[SamDetection]:
    """Run SAM 3.1 once on one ``uint8`` RGB frame, one ``model(batch)``
    forward for every concept: the backend of ``serving.sam_server``'s
    ``/segment``. Detections come per concept, best score first; ``pixel_mask``
    is the full ``(H, W)`` frame mask and ``pixel_bbox_rcrc`` its inclusive
    extent."""
    if not concepts:
        return []
    import torch as _torch

    loaded = _ensure_sam(device)
    dp, qids = _build_datapoint(rgb, concepts)
    batch = loaded.collate([loaded.transform(dp)], dict_key="x")["x"]
    batch = loaded.copy_to_device(batch, _torch.device(loaded.device), non_blocking=True)
    with _torch.inference_mode(), _torch.autocast("cuda", dtype=_torch.bfloat16):
        out = loaded.model(batch)
        results = loaded.postprocess.process_results(out, batch.find_metadatas)

    # No per-concept cap: every same-class instance stays countable for the
    # reasoner ("the Nth car from the intersection"); only the total
    # ``max_per_image`` backstop bounds a pathological frame.
    dets: list[SamDetection] = []
    for concept, qid in zip(concepts, qids):
        rec = results[qid]
        scores_t = rec.get("scores")
        masks_t = rec.get("masks")
        if scores_t is None or masks_t is None or scores_t.numel() == 0:
            continue
        scores_t = scores_t.detach()
        sel_idx = _torch.nonzero(scores_t >= SCORE_THRESHOLD, as_tuple=False).flatten()
        order = _torch.argsort(scores_t.index_select(0, sel_idx), descending=True)
        sel_idx = sel_idx.index_select(0, order)
        sel_scores_t = scores_t.index_select(0, sel_idx)
        masks_sel_t = masks_t.detach().index_select(0, sel_idx)
        if masks_sel_t.ndim == 4:
            masks_sel_t = masks_sel_t[:, 0]
        masks_bool_t = masks_sel_t > 0
        keep_idx_t = _torch.nonzero(masks_bool_t.sum(dim=(1, 2)) >= min_area_px, as_tuple=False).flatten()
        if keep_idx_t.numel() == 0:
            continue
        masks_bool_t = masks_bool_t.index_select(0, keep_idx_t)
        sel_scores_t = sel_scores_t.index_select(0, keep_idx_t)

        rows_any_t = masks_bool_t.any(dim=2)
        cols_any_t = masks_bool_t.any(dim=1)
        H, W = masks_bool_t.shape[1], masks_bool_t.shape[2]
        # The four bbox bounds move to the host in one transfer: each
        # ``.cpu()`` is a CUDA sync.
        r_min, c_min, r_max, c_max = _torch.stack([
            rows_any_t.float().argmax(dim=1),
            cols_any_t.float().argmax(dim=1),
            (H - 1) - rows_any_t.flip(dims=[1]).float().argmax(dim=1),
            (W - 1) - cols_any_t.flip(dims=[1]).float().argmax(dim=1),
        ], dim=0).cpu().numpy()
        scores_np = sel_scores_t.to(_torch.float32).cpu().numpy()
        masks_np = masks_bool_t.cpu().numpy()
        for k in range(int(masks_np.shape[0])):
            dets.append(SamDetection(
                label=concept, score=float(scores_np[k]), pixel_mask=masks_np[k],
                pixel_bbox_rcrc=(int(r_min[k]), int(c_min[k]), int(r_max[k]), int(c_max[k])),
            ))
            if len(dets) >= max_per_image:
                return dets
    return dets
