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
        print(
            f"[sam3_engine] seg-head act_ckpt disabled on {n_off} module(s) "
            "(inference: no backward, checkpoint is pure overhead)",
            flush=True,
        )
    except ImportError as exc:
        print(
            f"[sam3_engine] WARNING: could not import SegmentationHead "
            f"to disable inference act_ckpt ({exc}); leaving it on",
            flush=True,
        )

    _memoize_text_encoder(model)

    # Precision is bf16 autocast over fp32 master weights, the pattern the
    # model's internal `.to(torch.bfloat16)` calls expect. On an L40S,
    # torch.compile of the vision backbone moved detections past the parity
    # gate, and torchao weight quantization does not dispatch on the
    # backbone's custom ops; any speed change passes that gate before it ships.
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
        "(pure in the concept phrases; ~13% of a call)",
        flush=True,
    )


def _build_datapoint(rgb: np.ndarray, concepts: list[str], dp_counter: list[int]):
    """One Datapoint with one image and a text query per concept.
    ``dp_counter`` is a one-element list holding the next query id; each
    query's ``coco_image_id`` aligns its results back to its concept."""
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        FindQueryLoaded,
        Image as SAMImage,
        InferenceMetadata,
    )

    from PIL import Image

    img_pil = Image.fromarray(rgb)
    dp = Datapoint(find_queries=[], images=[])
    h, w = rgb.shape[:2]
    dp.images = [SAMImage(data=img_pil, objects=[], size=[h, w])]
    qids: list[int] = []
    for concept in concepts:
        cid = dp_counter[0]
        dp_counter[0] += 1
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
        qids.append(cid)
    return dp, qids


def segment(
    rgb: np.ndarray,
    concepts: list[str],
    *,
    device: str = "cuda",
    min_area_px: int,
    max_per_image: int,
) -> list[SamDetection]:
    """Run SAM 3.1 once on one ``uint8`` RGB frame: the backend of
    ``serving.sam_server``'s ``/segment``. ``pixel_mask`` is the full
    ``(H, W)`` frame mask and ``pixel_bbox_rcrc`` its inclusive extent."""
    if not concepts:
        return []
    return _segment_via_meta(
        [rgb],
        concepts,
        device=device,
        min_area_px=min_area_px,
        max_per_image=max_per_image,
    )[0]


def _segment_via_meta(
    rgb_tiles: list[np.ndarray],
    concepts: list[str],
    *,
    device: str,
    min_area_px: int,
    max_per_image: int,
) -> list[list[SamDetection]]:
    """Segment ``rgb_tiles`` through the Datapoint pipeline: a list aligned
    with ``rgb_tiles`` of per-concept-ordered detections. Tiles run in
    collated forwards of ``SAM_GPU_CHUNK`` (env, default 8), which bounds the
    per-forward peak memory; per-image outputs do not depend on the chunk."""
    import torch as _torch

    loaded = _ensure_sam(device)
    n_tiles = len(rgb_tiles)
    concepts_per_tile = [concepts] * n_tiles

    gpu_chunk = max(1, int(os.environ.get("SAM_GPU_CHUNK", "8")))
    dp_counter = [1]
    detections_per_tile: list[list[SamDetection]] = []
    for start in range(0, n_tiles, gpu_chunk):
        rgb_chunk = rgb_tiles[start : start + gpu_chunk]
        concepts_chunk = concepts_per_tile[start : start + gpu_chunk]
        detections_per_tile.extend(
            _forward_chunk(
                loaded,
                rgb_chunk,
                concepts_chunk,
                dp_counter,
                _torch,
                min_area_px=min_area_px,
                max_per_image=max_per_image,
            )
        )
    return detections_per_tile


def _forward_chunk(
    loaded: _LoadedSam,
    rgb_chunk: list[np.ndarray],
    concepts_chunk: list[list[str]],
    dp_counter: list[int],
    _torch: Any,
    *,
    min_area_px: int,
    max_per_image: int,
) -> list[list[SamDetection]]:
    """One collated ``model(batch)`` forward over one GPU chunk of tiles.
    ``dp_counter`` stays monotonic across chunks so query ids are unique."""
    dps = []
    qids_per_tile: list[list[int]] = []
    for rgb, tile_concepts in zip(rgb_chunk, concepts_chunk):
        dp, qids = _build_datapoint(rgb, tile_concepts, dp_counter)
        dps.append(loaded.transform(dp))
        qids_per_tile.append(qids)

    batch = loaded.collate(dps, dict_key="x")["x"]
    batch = loaded.copy_to_device(
        batch, _torch.device(loaded.device), non_blocking=True
    )

    with _torch.inference_mode(), _torch.autocast("cuda", dtype=_torch.bfloat16):
        out = loaded.model(batch)
        results = loaded.postprocess.process_results(out, batch.find_metadatas)

    detections_per_tile: list[list[SamDetection]] = []
    for tile_concepts, qids in zip(concepts_chunk, qids_per_tile):
        # No per-concept cap: every same-class instance stays countable for
        # the reasoner ("the Nth car from the intersection"); only the total
        # ``max_per_image`` backstop bounds a pathological frame.
        per_tile: list[SamDetection] = []
        for concept, qid in zip(tile_concepts, qids):
            rec = results[qid]
            scores_t = rec.get("scores")
            masks_t = rec.get("masks")
            if scores_t is None or masks_t is None or scores_t.numel() == 0:
                continue
            scores_t = scores_t.detach()
            keep_score_t = scores_t >= SCORE_THRESHOLD
            sel_idx = _torch.nonzero(keep_score_t, as_tuple=False).flatten()
            sel_scores_t = scores_t.index_select(0, sel_idx)
            order = _torch.argsort(sel_scores_t, descending=True)
            sel_idx = sel_idx.index_select(0, order)
            sel_scores_t = sel_scores_t.index_select(0, order)

            masks_t_sel = masks_t.detach().index_select(0, sel_idx)
            if masks_t_sel.ndim == 4:
                masks_t_sel = masks_t_sel[:, 0]
            masks_bool_t = masks_t_sel > 0
            areas_t = masks_bool_t.sum(dim=(1, 2))
            keep_area_t = areas_t >= min_area_px
            keep_idx_t = _torch.nonzero(keep_area_t, as_tuple=False).flatten()
            if keep_idx_t.numel() == 0:
                continue
            masks_bool_t = masks_bool_t.index_select(0, keep_idx_t)
            sel_scores_t = sel_scores_t.index_select(0, keep_idx_t)

            rows_any_t = masks_bool_t.any(dim=2)
            cols_any_t = masks_bool_t.any(dim=1)
            H = masks_bool_t.shape[1]
            W = masks_bool_t.shape[2]
            r_min_t = rows_any_t.float().argmax(dim=1)
            c_min_t = cols_any_t.float().argmax(dim=1)
            r_max_t = (H - 1) - rows_any_t.flip(dims=[1]).float().argmax(dim=1)
            c_max_t = (W - 1) - cols_any_t.flip(dims=[1]).float().argmax(dim=1)

            # The four bbox bounds move to the host in one transfer: each
            # ``.cpu()`` is a CUDA sync.
            bounds_t = _torch.stack([r_min_t, c_min_t, r_max_t, c_max_t], dim=0)
            bounds_np = bounds_t.cpu().numpy()
            r_min_np, c_min_np, r_max_np, c_max_np = (
                bounds_np[0],
                bounds_np[1],
                bounds_np[2],
                bounds_np[3],
            )
            scores_np = sel_scores_t.to(_torch.float32).cpu().numpy()
            masks_np_bool = masks_bool_t.cpu().numpy()

            for k in range(int(masks_np_bool.shape[0])):
                per_tile.append(
                    SamDetection(
                        label=concept,
                        score=float(scores_np[k]),
                        pixel_mask=masks_np_bool[k],
                        pixel_bbox_rcrc=(
                            int(r_min_np[k]),
                            int(c_min_np[k]),
                            int(r_max_np[k]),
                            int(c_max_np[k]),
                        ),
                    )
                )
                if len(per_tile) >= max_per_image:
                    break
            if len(per_tile) >= max_per_image:
                break
        detections_per_tile.append(per_tile)
    return detections_per_tile
