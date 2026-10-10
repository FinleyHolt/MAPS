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
from pathlib import Path
from typing import Any

import numpy as np




@dataclass
class SamDetection:
    label: str
    score: float
    pixel_mask: np.ndarray
    pixel_bbox_rcrc: tuple[int, int, int, int]
    object_id: int


@dataclass
class _LoadedSam:
    """Holds Meta's sam3 image model + the canonical Datapoint pipeline
    (transform, collate, postprocessor) so all concepts and (optionally)
    all tiles run in ONE ``model(batch)`` forward pass.

    This is the SAM 3.1 batched-concept path used by the official
    ``examples/sam3_image_batched_inference.ipynb`` notebook. It bypasses
    the per-concept ``Sam3Processor.set_text_prompt`` loop entirely —
    every concept is a separate ``FindQueryLoaded`` inside one Datapoint
    and the model handles them together.
    """

    model: Any
    transform: Any
    collate: Any
    postprocess: Any  # callable that takes (output, find_metadatas) -> list of dicts
    copy_to_device: Any
    device: str
    version: str


_SAM_SINGLETON: _LoadedSam | None = None


def release() -> None:
    """Drop the resident model and hand its device memory back to the driver.

    A process that ends with the model referenced leaves its allocations
    for the driver to reclaim on its own schedule; on a Jetson's unified
    memory that is host memory the box's free-memory checks read as gone.
    """
    global _SAM_SINGLETON
    _SAM_SINGLETON = None
    gc.collect()
    import torch as _torch

    if _torch.cuda.is_available():
        _torch.cuda.synchronize()
        _torch.cuda.empty_cache()

# Distinct concept unions one process may hold before the text-encoder cache is
# cleared. A runner process flies one episode and an episode freezes ONE union,
# so the live set is 1; this is a bound against a caller that never stops
# minting new phrase lists, not a working-set estimate.
_TEXT_MEMO_MAX = 32


def _ensure_sam(
    version: str, device: str, resolution: int, confidence_threshold: float
) -> _LoadedSam:
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
        f"[sam3_engine] building sam3 image model on {device} (version={version})",
        flush=True,
    )
    checkpoint_path = os.environ.get("SAM3_CHECKPOINT") or os.environ.get("SAM_CHECKPOINT")
    load_from_hf = checkpoint_path is None
    if checkpoint_path is None and version == "sam3.1":
        checkpoint_path = download_ckpt_from_hf(version="sam3.1")
        load_from_hf = False
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_hf,
    )
    _dev = _torch.device(device)
    if _dev.type == "cuda" and any(p.device.type == "cpu" for p in model.parameters()):
        model.to(_dev)
    model.eval()

    # Disable the segmentation-head activation checkpoint at inference.
    # `model_builder` hardcodes `act_ckpt=True` on the
    # `UniversalSegmentationHead`, and `SegmentationHead._embed_pixels`
    # gates the pixel-decoder `torch.utils.checkpoint.checkpoint` on a
    # bare `if self.act_ckpt:` with NO `self.training` guard, so it runs
    # on every forward even in eval/inference_mode. Activation
    # checkpointing only buys memory by recomputing in the backward
    # pass; under `torch.inference_mode()` there is no backward, so the
    # checkpoint wrapper is pure overhead — a per-call CUDA RNG-state
    # stash (a host<->device sync stall on the hot path) plus a
    # Dynamo/fusion barrier — for zero benefit. The pixel decoder is
    # Conv2d/GroupNorm/ReLU/interpolate with nothing RNG-dependent, so
    # disabling the checkpoint is mathematically identical to the
    # checkpointed path (verified by the parity gate). Quoted exactly so
    # a sam3 version that renames the class fails loudly rather than
    # silently leaving the stall in.
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

    # torch.compile is dead on this model ON ADA, and every clause of that
    # finding names an Ada mechanism. Whole-model max-autotune trips the
    # sm_89 101 KB shared-mem Triton ceiling and reduce-overhead
    # AssertionErrors on the dynamic DETR head; the one remaining avenue —
    # compiling ONLY `backbone.vision_backbone` via SAM3's own
    # `compile_visual` hook, `max-autotune-no-cudagraphs`, `dynamic=False`,
    # with `max_autotune_prune_choices_based_on_shared_mem` set so the sm_89
    # crash is pruned away — was measured on GPU 3: it compiled cleanly but
    # produced ~0 throughput change (386.6 vs 392 ms/frame) AND drifted
    # detections past the bit-identical gate (1223 frame mismatches, mask
    # IoU down to 0.975) because Inductor's autotuned bf16 GEMMs reassociate
    # differently from stock cuBLAS. No speedup + quality drift, so on an
    # L40S it is not an option and throughput comes from the host-overhead
    # removals above.
    #
    # Do not re-attempt ON sm_89. The shared-memory ceiling and the measured
    # autotune drift are both properties of that card, and neither transfers
    # to a Blackwell part. `build_sam3_image_model` also takes its own
    # `compile=` flag, a different hook from the `compile_visual` tried here.
    # Anything re-attempted elsewhere still passes the parity gate before it
    # ships, and updates this comment in the same change.

    # Precision is bf16-autocast over fp32 master weights, and that is
    # the only viable precision for this model on the paths measured here:
    #   * The FAIR sam3 backbone uses custom ops that torchao's
    #     Float8Tensor subclass cannot dispatch ("NotImplementedError:
    #     Float8Tensor dispatch"), so post-training FP8/INT8 weight
    #     quantization of this model does not run on this stack. That is a
    #     dead end for PTQ over the WEIGHTS specifically, and it says
    #     nothing about `sam3.perflib.fa3`, which is the package's own FP8
    #     ATTENTION: it casts q/k/v to `float8_e4m3fn` and calls
    #     `flash_attn_interface`, behind a `use_fa3` flag that every
    #     attention module defaults to False. Different mechanism, never
    #     tried here, and it needs FA3 installed to be switchable at all.
    #   * fp32-direct (no autocast) is ~2-3x slower on the 1008px ViT
    #     (half the Ada tensor-core rate, 2x memory traffic) for zero
    #     inference-accuracy gain — bf16 matmul is lossless for this
    #     model's scale.
    #   * Converting master weights to bf16 (vs fp32 + autocast) is ~no
    #     throughput gain (matmuls already run bf16 under autocast) and
    #     risks LayerNorm/reduction drift that autocast otherwise keeps
    #     in fp32 — i.e. accuracy risk for no speed. Keep fp32 weights +
    #     autocast.
    # The model also has hardcoded internal `.to(torch.bfloat16)` calls; the
    # master fp32 + autocast pattern is what they expect.
    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(
                sizes=resolution,
                max_size=resolution,
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
        # The postprocessor filters by score BEFORE the full-resolution
        # bilinear mask upscale, so it never upsamples masks the engine
        # would discard. Measured on the 8-concept aerial workload
        # (L40S, 1008 px): postprocess 46 ms -> ~0 at N=8, +12 %
        # single-stream throughput (392 -> 348 ms/call). Its internal
        # cut is strict ``>`` against ``score_threshold - 1e-6`` while
        # the engine's own filter is ``>=``, so a single near-threshold
        # (score ~= the 0.3 floor) detection is dropped on ~6 % of
        # frames: a measured 3 / 1775 detections (0.17 %), and only the
        # weakest one per affected frame. Every retained detection is
        # byte-identical (mask IoU 1.0, scores/bboxes unchanged) — this
        # is a pure drop of the flimsiest detections, never a shift.
        # That recall/throughput trade is the default.
        detection_threshold=max(0.0, float(confidence_threshold) - 1e-6),
        to_cpu=False,
    )
    _SAM_SINGLETON = _LoadedSam(
        model=model,
        transform=transform,
        collate=collate_fn_api,
        postprocess=postprocess,
        copy_to_device=copy_data_to_device,
        device=device,
        version=version,
    )
    print("[sam3_engine] sam3 image model ready", flush=True)
    return _SAM_SINGLETON


def _memoize_text_encoder(model: Any) -> None:
    """Cache the text encoder's output on the concept phrases that produced it.

    ``VETextEncoder.forward(text: list[str], ...)`` is a pure function of the
    phrase list: it tokenizes, embeds, runs the encoder and resizes, with no
    state and nothing random, under ``eval`` and ``inference_mode``. The same
    phrases therefore give the same three tensors every time.

    They are asked for on every single call, and the SAME ones over and over.
    The mission surface lets the model retarget the segmenter at any turn —
    ``perceive``'s watchlist channel calls ``declare_mission_concepts`` with
    the full set, latest-wins by class, and nothing freezes it — so the reuse
    is not a property of the protocol. It is a property of what the model
    actually asks for: job 1806's SAM log carries 936 ``/segment`` calls
    against **19 distinct concept lists**. One miss per distinct list is a 98%
    hit rate in one process.

    Measured on a B300 (job 1875), the encode costs ~16 ms whether the list
    holds one phrase or eight — a fixed shape, so this is ~13% of a ~119 ms
    call at EVERY concept count, not a tax that shrinks with the workload.

    The cache is PER PROCESS, so a replica plane pays one miss per distinct
    list per replica. At four replicas that is at most 76 misses of those
    936 calls, still ~92%. It gets worse as replicas grow,
    which is the one thing that would make this stop paying.

    Bit-identical by construction, which is the only reason this is a cache and
    not a trade: same weights, same input, deterministic module. The tensors
    are CLONED in and out so a downstream in-place write can never reach
    another call's copy — the cost of that is a few kB against the 16 ms.

    Bounded and cleared wholesale rather than LRU-evicted. The observed working
    set is ~19 lists across a whole sweep, so the bound exists to stop a
    pathological caller growing it forever rather than to manage pressure; at
    that size the eviction policy cannot matter, and a clear costs one re-encode
    per live list.

    The class is named exactly, so a sam3 version that renames or restructures
    it fails LOUD here instead of silently leaving the 16 ms in.
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
    """Construct one Datapoint with one image + N text queries, exactly
    like the FAIR notebook. ``dp_counter`` is a 1-element list used as a
    mutable monotonic id; each query gets a unique ``coco_image_id``
    that we use later to align results back to concepts."""
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        FindQueryLoaded,
        Image as SAMImage,
        InferenceMetadata,
    )

    img_pil = _to_pil(rgb)
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
                    # (H, W), the order the postprocessor unbinds it in:
                    # ``img_h, img_w = target_sizes.unbind(1)`` for boxes, and
                    # ``interpolate(masks, target_size.tolist())`` for masks.
                    # Reversed, a non-square frame comes back transposed --
                    # invisible for every square frame the runtime feeds
                    # (rgb_upscale_px, and SAHI tiles are 1008 square), and a
                    # silent wrong answer for anything else.
                    original_size=[h, w],
                    object_id=0,
                    frame_index=0,
                ),
            )
        )
        qids.append(cid)
    return dp, qids


def _to_pil(rgb: np.ndarray):
    from PIL import Image

    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    return Image.fromarray(rgb)


def segment(
    rgb: np.ndarray,
    concepts: list[str],
    *,
    version: str = "sam3.1",
    device: str = "cuda",
    score_threshold: float = 0.3,
    min_area_px: int = 32,
    max_per_image: int = 64,
    resolution: int = 1008,
) -> list[SamDetection]:
    """Run SAM 3.1 once on a single frame in-process.

    This is the in-process backend ``serving.sam_server`` drives for
    ``/segment``. Application code does NOT call this directly; the
    runtime adapter (``maps.perception.sam.detect_client``) POSTs
    ``/segment`` to that server. ``pixel_mask`` is the bbox crop; ``pixel_bbox_rcrc`` is the
    absolute origin in tile-local coords.
    """
    if not concepts:
        return []
    return _segment_via_meta(
        [rgb],
        concepts,
        version=version,
        device=device,
        score_threshold=score_threshold,
        min_area_px=min_area_px,
        max_per_image=max_per_image,
        resolution=resolution,
    )[0]


def _segment_via_meta(
    rgb_tiles: list[np.ndarray],
    concepts: list[str],
    *,
    version: str,
    device: str,
    score_threshold: float,
    min_area_px: int,
    max_per_image: int,
    resolution: int,
) -> list[list[SamDetection]]:
    """Segment ``rgb_tiles`` via the SAM 3.1 standalone Datapoint
    pipeline. Returns a list aligned with ``rgb_tiles`` of
    per-concept-ordered SamDetection.

    Tiles are processed in GPU-forward chunks of ``SAM_GPU_CHUNK`` (env,
    default 8). Each chunk is one collated ``model(batch)`` forward: the
    vision encoder runs once per tile in the chunk and every concept
    decoder for the chunk runs inside that single forward (the
    ``examples/sam3_image_batched_inference.ipynb`` pattern). Chunking
    decouples the caller's requested batch size from the GPU forward
    size: a caller may pass any number of tiles, while per-forward peak
    VRAM stays bounded by the chunk.

    This bound is required, not optional. The dominant transient is the
    MaskFormer segmentation-head FPN (``maskformer_segmentation`` group
    norm under activation checkpointing), whose peak grows ~linearly in
    the forward batch size (~2.85 GB/tile at 1008 px / 8 concepts on an
    L40S); an unbounded batch of 16 tiles tries a single >10 GiB
    allocation and OOMs a 44 GB card. The chunk is the ceiling.

    Per-image outputs are independent of chunk composition (per-image
    attention; ``LayerNorm``/``GroupNorm`` are per-sample, not
    ``BatchNorm``; ``process_results`` keys per query group), so chunking
    is detection-equivalent to one big forward.
    """
    import torch as _torch

    loaded = _ensure_sam(version, device, resolution, score_threshold)
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
                score_threshold=score_threshold,
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
    score_threshold: float,
    min_area_px: int,
    max_per_image: int,
) -> list[list[SamDetection]]:
    """One collated ``model(batch)`` forward over the tiles in one GPU
    chunk. ``dp_counter`` stays monotonic across chunks so query ids are
    globally unique. All chunk-local tensors (``out``, ``results``,
    ``batch``) are dropped on return, so the caching allocator reuses
    that memory for the next chunk and per-forward peak stays bounded by
    a single chunk."""
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
        # No per-concept cap: every same-class instance (e.g. EVERY car) must be
        # detected and exposed so the reasoner can do relational counting
        # ("the Nth car from the intersection", "between X and Y"). The old
        # per_concept_cap = max(8, max_per_image // n_concepts) silently dropped
        # same-class instances (~8-12/frame under the multi-concept union). Only
        # the total ``max_per_image`` backstop remains (set high via config to
        # be effectively unlimited while bounding a pathological frame -> OOM).
        per_tile: list[SamDetection] = []
        next_object_id = 0
        for concept, qid in zip(tile_concepts, qids):
            rec = results[qid]
            scores_t = rec.get("scores")
            masks_t = rec.get("masks")
            if scores_t is None or masks_t is None or scores_t.numel() == 0:
                continue
            scores_t = scores_t.detach()
            keep_score_t = scores_t >= score_threshold
            n_pass_score = int(keep_score_t.sum().item())
            if n_pass_score == 0:
                continue
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

            # Pack the 4 bbox-bound tensors into one transfer. Each
            # ``.cpu().numpy()`` forces a CUDA sync, and the bbox
            # bounds are tiny same-shape int tensors that benefit from
            # being moved together (stack on GPU, single sync). Drops
            # 6 syncs/concept to 3 (scores, bounds, masks).
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

            n_pass_area = int(masks_np_bool.shape[0])
            kept_for_concept = 0
            for k in range(n_pass_area):
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
                        object_id=next_object_id,
                    )
                )
                next_object_id += 1
                kept_for_concept += 1
                if len(per_tile) >= max_per_image:
                    break
            if len(per_tile) >= max_per_image:
                break
        detections_per_tile.append(per_tile)
    return detections_per_tile


__all__ = [
    "SamDetection",
    "segment",
]
