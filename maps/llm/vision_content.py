"""OpenAI-vision content helpers: package JPEG bytes into the multi-part
content-part dict shape.

The vLLM endpoint accepts the OpenAI multi-part content schema (verified on the
live ``qwen36-35b-a3b`` deployment, 2026-05-12). This module just packages JPEG
bytes into the content-part dict so a caller can append a multi-part user
message without scattering base64 + dict construction across call sites.

NOTE: the live CityNav path is TEXT-ONLY — the single persistent
MAPS Reasoner reasons over the structured world-state (SAM detections +
perceived attributes); no image crop is fed to it — so these helpers have no
production caller today (only the transport test). When used, the content built
by :func:`build_vision_user_content` reaches the reasoner's ``messages`` and the
only step mode, ``step_tool_call``, forwards the multi-part shape unchanged.
"""

from __future__ import annotations

import base64
import binascii
import io
import math

#: Pixels per vision token, and a flat per-image surcharge. AN UPPER BOUND ON
#: THE SERVER'S BILL, deliberately, not a reproduction of its tokenizer.
#:
#: Measured against the served build by differencing the client's own estimate
#: from the server's reported input count: 2,607 tok/megapixel by OLS over 207
#: real requests, with single-image checks at 400 px -> 427 tokens and
#: 1024 px -> 2,632. Those two points do NOT lie on one pixels-per-token line
#: (they imply 19.3 and 20.0 px per token side), and Inkling's own config
#: gives ``vision_config.patch_size 40`` with an ``hmlp`` encoder, so the true
#: law involves a resize this does not model. The surcharge covers the
#: per-image framing the grid does not -- the checkpoint declares an
#: ``image_bos_token`` -- and the pair brackets both measurements from ABOVE
#: (400 px -> 449 vs 427; 1024 px -> 2,763 vs 2,632).
#:
#: Erring high is the whole point. Under-counting is the defect this exists to
#: end: it sends a request the window cannot hold and the episode dies on a
#: 400. Over-counting fires compaction a little early, which costs a crop the
#: model was probably done with. When the exact law is wanted, take it from the
#: served model's image processor rather than refitting these constants.
#:
#: PER-MODEL. These belong to Inkling. Re-measure for any other served model:
#: send one crop of known size and difference the reported prompt tokens
#: against an otherwise identical text-only call.
_PIXELS_PER_VISION_TOKEN = 384
_VISION_TOKENS_PER_IMAGE = 32


def image_tokens_of(part: object) -> int:
    """What the SERVER will bill for one ``image_url`` content part.

    Zero for anything that is not an image part, so a caller can map this
    over a mixed parts list.

    Reads the dimensions out of the JPEG header (``PIL`` opens lazily and
    decodes no pixels), because the cost is set by the pixel count and
    nothing else on the wire reports it. An undecodable part reads 0 rather
    than raising: an estimate that dies on a malformed crop is worse than
    one that under-counts a crop the server will reject anyway.
    """
    if not isinstance(part, dict) or part.get("type") != "image_url":
        return 0
    url = (part.get("image_url") or {}).get("url")
    if not isinstance(url, str) or "base64," not in url:
        return 0
    try:
        from PIL import Image

        raw = base64.b64decode(url.split("base64,", 1)[1], validate=False)
        with Image.open(io.BytesIO(raw)) as im:
            width, height = im.size
    except (ValueError, OSError, binascii.Error, ImportError):
        return 0
    return (
        math.ceil(width * height / _PIXELS_PER_VISION_TOKEN)
        + _VISION_TOKENS_PER_IMAGE
    )


def image_tokens_in(messages: list[dict]) -> int:
    """The server's vision-token bill for a whole message list.

    THE CLIENT USED TO COUNT THIS AS ZERO. Every image part collapsed to
    the literal ``"[image crop]"`` (about 3 tokens) before tokenization, on
    the stated theory that the server's vision budget was governed
    separately by a per-request image cap. It is not, twice over. A count
    cap does not bill tokens -- they land in the same context window as the
    text -- and no cap is declared anyway: the sglang flag is
    ``--limit-mm-data-per-request`` (``--limit-mm-per-prompt`` is vLLM's
    spelling), no card or script in this tree passes either, and the serves
    report ``limit_mm_data_per_request=None``. A 1024 px map cost the
    server 2,632 tokens and the client 3,
    so a conversation carrying sixteen of them was 42,112 tokens -- 87% of
    the input budget -- into a window the client believed was empty.

    Measured consequence: on a reasoning-effort ladder the max-effort arm
    sent twelve requests 1,049-4,582 tokens past its own declared budget,
    the server 400'd every one, and 130,095 uncounted tokens over 136
    images is the entire gap. The two lower-effort arms never took enough
    turns to reach the window, which is why it read as an effort effect.
    """
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            total += sum(image_tokens_of(p) for p in content)
    return total


def build_image_part(jpeg_bytes: bytes, *, detail: str = "high") -> dict:
    """Wrap a JPEG byte-string in the OpenAI vision content-part shape.

    The vLLM server reads the ``data:image/jpeg;base64,...`` URL form
    directly; no HTTP fetch round-trip happens server-side.
    """

    if not jpeg_bytes:
        raise ValueError("jpeg_bytes must be non-empty")
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{b64}",
            "detail": detail,
        },
    }


def build_vision_user_content(
    text: str,
    jpegs: list[bytes],
    *,
    detail: str = "high",
) -> list[dict]:
    """Compose the multi-part user content list (text + image parts).

    The text part lands first so the prefix-stable head of the
    conversation matches the text-only reasoner path until the first
    image part. Image parts follow in caller order.
    """

    content: list[dict] = [{"type": "text", "text": text}]
    for jpeg in jpegs:
        if not jpeg:
            continue
        content.append(build_image_part(jpeg, detail=detail))
    return content


__all__ = [
    "build_image_part",
    "build_vision_user_content",
]
