"""OpenAI-vision content parts: JPEG bytes packaged as the multi-part user
content a reasoner's ``messages`` carry (``reasoner_agent.append_vision_user``),
and the server's vision-token bill for those parts.
"""

from __future__ import annotations

import base64
import binascii
import io
import math

#: Pixels per vision token and a flat per-image surcharge: an upper bound on
#: Inkling's bill, not its tokenizer. Under-counting sends a request the window
#: cannot hold; over-counting only compacts a little early.
#: ponytail: fitted for Inkling only; another served model needs its own pair
#: (or its image processor's exact law).
_PIXELS_PER_VISION_TOKEN = 384
_VISION_TOKENS_PER_IMAGE = 32


def image_tokens_of(part: object) -> int:
    """The server's bill for one ``image_url`` part, from the JPEG header's
    dimensions; 0 for any other part or an undecodable one."""
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
    """The server's vision-token bill for a whole message list: image tokens
    land in the same window as the text."""
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            total += sum(image_tokens_of(p) for p in content)
    return total


def build_image_part(jpeg_bytes: bytes) -> dict:
    """One JPEG as an OpenAI ``image_url`` part: a ``data:`` URL the serve
    reads directly, with no HTTP fetch."""
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}}


def build_vision_user_content(text: str, jpegs: list[bytes]) -> list[dict]:
    """The text part, then one image part per non-empty JPEG in caller order:
    the text-first head stays prefix-stable with the text-only path."""
    return [{"type": "text", "text": text}, *(build_image_part(j) for j in jpegs if j)]
