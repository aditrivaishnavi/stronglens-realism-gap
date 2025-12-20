"""Simple IO for local filesystem and S3.

The selection-function pipeline needs to write artifacts (CSV/JSON/PNG) to a
single destination. In practice we use S3 for long-running jobs.

This module supports:
  - Local paths
  - s3://bucket/key URIs (requires boto3)
  - s3a://bucket/key URIs (EMR convention, same S3 backend)
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple


def is_s3_uri(path_or_uri: str) -> bool:
    low = path_or_uri.lower()
    return low.startswith("s3://") or low.startswith("s3a://")


def _split_s3_uri(uri: str) -> Tuple[str, str]:
    if not is_s3_uri(uri):
        raise ValueError(f"Not an s3:// or s3a:// uri: {uri}")
    # strip scheme: handle both s3:// and s3a://
    low = uri.lower()
    if low.startswith("s3a://"):
        no_scheme = uri[6:]
    else:
        no_scheme = uri[5:]
    parts = no_scheme.split("/", 1)
    bucket = parts[0]
    key = "" if len(parts) == 1 else parts[1]
    return bucket, key


def join_uri(base: str, *parts: str) -> str:
    """Join path parts onto a base local path or s3 URI.

    Preserves the original scheme (s3:// or s3a://).
    """
    if is_s3_uri(base):
        # Preserve original scheme
        low = base.lower()
        scheme = "s3a://" if low.startswith("s3a://") else "s3://"
        b, k = _split_s3_uri(base)
        k = k.rstrip("/")
        for p in parts:
            k = f"{k}/{p.lstrip('/')}" if k else p.lstrip("/")
        return f"{scheme}{b}/{k}"
    # local
    return os.path.join(base, *parts)


def ensure_parent_dir(path: str) -> None:
    if is_s3_uri(path):
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


@dataclass
class S3Client:
    """Thin wrapper to lazily construct boto3 client."""

    profile: Optional[str] = None
    region: Optional[str] = None

    def client(self):
        import boto3

        if self.profile:
            import boto3.session

            sess = boto3.session.Session(profile_name=self.profile, region_name=self.region)
            return sess.client("s3")
        return boto3.client("s3", region_name=self.region)


def write_bytes(dst: str, data: bytes, content_type: Optional[str] = None, s3: Optional[S3Client] = None) -> None:
    if is_s3_uri(dst):
        bucket, key = _split_s3_uri(dst)
        s3c = (s3 or S3Client()).client()
        kwargs = {"Bucket": bucket, "Key": key, "Body": data}
        if content_type:
            kwargs["ContentType"] = content_type
        s3c.put_object(**kwargs)
        return

    ensure_parent_dir(dst)
    with open(dst, "wb") as f:
        f.write(data)


def write_text(dst: str, text: str, s3: Optional[S3Client] = None) -> None:
    write_bytes(dst, text.encode("utf-8"), content_type="text/plain", s3=s3)


def write_json(dst: str, obj: Any, indent: int = 2, s3: Optional[S3Client] = None) -> None:
    data = json.dumps(obj, indent=indent, sort_keys=True).encode("utf-8")
    write_bytes(dst, data, content_type="application/json", s3=s3)


def write_png(dst: str, pil_image, s3: Optional[S3Client] = None) -> None:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    write_bytes(dst, buf.getvalue(), content_type="image/png", s3=s3)


def open_for_write(dst: str, s3: Optional[S3Client] = None):
    """Return a file-like object for writing. For S3, returns BytesIO.

    Caller must pass resulting bytes to write_bytes.
    """

    if is_s3_uri(dst):
        return io.BytesIO()
    ensure_parent_dir(dst)
    return open(dst, "wb")
