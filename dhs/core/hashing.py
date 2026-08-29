"""SHA256 digests over files and byte strings."""

from __future__ import annotations

import hashlib
from pathlib import Path

# Files hashed here run to hundreds of megabytes, so read in chunks rather than
# loading whole files. The size affects only memory, never the digest.
_CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file, read incrementally."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA256 digest of an in-memory payload."""
    return hashlib.sha256(payload).hexdigest()
