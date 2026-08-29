"""Atomic publication of stage output directories."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from dhs.core.hashing import sha256_file


def write_sha256sums(
    paths: Iterable[Path],
    destination: Path,
) -> dict[str, str]:
    """Write a ``sha256sum``-compatible listing and return the digests by name.

    Entries are sorted by file name so that the listing is reproducible
    independently of the order in which a stage happened to create its outputs.
    """
    digests = {Path(path).name: sha256_file(path) for path in paths}
    Path(destination).write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(digests.items()))
    )
    return digests


@contextmanager
def atomic_output_dir(
    output_dir: Path,
    *,
    must_not_exist: bool = True,
) -> Iterator[Path]:
    """Build a stage's output in a sibling temporary directory, then swap it in.

    Yields the temporary directory to write into. On clean exit the directory is
    moved into place with ``os.replace``, which is atomic within a filesystem,
    so a reader sees either the previous state or the complete new one and never
    a half-written output. On any exception the temporary directory is removed
    and the exception propagates, leaving no partial artifact behind for a later
    run to mistake for a real result.

    The temporary directory is created as a sibling of the target, not in the
    system temp area, because ``os.replace`` is only atomic within one
    filesystem.

    Args:
        output_dir: Final location. Its parent is created if absent.
        must_not_exist: Refuse to overwrite an existing directory. Stages that
            publish a one-time artifact, such as the canonical split, rely on
            this to make a second run fail loudly instead of silently replacing
            evidence.
    """
    output_dir = Path(output_dir)
    if must_not_exist and output_dir.exists():
        raise FileExistsError(f"Output directory must not exist: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        yield temporary
        _swap_into_place(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _swap_into_place(temporary: Path, output_dir: Path) -> None:
    """Move ``temporary`` onto ``output_dir``, restoring the old one on failure.

    ``os.replace`` refuses to overwrite a non-empty directory, so replacing an
    existing output means moving it aside first. Between the two renames the
    target does not exist, which is why the old directory is preserved under a
    dotted sibling and put back if the second rename fails.
    """
    if not output_dir.exists():
        os.replace(temporary, output_dir)
        return
    superseded = output_dir.with_name(f".{output_dir.name}.replaced")
    if superseded.exists():
        raise FileExistsError(
            f"Refusing to replace while a previous backup remains: {superseded}"
        )
    os.replace(output_dir, superseded)
    try:
        os.replace(temporary, output_dir)
    except BaseException:
        os.replace(superseded, output_dir)
        raise
    shutil.rmtree(superseded, ignore_errors=True)
