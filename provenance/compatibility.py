"""Recognize recorded source identities without changing evidence hashes.

This module accepts exact hashes, or an explicitly recorded upstream-to-current
source pair. It never translates model, pool, manifest, or result-file hashes.
The separate record pin prevents a changed mapping from silently authorizing a
different runtime. Ordinary file-hashing helpers still report actual bytes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from provenance._record_pin import EXPECTED_MIGRATION_SHA256

RECORD_PATH = Path(__file__).with_name("source-migration.json")
UPSTREAM_COMMIT = "68755c84cdaeae0698a3b9d764b9c4c0bc6da7e0"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _relative_name(name: object) -> bool:
    if not isinstance(name, str) or "\\" in name or "\x00" in name:
        return False
    path = PurePosixPath(name)
    return (
        bool(name)
        and not path.is_absolute()
        and path.as_posix() == name
        and all(part not in (".", "..") for part in path.parts)
    )


def _source_name(name: object) -> bool:
    return _relative_name(name) and PurePosixPath(name).suffix == ".py"


def _digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _record() -> dict:
    """Read and verify every time so an in-process edit cannot use a cached map."""
    payload = RECORD_PATH.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if not _digest(EXPECTED_MIGRATION_SHA256) or not hmac.compare_digest(
        actual, EXPECTED_MIGRATION_SHA256
    ):
        raise ValueError("Source migration record integrity check failed")
    record = json.loads(payload)
    if (
        not isinstance(record, dict)
        or record.get("schema") != 1
        or record.get("upstream_commit") != UPSTREAM_COMMIT
        or not isinstance(record.get("files"), dict)
    ):
        raise ValueError("Unsupported source migration record")
    for name, entry in record["files"].items():
        if (
            not _source_name(name)
            or not isinstance(entry, dict)
            or not _digest(entry.get("original_sha256"))
            or not _digest(entry.get("migrated_sha256"))
            or not isinstance(entry.get("classification"), str)
        ):
            raise ValueError("Invalid source migration entry")
    return record["files"]


def _matches(name: str, expected: str, actual: str, entries: dict) -> bool:
    if not (_relative_name(name) and _digest(expected) and _digest(actual)):
        return False
    # Some recorded source tables also include unchanged plan/config JSON.
    # Such entries retain exact comparison and can never use a migrated pair.
    if PurePosixPath(name).suffix not in (".py", ".json"):
        return False
    if hmac.compare_digest(expected, actual):
        return True
    if not _source_name(name):
        return False
    entry = entries.get(name)
    return bool(
        entry
        and hmac.compare_digest(expected, entry["original_sha256"])
        and hmac.compare_digest(actual, entry["migrated_sha256"])
    )


def source_hash_matches(name: str, expected: str, actual: str) -> bool:
    """Compare one root-relative Python source file, failing closed on errors.

    Callers must hash the current file themselves. This function deliberately
    rejects non-source paths, even if their digests happen to match.
    """
    try:
        return _matches(name, expected, actual, _record())
    except (OSError, ValueError, TypeError):
        return False


def source_tables_match(expected: Mapping, actual: Mapping) -> bool:
    """Compare complete source tables with exactly the same path keys."""
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        return False
    if set(expected) != set(actual):
        return False
    try:
        entries = _record()
        return all(_matches(name, digest, actual[name], entries) for name, digest in expected.items())
    except (OSError, ValueError, TypeError):
        return False
