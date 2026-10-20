"""The reviewed source migration never permits a new scientific identity."""

import hashlib
import json
from pathlib import Path

import pytest

import provenance.compatibility as compatibility
from provenance import source_hash_matches, source_tables_match


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def migration(tmp_path, monkeypatch):
    old, new = digest("original source"), digest("reviewed source")
    record = {
        "schema": 1,
        "upstream_commit": compatibility.UPSTREAM_COMMIT,
        "files": {
            "env/combat.py": {
                "original_sha256": old,
                "migrated_sha256": new,
                "classification": "comments-and-docstrings",
            },
            # The first package may carry a complete map without later-stage files.
            "ui/app.py": {
                "original_sha256": digest("old app"),
                "migrated_sha256": digest("new app"),
                "classification": "application-integration",
            },
        },
    }
    payload = json.dumps(record).encode()
    path = tmp_path / "source-migration.json"
    path.write_bytes(payload)
    monkeypatch.setattr(compatibility, "RECORD_PATH", path)
    monkeypatch.setattr(compatibility, "EXPECTED_MIGRATION_SHA256", hashlib.sha256(payload).hexdigest())
    return old, new, path, record


def test_exact_and_explicit_forward_source_pair(migration):
    old, new, _, _ = migration
    assert source_hash_matches("env/combat.py", old, old)
    assert source_hash_matches("env/combat.py", new, new)
    assert source_hash_matches("env/combat.py", old, new)
    assert not source_hash_matches("env/combat.py", new, old)


def test_unknown_pair_and_other_file_rejected(migration):
    old, new, _, _ = migration
    assert not source_hash_matches("env/combat.py", digest("another original"), new)
    assert not source_hash_matches("env/combat.py", old, digest("unexpected runtime"))
    assert not source_hash_matches("env/config.py", old, new)


@pytest.mark.parametrize("name", ["../env/combat.py", "/env/combat.py", "env/../env/combat.py",
                                 "env//combat.py", "env\\combat.py", "./env/combat.py", ""])
def test_noncanonical_source_paths_rejected(migration, name):
    old, new, _, _ = migration
    assert not source_hash_matches(name, old, new)


def test_source_tables_require_exact_key_sets_and_values(migration):
    old, new, _, _ = migration
    original, current = {"env/combat.py": old}, {"env/combat.py": new}
    assert source_tables_match(original, current)
    assert not source_tables_match(original, {})
    assert not source_tables_match(original, {**current, "env/config.py": old})
    assert not source_tables_match({**original, "env/config.py": old}, current)
    assert not source_tables_match(original, {"env/combat.py": "changed"})
    assert not source_tables_match(None, current)
    assert not source_tables_match(original, [])


def test_evidence_files_cannot_receive_a_migrated_identity(migration):
    old, new, _, _ = migration
    for name in ("runs/P-0/model.zip", "runs/P-0/state.pt", "artifacts/episodes.csv",
                 "runs/P-0/manifest.json", "bases/train-v2.json"):
        assert not source_hash_matches(name, old, new)
    # Source-table JSON configurations remain byte-exact.
    assert source_tables_match({"version2/plan.json": old}, {"version2/plan.json": old})
    assert not source_tables_match({"version2/plan.json": old}, {"version2/plan.json": new})


def test_changed_record_fails_even_for_an_exact_source_hash(migration):
    old, new, path, _ = migration
    path.write_bytes(path.read_bytes() + b" ")
    assert not source_hash_matches("env/combat.py", old, new)
    assert not source_hash_matches("env/combat.py", old, old)
    assert not source_tables_match({"env/combat.py": old}, {"env/combat.py": new})


def test_record_is_rechecked_after_an_earlier_success(migration):
    old, new, path, _ = migration
    assert source_hash_matches("env/combat.py", old, new)
    path.unlink()
    assert not source_hash_matches("env/combat.py", old, new)


@pytest.mark.parametrize("mutation", ["upstream", "entry", "non-source", "schema"])
def test_invalid_record_rejected_despite_a_matching_record_pin(migration, monkeypatch, mutation):
    old, new, path, record = migration
    if mutation == "upstream":
        record["upstream_commit"] = "0" * 40
    elif mutation == "entry":
        record["files"]["env/combat.py"]["original_sha256"] = "not-a-digest"
    elif mutation == "non-source":
        record["files"]["runs/model.zip"] = record["files"]["env/combat.py"]
    else:
        record["schema"] = 99
    payload = json.dumps(record).encode()
    path.write_bytes(payload)
    monkeypatch.setattr(compatibility, "EXPECTED_MIGRATION_SHA256", hashlib.sha256(payload).hexdigest())
    assert not source_hash_matches("env/combat.py", old, new)


def test_only_supplied_subset_is_required_for_first_package(migration):
    old, new, _, _ = migration
    assert source_tables_match({"env/combat.py": old}, {"env/combat.py": new})


def test_repository_record_preserves_frozen_v2_source_identity():
    root = Path(__file__).resolve().parents[1]
    registry = root / "ui/v2_checkpoints.json"
    # The foundation package deliberately excludes the later-stage UI catalog.
    if not registry.is_file():
        pytest.skip("V2 catalog is supplied in the research stage")
    expected = json.loads(registry.read_text())["source_hashes"]
    current = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in expected}
    assert len(expected) == 28
    assert source_tables_match(expected, current)
    current["env/combat.py"] = digest("unreviewed simulator change")
    assert not source_tables_match(expected, current)
