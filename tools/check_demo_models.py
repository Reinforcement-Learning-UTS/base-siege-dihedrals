"""Load final study checkpoints and replay generated examples without training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check():
    from ui.deployment import load_cpu_policy
    from ui.policy_playback import record_policy, validate_compatibility
    from ui.v2_models import label_v2, read_registry
    from ui import ddqn_models, trpo_models

    inventory = json.loads((ROOT / "docs/migration/final-models.json").read_text())
    registry = read_registry()
    targets = {item["path"]: item for item in registry["models"]}
    reports = []
    for item in inventory["models"]:
        path = ROOT / item["path"]
        entry = {key: item[key] for key in ("study", "condition", "learner_seed", "path", "size")}
        start = perf_counter()
        try:
            if path.stat().st_size != item["size"]:
                raise ValueError("Model byte size differs")
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Model SHA-256 differs")
            if item["algorithm"] == "masked-double-dqn":
                policy = ddqn_models.load_selected_policy(path)
                mode, scheme = "DD", "legacy"
            elif item["algorithm"] == "MaskableTRPO":
                policy = trpo_models.load_selected_policy(path)
                mode, scheme = "SS", "v3-final-ds-v1"
            else:
                manifest_path = ROOT / item["manifest_path"]
                if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != item["manifest_sha256"]:
                    raise ValueError("Model manifest SHA-256 differs")
                manifest = json.loads(manifest_path.read_text())
                if item["study"] == "V2":
                    manifest = label_v2(manifest, targets[item["path"]], registry)
                validate_compatibility({**manifest, "model_sha256": item["sha256"]})
                stamp = path.stat()
                policy = load_cpu_policy(str(path), stamp.st_mtime_ns, stamp.st_size,
                                         item["sha256"], item["transitions"])
                if policy.model.seed != item["learner_seed"]:
                    raise ValueError("Loaded learner identity differs")
                mode = "DS" if item["study"] == "V2" else "SS"
                scheme = "v2-final-ds-v1" if item["study"] == "V2" else "legacy"
            replay = record_policy(policy, base_seed=42, motif="closed_ring", action_seed=41001,
                                   acting_mode=mode, seed_scheme=scheme)
            entry.update(status="passed", decisions=len(replay["actions"]), summary=replay["summary"])
        except Exception as exc:
            entry.update(status="blocked", error_type=type(exc).__name__, reason=str(exc))
        entry["seconds"] = round(perf_counter() - start, 3)
        reports.append(entry)
        print(json.dumps({key: value for key, value in entry.items() if key != "summary"}), flush=True)
    passed = sum(item["status"] == "passed" for item in reports)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"scope": "CPU checkpoint loading and one generated non-final example per model; no training or final evaluation",
            "python": platform.python_version(), "expected_models": len(reports),
            "passed_models": passed, "blocked_models": len(reports) - passed,
            "peak_process_rss_mib": round(peak / (1024 ** 2 if sys.platform == "darwin" else 1024), 1),
            "status": "passed" if passed == len(reports) else "incomplete", "models": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = check()
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "models"}))
    raise SystemExit(result["status"] != "passed")


if __name__ == "__main__":
    main()
