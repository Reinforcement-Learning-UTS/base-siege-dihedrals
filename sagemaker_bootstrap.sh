#!/usr/bin/env bash
# Prepare persistent research storage, the locked runtime and approved model files.
set -euo pipefail
PROJECT_DIR="${BASE_SIEGE_ROOT:-$HOME/base-siege}"
REPO_URL="${BASE_SIEGE_REPO:-https://github.com/Reinforcement-Learning-UTS/base-siege-dihedrals.git}"
LFS_MODE=models
SKIP_CLONE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) PROJECT_DIR="$2"; shift 2 ;;
    --repo) REPO_URL="$2"; shift 2 ;;
    --lfs) LFS_MODE="$2"; shift 2 ;;
    --skip-clone) SKIP_CLONE=1; shift ;;
    -h|--help) echo 'Usage: bash sagemaker_bootstrap.sh [--dir PATH] [--repo URL] [--lfs models|all|none] [--skip-clone]'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$LFS_MODE" in models|all|none) ;; *) echo 'Invalid LFS mode' >&2; exit 2 ;; esac
for program in git git-lfs uv; do
  command -v "$program" >/dev/null || { echo "Install $program before running this script." >&2; exit 1; }
done
if [[ "$SKIP_CLONE" -eq 0 && ! -d "$PROJECT_DIR/.git" ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone "$REPO_URL" "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"
[[ -f env/siege_env.py ]] || { echo 'Choose the project root.' >&2; exit 1; }
git lfs install --local
case "$LFS_MODE" in
  models) git lfs pull --include="$(cat models/lfs-paths.txt)" ;;
  all) git lfs pull --include='' --exclude='' ;;
  none) echo 'Model checks skipped: source-only checkout.' ;;
esac
uv python install 3.12.14
uv sync --frozen --python 3.12.14
uv run --frozen python -m ipykernel install --user --name base-siege --display-name 'Python 3 (Base Siege)'
if [[ "$LFS_MODE" != none ]]; then
  mkdir -p local-output
  report="local-output/studio-models-$(date -u +%Y%m%dT%H%M%SZ).json"
  uv run --frozen python tools/check_demo_models.py --integrity-only --out "$report"
fi
printf 'Project ready: %s\nOpen notebooks/00_sagemaker_setup.ipynb with the Base Siege kernel.\n' "$PROJECT_DIR"
