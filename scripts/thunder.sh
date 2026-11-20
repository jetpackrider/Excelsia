#!/usr/bin/env bash
# Run Excelsia's teacher on a rented GPU server. Written for Thunder Compute; works on any
# Linux machine with one NVIDIA card of 48 GB or more. Nothing here runs on a laptop.
#
#   bash scripts/thunder.sh setup       # once: install vLLM, download the teacher and her reading
#   bash scripts/thunder.sh pilot       # the teacher writes 50 conversations, to read before going big
#   bash scripts/thunder.sh chats N     # the teacher writes N conversations into data/chats-qwen.jsonl
#   bash scripts/thunder.sh stop        # stop the teacher and free the card
#
# Start long stages in the background so a dropped connection does not stop them:
#   nohup bash scripts/thunder.sh pilot > ~/pilot.log 2>&1 &
#
# The teacher is Qwen3.6-27B (Apache 2.0) in Qwen's own FP8 version, served by vLLM on the
# server's own card. No hosted AI service is called.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV="${EXCELSIA_ENV:-$HOME/env}"
MODEL="${EXCELSIA_TEACHER:-Qwen/Qwen3.6-27B-FP8}"
PORT=8000
LOGS="${EXCELSIA_LOGS:-$HOME}"

need_gpu() {
  nvidia-smi > /dev/null 2>&1 || { echo "No NVIDIA card here. This script is for the GPU server."; exit 1; }
}

activate() {
  [ -f "$ENV/bin/activate" ] || { echo "Not set up yet. Run: bash scripts/thunder.sh setup"; exit 1; }
  # shellcheck disable=SC1091
  source "$ENV/bin/activate"
}

up() { curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" > /dev/null; }

serve() {
  if up; then echo "the teacher is already running"; return; fi
  echo "== starting the teacher ($MODEL); its log is $LOGS/vllm.log"
  nohup vllm serve "$MODEL" --port "$PORT" --language-model-only --max-model-len 16384 \
    --gpu-memory-utilization 0.92 --max-num-seqs 64 > "$LOGS/vllm.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 240); do
    if up; then echo "the teacher is up"; return; fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "The teacher stopped while starting. Last lines of its log:"; tail -40 "$LOGS/vllm.log"; exit 1
    fi
    sleep 5
  done
  echo "The teacher took over 20 minutes to start. Last lines of its log:"; tail -40 "$LOGS/vllm.log"; exit 1
}

setup() {
  need_gpu
  command -v uv > /dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
  [ -d "$ENV" ] || uv venv "$ENV" --python 3.12
  activate
  uv pip install "vllm>=0.19" pyarrow --torch-backend=auto
  python -c "from huggingface_hub import snapshot_download as d; d('$MODEL')"
  [ -f work/stories.bin ] || python -m excelsia.train stories --n 0
  [ -f work/talk.bin ] || python -m excelsia.train talk
  echo "ALL DONE"
}

write_chats() {  # write_chats <how many> <file>
  need_gpu; activate; serve
  python -m excelsia.gen_chats --teacher vllm --talk --n "$1" --out "$2"
  echo; python -m excelsia.check_chats "$2"
}

stage="${1:-help}"; shift || true
case "$stage" in
  setup) setup ;;
  pilot) need_gpu; rm -f work/pilot-qwen.jsonl; mkdir -p work; write_chats 50 work/pilot-qwen.jsonl ;;
  chats) write_chats "${1:-1000}" data/chats-qwen.jsonl ;;
  serve) need_gpu; activate; serve ;;
  stop)  pkill -f "vllm serve" && echo "the teacher is stopped" || echo "the teacher was not running" ;;
  *)     sed -n '2,15p' "$0" ;;
esac
