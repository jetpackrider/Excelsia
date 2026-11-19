#!/usr/bin/env bash
# Run Excelsia's pipeline on a free Kaggle GPU notebook.
#
#   bash scripts/kaggle.sh check                  # tests + what GPU we got
#   bash scripts/kaggle.sh chats 300              # local Gemma writes 300 practice chats
#   bash scripts/kaggle.sh english 180 medium 1000000   # minutes, model size, stories
#   bash scripts/kaggle.sh excelsia               # fine-tune on the chats, then talk to her
#   bash scripts/kaggle.sh pack                   # copy the results where Kaggle saves them
#
# The teacher model is open-weight Gemma (Apache 2.0), downloaded and run on the
# notebook's own GPU. No hosted AI service is called.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_URL="https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/gemma-4-12b-it-Q4_K_M.gguf"
MODEL="/tmp/gemma-4-12b-it-Q4_K_M.gguf"      # 7.1 GB, kept out of the saved output
PORT=8080
OUT="${EXCELSIA_OUT:-/kaggle/working/excelsia_out}"
SERVER_PID=""

has_gpu_engine() {
  python -c "import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)" 2>/dev/null
}

install_engine() {
  if [ -x /tmp/llama.cpp/build/bin/llama-server ] || has_gpu_engine; then return; fi
  echo "== installing the model engine"
  pip install -q "llama-cpp-python[server]" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 || true
  if has_gpu_engine; then echo "engine: ready-built GPU version"; return; fi
  echo "ready-built engine can't use this GPU; building llama.cpp instead (about 10 minutes)"
  apt-get -qq update && apt-get -qq install -y build-essential cmake libcurl4-openssl-dev
  [ -d /tmp/llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
  cmake /tmp/llama.cpp -B /tmp/llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON > /tmp/cmake.log
  cmake --build /tmp/llama.cpp/build --config Release -j --target llama-server > /tmp/build.log
  echo "engine: built llama.cpp"
}

start_server() {
  if [ ! -f "$MODEL" ]; then
    echo "== downloading Gemma (7.1 GB)"
    curl -L --fail --silent --show-error -o "$MODEL.part" "$MODEL_URL" && mv "$MODEL.part" "$MODEL"
  fi
  echo "== starting the model server"
  if [ -x /tmp/llama.cpp/build/bin/llama-server ]; then
    /tmp/llama.cpp/build/bin/llama-server -m "$MODEL" -c 16384 -ngl 99 \
      --host 127.0.0.1 --port "$PORT" > /tmp/server.log 2>&1 &
  else
    python -m llama_cpp.server --model "$MODEL" --n_ctx 16384 --n_gpu_layers -1 \
      --host 127.0.0.1 --port "$PORT" > /tmp/server.log 2>&1 &
  fi
  SERVER_PID=$!
  for _ in $(seq 1 300); do
    if curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" > /dev/null; then echo "server is up"; return; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "the server stopped while starting. Last lines of its log:"; tail -25 /tmp/server.log; exit 1
    fi
    sleep 2
  done
  echo "the server never came up. Last lines of its log:"; tail -25 /tmp/server.log; exit 1
}

stop_server() { [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true; }

stage="${1:-help}"; shift || true
case "$stage" in
  check)
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "No GPU: turn on the GPU accelerator in the notebook settings."
    python -m tests.test_excelsia
    ;;
  chats)
    n="${1:-300}"
    install_engine
    start_server
    trap stop_server EXIT
    python -m excelsia.gen_chats --n "$n" --server "http://127.0.0.1:$PORT"
    echo; echo "== a random sample to read"
    python - <<'PY'
import json, random
rows = [json.loads(l) for l in open("data/chats.jsonl", encoding="utf-8") if l.strip()]
for chat in random.sample(rows, min(15, len(rows))):
    print(f"--- {chat['topic']}")
    for t in chat["turns"]:
        print(f"  person: {t['user']}\n  excelsia [{t['mood']}]: {t['reply']}")
PY
    ;;
  english)
    minutes="${1:-180}"; size="${2:-medium}"; stories="${3:-1000000}"
    python -m excelsia.train stories --n "$stories"
    python -m excelsia.train pretrain --size "$size" --minutes "$minutes" --steps 2000000 --batch 64
    ;;
  excelsia)
    python -m excelsia.train finetune
    python -m excelsia.chat --say "hello" --say "who are you?" --say "is your name excelsia?" \
      --say "you are wonderful, thank you" --say "what do you like?" \
      --say "you are stupid and i hate you" --say "sorry. tell me about your forests" \
      --say "i have to go now, goodbye"
    ;;
  pack)
    mkdir -p "$OUT"
    cp -v data/chats.jsonl "$OUT"/ 2>/dev/null || true
    cp -v work/pretrain.pt work/excelsia.pt "$OUT"/ 2>/dev/null || true
    echo "Results are in $OUT. Download them from the notebook's Output panel."
    ;;
  *)
    sed -n '2,12p' "$0"
    ;;
esac
