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

cuda_env() {
  # PyTorch ships its own copies of the NVIDIA libraries. Put them where the engine can find them.
  local extra
  extra=$(python - <<'PY'
import glob, os, site
dirs = []
for sp in site.getsitepackages():
    dirs += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
    dirs += glob.glob(os.path.join(sp, "nvidia", "*", "*", "lib"))
dirs += [d for d in ("/usr/local/cuda/lib64", "/usr/local/nvidia/lib64") if os.path.isdir(d)]
print(":".join(dirs))
PY
)
  export LD_LIBRARY_PATH="${extra}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

describe() {  # what kind of machine is this? Printed so problems are easy to report.
  echo "== this machine"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -4
  python -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| torch CUDA', torch.version.cuda)" 2>&1 | tail -1
  (nvcc --version 2>/dev/null | tail -2 | head -1) || echo "nvcc: not found"
  echo "libcuda: $(ldconfig -p 2>/dev/null | awk '/libcuda\.so/{print $NF}' | tr '\n' ' ')"
}

has_gpu_engine() {
  # No output hidden: if the ready-built engine can't load, we want to see why.
  python -c "import llama_cpp, sys; ok = llama_cpp.llama_supports_gpu_offload(); print('ready-built engine sees the GPU:', ok); sys.exit(0 if ok else 1)"
}

install_engine() {
  cuda_env
  if [ -x /tmp/llama.cpp/build/bin/llama-server ]; then return; fi
  describe
  echo "== installing the model engine"
  local major index
  major=$(python -c "import torch; print((torch.version.cuda or '12').split('.')[0])" 2>/dev/null || echo 12)
  index="cu124"; [ "$major" = "13" ] && index="cu130"
  local url="https://abetlen.github.io/llama-cpp-python/whl/$index"
  # Wheels only: never let pip quietly compile a slow CPU-only copy.
  pip install -q --force-reinstall --no-deps --only-binary=:all: llama-cpp-python --index-url "$url" \
    && pip install -q --only-binary=llama-cpp-python "llama-cpp-python[server]" --extra-index-url "$url" \
    || echo "ready-built engine did not install"
  if has_gpu_engine; then echo "engine: ready-built GPU version ($index)"; return; fi

  echo "== building llama.cpp instead (about 10 minutes)"
  (apt-get -qq update; apt-get -qq install -y build-essential cmake) > /tmp/apt.log 2>&1 || true
  [ -d /tmp/llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
  rm -rf /tmp/llama.cpp/build
  # GGML_CUDA_NO_VMM: Kaggle's image has no libcuda.so for the linker, and this is the
  # documented switch for building without it. It only changes how GPU memory is pooled.
  cmake /tmp/llama.cpp -B /tmp/llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
    -DGGML_CUDA_NO_VMM=ON > /tmp/cmake.log 2>&1 || { tail -30 /tmp/cmake.log; exit 1; }
  cmake --build /tmp/llama.cpp/build --config Release -j --target llama-server > /tmp/build.log 2>&1 \
    || { tail -40 /tmp/build.log; exit 1; }
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
    nvidia-smi > /dev/null 2>&1 || echo "No GPU: turn on the GPU accelerator in the notebook settings."
    describe
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
