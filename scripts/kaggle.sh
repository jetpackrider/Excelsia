#!/usr/bin/env bash
# Run Excelsia's pipeline on a free Kaggle GPU notebook.
#
#   bash scripts/kaggle.sh all        # everything, start to finish (what the notebook runs)
#
# `all` reads its settings from the environment:
#   EXCELSIA_CHATS=300      how many practice chats Gemma writes
#   EXCELSIA_MINUTES=0      minutes of English training; 0 = only write chats
#   EXCELSIA_SIZE=medium    model size for the English training
#   EXCELSIA_STORIES=0      how many stories to learn from; 0 = all of them
# With two GPUs it trains on one while Gemma writes chats on the other.
#
# The single stages still work on their own: check, chats N, english MIN SIZE STORIES,
# excelsia, pack.
#
# The teacher model is open-weight Gemma (Apache 2.0), downloaded and run on the
# notebook's own GPUs. No hosted AI service is called.
#
# Two engines can serve Gemma. llama.cpp's own server is the default: when it enforces
# the word-list grammar it checks only the word Gemma picked, where the ready-built
# Python engine checks her whole vocabulary for every word and pins a CPU core.
# Set EXCELSIA_ENGINE=python to skip the ten-minute build and use the slower one.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_URL="https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/gemma-4-12b-it-Q4_K_M.gguf"
MODEL="${EXCELSIA_MODEL:-/tmp/gemma-4-12b-it-Q4_K_M.gguf}"   # 7.1 GB, kept out of the saved output
ENGINE="${EXCELSIA_ENGINE:-llama.cpp}"
LLAMA_BIN=/tmp/llama.cpp/build/bin/llama-server
PORT=8080
CTX="${EXCELSIA_CTX:-8192}"    # room for the word list plus a short conversation
OUT="${EXCELSIA_OUT:-/kaggle/working/excelsia_out}"
PIDS=()
ONE_COPY=""

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
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "no NVIDIA GPU found"
  python -c "import sys, torch; print('python', sys.version.split()[0], '| torch', torch.__version__, '| torch CUDA', torch.version.cuda)" 2>/dev/null || true
  nvcc --version 2>/dev/null | grep -i release || echo "nvcc: not found"
}

has_gpu_engine() {
  # No output hidden: if the ready-built engine can't load, we want to see why.
  python -c "import llama_cpp, sys; ok = llama_cpp.llama_supports_gpu_offload(); print('ready-built engine sees the GPU:', ok); sys.exit(0 if ok else 1)"
}

python_engine_ok() {
  has_gpu_engine && return 0
  [ -n "${EXCELSIA_ALLOW_CPU:-}" ] && python -c "import llama_cpp"  # for testing off Kaggle
}

build_llama() {
  (apt-get -qq update; apt-get -qq install -y build-essential cmake) > /tmp/apt.log 2>&1 || true
  [ -d /tmp/llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp || return 1
  rm -rf /tmp/llama.cpp/build
  # GGML_CUDA_NO_VMM: Kaggle's image has no libcuda.so for the linker, and this is the
  # documented switch for building without it. It only changes how GPU memory is pooled.
  cmake /tmp/llama.cpp -B /tmp/llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
    -DGGML_CUDA_NO_VMM=ON > /tmp/cmake.log 2>&1 || { tail -30 /tmp/cmake.log; return 1; }
  # A fixed number of compile jobs. A bare -j starts every job at once and runs out of memory.
  cmake --build /tmp/llama.cpp/build --config Release -j "$(nproc)" --target llama-server > /tmp/build.log 2>&1 &
  local bp=$!
  while kill -0 "$bp" 2>/dev/null; do sleep 60; echo "  still building ($(wc -l < /tmp/build.log) lines of compiler output so far)"; done
  wait "$bp" || { tail -40 /tmp/build.log; return 1; }
  [ -x "$LLAMA_BIN" ]
}

install_wheel() {
  if python_engine_ok 2>/dev/null; then return 0; fi
  local major index
  major=$(python -c "import torch; print((torch.version.cuda or '12').split('.')[0])" 2>/dev/null || echo 12)
  index="cu124"; [ "$major" = "13" ] && index="cu130"
  local url="https://abetlen.github.io/llama-cpp-python/whl/$index"
  # Wheels only: never let pip quietly compile a slow CPU-only copy.
  pip install -q --force-reinstall --no-deps --only-binary=:all: llama-cpp-python --index-url "$url" \
    && pip install -q --only-binary=llama-cpp-python "llama-cpp-python[server]" --extra-index-url "$url" \
    || echo "ready-built engine did not install"
  python_engine_ok
}

install_engine() {
  cuda_env
  if [ -x "$LLAMA_BIN" ]; then return; fi
  describe
  if [ "$ENGINE" = "llama.cpp" ]; then
    echo "== building the llama.cpp engine (about 10 minutes, once per session)"
    if build_llama; then echo "engine: llama.cpp"; return; fi
    echo "The build failed. Falling back to the ready-built engine, which is slower with the grammar."
  fi
  echo "== installing the ready-built engine"
  install_wheel || { echo "No working model engine. Paste everything above to get help."; exit 1; }
  echo "engine: ready-built"
  ONE_COPY=1  # this engine needs far more memory per copy, so never run two of it
}

get_model() {
  if [ ! -f "$MODEL" ]; then
    echo "== downloading Gemma (7.1 GB)"
    curl -L --fail --silent --show-error -o "$MODEL.part" "$MODEL_URL" && mv "$MODEL.part" "$MODEL"
  fi
}

start_server() {  # start_server <gpu number> <port>
  local gpu="$1" port="$2"
  if [ -x "$LLAMA_BIN" ]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$LLAMA_BIN" -m "$MODEL" -c "$CTX" -np 1 -ngl 99 --reasoning-budget 0 \
      --host 127.0.0.1 --port "$port" > "/tmp/server$gpu.log" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$gpu" python -m llama_cpp.server --model "$MODEL" --n_ctx "$CTX" \
      --n_gpu_layers -1 --host 127.0.0.1 --port "$port" > "/tmp/server$gpu.log" 2>&1 &
  fi
  PIDS+=($!)
}

wait_for_server() {  # wait_for_server <gpu number> <port> <pid>
  for _ in $(seq 1 300); do
    if curl -sf -m 2 "http://127.0.0.1:$2/v1/models" > /dev/null; then echo "copy $1 is up"; return; fi
    if ! kill -0 "$3" 2>/dev/null; then
      echo "copy $1 stopped while starting. Last lines of its log:"; tail -25 "/tmp/server$1.log"; exit 1
    fi
    sleep 2
  done
  echo "copy $1 never came up. Last lines of its log:"; tail -25 "/tmp/server$1.log"; exit 1
}

stop_servers() { for pid in "${PIDS[@]:-}"; do [ -n "$pid" ] && kill "$pid" 2>/dev/null || true; done; }

gpu_count() { { nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true; } | wc -l; }

run_chats() {  # run_chats <how many> <gpu numbers...>   one copy of Gemma per GPU listed
  local n="$1"; shift
  local gpus=("$@") g i=0
  mkdir -p "$OUT"
  install_engine
  if [ -n "${ONE_COPY:-}" ] && [ "${#gpus[@]}" -gt 1 ]; then gpus=("${gpus[0]}"); fi
  get_model
  trap stop_servers EXIT
  echo "== starting ${#gpus[@]} copy/copies of the model server"
  for g in "${gpus[@]}"; do start_server "$g" $((PORT + g)); done
  for g in "${gpus[@]}"; do wait_for_server "$g" $((PORT + g)) "${PIDS[$i]}"; i=$((i + 1)); done
  local each=$(( (n + ${#gpus[@]} - 1) / ${#gpus[@]} )) writers=()
  for g in "${gpus[@]}"; do
    # Written straight into the folder Kaggle keeps, one chat at a time.
    python -m excelsia.gen_chats --n "$each" --server "http://127.0.0.1:$((PORT + g))" \
      --out "$OUT/chats.part$g.jsonl" &
    writers+=($!)
  done
  for w in "${writers[@]}"; do wait "$w" || echo "one writer stopped early (see above)"; done
  stop_servers
  merge_chats
}

merge_chats() {
  EXCELSIA_OUT="$OUT" python - <<'PY'
import glob, json, os, random
out = os.environ["EXCELSIA_OUT"]
main = os.path.join(out, "chats.jsonl")
kept, uses, seen_chats = [], {}, set()
files = ([main] if os.path.exists(main) else []) + sorted(glob.glob(os.path.join(out, "chats.part*.jsonl")))
for path in files:
    for line in open(path, encoding="utf-8"):
        if not line.strip() or line in seen_chats:
            continue
        try:
            chat = json.loads(line)
        except ValueError:
            continue  # a line cut off mid-write
        replies = [t["reply"] for t in chat["turns"]]
        if any(uses.get(r, 0) >= 2 for r in replies):  # the same line said too often across copies
            continue
        for r in replies:
            uses[r] = uses.get(r, 0) + 1
        seen_chats.add(line)
        kept.append(line if line.endswith("\n") else line + "\n")
open(main, "w", encoding="utf-8").writelines(kept)
open("data/chats.jsonl", "w", encoding="utf-8").writelines(kept)
for path in glob.glob(os.path.join(out, "chats.part*.jsonl")):
    os.remove(path)
print(f"\n== {len(kept)} chats saved. A random sample to read:")
for line in random.sample(kept, min(15, len(kept))):
    chat = json.loads(line)
    print(f"--- {chat['topic']}")
    for t in chat["turns"]:
        print(f"  person: {t['user']}\n  excelsia [{t['mood']}]: {t['reply']}")
PY
}

run_english() {  # run_english <minutes> <size> <stories>
  python -m excelsia.train stories --n "$3"
  python -m excelsia.train pretrain --size "$2" --minutes "$1" --steps 2000000 --batch 64
}

run_excelsia() {
  python -m excelsia.train finetune &&
  python -m excelsia.chat --say "hello" --say "who are you?" --say "is your name excelsia?" \
    --say "you are wonderful, thank you" --say "what do you like?" \
    --say "you are stupid and i hate you" --say "sorry. tell me about your forests" \
    --say "i have to go now, goodbye"
}

pack() {
  mkdir -p "$OUT"
  cp work/pretrain.pt work/excelsia.pt "$OUT"/ 2>/dev/null || true
  echo "== saved in $OUT:"; ls -lh "$OUT" | tail -n +2
}

stage="${1:-help}"; shift || true
case "$stage" in
  check)
    nvidia-smi > /dev/null 2>&1 || echo "No GPU: turn on the GPU accelerator in the notebook settings."
    describe
    python -m tests.test_excelsia
    ;;
  chats)
    n="${1:-300}"
    copies="${EXCELSIA_COPIES:-$(gpu_count)}"
    [ "$copies" -lt 1 ] && copies=1
    [ "$copies" -gt 2 ] && copies=2
    run_chats "$n" $(seq 0 $((copies - 1)))
    ;;
  english)
    run_english "${1:-180}" "${2:-medium}" "${3:-1000000}"
    ;;
  excelsia)
    run_excelsia
    ;;
  pack)
    pack
    ;;
  all)
    n="${EXCELSIA_CHATS:-300}"; minutes="${EXCELSIA_MINUTES:-0}"
    size="${EXCELSIA_SIZE:-medium}"; stories="${EXCELSIA_STORIES:-0}"
    mkdir -p "$OUT"
    describe
    python -m tests.test_excelsia
    gpus="${EXCELSIA_COPIES:-$(gpu_count)}"
    en_pid=""
    if [ "$minutes" != "0" ] && [ "$gpus" -ge 2 ]; then
      echo "== English training started on the first GPU ($minutes minutes). Its log: $OUT/english.log"
      ( export CUDA_VISIBLE_DEVICES=0; run_english "$minutes" "$size" "$stories" ) > "$OUT/english.log" 2>&1 &
      en_pid=$!
      chat_gpus="1"
    elif [ "$gpus" -ge 2 ]; then
      chat_gpus="0 1"
    else
      chat_gpus="0"
      if [ "$minutes" != "0" ]; then run_english "$minutes" "$size" "$stories"; fi
    fi
    chats_ok=1
    ( run_chats "$n" $chat_gpus ) || { chats_ok=0; echo "!! Writing the chats failed. See the messages above."; }
    if [ -n "$en_pid" ]; then
      while kill -0 "$en_pid" 2>/dev/null; do
        echo "english: $(tail -n 1 "$OUT/english.log" 2>/dev/null)"; sleep 120
      done
      wait "$en_pid" || { echo "!! English training failed. Last lines of its log:"; tail -30 "$OUT/english.log"; }
      grep "check:" "$OUT/english.log" | tail -3 || true
    fi
    if [ -f work/pretrain.pt ] && [ "$chats_ok" = 1 ] && [ -f data/chats.jsonl ]; then
      run_excelsia || echo "!! The last training step failed. See above."
    elif [ "$minutes" != "0" ]; then
      echo "Skipping the last step: it needs both the English model and the chats."
    fi
    pack
    ;;
  *)
    sed -n '2,16p' "$0"
    ;;
esac
