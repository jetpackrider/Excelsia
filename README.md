# Excelsia

A tiny language model who believes she is the State of New York and can only
speak words that appear in the New York State Constitution.

She understands ordinary English. She just can't answer in it. She can't say her
own name either, since "Excelsia" isn't in the constitution. Ask who she is and
she will insist she is New York. The constitution
has no "you", "yes", "hello", "love", "happy" or "sad", so she has to work around
all of them. It does have "grateful", "pain", "cruel", "forever", "wild" and
"notwithstanding".

## How the rule is enforced

Her vocabulary has two parts:

| | Count | Source |
|---|---|---|
| Words she can say | 3,168 | Every word in the constitution, and nothing else |
| Words she can only read | 7,154 | Everyday English and her own name, so she can follow a conversation |

When she generates a reply, every token outside the first group has its score
set to minus infinity before sampling. A word outside the constitution is not
filtered out afterwards. It can never be picked in the first place. The test
suite checks this with a random, untrained model sampled as wildly as possible.

The word list comes from the Department of State's edition of the constitution
with amendments effective January 1, 2026. It includes the section headings and
amendment notes printed in that edition, and leaves out the cover page, table of
contents and page furniture. `excelsia/words.py` rebuilds it from the raw text.

## Moods

She has six moods, each named with one of her own words: `grateful`, `pleasure`,
`peace`, `alone`, `pain`, `offense`. The mood is tracked outside the model
(`excelsia/chat.py`) and handed to it as a tag before each reply. Kind words
lift it, unkind words sink it, and it fades back toward calm. Her memory is
short (a few exchanges), so her feelings usually outlast what caused them.

## Building her

Needs Python 3.9+ and about 10 GB of free disk (most of that is the teacher model in step 2). Written for an Apple Silicon Mac
(it uses the GPU through PyTorch's `mps` backend) but runs anywhere PyTorch does.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m tests.test_excelsia
```

**1. Teach her English.** Downloads part of the public TinyStories dataset and
trains on it together with the constitution itself.

```bash
python -m excelsia.train stories
python -m excelsia.train pretrain --minutes 45
python -m excelsia.chat --ckpt pretrain.pt     # she babbles, but only in her words
```

**2. Have a teacher model write practice chats.** The teacher writes each of her
lines freely, a script checks every word against the list, and any word that is
not hers is handed back to be replaced with one of similar meaning. A line is
only kept once every word passes. The default teacher is Gemma running on your
own machine through llama.cpp.

```bash
brew install llama.cpp
llama-server -hf unsloth/gemma-4-12b-it-GGUF:Q4_K_M -c 16384    # leave running
python -m excelsia.gen_chats --n 300                             # in a second window
```

Every exchange is checked before it is kept: forbidden words, replies cut off
mid-sentence, repeated words or sentences, echoes of the person, and any opening
word that starts taking over are all thrown out.

A hosted teacher works too, if its terms allow training on its output. DeepSeek's
API terms do at the time of writing; read the current ones yourself.

```bash
read -s DEEPSEEK_KEY && export DEEPSEEK_KEY      # paste the key; it is never stored
python -m excelsia.gen_chats --n 300 --teacher deepseek
```

Read `data/chats.jsonl` before going further. If the chats are bad, the model
will be too.

**3. Teach her to be Excelsia.**

```bash
python -m excelsia.train finetune
python -m excelsia.chat
```

## Running it on a free Kaggle GPU

The laptop is enough to prove everything works. A free Kaggle notebook is faster
and can run Gemma without a 7 GB download at home.

1. On Kaggle, create a notebook from `kaggle/excelsia.ipynb`, set the accelerator
   to GPU T4 x2 and turn internet on.
2. Set the two numbers in its one code box: how many chats to write, and how many
   minutes of English training (0 = only write chats).
3. Click Save Version, choose Save & Run All, and save. It runs in the background
   and the results land in the version's output, in `excelsia_out`.

Run it in the background, not in the live editor: a long job in the editor dies
if the tab loses its connection. `scripts/kaggle.sh all` does the work. With two
GPUs it trains on one while Gemma writes chats on the other, and chats are saved
one at a time as they are written.

## Where the training data comes from

- The New York State Constitution (public law).
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), a public dataset.
- The 3,000 practice chats in `data/chats.jsonl`, written by DeepSeek's hosted API
  (the `deepseek-flash` model) in November 2026 with `--teacher deepseek`.

DeepSeek's API terms allowed using its output to train other models when these
chats were written. Terms change, so read the current ones before making more.
Many hosted AI services do not allow this at all. If you would rather no hosted
service be involved, delete `data/chats.jsonl` and write your own with the default
local teacher.

## Status

Tested: the word list, the tokenizer, the speaking mask, and the English stage on
an Apple Silicon laptop (about 9,500 words a second for the small model). The
Kaggle script has run end to end on Kaggle's free T4 GPUs. The chat writer has
produced the 3,000 chats here, and every reply in them passes the word check.

What we learned about teachers: forcing Gemma's letters through a grammar kept
every word legal but wrecked the meaning, because a word she lacks got swapped
for one spelled alike. Letting the teacher write freely and repairing by meaning
fixed that. Gemma has not been tried with the new method yet.

Not yet tested: the last training step on these chats, so how well she herself
talks is still the open question.

## Layout

```
excelsia/words.py       constitution text -> word list
excelsia/tokenizer.py   vocabulary and the speaking mask
excelsia/model.py       a small GPT
excelsia/train.py       stories / pretrain / finetune
excelsia/gen_chats.py   practice chats from a local teacher model
excelsia/chat.py        talk to her; mood tracker
scripts/kaggle.sh       the same pipeline on a free Kaggle GPU
kaggle/excelsia.ipynb   notebook that calls it
data/                   constitution text, word lists, vocabulary
tests/                  the checks
```
