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

**3. Let her read ordinary conversations.** Before her own practice chats she
reads SODA, about 1.2 million everyday two-person conversations, in the same shape
as her chats: the first speaker is the person, the second answers where she would.
She reads them but never says them, since the speaking mask still holds. The point
is to learn that a reply answers what was just said. This step is optional, and
best done on a GPU (about an hour on Kaggle).

```bash
python -m excelsia.train talk         # download and tokenize SODA (690 MB, once)
python -m excelsia.train converse     # work/pretrain.pt -> work/talker.pt
```

**4. Teach her to be Excelsia.** She starts from `talker.pt` if it exists, else
from `pretrain.pt`. Give the chats she should sound most like with `--then`: she
studies the first set, then the `--then` set last.

```bash
python -m excelsia.train finetune --chats data/chats.jsonl data/chats-talk.jsonl \
    --then data/chats-gpt.jsonl
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

## A page of her own

`web/` is a small website where anyone can talk to her. She runs in the visitor's
browser: the page downloads her once (about 18 MB) and does the arithmetic
itself, so there is no server and nothing typed to her leaves the page. The
scene is a lake in the Adirondacks, and the sky follows her mood.

```bash
python -m excelsia.export --web       # work/excelsia.pt -> web/excelsia.bin
cd web && python -m http.server       # then open http://localhost:8000
```

To publish it, copy the three files in `web/` to any static host. `engine.js` is
the same network as `excelsia/model.py`, written in plain JavaScript and checked
against the Python version.

## Tuning her by hand

Two dials that need no training.

```bash
python -m excelsia.chat --repeat 1                       # she tires of words she just used
python -m excelsia.tweak --down people body --by 2       # edit her weights directly
python -m excelsia.chat --ckpt excelsia-tweaked.pt
```

`tweak` changes 384 of her roughly 18 million numbers so the words you name
become rarer or more common everywhere, and saves the result as a new file. It
is a blunt tool: it changes which words she reaches for, not what she has to
say. Try `--up forest waters trees --by 3` to see it plainly.

## Putting her on Discord

She can live in a Discord channel and answer everyone who talks there. The bot
runs her with plain numpy (no PyTorch), so it fits on the smallest Fly.io machine.

1. In Discord's developer portal, make an application with a bot, turn on
   **Message Content Intent**, and invite it to your server with permission to
   read and send messages.
2. Pack her and send her up:

```bash
python -m excelsia.export        # work/excelsia.pt -> bot/excelsia.npz
bash scripts/fly.sh              # asks for the bot token and channel id once
```

The token is typed blind and goes straight to Fly as a secret. It is never
written to a file in this folder. To update her later, export again and rerun
the script.

In the channel, each person has their own thread with her and her mood is shared
by everyone. Her replies begin with her mood in brackets. Moderators can use
`/reset` to clear her memory and mood.

## Where the training data comes from

- The New York State Constitution (public law).
- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), a public dataset.
- [SODA](https://huggingface.co/datasets/allenai/soda) by Hyunwoo Kim and others
  ([paper](https://arxiv.org/abs/2212.10465)), licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its conversations were
  written by an OpenAI model for that project. She reads its training split in step 3,
  reformatted into her chat shape; conversations that are not two people taking
  turns are left out. It is downloaded at training time, not stored here.
- The practice chats in `data/chats.jsonl` and `data/chats-talk.jsonl`, written by
  DeepSeek's hosted API (the `deepseek-flash` model) in November 2026 with
  `--teacher deepseek`, the second set with `--talk`.
- The practice chats in `data/chats-gpt.jsonl`, written by GPT-6.2 Astra in
  ChatGPT in November 2026 from a prompt asking for longer conversations with
  follow-ups and changes of subject. They were then checked line by line with
  `python -m excelsia.check_chats`.

Every practice chat in this repository is AI-generated.

DeepSeek's API terms allowed using its output to train other models when these
chats were written. OpenAI's usage policies bar using its output to train models
meant as alternatives to or substitutes for its own, which this is not. Terms change, so read the current ones before making more.
Many hosted AI services do not allow this at all. If you would rather no hosted
service be involved, delete `data/chats.jsonl` and write your own with the default
local teacher.

## Status

Tested: the word list, the tokenizer, the speaking mask, and every stage end to
end. English was learned on Kaggle's free T4 GPUs (18 million parameters, all 2.7
million stories, about two and a half hours), then she studied the practice chats
for a few minutes, and she talks.

What we learned about teachers: forcing a teacher's letters through a grammar kept
every word legal but wrecked the meaning, because a word she lacks got swapped for
one spelled alike. Letting the teacher write freely and repairing by meaning fixed
that.

What we learned about her: a model this small exaggerates whatever its practice
chats lean on. The first set used "the people" in over half its replies, and she
used it in three quarters of hers. The chat writer now steers the teacher away
from any word that starts turning up too often.

## Layout

```
excelsia/words.py       constitution text -> word list
excelsia/tokenizer.py   vocabulary and the speaking mask
excelsia/model.py       a small GPT
excelsia/train.py       stories / pretrain / talk / converse / finetune
excelsia/gen_chats.py   practice chats written by a teacher model
excelsia/check_chats.py check chats written elsewhere before she trains on them
excelsia/chat.py        talk to her; mood tracker
excelsia/tweak.py       turn words up or down by editing her weights
excelsia/lite.py        the same model in plain numpy, for small servers
excelsia/export.py      pack her into the file lite.py reads
bot/                    the Discord bot
web/                    her website; she runs in the browser
Dockerfile, fly.toml    how the bot is run on Fly.io
scripts/kaggle.sh       the same pipeline on a free Kaggle GPU
kaggle/excelsia.ipynb   notebook that calls it
data/                   constitution text, word lists, vocabulary
tests/                  the checks
```
