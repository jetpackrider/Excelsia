"""One-time build of data/input_words.txt: everyday words Excelsia can READ.

She can only SAY constitution words, but she should understand ordinary English.
This list is the extra reading vocabulary: common words from a sample of the
public TinyStories dataset plus the most common English words from `wordfreq`.

    pip install wordfreq
    python scripts/build_input_words.py path/to/TinyStoriesV2-GPT4-train.txt

(The committed list was built from the first 300 MB of that file.)
"""
import collections
import re
import sys
from pathlib import Path

from wordfreq import top_n_list

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
MIN_STORY_COUNT = 40     # keep story words seen at least this often in the sample
TOP_GENERAL = 5000       # plus this many of the most common English words

root = Path(__file__).resolve().parent.parent
mine = {line.split("\t")[0] for line in (root / "data/excelsia_words.tsv").open()}

counts = collections.Counter()
with open(sys.argv[1], encoding="utf-8", errors="ignore") as f:
    for line in f:
        counts.update(WORD.findall(line.lower().replace("’", "'")))

words = {w for w, n in counts.items() if n >= MIN_STORY_COUNT}
words |= {w for w in top_n_list("en", TOP_GENERAL) if WORD.fullmatch(w)}
words = sorted(w for w in words - mine if len(w) > 1 or w in ("a", "i"))
(root / "data/input_words.txt").write_text("\n".join(words) + "\n", encoding="utf-8")
print(f"{len(words):,} reading-only words written")
