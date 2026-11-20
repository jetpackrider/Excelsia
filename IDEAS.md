# Ideas for later

## Imperium

A second, larger model: Excelsia's older, grander self. She can say any word printed
in any New York constitution ever, including every amendment and all the text that
has since been repealed.

**Her words.** The four full constitutions (1777, 1821, 1846, and 1894 as first
adopted) add about 1,000 words to Excelsia's 3,168, after leaving out words from
editors' notes in the scanned copies. Repealed amendments would add a few hundred
more, mostly legal and money words; those are scattered across the session laws,
Lincoln's *Constitutional History of New York* and the 50 Constitutions timeline,
so gathering them is the slow part.

- About 380 of the new words are ones Excelsia already understands but cannot say,
  including some that matter for conversation: us, can, see, say, here, too, many,
  hope, friends, happiness. ("Us" comes from the Declaration of Independence, which
  the 1777 constitution quotes in full.)
- About 700 are words she has never seen: chancellor, hath, doth, oyer, terminer,
  specie, escheat. Most appear only once or twice.

Sources for the full texts: the 1777 constitution at Yale's Avalon Project; the 1821,
1846 and 1894 texts from the Historical Society of the New York Courts.

**Shared brain.** To say the words Excelsia has never seen, both models need one
shared word list that includes them. That means redoing the English stage (about
2.5 hours on Kaggle) and the conversation stage (about an hour) once; both models
then start from the result. Excelsia's speaking list stays exactly as it is.
Imperium needs her own practice chats.

**How big.** Estimated from Excelsia's measured speed on one Kaggle T4, with the same
amount of reading she had (about 800 million words):

| Size | Kaggle training | Web download | Slower than Excelsia |
|---|---|---|---|
| 19M (Excelsia) | about 3.5 h | 19 MB | 1x |
| 44M (12 layers x 512) | about 8 h on one GPU, 4.5 h on two | 44 MB | 2.4x |
| 67M (12 x 640) | about 12 h on one, 7 h on two | 67 MB | 3.6x |
| 94M (12 x 768) | about 17 h on one, 10 h on two | 94 MB | 5x |

About 45M is the sweet spot: one Kaggle session, a reasonable download, and the bot
still fits the smallest Fly machine if packed as 8-bit. About 95M is the most that
stays free, and it needs training on both GPUs at once (not built yet) and a larger
Fly machine for the bot. Kaggle allows 12-hour sessions and about 30 GPU hours a week.

## Two linked models

Two small models that read each other's hidden states as they write, as in
[neural-pairing](https://github.com/SauersML/neural-pairing): a New Jersey across the
river, or Imperium beside Excelsia. In that work the link passes private information
and lets one model steer the other, but the pair did not reason better than one
model, and one learned to copy the other's wrong answers. Worth doing for fun, not as
a fix for coherence. Two models that share one word list and brain would be much
easier to link than two unrelated ones.
