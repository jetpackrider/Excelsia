"""Checks that matter: she can never say a word outside the constitution.

    python -m tests.test_excelsia        (or: pytest)
"""
import random

import torch

from excelsia.gen_chats import build_grammar, usable, word_rules, write_reply
from excelsia.model import GPT, Config
from excelsia.tokenizer import DATA, Tokenizer
from excelsia.train import chat_batcher, constitution_ids, encode_chat, lm_batcher
from excelsia.words import count_words

tok = Tokenizer()


def test_word_list_matches_the_text():
    text = (DATA / "ny_constitution.txt").read_text(encoding="utf-8")
    assert set(count_words(text)) == tok.speak_words


def test_reading_and_speaking_words_do_not_overlap():
    assert not tok.speak_words & set(tok.vocab["read_only"])
    assert len(tok) < 65536  # token ids are stored as 16-bit numbers


def test_what_she_can_and_cannot_say():
    assert tok.can_say("I do solemnly affirm.")[0]
    assert tok.can_say("The forest of New York is forever wild!")[0]
    for word in ["you", "yes", "hello", "love", "happy", "sad", "me", "am", "excelsia"]:
        assert not tok.can_say(word)[0], word


def test_she_knows_her_name_but_cannot_say_it():
    assert tok.encode("excelsia") != [tok.unk]
    assert not tok.speak_mask[tok.stoi["excelsia"]]
    assert tok.can_say("I am New York.") == (False, ["am"])
    assert tok.can_say("I, New York.")[0]


def test_round_trip():
    ids = tok.encode("i shall be grateful forever . the pain is cruel and unusual !")
    assert tok.unk not in ids
    assert tok.decode(ids) == "I shall be grateful forever. The pain is cruel and unusual!"


def test_unknown_words_are_read_as_unknown():
    assert tok.encode("zzyzx") == [tok.unk]


def test_generation_cannot_leave_the_word_list():
    """Even a model with random weights, sampled hot, stays inside the list."""
    torch.manual_seed(1)
    model = GPT(Config(len(tok), block_size=64, n_layer=2, n_head=2, n_embd=64))
    allowed = torch.tensor(tok.speak_mask)
    total = 0
    for _ in range(40):
        start = torch.tensor([[tok.user] + tok.encode("hello, do you love me?") + [tok.excelsia]])
        out = model.generate(start, allowed, tok.end, max_new=80, temperature=2.0, top_k=0)
        assert all(tok.speak_mask[i] for i in out)
        ok, bad = tok.can_say(tok.decode(out))
        assert ok, bad
        total += len(out)
    assert total > 1000


def test_chat_loss_is_only_on_her_words():
    chat = {"turns": [{"user": "hello you", "mood": "peace", "reply": "peace ."}]}
    x, y = encode_chat(tok, chat, 64)
    assert len(x) == len(y)
    wanted = [t for t in y if t != -100]
    assert wanted == tok.encode("peace .") + [tok.end]
    xb, yb = chat_batcher([(x, y)] * 4, 4, tok.pad, "cpu")()
    assert xb.shape == yb.shape


def test_training_makes_progress():
    torch.manual_seed(0)
    random.seed(0)
    law = constitution_ids(tok)
    model = GPT(Config(len(tok), block_size=32, n_layer=2, n_head=2, n_embd=64, dropout=0.0))
    get = lm_batcher([(law, 1.0)], 32, 16, tok.unk, "cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = None
    for _ in range(60):
        _, loss = model(*get())
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
        last = loss.item()
    assert last < first - 1.0, (first, last)


def test_bad_practice_chats_are_rejected():
    ok = lambda user, reply, turns=(), seen=None: usable(tok, user, reply, list(turns), seen or {})
    assert ok("hello there", "peace, the people are together.")
    assert not ok("hello there", "hello, i love you.")                # forbidden words
    assert not ok("hello there", "peace and the people are")          # cut off mid-sentence
    assert not ok("hello there", "excise excise excise excise.")      # stuck in a loop
    assert not ok("", "peace.")
    before = [{"user": "hi", "mood": "peace", "reply": "peace."}]
    assert not ok("Peace.", "the forest is wild.", before)            # teacher parroted her
    assert not ok("and now?", "peace.", before)                       # same reply twice in a chat
    assert not ok("HI", "the forest is wild.", before)                # person repeated themselves
    assert not ok("hello", "peace.", (), {"peace.": 2})               # overused across chats
    assert not ok("I shall be leaving now.", "i shall be leaving now.")  # echoes the person
    assert not ok("hello", "the forest is wild. the forest is wild.")  # same sentence twice
    assert not ok("more?", "the forest is wild. peace shall remain.",
                  [{"user": "hi", "mood": "peace", "reply": "peace shall remain."}])  # reused sentence


class FakeTeacher:
    """Stands in for a model: hands back prepared lines and records what it was asked."""

    def __init__(self, lines):
        self.lines, self.asked = list(lines), []

    def ask(self, messages, temperature, max_tokens, grammar=None):
        self.asked.append(messages)
        return self.lines.pop(0)


def test_a_bad_word_is_sent_back_for_repair_by_meaning():
    teacher = FakeTeacher(['"I am happy, thank you."', "i shall be grateful."])
    reply, tries = write_reply(teacher, tok, "system", "task", tries=3)
    assert (reply, tries) == ("i shall be grateful.", 2)
    repair = teacher.asked[1][-1]["content"]
    assert "am" in repair and "happy" in repair and "MEANING" in repair
    assert teacher.asked[1][-2] == {"role": "assistant", "content": "i am happy, thank you."}


def test_a_line_that_never_passes_is_dropped():
    teacher = FakeTeacher(["hello you."] * 3)
    assert write_reply(teacher, tok, "system", "task", tries=3) == ("", 3)


def test_grammar_spells_exactly_her_words():
    """Walk the grammar's prefix tree and check it spells her word list and nothing else."""
    rules, start = word_rules(tok.vocab["speak"])
    spelled = set()

    def walk(name, prefix):
        branches, may_stop = rules[name]
        if may_stop:
            spelled.add(prefix)
        for letters, nxt in branches:
            if nxt:
                walk(nxt, prefix + letters)
            else:
                spelled.add(prefix + letters)

    walk(start, "")
    assert spelled == tok.speak_words
    grammar = build_grammar(tok.vocab["speak"])
    assert grammar.startswith("root ::=") and "word ::= " + start in grammar


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
