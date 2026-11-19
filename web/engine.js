/* Excelsia, running in the browser.
 *
 * This is the same small network as excelsia/model.py, written in plain JavaScript.
 * It runs in a Web Worker so the page stays smooth while she thinks. Nothing typed
 * to her leaves the browser: there is no server behind this.
 *
 * The rule that matters is kept here too: before she picks each word, every word
 * that is not printed in the New York State Constitution has its score set to
 * minus infinity, so she cannot say it.
 */
"use strict";

const MOODS = ["grateful", "pleasure", "peace", "alone", "pain", "offense"];
const SPEAK_PUNCT = [".", ",", "?", "!", ";", ":"];
const KIND = new Set(("thank thanks love lovely like nice good great kind sweet friend friends happy " +
  "glad wonderful beautiful pretty best awesome amazing proud please sorry welcome hug cute " +
  "smart brave care gentle fun funny yay").split(" "));
const UNKIND = new Set(("hate stupid dumb ugly bad worst shut boring useless idiot terrible awful " +
  "annoying horrible gross mean liar silly wrong hush lame pathetic worthless").split(" "));
const LEAVING = new Set("bye goodbye leaving leave gone goodnight later farewell".split(" "));
const TOKEN = /<[a-z:]+>|[a-z]+(?:'[a-z]+)?|[0-9]+|[.,?!;:'"()\-]/g;
const FOLD = { "‘": "'", "’": "'", "“": '"', "”": '"', "—": "-", "–": "-" };
const REMEMBER = 2;   // earlier exchanges she keeps in view
const MAX_NEW = 48;   // the longest reply she may give, in words and marks

/* ---------------------------------------------------------------- her mood */

class Mood {
  constructor() { this.level = 0; this.left = false; }
  hear(text) {
    const words = text.toLowerCase().match(/[a-z']+/g) || [];
    this.level *= 0.7;                                   // feelings fade
    for (const w of words) {
      if (KIND.has(w)) this.level += 0.45;
      if (UNKIND.has(w)) this.level -= 0.55;
    }
    this.level = Math.max(-1, Math.min(1, this.level));
    this.left = words.some((w) => LEAVING.has(w));
    return this.name;
  }
  get name() {
    if (this.left && this.level > -0.2) return "alone";   // a goodbye, unless she is already hurt
    if (this.level > 0.6) return "pleasure";
    if (this.level > 0.2) return "grateful";
    if (this.level < -0.6) return "offense";
    if (this.level < -0.2) return "pain";
    return "peace";
  }
}

/* ---------------------------------------------------------------- her words */

class Tokenizer {
  constructor(vocab) {
    this.itos = [].concat(vocab.specials, vocab.punct, vocab.speak, vocab.read_only);
    this.stoi = new Map(this.itos.map((t, i) => [t, i]));
    this.display = vocab.display;
    this.unk = this.stoi.get("<unk>");
    this.user = this.stoi.get("<user>");
    this.excelsia = this.stoi.get("<excelsia>");
    this.end = this.stoi.get("<end>");
    const allowed = new Set([].concat(vocab.speak, SPEAK_PUNCT, ["<end>"]));
    this.speakIds = [];                                  // the only ids she may ever produce
    this.itos.forEach((t, i) => { if (allowed.has(t)) this.speakIds.push(i); });
    this.punct = SPEAK_PUNCT.map((p) => this.stoi.get(p));
  }
  mood(name) { return this.stoi.get("<mood:" + name + ">"); }
  split(text) {
    const folded = text.toLowerCase().replace(/[‘’“”—–]/g, (c) => FOLD[c]);
    return folded.match(TOKEN) || [];
  }
  encode(text) {
    return this.split(text).map((t) => {
      let id = this.stoi.get(t);
      if (id === undefined && t.endsWith("'s")) id = this.stoi.get(t.slice(0, -2));
      return id === undefined ? this.unk : id;
    });
  }
  decode(ids) {
    const out = [];
    let start = true;
    for (const i of ids) {
      const t = this.itos[i];
      if (t === "<pad>" || t === "<end>" || t === "<excelsia>" || t.startsWith("<mood:")) continue;
      if (".,?!;:)".includes(t) && t.length === 1) {
        out.push(t);
        start = start || ".?!".includes(t);
        continue;
      }
      let word = Object.prototype.hasOwnProperty.call(this.display, t) ? this.display[t] : t;
      if (start && /^[a-zA-Z]/.test(word)) word = word[0].toUpperCase() + word.slice(1);
      out.push((out.length && out[out.length - 1] !== "(" ? " " : "") + word);
      start = false;
    }
    return out.join("").trim().replace(/\b[Nn]ew York\b/g, "New York");
  }
}

/* ---------------------------------------------------------------- her network */

function erf(x) {                                        // Abramowitz and Stegun 7.1.26
  const s = x < 0 ? -1 : 1, a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  const p = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return s * (1 - p * Math.exp(-a * a));
}

function norm(x, weight, bias, out) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) { const d = x[i] - mean; v += d * d; }
  const inv = 1 / Math.sqrt(v / n + 1e-5);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * weight[i] + bias[i];
  return out;
}

/* out = W x, where W has `rows` rows of x.length numbers, stored row after row. */
function matvec(W, x, rows, out, bias) {
  const n = x.length;
  for (let r = 0, o = 0; r < rows; r++, o += n) {
    let a = 0, b = 0, c = 0, d = 0, i = 0;
    for (; i + 3 < n; i += 4) {
      a += W[o + i] * x[i]; b += W[o + i + 1] * x[i + 1];
      c += W[o + i + 2] * x[i + 2]; d += W[o + i + 3] * x[i + 3];
    }
    for (; i < n; i++) a += W[o + i] * x[i];
    out[r] = a + b + c + d + (bias ? bias[r] : 0);
  }
  return out;
}

class Model {
  constructor(buffer) {
    const bytes = new Uint8Array(buffer);
    if (String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]) !== "EXW1") throw new Error("not an Excelsia file");
    const headerLength = new DataView(buffer).getUint32(4, true);
    const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + headerLength)));
    const base = 8 + headerLength;
    this.config = header.config;
    this.vocab = header.vocab;
    this.w = {};
    for (const t of header.tensors) {
      const size = t.shape.reduce((a, b) => a * b, 1);
      if (t.kind === "f32") {
        this.w[t.name] = new Float32Array(buffer.slice(base + t.data, base + t.data + 4 * size));
      } else {                                           // int8 with one scale per row
        const rows = t.shape[0], cols = t.shape[1];
        const scale = new Float32Array(buffer.slice(base + t.scale, base + t.scale + 4 * rows));
        const q = new Int8Array(buffer, base + t.data, size);
        const full = new Float32Array(size);
        for (let r = 0, o = 0; r < rows; r++) {
          const s = scale[r];
          for (let c = 0; c < cols; c++, o++) full[o] = q[o] * s;
        }
        this.w[t.name] = full;
      }
    }
    const c = this.config;
    this.C = c.n_embd; this.H = c.n_head; this.D = this.C / this.H;
    this.layers = c.n_layer; this.block = c.block_size;
    // working space, made once
    this.x = new Float32Array(this.C); this.h = new Float32Array(this.C);
    this.qkv = new Float32Array(3 * this.C); this.y = new Float32Array(this.C);
    this.p = new Float32Array(this.C); this.m = new Float32Array(4 * this.C);
    this.att = new Float32Array(this.block);
    this.keys = []; this.vals = [];
    for (let i = 0; i < this.layers; i++) {
      this.keys.push(new Float32Array(this.block * this.C));
      this.vals.push(new Float32Array(this.block * this.C));
    }
    this.seen = 0;                                       // positions already in the cache
  }

  reset() { this.seen = 0; }

  /* Feed one token at the next position. Leaves the final hidden state in this.h. */
  feed(id) {
    const { w, C, H, D, x, h, qkv, y, p, m, att } = this;
    const pos = this.seen, te = w["tok_emb.weight"], pe = w["pos_emb.weight"];
    for (let i = 0; i < C; i++) x[i] = te[id * C + i] + pe[pos * C + i];
    for (let l = 0; l < this.layers; l++) {
      const pre = "blocks." + l + ".", K = this.keys[l], V = this.vals[l];
      norm(x, w[pre + "ln1.weight"], w[pre + "ln1.bias"], h);
      matvec(w[pre + "qkv.weight"], h, 3 * C, qkv);
      K.set(qkv.subarray(C, 2 * C), pos * C);            // remember this position's key and value
      V.set(qkv.subarray(2 * C, 3 * C), pos * C);
      const scale = 1 / Math.sqrt(D);
      for (let hd = 0; hd < H; hd++) {
        const off = hd * D;
        let max = -Infinity;
        for (let t = 0; t <= pos; t++) {
          let s = 0;
          const ko = t * C + off;
          for (let i = 0; i < D; i++) s += qkv[off + i] * K[ko + i];
          s *= scale; att[t] = s; if (s > max) max = s;
        }
        let sum = 0;
        for (let t = 0; t <= pos; t++) { att[t] = Math.exp(att[t] - max); sum += att[t]; }
        for (let i = 0; i < D; i++) y[off + i] = 0;
        for (let t = 0; t <= pos; t++) {
          const a = att[t] / sum, vo = t * C + off;
          for (let i = 0; i < D; i++) y[off + i] += a * V[vo + i];
        }
      }
      matvec(w[pre + "proj.weight"], y, C, p);
      for (let i = 0; i < C; i++) x[i] += p[i];
      norm(x, w[pre + "ln2.weight"], w[pre + "ln2.bias"], h);
      matvec(w[pre + "mlp.0.weight"], h, 4 * C, m, w[pre + "mlp.0.bias"]);
      for (let i = 0; i < 4 * C; i++) m[i] = 0.5 * m[i] * (1 + erf(m[i] * Math.SQRT1_2));
      matvec(w[pre + "mlp.2.weight"], m, C, p, w[pre + "mlp.2.bias"]);
      for (let i = 0; i < C; i++) x[i] += p[i];
    }
    norm(x, w["ln_f.weight"], w["ln_f.bias"], h);
    this.seen = pos + 1;
  }

  /* Scores for the next word, worked out only for the ids listed. */
  scores(ids, out) {
    const te = this.w["tok_emb.weight"], C = this.C, h = this.h;
    for (let k = 0; k < ids.length; k++) {
      let s = 0;
      const o = ids[k] * C;
      for (let i = 0; i < C; i++) s += te[o + i] * h[i];
      out[k] = s;
    }
    return out;
  }
}

class Excelsia {
  constructor(buffer) {
    this.model = new Model(buffer);
    this.tok = new Tokenizer(this.model.vocab);
    this.mood = new Mood();
    this.turns = [];
    this.fed = [];                                       // the tokens now sitting in the model's cache
    this.scratch = new Float64Array(this.tok.speakIds.length);
  }

  forget() { this.turns = []; this.fed = []; this.mood = new Mood(); }

  /* Reply to `text`. `onWord(textSoFar)` is called as each word arrives. */
  reply(text, { temperature = 0.8, topK = 40, random = Math.random, onWord = null } = {}) {
    const tok = this.tok, model = this.model, ids = tok.speakIds, s = this.scratch;
    const mood = this.mood.hear(text);
    const asked = [tok.user].concat(tok.encode(text).slice(0, 120), [tok.mood(mood), tok.excelsia]);
    let prompt = [].concat(...this.turns.slice(-REMEMBER), asked);
    prompt = prompt.slice(-(model.block - MAX_NEW));     // she only sees what fits
    // Whatever she has already read at the same positions need not be read again.
    let same = 0;
    while (same < this.fed.length && same < prompt.length - 1 && this.fed[same] === prompt[same]) same++;
    model.seen = same;
    for (let i = same; i < prompt.length; i++) model.feed(prompt[i]);
    const out = [];
    for (let step = 0; step < MAX_NEW; step++) {
      model.scores(ids, s);
      for (let k = 0; k < ids.length; k++) s[k] /= Math.max(temperature, 1e-5);
      const banned = new Set();
      if (step === 0) { banned.add(tok.end); for (const p of tok.punct) banned.add(p); }   // open with a word
      const n = out.length;
      if (n >= 2 && out[n - 1] === out[n - 2]) banned.add(out[n - 1]);                     // never one word three times running
      if (banned.size) for (let k = 0; k < ids.length; k++) if (banned.has(ids[k])) s[k] = -Infinity;
      // keep the top few, then choose among them by chance
      const order = Array.from(s.keys()).sort((a, b) => s[b] - s[a]).slice(0, topK);
      const max = s[order[0]];
      let total = 0;
      const p = order.map((k) => { const e = Math.exp(s[k] - max); total += e; return e; });
      let r = random() * total, pick = order[order.length - 1];
      for (let j = 0; j < order.length; j++) { r -= p[j]; if (r <= 0) { pick = order[j]; break; } }
      const next = ids[pick];
      if (next === tok.end) break;
      out.push(next);
      if (onWord) onWord(tok.decode(out));
      model.feed(next);
    }
    this.fed = prompt.concat(out);
    this.turns = this.turns.concat([asked.concat(out, [tok.end])]).slice(-12);
    return { mood, text: tok.decode(out) };
  }
}

/* ---------------------------------------------------------------- as a worker */

if (typeof self !== "undefined" && typeof self.postMessage === "function" && typeof window === "undefined") {
  let her = null;
  self.onmessage = (event) => {
    const msg = event.data;
    try {
      if (msg.type === "load") {
        her = new Excelsia(msg.buffer);
        self.postMessage({ type: "ready", words: her.tok.speakIds.length - SPEAK_PUNCT.length - 1 });
      } else if (msg.type === "say" && her) {
        const feeling = new Mood(); feeling.level = her.mood.level; feeling.left = her.mood.left;
        self.postMessage({ type: "mood", mood: feeling.hear(msg.text), id: msg.id });    // the sky can turn before she speaks
        const result = her.reply(msg.text, { onWord: (text) => self.postMessage({ type: "word", text, id: msg.id }) });
        self.postMessage({ type: "done", mood: result.mood, text: result.text, id: msg.id });
      } else if (msg.type === "forget" && her) {
        her.forget();
        self.postMessage({ type: "forgotten" });
      }
    } catch (problem) {
      self.postMessage({ type: "error", message: String(problem && problem.message || problem) });
    }
  };
}

if (typeof module !== "undefined") module.exports = { Excelsia, Model, Tokenizer, Mood };
