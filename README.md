# warp_pos_spacy

A tiny structural diversity meter for generated text.

A speed nerd met spaCy and said... what if we did it this way?

`warp_pos_spacy` keeps spaCy where it is good: tokenization, POS tagging, dependency parsing, and linguistic judgment. It moves the boring hot loop into Rust: hash small POS windows, count repeated shapes, and report whether a generated corpus is drifting into the same hidden template over and over.

It is small on purpose. It is not a parser, not an embedding model, and not a synthetic-data platform. It is a pressure gauge for template collapse.

## Why This Exists

Generated text often looks varied while repeating the same move underneath:

```text
PRON AUX PART VERB DET NOUN
PRON AUX PART VERB DET NOUN
PRON AUX PART VERB DET NOUN
```

For hard negation data, that repetition made examples weaker. A model could learn the template instead of learning the semantic hinge. The same failure shows up in other places too:

- synthetic training data that quietly reuses one sentence shape
- eval questions that all test the same trick
- RAG hard negatives that differ lexically but not structurally
- prompt batches where the model falls into a phrasing groove
- agent traces that repeat the same action pattern

`warp_pos_spacy` gives you a cheap way to notice.

## What It Measures

The crate treats a fixed POS window `[-2, +3]` around an anchor as a structural "species." It then scores the species distribution.

The high-level APIs return:

- `total_windows`: number of windows scored
- `unique_patterns`: number of distinct structural patterns
- `simpson_d`: Simpson dominance, where higher means one pattern is taking over
- `shannon_h`: Shannon entropy, where higher means broader variety
- `dominant_share`: share held by the most common pattern
- `collapse_pressure`: currently the same as Simpson dominance, exposed under a plain name
- `top_patterns`: readable pattern/count/share diagnostics

The older tuple API remains available for callers that only need raw speed and hashes.

## Install For Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --release
python examples/pretagged.py
```

To build a wheel without activating the environment:

```bash
python -m maturin build --release
pip install target/wheels/*.whl
python examples/pretagged.py
```

## Quick Start

Use `score_anchored` when you already know the important token position: a negation marker, answer span, entity, citation, number, verb, or any domain-specific hinge.

```python
from warp_pos_spacy import score_anchored

pos_sequences = [
    ["PRON", "AUX", "PART", "VERB", "DET", "NOUN"],
    ["PRON", "AUX", "PART", "VERB", "DET", "NOUN"],
    ["NOUN", "AUX", "PART", "VERB", "ADJ", "NOUN"],
]
anchor_indices = [2, 2, 2]

summary = score_anchored(pos_sequences, anchor_indices, top_n=3)
print(summary["dominant_share"])
print(summary["top_patterns"][0])
```

Example output shape:

```python
{
    "total_windows": 3,
    "unique_patterns": 2,
    "simpson_d": 0.5555555555555556,
    "shannon_h": 0.6365141682948128,
    "dominant_hash": 15450595141421276804,
    "dominant_count": 2,
    "dominant_share": 0.6666666666666666,
    "collapse_pressure": 0.5555555555555556,
    "top_patterns": [
        {
            "hash": 15450595141421276804,
            "pattern": "PRON AUX PART VERB DET NOUN",
            "count": 2,
            "share": 0.6666666666666666,
        }
    ],
}
```

Use `score_all_windows` when you do not have anchors and just want a broad structural scan.

```python
from warp_pos_spacy import score_all_windows

summary = score_all_windows(pos_sequences, top_n=5)
for pattern in summary["top_patterns"]:
    print(pattern["count"], pattern["pattern"])
```

## With spaCy

spaCy should still do the linguistic work. This package expects POS tags and optional anchor positions.

```python
import spacy
from warp_pos_spacy import score_anchored

nlp = spacy.load("en_core_web_sm")
texts = [
    "The system does not accept stale credentials.",
    "The policy cannot survive that exception.",
]

pos_sequences = []
anchor_indices = []

for doc in nlp.pipe(texts):
    for i, token in enumerate(doc):
        if token.dep_ == "neg":
            pos_sequences.append([token.pos_ for token in doc])
            anchor_indices.append(i)
            break

print(score_anchored(pos_sequences, anchor_indices))
```

If your spaCy pipeline can use GPU, enable it before loading the model:

```python
import spacy

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")
```

## API

`hash_window(pos_sequence, anchor_idx) -> int`

Returns a deterministic species hash for one POS window. Empty POS arrays or out-of-range anchors raise `ValueError`.

`accumulate_and_score(pos_sequences, anchor_indices) -> tuple[float, float, int, int]`

Legacy fast path. Returns `(simpson_d, shannon_h, dominant_hash, dominant_count)`.

`score_anchored(pos_sequences, anchor_indices, top_n=8) -> dict`

Scores one supplied anchor per POS sequence and returns readable diagnostics.

`score_all_windows(pos_sequences, top_n=8) -> dict`

Scores every token position in every POS sequence. This is the simplest mode for broad corpus QA.

## Performance

On the original Super-Neg SBD scoring job, the Rust/PyO3/Rayon path was roughly **1,500x faster** than the Python loop used for the same POS-window math. The working benchmark note was about **3.5 minutes** for Python loops versus about **6.6 ms** for the Rust path.

That claim is intentionally scoped:

- spaCy still performs tokenization, tagging, and dependency parsing.
- `warp_pos_spacy` accelerates POS-window hashing, accumulation, and diversity scoring.
- End-to-end throughput still depends on the spaCy model, CPU/GPU setup, text length, and batch size.

## Design Notes

This crate stays boring:

- fixed six-token POS windows
- deterministic hash seeds
- no model downloads
- no NLP opinions beyond the features you pass in
- no hidden state

You can pass POS tags, POS plus dependency labels, entity tags, morphology labels, or any other short feature strings. The Rust side only sees windows of strings.

Hash collisions are possible in theory, so use this as a fast diagnostic and gating signal rather than a permanent identifier system.

## Development

```bash
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo test
python -m maturin build --release
```

GitHub Actions runs Rust formatting, clippy, unit tests, a Python wheel build, wheel install, and the pre-tagged example.
