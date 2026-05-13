# warp_pos_spacy

`warp_pos_spacy` is what happened when a speed nerd met spaCy and asked: what if we kept spaCy's linguistic judgment, but moved the repetitive scoring math into Rust?

It does not replace spaCy as a linguistic parser. spaCy still does the high-fidelity POS tagging and anchor detection. This crate takes the POS arrays and runs the hot scoring loop in Rust with Rayon.

The original Super-Neg workload used this for Syntactic Biome Diversity (SBD): detect whether an LLM generation loop has collapsed into the same grammatical template over and over.

## Why

When generating hard negation data, lexical variety is not enough. A model can produce thousands of examples that look different at the word level while quietly reusing the same syntactic move:

```text
PRON AUX PART VERB DET NOUN
PRON AUX PART VERB DET NOUN
PRON AUX PART VERB DET NOUN
```

That is template monoculture. It trains models to memorize shortcuts instead of learning the harder semantic boundary.

SBD treats the POS window around a negation anchor as a "species" and tracks the species distribution with ecological diversity metrics:

- Simpson's dominance, where high values mean one template is taking over.
- Shannon entropy, where higher values mean a broader syntactic spread.

## Performance

On the Super-Neg SBD scoring job, the Rust/PyO3/Rayon path was roughly **1,500x faster** than the Python loop used for the same POS-window math. The working benchmark note was about **3.5 minutes** for Python loops versus about **6.6 ms** for the Rust path.

That claim is intentionally scoped:

- spaCy still performs tokenization, tagging, and dependency parsing.
- `warp_pos_spacy` accelerates the POS-window hashing, accumulation, and diversity scoring.
- End-to-end throughput still depends on the spaCy model, CPU/GPU setup, text length, and batch size.

## API

```python
from warp_pos_spacy import accumulate_and_score, hash_window

species = hash_window(["PRON", "AUX", "PART", "VERB", "DET", "NOUN"], 2)

simpson_d, shannon_h, dominant_hash, dominant_count = accumulate_and_score(
    [
        ["PRON", "AUX", "PART", "VERB", "DET", "NOUN"],
        ["PRON", "AUX", "PART", "VERB", "DET", "NOUN"],
        ["NOUN", "AUX", "PART", "VERB", "ADJ", "NOUN"],
    ],
    [2, 2, 2],
)
```

`hash_window(pos_sequence, anchor_idx) -> int`

Returns a deterministic species hash for the POS window `[-2, +3]` around `anchor_idx`. Out-of-range window positions are padded. Empty POS arrays or out-of-range anchors raise `ValueError`.

`accumulate_and_score(pos_sequences, anchor_indices) -> tuple[float, float, int, int]`

Returns:

- `simpson_d`: Simpson's dominance index, `sum(p_i^2)`.
- `shannon_h`: Shannon entropy, `-sum(p_i * ln(p_i))`.
- `dominant_hash`: species hash for the most common POS window.
- `dominant_count`: count for the dominant species.

## Build

Install Rust and Python build tooling, then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --release
```

For the spaCy example:

```bash
pip install spacy
python -m spacy download en_core_web_sm
python examples/spacy_pipeline.py
```

To build and install a wheel without relying on an activated environment:

```bash
python -m maturin build --release
pip install target/wheels/*.whl
python examples/pretagged.py
```

If you have a spaCy pipeline that can use GPU, enable it before loading the model:

```python
import spacy

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")
```

## Development

```bash
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

## Design Notes

The crate is deliberately narrow. It expects pre-tagged POS sequences and explicit anchors because that keeps the Rust side deterministic, small, and easy to audit. This also avoids pretending Rust is doing the NLP work. spaCy is the parser. Rust is the fast math lane.

The hash seed is fixed so a given POS window maps to the same species hash across runs. That makes SBD feedback stable enough to feed back into prompt penalties and regression tests.
