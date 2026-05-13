from __future__ import annotations

from collections.abc import Iterable

import spacy
from warp_pos_spacy import accumulate_and_score

NEG_TRIGGERS = {
    "not",
    "n't",
    "never",
    "no",
    "none",
    "nobody",
    "nothing",
    "nowhere",
    "neither",
    "nor",
    "without",
    "cannot",
}


def find_negation_anchor(doc) -> int | None:
    for i, tok in enumerate(doc):
        if tok.dep_ == "neg":
            return i

    for i, tok in enumerate(doc):
        lower = tok.lower_
        if lower in NEG_TRIGGERS or lower.endswith("n't"):
            return i

    return None


def prepare_spacy_inputs(sentences: Iterable[str], nlp) -> tuple[list[list[str]], list[int], int]:
    pos_sequences: list[list[str]] = []
    anchor_indices: list[int] = []
    skipped = 0

    for doc in nlp.pipe(sentences, batch_size=2000, n_process=-1):
        anchor_idx = find_negation_anchor(doc)
        if anchor_idx is None:
            skipped += 1
            continue

        pos_sequences.append([tok.pos_ for tok in doc])
        anchor_indices.append(anchor_idx)

    return pos_sequences, anchor_indices, skipped


def main() -> None:
    sentences = [
        "I do not agree with this fragile pattern.",
        "The service cannot tolerate silent failures.",
        "We never allow monoculture templates in training data.",
        "No researcher should skip adversarial negation checks.",
        "They weren't expecting this semantic inversion.",
    ]

    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_sm")

    pos_sequences, anchor_indices, skipped = prepare_spacy_inputs(sentences, nlp)
    simpson_d, shannon_h, dominant_hash, dominant_count = accumulate_and_score(
        pos_sequences, anchor_indices
    )

    print(f"Scored sentences:  {len(pos_sequences)}")
    print(f"Skipped:           {skipped}")
    print(f"Simpson dominance: {simpson_d:.6f}")
    print(f"Shannon entropy:   {shannon_h:.6f}")
    print(f"Dominant hash:     {dominant_hash}")
    print(f"Dominant count:    {dominant_count}")


if __name__ == "__main__":
    main()
