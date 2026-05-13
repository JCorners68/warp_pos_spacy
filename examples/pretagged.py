from warp_pos_spacy import accumulate_and_score, hash_window


def main() -> None:
    pos_sequences = [
        ["PRON", "AUX", "PART", "VERB", "DET", "NOUN"],
        ["PRON", "AUX", "PART", "VERB", "DET", "NOUN"],
        ["NOUN", "AUX", "PART", "VERB", "ADJ", "NOUN"],
        ["ADV", "AUX", "PART", "VERB", "ADP", "NOUN"],
        ["PRON", "AUX", "PART", "VERB", "DET", "NOUN"],
    ]
    anchor_indices = [2, 2, 2, 2, 2]

    print("Example species:", hash_window(pos_sequences[0], anchor_indices[0]))

    simpson_d, shannon_h, dominant_hash, dominant_count = accumulate_and_score(
        pos_sequences, anchor_indices
    )

    print(f"Simpson dominance: {simpson_d:.6f}")
    print(f"Shannon entropy:   {shannon_h:.6f}")
    print(f"Dominant hash:     {dominant_hash}")
    print(f"Dominant count:    {dominant_count}")


if __name__ == "__main__":
    main()
