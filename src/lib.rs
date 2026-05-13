// PyO3's proc-macro wrappers currently trip clippy::useless_conversion on
// PyResult return types. Keep the rest of the warning set strict.
#![allow(clippy::useless_conversion)]

use ahash::{AHashMap, RandomState};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use rayon::prelude::*;
use std::hash::{BuildHasher, Hasher};
use std::sync::LazyLock;

const PAD_TOKEN: &[u8] = b"<PAD>";
const SEP_TOKEN: u8 = 0xFF;
const WINDOW_START: i32 = -2;
const WINDOW_END: i32 = 3;

const HASH_SEED_0: u64 = 0x9E37_79B9_7F4A_7C15;
const HASH_SEED_1: u64 = 0xD1B5_4A32_D192_ED03;
const HASH_SEED_2: u64 = 0x94D0_49BB_1331_11EB;
const HASH_SEED_3: u64 = 0x2545_F491_4F6C_DD1D;

static HASH_BUILDER: LazyLock<RandomState> =
    LazyLock::new(|| RandomState::with_seeds(HASH_SEED_0, HASH_SEED_1, HASH_SEED_2, HASH_SEED_3));

#[inline]
fn validate_anchor<T>(pos_sequence: &[T], anchor_idx: usize) -> Result<(), String> {
    if pos_sequence.is_empty() {
        return Err("pos_sequence must not be empty".to_string());
    }

    if anchor_idx >= pos_sequence.len() {
        return Err(format!(
            "anchor_idx={} out of range for pos_sequence length={}",
            anchor_idx,
            pos_sequence.len()
        ));
    }

    Ok(())
}

#[inline]
fn hash_window_internal<T: AsRef<str>>(pos_sequence: &[T], anchor_idx: usize) -> u64 {
    let mut hasher = HASH_BUILDER.build_hasher();

    for rel in WINDOW_START..=WINDOW_END {
        let maybe_idx = if rel < 0 {
            anchor_idx.checked_sub(rel.unsigned_abs() as usize)
        } else {
            anchor_idx.checked_add(rel as usize)
        };

        match maybe_idx {
            Some(idx) if idx < pos_sequence.len() => {
                hasher.write(pos_sequence[idx].as_ref().as_bytes())
            }
            _ => hasher.write(PAD_TOKEN),
        }
        hasher.write_u8(SEP_TOKEN);
    }

    hasher.finish()
}

fn counts_from_pretagged<T: AsRef<str> + Sync>(
    pos_sequences: &[Vec<T>],
    anchor_indices: &[usize],
) -> AHashMap<u64, u64> {
    pos_sequences
        .par_iter()
        .zip(anchor_indices.par_iter())
        .fold(
            || AHashMap::<u64, u64>::with_capacity(4096),
            |mut local_counts, (pos_seq, &anchor_idx)| {
                let species = hash_window_internal(pos_seq, anchor_idx);
                *local_counts.entry(species).or_insert(0) += 1;
                local_counts
            },
        )
        .reduce(AHashMap::<u64, u64>::new, |mut acc, local| {
            for (species, count) in local {
                *acc.entry(species).or_insert(0) += count;
            }
            acc
        })
}

fn validate_inputs<T>(pos_sequences: &[Vec<T>], anchor_indices: &[usize]) -> Result<(), String> {
    if pos_sequences.len() != anchor_indices.len() {
        return Err(format!(
            "length mismatch: pos_sequences={} anchor_indices={}",
            pos_sequences.len(),
            anchor_indices.len()
        ));
    }

    for (row, (pos_sequence, &anchor_idx)) in
        pos_sequences.iter().zip(anchor_indices.iter()).enumerate()
    {
        validate_anchor(pos_sequence, anchor_idx)
            .map_err(|message| format!("row {row}: {message}"))?;
    }

    Ok(())
}

fn score_counts(counts: &AHashMap<u64, u64>, total_count: u64) -> (f64, f64, u64, u64) {
    if total_count == 0 {
        return (0.0, 0.0, 0, 0);
    }

    let mut simpsons_dominance = 0.0_f64;
    let mut shannon_entropy = 0.0_f64;
    let mut dominant_species_hash = 0_u64;
    let mut dominant_species_count = 0_u64;
    let total = total_count as f64;

    for (&species, &count) in counts {
        if count > dominant_species_count {
            dominant_species_count = count;
            dominant_species_hash = species;
        }

        let p = count as f64 / total;
        simpsons_dominance += p * p;
        shannon_entropy -= p * p.ln();
    }

    (
        simpsons_dominance,
        shannon_entropy,
        dominant_species_hash,
        dominant_species_count,
    )
}

#[pyfunction]
#[allow(clippy::useless_conversion)]
fn hash_window(pos_sequence: Vec<PyBackedStr>, anchor_idx: usize) -> PyResult<u64> {
    validate_anchor(&pos_sequence, anchor_idx).map_err(PyValueError::new_err)?;
    Ok(hash_window_internal(&pos_sequence, anchor_idx))
}

#[allow(clippy::useless_conversion)]
#[pyfunction]
fn accumulate_and_score(
    py: Python<'_>,
    pos_sequences: Vec<Vec<PyBackedStr>>,
    anchor_indices: Vec<usize>,
) -> PyResult<(f64, f64, u64, u64)> {
    validate_inputs(&pos_sequences, &anchor_indices).map_err(PyValueError::new_err)?;
    let counts = py.allow_threads(|| counts_from_pretagged(&pos_sequences, &anchor_indices));
    Ok(score_counts(&counts, anchor_indices.len() as u64))
}

#[pymodule]
fn warp_pos_spacy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hash_window, m)?)?;
    m.add_function(wrap_pyfunction!(accumulate_and_score, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_window_is_deterministic() {
        let pos = vec!["PRON", "AUX", "PART", "VERB", "DET", "NOUN"];
        assert_eq!(hash_window_internal(&pos, 2), hash_window_internal(&pos, 2));
    }

    #[test]
    fn hash_window_uses_padding_at_boundaries() {
        let pos = vec!["PART", "VERB"];
        let at_start = hash_window_internal(&pos, 0);
        let at_end = hash_window_internal(&pos, 1);
        assert_ne!(at_start, at_end);
    }

    #[test]
    fn validate_anchor_rejects_empty_or_out_of_range() {
        let empty: Vec<&str> = vec![];
        assert!(validate_anchor(&empty, 0).is_err());

        let pos = vec!["NOUN"];
        assert!(validate_anchor(&pos, 1).is_err());
        assert!(validate_anchor(&pos, 0).is_ok());
    }

    #[test]
    fn score_counts_returns_expected_metrics() {
        let mut counts = AHashMap::new();
        counts.insert(10, 3);
        counts.insert(20, 1);

        let (simpson, shannon, _dominant_hash, dominant_count) = score_counts(&counts, 4);
        assert!((simpson - 0.625).abs() < 1e-12);
        assert!((shannon - 0.5623351446188083).abs() < 1e-12);
        assert_eq!(dominant_count, 3);
    }

    #[test]
    fn counts_merge_parallel_results() {
        let pos_sequences = vec![
            vec!["PRON", "AUX", "PART", "VERB", "DET", "NOUN"],
            vec!["PRON", "AUX", "PART", "VERB", "DET", "NOUN"],
            vec!["NOUN", "AUX", "PART", "VERB", "ADJ", "NOUN"],
        ];
        let anchor_indices = vec![2, 2, 2];

        let counts = counts_from_pretagged(&pos_sequences, &anchor_indices);
        let total: u64 = counts.values().sum();
        assert_eq!(total, 3);
        assert_eq!(counts.len(), 2);
    }
}
