// PyO3's proc-macro wrappers currently trip clippy::useless_conversion on
// PyResult return types. Keep the rest of the warning set strict.
#![allow(clippy::useless_conversion)]

use ahash::{AHashMap, RandomState};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use std::hash::{BuildHasher, Hasher};
use std::sync::LazyLock;

const PAD_TOKEN: &[u8] = b"<PAD>";
const SEP_TOKEN: u8 = 0xFF;
const WINDOW_START: i32 = -2;
const WINDOW_END: i32 = 3;
const WINDOW_LEN: usize = (WINDOW_END - WINDOW_START + 1) as usize;

const HASH_SEED_0: u64 = 0x9E37_79B9_7F4A_7C15;
const HASH_SEED_1: u64 = 0xD1B5_4A32_D192_ED03;
const HASH_SEED_2: u64 = 0x94D0_49BB_1331_11EB;
const HASH_SEED_3: u64 = 0x2545_F491_4F6C_DD1D;

static HASH_BUILDER: LazyLock<RandomState> =
    LazyLock::new(|| RandomState::with_seeds(HASH_SEED_0, HASH_SEED_1, HASH_SEED_2, HASH_SEED_3));

#[derive(Clone, Debug)]
struct PatternCount {
    pattern: String,
    count: u64,
}

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
fn window_index(anchor_idx: usize, rel: i32) -> Option<usize> {
    if rel < 0 {
        anchor_idx.checked_sub(rel.unsigned_abs() as usize)
    } else {
        anchor_idx.checked_add(rel as usize)
    }
}

#[inline]
fn hash_window_internal<T: AsRef<str>>(pos_sequence: &[T], anchor_idx: usize) -> u64 {
    let mut hasher = HASH_BUILDER.build_hasher();

    for rel in WINDOW_START..=WINDOW_END {
        match window_index(anchor_idx, rel) {
            Some(idx) if idx < pos_sequence.len() => {
                hasher.write(pos_sequence[idx].as_ref().as_bytes())
            }
            _ => hasher.write(PAD_TOKEN),
        }
        hasher.write_u8(SEP_TOKEN);
    }

    hasher.finish()
}

fn window_pattern_internal<T: AsRef<str>>(pos_sequence: &[T], anchor_idx: usize) -> String {
    let mut parts = Vec::with_capacity(WINDOW_LEN);

    for rel in WINDOW_START..=WINDOW_END {
        match window_index(anchor_idx, rel) {
            Some(idx) if idx < pos_sequence.len() => parts.push(pos_sequence[idx].as_ref()),
            _ => parts.push("<PAD>"),
        }
    }

    parts.join(" ")
}

fn increment_pattern_count<T: AsRef<str>>(
    counts: &mut AHashMap<u64, PatternCount>,
    pos_sequence: &[T],
    anchor_idx: usize,
) {
    let species = hash_window_internal(pos_sequence, anchor_idx);

    counts
        .entry(species)
        .and_modify(|entry| entry.count += 1)
        .or_insert_with(|| PatternCount {
            pattern: window_pattern_internal(pos_sequence, anchor_idx),
            count: 1,
        });
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

fn pattern_counts_from_anchored<T: AsRef<str> + Sync>(
    pos_sequences: &[Vec<T>],
    anchor_indices: &[usize],
) -> AHashMap<u64, PatternCount> {
    pos_sequences
        .par_iter()
        .zip(anchor_indices.par_iter())
        .fold(
            AHashMap::<u64, PatternCount>::new,
            |mut local_counts, (pos_seq, &anchor_idx)| {
                increment_pattern_count(&mut local_counts, pos_seq, anchor_idx);
                local_counts
            },
        )
        .reduce(AHashMap::<u64, PatternCount>::new, merge_pattern_counts)
}

fn pattern_counts_from_all_windows<T: AsRef<str> + Sync>(
    pos_sequences: &[Vec<T>],
) -> AHashMap<u64, PatternCount> {
    pos_sequences
        .par_iter()
        .fold(
            AHashMap::<u64, PatternCount>::new,
            |mut local_counts, pos_seq| {
                for anchor_idx in 0..pos_seq.len() {
                    increment_pattern_count(&mut local_counts, pos_seq, anchor_idx);
                }
                local_counts
            },
        )
        .reduce(AHashMap::<u64, PatternCount>::new, merge_pattern_counts)
}

fn merge_pattern_counts(
    mut acc: AHashMap<u64, PatternCount>,
    local: AHashMap<u64, PatternCount>,
) -> AHashMap<u64, PatternCount> {
    for (species, pattern_count) in local {
        acc.entry(species)
            .and_modify(|entry| entry.count += pattern_count.count)
            .or_insert(pattern_count);
    }
    acc
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

fn total_windows_from_all<T>(pos_sequences: &[Vec<T>]) -> u64 {
    pos_sequences
        .iter()
        .map(|pos_sequence| pos_sequence.len() as u64)
        .sum()
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

fn score_pattern_counts(
    counts: &AHashMap<u64, PatternCount>,
    total_count: u64,
) -> (f64, f64, u64, u64) {
    if total_count == 0 {
        return (0.0, 0.0, 0, 0);
    }

    let mut simpsons_dominance = 0.0_f64;
    let mut shannon_entropy = 0.0_f64;
    let mut dominant_species_hash = 0_u64;
    let mut dominant_species_count = 0_u64;
    let total = total_count as f64;

    for (&species, pattern_count) in counts {
        if pattern_count.count > dominant_species_count {
            dominant_species_count = pattern_count.count;
            dominant_species_hash = species;
        }

        let p = pattern_count.count as f64 / total;
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

fn sorted_top_patterns(
    counts: &AHashMap<u64, PatternCount>,
    top_n: usize,
) -> Vec<(u64, PatternCount)> {
    let mut patterns: Vec<(u64, PatternCount)> = counts
        .iter()
        .map(|(&species, pattern_count)| (species, pattern_count.clone()))
        .collect();

    patterns.sort_by(|a, b| {
        b.1.count
            .cmp(&a.1.count)
            .then_with(|| a.1.pattern.cmp(&b.1.pattern))
            .then_with(|| a.0.cmp(&b.0))
    });
    patterns.truncate(top_n);
    patterns
}

fn build_summary(
    py: Python<'_>,
    counts: AHashMap<u64, PatternCount>,
    total_count: u64,
    top_n: usize,
) -> PyResult<PyObject> {
    let (simpson_d, shannon_h, _dominant_hash, _dominant_count) =
        score_pattern_counts(&counts, total_count);
    let leading_patterns = sorted_top_patterns(&counts, top_n.max(1));
    let (dominant_hash, dominant_count) = leading_patterns
        .first()
        .map(|(species, pattern_count)| (*species, pattern_count.count))
        .unwrap_or((0, 0));
    let dominant_share = if total_count == 0 {
        0.0
    } else {
        dominant_count as f64 / total_count as f64
    };

    let top_patterns = PyList::empty_bound(py);
    for (species, pattern_count) in leading_patterns.into_iter().take(top_n) {
        let item = PyDict::new_bound(py);
        item.set_item("hash", species)?;
        item.set_item("pattern", pattern_count.pattern)?;
        item.set_item("count", pattern_count.count)?;
        item.set_item(
            "share",
            if total_count == 0 {
                0.0
            } else {
                pattern_count.count as f64 / total_count as f64
            },
        )?;
        top_patterns.append(item)?;
    }

    let summary = PyDict::new_bound(py);
    summary.set_item("total_windows", total_count)?;
    summary.set_item("unique_patterns", counts.len())?;
    summary.set_item("simpson_d", simpson_d)?;
    summary.set_item("shannon_h", shannon_h)?;
    summary.set_item("dominant_hash", dominant_hash)?;
    summary.set_item("dominant_count", dominant_count)?;
    summary.set_item("dominant_share", dominant_share)?;
    summary.set_item("collapse_pressure", simpson_d)?;
    summary.set_item("top_patterns", top_patterns)?;
    Ok(summary.into_py(py))
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

#[allow(clippy::useless_conversion)]
#[pyfunction(signature = (pos_sequences, anchor_indices, top_n = 8))]
fn score_anchored(
    py: Python<'_>,
    pos_sequences: Vec<Vec<PyBackedStr>>,
    anchor_indices: Vec<usize>,
    top_n: usize,
) -> PyResult<PyObject> {
    validate_inputs(&pos_sequences, &anchor_indices).map_err(PyValueError::new_err)?;
    let counts = py.allow_threads(|| pattern_counts_from_anchored(&pos_sequences, &anchor_indices));
    build_summary(py, counts, anchor_indices.len() as u64, top_n)
}

#[allow(clippy::useless_conversion)]
#[pyfunction(signature = (pos_sequences, top_n = 8))]
fn score_all_windows(
    py: Python<'_>,
    pos_sequences: Vec<Vec<PyBackedStr>>,
    top_n: usize,
) -> PyResult<PyObject> {
    let total_windows = total_windows_from_all(&pos_sequences);
    let counts = py.allow_threads(|| pattern_counts_from_all_windows(&pos_sequences));
    build_summary(py, counts, total_windows, top_n)
}

#[pymodule]
fn warp_pos_spacy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hash_window, m)?)?;
    m.add_function(wrap_pyfunction!(accumulate_and_score, m)?)?;
    m.add_function(wrap_pyfunction!(score_anchored, m)?)?;
    m.add_function(wrap_pyfunction!(score_all_windows, m)?)?;
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
    fn window_pattern_is_readable_and_padded() {
        let pos = vec!["PART", "VERB"];
        assert_eq!(
            window_pattern_internal(&pos, 0),
            "<PAD> <PAD> PART VERB <PAD> <PAD>"
        );
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

    #[test]
    fn anchored_pattern_counts_keep_readable_top_pattern() {
        let pos_sequences = vec![
            vec!["PRON", "AUX", "PART", "VERB", "DET", "NOUN"],
            vec!["PRON", "AUX", "PART", "VERB", "DET", "NOUN"],
            vec!["NOUN", "AUX", "PART", "VERB", "ADJ", "NOUN"],
        ];
        let anchor_indices = vec![2, 2, 2];

        let counts = pattern_counts_from_anchored(&pos_sequences, &anchor_indices);
        let (simpson, _shannon, _dominant_hash, dominant_count) =
            score_pattern_counts(&counts, anchor_indices.len() as u64);
        let top = sorted_top_patterns(&counts, 1);

        assert_eq!(counts.len(), 2);
        assert_eq!(dominant_count, 2);
        assert!((simpson - 5.0 / 9.0).abs() < 1e-12);
        assert_eq!(top[0].1.pattern, "PRON AUX PART VERB DET NOUN");
        assert_eq!(top[0].1.count, 2);
    }

    #[test]
    fn all_windows_scores_every_token_position() {
        let pos_sequences = vec![vec!["DET", "ADJ", "NOUN"], vec!["DET", "ADJ", "NOUN"]];

        let counts = pattern_counts_from_all_windows(&pos_sequences);
        let total = total_windows_from_all(&pos_sequences);

        assert_eq!(total, 6);
        assert_eq!(counts.values().map(|pattern| pattern.count).sum::<u64>(), 6);
        assert_eq!(counts.len(), 3);
    }
}
