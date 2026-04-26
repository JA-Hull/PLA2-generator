"""
Hull et al. (2025) Markov-style PLA2 design via ESM2 per-sequence contacts.
Steps: (1) predict contacts per sequence, (2) build (frequency, AA-pair) model,
(3) BLOSUM90 wobble, (4) sample from PSFM with contact-frequency weighting.
Per-sequence binary maps; contact strength is the continuous mean frequency
across training maps (e.g. 80% more pressure than 20%), not a single pooled map.
"""

import sys
import numpy as np

from generate_pla2 import (
    parse_fasta, load_esm2_model, predict_contacts_esm2,
    binarize_contacts, build_psfm, find_catalytic_motif,
    get_key_positions, AA_TO_IDX, IDX_TO_AA, BLOSUM90_POSITIVE,
    CATALYTIC_MOTIF_RE,
)


# --- Diversity selection (ESM2 tractability) ---

def select_diverse_subset(sequences, n=50, seed=42):
    """Greedy farthest-first selection on Hamming distance."""
    if not sequences:
        return []
    rng = np.random.default_rng(seed)
    n = min(n, len(sequences))
    seq_arr = np.array([[AA_TO_IDX.get(aa, 20) for aa in s] for s in sequences])

    selected = [rng.integers(len(sequences))]
    remaining = set(range(len(sequences))) - set(selected)

    for _ in range(n - 1):
        candidates = list(remaining)
        if len(candidates) > 500:
            candidates = rng.choice(candidates, 500, replace=False).tolist()
        best_idx, best_dist = None, -1
        for c in candidates:
            d = min(np.sum(seq_arr[c] != seq_arr[s]) for s in selected)
            if d > best_dist:
                best_dist, best_idx = d, c
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected

# --- Per-sequence contact profiling ---

def build_contact_frequency_model(sequences, contact_maps, min_sep=1):
    """Mean binary contact map and, per (i,j) with |i-j|>=min_sep, 20x20 AA^2
    counts from sequences that actually have that contact."""
    n_seqs = len(sequences)
    L = len(sequences[0])
    binary_maps = [binarize_contacts(cm) for cm in contact_maps]
    contact_freq = np.zeros((L, L), dtype=np.float64)
    for bm in binary_maps:
        contact_freq += bm.astype(np.float64)
    contact_freq /= n_seqs

    sep_mask = np.abs(np.subtract.outer(np.arange(L), np.arange(L))) < min_sep
    contact_freq[sep_mask] = 0

    pair_aa_counts = {}
    for i in range(L):
        for j in range(i + min_sep, L):
            if contact_freq[i, j] <= 0:
                continue
            counts = np.zeros((20, 20), dtype=np.float64)
            for seq, bm in zip(sequences, binary_maps):
                if not bm[i, j]:
                    continue
                ai, aj = AA_TO_IDX.get(seq[i]), AA_TO_IDX.get(seq[j])
                if ai is not None and aj is not None:
                    counts[ai, aj] += 1
            pair_aa_counts[(i, j)] = counts

    return contact_freq, pair_aa_counts

# --- Contact-weighted scoring ---

def contact_score(seq, contact_freq, pair_aa_counts):
    """Frequency-weighted fraction of contacts satisfied (observed or BLOSUM wobble)."""
    total_weight = 0.0
    satisfied_weight = 0.0
    for (i, j), counts in pair_aa_counts.items():
        freq = contact_freq[i, j]
        if freq <= 0:
            continue
        total_weight += freq
        ai = AA_TO_IDX.get(seq[i])
        aj = AA_TO_IDX.get(seq[j])
        if ai is None or aj is None:
            continue
        if counts[ai, aj] > 0:
            satisfied_weight += freq
            continue
        for oi, oj in np.argwhere(counts > 0):
            if BLOSUM90_POSITIVE[oi, ai] and BLOSUM90_POSITIVE[oj, aj]:
                satisfied_weight += freq
                break
    return satisfied_weight / total_weight if total_weight > 0 else 1.0


def pseudo_log_likelihood(seq, psfm, contact_freq, pair_aa_counts):
    """Pseudo-log-likelihood: sum of log conditional probabilities at each position.

    For each position, computes the probability the Gibbs model assigns to
    the actual amino acid given all its neighbors' current values (PSFM base
    probability × contact compatibility penalties), then takes the log.
    Summing over all positions gives a joint score from the generative model.
    """
    L = len(seq)
    neighbors = {i: [] for i in range(L)}
    for (i, j), counts in pair_aa_counts.items():
        freq = contact_freq[i, j]
        if freq > 0:
            neighbors[i].append((j, freq, counts))
            neighbors[j].append((i, freq, counts))

    total_ll = 0.0
    for pos in range(L):
        probs = psfm[pos].copy()
        for aa_idx in range(20):
            penalty = 1.0
            for nb, freq, counts in neighbors[pos]:
                nb_idx = AA_TO_IDX.get(seq[nb])
                if nb_idx is None:
                    continue
                if not _is_compatible(counts, aa_idx, nb_idx, pos, nb):
                    penalty *= (1.0 - freq)
            probs[aa_idx] *= penalty
        total = probs.sum()
        if total > 0:
            probs /= total
        aa_idx = AA_TO_IDX.get(seq[pos])
        if aa_idx is not None and probs[aa_idx] > 0:
            total_ll += np.log(probs[aa_idx])
        else:
            total_ll += -20.0  # floor for unsupported residues
    return total_ll

# --- Gibbs sampler ---

def _is_compatible(counts, aa_idx, nb_idx, pos, nb):
    """(aa at pos, aa at nb) vs stored counts for pair (min,max); keys always i<j."""
    ai, aj = (aa_idx, nb_idx) if pos < nb else (nb_idx, aa_idx)
    if counts[ai, aj] > 0:
        return True
    return any(
        BLOSUM90_POSITIVE[oi, ai] and BLOSUM90_POSITIVE[oj, aj]
        for oi, oj in np.argwhere(counts > 0)
    )


def gibbs_sample(psfm, contact_freq, pair_aa_counts,
                 n_sequences=30, n_sweeps=20,
                 min_score=0.90, rng=None):
    """Gibbs on PSFM; per-neighbor step down-weight incompatible pairs by
    (1 - contact_freq) so frequent contacts apply stronger pressure."""
    if rng is None:
        rng = np.random.default_rng(42)
    L = psfm.shape[0]
    neighbors = {i: [] for i in range(L)}
    for (i, j), counts in pair_aa_counts.items():
        freq = contact_freq[i, j]
        if freq > 0:
            neighbors[i].append((j, freq, counts))
            neighbors[j].append((i, freq, counts))

    generated = []
    attempts = 0
    max_attempts = n_sequences * 10
    while len(generated) < n_sequences and attempts < max_attempts:
        attempts += 1
        seq = [IDX_TO_AA[rng.choice(20, p=psfm[pos])] for pos in range(L)]
        positions = list(range(L))
        for _ in range(n_sweeps):
            rng.shuffle(positions)
            for pos in positions:
                if not neighbors[pos]:
                    continue
                probs = psfm[pos].copy()
                for aa_idx in range(20):
                    penalty = 1.0
                    for nb, freq, counts in neighbors[pos]:
                        nb_idx = AA_TO_IDX.get(seq[nb])
                        if nb_idx is None:
                            continue
                        if not _is_compatible(counts, aa_idx, nb_idx, pos, nb):
                            penalty *= (1.0 - freq)
                    probs[aa_idx] *= penalty
                total = probs.sum()
                if total > 0:
                    probs /= total
                else:
                    probs = psfm[pos].copy()
                seq[pos] = IDX_TO_AA[rng.choice(20, p=probs)]
        seq_str = "".join(seq)
        score = contact_score(seq_str, contact_freq, pair_aa_counts)
        if score >= min_score and seq_str not in generated:
            generated.append(seq_str)
    return generated

# --- main ---

def main():
    input_fasta = "data/natural_pla2_domains.fasta"
    output_fasta = "output/generated.fasta"
    n_contact_subset = None  # None = use all sequences for ESM2 contacts
    n_generate = 30
    rng = np.random.default_rng(42)

    # 1. Load
    print(f"Loading {input_fasta} ...")
    raw = parse_fasta(input_fasta)
    all_names = [n for n, _ in raw]
    all_seqs = [s for _, s in raw]
    L = len(all_seqs[0])
    print(f"  {len(all_seqs)} seqs, L={L}")
    ref_idx = next((i for i, n in enumerate(all_names) if "AAV9" in n), None)
    if ref_idx is None:
        print("ERROR: no AAV9 reference")
        sys.exit(1)
    ref_seq = all_seqs[ref_idx]
    print(f"  ref: {all_names[ref_idx]}")
    motif_off = find_catalytic_motif(ref_seq)
    if motif_off is None:
        print("ERROR: DxxxxxHD motif not found")
        sys.exit(1)
    ca_binding, catalytic = get_key_positions(motif_off, ref_seq)
    conserved = ca_binding | catalytic
    print(
        f"  motif {ref_seq[motif_off:motif_off+8]}  "
        f"Ca-binding={sorted(ca_binding)}  catalytic={sorted(catalytic)}"
    )

    # 2. PSFM
    print(f"\nPSFM from {len(all_seqs)} seqs")
    psfm = build_psfm(all_seqs)
    n_var = sum(1 for j in range(L) if np.max(psfm[j]) < 0.95)
    print(f"  variable {n_var}/{L}")
    for pos in sorted(conserved):
        p = np.argmax(psfm[pos])
        label = "cat" if pos in catalytic else "Ca"
        print(
            f"  {label} {pos}: {IDX_TO_AA[p]} ({np.max(psfm[pos]):.0%})"
        )

    # 3. Select sequences for ESM2 contacts
    if n_contact_subset is None or n_contact_subset >= len(all_seqs):
        subset_idx = list(range(len(all_seqs)))
        print(f"\nUsing all {len(subset_idx)} sequences for ESM2 contacts")
    else:
        print(f"\nDiverse subset n={n_contact_subset} for ESM2")
        subset_idx = select_diverse_subset(all_seqs, n=n_contact_subset)
    if ref_idx not in subset_idx:
        subset_idx[0] = ref_idx
    subset_seqs = [all_seqs[i] for i in subset_idx]
    subset_names = [all_names[i] for i in subset_idx]

    # 4. ESM2 contacts
    print("\nESM2 load + per-seq contacts")
    model, alphabet, batch_converter = load_esm2_model("cpu")
    contact_maps = []
    for i, (name, seq) in enumerate(zip(subset_names, subset_seqs)):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {i+1}/{len(subset_seqs)} {name[:30]}")
        contact_maps.append(
            predict_contacts_esm2(seq, model, alphabet, batch_converter, "cpu")
        )
    del model

    # 5. Contact model
    print("\nContact frequency model (per-seq maps)")
    contact_freq, pair_aa_counts = build_contact_frequency_model(
        subset_seqs, contact_maps
    )
    n_cp = len(pair_aa_counts)
    freq_values = np.array([contact_freq[i, j] for i, j in pair_aa_counts])
    c_per_pos = np.zeros(L)
    for i, j in pair_aa_counts:
        c_per_pos[i] += 1
        c_per_pos[j] += 1
    print(
        f"  pairs={n_cp}  freq "
        f"[{freq_values.min():.2f}, {freq_values.max():.2f}] "
        f"mean={freq_values.mean():.2f}"
    )
    print(
        f"  contacts/pos: mean={c_per_pos.mean():.1f} max={c_per_pos.max():.0f} "
        f"pos0={(c_per_pos == 0).sum()}"
    )

    # 6. Gibbs
    min_score = 0.90
    print(f"\nGibbs: n={n_generate}  min contact score {min_score:.0%}")
    novel_seqs = gibbs_sample(
        psfm=psfm, contact_freq=contact_freq, pair_aa_counts=pair_aa_counts,
        n_sequences=n_generate, n_sweeps=20, min_score=min_score, rng=rng,
    )
    print(f"  generated {len(novel_seqs)}")

    # 7. Check
    n_as_bad = sum(
        1 for seq in novel_seqs for pos in conserved if seq[pos] != ref_seq[pos]
    )
    n_motif = sum(1 for seq in novel_seqs if CATALYTIC_MOTIF_RE.search(seq))
    print(
        f"  active site edits: {n_as_bad}  DxxxxxHD ok: {n_motif}/{len(novel_seqs)}"
    )
    if novel_seqs:
        ids = [sum(a == b for a, b in zip(s, ref_seq)) / L * 100 for s in novel_seqs]
        print(
            f"  id vs AAV9: {np.mean(ids):.1f}% "
            f"[{np.min(ids):.1f}-{np.max(ids):.1f}]"
        )
        scs = [contact_score(s, contact_freq, pair_aa_counts) for s in novel_seqs]
        print(
            f"  contact score: {np.mean(scs):.3f}  min {np.min(scs):.3f}"
        )
        plls = [pseudo_log_likelihood(s, psfm, contact_freq, pair_aa_counts) for s in novel_seqs]
        ref_pll = pseudo_log_likelihood(ref_seq, psfm, contact_freq, pair_aa_counts)
        print(
            f"  pseudo-LL: {np.mean(plls):.1f}  "
            f"[{np.min(plls):.1f}, {np.max(plls):.1f}]  "
            f"(AAV9: {ref_pll:.1f})"
        )

    # 8. Write
    with open(output_fasta, "w") as f:
        ref_pll = pseudo_log_likelihood(ref_seq, psfm, contact_freq, pair_aa_counts)
        f.write(f">AAV9_PLA2_reference|pll={ref_pll:.1f}\n{ref_seq}\n")
        for i, seq in enumerate(novel_seqs):
            iden = sum(a == b for a, b in zip(seq, ref_seq)) / L
            sc = contact_score(seq, contact_freq, pair_aa_counts)
            pll = pseudo_log_likelihood(seq, psfm, contact_freq, pair_aa_counts)
            f.write(
                f">generated_PLA2_{i+1:03d}|id={iden:.2f}|cs={sc:.3f}|pll={pll:.1f}\n{seq}\n"
            )
    print(f"\nWrote {len(novel_seqs) + 1} -> {output_fasta}")

    c_pairs = np.array(list(pair_aa_counts.keys()))
    np.save("output/contact_subset_indices.npy", np.array(subset_idx))
    np.save("output/contact_freq_matrix.npy", contact_freq)
    np.savez(
        "output/contact_data.npz", contact_pairs=c_pairs, contacts_per_pos=c_per_pos
    )
    print("  saved contact metadata")


if __name__ == "__main__":
    main()