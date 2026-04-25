"""
PLA2 domain sequence generator — shared library.

Provides BLOSUM90 matrix, FASTA I/O, ESM2 contact prediction,
contact binarization, PSFM construction, and catalytic motif utilities.
Used by: run_pipeline.py, make_figures.py
"""

import re

import numpy as np
import torch

# --- Constants ---

AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
IDX_TO_AA = {i: aa for aa, i in AA_TO_IDX.items()}

# DxxxxxHD: D, 5 variable residues, then H, then D (8 residues total).
# H is at offset +6, second D at offset +7 from the first D.
CATALYTIC_MOTIF_RE = re.compile(r"D.{5}HD")
H_OFFSET = 6
D2_OFFSET = 7

# --- BLOSUM90 (contact wobble only) ---

# fmt: off
BLOSUM90_RAW = [
 [  5, -2, -2, -3, -1, -1, -1,  0, -2, -2, -2, -1, -2, -3, -1,  1,  0, -4, -3, -1],
 [ -2,  6, -1, -3, -5,  1, -1, -3,  0, -4, -3,  2, -2, -4, -3, -1, -2, -4, -3, -3],
 [ -2, -1,  7,  1, -4,  0, -1, -1,  0, -4, -4,  0, -3, -4, -3,  0,  0, -5, -3, -4],
 [ -3, -3,  1,  7, -5, -1,  1, -2, -2, -5, -5, -1, -4, -5, -3, -1, -2, -6, -4, -5],
 [ -1, -5, -4, -5,  9, -4, -6, -4, -5, -2, -2, -4, -2, -3, -4, -2, -2, -4, -4, -2],
 [ -1,  1,  0, -1, -4,  7,  2, -3,  1, -4, -3,  1,  0, -4, -2,  0, -1, -3, -2, -3],
 [ -1, -1, -1,  1, -6,  2,  6, -3, -1, -4, -4,  0, -3, -5, -2,  0, -1, -5, -4, -3],
 [  0, -3, -1, -2, -4, -3, -3,  6, -3, -5, -5, -2, -4, -5, -3, -1, -3, -4, -5, -5],
 [ -2,  0,  0, -2, -5,  1, -1, -3,  8, -4, -4, -1, -3, -2, -3, -2, -2, -3,  1, -4],
 [ -2, -4, -4, -5, -2, -4, -4, -5, -4,  5,  1, -4,  1, -1, -4, -3, -1, -4, -2,  3],
 [ -2, -3, -4, -5, -2, -3, -4, -5, -4,  1,  5, -3,  2,  0, -4, -3, -2, -3, -2,  0],
 [ -1,  2,  0, -1, -4,  1,  0, -2, -1, -4, -3,  6, -2, -4, -2, -1, -1, -5, -3, -3],
 [ -2, -2, -3, -4, -2,  0, -3, -4, -3,  1,  2, -2,  7, -1, -3, -2, -1, -2, -2,  0],
 [ -3, -4, -4, -5, -3, -4, -5, -5, -2, -1,  0, -4, -1,  7, -4, -3, -3,  0,  3, -2],
 [ -1, -3, -3, -3, -4, -2, -2, -3, -3, -4, -4, -2, -3, -4,  8, -2, -2, -5, -4, -3],
 [  1, -1,  0, -1, -2,  0,  0, -1, -2, -3, -3, -1, -2, -3, -2,  5,  1, -4, -3, -2],
 [  0, -2,  0, -2, -2, -1, -1, -3, -2, -1, -2, -1, -1, -3, -2,  1,  6, -4, -2,  0],
 [ -4, -4, -5, -6, -4, -3, -5, -4, -3, -4, -3, -5, -2,  0, -5, -4, -4, 11,  2, -3],
 [ -3, -3, -3, -4, -4, -2, -4, -5,  1, -2, -2, -3, -2,  3, -4, -3, -2,  2,  8, -3],
 [ -1, -3, -4, -5, -2, -3, -3, -5, -4,  3,  0, -3,  0, -2, -3, -2,  0, -3, -3,  5],
]
# fmt: on

BLOSUM90 = np.array(BLOSUM90_RAW, dtype=np.float64)
BLOSUM90_POSITIVE = BLOSUM90 > 0  # wobble tolerance mask

# --- Sequence I/O ---

def parse_fasta(path):
    """Parse a FASTA file into (name, sequence) pairs."""
    out, name, parts = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] == ">":
                if name is not None:
                    out.append((name, "".join(parts).replace(" ", "")))
                name, parts = line[1:], []
            else:
                parts.append(line)
    if name is not None:
        out.append((name, "".join(parts).replace(" ", "")))
    return out

# --- Catalytic motif and active site ---
def find_catalytic_motif(seq):
    """Find the first DxxxxxHD motif (D, five variable residues, H, D) in a sequence.

    Returns the 0-based index of the leading D, or None if not found.
    """
    m = CATALYTIC_MOTIF_RE.search(seq)
    return m.start() if m is not None else None


def get_active_site_positions(motif_offset):
    """Return domain-local indices of the calcium-binding D, His, and catalytic Asp.

    `motif_offset` is the 0-based index of the first D in DxxxxxHD (as from
    `find_catalytic_motif`).
    """
    return frozenset((motif_offset, motif_offset + H_OFFSET, motif_offset + D2_OFFSET))


def extract_pla2_by_motif(seq, domain_len=81, upstream_of_motif=21):
    """Extract PLA2 domain aligned by the DxxxxxHD catalytic motif.

    Returns (domain_sequence, motif_offset_in_domain) or (None, None).
    """
    motif_pos = find_catalytic_motif(seq)
    if motif_pos is None:
        return None, None
    domain_start = motif_pos - upstream_of_motif
    domain_end = domain_start + domain_len
    if domain_start < 0 or domain_end > len(seq):
        return None, None
    return seq[domain_start:domain_end], motif_pos - domain_start

# --- ESM2 contact prediction ---
def load_esm2_model(device="cpu"):
    """Load ESM2-650M. Returns (model, alphabet, batch_converter)."""
    import esm as fair_esm
    model, alphabet = fair_esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    return model.to(device).eval(), alphabet, batch_converter

def predict_contacts_esm2(sequence, model, _alphabet, batch_converter, device="cpu"):
    """Predict residue-residue contacts from ESM2 attention maps.

    Extracts attention weights across all 33 layers and 20 heads,
    applies APC (average product correction), symmetrizes, and
    averages into an L x L contact score matrix.
    """
    _, _, batch_tokens = batch_converter([("query", sequence)])
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], need_head_weights=True)
    attn = results["attentions"][0]
    attn = (attn[:, :, 1:-1, 1:-1] + attn[:, :, 1:-1, 1:-1].transpose(-1, -2)) / 2.0
    row_m, col_m, g = attn.mean(-1, True), attn.mean(-2, True), attn.mean((-1, -2), True)
    apc = attn - (row_m * col_m) / (g + 1e-8)
    contact_map = apc.mean((0, 1)).cpu().numpy()
    np.fill_diagonal(contact_map, 0)
    return contact_map


def binarize_contacts(contact_map, top_fraction=0.15, min_seq_sep=6):
    """Convert continuous contact scores to a binary contact map.

    Per-position binarization: for each position i, take the top
    `top_fraction` of its long-range contact scores (|i-j| >= min_seq_sep).
    This ensures every position gets roughly the same number of contacts
    (~top_fraction * eligible_partners ≈ 11 for L=81, sep=6, frac=0.15).

    The result is symmetrized: if i selects j, j also contacts i.
    """
    L = contact_map.shape[0]
    binary = np.zeros((L, L), dtype=np.int32)
    j = np.arange(L, dtype=np.int32)
    for i in range(L):
        elig = np.abs(j - i) >= min_seq_sep
        n_elig = int(elig.sum())
        if n_elig == 0:
            continue
        row = contact_map[i]
        scores = row[elig]
        n_k = max(1, int(top_fraction * n_elig))
        thr = np.partition(scores, n_elig - n_k)[n_elig - n_k]
        binary[i, elig & (row >= thr)] = 1
    binary = np.maximum(binary, binary.T)
    np.fill_diagonal(binary, 0)
    return binary

# --- Position-specific frequency matrix ---

def build_psfm(sequences, pseudocount=0.01):
    """Build a position-specific frequency matrix from aligned sequences.

    Returns: (L, 20) array of amino acid probabilities at each position.
    Nonstandard characters (X, U, etc.) are ignored in counts.
    """
    L = len(sequences[0])
    assert all(len(s) == L for s in sequences), "All sequences must have equal length"
    counts = np.full((L, 20), pseudocount, dtype=np.float64)
    for seq in sequences:
        for p, aa in enumerate(seq):
            k = AA_TO_IDX.get(aa)
            if k is not None:
                counts[p, k] += 1
    return counts / counts.sum(axis=1, keepdims=True)
