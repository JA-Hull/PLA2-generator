"""
Generate publication-quality figures for PLA2 domain analysis:
  1. Contact frequency model overview (frequency heatmap + per-position bar)
  2. Individual contact map PNGs with frequency overlay
  3. Three-panel heatmap: Jaccard contact similarity, % identity, % similarity
"""

import re
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from generate_pla2 import (
    parse_fasta, load_esm2_model, predict_contacts_esm2,
    binarize_contacts, find_catalytic_motif, get_key_positions,
    infer_catalytic_asp, get_all_active_site_positions,
    BLOSUM90, AA_TO_IDX,
)

OUTDIR = "output/figures"

# Global style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

# --- Metrics ---

def pairwise_identity(seq_a, seq_b):
    assert len(seq_a) == len(seq_b)
    return sum(a == b for a, b in zip(seq_a, seq_b)) / len(seq_a) * 100


def pairwise_similarity(seq_a, seq_b):
    assert len(seq_a) == len(seq_b)
    n = sum(1 for a, b in zip(seq_a, seq_b)
            if (i := AA_TO_IDX.get(a)) is not None
            and (j := AA_TO_IDX.get(b)) is not None
            and BLOSUM90[i, j] >= 0)
    return n / len(seq_a) * 100


def jaccard_binary(a, b, min_sep=1):
    L = a.shape[0]
    mask = np.triu(np.ones((L, L), dtype=bool), k=min_sep)
    af, bf = a[mask].astype(bool), b[mask].astype(bool)
    u, n = np.sum(af | bf), np.sum(af & bf)
    return n / u if u > 0 else 0.0


# --- Contact frequency model ---

def load_contact_data():
    return np.load("output/contact_freq_matrix.npy"), np.load("output/contact_data.npz")["contacts_per_pos"]


def _upper_tri_nan(mat, L, k=1):
    m = np.triu(np.ones((L, L), dtype=bool), k=k)
    d = np.full((L, L), np.nan)
    d[m] = mat[m]
    return m, d


def _cbar_labeled(fig, im, ax, ylabel, tick, shrink=0.72):
    cb = fig.colorbar(im, ax=ax, shrink=shrink, pad=0.02, aspect=25)
    cb.set_label(ylabel, fontsize=9)
    cb.ax.tick_params(labelsize=tick)
    return cb


def _active_site_lines(ax, sites):
    """Draw annotation lines for all active site positions."""
    styles = {
        'ca_D1':  dict(color="#2980b9", linewidth=0.6, alpha=0.5, linestyle=":"),
        'ca_D2':  dict(color="#2980b9", linewidth=0.6, alpha=0.5, linestyle=":"),
        'cat_H':  dict(color="#c0392b", linewidth=0.8, alpha=0.6, linestyle="--"),
        'cat_D_inferred': dict(color="#c0392b", linewidth=0.8, alpha=0.6, linestyle="--"),
        'yxgxg':  dict(color="#27ae60", linewidth=0.6, alpha=0.5, linestyle="-."),
        'phos_K': dict(color="#8e44ad", linewidth=0.6, alpha=0.5, linestyle="-."),
    }
    for key, pos in sites.items():
        if key in styles:
            ax.axvline(pos, **styles[key])
            ax.axhline(pos, **styles[key])


def plot_frequency_map(contact_freq, contacts_per_pos, ref_sites, filepath):
    L = contact_freq.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6),
                             gridspec_kw={"width_ratios": [1, 0.5], "wspace": 0.08})
    ax = axes[0]
    mask_upper, display = _upper_tri_nan(contact_freq, L, k=1)
    freq_cmap = LinearSegmentedColormap.from_list(
        "freq", ["#f7f7f7", "#fdd49e", "#ef6548", "#990000"])
    im = ax.imshow(display, cmap=freq_cmap, aspect="equal",
                   interpolation="nearest", vmin=0, vmax=1)
    _cbar_labeled(fig, im, ax,
                  "Contact frequency\n(fraction of input sequences)", 8)
    _active_site_lines(ax, ref_sites)
    n_pairs = int(np.sum(contact_freq[mask_upper] > 0))
    ax.set_xlabel("Residue position (PLA2 domain)", fontsize=10)
    ax.set_ylabel("Residue position (PLA2 domain)", fontsize=10)
    ax.set_title(
        f"Per-Sequence Contact Frequency Model\n"
        f"{n_pairs} contact pairs  |  15% per-position binarization",
        fontsize=11, fontweight="bold", pad=8)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.legend(handles=[
        Patch(facecolor="#990000", label="High frequency"),
        Patch(facecolor="#fdd49e", label="Low frequency"),
        Patch(facecolor="none", edgecolor="#c0392b", linestyle="--", label="Catalytic dyad (H + D)"),
        Patch(facecolor="none", edgecolor="#2980b9", linestyle=":", label="Ca²⁺-binding (DxxxxxHD Ds)"),
        Patch(facecolor="none", edgecolor="#27ae60", linestyle="-.", label="YxGxG Ca²⁺ loop"),
        Patch(facecolor="none", edgecolor="#8e44ad", linestyle="-.", label="Phospholipid-binding Lys"),
    ], loc="lower right", fontsize=7, framealpha=0.95, edgecolor="#cccccc")
    ax2, positions = axes[1], np.arange(L)
    all_sites = set(ref_sites.values())
    cat_pos = {ref_sites.get('cat_H'), ref_sites.get('cat_D_inferred'), ref_sites.get('cat_D')} - {None}
    ca_pos = {ref_sites.get('ca_D1'), ref_sites.get('ca_D2')} - {None}
    bar_colors = ["#c0392b" if p in cat_pos else "#2980b9" if p in ca_pos
                  else "#27ae60" if p == ref_sites.get('yxgxg')
                  else "#8e44ad" if p == ref_sites.get('phos_K')
                  else "#7f8c8d" for p in positions]
    ax2.barh(positions, contacts_per_pos, color=bar_colors, edgecolor="none",
             height=0.75, alpha=0.85)
    ax2.set_ylim(-0.5, L - 0.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Contact pairs", fontsize=10)
    ax2.set_ylabel("Position", fontsize=10)
    ax2.set_title("Per-Position\nCoverage", fontsize=11, fontweight="bold", pad=8)
    ax2.tick_params(labelsize=8)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(10))
    med = np.median(contacts_per_pos)
    ax2.axvline(x=med, color="#7f8c8d", linewidth=0.8, linestyle=":", alpha=0.8)
    ax2.text(med + 0.5, L - 2, f"median={med:.0f}", fontsize=7, color="#7f8c8d", va="bottom")
    plt.savefig(filepath, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved {filepath}")


# --- Individual contact maps ---

def plot_single_contact(cmap_raw, cmap_bin, contact_freq, name, seq, ref_seq, sites, filepath):
    L = cmap_raw.shape[0]
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.06], width_ratios=[1.1, 1],
                          hspace=0.25, wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    ax_legend = fig.add_subplot(gs[1, :])
    ax2 = fig.add_subplot(gs[0, 1])

    mask_upper, display = _upper_tri_nan(cmap_raw, L, k=1)
    vmax = np.percentile(cmap_raw[mask_upper], 98)
    esm_cmap = LinearSegmentedColormap.from_list(
        "esm", ["#ffffff", "#fff7bc", "#fec44f", "#d95f0e", "#993404"])
    im = ax.imshow(display, cmap=esm_cmap, aspect="equal",
                   interpolation="nearest", vmin=0, vmax=vmax)
    _cbar_labeled(fig, im, ax, "ESM2 contact probability", 7)
    mask_lower = np.tril(np.ones((L, L), dtype=bool), k=-1)
    bdisp = np.full((L, L), np.nan)
    bdisp[mask_lower & (cmap_bin > 0)] = 1
    ax.imshow(bdisp, cmap=ListedColormap(["#2980b9"]), aspect="equal",
              interpolation="nearest", alpha=0.35)
    _active_site_lines(ax, sites)
    n_contacts, ident = int(cmap_bin.sum() / 2), pairwise_identity(seq, ref_seq)
    ax.set_xlabel("Residue position", fontsize=9)
    ax.set_ylabel("Residue position", fontsize=9)
    ax.set_title(
        f"{name}\n{n_contacts} per-position contacts  |  {ident:.0f}% identity to AAV9",
        fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))

    ax_legend.axis("off")
    legend_patches = [
        Patch(facecolor="#fec44f", label="Upper tri: ESM2 contact prob."),
        Patch(facecolor="#2980b9", alpha=0.4, label="Lower tri: binary contacts (top 15%)"),
        Patch(facecolor="none", edgecolor="#c0392b", linestyle="--", label="Catalytic dyad (H + inferred D)"),
        Patch(facecolor="none", edgecolor="#2980b9", linestyle=":", label="Ca²⁺-binding (DxxxxxHD Ds)"),
        Patch(facecolor="none", edgecolor="#27ae60", linestyle="-.", label="YxGxG Ca²⁺ loop"),
        Patch(facecolor="none", edgecolor="#8e44ad", linestyle="-.", label="Phospholipid-binding Lys"),
    ]
    ax_legend.legend(handles=legend_patches, loc="center", ncol=3, fontsize=7.5,
                     framealpha=0.95, edgecolor="#cccccc")

    # Right panel: sequence annotation
    ax2.axis("off")
    n_mut, simil = sum(1 for a, b in zip(seq, ref_seq) if a != b), pairwise_similarity(seq, ref_seq)
    match_str = "".join("|" if a == b else "." for a, b in zip(seq, ref_seq))

    site_lines = []
    if 'ca_D1' in sites and 'ca_D2' in sites:
        d1, d2 = sites['ca_D1'], sites['ca_D2']
        site_lines.append(f"  DxxxxxHD:     pos {d1}-{d2}  {seq[d1:d2+1]}")
    if 'cat_H' in sites:
        h = sites['cat_H']
        site_lines.append(f"  Catalytic H:  pos {h}  ({seq[h]})")
    if 'cat_D_inferred' in sites:
        d = sites['cat_D_inferred']
        site_lines.append(f"  Catalytic D:  pos {d}  ({seq[d]})  [contact-inferred]")
    elif 'cat_D' in sites:
        d = sites['cat_D']
        site_lines.append(f"  Catalytic D:  pos {d}  ({seq[d]})  [HAD motif]")
    if 'yxgxg' in sites:
        y = sites['yxgxg']
        site_lines.append(f"  YxGxG loop:   pos {y}-{y+4}  {seq[y:y+5]}")
    if 'phos_K' in sites:
        k = sites['phos_K']
        site_lines.append(f"  Phos. Lys:    pos {k}  ({seq[k]})")

    text = (
        f"  Sequence:   {name}\n"
        f"  Length:     {L} aa\n"
        f"  Identity:   {ident:.1f}%  ({L - n_mut}/{L} identical)\n"
        f"  Similarity: {simil:.1f}%  (BLOSUM90 >= 0)\n"
        f"  Mutations:  {n_mut} substitutions\n"
        f"\n"
        f"  AAV9:  {ref_seq}\n"
        f"  Query: {seq}\n"
        f"  Match: {match_str}\n"
        f"\n"
        + "\n".join(site_lines))
    ax2.text(0.02, 0.96, text, transform=ax2.transAxes, fontsize=7.5, fontfamily="monospace",
             verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#fafafa", edgecolor="#d0d0d0", linewidth=0.6))
    plt.savefig(filepath, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close()


# --- Heatmap helpers ---

def _lum_color(val, vmin, vmax, cm):
    t = np.clip((val - vmin) / (vmax - vmin + 1e-8), 0, 1)
    r, g, b, _ = plt.get_cmap(cm)(t)
    return "black" if 0.299*r + 0.587*g + 0.114*b > 0.45 else "white"


def _select_diverse(items, n, ref_idx=None):
    if not items:
        return []
    items.sort(key=lambda x: x[1])
    n = min(n, len(items))
    step = max(1, len(items) // n)
    picks = [items[min(i * step, len(items) - 1)][0] for i in range(n)]
    if ref_idx is not None and ref_idx not in picks:
        picks[0] = ref_idx
    return sorted(set(picks))


# --- Main ---

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    raw = parse_fasta("data/natural_pla2_domains.fasta")
    all_names, all_seqs = [n for n, _ in raw], [s for _, s in raw]
    ref_idx = next((i for i, n in enumerate(all_names) if "AAV9" in n), None)
    if ref_idx is None:
        print("ERROR: No AAV9 reference")
        return
    ref_seq = all_seqs[ref_idx]
    motif_off = find_catalytic_motif(ref_seq)
    if motif_off is None:
        print("ERROR: No DxxxxxHD motif")
        return
    ref_sites = get_all_active_site_positions(ref_seq)
    gen_pairs = [(n.split("|")[0], s) for n, s in parse_fasta("output/generated.fasta")
                   if "reference" not in n]
    gen_seqs = [s for _, s in gen_pairs]
    contact_freq, contacts_per_pos = load_contact_data()
    n_pairs = int(np.sum(contact_freq > 0) // 2)
    print(f"Contact model: {n_pairs} pairs, per-position "
          f"[{contacts_per_pos.min():.0f}–{contacts_per_pos.max():.0f}], "
          f"median={np.median(contacts_per_pos):.0f}")
    print("\nFigure 1: Contact frequency model")
    plot_frequency_map(contact_freq, contacts_per_pos, ref_sites,
                       os.path.join(OUTDIR, "contact_frequency_model.png"))
    # Named type members grouped by genus for heatmap readability
    type_members_ordered = [
        ("AAV9(AAS99264.1)", "AAV9"),
        ("NP_041244.1",      "AAV2"),
        ("YP_068411.1",      "AAV5"),
        ("NP_852781.1",      "GPV"),
        ("YP_009111340.1",   "BtPV"),
        ("NP_041404.1",      "MVM"),
        ("P07299.1",         "CPV"),
        ("NP_542611.1",      "B19"),
        ("NP_046815.1",      "AMDV"),
        ("NP_694840.1",      "HBoV"),
        ("YP_009507374.1",   "ChPV"),
        ("NP_874379.2",      "GmDNV"),
    ]
    genus_groups = [
        ("Dependoparvovirus", 5),
        ("Protoparvovirus", 2),
        ("Erythroparvovirus", 1),
        ("Amdoparvovirus", 1),
        ("Bocaparvovirus", 1),
        ("Aveparvovirus", 1),
        ("Densovirus", 1),
    ]
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    key_idx, key_names = [], []
    for acc, short in type_members_ordered:
        if acc in name_to_idx:
            key_idx.append(name_to_idx[acc])
            key_names.append(short)
    key_seqs = [all_seqs[i] for i in key_idx]

    gen_pick = _select_diverse(
        [(i, pairwise_identity(s, ref_seq)) for i, s in enumerate(gen_seqs)], 6)
    picked_names = [f"GEN_{k+1}" for k in range(len(gen_pick))]
    picked_seqs = [gen_seqs[i] for i in gen_pick]
    viz_names, viz_seqs = key_names + picked_names, key_seqs + picked_seqs
    viz_is_gen = [False] * len(key_names) + [True] * len(picked_names)
    n_nat = len(key_names)
    print(f"\n  Natural: {n_nat}, Generated: {len(picked_names)}")
    print("\nLoading ESM2...")
    model, alphabet, bc = load_esm2_model("cpu")
    print(f"Predicting contacts for {len(viz_seqs)} sequences...")
    raw_cmaps, bin_cmaps = [], []
    for i, (name, seq) in enumerate(zip(viz_names, viz_seqs)):
        print(f"  {i+1}/{len(viz_seqs)}: {name[:30]}...")
        cmap = predict_contacts_esm2(seq, model, alphabet, bc, "cpu")
        raw_cmaps.append(cmap)
        bin_cmaps.append(binarize_contacts(cmap))
    del model

    # --- Per-sequence active site positions ---
    all_sites = []
    for i, (name, seq) in enumerate(zip(viz_names, viz_seqs)):
        sites = get_all_active_site_positions(seq, contact_map=raw_cmaps[i])
        all_sites.append(sites)

    # --- Catalytic Asp inference table ---
    h_pos = motif_off + 6
    print(f"\n{'='*80}")
    print(f"  Inferred Active Site Residues (per-sequence)")
    print(f"{'='*80}")
    print(f"  {'Variant':<16s} {'DxxxxxHD':>8s} {'cat-H':>5s} {'cat-D':>5s} {'YxGxG':>7s} {'phos-K':>6s} {'D score':>8s}")
    print(f"  {'-'*16} {'-'*8} {'-'*5} {'-'*5} {'-'*7} {'-'*6} {'-'*8}")
    for i, (name, seq) in enumerate(zip(viz_names, viz_seqs)):
        s = all_sites[i]
        tag = "*" if viz_is_gen[i] else " "
        dxh = f"{s.get('ca_D1','?')}-{s.get('ca_D2','?')}" if 'ca_D1' in s else "N/A"
        cat_h = str(s.get('cat_H', 'N/A'))
        cat_d = str(s.get('cat_D_inferred', s.get('cat_D', 'N/A')))
        yxg = f"{s['yxgxg']}-{s['yxgxg']+4}" if 'yxgxg' in s else "N/A"
        pk = str(s.get('phos_K', 'N/A'))
        asp_pos, asp_score = infer_catalytic_asp(seq, raw_cmaps[i], h_pos=h_pos)
        sc = f"{asp_score:.3f}" if asp_pos is not None else "-"
        print(f" {tag}{name:<16s} {dxh:>8s} {cat_h:>5s} {cat_d:>5s} {yxg:>7s} {pk:>6s} {sc:>8s}")
    print(f"{'='*80}")
    print("  (* = generated sequence)\n")

    print(f"Figure 2: Individual contact maps → {OUTDIR}/")
    for i, (name, seq) in enumerate(zip(viz_names, viz_seqs)):
        safe = re.sub(r'[^\w\-.]', '_', name)
        tag = "generated" if viz_is_gen[i] else "natural"
        path = os.path.join(OUTDIR, f"{tag}_{safe}.png")
        plot_single_contact(raw_cmaps[i], bin_cmaps[i], contact_freq,
                            name, seq, ref_seq, all_sites[i], path)
        print(f"  {path}")
    print("\nFigure 3: Pairwise comparison heatmaps")
    n_total = len(viz_names)
    sim_j, sim_id, sim_bl = [np.zeros((n_total, n_total)) for _ in range(3)]
    for i in range(n_total):
        for j in range(n_total):
            sim_j[i, j] = jaccard_binary(bin_cmaps[i], bin_cmaps[j])
            sim_id[i, j] = pairwise_identity(viz_seqs[i], viz_seqs[j])
            sim_bl[i, j] = pairwise_similarity(viz_seqs[i], viz_seqs[j])
    labels = [n[:25] for n in viz_names]
    cmap_shared = "YlOrRd"
    fig, axes = plt.subplots(3, 1, figsize=(12, 28))
    fig.subplots_adjust(hspace=0.08, left=0.25, right=0.92, bottom=0.02, top=0.94)
    panels = [
        (axes[0], sim_j * 100, "A. Contact Map Similarity (Jaccard Index × 100)", 0, 100, ".0f", "Jaccard × 100"),
        (axes[1], sim_id, "B. Sequence Identity (% identical residues)", 0, 100, ".0f", "% Identity"),
        (axes[2], sim_bl, "C. Sequence Similarity (% BLOSUM90 >= 0)", 0, 100, ".0f", "% Similarity"),
    ]
    for ax, mat, title, vmin, vmax, fmt, cbar_label in panels:
        im = ax.imshow(mat, cmap=cmap_shared, vmin=vmin, vmax=vmax, aspect="equal", interpolation="nearest")
        _cbar_labeled(fig, im, ax, cbar_label, 8, shrink=0.75)
        for i in range(n_total):
            for j in range(n_total):
                c = _lum_color(mat[i, j], vmin, vmax, cmap_shared)
                ax.text(j, i, f"{mat[i, j]:{fmt}}", ha="center", va="center",
                        fontsize=6.5, color=c, fontweight="medium")
        ax.set_xticks(range(n_total))
        ax.set_xticklabels(labels, fontsize=8, rotation=48, ha="right")
        ax.set_yticks(range(n_total))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axhline(y=n_nat - 0.5, color="white", linewidth=3)
        ax.axvline(x=n_nat - 0.5, color="white", linewidth=3)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=18)
    fig.suptitle(
        "PLA2 Domain Comparison: Natural Parvoviral vs Generated Sequences",
        fontsize=14, fontweight="bold", y=0.97)
    plt.savefig("output/figures/similarity_heatmap.png", dpi=250, bbox_inches="tight", facecolor="white")
    plt.savefig("output/figures/similarity_heatmap.pdf", bbox_inches="tight", facecolor="white")
    print("  Saved similarity_heatmap.png and .pdf")
    plt.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
