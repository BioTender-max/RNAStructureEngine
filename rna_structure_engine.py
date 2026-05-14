"""
RNAStructureEngine: RNA Secondary Structure Prediction and Analysis
- Minimum free energy (MFE) folding (Nussinov dynamic programming algorithm)
- Base-pair probability matrix (partition function approximation)
- SHAPE reactivity integration (constrained folding)
- Structural conservation scoring across homologs
- Structure-function correlation (UTR structure vs expression)
"""

import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ─── Data Simulation ────────────────────────────────────────────────────────

N_SEQ = 200
BASES = ['A', 'U', 'G', 'C']
BASE_PAIRS = {('A','U'), ('U','A'), ('G','C'), ('C','G'), ('G','U'), ('U','G')}

def random_rna(length):
    probs = [0.25, 0.25, 0.30, 0.20]
    return ''.join(np.random.choice(BASES, size=length, p=probs))

lengths = np.random.randint(100, 501, size=N_SEQ)
sequences = [random_rna(l) for l in lengths]
expression = np.random.lognormal(mean=3.0, sigma=1.2, size=N_SEQ)
shape_profiles = [np.random.beta(0.8, 2.5, size=l) for l in lengths]

# ─── Algorithm 1: Nussinov MFE ──────────────────────────────────────────────

def can_pair(b1, b2):
    return (b1, b2) in BASE_PAIRS

def nussinov(seq, min_loop=3):
    n = len(seq)
    dp = np.zeros((n, n), dtype=np.int32)
    for gap in range(min_loop + 1, n):
        for i in range(n - gap):
            j = i + gap
            dp[i][j] = dp[i][j-1]
            for k in range(i, j - min_loop):
                pair_score = 1 if can_pair(seq[k], seq[j]) else 0
                left = dp[i][k-1] if k > i else 0
                dp[i][j] = max(dp[i][j], left + pair_score + dp[k+1][j-1])
    return dp

def traceback(dp, seq, i, j, pairs, min_loop=3):
    if i >= j:
        return
    if dp[i][j] == dp[i][j-1]:
        traceback(dp, seq, i, j-1, pairs, min_loop)
    else:
        for k in range(i, j - min_loop):
            pair_score = 1 if can_pair(seq[k], seq[j]) else 0
            left = dp[i][k-1] if k > i else 0
            if left + pair_score + dp[k+1][j-1] == dp[i][j] and pair_score:
                pairs.append((k, j))
                if k > i:
                    traceback(dp, seq, i, k-1, pairs, min_loop)
                traceback(dp, seq, k+1, j-1, pairs, min_loop)
                return

def get_structure(seq, min_loop=3):
    n = len(seq)
    dp = nussinov(seq, min_loop)
    pairs = []
    traceback(dp, seq, 0, n-1, pairs, min_loop)
    dot_bracket = ['.'] * n
    for (i, j) in pairs:
        dot_bracket[i] = '('
        dot_bracket[j] = ')'
    return ''.join(dot_bracket), dp[0][n-1], pairs

print("Computing MFE structures (Nussinov DP)...")
mfe_scores = []
structures = []
all_pairs = []
for idx, seq in enumerate(sequences):
    s = seq[:80]
    struct, mfe, pairs = get_structure(s)
    mfe_scores.append(mfe)
    structures.append(struct)
    all_pairs.append(pairs)
    if (idx+1) % 50 == 0:
        print(f"  Processed {idx+1}/{N_SEQ} sequences")

mfe_scores = np.array(mfe_scores)
mfe_norm = mfe_scores / np.minimum(lengths, 80)

# ─── Algorithm 2: Base-pair probability (partition function approx) ──────────

def approx_bp_probability(seq, n_samples=50, min_loop=3):
    n = len(seq)
    bp_count = np.zeros((n, n))
    for _ in range(n_samples):
        perturbed = list(seq)
        n_mut = max(1, int(n * 0.05))
        mut_pos = np.random.choice(n, n_mut, replace=False)
        for pos in mut_pos:
            perturbed[pos] = np.random.choice(BASES)
        p_seq = ''.join(perturbed)
        _, _, pairs = get_structure(p_seq, min_loop)
        for (i, j) in pairs:
            if i < n and j < n:
                bp_count[i][j] += 1
                bp_count[j][i] += 1
    bp_prob = bp_count / n_samples
    return bp_prob

print("Computing base-pair probability matrix (example sequence)...")
example_seq = sequences[0][:60]
bp_prob_matrix = approx_bp_probability(example_seq, n_samples=80)

# ─── Algorithm 3: SHAPE integration ─────────────────────────────────────────

def nussinov_shape(seq, shape, min_loop=3, shape_thresh=0.7, penalty=5):
    n = len(seq)
    dp = np.zeros((n, n), dtype=np.float32)
    for gap in range(min_loop + 1, n):
        for i in range(n - gap):
            j = i + gap
            dp[i][j] = dp[i][j-1]
            for k in range(i, j - min_loop):
                if can_pair(seq[k], seq[j]):
                    shape_pen = 0
                    if k < len(shape) and shape[k] > shape_thresh:
                        shape_pen += penalty
                    if j < len(shape) and shape[j] > shape_thresh:
                        shape_pen += penalty
                    pair_score = max(0, 1 - shape_pen * 0.1)
                else:
                    pair_score = 0
                left = dp[i][k-1] if k > i else 0
                dp[i][j] = max(dp[i][j], left + pair_score + dp[k+1][j-1])
    return dp[0][n-1]

print("Computing SHAPE-constrained folding...")
shape_mfe = []
for idx in range(min(100, N_SEQ)):
    seq = sequences[idx][:80]
    shape = shape_profiles[idx][:80]
    s_mfe = nussinov_shape(seq, shape)
    shape_mfe.append(s_mfe)
shape_mfe = np.array(shape_mfe)

shape_vals = shape_profiles[0][:60]
bp_prob_diag = np.array([bp_prob_matrix[i, min(i+5, 59)] for i in range(60)])

# ─── Algorithm 4: Conservation scoring ──────────────────────────────────────

def mountain_representation(structure):
    mountain = []
    height = 0
    for c in structure:
        if c == '(':
            height += 1
        elif c == ')':
            height -= 1
        mountain.append(height)
    return np.array(mountain)

def mountain_distance(s1, s2):
    m1 = mountain_representation(s1)
    m2 = mountain_representation(s2)
    min_len = min(len(m1), len(m2))
    m1 = m1[:min_len]
    m2 = m2[:min_len]
    return np.sum(np.abs(m1 - m2)) / (min_len + 1e-9)

print("Computing structural conservation scores...")
n_families = 20
conservation_scores = []
for fam in range(n_families):
    idxs = list(range(fam * 10, (fam + 1) * 10))
    fam_structs = [structures[i] for i in idxs]
    dists = []
    for i in range(len(fam_structs)):
        for j in range(i+1, len(fam_structs)):
            d = mountain_distance(fam_structs[i], fam_structs[j])
            dists.append(d)
    conservation_scores.append(1.0 - np.mean(dists))

conservation_scores = np.array(conservation_scores)

# ─── Algorithm 5: Structure-function correlation ─────────────────────────────

corr_result = stats.pearsonr(mfe_norm, np.log1p(expression))
spearman_result = stats.spearmanr(mfe_norm, np.log1p(expression))

# ─── Stem-loop counting ──────────────────────────────────────────────────────

def count_stem_loops(structure):
    count = 0
    i = 0
    while i < len(structure):
        if structure[i] == '(':
            depth = 1
            j = i + 1
            while j < len(structure) and depth > 0:
                if structure[j] == '(':
                    depth += 1
                elif structure[j] == ')':
                    depth -= 1
                j += 1
            inner = structure[i+1:j-1]
            if '(' not in inner:
                count += 1
            i = j
        else:
            i += 1
    return count

stem_loop_counts = np.array([count_stem_loops(s) for s in structures])

example_struct = structures[5]
mountain_vals = mountain_representation(example_struct)

# ─── Dashboard ───────────────────────────────────────────────────────────────

print("Generating dashboard...")
fig = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor('#0a0a0a')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

COLORS = ['#00ff88', '#ff6b6b', '#4ecdc4', '#ffe66d', '#a29bfe', '#fd79a8', '#74b9ff', '#55efc4']
TEXT_COLOR = 'white'
GRID_COLOR = '#333333'

def style_ax(ax, title):
    ax.set_facecolor('#111111')
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.set_title(title, fontsize=10, fontweight='bold', color=TEXT_COLOR, pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    ax.grid(True, color=GRID_COLOR, alpha=0.4, linewidth=0.5)

# Panel 1: MFE distribution
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(mfe_scores, bins=30, color=COLORS[0], alpha=0.85, edgecolor='#003322')
ax1.axvline(np.mean(mfe_scores), color=COLORS[1], linestyle='--', linewidth=1.5,
            label=f'Mean={np.mean(mfe_scores):.1f}')
ax1.set_xlabel('MFE Score (base pairs)', color=TEXT_COLOR, fontsize=8)
ax1.set_ylabel('Count', color=TEXT_COLOR, fontsize=8)
ax1.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TEXT_COLOR)
style_ax(ax1, 'Panel 1: MFE Distribution')

# Panel 2: Base-pair probability matrix
ax2 = fig.add_subplot(gs[0, 1])
cmap_bp = LinearSegmentedColormap.from_list('bp', ['#111111', '#00ff88', '#ffe66d'])
im2 = ax2.imshow(bp_prob_matrix, cmap=cmap_bp, aspect='auto', origin='lower', vmin=0, vmax=0.5)
cb2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cb2.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
ax2.set_xlabel('Position j', color=TEXT_COLOR, fontsize=8)
ax2.set_ylabel('Position i', color=TEXT_COLOR, fontsize=8)
style_ax(ax2, 'Panel 2: BP Probability Matrix')

# Panel 3: SHAPE reactivity vs BP probability
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(shape_vals, bp_prob_diag, c=np.arange(60), cmap='plasma', alpha=0.7, s=25)
ax3.axvline(0.7, color=COLORS[1], linestyle='--', linewidth=1.2, label='SHAPE=0.7')
ax3.set_xlabel('SHAPE Reactivity', color=TEXT_COLOR, fontsize=8)
ax3.set_ylabel('BP Probability', color=TEXT_COLOR, fontsize=8)
ax3.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TEXT_COLOR)
style_ax(ax3, 'Panel 3: SHAPE vs BP Probability')

# Panel 4: Mountain plot
ax4 = fig.add_subplot(gs[1, 0])
x_mtn = np.arange(len(mountain_vals))
ax4.fill_between(x_mtn, mountain_vals, alpha=0.6, color=COLORS[3])
ax4.plot(x_mtn, mountain_vals, color=COLORS[3], linewidth=1.2)
ax4.set_xlabel('Position (nt)', color=TEXT_COLOR, fontsize=8)
ax4.set_ylabel('Nesting Depth', color=TEXT_COLOR, fontsize=8)
style_ax(ax4, 'Panel 4: Mountain Plot (Example Structure)')

# Panel 5: Conservation score distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.bar(range(n_families), conservation_scores, color=COLORS[4], alpha=0.85, edgecolor='#220044')
ax5.axhline(np.mean(conservation_scores), color=COLORS[1], linestyle='--', linewidth=1.5,
            label=f'Mean={np.mean(conservation_scores):.3f}')
ax5.set_xlabel('Homolog Family', color=TEXT_COLOR, fontsize=8)
ax5.set_ylabel('Conservation Score', color=TEXT_COLOR, fontsize=8)
ax5.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TEXT_COLOR)
style_ax(ax5, 'Panel 5: Structure Conservation')

# Panel 6: MFE vs expression
ax6 = fig.add_subplot(gs[1, 2])
sc6 = ax6.scatter(mfe_norm, np.log1p(expression), c=lengths, cmap='viridis', alpha=0.6, s=20)
cb6 = plt.colorbar(sc6, ax=ax6, fraction=0.046, pad=0.04, label='Length')
cb6.ax.yaxis.set_tick_params(color=TEXT_COLOR, labelcolor=TEXT_COLOR)
m, b = np.polyfit(mfe_norm, np.log1p(expression), 1)
x_line = np.linspace(mfe_norm.min(), mfe_norm.max(), 100)
ax6.plot(x_line, m*x_line + b, color=COLORS[1], linewidth=1.5, label=f'r={corr_result[0]:.3f}')
ax6.set_xlabel('Normalized MFE', color=TEXT_COLOR, fontsize=8)
ax6.set_ylabel('log(Expression+1)', color=TEXT_COLOR, fontsize=8)
ax6.legend(fontsize=7, facecolor='#1a1a1a', labelcolor=TEXT_COLOR)
style_ax(ax6, 'Panel 6: MFE vs Expression')

# Panel 7: Sequence length vs MFE
ax7 = fig.add_subplot(gs[2, 0])
ax7.scatter(lengths, mfe_scores, c=COLORS[5], alpha=0.5, s=18)
m7, b7 = np.polyfit(lengths, mfe_scores, 1)
x7 = np.linspace(lengths.min(), lengths.max(), 100)
ax7.plot(x7, m7*x7 + b7, color=COLORS[0], linewidth=1.5)
ax7.set_xlabel('Sequence Length (nt)', color=TEXT_COLOR, fontsize=8)
ax7.set_ylabel('MFE Score', color=TEXT_COLOR, fontsize=8)
style_ax(ax7, 'Panel 7: Length vs MFE')

# Panel 8: Stem-loop count distribution
ax8 = fig.add_subplot(gs[2, 1])
unique_sl, counts_sl = np.unique(stem_loop_counts, return_counts=True)
ax8.bar(unique_sl, counts_sl, color=COLORS[6], alpha=0.85, edgecolor='#002244')
ax8.set_xlabel('Stem-loop Count', color=TEXT_COLOR, fontsize=8)
ax8.set_ylabel('Number of Sequences', color=TEXT_COLOR, fontsize=8)
style_ax(ax8, 'Panel 8: Stem-loop Count Distribution')

# Panel 9: Summary text
ax9 = fig.add_subplot(gs[2, 2])
ax9.set_facecolor('#111111')
ax9.axis('off')
for spine in ax9.spines.values():
    spine.set_edgecolor('#444444')

summary_lines = [
    "RNA STRUCTURE ENGINE SUMMARY",
    "─" * 32,
    f"Sequences analyzed:    {N_SEQ}",
    f"Length range:          {lengths.min()}-{lengths.max()} nt",
    f"Mean MFE score:        {np.mean(mfe_scores):.2f} bp",
    f"Mean norm. MFE:        {np.mean(mfe_norm):.4f}",
    f"Mean stem-loops:       {np.mean(stem_loop_counts):.2f}",
    f"Mean conservation:     {np.mean(conservation_scores):.4f}",
    f"MFE-expr Pearson r:    {corr_result[0]:.4f}",
    f"MFE-expr p-value:      {corr_result[1]:.4e}",
    f"MFE-expr Spearman r:   {spearman_result[0]:.4f}",
    f"SHAPE-constrained MFE: {np.mean(shape_mfe):.2f}",
    f"High-SHAPE fraction:   {np.mean([np.mean(s>0.7) for s in shape_profiles[:100]]):.3f}",
]
ax9.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax9.transAxes,
         fontsize=8, verticalalignment='top', fontfamily='monospace',
         color=TEXT_COLOR, bbox=dict(boxstyle='round', facecolor='#1a1a1a', alpha=0.8))
ax9.set_title('Panel 9: Summary', fontsize=10, fontweight='bold', color=TEXT_COLOR, pad=8)

fig.suptitle('RNAStructureEngine: RNA Secondary Structure Analysis Dashboard',
             fontsize=14, fontweight='bold', color=TEXT_COLOR, y=0.98)

plt.savefig('/workspace/subagents/7c45dd59/rna_structure_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor='#0a0a0a', edgecolor='none')
plt.close()
print("Dashboard saved: /workspace/subagents/7c45dd59/rna_structure_dashboard.png")

# ─── Structured Summary ──────────────────────────────────────────────────────

print("\n" + "="*60)
print("RNA STRUCTURE ENGINE — STRUCTURED SUMMARY")
print("="*60)
print(f"Total sequences analyzed:        {N_SEQ}")
print(f"Sequence length range:           {lengths.min()} – {lengths.max()} nt")
print(f"Mean sequence length:            {np.mean(lengths):.1f} nt")
print(f"Mean MFE score (base pairs):     {np.mean(mfe_scores):.2f}")
print(f"Std MFE score:                   {np.std(mfe_scores):.2f}")
print(f"Mean normalized MFE:             {np.mean(mfe_norm):.4f}")
print(f"Mean stem-loop count:            {np.mean(stem_loop_counts):.2f}")
print(f"Max stem-loop count:             {stem_loop_counts.max()}")
print(f"Mean conservation score:         {np.mean(conservation_scores):.4f}")
print(f"Min/Max conservation:            {conservation_scores.min():.4f} / {conservation_scores.max():.4f}")
print(f"MFE–Expression Pearson r:        {corr_result[0]:.4f}  (p={corr_result[1]:.4e})")
print(f"MFE–Expression Spearman r:       {spearman_result[0]:.4f}  (p={spearman_result[1]:.4e})")
print(f"Mean SHAPE-constrained MFE:      {np.mean(shape_mfe):.2f}")
print(f"Fraction high-SHAPE (>0.7):      {np.mean([np.mean(s>0.7) for s in shape_profiles[:100]]):.3f}")
print(f"BP probability matrix shape:     {bp_prob_matrix.shape}")
print(f"Mean BP probability:             {np.mean(bp_prob_matrix):.4f}")
print("="*60)
