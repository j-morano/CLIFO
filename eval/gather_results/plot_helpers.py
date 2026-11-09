"""Helpers of make_plots.py: paired significance tests, LaTeX table helpers and box / bar plots."""
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from plot_utils import get_palette
from paths import GEN_RESULTS, RAW_RESULTS


BASE_PATH = GEN_RESULTS


CSV_DIR = RAW_RESULTS


FIG_FORMAT = 'svg'


def get_sig_stars(p_value):
    if p_value is None or pd.isna(p_value):
        return ''
    if p_value < 0.001:
        return '$^{***}$'
    elif p_value < 0.01:
        return '$^{**}$'
    elif p_value < 0.05:
        return '$^{*}$'
    return ''


def longtable_header(lines, n_cols, header_row, caption, label):
    """Appends the longtable preamble (caption, first/repeat heads, feet)."""
    lines.append(caption + f' \\label{{{label}}} \\\\')
    lines.append('\\toprule')
    lines.append(header_row)
    lines.append('\\midrule')
    lines.append('\\endfirsthead')
    lines.append(f'\\multicolumn{{{n_cols}}}{{c}}{{\\tablename\\ \\thetable{{}} -- continued from previous page}} \\\\')
    lines.append('\\toprule')
    lines.append(header_row)
    lines.append('\\midrule')
    lines.append('\\endhead')
    lines.append('\\midrule')
    lines.append(f'\\multicolumn{{{n_cols}}}{{r}}{{\\textit{{continued on next page}}}} \\\\')
    lines.append('\\endfoot')
    lines.append('\\bottomrule')
    lines.append('\\endlastfoot')


def save_latex(lines, save_fn):
    latex_str = '\n'.join(lines)
    print('\n' + '=' * 60)
    print(f'LATEX TABLE -> {save_fn}')
    print('=' * 60)
    print(latex_str)
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    with open(save_fn, 'w') as f:
        f.write(latex_str)
    print(f'Saved to {save_fn}')


def sig_str_from_pvalue(p_value):
    if p_value < 0.001:
        return '$p<0.001$'
    elif p_value < 0.01:
        return '$p<0.01$'
    elif p_value < 0.05:
        return '$p<0.05$'
    return f'$p={p_value:.3f}$'


def wrap_for_plot(label):
    """Insert a line break before a trailing '(...)' for cleaner plot
    x-axis labels, without needing '\n' stored in the data itself."""
    if ' (' in label:
        return label.replace(' (', '\n(', 1)
    return label


def draw_mean_annotation(ax, x, mean_val, value_fmt='{:.3f}', mean_style='marker', box_half_width=0.38,
                          dot_spacing_px=6):
    """
    Draws the mean marker/label for a single box at x position `x`.
    mean_style options:
      - 'marker' (default): standard seaborn-style circle marker + boxed
        number placed just above it (caller must pass showmeans=True to
        sns.boxplot for the marker itself to render).
      - 'dots': no marker; a row of small white squares (thin black
        border) spans the box width at the mean height, spaced roughly
        `dot_spacing_px` pixels apart regardless of figure size, with the
        value centered directly on top (outlined text for legibility).
      - 'diamond': a white diamond marker with a black border sits at
        the mean height (shape deliberately distinct from the circular
        outlier/stripplot points), with the value labeled just above it
        (close by, but not touching the marker).
    """
    if mean_style == 'dots':
        p_left = ax.transData.transform((x - box_half_width, mean_val))
        p_right = ax.transData.transform((x + box_half_width, mean_val))
        width_px = abs(p_right[0] - p_left[0])
        n_dots = max(3, int(round(width_px / dot_spacing_px)) + 1)
        dot_xs = np.linspace(x - box_half_width, x + box_half_width, n_dots)
        ax.scatter(
            dot_xs, [mean_val] * n_dots,
            s=6, marker='s', color='white', edgecolor='black', linewidth=0.8,
            zorder=9,
        )
        ax.text(
            x, mean_val, value_fmt.format(mean_val),
            va='center', ha='center', color='white', fontweight='bold', fontsize=9,
            path_effects=[pe.withStroke(linewidth=2.2, foreground='black')],
            zorder=10,
        )
    elif mean_style == 'diamond':
        ax.scatter(
            [x], [mean_val],
            s=32, marker='D', color='white', edgecolor='black', linewidth=1.0,
            zorder=9,
        )
        # Text placed a small fixed pixel offset above the marker (close
        # by, not touching it), regardless of the y-axis scale/units of
        # the plot (white fill with a black outline, same style as the
        # 'dots' variant, for legibility over any box color)
        text = ax.annotate(
            value_fmt.format(mean_val),
            xy=(x, mean_val), xycoords='data',
            xytext=(0, 5), textcoords='offset points',
            ha='center', va='bottom', color='white', fontweight='bold', fontsize=9,
            zorder=10,
        )
        text.set_path_effects([pe.withStroke(linewidth=2.2, foreground='black')])
    else:
        ax.text(
            x, mean_val, value_fmt.format(mean_val),
            va='bottom', ha='center', color='black', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
        )


def signflip_pvalue(diff, alternative='greater', exact_max_n=16, n_mc=200000, seed=0):
    """Paired sign-flip permutation test on the mean of the differences. Exact over all 2^n
    sign patterns when n <= exact_max_n, Monte Carlo (n_mc random patterns) otherwise."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n < 2:
        return None
    obs = diff.mean()
    if n <= exact_max_n:
        signs = np.array(list(itertools.product([1, -1], repeat=n)))
    else:
        signs = np.random.default_rng(seed).choice([1, -1], size=(n_mc, n))
    null = (signs * diff).mean(axis=1)
    tol = 1e-12
    if alternative == 'less':
        return float(np.mean(null <= obs + tol))
    return float(np.mean(null >= obs - tol))


def paired_pvalue(x, y, alternative='greater'):
    """One-sided paired sign-flip permutation test of x vs y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if len(x) < 2:
        return None
    try:
        return signflip_pvalue(x - y, alternative)
    except Exception:
        return None


def paired_test_name():
    return 'one-sided paired sign-flip permutation test'


def paired_unit_pvalue(unit_means, best_model, other_model, metric):
    """One-sided paired sign-flip test, best vs other, on one value per unit
    (unit = dataset, or dataset-task for segmentation). Direction: 'less' for
    HD/MAD (lower is better), 'greater' otherwise. unit_means has columns
    Unit, Model and metric, with one row per (Unit, Model)."""
    alt = 'less' if (metric.startswith('HD') or metric == 'MAD') else 'greater'
    wide = unit_means.pivot(index='Unit', columns='Model', values=metric)
    if best_model not in wide.columns or other_model not in wide.columns:
        return None
    pair = wide[[best_model, other_model]].dropna()
    if len(pair) < 2:
        return None
    try:
        return paired_pvalue(pair[best_model].values, pair[other_model].values, alt)
    except Exception:
        return None


def compute_pvalue_avg(full_df, best_model, other_model, metric, single_dataset):
    if single_dataset:
        best_data = full_df[full_df['Model'] == best_model].sort_values('Run')[metric].values
        other_data = full_df[full_df['Model'] == other_model].sort_values('Run')[metric].values
        try:
            return paired_pvalue(best_data, other_data, 'greater')
        except Exception:
            return None
    # One value per dataset (runs averaged), paired one-sided test across datasets
    units = full_df.groupby(['Dataset', 'Model'], observed=True)[metric].mean().reset_index()
    return paired_unit_pvalue(units.rename(columns={'Dataset': 'Unit'}), best_model, other_model, metric)


def compute_pvalue_per_dataset(dataset_df, best_model, other_model, metric):
    best_data = pd.to_numeric(dataset_df[dataset_df['Model'] == best_model].sort_values('Run')[metric], errors='coerce').values
    other_data = pd.to_numeric(dataset_df[dataset_df['Model'] == other_model].sort_values('Run')[metric], errors='coerce').values
    if len(best_data) != len(other_data) or len(best_data) < 2:
        return None
    try:
        p = paired_pvalue(best_data, other_data, 'greater')
        return p
    except Exception:
        return None


def cls_plot_boxplot(full_df, prescribed_order, experiment, metric, single_dataset,
                      fig_width=None, second_vs_third=False, mean_style='marker', brackets=None):
    # One value per dataset (runs averaged), matching the paired test
    plot_df = full_df.groupby(['Dataset', 'Model'], observed=True)[metric].mean().reset_index()
    means_by_model = plot_df.groupby('Model', observed=True)[metric].mean()
    ranked = means_by_model.sort_values(ascending=False)
    best_model = ranked.index[0]
    second_best_model = ranked.index[1]
    p_value_1 = compute_pvalue_avg(full_df, best_model, second_best_model, metric, single_dataset)
    best_idx = prescribed_order.index(best_model)
    second_best_idx = prescribed_order.index(second_best_model)

    p_value_2 = None
    third_idx = None
    if second_vs_third and len(ranked) >= 3:
        third_model = ranked.index[2]
        p_value_2 = compute_pvalue_avg(full_df, second_best_model, third_model, metric, single_dataset)
        third_idx = prescribed_order.index(third_model)

    width = fig_width if fig_width is not None else len(prescribed_order)
    plt.figure(figsize=(width, 5))
    sns.set_style("whitegrid")
    ax = sns.boxplot(
        data=plot_df, hue='Model', x='Model', y=metric,
        palette=get_palette(prescribed_order), order=prescribed_order,
        showmeans=(mean_style == 'marker'),
        meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "8"} if mean_style == 'marker' else None,
        legend=False,
        showfliers=False,
    )
    sns.stripplot(data=plot_df, x='Model', y=metric, order=prescribed_order, color="black", alpha=0.3, jitter=True, size=4)
    ax.set_xticklabels([wrap_for_plot(m) for m in prescribed_order])

    means = plot_df.groupby('Model', observed=True)[metric].mean()
    for i, model_name in enumerate(prescribed_order):
        if model_name in means:
            draw_mean_annotation(ax, i, means[model_name], mean_style=mean_style)

    y_min, y_max = ax.get_ylim()
    span = y_max - y_min

    def draw_bracket(idx_a, idx_b, p_value, height):
        ax.plot([idx_a, idx_b], [height, height], 'k-', linewidth=2)
        ax.plot([idx_a, idx_a], [height - span * 0.02, height], 'k-', linewidth=2)
        ax.plot([idx_b, idx_b], [height - span * 0.02, height], 'k-', linewidth=2)
        ax.text((idx_a + idx_b) / 2, height + span * 0.02, sig_str_from_pvalue(p_value),
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    top_extra = 0.1
    if brackets is not None:
        # Explicit comparisons (a, b): one-sided test that a is better than b, drawn between their boxes
        for k, (a, b) in enumerate(brackets):
            p = compute_pvalue_avg(full_df, a, b, metric, single_dataset)
            ia, ib = prescribed_order.index(a), prescribed_order.index(b)
            draw_bracket(min(ia, ib), max(ia, ib), p, y_max + span * 0.12 * k)
        top_extra += 0.12 * len(brackets)
    else:
        if p_value_1 is not None:
            draw_bracket(best_idx, second_best_idx, p_value_1, y_max)
            top_extra += 0.07
        if p_value_2 is not None:
            draw_bracket(second_best_idx, third_idx, p_value_2, y_max + span * 0.12)
            top_extra += 0.07

    ax.set_ylim(y_min, y_max + span * top_extra)

    plt.ylabel(metric)
    plt.xlabel('Model')
    plt.tight_layout()
    save_fn = BASE_PATH / experiment / f'{experiment}_boxplot_{metric}.{FIG_FORMAT}'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().patch.set_alpha(0)
    plt.gca().set_facecolor('white')
    plt.savefig(save_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Boxplot saved to {save_fn}")


def cls_plot_barplot(full_df, prescribed_order, experiment, metric):
    datasets = sorted(full_df['Dataset'].unique())
    plt.figure(figsize=(max(8, 5 * len(datasets)), 8))
    ax = sns.barplot(
        data=full_df, x='Dataset', y=metric, hue='Model',
        palette=get_palette(prescribed_order), hue_order=prescribed_order,
        errorbar='sd', capsize=.1,
    )
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            ax.text(p.get_x() + p.get_width() / 2., height + 0.005, f'{height * 100:.1f}',
                     ha='center', va='bottom', color='black', fontweight='bold')
    plt.ylabel(metric)
    plt.xlabel('Dataset')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, [wrap_for_plot(l) for l in labels], title='Model',
               bbox_to_anchor=(0.5, -0.15), loc='center', ncol=len(prescribed_order))
    plt.tight_layout()
    save_fn = BASE_PATH / experiment / f'{experiment}_barplot_{metric}.{FIG_FORMAT}'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().patch.set_alpha(0)
    plt.gca().set_facecolor('white')
    plt.savefig(save_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Barplot saved to {save_fn}")


CLS_PAPER_DS_ORDER = [
    ('AMD_DME_3D', 'AMD-DME-3D'), ('BanglaOCT2025', 'BanglaOCT2025'), ('GAMMA', 'GAMMA'),
    ('Harvard_Glaucoma', 'Harvard Glaucoma'), ('M3Ret', 'M3Ret'), ('Noor_Eye_Hospital', 'NEH'),
    ('OCTA500', 'OCTA-500'), ('OCTAVE', 'OCTAVE'), ('OCT_MD_MS', 'OCT MD MS'), ('OIMHS', 'OIMHS'),
    ('OLIVES_baseline', 'OLIVES'), ('POAGD', 'POAGD'),
    ('Noor_to_UMNandDukeSrinivasan_cross', 'UMN + Duke Srinivasan (trained on NEH)'),
    ('UMNandDukeSrinivasan_to_Noor_cross', 'NEH (trained on UMN + Duke Srinivasan)'),
]


def seg_volume_level(full_df, metric, pool_seeds=False):
    """Average across classes within each volume. Seed-aware (default): one row per
    (Dataset, Model, Seed, ID). pool_seeds=True: old behaviour, seeds averaged per ID."""
    keys = ['Dataset', 'Model', 'ID']
    if not pool_seeds and 'Seed' in full_df.columns:
        keys = ['Dataset', 'Model', 'Seed', 'ID']
    return full_df.groupby(keys, observed=True)[metric].mean().reset_index()


def seg_level_df(full_df, metric, level, pool_seeds=False):
    """
    Returns the granularity that mean/std and the significance test
    should be computed on, matching the original per-level semantics:
      - 'volume' / 'volume_wilcoxon': patient-level (one row per ID),
        pooled across all datasets.
      - 'dataset': one row per (Dataset, Model), averaged across patients
        within each dataset first (so mean/std afterward is computed
        ACROSS DATASETS, not across patients).
    """
    vol_df = seg_volume_level(full_df, metric, pool_seeds)
    if level in ('volume', 'volume_wilcoxon'):
        return vol_df
    keys = ['Dataset', 'Model'] + (['Seed'] if 'Seed' in vol_df.columns else [])
    return vol_df.groupby(keys, observed=True)[metric].mean().reset_index()


def seg_compute_pvalue_avg(level_df, best_model, other_model, metric):
    """One value per dataset-task (seeds and volumes averaged), paired one-sided test across dataset-tasks.
    level_df must be at the granularity produced by seg_level_df(..., 'dataset')."""
    units = level_df.groupby(['Dataset', 'Model'], observed=True)[metric].mean().reset_index()
    return paired_unit_pvalue(units.rename(columns={'Dataset': 'Unit'}), best_model, other_model, metric)


def seg_compute_pvalue_per_dataset(dataset_df, best_model, other_model, metric, higher_is_better):
    """Always volume-level within a single dataset, the only granularity
    with repeated (patient) measures to pair on inside one dataset.
    Seed-aware data are paired on (ID, Seed); pooled data on ID."""
    alt = 'greater' if higher_is_better else 'less'
    # Average seeds per volume, then pair by volume ID (one unit per volume).
    best_data = dataset_df[dataset_df['Model'] == best_model].groupby('ID')[metric].mean().sort_index()
    other_data = dataset_df[dataset_df['Model'] == other_model].groupby('ID')[metric].mean().sort_index()
    common = best_data.index.intersection(other_data.index)
    if len(common) < 2:
        return None
    try:
        return paired_pvalue(best_data.loc[common].values, other_data.loc[common].values, alt)
    except Exception:
        return None


PAPER_DS_BASE = {
    'AMD_DME_3D': 'AMD-DME-3D', 'AMD_SD_vol': 'AMD-SD', 'AROI_nosclera': 'AROI',
    'Duke_DME_nosclera': 'Duke DME', 'Duke_iAMD_labeled_nosclera_70': 'Duke iAMD',
    'GOALS': 'GOALS', 'OCTAVE': 'OCTAVE', 'OIMHS': 'OIMHS', 'RETOUCH': 'RETOUCH',
    'RVO_Lesion': 'RVO-ME', 'UMN': 'UMN',
}


def paper_dataset_label(dataset_name):
    """'AMD_SD_vol_layers' -> 'AMD-SD -- Layers' (paper table style)."""
    if dataset_name.endswith('_layers'):
        base, grp = dataset_name[:-len('_layers')], 'Layers'
    elif dataset_name.endswith('_lesions'):
        base, grp = dataset_name[:-len('_lesions')], 'Lesions'
    else:
        base, grp = dataset_name, ''
    label = PAPER_DS_BASE.get(base, base)
    label = f'{label} -- {grp}' if grp else label
    if base == 'Duke_iAMD_labeled_nosclera_70':
        label += ' (trained on AROI)'
    return label


def ret_plot_barplot(df, prescribed_order, metric, metric_label):
    datasets = sorted(df['dataset_name'].unique())
    plt.figure(figsize=(max(8, 1.2 * len(datasets)), 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(
        data=df, x='dataset_name', y=metric, hue='model_name',
        palette=get_palette(prescribed_order), hue_order=prescribed_order,
    )
    for p in ax.patches:
        height = p.get_height()
        if not pd.isna(height):
            ax.text(p.get_x() + p.get_width() / 2., height + 0.005, f'{height:.3f}',
                     ha='center', va='bottom', color='black', fontsize=7, fontweight='bold', rotation=90)
    plt.ylabel(metric_label)
    plt.xlabel('Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model', bbox_to_anchor=(0.5, -0.25), loc='center', ncol=len(prescribed_order))
    plt.tight_layout()
    save_fn = BASE_PATH / 'retrieval' / f'barplot_{metric}.{FIG_FORMAT}'
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().patch.set_alpha(0)
    plt.gca().set_facecolor('white')
    plt.savefig(save_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Barplot saved to {save_fn}")
