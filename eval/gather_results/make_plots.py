"""
Supplementary tables with explicit comparisons.

Statistics and LaTeX helpers are imported from plot_helpers.py, so every test is the
same one-sided paired sign-flip permutation test. What changes is which models are
compared. Each table lists comparisons as (model, reference) pairs: the stars go on
`model`, tested against `reference`. Where pairs is None, the comparison is the best
model against the second best, with stars on the best.

Bold marks the best model per column. Underline marks the second best, only where
underline=True. Tables only, no plots.

Usage (run from eval/gather_results with the eval folder on PYTHONPATH):
    python make_plots.py              # all supplementary tables
    python make_plots.py --kind 3d    # one table
Output: eval/__gen_results/supp/<experiment>/
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import plot_helpers as ph
from plot_helpers import get_sig_stars, save_latex, paired_test_name

OUT_DIR = Path(ph.BASE_PATH) / 'supp'

CLS_METRICS = ['MCC', 'BAcc', 'AUROC', 'AP']
SEG_METRICS = ['Dice', 'HD952D']
DET_F1 = False   # add the Det-F1 column (B-scan lesion detection F1, lesion tasks only) to the segmentation tables; set by --detf1
SEG_HEADER = {'HD952D': 'HD95', 'F1': 'Det-F1'}

# Display name -> (LaTeX label, colour macro or None). Colour macros are defined in the paper preamble.
CLS_2D_DISPLAY = {
    'RETFound': ('RETFound', 'colRETFound'), 'VisionFM': ('VisionFM', 'colVisionFM'),
    'UrFound': ('UrFound', 'colUrFound'), 'OCTCube': ('OCTCube', 'colOCTCube'),
    'MIRAGE': ('MIRAGE', 'colMIRAGE'), 'CLIFO': ('\\mname', 'colOURS'),
}
CLS_3D_RENAME = {'OCTCube (60 B-scans)': 'OCTCube (60)', 'CLIFO (60 B-scans)': 'CLIFO (60)',
                 'CLIFO (all B-scans)': 'CLIFO (all)'}
CLS_3D_DISPLAY = {
    'OCTCube (60)': ('OCTCube (60)', 'colOCTCube'), 'CLIFO (60)': ('\\mnames (60)', 'colOURSBscans'),
    'CLIFO (all)': ('\\mnames (all)', 'colOURS'),
}

# 2D per-dataset order and labels, as in the supplement.
# Cross-dataset tasks always come last, right before the Average: NEH (trained on UMN + Duke) first.
CROSS_LAST = [
    ('UMNandDukeSrinivasan_to_Noor_cross', 'NEH (trained on UMN + Duke Srinivasan)'),
    ('Noor_to_UMNandDukeSrinivasan_cross', 'UMN + Duke Srinivasan (trained on NEH)'),
]
DS_3D = [(d, l) for d, l in ph.CLS_PAPER_DS_ORDER if d not in dict(CROSS_LAST)] + CROSS_LAST

# Long dataset labels are split over the rows of their block, one piece per model row (by number of models).
LABEL_LINES = {
    'NEH (trained on UMN + Duke Srinivasan)': {
        2: ['NEH (trained on UMN', '+ Duke Srinivasan)'], 3: ['NEH (trained', 'on UMN + Duke', 'Srinivasan)'],
        6: ['NEH (trained', 'on UMN', '+ Duke', 'Srinivasan)']},
    'UMN + Duke Srinivasan (trained on NEH)': {
        2: ['UMN + Duke Srinivasan', '(trained on NEH)'], 3: ['UMN + Duke', 'Srinivasan', '(trained on NEH)'],
        6: ['UMN + Duke', 'Srinivasan', '(trained on', 'NEH)']},
    'Duke iAMD -- Layers (trained on AROI)': {6: ['Duke iAMD -- Layers', '(trained on AROI)']},
}


def label_piece(label, n_models, i):
    """Text of the dataset column in model row i of a block with n_models rows."""
    pieces = LABEL_LINES.get(label, {}).get(n_models, [label])
    return pieces[i] if i < len(pieces) else ''


DS_2D = [
    ('AMD_DME_3D', 'AMD-DME-3D'), ('BanglaOCT2025', 'BanglaOCT2025'), ('Duke_iAMD', 'Duke iAMD'),
    ('GAMMA', 'GAMMA'), ('GOALS', 'GOALS'), ('Harvard_Glaucoma', 'Harvard Glaucoma'),
    ('Kermany_correct', 'Kermany'), ('M3Ret', 'M3Ret'), ('MMC_AMD', 'MMC-AMD'), ('Noor_Eye_Hospital', 'NEH'),
    ('OAEFD_DME', 'OAEFD DME'), ('OAEFD_DR', 'OAEFD DR'), ('OCTA500', 'OCTA-500'), ('OCTAVE', 'OCTAVE'),
    ('OCTDL', 'OCTDL'), ('OCTID', 'OCTID'), ('OCT_C8', 'OCT C8'), ('OCT_MD_MS', 'OCT MD MS'),
    ('OIMHS', 'OIMHS'), ('OLIVES_baseline', 'OLIVES'), ('POAGD', 'POAGD'),
] + CROSS_LAST

CONFIG = {
    '2d': dict(kind='cls_ds', csv='2d_raw_results.csv', experiment='2d', metrics=CLS_METRICS,
               order=['RETFound', 'VisionFM', 'UrFound', 'OCTCube', 'MIRAGE', 'CLIFO'],
               display=CLS_2D_DISPLAY, ds_order=DS_2D, pairs=None, underline=True),
    '3d': dict(kind='cls_ds', csv='3d_raw_results.csv', experiment='3d', metrics=CLS_METRICS,
               rename=CLS_3D_RENAME, order=['OCTCube (60)', 'CLIFO (60)', 'CLIFO (all)'],
               display=CLS_3D_DISPLAY, ds_order=DS_3D,
               pairs=[('CLIFO (60)', 'OCTCube (60)'), ('CLIFO (all)', 'OCTCube (60)')], underline=False,
               plot_pairs=[('CLIFO (all)', 'CLIFO (60)'), ('CLIFO (60)', 'OCTCube (60)')], fig_width=4),  # first drawn lowest
    'chd': dict(kind='cls_avg', single=True, csv='chd_raw_results.csv', experiment='chd', metrics=CLS_METRICS,
                order=['OCTCube', 'MIRAGE', 'RETFound', 'UrFound', 'VisionFM', 'CLIFO'],
                display=CLS_2D_DISPLAY, pairs=None, underline=True),
    'seg_sota': dict(kind='seg_ds', csv='sota_raw_seg_results.csv', experiment='sota', metrics=SEG_METRICS,
                     order=['VisionFM', 'UrFound', 'OCTCube', 'RETFound', 'MIRAGE', 'CLIFO'],
                     pairs=None, underline=True),
    'retrieval': dict(kind='ret', csv='retrieval_raw_results.csv', experiment='retrieval',
                      order=['OCTCube', 'CLIFO'], display=CLS_2D_DISPLAY,
                      pairs=[('CLIFO', 'OCTCube')], underline=False),
}


# ============================================================
# Shared helpers
# ============================================================
def comparisons(means, pairs, higher=True, stars_on='first'):
    """List of (row, a, b): one-sided test of a better than b, with the stars on the `row` model.
    pairs=None means best vs second best (stars on the best). stars_on='second' puts the stars on
    b's row, so one model can be compared against several baselines without the pairs overwriting
    each other (each baseline row carries its own comparison)."""
    if pairs is None:
        ranked = means.dropna().sort_values(ascending=not higher).index
        return [(ranked[0], ranked[0], ranked[1])] if len(ranked) > 1 else []
    return [(a if stars_on == 'first' else b, a, b) for a, b in pairs if a in means.index and b in means.index]


def bests(means, underline, higher=True):
    """(best, second) for one column. second is None unless underline=True."""
    ranked = means.dropna().sort_values(ascending=not higher).index
    best = ranked[0] if len(ranked) else None
    second = ranked[1] if (underline and len(ranked) > 1) else None
    return best, second


SD_PAREN = True   # mean followed by the SD in parentheses (smaller); False: mean$\\pm$SD. Set by --plusminus.


def seg_table_metrics(cfg):
    return cfg['metrics'] + (['F1'] if DET_F1 else [])


def det_f1_note():
    return 'Det-F1 is the B-scan-level lesion detection F1 (lesion tasks only; the average covers lesion tasks only). ' if DET_F1 else ''


def mean_sd_label():
    return 'Mean (SD)' if SD_PAREN else 'Mean $\\pm$ SD'


def fmt_cell(mean_val, std_val, stars, is_best, is_second, colour=None):
    if pd.isna(mean_val):
        return '--'
    std_txt = f'{std_val:.3f}' if pd.notna(std_val) else '0.000'
    if SD_PAREN:   # stars stay on the mean; \\mdseries keeps the SD out of bold
        cell = f'{mean_val:.3f}{stars} {{\\scriptsize\\mdseries({std_txt})}}'
    else:
        cell = f'{mean_val:.3f}$\\pm$' + std_txt + stars
    if is_best:
        cell = f'\\textbf{{{cell}}}'
        return f'\\textcolor{{{colour}}}{{{cell}}}' if colour else cell
    if is_second:
        return f'\\underline{{{cell}}}'
    return cell


def name_cell(model, display):
    text, colour = display.get(model, (model, None))
    return f'\\textcolor{{{colour}}}{{{text}}}' if colour else text


def render_rows(label, order, display, metrics, per_metric, pvals, best_second, with_label=True):
    """Rows of one block (a dataset, or the Average). per_metric[m] is indexed by model with mean/std."""
    out = []
    models = [m for m in order if any(k in per_metric and m in per_metric[k].index for k in metrics)]
    for i, model in enumerate(models):
        if i % 2 == 0:
            out.append('\\rowcolor{gray!10}\\cellcolor{white}%')
        _, colour = display.get(model, (model, None))
        cells = [label_piece(label, len(models), i), name_cell(model, display)] if with_label else [name_cell(model, display)]
        for m in metrics:
            st = per_metric.get(m)
            if st is None or model not in st.index:
                cells.append('--')
                continue
            best, second = best_second[m]
            stars = get_sig_stars(pvals.get(m, {}).get(model))
            cells.append(fmt_cell(st.loc[model, 'mean'], st.loc[model, 'std'], stars,
                                  model == best, model == second, colour))
        out.append(' & '.join(cells) + ' \\\\')
    return out


def caption(title, cfg, scope):
    """scope: what the average or test runs over, e.g. 'across datasets'."""
    if cfg['pairs'] is None:
        comp = 'Stars mark the best model against the second best.'
    elif cfg.get('stars_on', 'first') == 'second':
        first = cfg['pairs'][0][0]
        comp = (f'Stars on each baseline mark the significance of {cfg["display"].get(first, (first,))[0]} '
                'vs. that baseline.')
    else:
        comp = 'Stars mark ' + '; '.join(
            f'{cfg["display"].get(m, (m,))[0]} vs {cfg["display"].get(r, (r,))[0]}' for m, r in cfg['pairs']) + '.'
    best_txt = 'Best results are shown in \\textbf{bold}' + (
        ', and second best, \\underline{underlined}' if cfg['underline'] else '') + '.'
    return (f'\\caption{{{title} {best_txt} {comp} '
            f'Significance: {paired_test_name()} {scope}. '
            '$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.}')


def save(lines, experiment, name):
    save_latex(lines, OUT_DIR / experiment / f'{experiment}_{name}.tex')


# ============================================================
# Classification
# ============================================================
def cls_ds_table(full_df, cfg):
    """Per-dataset table (paper layout) with an Average block."""
    metrics, order, display = cfg['metrics'], cfg['order'], cfg['display']
    present = set(full_df['Dataset'].unique())
    n = len(metrics)
    header = ' & '.join(f'\\textbf{{{m}}}' for m in metrics)
    title = '\\textbf{OCT diagnosis results.} Per-dataset performance comparison across models.'
    scope = 'across training runs within each dataset, and across datasets for the average'
    lines = ['{\\setlength{\\tabcolsep}{4pt}', '\\footnotesize', f'\\begin{{longtable}}{{{"l" * (2 + n)}}}']
    ph.longtable_header(lines, 2 + n, f'\\textbf{{Dataset}} & \\textbf{{Model}} & {header} \\\\',
                        caption(title, cfg, scope), f'tab:{cfg["experiment"]}_per_dataset')

    for ds, label in cfg['ds_order']:
        if ds not in present:
            continue
        dds = full_df[full_df['Dataset'] == ds]
        per_metric, pvals, bs = {}, {}, {}
        for m in metrics:
            st = dds.groupby('Model', observed=True)[m].agg(['mean', 'std'])
            per_metric[m] = st
            pm = comparisons(st['mean'], cfg['pairs'], stars_on=cfg.get('stars_on', 'first'))
            pvals[m] = {row: ph.compute_pvalue_per_dataset(dds, a, b, m) for row, a, b in pm}
            bs[m] = bests(st['mean'], cfg['underline'])
        lines += render_rows(label, order, display, metrics, per_metric, pvals, bs)
        lines.append('\\midrule')

    per_metric, pvals, bs = {}, {}, {}
    for m in metrics:
        per_ds = full_df.groupby(['Dataset', 'Model'], observed=True)[m].mean().reset_index()
        st = per_ds.groupby('Model', observed=True)[m].agg(['mean', 'std'])
        per_metric[m] = st
        pm = comparisons(st['mean'], cfg['pairs'], stars_on=cfg.get('stars_on', 'first'))
        pvals[m] = {row: ph.compute_pvalue_avg(full_df, a, b, m, False) for row, a, b in pm}
        bs[m] = bests(st['mean'], cfg['underline'])
    if lines[-1] == '\\midrule':
        lines.pop()
    lines.append('\\midrule')
    lines += render_rows('\\textit{Average}', order, display, metrics, per_metric, pvals, bs)
    lines += ['\\end{longtable}', '}']
    save(lines, cfg['experiment'], 'per_dataset')


def cls_avg_table(full_df, cfg):
    """Single table over datasets (mean and SD across datasets). For chd (single=True): across runs."""
    metrics, order, display = cfg['metrics'], cfg['order'], cfg['display']
    single = cfg.get('single', False)
    scope = 'across training runs' if single else 'across datasets'
    title = 'Performance comparison across models.'
    lines = ['\\footnotesize', f'\\begin{{longtable}}{{{"l" * (1 + len(metrics))}}}']
    header = ' & '.join(f'\\textbf{{{m}}}' for m in metrics)
    ph.longtable_header(lines, 1 + len(metrics), f'\\textbf{{Model}} & {header} \\\\',
                        caption(title, cfg, scope), f'tab:{cfg["experiment"]}')
    per_metric, pvals, bs = {}, {}, {}
    for m in metrics:
        if single:
            st = full_df.groupby('Model', observed=True)[m].agg(['mean', 'std'])
        else:
            per_ds = full_df.groupby(['Dataset', 'Model'], observed=True)[m].mean().reset_index()
            st = per_ds.groupby('Model', observed=True)[m].agg(['mean', 'std'])
        per_metric[m] = st
        pm = comparisons(st['mean'], cfg['pairs'], stars_on=cfg.get('stars_on', 'first'))
        if single:
            pvals[m] = {row: ph.compute_pvalue_per_dataset(full_df, a, b, m) for row, a, b in pm}
        else:
            pvals[m] = {row: ph.compute_pvalue_avg(full_df, a, b, m, False) for row, a, b in pm}
        bs[m] = bests(st['mean'], cfg['underline'])
    lines += render_rows(None, order, display, metrics, per_metric, pvals, bs, with_label=False)
    lines += ['\\end{longtable}', '\\normalsize']
    save(lines, cfg['experiment'], 'table')


# ============================================================
# Segmentation
# ============================================================
def add_detection_f1(df):
    """B-scan detection F1 per volume and class, from the slice counts in the CSV: a B-scan is a hit when the
    class is present in both prediction and reference. NaN where the class is absent from both everywhere, for
    layer tasks, and for rows without slice counts."""
    tp = df['n_valid_slices'] - df['FN_slices'] - df['FP_slices']
    denom = 2 * tp + df['FP_slices'] + df['FN_slices']
    df = df.copy()
    df['F1'] = (2 * tp / denom).where(denom > 0)
    df.loc[~df['Dataset'].str.endswith('_lesions'), 'F1'] = float('nan')
    return df


def seg_vol_sd(df, metric):
    """Mean and SD per model within one dataset. Mean: average over volumes within each seed, then over seeds
    (as in the headline numbers). SD: across volumes, with seeds averaged per volume."""
    per_seed = ph.seg_volume_level(df, metric)
    mean = per_seed.groupby(['Model', 'Seed'], observed=True)[metric].mean().groupby('Model', observed=True).mean()
    vol = per_seed.groupby(['Model', 'ID'], observed=True)[metric].mean()
    return pd.DataFrame({'mean': mean, 'std': vol.groupby('Model', observed=True).std()})


def seg_unit_stats(full_df, metric):
    """Mean and SD across dataset-tasks per model (volumes and seeds averaged within each dataset-task)."""
    per_ds = ph.seg_level_df(full_df, metric, 'dataset').groupby(['Dataset', 'Model'], observed=True)[metric].mean()
    return per_ds.groupby('Model', observed=True).agg(['mean', 'std'])


def seg_ds_table(full_df, cfg):
    """Per-dataset segmentation table (paper layout) with an Average block. Within a dataset: mean and SD
    across volumes (seeds averaged per volume), stars from the paired test over volumes. Average: mean and
    SD across dataset-tasks, stars from the paired test over dataset-tasks."""
    metrics, order, display = seg_table_metrics(cfg), cfg['order'], CLS_2D_DISPLAY
    n = len(metrics)
    header = ' & '.join(f'\\textbf{{{SEG_HEADER.get(m, m)}}}' for m in metrics)
    title = ('Per-dataset segmentation performance comparison across models. ' + mean_sd_label() + ' across volumes '
             'within each dataset (seeds averaged per volume), and across dataset-tasks for the average. '
             + det_f1_note().rstrip())
    scope = 'over volumes within each dataset, and over dataset-tasks for the average'
    lines = ['\\footnotesize', f'\\begin{{longtable}}{{{"l" * (2 + n)}}}']
    ph.longtable_header(lines, 2 + n, f'\\textbf{{Dataset}} & \\textbf{{Model}} & {header} \\\\',
                        caption(title, cfg, scope), f'tab:seg_{cfg["experiment"]}_per_dataset')
    for ds in sorted(full_df['Dataset'].unique(), key=lambda d: d.startswith('Duke_iAMD')):   # cross-dataset last
        dds = full_df[full_df['Dataset'] == ds]
        per_metric, pvals, bs = {}, {}, {}
        for m in metrics:
            higher = m in ('Dice', 'F1')
            vol = ph.seg_volume_level(dds, m, pool_seeds=True).dropna(subset=[m])
            st = seg_vol_sd(dds, m)
            per_metric[m] = st
            pm = comparisons(st['mean'], cfg['pairs'], higher)
            pvals[m] = {row: ph.seg_compute_pvalue_per_dataset(vol, a, b, m, higher) for row, a, b in pm}
            bs[m] = bests(st['mean'], cfg['underline'], higher)
        lines += render_rows(ph.paper_dataset_label(ds), order, display, metrics, per_metric, pvals, bs)
        lines.append('\\midrule')
    per_metric, pvals, bs = {}, {}, {}
    for m in metrics:
        higher = m in ('Dice', 'F1')
        st = seg_unit_stats(full_df, m)
        per_metric[m] = st
        level = ph.seg_level_df(full_df, m, 'dataset')
        pm = comparisons(st['mean'], cfg['pairs'], higher)
        pvals[m] = {row: ph.seg_compute_pvalue_avg(level, a, b, m) for row, a, b in pm}
        bs[m] = bests(st['mean'], cfg['underline'], higher)
    lines += render_rows('\\textit{Average}', order, display, metrics, per_metric, pvals, bs)
    lines += ['\\end{longtable}', '\\normalsize']
    save(lines, cfg['experiment'], 'per_dataset')


# ============================================================
# Retrieval (one value per dataset and model, no seeds)
# ============================================================
# Retrieval metrics shown, in this order (--top10 adds Top-10 at the end)
RET_METRICS = ['mAP', 'top_1_acc', 'top_5_acc']


def retrieval_table(cfg):
    df = pd.read_csv(ph.CSV_DIR / cfg['csv'])
    metrics = list(RET_METRICS)
    labels = {'top_1_acc': 'Top-1', 'top_5_acc': 'Top-5', 'top_10_acc': 'Top-10', 'mAP': 'mAP'}
    df[metrics] = df[metrics] / 100
    df = df.rename(columns={'model_name': 'Model', 'dataset_name': 'Dataset'})
    order, display = cfg['order'], cfg['display']
    n = len(metrics)
    header = ' & '.join(f'\\textbf{{{labels[m]}}}' for m in metrics)
    title = 'Per-dataset retrieval performance.'
    lines = ['{\\setlength{\\tabcolsep}{4pt}', '\\footnotesize', f'\\begin{{longtable}}{{{"l" * (2 + n)}}}']
    ph.longtable_header(lines, 2 + n, f'\\textbf{{Dataset}} & \\textbf{{Model}} & {header} \\\\',
                        caption(title, cfg, 'across datasets for the average (one value per dataset)'),
                        f'tab:{cfg["experiment"]}_per_dataset')

    present = set(df['Dataset'].unique())
    for ds, label in DS_3D:
        if ds not in present:
            continue
        dds = df[df['Dataset'] == ds].set_index('Model')
        best = {m: dds[m].idxmax() for m in metrics}
        for i, model in enumerate(order):
            if model not in dds.index:
                continue
            _, colour = display.get(model, (model, None))
            cells = [label_piece(label, len(order), i), name_cell(model, display)]
            for m in metrics:
                val = f'{dds.loc[model, m]:.3f}'
                if best[m] == model:
                    val = f'\\textbf{{{val}}}'
                    val = f'\\textcolor{{{colour}}}{{{val}}}' if colour else val
                cells.append(val)
            lines.append(' & '.join(cells) + ' \\\\')
        lines.append('\\midrule')

    per_metric, pvals, bs = {}, {}, {}
    for m in metrics:
        per_ds = df.groupby(['Dataset', 'Model'], observed=True)[m].mean().reset_index()
        st = per_ds.groupby('Model', observed=True)[m].agg(['mean', 'std'])
        per_metric[m] = st
        pm = comparisons(st['mean'], cfg['pairs'], stars_on=cfg.get('stars_on', 'first'))
        pvals[m] = {row: ph.compute_pvalue_avg(df, a, b, m, False) for row, a, b in pm}
        bs[m] = bests(st['mean'], cfg['underline'])
    lines += render_rows('\\textit{Average}', order, display, metrics, per_metric, pvals, bs)
    lines += ['\\end{longtable}', '}']
    save(lines, cfg['experiment'], 'per_dataset')


# ============================================================
# Plots: one point per unit (dataset, or dataset-task for segmentation), brackets = the table's
# comparisons, p-values from the same sign-flip tests.
# ============================================================
def save_fig(save_fn):
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().patch.set_alpha(0)
    plt.gca().set_facecolor('white')
    plt.savefig(save_fn, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {save_fn}')


def plot_box(plot_df, cfg, metric, ylabel, pvalue_fn, save_fn, higher=True, width=None, single_extra=0.1, wrap=True):
    """plot_df: one row per unit with columns Model and metric. pvalue_fn(a, b) gives the one-sided p for a > b.
    width: figure width in inches (default one per model). single_extra: headroom above one bracket.
    wrap: wrap x labels (as in the original retrieval plot; the original segmentation plot did not)."""
    order = [m for m in cfg['order'] if m in set(plot_df['Model'])]
    means = plot_df.groupby('Model', observed=True)[metric].mean()
    pairs = cfg.get('plot_pairs', cfg['pairs'])  # plots may compare different pairs than the table
    if pairs is None:
        best, second = bests(means, True, higher)
        brackets = [(best, second)] if second is not None else []
    else:
        brackets = [(a, b) for a, b in pairs if a in means.index and b in means.index]

    plt.figure(figsize=(width if width is not None else len(order), 5))
    sns.set_style('whitegrid')
    ax = sns.boxplot(data=plot_df, hue='Model', x='Model', y=metric, palette=ph.get_palette(order), order=order,
                     showmeans=False, legend=False, showfliers=False)
    sns.stripplot(data=plot_df, x='Model', y=metric, order=order, color='black', alpha=0.3, jitter=True, size=4)
    if wrap:
        ax.set_xticklabels([ph.wrap_for_plot(m) for m in order])
    for i, model in enumerate(order):
        ph.draw_mean_annotation(ax, i, means[model], value_fmt='{:.3f}', mean_style='diamond')

    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    for k, (a, b) in enumerate(brackets):
        p = pvalue_fn(a, b)
        ia, ib = order.index(a), order.index(b)
        h = y_max + span * 0.12 * k
        ax.plot([ia, ib], [h, h], 'k-', linewidth=2)
        ax.plot([ia, ia], [h - span * 0.02, h], 'k-', linewidth=2)
        ax.plot([ib, ib], [h - span * 0.02, h], 'k-', linewidth=2)
        label = ph.sig_str_from_pvalue(p) if p is not None else 'n/a'
        ax.text((ia + ib) / 2, h + span * 0.02, label, ha='center', va='bottom', fontsize=11, fontweight='bold')
    if len(brackets) > 1:
        ax.set_ylim(y_min, y_max + span * (0.1 + 0.12 * len(brackets)))
    else:
        ax.set_ylim(y_min, y_max + span * single_extra)

    plt.ylabel(ylabel)
    plt.xlabel('Model')
    plt.tight_layout()
    save_fig(save_fn)


def make_plots(cfg, full_df):
    kind, experiment = cfg['kind'], cfg['experiment']
    if kind in ('cls_ds', 'cls_avg'):
        single = cfg.get('single', False)
        for m in cfg['metrics']:
            # Original classification boxplot (sizes, headroom, styling); brackets are the plot comparisons.
            brackets = cfg.get('plot_pairs', cfg['pairs'])
            ph.cls_plot_boxplot(full_df, cfg['order'], experiment, m, single, cfg.get('fig_width'), False,
                                'diamond', brackets=brackets)
            ph.cls_plot_barplot(full_df, cfg['order'], experiment, m)
    elif kind == 'seg_ds':
        for m in cfg['metrics']:
            higher = m in ('Dice', 'F1')
            level = ph.seg_level_df(full_df, m, 'dataset')
            plot_df = level.groupby(['Dataset', 'Model'], observed=True)[m].mean().reset_index()
            pfn = lambda a, b, m=m, higher=higher, level=level: ph.seg_compute_pvalue_avg(level, a, b, m)
            plot_box(plot_df, cfg, m, m, pfn, OUT_DIR / f'seg_{experiment}' / f'boxplot_{m}.{ph.FIG_FORMAT}',
                     higher=higher, wrap=False)


def retrieval_plots(cfg):
    raw = pd.read_csv(ph.CSV_DIR / cfg['csv'])
    metrics = list(RET_METRICS)
    labels = {'top_1_acc': 'Top-1', 'top_5_acc': 'Top-5', 'top_10_acc': 'Top-10', 'mAP': 'mAP'}
    raw[metrics] = raw[metrics] / 100
    plot_df = raw.rename(columns={'model_name': 'Model', 'dataset_name': 'Dataset'})
    for m in metrics:
        pfn = lambda a, b, m=m: ph.compute_pvalue_avg(plot_df, a, b, m, False)
        plot_box(plot_df, cfg, m, labels[m], pfn, OUT_DIR / 'retrieval' / f'boxplot_{m}.{ph.FIG_FORMAT}',
                 width=max(2.4, len(cfg['order']) * 1.2))
        ph.ret_plot_barplot(raw, cfg['order'], m, labels[m])


# ============================================================
# Driver
# ============================================================
def build(name):
    cfg = CONFIG[name]
    if cfg['kind'] == 'ret':
        retrieval_table(cfg)
        retrieval_plots(cfg)
        return
    full_df = pd.read_csv(ph.CSV_DIR / cfg['csv'])
    if cfg.get('rename'):
        full_df['Model'] = full_df['Model'].replace(cfg['rename'])
    # Best/second are chosen among the models in the table only (the CSVs may hold extra models).
    full_df = full_df[full_df['Model'].isin(cfg['order'])]
    if cfg['kind'] == 'seg_ds':
        full_df = add_detection_f1(full_df)
    if cfg['kind'] == 'cls_ds':
        cls_ds_table(full_df, cfg)
    elif cfg['kind'] == 'cls_avg':
        cls_avg_table(full_df, cfg)
    elif cfg['kind'] == 'seg_ds':
        seg_ds_table(full_df, cfg)
    make_plots(cfg, full_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', default='all', choices=list(CONFIG) + ['all'])
    parser.add_argument('--detf1', action='store_true', help='add the Det-F1 column to the segmentation tables')
    parser.add_argument('--plusminus', action='store_true', help='mean$\\pm$SD instead of mean (SD)')
    parser.add_argument('--top10', action='store_true', help='add Top-10 to the retrieval table and plots')
    args = parser.parse_args()
    if args.top10:
        RET_METRICS.append('top_10_acc')
    SD_PAREN = not args.plusminus
    DET_F1 = args.detf1
    ph.BASE_PATH = OUT_DIR  # plot_helpers writes its barplots and seg per-dataset table under BASE_PATH
    for name in (list(CONFIG) if args.kind == 'all' else [args.kind]):
        print(f'\n##### {name}')
        build(name)
