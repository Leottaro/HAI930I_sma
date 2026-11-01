"""Plots of the LLM matches, from a results CSV (one row per match).

    python visualizations.py                         # results_matches.csv → PNGs in the current folder
    python visualizations.py sorted_history          # every condition made by sort_history.py
    python visualizations.py path/to/matches.csv ... # given CSVs → PNGs in a plots/ folder next to each

With arguments, a folder means its matches.csv, or else every */matches.csv
inside it; the plots of each CSV go to a plots/ folder next to it, and carry
the name of that folder as a subtitle.

Each CSV gives 15 plots: plots 1-6 (overview) and plots 7-15 (strategy and
parameter analysis, see make_extra_plots). The extra plots need seaborn >= 0.12
and are skipped individually when the columns they need are missing or constant.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Set global smaller font sizes and compact layout
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.autolayout': True
})

# Experiment knobs worth sweeping in the extra plots (only used when they actually vary).
PARAMS = ['temperature', 'mapek_every', 'window', 'poor_threshold']
CI = ('ci', 95)
# MAPE-K settings only mean something on adaptive runs.
MAPEK_KNOBS = {'mapek_every', 'window', 'poor_threshold'}
# Value of the `trend` column that marks a comeback (case-insensitive substring).
COMEBACK_TREND = 'comeback'

def _has(df, *cols):
    return all(c in df.columns for c in cols)

def _varies(df, col):
    return col in df.columns and pd.to_numeric(df[col], errors='coerce').nunique() > 1
 
def _comebacks(df):
    """Flag the comebacks: (0/1 Series, how it is defined, is_proxy), or (None, None, False).
 
    The CSV has no explicit comeback column, so:
    1. if the `trend` column has a value containing COMEBACK_TREND, that is the flag;
    2. otherwise a proxy from the other columns: MAPE-K switched strategy at least
       once and the LLM still won the match (llm_score > opp_score).
    Change this function if comebacks are defined differently in your data.
    """
    if 'trend' in df.columns:
        flag = df['trend'].astype(str).str.lower().str.contains(COMEBACK_TREND.lower(), regex=False)
        if flag.any():
            return flag.astype(float), f"comeback = trend '{COMEBACK_TREND}'", False
    if _has(df, 'n_switches', 'llm_score', 'opp_score'):
        flag = (pd.to_numeric(df['n_switches'], errors='coerce') > 0) & (df['llm_score'] > df['opp_score'])
        return flag.astype(float), 'comeback (proxy) = win after >= 1 MAPE-K switch', True
    return None, None, False
 
def _levels(d, p):
    """Group labels (as str) of column p, and the order to display them in."""
    s = d[p]
    if p == 'adaptive':
        return s.map({False: 'fixed', True: 'adaptive'}), ['fixed', 'adaptive']
    if pd.api.types.is_numeric_dtype(s):
        if s.nunique() > 8:  # continuous: 4 quantile bins
            s = pd.qcut(s, 4, duplicates='drop')
            return s.astype(str), [str(c) for c in s.cat.categories]
        return s.map(lambda v: f'{v:g}'), [f'{v:g}' for v in sorted(s.dropna().unique())]
    rate = d.groupby(s)['comeback'].mean().sort_values(ascending=False)
    return s.astype(str), [str(k) for k in rate.index]
 

def make_plots(df: pd.DataFrame, out: Path, label: str | None = None,
                     min_n: int = 3) -> None:
    """Write the 9 extra analysis plots (7-15) of `df` to the folder `out`.

    Strategy analysis: 7 (share of final strategies), 8 (strategy ranking),
    9 (strategy x opponent matrix). Parameter effects: 10 (adaptive vs fixed),
    11 (parameter sweeps), 12 (parameter interactions), 15 (parameter importance).
    Behaviour: 13 (switches vs score), 14 (score margin).

    min_n: cells of the strategy x opponent matrix with fewer runs are blanked.
    """
    
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df_adaptative = df[df['adaptive'] == True]

    # ---- derived columns -------------------------------------------------
    if 'adaptive' in df.columns:
        df['adaptive'] = df['adaptive'].astype(str).str.lower().isin(['true', '1', '1.0', 'yes'])
    if _has(df, 'llm_score', 'opp_score'):
        df['win'] = (df['llm_score'] > df['opp_score']).astype(float)
    if _has(df, 'llm_avg', 'opp_avg'):
        df['margin'] = df['llm_avg'] - df['opp_avg']
    if _has(df, 'llm_avg', 'opponent'):
        # score relative to what that opponent gives on average: removes the
        # "some opponents are just easier" effect from the comparisons below
        df['llm_avg_c'] = df['llm_avg'] - df.groupby('opponent')['llm_avg'].transform('mean')

    def save(fig, name, note=None):
        sub = ' | '.join(s for s in (label, note) if s)
        if sub:
            fig.suptitle(sub, fontsize=6, color='gray')
        fig.tight_layout()
        fig.savefig(out / name, dpi=300)
        plt.close(fig)

    # -------------------------------------------------------------
    # Plot 1: LLM Average Score per Opponent Type (Bar Plot)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    if 'opponent' in df.columns and 'llm_avg' in df.columns:
        sns.barplot(data=df, x='opponent', y='llm_avg', ax=ax, palette='mako', ci=None)
        ax.set_title('Avg Payoff vs. Opponent Strategy', fontweight='bold')
        ax.set_xlabel('Opponent')
        ax.set_ylabel('LLM Avg Score / Round')
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
    save(fig, 'plot_1_avg_payoff.png')

    # -------------------------------------------------------------
    # Plot 2: Match Outcome Breakdown (Pie Chart)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    if 'outcome' in df.columns:
        outcome_counts = df['outcome'].value_counts()
        ax.pie(outcome_counts, labels=outcome_counts.index, autopct='%1.1f%%',
                colors=sns.color_palette('Set2'), startangle=140, wedgeprops={'edgecolor': 'white'})
        ax.set_title('Match Outcome Proportions', fontweight='bold')
    save(fig, 'plot_2_outcomes.png')

    # -------------------------------------------------------------
    # Plot 3: LLM Score vs. Opponent Score Parity Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    if 'llm_score' in df.columns and 'opp_score' in df.columns:
        sns.scatterplot(data=df, x='opp_score', y='llm_score', hue='outcome', style='opponent', s=80, ax=ax, palette='deep')
        max_val = max(df['llm_score'].max(), df['opp_score'].max()) if len(df) > 0 else 50
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Parity (y=x)')
        ax.set_title('Score Parity: LLM vs. Opponent', fontweight='bold')
        ax.set_xlabel('Opponent Total Score')
        ax.set_ylabel('LLM Total Score')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    save(fig, 'plot_3_parity.png')

    # -------------------------------------------------------------
    # Plot 4: MAPE-K Adaptation Activity (Switches by Opponent)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    if 'opponent' in df.columns and 'n_switches' in df.columns:
        sns.boxplot(data=df, x='opponent', y='n_switches', ax=ax, palette='coolwarm')
        sns.stripplot(data=df, x='opponent', y='n_switches', ax=ax, color='black', alpha=0.6, jitter=0.2)
        ax.set_title('MAPE-K Adaptations by Opponent', fontweight='bold')
        ax.set_xlabel('Opponent')
        ax.set_ylabel('Number of Strategy Switches')
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
    save(fig, 'plot_4_switches.png')

    # -------------------------------------------------------------
    # Plot 5: Initial Strategy vs Final Strategy Heatmap
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    if 'initial_strategy' in df_adaptative.columns and 'final_strategy' in df_adaptative.columns:
        pivot_strat = pd.crosstab(df_adaptative['initial_strategy'], df_adaptative['final_strategy'])
        sns.heatmap(pivot_strat, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax, annot_kws={'size': 8})
        ax.set_title('Strategy Evolution: Initial -> Final', fontweight='bold')
        ax.set_xlabel('Final Strategy')
        ax.set_ylabel('Initial Strategy')
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
    save(fig, 'plot_5_evolution.png')

    # -------------------------------------------------------------
    # Plot 6: Tournament Ranks / Distribution across Models or Temperatures
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    if 'llm_rank' in df.columns and 'model' in df.columns and df['llm_rank'].notna().any():
        sns.boxplot(data=df, x='model', y='llm_rank', hue='temperature', ax=ax, palette='Set3')
        ax.set_title('Tournament Rank Distribution', fontweight='bold')
        ax.set_xlabel('LLM Model')
        ax.set_ylabel('Tournament Rank (Lower = Better)')
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'Rank data will appear here once multi-player tournaments are run.',
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=8, color='gray')
        ax.set_title('Tournament Rank Distribution', fontweight='bold')
    save(fig, 'plot_6_rank.png')

    # -------------------------------------------------------------
    # Plot 7: How often is each strategy the FINAL one? (per opponent)
    # -------------------------------------------------------------
    if _has(df, 'opponent', 'final_strategy'):
        share = pd.crosstab(df['opponent'], df['final_strategy'], normalize='index')
        overall = df['final_strategy'].value_counts(normalize=True).reindex(share.columns).fillna(0)
        share = pd.concat([overall.to_frame('ALL').T, share])
        share = share[overall.sort_values(ascending=False).index]
        fig, ax = plt.subplots(figsize=(max(5, 0.9 * share.shape[1] + 2), 4))
        sns.heatmap(share, annot=True, fmt='.0%', cmap='mako_r', vmin=0, vmax=1,
                    cbar=False, ax=ax, annot_kws={'size': 7})
        ax.axhline(1, color='white', lw=3)  # separates the ALL row
        ax.set_title('Share of runs ending on each strategy', fontweight='bold')
        ax.set_xlabel('Final strategy')
        ax.set_ylabel('Opponent')
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
        save(fig, 'plot_7_final_strategy_share.png')

    # -------------------------------------------------------------
    # Plot 8: Strategy ranking by final strategy (score + win rate, 95% CI)
    # -------------------------------------------------------------
    if _has(df, 'final_strategy', 'llm_avg'):
        g = df.groupby('final_strategy')
        order = list(g['llm_avg'].mean().sort_values(ascending=False).index)
        n = g.size()
        has_win = 'win' in df.columns
        fig, axes = plt.subplots(1, 2 if has_win else 1, figsize=(9 if has_win else 5, 4),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        sns.barplot(data=df, y='final_strategy', x='llm_avg', order=order, ax=axes[0],
                    color='steelblue', errorbar=CI)
        axes[0].set_title('Avg score / round', fontweight='bold')
        axes[0].set_xlabel('LLM avg score')
        axes[0].set_ylabel('Final strategy')
        axes[0].set_yticks(range(len(order)))
        axes[0].set_yticklabels([f'{s} (n={n[s]})' for s in order], fontsize=7)
        if has_win:
            sns.barplot(data=df, y='final_strategy', x='win', order=order, ax=axes[1],
                        color='indianred', errorbar=CI)
            axes[1].axvline(0.5, color='k', ls='--', alpha=0.5)
            axes[1].set_title('Win rate (LLM score > opp)', fontweight='bold')
            axes[1].set_xlabel('Win rate')
            axes[1].set_ylabel('')
        save(fig, 'plot_8_strategy_ranking.png')

    # -------------------------------------------------------------
    # Plot 9: Best-response matrix: final strategy x opponent -> avg score
    # -------------------------------------------------------------
    if _has(df, 'final_strategy', 'opponent', 'llm_avg'):
        piv = df.pivot_table(index='final_strategy', columns='opponent',
                             values='llm_avg', aggfunc='mean')
        cnt = df.pivot_table(index='final_strategy', columns='opponent',
                             values='llm_avg', aggfunc='count')
        piv = piv.where(cnt >= min_n)
        fig, ax = plt.subplots(figsize=(max(5, 0.9 * piv.shape[1] + 2), max(3.5, 0.5 * piv.shape[0] + 1.5)))
        sns.heatmap(piv, annot=True, fmt='.2f', cmap='viridis', ax=ax,
                    annot_kws={'size': 7}, cbar_kws={'label': 'LLM avg score'})
        for j, col in enumerate(piv.columns):  # red box = best strategy per opponent
            if piv[col].notna().any():
                i = piv.index.get_loc(piv[col].idxmax())
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2))
        ax.set_title(f'Final strategy vs opponent (cells with n<{min_n} blank; red = best)',
                     fontweight='bold', fontsize=9)
        ax.set_xlabel('Opponent')
        ax.set_ylabel('Final strategy')
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
        save(fig, 'plot_9_strategy_vs_opponent.png')

    # -------------------------------------------------------------
    # Plot 10: Does adapting help? adaptive vs fixed, per opponent
    # -------------------------------------------------------------
    if _has(df, 'adaptive', 'opponent', 'llm_avg') and df['adaptive'].nunique() > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df, x='opponent', y='llm_avg', hue='adaptive', ax=ax,
                    errorbar=CI, palette={True: 'green', False: 'red'})
        ax.set_title('Adaptive (MAPE-K) vs fixed strategy', fontweight='bold')
        ax.set_xlabel('Opponent')
        ax.set_ylabel('LLM Avg Score / Round')
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
        save(fig, 'plot_10_adaptive_vs_fixed.png')

    # -------------------------------------------------------------
    # Plot 11: One-at-a-time parameter sweeps (score vs each knob)
    # -------------------------------------------------------------
    params = [p for p in PARAMS if _varies(df, p)]
    if params and 'llm_avg' in df.columns:
        fig, axes = plt.subplots(1, len(params), figsize=(4 * len(params), 3.6),
                                 squeeze=False, sharey=True)
        hue = 'opponent' if 'opponent' in df.columns else None
        for k, (ax, p) in enumerate(zip(axes[0], params)):
            sns.lineplot(data=df, x=p, y='llm_avg', hue=hue, marker='o', errorbar=CI, ax=ax,
                         legend='auto' if k == len(params) - 1 else False)
            vals = sorted(pd.to_numeric(df[p], errors='coerce').dropna().unique())
            if len(vals) <= 8:
                ax.set_xticks(vals)
            ax.set_title(f'Score vs {p}', fontweight='bold', fontsize=9)
            ax.set_ylabel('LLM Avg Score / Round' if k == 0 else '')
        if hue and axes[0][-1].get_legend():
            axes[0][-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
        save(fig, 'plot_11_param_sweeps.png')

    # -------------------------------------------------------------
    # Plot 12: Parameter interactions (2D heatmaps of mean score)
    # -------------------------------------------------------------
    cand = [('mapek_every', 'window'), ('mapek_every', 'temperature')]
    pairs = [(a, b) for a, b in cand if _varies(df, a) and _varies(df, b)]
    if pairs and 'llm_avg' in df.columns:
        fig, axes = plt.subplots(1, len(pairs), figsize=(4.5 * len(pairs), 3.8), squeeze=False)
        for ax, (a, b) in zip(axes[0], pairs):
            piv = df.pivot_table(index=b, columns=a, values='llm_avg', aggfunc='mean')
            sns.heatmap(piv, annot=True, fmt='.2f', cmap='viridis', cbar=False, ax=ax,
                        annot_kws={'size': 7})
            ax.invert_yaxis()
            ax.set_title(f'{b} x {a}', fontweight='bold', fontsize=9)
        save(fig, 'plot_12_param_interactions.png')

    # -------------------------------------------------------------
    # Plot 13: Is switching more often better or just thrashing?
    # -------------------------------------------------------------
    if _has(df, 'n_switches', 'llm_avg'):
        ycol = 'llm_avg_c' if 'llm_avg_c' in df_adaptative.columns else 'llm_avg'
        d = df_adaptative.dropna(subset=['n_switches', ycol])
        if len(d) > 2 and d['n_switches'].nunique() > 1:
            rho = d['n_switches'].corr(d[ycol], method='spearman')
            fig, ax = plt.subplots(figsize=(5.5, 4))
            if d['n_switches'].nunique() <= 15:  # few distinct counts: show them as columns
                sns.stripplot(data=d, x='n_switches', y=ycol, color='steelblue', alpha=0.5,
                              jitter=0.2, ax=ax)
            else:
                sns.scatterplot(data=d, x='n_switches', y=ycol, alpha=0.5, ax=ax)
                sns.regplot(data=d, x='n_switches', y=ycol, scatter=False, color='k', ax=ax,
                            line_kws={'lw': 1.5})
            ax.axhline(0, color='gray', ls='--', alpha=0.5)
            ax.set_title(f'Switches vs score (Spearman rho = {rho:.2f})', fontweight='bold')
            ax.set_xlabel('Number of strategy switches')
            ax.set_ylabel('LLM score (centred per opponent)' if ycol == 'llm_avg_c' else 'LLM Avg Score')
            save(fig, 'plot_13_switches_vs_score.png')

    # -------------------------------------------------------------
    # Plot 14: Score margin distribution per opponent (LLM - opponent)
    # -------------------------------------------------------------
    if _has(df, 'margin', 'opponent'):
        fig, ax = plt.subplots(figsize=(5.5, 4))
        sns.violinplot(data=df, x='opponent', y='margin', inner='point', cut=0,
                       color='lightsteelblue', ax=ax)
        ax.axhline(0, color='k', ls='--', alpha=0.5)
        ax.set_title('Margin per round (LLM avg - opponent avg)', fontweight='bold')
        ax.set_xlabel('Opponent')
        ax.set_ylabel('Margin')
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
        save(fig, 'plot_14_margin.png')

    # -------------------------------------------------------------
    # Plot 15: Which knob matters most? (Spearman vs centred score)
    # -------------------------------------------------------------
    if 'llm_avg_c' in df.columns:
        num = pd.DataFrame(index=df.index)
        for c in PARAMS + ['length', 'n_players', 'adaptive']:
            if c in df.columns:
                num[c] = pd.to_numeric(df[c].astype(float) if c == 'adaptive' else df[c],
                                       errors='coerce')
        num = num.loc[:, num.nunique() > 1]
        if num.shape[1]:
            rho = num.apply(lambda s: s.corr(df['llm_avg_c'], method='spearman'))
            rho = rho.reindex(rho.abs().sort_values().index)
            fig, ax = plt.subplots(figsize=(5, 0.5 * len(rho) + 1.5))
            ax.barh(rho.index, rho.values, color=np.where(rho.values >= 0, 'seagreen', 'indianred'))
            ax.axvline(0, color='k', lw=0.8)
            ax.set_title('Parameter vs score (Spearman rho)', fontweight='bold')
            ax.set_xlabel('rho with LLM score centred per opponent')
            save(fig, 'plot_15_param_importance.png')
            
    # -------------------------------------------------------------
    # Plot 16: What drives comebacks? Comeback rate vs each parameter
    # -------------------------------------------------------------
    flag, how, proxy = _comebacks(df_adaptative)
    if flag is not None:
        df_adaptative['comeback'] = flag
        # a proxy comeback needs a MAPE-K switch, so fixed runs can never count: leave them out
        base = df_adaptative[df_adaptative['adaptive']] if proxy and 'adaptive' in df_adaptative.columns else df_adaptative
        panels = []
        for p in ['opponent', 'initial_strategy', 'final_strategy']:
            if p not in base.columns or (p == 'adaptive' and proxy):
                continue
            d = base[base['adaptive']] if p in MAPEK_KNOBS and 'adaptive' in base.columns else base
            if d[p].nunique() > 1:
                panels.append((p, d))
        if panels and len(base):
            ncols = min(4, len(panels))
            nrows = -(-len(panels) // ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows),
                                     squeeze=False, sharey=True)
            overall = base['comeback'].mean()
            drawn = 0
            for k, (ax, (p, d)) in enumerate(zip(axes.flat, panels)):
                lab, order = _levels(d, p)
                d = d.assign(level=lab)
                n = d['level'].value_counts()
                order = [o for o in order if n.get(o, 0) >= min_n]
                if len(order) < 2:
                    ax.set_visible(False)
                    continue
                drawn += 1
                sns.barplot(data=d, x='level', y='comeback', order=order, ax=ax,
                            color='teal', errorbar=CI, seed=0)
                ax.axhline(overall, color='k', ls='--', lw=0.8, alpha=0.6)
                ax.set_title(f'vs {p}' + (' (adaptive runs)' if p in MAPEK_KNOBS else ''),
                             fontweight='bold', fontsize=8)
                ax.set_xlabel('')
                ax.set_ylabel('Comeback rate' if k % ncols == 0 else '')
                ax.yaxis.set_major_formatter(PercentFormatter(1.0))
                ax.set_xticks(range(len(order)))
                if max(len(o) for o in order) > 6:  # long names: one slanted line each
                    ax.set_xticklabels([f'{o} (n={n[o]})' for o in order], rotation=30,
                                       ha='right', rotation_mode='anchor', fontsize=6)
                else:
                    ax.set_xticklabels([f'{o}\n(n={n[o]})' for o in order])
            for ax in axes.flat[len(panels):]:
                ax.set_visible(False)
            if drawn:
                save(fig, 'plot_16_comeback_drivers.png',
                     note=f'{how}; dashed = overall rate {overall:.0%}; bars = 95% CI; n = matches')
            else:
                plt.close(fig)


def csv_files(args: list[str]) -> list[Path]:
    """The CSVs named on the command line (a folder = its matches.csv, or every */matches.csv)."""
    files = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            files += [p / 'matches.csv'] if (p / 'matches.csv').exists() else sorted(p.glob('*/matches.csv'))
        elif p.exists():
            files.append(p)
        else:
            print(f"  ! {p} not found")
    return files


def main(args: list[str]) -> None:
    if not args:
        make_plots(pd.read_csv('results_matches.csv'), Path('.'))
        print("All individual plots generated successfully with smaller fonts!")
        return
    files = csv_files(args)
    for f in files:
        df = pd.read_csv(f)
        if df.empty:
            print(f"  ! {f}: no match, skipped")
            continue
        make_plots(df, f.parent / 'plots', label=f.parent.name)
        print(f"  {f.parent / 'plots'}/  ({len(df)} matches)")
    print(f"{len(files)} CSV file(s) plotted.")


if __name__ == '__main__':
    main(sys.argv[1:])