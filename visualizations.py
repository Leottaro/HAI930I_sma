"""Plots of the LLM matches, from a results CSV (one row per match).

    python visualizations.py                         # results_matches.csv → PNGs in the current folder
    python visualizations.py sorted_history          # every condition made by sort_history.py
    python visualizations.py path/to/matches.csv ... # given CSVs → PNGs in a plots/ folder next to each

With arguments, a folder means its matches.csv, or else every */matches.csv
inside it; the plots of each CSV go to a plots/ folder next to it, and carry
the name of that folder as a subtitle.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


def make_plots(df: pd.DataFrame, out: Path, label: str | None = None) -> None:
    """Write the 6 plots of `df` to the folder `out` (subtitle: `label`)."""
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        if label:
            fig.suptitle(label, fontsize=6, color='gray')
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
    if 'initial_strategy' in df.columns and 'final_strategy' in df.columns:
        pivot_strat = pd.crosstab(df['initial_strategy'], df['final_strategy'])
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
