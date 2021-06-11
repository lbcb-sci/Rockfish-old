import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_distribution(bins=15):
    """ Distribution of signal points """
    # Data for matches
    matches_mod = pd.read_pickle('../plots/matches-mod.pkl')
    matches_nomod = pd.read_pickle('../plots/matches-nomod.pkl')
    matches_mod_avg = round(sum(matches_mod) / len(matches_mod), 2)
    matches_nomod_avg = round(sum(matches_nomod) / len(matches_nomod), 2)
    matches_mod = list(filter(lambda x: x < 100, matches_mod))
    matches_nomod = list(filter(lambda x: x < 100, matches_nomod))

    # Data for deletions
    deletions_mod = pd.read_pickle('../plots/deletions-mod.pkl')
    deletions_nomod = pd.read_pickle('../plots/deletions-nomod.pkl')
    deletions_mod_avg = round(sum(deletions_mod) / len(deletions_mod), 2)
    deletions_nomod_avg = round(sum(deletions_nomod) / len(deletions_nomod), 2)
    deletions_mod = list(filter(lambda x: x < 100, deletions_mod))
    deletions_nomod = list(filter(lambda x: x < 100, deletions_nomod))

    # Plot distribution of signal points for matches
    plt.hist([matches_mod, matches_nomod], bins=bins, color=['red', 'green'],
             label=[f'mod (avg: {matches_mod_avg})', f'nomod (avg: {matches_nomod_avg})'])
    plt.legend()
    plt.title(f'Distribution of Signal Points Around Matches')
    plt.xlabel('Number of Signal Points')
    plt.ylabel('Count')
    plt.xticks(np.arange(0, 101, 10))
    plt.savefig(f'../plots/signal_points_M.png')
    plt.clf()

    # Plot distribution of signal points for deltions
    plt.hist([deletions_mod, deletions_nomod], bins=bins, color=['red', 'green'],
             label=[f'mod (avg: {deletions_mod_avg})', f'nomod (avg: {deletions_nomod_avg})'])
    plt.legend()
    plt.title(f'Distribution of Signal Points Around Deletions')
    plt.xlabel('Number of Signal Points')
    plt.ylabel('Count')
    plt.xticks(np.arange(0, 101, 10))
    plt.savefig(f'../plots/signal_points_D.png')


if __name__ == '__main__':
    plot_distribution()
