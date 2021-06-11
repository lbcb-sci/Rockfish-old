import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_distribution(data, modification, bins=15):
    """ Distribution of signal points ratio """
    plt.hist(data, bins=bins, color='green' if modification == 'nomod' else 'red')
    plt.title(f'Distribution of Signal Points Ratio ({modification})')
    plt.xlabel('Signal Points Ratio')
    plt.ylabel('Count')
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.arange(0, 14001, 2000))
    plt.savefig(f'../plots/ratio_D_{modification}.png')


def plot_final_distribution(data, modification, bins=10):
    """ Distribution of signal points ratio, taking the number of deletions into consideration """
    plt.hist(data.values(), bins, histtype='bar', label=list(data_mod.keys()))
    plt.legend(title="Number of Deletions")
    plt.title(f'Distribution of Signal Points Ratio ({modification})')
    plt.xlabel('Signal Points Ratio')
    plt.ylabel('Count')
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.arange(0, 12001, 2000))
    plt.savefig(f'../plots/ratio_D_{modification}_final.png')


if __name__ == '__main__':
    modification = 'mod'  # 'mod' or 'nomod'

    # ratios = []
    # plot_distribution(ratios, modification)

    data_mod = pd.read_pickle('../plots/mod-D.pkl')
    data_nomod = pd.read_pickle('../plots/nomod-D.pkl')
    data = data_mod if modification == 'mod' else data_nomod
    plot_final_distribution(data, modification)
