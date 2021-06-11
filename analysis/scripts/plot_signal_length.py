import os
from ont_fast5_api.fast5_interface import get_fast5_file
import numpy as np
import matplotlib.pyplot as plt


def get_signal_length(fast5_filepath):
    """
    :param fast5_filepath: can be a single- or multi-read file
    :return: raw signal length
    """
    with get_fast5_file(fast5_filepath, mode="r") as f5:
        for read in f5.get_reads():
            raw_data = read.get_raw_data()
            # print(read.read_id, raw_data, len(raw_data))

    return len(raw_data)


def plot_signal_length(dir_in, bins=50):
    """ Distribution of raw signal length """
    data = []
    for file in os.listdir(dir_in):
        data.append(get_signal_length(os.path.join(dir_in, file)))

    plt.hist(data, bins=bins, color='green' if modification == 'nomod' else 'red')
    plt.title(f'Distribution of raw signal length ({modification})')
    plt.xlabel('raw signal length')
    plt.ylabel('count')
    plt.xticks(np.arange(0, 600_001, 100_000))  # Change if needed
    plt.yticks(np.arange(0, 301, 50))  # Change if needed
    plt.savefig(f'../plots/signal_length_{modification}.png')


if __name__ == '__main__':
    modification = 'nomod'  # 'mod' or 'nomod'
    dir_in = f'/home/sdeur/tmp-guppy/{modification}/workspace'

    plot_signal_length(dir_in)
