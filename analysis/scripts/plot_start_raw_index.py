import os
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file
import matplotlib.pyplot as plt
import statistics


def get_raw_signal_block(fast5_filename, read_id, start_seq_position, end_seq_position):
    raw_data = _get_raw_data(fast5_filename, read_id)
    (move_table, first_sample_template, block_stride) = _get_fast5_data(fast5_filename, read_id)
    seq_to_move_map = _create_map_from_basecall_to_signal(move_table)
    _, start_raw_index = _get_raw_signal_block(raw_data, seq_to_move_map, first_sample_template, block_stride,
                                               start_seq_position, end_seq_position)
    return list(raw_data), start_raw_index


def _get_raw_signal_block(raw_data, seq_to_move_map, first_sample_template, block_stride,
                          start_seq_position, end_seq_position):
    start_raw_index = first_sample_template + seq_to_move_map[start_seq_position] * block_stride
    # Get the raw index of the *next* entry in the seq_to_move map,
    # so that we include all the raw data for end_seq_position
    if end_seq_position + 1 == len(seq_to_move_map):
        end_raw_index = len(raw_data)
    else:
        end_raw_index = first_sample_template + seq_to_move_map[end_seq_position + 1] * block_stride
    return raw_data[start_raw_index:end_raw_index], start_raw_index


def _get_raw_data(fast5_filename, read_id):
    with get_fast5_file(fast5_filename, mode="r") as f5file:
        read = f5file.get_read(read_id)
        raw_data = read.get_raw_data()
    return raw_data


def _create_map_from_basecall_to_signal(move_table):
    cumulative_moves = np.add.accumulate(move_table)
    sequence_to_move_index = [np.min(np.where(cumulative_moves == x))
                              for x in
                              range(1, int(np.max(cumulative_moves) + 1))]
    return sequence_to_move_index


def _get_fast5_data(fast5_filename, read_id):
    with get_fast5_file(fast5_filename) as f5file:
        read = f5file.get_read(read_id)
        basecall_analysis = read.get_latest_analysis('Basecall_1D')
        segmentation_analysis = read.get_latest_analysis('Segmentation')
        segmentation_summary = read.get_summary_data(segmentation_analysis)
        basecall_summary = read.get_summary_data(basecall_analysis)
        move_table = read.get_analysis_dataset(basecall_analysis, 'BaseCalled_template/Move')
        first_sample_template = segmentation_summary['segmentation']['first_sample_template']
        block_stride = basecall_summary['basecall_1d_template']['block_stride']
    return move_table, first_sample_template, block_stride


def plot_distribution(dir_in, files_fastq, bins=50):
    """ Distribution of start_raw_index """
    data = []

    for file in os.listdir(dir_in):
        fast5_filename = os.path.join(dir_in, file)
        read_id = file[:-6]
        start_seq_position = 0
        end_seq_position = files_fastq[read_id] - 1
        _, start_raw_index = get_raw_signal_block(fast5_filename, read_id, start_seq_position, end_seq_position)
        data.append(start_raw_index)

    plt.hist(data, bins=bins, color='green' if modification == 'nomod' else 'red')
    plt.title(f'Distribution of "start_raw_index" ({modification})')
    plt.xlabel('start_raw_index')
    plt.ylabel('count')
    plt.xticks(np.arange(0, 5001, 500))  # Change if needed
    plt.yticks(np.arange(0, 151, 25))  # Change if needed
    plt.savefig(f'../plots/start_raw_index_{modification}.png')


def plot_raw_signal(dir_in, files_fastq, fast5_file, n=1000):
    """ raw signal data + start_raw_index line """
    fast5_filename = os.path.join(dir_in, fast5_file)
    read_id = fast5_file[:-6]
    start_seq_position = 0
    end_seq_position = files_fastq[read_id] - 1
    raw_data, start_raw_index = get_raw_signal_block(fast5_filename, read_id, start_seq_position, end_seq_position)

    x_first_n = [str(x) for x in range(0, n)]
    x_last_n = [str(x) for x in range(len(raw_data) - n, len(raw_data))]
    y_first_n = raw_data[:n]
    y_last_n = raw_data[-n:]

    x = x_first_n + x_last_n
    y = y_first_n + y_last_n

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, color='green' if modification == 'nomod' else 'red')
    plt.axvline(x=str(start_raw_index), color='darkblue', linestyle='--', label=f'start_raw_index = {start_raw_index}')

    ax.set_xticks(x[::100])  # Change if needed
    ax.set_xticklabels(x[::100], rotation=45)  # Change if needed
    plt.title(f'read_id: {read_id}')
    plt.xlabel('time')
    plt.ylabel('raw signal data')
    plt.legend()
    plt.savefig(f'../plots/raw_signal_{read_id}.png')


def plot_stdev(dir_in, files_fastq, fast5_file, n=1000, window=50, step=5):
    """ Standard deviation of the raw signal """
    fast5_filename = os.path.join(dir_in, fast5_file)
    read_id = fast5_file[:-6]
    start_seq_position = 0
    end_seq_position = files_fastq[read_id] - 1
    raw_data, start_raw_index = get_raw_signal_block(fast5_filename, read_id, start_seq_position, end_seq_position)

    first_n = raw_data[:n]
    last_n = raw_data[-n:]

    stdev_first_n = []
    stdev_last_n = []
    for i in range(0, n - window, step):
        points_first_n = np.array(first_n[i: i + window], dtype=np.float64)
        stdev_first_n.append(statistics.stdev(points_first_n))
        points_last_n = np.array(last_n[i: i + window], dtype=np.float64)
        stdev_last_n.append(statistics.stdev(points_last_n))

    fig, axs = plt.subplots(2)
    fig.suptitle(f'Standard deviation - {read_id}')
    axs[0].plot(range(0, n - window, step), stdev_first_n, color='green' if modification == 'nomod' else 'red')
    axs[1].plot(range(0, n - window, step), stdev_last_n, color='green' if modification == 'nomod' else 'red')
    plt.xlabel('time')
    axs[0].set_ylabel(f'stdev - first {n}')
    axs[1].set_ylabel(f'stdev - last {n}')
    plt.savefig(f'../plots/stdev_{read_id}.png')


if __name__ == '__main__':
    modification = 'nomod'  # 'mod' or 'nomod'
    dir_in_fastq = f'/home/sdeur/tmp-guppy/{modification}-fastq'
    dir_in = f'/home/sdeur/tmp-guppy/{modification}/workspace'

    files_fastq = {}  # {read_id : fastq_len}
    for file_fastq in os.listdir(dir_in_fastq):
        with open(os.path.join(dir_in_fastq, file_fastq)) as fin:
            files_fastq[file_fastq[:-6]] = len(fin.readlines()[1].strip())

    # Choose between the following plotting methods
    plot_distribution(dir_in, files_fastq)
    plot_raw_signal(dir_in, files_fastq, 'a40f1242-df43-4b9d-a97e-d9af64f3c956.fast5')
    plot_stdev(dir_in, files_fastq, 'a40f1242-df43-4b9d-a97e-d9af64f3c956.fast5')
