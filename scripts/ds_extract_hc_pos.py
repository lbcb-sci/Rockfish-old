from contextlib import ExitStack

import argparse

DS_LINE = '{}\t{}\t{}\n'


def extract_hc_pos(bed_file, nomod_out, mod_out):
    with ExitStack() as stack:
        bed = stack.enter_context(open(bed_file, 'r'))
        nomod = stack.enter_context(open(nomod_out, 'w'))
        mod = stack.enter_context(open(mod_out, 'w'))

        for line in bed:
            data = line.strip().split('\t')

            n_reads = int(data[-2])
            m_freq = int(data[-1])

            if n_reads >= 10:
                if m_freq == 0:
                    example = DS_LINE.format(data[0], data[1], data[5])
                    nomod.write(example)
                if m_freq == 100:
                    example = DS_LINE.format(data[0], data[1], data[5])
                    mod.write(example)


def create_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('bed', type=str)
    parser.add_argument('--nomod', type=str, default='hc_nomod.tsv')
    parser.add_argument('--mod', type=str, default='hc_mod.tsv')

    return parser.parse_args()


if __name__ == '__main__':
    args = create_arguments()

    extract_hc_pos(args.bed, args.nomod, args.mod)
