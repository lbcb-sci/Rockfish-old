import h5py
import numpy as np
from pathlib import Path
import subprocess
import io

from abc import ABC, abstractmethod
from typing import *

from .models import *


class DataWriter(ABC):
    """ Abstract class for writing generated dataset. """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def write_data(self, data: FeaturesData, bed_info: Optional[BEDData]=None) -> None:
        """ Function that writes read's data.

        This abstract method writes generated data for one read. All concrete implementations of data writer should
        override this method.

        :param data: Generated data corresponding to one read
        :param bed_info: Optional BED info containing modification information used for labeling
        """
        pass

    @staticmethod
    def write_extraction_info(path: Path, info: Dict[str, Any], **kwargs) -> None:
        with path.open('w') as f:
            for k, v in info.items():
                f.write(f'{k}\t{v}\n')

    @staticmethod
    def on_extraction_finish(*args, **kwargs) -> None:
        pass


class HDF5Writer(DataWriter):
    """ Implementation of DataWriter that stores data in HDF5 file. """

    def __init__(self, filename: str, generation_info: Dict[str, Any]) -> None:
        super().__init__()

        self.filename = filename
        self.generation_info = generation_info


    def __enter__(self):
        self.fd = h5py.File(self.filename, 'w', libver='latest')

        info_group = self.fd.create_group('info')
        info_attrs = info_group.attrs
        for k, v in self.generation_info.items():
            info_attrs[k] = v

        return self

    def __exit__(self, type, value, traceback):
        self.fd.close()

    def write_data(self, data: FeaturesData, bed_info: Optional[BEDData]=None) -> None:
        """ Function that writes data into HDF5 file.

        This function creates group for the read given by the data object. Stores alignment information as attributes
        and creates three datasets containing information about positions, reference k-mers and signal points.

        :param data: Generated data and alignment information for one read
        :param bed_info: Optional BED modification information used for labeling
        """
        group = self.fd.create_group(data.read_id)
        attrs = group.attrs

        attrs['chromosome'] = data.chromosome
        attrs['read_id'] = data.read_id
        attrs['strand'] = data.strand.value

        positions = []
        ref_kmers = []
        points = []

        if bed_info is not None:
            labels = []

        for example in data.examples:
            positions.append(example.position)
            ref_kmers.append([base_encoding[base] for base in example.ref_kmer])

            points.append(example.signal_points)

            if bed_info is not None:
                mod_info = bed_info[GenomicPos(data.chromosome, example.position, data.strand)]
                label = 1 if mod_info.mod_freq > 50 else 0
                labels.append(label)

        positions = np.array(positions)
        ref_kmer = np.array(ref_kmers, dtype=np.ubyte)
        points = np.array(points, dtype=np.float16)

        group.create_dataset('positions', data=positions, chunks=None)
        group.create_dataset('ref_kmer', data=ref_kmer, chunks=None)
        group.create_dataset('points', data=points, chunks=None)

        if bed_info is not None:
            labels = np.array(labels, dtype=np.ubyte)
            group.create_dataset('labels', data=labels, chunks=None)


class BinaryWriter(DataWriter):
    """ Implementation of DataWriter that stores data in binary file. """

    TRAIN_DTYPE = np.dtype([('signal', np.float16, (340,)), ('kmer', np.uint8, (17,)), ('label', np.uint8)])

    def __init__(self, filename: str) -> None:
        super().__init__()

        self.filename = filename

    def __enter__(self):
        self.fd = io.open(self.filename, 'wb')
        return self

    def __exit__(self, type, value, traceback):
        self.fd.flush()
        self.fd.close()

    def write_data(self, data: FeaturesData, bed_info: Optional[BEDData]=None, label: Optional[int]=None) -> None:
        """ Function that writes data into binary file.

        :param data: Generated data and alignment information for one read
        :param bed_info: Optional BED modification information used for labeling
        :param label: Optional integer present if label is explicitly given
        """
        array = np.empty((len(data.examples),), dtype=BinaryWriter.TRAIN_DTYPE)

        for i, example in enumerate(data.examples):
            array[i]['signal'] = example.signal_points
            array[i]['kmer'] = [base_encoding[base] for base in example.ref_kmer]

            if bed_info is not None:
                mod_info = bed_info[GenomicPos(data.chromosome, example.position, data.strand)]
                bed_label = 1 if mod_info.mod_freq > 50 else 0
                array[i]['label'] = bed_label
            elif label is not None:
                array[i]['label'] = label
            else:
                raise ValueError('No label was provided.')

        array_bytes = array.tobytes()
        self.fd.write(array_bytes)

    @staticmethod
    def on_extraction_finish(*args, **kwargs) -> None:
        """Function that concatenates temporary files into a single output and removes temporary files."""

        src_path = Path(kwargs['path'], '*.bin.tmp')
        dest_path = Path(kwargs['path'], 'data.bin')
        cat_command = f'cat {src_path} > {dest_path}'

        subprocess.run(cat_command, shell=True)

        rm_command = f'rm {src_path}'
        subprocess.run(rm_command, shell=True)
