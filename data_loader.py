import os
import math
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import csv
import re
import logging
from dataclasses import dataclass

import config
from process_spectrum import get_ion_index, process_peaks

import config

logger = logging.getLogger(__name__)


def parse_raw_sequence(raw_sequence: str):
    raw_sequence_len = len(raw_sequence)
    peptide = []
    index = 0
    while index < raw_sequence_len:
        if raw_sequence[index] == "(":
            if peptide[-1] == "C" and raw_sequence[index:index + 8] == "(+57.02)":  # +57.021
                peptide[-1] = "C(Carbamidomethylation)"
                index += 8
            elif peptide[-1] == 'M' and raw_sequence[index:index + 8] == "(+15.99)":  # +15.995
                peptide[-1] = 'M(Oxidation)'
                index += 8
            elif peptide[-1] == 'C' and raw_sequence[index] != "(":  # +15.995
                return False, peptide
            # elif peptide[-1] == 'N' and raw_sequence[index:index + 6] == "(+.98)":  # +0.984
            #     peptide[-1] = 'N(Deamidation)'
            #     index += 6
            # elif peptide[-1] == 'Q' and raw_sequence[index:index + 6] == "(+.98)":
            #     peptide[-1] = 'Q(Deamidation)'
            #     index += 6
            else:  # unknown modification
                logger.warning(f"unknown modification in seq {raw_sequence}")
                return False, peptide
        else:
            peptide.append(raw_sequence[index])
            index += 1

    return True, peptide
# print(parse_raw_sequence('AIIISC(+57.02)TYIK'))

def to_tensor(data_dict: dict) -> dict:
    temp = [(k, torch.from_numpy(v)) for k, v in data_dict.items()]
    return dict(temp)


def pad_to_length(input_data: list, pad_token, max_length: int) -> list:
    assert len(input_data) <= max_length
    result = input_data[:]
    for i in range(max_length - len(result)):
        result.append(pad_token)
    return result


@dataclass
class DDAFeature:
    feature_id: str
    category: str
    mz: float
    z: float
    peptide: list
    mass: float

    lys_mz: float
    lys_z: float
    lys_peptide: list
    lys_mass: float


@dataclass
class DenovoData:
    peak_location: np.ndarray
    peak_intensity: np.ndarray
    # spectrum_representation: np.ndarray
    mirror_peak_location: np.ndarray
    mirror_peak_intensity: np.ndarray
    original_dda_feature: DDAFeature

@dataclass
class MGFfeature:
    PEPMASS: float
    CHARGE: int
    SCANS: str
    SEQ: str
    RTINSECONDS: float
    MOZ_LIST: list
    INTENSITY_LIST: list

@dataclass
class TrainData:
    peak_location: np.ndarray
    peak_intensity: np.ndarray
    lys_peak_location: np.ndarray
    lys_peak_intensity: np.ndarray

    forward_id_target: list
    backward_id_target: list
    forward_ion_location_index_list: list
    backward_ion_location_index_list: list
    forward_id_input: list
    backward_id_input: list

    lys_forward_ion_location_index_list: list
    lys_backward_ion_location_index_list: list


class DeepNovoTrainDataset(Dataset):
    def __init__(self, args, spectrum_path, mirror_spectrum_path, feature_path, transform=None):
        """
        read all feature information and store in memory,
        :param feature_filename:
        :param spectrum_filename:
        """
        print('start')
        logger.info(f"input spectrum file: {spectrum_path}")
        logger.info(f"input feature file: {feature_path}")
        self.args = args
        self.try_spectrum_filename = spectrum_path
        self.lys_spectrum_filename = mirror_spectrum_path
        self.input_spectrum_handle = None
        self.input_mirror_spectrum_handle = None
        self.feature_list = []
        self.try_spectrum_location_dict = self.load_MSData(spectrum_path)
        self.lys_spectrum_location_dict = self.load_MSData(mirror_spectrum_path)
        self.transform = transform
        # read spectrum location file

        # read feature file
        skipped_by_mass = 0
        skipped_by_ptm = 0
        skipped_by_length = 0
        skipped_by_file = 0
        with open(feature_path, 'r') as fr:
            reader = csv.reader(fr, delimiter=',')
            header = next(reader)
            feature_id_index = header.index(config.col_feature_id)
            category_index = header.index(config.col_mirror_category)
            mz_index = header.index(config.col_precursor_mz)
            z_index = header.index(config.col_precursor_charge)
            seq_index = header.index(config.col_raw_sequence)

            lys_mz_index = header.index(config.lys_col_precursor_mz)
            lys_z_index = header.index(config.lys_col_precursor_charge)
            lys_seq_index = header.index(config.lys_col_raw_sequence)

            for line in reader:
                mass = (float(line[mz_index]) - config.mass_H) * float(line[z_index])
                lys_mass = (float(line[lys_mz_index]) - config.mass_H) * float(line[lys_z_index])
                ok, try_peptide = parse_raw_sequence(line[seq_index])
                lys_ok, lys_peptide = parse_raw_sequence(line[lys_seq_index])
                try_title, lys_title = line[feature_id_index].split("@")
                if not try_title in self.try_spectrum_location_dict.keys() and not lys_title in self.lys_spectrum_location_dict.keys():
                    skipped_by_file += 1
                    logger.debug(f"{line[skipped_by_file]} skipped by spectrum")
                    continue
                if not ok and not lys_ok:
                    skipped_by_ptm += 1
                    logger.debug(f"{line[seq_index]} skipped by ptm")
                    continue
                if mass > self.args.MZ_MAX:
                    skipped_by_mass += 1
                    logger.debug(f"{line[seq_index],mass} skipped by mass")
                    continue
                if len(try_peptide) >= self.args.MAX_LEN:
                    skipped_by_length += 1
                    logger.debug(f"{line[seq_index]} skipped by length")
                    continue
                new_feature = DDAFeature(feature_id=line[feature_id_index],
                                         category=line[category_index],
                                         mz=float(line[mz_index]),
                                         z=float(line[z_index]),
                                         peptide=try_peptide,
                                         mass=mass,
                                         lys_mz=float(line[lys_mz_index]),
                                         lys_z=float(line[lys_z_index]),
                                         lys_peptide=lys_peptide,
                                         lys_mass=lys_mass
                                         )
                self.feature_list.append(new_feature)
        logger.info(f"read {len(self.feature_list)} features, {skipped_by_mass} skipped by mass, "
                    f"{skipped_by_ptm} skipped by unknown modification, {skipped_by_length} skipped by length"
                    f"{skipped_by_file} skipped by spectrum")

    def __len__(self):
        return len(self.feature_list)

    def close(self):
        self.input_spectrum_handle.close()

    def load_MSData(self, spectrum_path):
        spectrum_location_file = spectrum_path + '.location.pytorch.pkl'
        if os.path.exists(spectrum_location_file):
            logger.info(f"read cached spectrum locations")
            with open(spectrum_location_file, 'rb') as fr:
                spectrum_location_dict = pickle.load(fr)
        else:
            logger.info("build spectrum location from scratch")
            spectrum_location_dict = {}
            line = True
            with open(spectrum_path, 'r') as f:
                while line:
                    current_location = f.tell()
                    line = f.readline()
                    if "BEGIN IONS" in line:
                        spectrum_location = current_location
                    elif "TITLE=" in line:
                        title = re.split('[=\r\n]', line)[1]
                        spectrum_location_dict[title] = spectrum_location
            with open(spectrum_location_file, 'wb') as fw:
                pickle.dump(spectrum_location_dict, fw)
        return spectrum_location_dict

    def _parse_spectrum_ion(self, input_spectrum_handle):
        mz_list = []
        intensity_list = []
        line = input_spectrum_handle.readline()
        while not "END IONS" in line:
            mz, intensity = re.split(' |\r|\n', line)[:2]
            mz_float = float(mz)
            intensity_float = float(intensity)
            # skip an ion if its mass > MZ_MAX
            if mz_float > self.args.MZ_MAX:
                line = input_spectrum_handle.readline()
                continue
            mz_list.append(mz_float)
            intensity_list.append(math.sqrt(intensity_float))
            line = input_spectrum_handle.readline()
        return mz_list, intensity_list

    def getmzandintensitylist(self, input_spectrum_handle):
        line = input_spectrum_handle.readline()
        assert "BEGIN IONS" in line, "Error: wrong input BEGIN IONS"
        line = input_spectrum_handle.readline()
        assert "TITLE=" in line, "Error: wrong input TITLE="
        line = input_spectrum_handle.readline()
        assert "CHARGE=" in line, "Error: wrong input CHARGE="
        line = input_spectrum_handle.readline()
        assert "RTINSECONDS=" in line, "Error: wrong input RTINSECONDS="
        line = input_spectrum_handle.readline()
        assert "PEPMASS=" in line, "Error: wrong input PEPMASS="

        mz_list, intensity_list = self._parse_spectrum_ion(input_spectrum_handle)
        return mz_list, intensity_list

    def _get_theory_ions(self, peptide, mass):
        peptide_id_list = [config.vocab["L"] if x == "I" else config.vocab[x] for x in peptide]
        # peptide_id_list = [config.vocab[x] for x in peptide]
        forward_id_input = [config.GO_ID] + peptide_id_list
        forward_id_target = peptide_id_list + [config.EOS_ID]
        forward_ion_location_index_list = []
        prefix_mass = 0.
        for i, id in enumerate(forward_id_input):
            prefix_mass += config.mass_ID[id]
            ion_location = get_ion_index(mass, prefix_mass, forward_id_input[:i + 1], 0, args=self.args)
            forward_ion_location_index_list.append(ion_location)

        backward_id_input = [config.EOS_ID] + peptide_id_list[::-1]
        backward_id_target = peptide_id_list[::-1] + [config.GO_ID]
        backward_ion_location_index_list = []
        suffix_mass = 0
        for i, id in enumerate(backward_id_input):
            suffix_mass += config.mass_ID[id]
            ion_location = get_ion_index(mass, suffix_mass, backward_id_input[:i + 1], 1, args=self.args)
            backward_ion_location_index_list.append(ion_location)
        return forward_id_input, forward_id_target, forward_ion_location_index_list, backward_id_input, backward_id_target, backward_ion_location_index_list

    def _get_feature(self, feature: DDAFeature) -> TrainData:
        try_filename, lys_filename = feature.feature_id.split("@")
        try_spectrum_location = self.try_spectrum_location_dict[try_filename]
        lys_spectrum_location = self.lys_spectrum_location_dict[lys_filename]
        self.input_spectrum_handle.seek(try_spectrum_location)
        self.input_mirror_spectrum_handle.seek(lys_spectrum_location)
        mz_list, intensity_list = self.getmzandintensitylist(self.input_spectrum_handle)
        lys_mz_list, lys_intensity_list = self.getmzandintensitylist(self.input_mirror_spectrum_handle)

        peak_location, peak_intensity = process_peaks(mz_list, intensity_list, feature.mass, self.args)
        lys_peak_location, lys_peak_intensity = process_peaks(lys_mz_list, lys_intensity_list, feature.lys_mass, self.args)

        assert np.max(peak_intensity) < 1.0 + 1e-5

        try_forward_id_input, try_forward_id_target, try_forward_ion_location_index_list, try_backward_id_input, try_backward_id_target, try_backward_ion_location_index_list = self._get_theory_ions(feature.peptide, feature.mass)
        lys_forward_id_input, lys_forward_id_target, lys_forward_ion_location_index_list, lys_backward_id_input, lys_backward_id_target, lys_backward_ion_location_index_list = self._get_theory_ions(
            feature.lys_peptide, feature.lys_mass)
        # padd_zeros = np.zeros(self.args.num_aa, self.args.num_ions)
        # lys_backward_ion_location_index_list.insert(1, padd_zeros)
        try_forward_id_input = try_forward_id_input[:-1]#1PEPTIDEK->1PEPTIDE
        try_forward_id_target.pop(-2)#PEPTIDEK2->PEPTIDE2
        lys_forward_ion_location_index_list.pop(1)#1KPEPTIDE->1PEPTIDE feature
        try_forward_ion_location_index_list = try_forward_ion_location_index_list[:-1]#1PEPTIDEK->1PEPTIDE feature

        try_backward_id_input.pop(1)#1KEDITPEP->1EDITPEP
        try_backward_id_target=try_backward_id_target[1:]#KEDITPEP2->EDITPEP2
        try_backward_ion_location_index_list.pop(1)#1KEDITPEP->1EDITPEP feature
        lys_backward_ion_location_index_list=lys_backward_ion_location_index_list[:-1]#1EDITPEPK->1EDITPEP feature
        return TrainData(peak_location=peak_location,
                         peak_intensity=peak_intensity,
                         lys_peak_location=lys_peak_location,
                         lys_peak_intensity=lys_peak_intensity,
                         forward_id_target=try_forward_id_target,
                         backward_id_target=try_backward_id_target,
                         forward_ion_location_index_list=try_forward_ion_location_index_list,
                         backward_ion_location_index_list=try_backward_ion_location_index_list,
                         forward_id_input=try_forward_id_input,
                         backward_id_input=try_backward_id_input,
                         lys_forward_ion_location_index_list=lys_forward_ion_location_index_list,
                         lys_backward_ion_location_index_list=lys_backward_ion_location_index_list)

    def __getitem__(self, idx):
        if self.input_spectrum_handle is None:
            self.input_spectrum_handle = open(self.try_spectrum_filename, 'r')
        if self.input_mirror_spectrum_handle is None:
            self.input_mirror_spectrum_handle = open(self.lys_spectrum_filename, 'r')
        feature = self.feature_list[idx]
        return self._get_feature(feature)


def collate_func(train_data_list):
    """
    :param train_data_list: list of TrainData
    :return:
        peak_location: [batch, N]
        peak_intensity: [batch, N]
        forward_target_id: [batch, T]
        backward_target_id: [batch, T]
        forward_ion_index_list: [batch, T, 26, 8]
        backward_ion_index_list: [batch, T, 26, 8]
    """
    # sort data by seq length (decreasing order)
    train_data_list.sort(key=lambda x: len(x.forward_id_target), reverse=True)
    batch_max_seq_len = len(train_data_list[0].forward_id_target)
    ion_index_shape = train_data_list[0].forward_ion_location_index_list[0].shape

    peak_location = [x.peak_location for x in train_data_list]
    peak_location = np.stack(peak_location) # [batch_size, N]
    peak_location = torch.from_numpy(peak_location)

    peak_intensity = [x.peak_intensity for x in train_data_list]
    peak_intensity = np.stack(peak_intensity) # [batch_size, N]
    peak_intensity = torch.from_numpy(peak_intensity)

    lys_peak_location = [x.lys_peak_location for x in train_data_list]
    lys_peak_location = np.stack(lys_peak_location)  # [batch_size, N]
    lys_peak_location = torch.from_numpy(lys_peak_location)

    lys_peak_intensity = [x.lys_peak_intensity for x in train_data_list]
    lys_peak_intensity = np.stack(lys_peak_intensity)  # [batch_size, N]
    lys_peak_intensity = torch.from_numpy(lys_peak_intensity)

    batch_forward_ion_index = []
    lys_batch_forward_ion_index = []
    batch_forward_id_target = []
    batch_forward_id_input = []
    for data in train_data_list:
        ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                               np.float32)
        lys_ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                                np.float32)
        forward_ion_index = np.stack(data.forward_ion_location_index_list)
        lys_forward_ion_index = np.stack(data.lys_forward_ion_location_index_list)
        ion_index[:forward_ion_index.shape[0], :, :] = forward_ion_index
        lys_ion_index[:lys_forward_ion_index.shape[0], :, :] = lys_forward_ion_index
        batch_forward_ion_index.append(ion_index)
        lys_batch_forward_ion_index.append(lys_ion_index)

        f_target = np.zeros((batch_max_seq_len,), np.int64)
        forward_target = np.array(data.forward_id_target, np.int64)
        f_target[:forward_target.shape[0]] = forward_target
        batch_forward_id_target.append(f_target)

        f_input = np.zeros((batch_max_seq_len,), np.int64)
        forward_input = np.array(data.forward_id_input, np.int64)
        f_input[:forward_input.shape[0]] = forward_input
        batch_forward_id_input.append(f_input)
    batch_forward_id_target = torch.from_numpy(np.stack(batch_forward_id_target))  # [batch_size, T]
    batch_forward_ion_index = torch.from_numpy(np.stack(batch_forward_ion_index))  # [batch, T, 26, 8]
    batch_forward_id_input = torch.from_numpy(np.stack(batch_forward_id_input))
    lys_batch_forward_ion_index = torch.from_numpy(np.stack(lys_batch_forward_ion_index))

    batch_backward_ion_index = []
    lys_batch_backward_ion_index = []
    batch_backward_id_target = []
    batch_backward_id_input = []
    for data in train_data_list:
        ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                             np.float32)
        lys_ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                               np.float32)
        backward_ion_index = np.stack(data.backward_ion_location_index_list)
        ion_index[:backward_ion_index.shape[0], :, :] = backward_ion_index
        batch_backward_ion_index.append(ion_index)
        lys_backward_ion_index = np.stack(data.lys_backward_ion_location_index_list)
        lys_ion_index[:lys_backward_ion_index.shape[0], :, :] = lys_backward_ion_index
        lys_batch_backward_ion_index.append(lys_ion_index)

        b_target = np.zeros((batch_max_seq_len,), np.int64)
        backward_target = np.array(data.backward_id_target, np.int64)
        b_target[:backward_target.shape[0]] = backward_target
        batch_backward_id_target.append(b_target)

        b_input = np.zeros((batch_max_seq_len,), np.int64)
        backward_input = np.array(data.backward_id_input, np.int64)
        b_input[:backward_input.shape[0]] = backward_input
        batch_backward_id_input.append(b_input)
    batch_backward_id_target = torch.from_numpy(np.stack(batch_backward_id_target))  # [batch_size, T]
    batch_backward_ion_index = torch.from_numpy(np.stack(batch_backward_ion_index))  # [batch, T, 26, 8]
    batch_backward_id_input = torch.from_numpy(np.stack(batch_backward_id_input))
    lys_batch_backward_ion_index = torch.from_numpy(np.stack(lys_batch_backward_ion_index))

    # lys_batch_forward_ion_index = []
    # for data in train_data_list:
    #     ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
    #                          np.float32)
    #     forward_ion_index = np.stack(data.lys_forward_ion_location_index_list[:1] + data.lys_forward_ion_location_index_list[2:])
    #     ion_index[:forward_ion_index.shape[0], :, :] = forward_ion_index
    #     lys_batch_forward_ion_index.append(ion_index)
    #
    # lys_batch_forward_ion_index = torch.from_numpy(np.stack(lys_batch_forward_ion_index))  # [batch, T, 26, 8]
    #
    # lys_batch_backward_ion_index = []
    # for data in train_data_list:
    #     ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
    #                          np.float32)
    #     backward_ion_index = np.stack(data.lys_backward_ion_location_index_list[:-1])
    #     ion_index[:backward_ion_index.shape[0], :, :] = backward_ion_index
    #     lys_batch_backward_ion_index.append(ion_index)
    #
    # lys_batch_backward_ion_index = torch.from_numpy(np.stack(lys_batch_backward_ion_index))  # [batch, T, 26, 8]

    return (peak_location,
            peak_intensity,
            lys_peak_location,
            lys_peak_intensity,
            batch_forward_id_target,
            batch_backward_id_target,
            batch_forward_ion_index,
            batch_backward_ion_index,
            batch_forward_id_input,
            batch_backward_id_input,
            lys_batch_forward_ion_index,
            lys_batch_backward_ion_index
            )

# helper functions
def chunks(l, n: int):
    for i in range(0, len(l), n):
        yield l[i:i + n]


class DeepNovoDenovoDataset(DeepNovoTrainDataset):
    # override _get_feature method
    def _get_feature(self, feature: DDAFeature) -> DenovoData:
        try_filename, lys_filename = feature.feature_id.split("@")
        try_spectrum_location = self.try_spectrum_location_dict[try_filename]
        lys_spectrum_location = self.lys_spectrum_location_dict[lys_filename]
        self.input_spectrum_handle.seek(try_spectrum_location)
        self.input_mirror_spectrum_handle.seek(lys_spectrum_location)
        mz_list, intensity_list = self.getmzandintensitylist(self.input_spectrum_handle)
        lys_mz_list, lys_intensity_list = self.getmzandintensitylist(self.input_mirror_spectrum_handle)

        peak_location, peak_intensity = process_peaks(mz_list, intensity_list, feature.mass, self.args)
        lys_peak_location, lys_peak_intensity = process_peaks(lys_mz_list, lys_intensity_list, feature.lys_mass,
                                                              self.args)
        # print("mz list: ", mz_list[0], try_filename)
        # print("lys mz list: ", lys_mz_list[0], lys_filename)
        return DenovoData(peak_location=peak_location,
                          peak_intensity=peak_intensity,
                          mirror_peak_location=lys_peak_location,
                          mirror_peak_intensity=lys_peak_intensity,
                          original_dda_feature=feature)
