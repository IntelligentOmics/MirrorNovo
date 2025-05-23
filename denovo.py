import os
from _codecs import latin_1_encode

import torch
import time
import logging
import config
import numpy as np
from dataclasses import dataclass
# from model_gru_serise import Direction, InferenceModelWrapper, device
from model_gru import Direction, InferenceModelWrapper, device
from process_spectrum import get_ion_index
from data_loader import DeepNovoDenovoDataset, chunks
from writer import BeamSearchedSequence, DenovoWriter, DDAFeature
import copy

logger = logging.getLogger(__name__)


@dataclass
class BeamSearchStartPoint:
    prefix_mass: float
    suffix_mass: float
    mass_tolerance: float
    direction: Direction


@dataclass
class DenovoResult:
    dda_feature: DDAFeature
    best_beam_search_sequence: BeamSearchedSequence

@dataclass
class BeamSearchResult:
    dda_feature: DDAFeature
    beam_search_batch: list

class KnapsackSearcher(object):
    def __init__(self, MZ_MAX, knapsack_file):
        self.knapsack_file = knapsack_file
        self.MZ_MAX = MZ_MAX
        self.knapsack_aa_resolution = config.KNAPSACK_AA_RESOLUTION
        if os.path.isfile(knapsack_file):
            logging.info("KnapsackSearcher.__init__(): load knapsack matrix")
            self.knapsack_matrix = np.load(knapsack_file)
        else:
            logging.info("KnapsackSearcher.__init__(): build knapsack matrix from scratch")
            self.knapsack_matrix = self._build_knapsack()

    def _build_knapsack(self):
        max_mass = self.MZ_MAX - config.mass_N_terminus - config.mass_C_terminus
        max_mass_round = int(round(max_mass * self.knapsack_aa_resolution))
        max_mass_upperbound = max_mass_round + self.knapsack_aa_resolution
        knapsack_matrix = np.zeros(shape=(config.vocab_size, max_mass_upperbound), dtype=bool)
        for aa_id in range(3, config.vocab_size):
            mass_aa = int(round(config.mass_ID[aa_id] * self.knapsack_aa_resolution))

            for col in range(max_mass_upperbound):
                current_mass = col + 1
                if current_mass < mass_aa:
                    knapsack_matrix[aa_id, col] = False
                elif current_mass == mass_aa:
                    knapsack_matrix[aa_id, col] = True
                elif current_mass > mass_aa:
                    sub_mass = current_mass - mass_aa
                    sub_col = sub_mass - 1
                    if np.sum(knapsack_matrix[:, sub_col]) > 0:
                        knapsack_matrix[aa_id, col] = True
                        knapsack_matrix[:, col] = np.logical_or(knapsack_matrix[:, col], knapsack_matrix[:, sub_col])
                    else:
                        knapsack_matrix[aa_id, col] = False
        np.save(self.knapsack_file, knapsack_matrix)
        return knapsack_matrix

    def search_knapsack(self, mass, knapsack_tolerance):
        mass_round = int(round(mass * self.knapsack_aa_resolution))
        mass_upperbound = mass_round + knapsack_tolerance
        mass_lowerbound = mass_round - knapsack_tolerance
        if mass_upperbound < config.mass_AA_min_round:
            return []
        mass_lowerbound_col = mass_lowerbound - 1
        mass_upperbound_col = mass_upperbound - 1
        candidate_aa_id = np.flatnonzero(np.any(self.knapsack_matrix[:, mass_lowerbound_col:(mass_upperbound_col + 1)],
                                                axis=1))
        return candidate_aa_id.tolist()
    def search_knapsack_ppm(self, residual_mass,precursor_mass, peak_mass_tolerance):

        # mass_upperbound = int(round(((10 ** 6 * precursor_mass / (10 ** 6 - peak_mass_tolerance)) - precursor_mass + residual_mass)* self.knapsack_aa_resolution))
        mass_upperbound = int(round((residual_mass + (peak_mass_tolerance * precursor_mass / (10 ** 6 - peak_mass_tolerance)))* self.knapsack_aa_resolution))
        # mass_lowerbound = int(round((residual_mass - precursor_mass + (10 ** 6 * precursor_mass / (10 ** 6 + peak_mass_tolerance)))* self.knapsack_aa_resolution))
        mass_lowerbound = int(round((residual_mass - (peak_mass_tolerance * precursor_mass / (10 ** 6 + peak_mass_tolerance)))* self.knapsack_aa_resolution))
        if mass_upperbound < config.mass_AA_min_round:
            return []
        mass_lowerbound_col = mass_lowerbound - 1
        mass_upperbound_col = mass_upperbound - 1
        candidate_aa_id = np.flatnonzero(np.any(self.knapsack_matrix[:, mass_lowerbound_col:(mass_upperbound_col + 1)],
                                                axis=1))
        return candidate_aa_id.tolist()


@dataclass
class SearchPath:
    aa_id_list: list
    # mirror_aa_id_list: list
    aa_seq_mass: float
    # mirror_aa_seq_mass: float
    score_list: list
    score_sum: float
    # lstm_state: tuple  # state tupe store in search path is of shape [num_lstm_layers, num_units]
    direction: Direction


@dataclass
class SearchEntry:
    feature_index: int
    current_path_list: list  # list of search paths
    spectrum_state: tuple  # tuple of (peak_location, peak_intensity)
    error_mass: float
    error_aa_id: int


class IonCNNDenovo(object):
    def __init__(self, args):
        self.args = args
        self.MZ_MAX = args.MZ_MAX  # legacy, not used here
        self.beam_size = args.beam_size
        self.knapsack_searcher = KnapsackSearcher(self.MZ_MAX, args.knapsack)

    def _beam_search(self, model_wrapper: InferenceModelWrapper,
                     feature_dp_batch: list, start_point_batch: list) -> list:
        """

        :param model_wrapper:
        :param feature_dp_batch: list of DenovoData
        :param start_point_batch:
        :return:
        """
        num_features = len(feature_dp_batch)
        top_path_batch = [[] for _ in range(num_features)]
        direction_cint_map = {Direction.forward: 0, Direction.backward: 1}

        direction = start_point_batch[0].direction
        # print("direction: ", direction)
        if direction == Direction.forward:
            get_start_mass = lambda x: x.prefix_mass
            first_label = config.GO_ID
            last_label = config.EOS_ID
        elif direction == Direction.backward:
            get_start_mass = lambda x: x.suffix_mass
            first_label = config.EOS_ID
            last_label = config.GO_ID
        else:
            raise ValueError('direction neither forward nor backward')

        # step 1: extract original spectrum
        batch_peak_location = np.array([x.peak_location for x in feature_dp_batch])
        batch_peak_intensity = np.array([x.peak_intensity for x in feature_dp_batch])
        # batch_spectrum_representation = np.array([x.spectrum_representation for x in feature_dp_batch])
        batch_mirror_peak_location = np.array([x.mirror_peak_location for x in feature_dp_batch])
        batch_mirror_peak_intensity = np.array([x.mirror_peak_intensity for x in feature_dp_batch])
        batch_category = [x.original_dda_feature.category for x in feature_dp_batch]

        batch_peak_location = torch.from_numpy(batch_peak_location).to(device)
        batch_peak_intensity = torch.from_numpy(batch_peak_intensity).to(device)
        batch_mirror_peak_location = torch.from_numpy(batch_mirror_peak_location).to(device)
        batch_mirror_peak_intensity = torch.from_numpy(batch_mirror_peak_intensity).to(device)
        # batch_spectrum_representation = torch.from_numpy(batch_spectrum_representation).to(device)

        # initialize activate search list
        active_search_list = []
        for feature_index in range(num_features):
            # all feature in the same batch should be from same direction
            assert direction == start_point_batch[feature_index].direction
            category = batch_category[feature_index]

            spectrum_state = (batch_peak_location[feature_index],
                              batch_peak_intensity[feature_index],
                              batch_mirror_peak_location[feature_index],
                              batch_mirror_peak_intensity[feature_index]
                              )
            #
            if direction == Direction.forward:
                path = SearchPath(
                    aa_id_list=[first_label],
                    # mirror_aa_id_list=[first_label, config.lys_first_aa_id_dict[batch_category[feature_index]]],
                    aa_seq_mass=get_start_mass(start_point_batch[feature_index]),
                    # mirror_aa_seq_mass=get_start_mass(start_point_batch[feature_index])+config.lys_add_direction_dict[direction][batch_category[feature_index]],
                    score_list=[0.0],
                    score_sum=0.0,
                    direction=direction,
                )
                search_entry = SearchEntry(
                    feature_index=feature_index,
                    current_path_list=[path],
                    spectrum_state=spectrum_state,
                    error_mass=config.lys_first_aa_mass_dict[category],
                    error_aa_id=config.lys_first_aa_id_dict[category]
                )
                active_search_list.append(search_entry)
            elif direction == Direction.backward:
                path = SearchPath(
                    aa_id_list=[first_label, config.try_last_aa_id_dict[category]],
                    aa_seq_mass=get_start_mass(start_point_batch[feature_index])+config.try_last_aa_mass_dict[category],
                    score_list=[0.0, 0.0],
                    score_sum=0.0,
                    direction=direction,
                )
                search_entry = SearchEntry(
                    feature_index=feature_index,
                    current_path_list=[path],
                    spectrum_state=spectrum_state,
                    error_mass=config.try_last_aa_mass_dict[category],
                    error_aa_id=config.try_last_aa_id_dict[category]
                )
                active_search_list.append(search_entry)

        # repeat STEP 2, 3, 4 until the active_search_list is empty.
        while True:
            # STEP 2: gather data from active search entries and group into blocks.

            # model input
            block_aa_id_input = []
            block_ion_location = []
            block_peak_location = []
            block_peak_intensity = []

            block_mirror_ion_location = []
            block_mirror_peak_location = []
            block_mirror_peak_intensity = []
            # block_lstm_h = []
            # block_lstm_c = []
            # data stored in path
            block_aa_id_list = []
            block_aa_seq_mass = []
            block_score_list = []
            block_score_sum = []
            block_knapsack_candidates = []

            # store the number of paths of each search entry in the big blocks
            #     to retrieve the info of each search entry later in STEP 4.
            search_entry_size = [0] * len(active_search_list)

            for entry_index, search_entry in enumerate(active_search_list):
                feature_index = search_entry.feature_index
                current_path_list = search_entry.current_path_list
                precursor_mass = feature_dp_batch[feature_index].original_dda_feature.mass
                lys_precursor_mass = feature_dp_batch[feature_index].original_dda_feature.lys_mass
                peak_mass_tolerance = start_point_batch[feature_index].mass_tolerance
                error_mass = search_entry.error_mass
                error_aa_id = search_entry.error_aa_id

                for path in current_path_list:
                    aa_id_list = path.aa_id_list
                    aa_id = aa_id_list[-1]
                    score_sum = path.score_sum
                    aa_seq_mass = path.aa_seq_mass
                    score_list = path.score_list
                    original_spectrum_tuple = search_entry.spectrum_state
                    # lstm_state_tuple = path.lstm_state
                    # print("aa id list: ", aa_id_list)
                    if aa_id == last_label:
                        # if abs(aa_seq_mass - precursor_mass) <= peak_mass_tolerance:
                        # if precursor_mass - aa_seq_mass <= peak_mass_tolerance and precursor_mass - aa_seq_mass >= 0:
                        if abs((precursor_mass - aa_seq_mass) / aa_seq_mass * 10 ** 6) <= peak_mass_tolerance:
                            seq = aa_id_list[1:-1]
                            trunc_score_list = score_list[1:-1]
                            if direction == Direction.backward:
                                seq = seq[::-1]
                                trunc_score_list = trunc_score_list[::-1]

                            top_path_batch[feature_index].append(
                                BeamSearchedSequence(sequence=seq,
                                                     position_score=trunc_score_list,
                                                     score=path.score_sum / len(seq),
                                                     direction=direction.value)
                            )
                        continue
                    ion_location = get_ion_index(precursor_mass, aa_seq_mass, aa_id_list,direction_cint_map[direction], args=self.args)  # [26,8]
                    # print(aa_id_list)
                    if direction == Direction.forward:
                        mirror_aa_seq_mass = aa_seq_mass + error_mass
                        mirror_aa_id_list = aa_id_list.copy()
                        mirror_aa_id_list.insert(1, error_aa_id)
                    elif direction == Direction.backward:
                        mirror_aa_seq_mass = aa_seq_mass - error_mass
                        mirror_aa_id_list = aa_id_list.copy()
                        mirror_aa_id_list.pop(1)
                    # print("mirror aa id list: ", mirror_aa_id_list)
                    # print("mirror aa seq mass: ", mirror_aa_seq_mass)
                    # print("mirror aa id: ", mirror_aa_id_list, aa_id_list)
                    mirror_ion_location = get_ion_index(lys_precursor_mass, mirror_aa_seq_mass, mirror_aa_id_list, direction_cint_map[direction],
                                                 args=self.args)

                    residual_mass = precursor_mass - aa_seq_mass - config.mass_ID[last_label]
                    '''
                    这是以da为单位计算误差
                    knapsack_tolerance = int(round(peak_mass_tolerance * config.KNAPSACK_AA_RESOLUTION))
                    knapsack_candidates = self.knapsack_searcher.search_knapsack(residual_mass, knapsack_tolerance)
                    '''
                    # 这是以ppm为单位计算误差
                    knapsack_candidates = self.knapsack_searcher.search_knapsack_ppm(residual_mass, precursor_mass,peak_mass_tolerance)
                    # print("knapsack_candidates: ", knapsack_candidates)
                    if not knapsack_candidates:
                        # if not possible aa, force it to stop.
                        knapsack_candidates.append(last_label)

                    block_ion_location.append(ion_location)
                    block_mirror_ion_location.append(mirror_ion_location)
                    block_aa_id_input.append(aa_id)
                    # get hidden state block
                    block_peak_location.append(original_spectrum_tuple[0])
                    block_peak_intensity.append(original_spectrum_tuple[1])
                    block_mirror_peak_location.append(original_spectrum_tuple[2])
                    block_mirror_peak_intensity.append(original_spectrum_tuple[3])
                    # if deepnovo_config.use_lstm:
                    #     block_lstm_h.append(lstm_state_tuple[0])
                    #     block_lstm_c.append(lstm_state_tuple[1])

                    block_aa_id_list.append(aa_id_list)
                    block_aa_seq_mass.append(aa_seq_mass)
                    block_score_list.append(score_list)
                    block_score_sum.append(score_sum)
                    block_knapsack_candidates.append(knapsack_candidates)
                    # record the size of each search entry in the blocks
                    search_entry_size[entry_index] += 1

            # step 3 run model on data blocks to predict next AA.
            #     output is stored in current_log_prob
            # assert block_aa_id_list, 'IonCNNDenovo._beam_search(): aa_id_list is empty.'
            if not block_ion_location:
                # all search entry finished in the previous step
                break

            block_ion_location = torch.from_numpy(np.array(block_ion_location)).to(device)  # [batch, 26, 8, 10]
            block_ion_location = torch.unsqueeze(block_ion_location, dim=1)  # [batch, 1, 26, 8]
            block_peak_location = torch.stack(block_peak_location, dim=0).contiguous()
            block_peak_intensity = torch.stack(block_peak_intensity, dim=0).contiguous()
            block_mirror_ion_location = torch.from_numpy(np.array(block_mirror_ion_location)).to(device)  # [batch, 26, 8, 10]
            block_mirror_ion_location = torch.unsqueeze(block_mirror_ion_location, dim=1)  # [batch, 1, 26, 8]
            block_mirror_peak_location = torch.stack(block_mirror_peak_location, dim=0).contiguous()
            block_mirror_peak_intensity = torch.stack(block_mirror_peak_intensity, dim=0).contiguous()
            # if deepnovo_config.use_lstm:
            #     block_lstm_h = torch.stack(block_lstm_h, dim=1).contiguous()
            #     block_lstm_c = torch.stack(block_lstm_c, dim=1).contiguous()
            #     block_state_tuple = (block_lstm_h, block_lstm_c)
            #     block_aa_id_input = torch.from_numpy(np.array(block_aa_id_input, dtype=np.int64)).unsqueeze(1).to(
            #         device)
            # else:
            #     block_state_tuple = None
            #     block_aa_id_input = None
            # print("block ion location:　", block_ion_location.shape, num_features)
            current_log_prob = model_wrapper.step(block_ion_location,
                                                   block_peak_location,
                                                   block_peak_intensity,
                                                   block_mirror_ion_location,
                                                   block_mirror_peak_location,
                                                   block_mirror_peak_intensity,
                                                   direction)
            # transfer log_prob back to cpu
            current_log_prob = current_log_prob.cpu().numpy()
            # print("current_log_prob: ", current_log_prob)
            # STEP 4: retrieve data from blocks to update the active_search_list
            #     with knapsack dynamic programming and beam search.
            block_index = 0
            for entry_index, search_entry in enumerate(active_search_list):

                new_path_list = []
                direction = search_entry.current_path_list[0].direction
                for index in range(block_index, block_index + search_entry_size[entry_index]):
                    for aa_id in block_knapsack_candidates[index]:
                        if len(block_knapsack_candidates[index])==1 and block_knapsack_candidates[index][0] in (4,13):
                            new_score_list = block_score_list[index] + [0.0]
                            new_score_sum = block_score_sum[index] + 0.0
                        elif aa_id > 2:
                            # do not add score of GO, EOS, PAD
                            # if aa_id == 12 or aa_id == 13:
                            #     new_score_list = block_score_list[index] + [current_log_prob[index][13]]
                            #     new_score_sum = block_score_sum[index] + current_log_prob[index][13]
                            new_score_list = block_score_list[index] + [current_log_prob[index][aa_id]]
                            new_score_sum = block_score_sum[index] + current_log_prob[index][aa_id]
                        else:
                            new_score_list = block_score_list[index] + [0.0]
                            new_score_sum = block_score_sum[index] + 0.0

                        # if deepnovo_config.use_lstm:
                        #     new_path_state_tuple = (new_state_tuple[0][:, index, :], new_state_tuple[1][:, index, :])
                        # else:
                        #     new_path_state_tuple = None

                        new_path = SearchPath(
                            aa_id_list=block_aa_id_list[index] + [aa_id],
                            aa_seq_mass=block_aa_seq_mass[index] + config.mass_ID[aa_id],
                            score_list=new_score_list,
                            score_sum=new_score_sum,
                            direction=direction
                        )
                        new_path_list.append(new_path)
                if len(new_path_list) > self.beam_size:
                    new_path_score = np.array([x.score_sum for x in new_path_list])
                    top_k_index = np.argpartition(-new_path_score, self.beam_size)[:self.beam_size]
                    search_entry.current_path_list = [new_path_list[ii] for ii in top_k_index]
                else:
                    search_entry.current_path_list = new_path_list
                # print("current_path_list: ", search_entry.current_path_list)

                block_index += search_entry_size[entry_index]

            active_search_list = [x for x in active_search_list if x.current_path_list]

            if not active_search_list:
                break
        return top_path_batch

    @staticmethod
    def _get_start_point(feature_dp_batch: list) -> tuple:
        mass_GO = config.mass_ID[config.GO_ID]
        forward_start_point_lists = [BeamSearchStartPoint(prefix_mass=mass_GO,
                                                          suffix_mass=feature_dp.original_dda_feature.mass - mass_GO,
                                                          # mass_tolerance=config.PRECURSOR_MASS_PRECISION_TOLERANCE,
                                                          mass_tolerance=config.PRECURSOR_MASS_PRECISION_TOLERANCE_PPM,
                                                          direction=Direction.forward)
                                     for feature_dp in feature_dp_batch]

        mass_EOS = config.mass_ID[config.EOS_ID]
        backward_start_point_lists = [BeamSearchStartPoint(prefix_mass=feature_dp.original_dda_feature.mass - mass_EOS,
                                                           suffix_mass=mass_EOS,
                                                           # mass_tolerance=config.PRECURSOR_MASS_PRECISION_TOLERANCE,
                                                           mass_tolerance=config.PRECURSOR_MASS_PRECISION_TOLERANCE_PPM,
                                                           direction=Direction.backward)
                                      for feature_dp in feature_dp_batch]
        return forward_start_point_lists, backward_start_point_lists

    @staticmethod
    def _select_path(feature_dp_batch: list, top_candidate_batch: list) -> tuple:
        """
        for each feature, select the best denovo sequence given by DeepNovo model
        :param feature_dp_batch: list of DenovoData
        :param top_candidate_batch: defined in _search_denovo_batch
        :return:
        list of DenovoResult
        """
        feature_batch_size = len(feature_dp_batch)

        refine_batch = [[] for x in range(feature_batch_size)]
        for feature_index in range(feature_batch_size):
            precursor_mass = feature_dp_batch[feature_index].original_dda_feature.mass
            candidate_list = top_candidate_batch[feature_index]
            for beam_search_sequence in candidate_list:
                sequence = beam_search_sequence.sequence
                sequence_mass = sum(config.mass_ID[x] for x in sequence)
                sequence_mass += config.mass_ID[config.GO_ID] + config.mass_ID[
                    config.EOS_ID]
                # if abs(sequence_mass - precursor_mass) <= config.PRECURSOR_MASS_PRECISION_TOLERANCE:#Da
                # if precursor_mass - sequence_mass <= config.PRECURSOR_MASS_PRECISION_TOLERANCE and precursor_mass - sequence_mass >= 0:
                if abs(((precursor_mass - sequence_mass) / sequence_mass) * 10 ** 6) <= config.PRECURSOR_MASS_PRECISION_TOLERANCE_PPM:#ppm
                    logger.debug(f"sequence {sequence} of feature "
                                 f"{feature_dp_batch[feature_index].original_dda_feature.feature_id} refined")
                    refine_batch[feature_index].append(beam_search_sequence)
        predicted_batch = []
        beam_search_batch_predicted = []
        for feature_index in range(feature_batch_size):
            candidate_list = refine_batch[feature_index]
            if not candidate_list:
                best_beam_search_sequence = BeamSearchedSequence(
                    sequence=[],
                    position_score=[],
                    score=-float('inf'),
                    direction=0
                )
            else:
                # sort candidate sequence by average position score
                best_beam_search_sequence = max(candidate_list, key=lambda x: x.score)

            denovo_result = DenovoResult(
                dda_feature=feature_dp_batch[feature_index].original_dda_feature,
                best_beam_search_sequence=best_beam_search_sequence
            )
            predicted_batch.append(denovo_result)
            # print(candidate_list)
            candidate_list.sort(key=lambda x: (len(sequence), -x.score))
            # print(candidate_list)
            beam_search_result = BeamSearchResult(dda_feature=feature_dp_batch[feature_index].original_dda_feature,
                                                  beam_search_batch=candidate_list)
            beam_search_batch_predicted.append(beam_search_result)
        return predicted_batch,beam_search_batch_predicted

    def _search_denovo_batch(self, feature_dp_batch: list, model_wrapper: InferenceModelWrapper) -> tuple:
        start_time = time.time()
        feature_batch_size = len(feature_dp_batch)
        start_points_tuple = self._get_start_point(feature_dp_batch)
        top_candidate_batch = [[] for x in range(feature_batch_size)]

        for start_points in start_points_tuple:
            beam_search_result_batch = self._beam_search(model_wrapper, feature_dp_batch, start_points)
            # print("beam_search_result_batch: ", beam_search_result_batch)
            for feature_index in range(feature_batch_size):
                top_candidate_batch[feature_index].extend(beam_search_result_batch[feature_index])
        predicted_batch , beam_search_batch_predicted = self._select_path(feature_dp_batch, top_candidate_batch)
        # print("predicted batch: ", predicted_batch)
        # print("beam search batch predicted: ", beam_search_batch_predicted)
        test_time = time.time() - start_time
        logger.info("beam_search(): batch time {}s".format(test_time))
        return predicted_batch, beam_search_batch_predicted

    def search_denovo(self, model_wrapper: InferenceModelWrapper,
                      beam_search_reader: DeepNovoDenovoDataset, denovo_writer: DenovoWriter):
        logger.info("start beam search denovo")
        predicted_denovo_list = []

        test_set_iter = chunks(list(range(len(beam_search_reader))), n=self.args.batch_size)
        total_batch_num = int(len(beam_search_reader) / self.args.batch_size)
        for index, feature_batch_index in enumerate(test_set_iter):
            feature_dp_batch = [beam_search_reader[i] for i in feature_batch_index]
            logger.info("Read {}th/{} batches".format(index, total_batch_num))
            # print("feature_batch_index: ", len(feature_batch_index))
            predicted_batch, beam_search_batch_predicted = self._search_denovo_batch(feature_dp_batch, model_wrapper)
            predicted_denovo_list += predicted_batch
            for denovo_result in predicted_batch:
                denovo_writer.write(denovo_result.dda_feature, denovo_result.best_beam_search_sequence)
            for beamsearch_result in beam_search_batch_predicted:
                denovo_writer.write_beamsearch(beamsearch_result.dda_feature, beamsearch_result.beam_search_batch)
        return predicted_denovo_list

# if __name__ == '__main__':
#     k = KnapsackSearcher(MZ_MAX=6000.0, knapsack_file="./knapsackfile/knapsack_C_M_IL.npy")
