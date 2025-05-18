import argparse
import configparser
from builtins import print


def init_args(file):
    parser = argparse.ArgumentParser()
    config = configparser.ConfigParser()
    print(file)
    config.read(file)
    # add train params
    parser.add_argument("--engine_model", default=config["train"]["engine_model"], type=int)
    parser.add_argument("--train_dir", default=config["train"]["train_dir"], type=str)
    parser.add_argument("--num_workers", default=config["train"]["num_workers"], type=int)
    parser.add_argument("--batch_size", default=config["train"]["batch_size"], type=int)
    parser.add_argument("--num_epoch", default=config["train"]["num_epoch"], type=int)
    parser.add_argument("--init_lr", default=config["train"]["init_lr"], type=float)
    parser.add_argument("--steps_per_validation", default=config["train"]["steps_per_validation"], type=int)
    parser.add_argument("--weight_decay", default=config["train"]["weight_decay"], type=float)
    parser.add_argument("--MAX_NUM_PEAK", default=config["train"]["MAX_NUM_PEAK"], type=int)
    parser.add_argument("--MZ_MAX", default=config["train"]["MZ_MAX"], type=float)
    parser.add_argument("--MAX_LEN", default=config["train"]["MAX_LEN"], type=int)
    parser.add_argument("--num_ions", default=config["train"]["num_ions"], type=int)

    # add model params
    parser.add_argument("--input_dim", default=config["model"]["input_dim"], type=int)
    parser.add_argument("--output_dim", default=config["model"]["output_dim"], type=int)
    parser.add_argument("--n_classes", default=config["model"]["n_classes"], type=int)
    parser.add_argument("--units", default=config["model"]["units"], type=int)
    parser.add_argument("--dropout", default=config["model"]["dropout"], type=float)

    parser.add_argument("--beam_size", default=config["search"]["beam_size"], type=int)
    parser.add_argument("--knapsack", default=config["search"]["knapsack"], type=str)

    # add data params
    parser.add_argument("--train_try_spectrum_path", default=config["data"]["train_try_spectrum_path"], type=str)
    parser.add_argument("--train_feature_path", default=config["data"]["train_feature_path"], type=str)
    parser.add_argument("--valid_try_spectrum_path", default=config["data"]["valid_try_spectrum_path"], type=str)
    parser.add_argument("--valid_feature_path", default=config["data"]["valid_feature_path"], type=str)
    parser.add_argument("--train_lys_spectrum_path", default=config["data"]["train_lys_spectrum_path"], type=str)
    parser.add_argument("--valid_lys_spectrum_path", default=config["data"]["valid_lys_spectrum_path"], type=str)

    parser.add_argument("--denovo_input_spectrum_file", default=config["data"]["denovo_input_spectrum_file"], type=str)
    parser.add_argument("--denovo_input_mirror_spectrum_file", default=config["data"]["denovo_input_mirror_spectrum_file"], type=str)
    parser.add_argument("--denovo_input_feature_file", default=config["data"]["denovo_input_feature_file"], type=str)
    parser.add_argument("--denovo_output_file", default=config["data"]["denovo_output_file"], type=str)
    parser.add_argument("--accuracy_file", default=config["data"]["denovo_output_file"] + "_model", type=str)
    parser.add_argument("--denovo_only_file", default=config["data"]["denovo_output_file"] + ".denovo_only", type=str)
    parser.add_argument("--scan2fea_file", default=config["data"]["denovo_output_file"] + ".scan2fea", type=str)
    parser.add_argument("--multifea_file", default=config["data"]["denovo_output_file"] + ".multifea", type=str)

    parser.add_argument("--pnovom_path", default=config["calculate"]["pnovom_path"], type=str)
    parser.add_argument("--startswith", default=config["calculate"]["startswith"], type=str)
    parser.add_argument("--topk", default=config["calculate"]["topk"], type=int)

    args = parser.parse_args()

    return args



