# Copyright 2019 Rui Qiao. All Rights Reserved.
#
# DeepNovoV2 is publicly available for non-commercial uses.
# ==============================================================================
import datetime
import time
import torch
import logging
import logging.config
import config
import os
from train_func1 import train, build_model, validation, perplexity
from data_loader import DeepNovoDenovoDataset, collate_func, DeepNovoTrainDataset
from model_gru import InferenceModelWrapper
from denovo import IonCNNDenovo
from writer import DenovoWriter
from init_args import init_args
from Recall_Precision import compareFeature

logger = logging.getLogger(__name__)


def engine_1(args):
    # train + search denovo + test
    logger.info(f"training mode")
    torch.cuda.empty_cache()
    train(args=args)
    """
    search denovo
    """
    engine_2(args)



def engine_2(args):
    # search denovo + test
    """
    search denovo
    """
    torch.cuda.empty_cache()
    start = time.time()
    logger.info("denovo mode")
    data_reader = DeepNovoDenovoDataset(spectrum_path=args.denovo_input_spectrum_file,
                                        mirror_spectrum_path=args.denovo_input_mirror_spectrum_file,
                                        feature_path=args.denovo_input_feature_file,
                                        args=args)
    denovo_worker = IonCNNDenovo(args=args)
    # forward_deepnovo, backward_deepnovo, init_net = build_model(training=False)
    # model_wrapper = InferenceModelWrapper(forward_deepnovo, backward_deepnovo, init_net)
    forward_deepnovo, backward_deepnovo = build_model(args=args, training=False)
    model_wrapper = InferenceModelWrapper(forward_deepnovo, backward_deepnovo)
    writer = DenovoWriter(args=args)
    denovo_worker.search_denovo(model_wrapper, data_reader, writer)
    torch.cuda.empty_cache()
    logger.info(f'using time:{time.time() - start}')

    '''
    calculate accuracy
    '''
    # engine_3(args)
#
# def engine_3(args):
#     # test
#     logger.info("calculate accuracy")
#     compareFeature(
#         mirror_path=args.denovo_input_feature_file,
#         MirrorNovo_res_path=args.denovo_output_file+".beamsearch",
#         pNovoM_res_path=args.pnovom_path,
#         title=args.startswith,
#         top=args.topk
#     )

def init_log(log_file_name):
    d = {
        'version': 1,
        'disable_existing_loggers': False,  # this fixes the problem
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': log_file_name,
                'mode': 'w',
                'formatter': 'standard',
            }
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        }
    }
    logging.config.dictConfig(d)


if __name__ == '__main__':
    param_path = [
        "./Params/B1-TNet_GRU_Beam10_PEAKS500_T20PPM[E50M30T4V60Y999][Yeast].cfg",
    ]
    log_path = "./log/"
    if isinstance(param_path,list):
        print(param_path)
        for _param_path in param_path:
            dir, param_file = os.path.split(_param_path)
            # log_file_name = "top5_" + param_file[-4] + ".log"
            now = datetime.datetime.now().strftime("%Y%m%d%H%M")
            args = init_args(_param_path)
            # log_file_name = "./log/" + now + "(" + str(args.engine_model) + ").log"
            log_file_name = log_path + param_file + now + "(" + str(args.engine_model) + ").log"
            init_log(log_file_name=log_file_name)
            if os.path.exists(args.train_dir):
                pass
            else:
                os.makedirs(args.train_dir)
            if args.engine_model == 1:
                # print("engine model 1")
                engine_1(args=args)
            elif args.engine_model == 2:
                engine_2(args=args)
            # elif args.engine_model == 3:
            #     engine_3(args=args)
    elif os.path.isfile(param_path):
        dir, param_file = os.path.split(param_path)
        # log_file_name = "top5_" + param_file[-4] + ".log"
        now = datetime.datetime.now().strftime("%Y%m%d%H%M")
        args = init_args(param_path)
        # log_file_name = "./log/" + now + "(" + str(args.engine_model) + ").log"
        log_file_name = log_path + now + "(" + str(args.engine_model) + ").log"
        init_log(log_file_name=log_file_name)
        if os.path.exists(args.train_dir):
            pass
        else:
            os.makedirs(args.train_dir)
        if args.engine_model == 1:
            # print("engine model 1")
            engine_1(args=args)
        elif args.engine_model == 2:
            engine_2(args=args)
        # elif args.engine_model == 3:
        #     engine_3(args=args)
    elif os.path.isdir(param_path):
        for file in os.listdir(param_path):
            one_param_path = os.path.join(param_path, file)
            if os.path.isfile(one_param_path):
                now = datetime.datetime.now().strftime("%Y%m%d%H%M")
                args = init_args(param_path)
                log_file_name = log_path + now + "(" + str(args.engine_model) + ").log"
                init_log(log_file_name=log_file_name)
                if os.path.exists(args.train_dir):
                    pass
                else:
                    os.makedirs(args.train_dir)
                if args.engine_model == 1:
                    engine_1(args=args)
                elif args.engine_model == 2:
                    engine_2(args=args)
                # elif args.engine_model == 3:
                #     engine_3(args=args)
