import os
import sys


# work_root = "/home/liuxk/PycharmProjects/LeKAN"
# work_root = "D:/PycharmProjects/LeKAN/"

import platform
if 'Windows' in platform.platform():
    work_root = "D:/PycharmProjects/LeKAN"
else:
    work_root = "/home/liuxk/PycharmProjects/LeKAN"

os.chdir(work_root)
sys.path.append("./")
sys.path.append("./plot")
sys.path.append("./")



import logging
import torch.nn as nn
import numpy as np

from lekankit.networks.encoders import SentenceEncoder
from lekankit.networks.models import Model
from lekankit.config import *
from lekankit.dataloaders import WordVec, LoadNYT, LoadHierData
from lekankit.frameworks import Trainer

from torch import optim

project_name = 'LeKAN'

logger = logging.getLogger(project_name)

def main(args):
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if use_cuda else "cpu")

    print('torch.cuda.available:{}'.format(torch.cuda.is_available()))
    print('args["no_cuda"]:{}'.format(args['no_cuda']))
    print("use_cuda:{}".format(use_cuda))
    print("device:{}".format(device))

    training_batch_size = args['training_batch_size']
    learning_rate = args['learning_rate']
    weight_decay = args['weight_decay']
    keep_prob = args['keep_prob']
    word_size = args['word_size']
    pos_size = args['pos_size']
    hidden_size = args['hidden_size']

    save_file_suffix = str("-tb_" + str(training_batch_size) + "-lr_" + str(learning_rate) + "-weight_decay_"
                           + str(weight_decay) + "-keep_prob_" + str(keep_prob) + "-pos_size_" + str(
        pos_size) + "-hidden_size_" + str(hidden_size))

    pattern = "ag-0"

    HierDataLoader = LoadHierData.HierDataLoader(workdir=os.getcwd(), pattern=pattern, device=device)
    relation_levels_Tensor = HierDataLoader.relation_levels_Tensor.to(device)
    relation_level_layer = HierDataLoader.relation_level_layer
    trainDataLoader = LoadNYT.NYTTrainDataLoader(device=device)
    testDataLoader = LoadNYT.NYTTestDataLoader(mode="pr", device=device)
    SkipGramVec = WordVec.SkipGram(data_path='./data/').SkipGramVec

    encoderName = "cnn"
    if encoderName == "cnn":
        sentence_encoder = SentenceEncoder.CNNSentenceEncoder(SkipGramVec, 120, args['pos_num'], word_size, pos_size,
                                                              hidden_size)
    elif encoderName == "pcnn":
        sentence_encoder = SentenceEncoder.PCNNSentenceEncoder(SkipGramVec, 120, args['pos_num'], word_size, pos_size,
                                                               hidden_size)

    model = Model.LeKAN(sentence_encoder = sentence_encoder,
                        relation_levels = relation_levels_Tensor,
                        relation_level_layer = relation_level_layer,
                        keep_prob = keep_prob,
                        train_batch_size= training_batch_size,
                        test_batch_size = args['testing_batch_size'],
                        num_classes = args['num_classes'],
                        device = device).to(device)


    criterion = nn.CrossEntropyLoss().to(device)
    parameters_to_optimize = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters_to_optimize,
                          learning_rate,
                          weight_decay=weight_decay)

    trainer = Trainer.Trainer(model=model,
                              NYTTrainDataLoader=trainDataLoader,
                              NYTTestDataLoader=testDataLoader,
                              args=args,
                              epoch=0,
                              training_batch_size=training_batch_size,
                              criterion=criterion,
                              optimizer=optimizer, learning_rate=learning_rate,
                              device=device)

    best_auc = 0
    for epoch in range(0, 60):
        losses, na_acc, not_na_acc = trainer.train_epoch()
        auc, mi_ma_100, mi_ma_200, pr, exclude_na_flatten_output = trainer.eval_epoch()
        trainer.epoch = trainer.epoch + 1
        result = {"default": auc,
                  "mi_100_10": mi_ma_100['mi_10'],
                  "mi_100_15": mi_ma_100['mi_15'],
                  "mi_100_20": mi_ma_100['mi_20'],
                  "ma_100_10": mi_ma_100['ma_10'],
                  "ma_100_15": mi_ma_100['ma_15'],
                  "ma_100_20": mi_ma_100['ma_20'],
                  "mi_200_10": mi_ma_200['mi_10'],
                  "mi_200_15": mi_ma_200['mi_15'],
                  "mi_200_20": mi_ma_200['mi_20'],
                  "ma_200_10": mi_ma_200['ma_10'],
                  "ma_200_15": mi_ma_200['ma_15'],
                  "ma_200_20": mi_ma_200['ma_20'],
                  "pr_m": pr['m'],
                  "pr_M": pr['M'],
                  "losses": losses,
                  "na_acc": na_acc,
                  "not_na_acc": not_na_acc
                  }

        if best_auc < auc:
            best_auc = auc
            best_result = result
            if auc > 0.40:
                torch.save(trainer.model, args['model_dir']  + encoderName  + "-auc-" + str(
                    auc) + save_file_suffix + '.pkl')
                torch.save(trainer.model.state_dict(),
                           args['model_dir']  + encoderName  + "-auc-" + str(auc) + "-params" + save_file_suffix + '.pkl')
                np.save(args['logits_path']  +  encoderName  + "-auc-" + str(auc) + "-params" + save_file_suffix, exclude_na_flatten_output)

def get_params():
    # Training settings
    parser = Parser(work_root + "/data/config", "LeKAN")
    oneParser = parser.oneParser
    args, _ = oneParser.parse_known_args(args=[])
    return args

if __name__ == '__main__':
    '''@nni.get_next_parameter()'''
    try:
        params = vars(get_params())
        # get parameters form tuner
        # tuner_params = nni.get_next_parameter()
        # logger.debug(tuner_params)
        # params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
