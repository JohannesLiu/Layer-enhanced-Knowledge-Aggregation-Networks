import datetime
import json
import os
import pickle
import sys
import time
import torch
import math



import argparse


class Parser(object):

    def __init__(self, config_path, model , is_training = None):
        self.config = json.loads(open(config_path,'r').read())
        self.is_training = is_training
        self.model = model
        self._trainParser = argparse.ArgumentParser(description ="training-" + model)
        self._testParser = argparse.ArgumentParser(description ="testing-" + model)
        self._oneParser = argparse.ArgumentParser(description ="one-" + model)
        if self.is_training == True:
            self.reset_train_parser()
        elif self.is_training == False:
            self.reset_test_parser()
        else :
            self.reset_one_parser()

    @property
    def trainParser(self):
        return self._trainParser

    @property
    def testParser(self):
        return self._testParser
    @property
    def oneParser(self):
        return self._oneParser

    def reset_train_parser(self):
        # training
        self._trainParser.add_argument('--model', help='neural models to encode sentences', type=str,
                                       default=self.model)
        self._trainParser.add_argument('--use_baseline', help='baseline or hier', type=bool, default=False)
        self._trainParser.add_argument('--mode', help='test mode', type=str, default='pr')
        self._trainParser.add_argument('--gpu', help='gpu(s) to use', type=str, default='0')
        self._trainParser.add_argument('--no_cuda', action='store_true', default=False,
                            help='disables CUDA training')
        self._trainParser.add_argument('--data_path', help ='path to load data', type=str, default='./data/')
        self._trainParser.add_argument('--model_dir', help ='path to store model', type= str, default ='./outputs/ckpt/')
        self._trainParser.add_argument('--summary_dir', help ='path to store summary_dir', type=str, default='./outputs/summary')
        self._trainParser.add_argument('--batch_size', help ='entity numbers used each training time', type= int, default= 160)
        self._trainParser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')


        self._trainParser.add_argument('--max_epoch', help='maximum of training epochs', type=int, default= 40)
        self._trainParser.add_argument('--save_epoch', help='frequency of training epochs', type=int, default=2)
        self._trainParser.add_argument('--restore_epoch', help='epoch to continue training', type=int, default=0)
        self._trainParser.add_argument('--learning_rate', help='learning rate', type=float, default=0.2)
        self._trainParser.add_argument('--weight_decay', help='weight_decay', type=float, default=0.00001)
        self._trainParser.add_argument('--keep_prob', help='dropout rate', type=float, default=0.5)


        self._trainParser.add_argument('--word_size', help='maximum of relations', type=int, default=self.config['word_size'])
        self._trainParser.add_argument('--hidden_size', help='hidden feature size', type=int, default=230)
        self._trainParser.add_argument('--pos_size', help='position embedding size', type=int, default=5)

        # statistics
        self._trainParser.add_argument('--max_length', help='maximum of number of words in one sentence', type=int,
                                       default=self.config['fixlen'])
        self._trainParser.add_argument('--pos_num', help='number of position embedding vectors', type=int,
                                       default=self.config['maxlen']*2 +1)
        self._trainParser.add_argument('--num_classes', help='maximum of relations', type=int,
                                       default=len(self.config['relation2id']))
        self._trainParser.add_argument('--vocabulary_size', help='maximum of relations', type=int,
                                       default=len(self.config['word2id']))

    def reset_test_parser(self):
        # test_settings
        self._testParser.add_argument('--model', help='neural models to encode sentences', type=str, default=self.model)
        self._testParser.add_argument('--use_baseline', help='baseline or hier', type=bool, default=False)
        self._testParser.add_argument('--mode', help='test mode', type=str, default='pr')
        self._testParser.add_argument('--gpu', help='gpu(s) to use', type=str, default='0')
        self._testParser.add_argument('--no_cuda', action='store_true', default=False,
                            help='disables CUDA training')
        self._testParser.add_argument('--allow_growth', help='occupying gpu(s) gradually', type=bool, default=True)
        self._testParser.add_argument('--checkpoint_path', help='path to store model', type=str, default='./outputs/ckpt/')
        self._testParser.add_argument('--logits_path', help='path to store model', type=str, default='./outputs/logits/')
        self._testParser.add_argument('--data_path', help='path to load data', type=str, default='./data/')
        self._testParser.add_argument('--batch_size',
                                      help='instance(entity pair) numbers to use each training(testing) time', type=int,
                                      default=262)
        self._testParser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')

        # training settings
        self._testParser.add_argument('--max_epoch', help='maximum of training epochs', type=int, default=30)
        self._testParser.add_argument('--save_epoch', help='frequency of training epochs', type=int, default=2)
        self._testParser.add_argument('--learning_rate', help='entity numbers used each training time', type=float,
                                      default=0.2)
        self._testParser.add_argument('--weight_decay', help='weight_decay', type=float, default=0.00001)
        self._testParser.add_argument('--keep_prob', help='dropout rate', type=float, default=1.0)

        # test_settings
        self._testParser.add_argument('--test_single', help='only test one checkpoint', type=bool, default=True)
        self._testParser.add_argument('--test_start_ckpt', help='first epoch to test', type=int, default=1)
        self._testParser.add_argument('--test_end_ckpt', help='last epoch to test', type=int, default=30)
        self._testParser.add_argument('--test_sleep', help='time units to sleep ', type=float, default=10)
        self._testParser.add_argument('--test_use_step', help='test step instead of epoch', type=bool, default=False)
        self._testParser.add_argument('--test_start_step', help='first step to test', type=int, default=0 * 1832)
        self._testParser.add_argument('--test_end_step', help='last step to test', type=int, default=30 * 1832)
        self._testParser.add_argument('--test_step', help='step to add per test', type=int, default=1832)

        # parameters
        # self._testParser.add_argument('--word_size', help='maximum of relations', type=int, default=self.config['word_size'])
        self._testParser.add_argument('--word_size', help='maximum of relations', type=int, default=50)
        self._testParser.add_argument('--hidden_size', help='hidden feature size', type=int, default=230)
        self._testParser.add_argument('--pos_size', help='position embedding size', type=int, default=5)

        # statistics
        self._testParser.add_argument('--max_length', help='maximum of number of words in one sentence', type=int,
                                      default=self.config['fixlen'])
        self._testParser.add_argument('--pos_num', help='number of position embedding vectors', type=int,
                                      default=self.config['maxlen']*2+1)
        self._testParser.add_argument('--num_classes', help='maximum of relations', type=int,
                                      default=len(self.config['relation2id']))
        self._testParser.add_argument('--vocabulary_size', help='maximum of relations', type=int,
                                      default=len(self.config['word2id']))

    def reset_one_parser(self):
        #traning
        # overall
        self._oneParser.add_argument('--model', help='neural models to encode sentences', type=str,
                                     default=self.model)
        self._oneParser.add_argument('--use_baseline', help='baseline or hier', type=bool, default=False)
        self._oneParser.add_argument('--mode', help='test mode', type=str, default='pr')
        self._oneParser.add_argument('--gpu', help='gpu(s) to use', type=str, default='0')
        self._oneParser.add_argument('--no_cuda', action='store_true', default=False,
                            help='disables CUDA training')
        self._oneParser.add_argument('--allow_growth', help='occupying gpu(s) gradually', type=bool, default=True)

        self._oneParser.add_argument('--data_path', help ='path to load data', type=str, default='./data/')
        self._oneParser.add_argument('--model_dir', help ='path to store model', type= str, default ='./outputs/ckpt/')
        self._oneParser.add_argument('--summary_dir', help ='path to store summary_dir', type=str, default='./outputs/summary')
        self._oneParser.add_argument('--training_batch_size', help ='entity numbers used each training time', type= int, default= 160)
        self._oneParser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')

        self._oneParser.add_argument('--layer_pattern', help='default, ag-0, ag-1, ag-2', type=str, default='default')

        # training
        self._oneParser.add_argument('--max_epoch', help='maximum of training epochs', type=int, default= 80)
        self._oneParser.add_argument('--save_epoch', help='frequency of training epochs', type=int, default=2)
        self._oneParser.add_argument('--restore_epoch', help='epoch to continue training', type=int, default=0)
        self._oneParser.add_argument('--learning_rate', help='learning rate', type=float, default=0.2)
        self._oneParser.add_argument('--weight_decay', help='weight_decay', type=float, default=0.00001)
        self._oneParser.add_argument('--keep_prob', help='dropout rate', type=float, default=0.5)

        # parameters
        self._oneParser.add_argument('--word_size', help='maximum of relations', type=int, default=self.config['word_size'])
        self._oneParser.add_argument('--hidden_size', help='hidden feature size', type=int, default=230)
        self._oneParser.add_argument('--pos_size', help='position embedding size', type=int, default=5)
        self._oneParser.add_argument('--losses', help='loss_function', type=str, default='cross_entropy')


        # statistics
        self._oneParser.add_argument('--max_length', help='maximum of number of words in one sentence', type=int,
                                     default=self.config['fixlen'])
        self._oneParser.add_argument('--pos_num', help='number of position embedding vectors', type=int,
                                     default=self.config['maxlen']*2 +1)
        self._oneParser.add_argument('--num_classes', help='maximum of relations', type=int,
                                     default=len(self.config['relation2id']))
        self._oneParser.add_argument('--vocabulary_size', help='maximum of relations', type=int,
                                     default=len(self.config['word2id']))

        #testing
        #overall
        self._oneParser.add_argument('--checkpoint_path', help='path to store model', type=str, default='./outputs/ckpt/')
        self._oneParser.add_argument('--logits_path', help='path to store model', type=str, default='./outputs/logits/')
        self._oneParser.add_argument('--testing_batch_size',
                                     help='instance(entity pair) numbers to use each training(testing) time', type=int,
                                     default=262)

        # test_settings
        self._oneParser.add_argument('--test_single', help='only test one checkpoint', type=bool, default=True)
        self._oneParser.add_argument('--test_start_ckpt', help='first epoch to test', type=int, default=1)
        self._oneParser.add_argument('--test_end_ckpt', help='last epoch to test', type=int, default=30)
        self._oneParser.add_argument('--test_sleep', help='time units to sleep ', type=float, default=10)
        self._oneParser.add_argument('--test_use_step', help='test step instead of epoch', type=bool, default=False)
        self._oneParser.add_argument('--test_start_step', help='first step to test', type=int, default=0 * 1832)
        self._oneParser.add_argument('--test_end_step', help='last step to test', type=int, default=30 * 1832)
        self._oneParser.add_argument('--test_step', help='step to add per test', type=int, default=1832)

if __name__=="__main__":
    args = Parser("./data/config", "trials")
    trainParser = args.trainParser
    testParser = args.testParser
    oneParser = args.oneParser
    for key in args.__dict__:
        print(f"{key}:{args.__dict__[key]}")