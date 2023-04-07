import numpy as np
import torch

class NYTDataLoader(object):
    def __init__(self):
        super(NYTDataLoader, self).__init__()
        self.relationVectorFile_NYT = "./raw_data/TransEEmbeddings/NYT-10/relationVector.txt"
        self.data_path = "./data/"

class NYTTrainDataLoader(NYTDataLoader):
    def __init__(self, device= 'cpu'):
        super(NYTTrainDataLoader, self).__init__()
        self.num_classes = 53
        self.instance_triple = np.load(self.data_path + 'train_instance_triple.npy')
        self.instance_scope = np.load(self.data_path + 'train_instance_scope.npy')
        self.len = np.load(self.data_path + 'train_len.npy')
        self.label = np.load(self.data_path + 'train_label.npy')
        self.word = np.load(self.data_path + 'train_word.npy')
        self.pos1 = np.load(self.data_path + 'train_pos1.npy')
        self.pos2 = np.load(self.data_path + 'train_pos2.npy')
        self.mask = np.load(self.data_path + 'train_mask.npy')
        self.instance_scope_Tensor = torch.LongTensor(self.instance_scope).to(device)
        self.len_Tensor = torch.LongTensor(self.len).to(device)
        self.label_Tensor = torch.LongTensor(self.label).to(device)
        self.word_Tensor = torch.LongTensor(self.word).to(device)
        self.pos1_Tensor = torch.LongTensor(self.pos1).to(device)
        self.pos2_Tensor = torch.LongTensor(self.pos2).to(device)
        self.mask_Tensor = torch.LongTensor(self.mask).to(device)

class NYTTestDataLoader(NYTDataLoader):
    def __init__(self, mode, device= 'cpu'):
        super(NYTTestDataLoader, self).__init__()
        self._mode = mode
        if self._mode == 'pr' or self._mode == 'hit_k_100' or self._mode == 'hit_k_200':
            self.instance_triple = np.load(self.data_path + 'test_entity_pair.npy')
            self.instance_scope = np.load(self.data_path + 'test_entity_scope.npy')
            self.len = np.load(self.data_path + 'test_len.npy')
            self.label = np.load(self.data_path + 'test_label.npy')
            self.word = np.load(self.data_path + 'test_word.npy')
            self.pos1 = np.load(self.data_path + 'test_pos1.npy')
            self.pos2 = np.load(self.data_path + 'test_pos2.npy')
            self.mask = np.load(self.data_path + 'test_mask.npy')
            self.exclude_na_flatten_label = np.load(self.data_path + 'all_true_label.npy')
        else:
            self.instance_triple = np.load(self.data_path + 'pn/test_entity_pair_pn.npy')
            self.instance_scope = np.load(self.data_path + 'pn/test_entity_scope_' + mode + '.npy')
            self.len = np.load(self.data_path + 'pn/test_len_' + mode + '.npy')
            self.label = np.load(self.data_path + 'pn/test_label_' + mode + '.npy')
            self.word = np.load(self.data_path + 'pn/test_word_' + mode + '.npy')
            self.pos1 = np.load(self.data_path + 'pn/test_pos1_' + mode + '.npy')
            self.pos2 = np.load(self.data_path + 'pn/test_pos2_' + mode + '.npy')
            self.mask = np.load(self.data_path + 'pn/test_mask_' + mode + '.npy')
            self.exclude_na_flatten_label = np.load(self.data_path + 'pn/true_label.npy')
        self.instance_scope_Tensor = torch.LongTensor(self.instance_scope).to(device)
        self.len_Tensor = torch.LongTensor(self.len).to(device)
        self.label_Tensor = torch.LongTensor(self.label).to(device)
        self.word_Tensor = torch.LongTensor(self.word).to(device)
        self.pos1_Tensor = torch.LongTensor(self.pos1).to(device)
        self.pos2_Tensor = torch.LongTensor(self.pos2).to(device)
        self.mask_Tensor = torch.LongTensor(self.mask).to(device)
        self.exclude_na_flatten_label_Tensor = torch.LongTensor(self.exclude_na_flatten_label).to(device)

        f = open("raw_data/relation2id.txt", "r")
        content = f.readlines()[1:]
        self.id2rel = {}
        for i in content:
            rel, rid = i.strip().split()
            self.id2rel[(int)(rid)] = rel
        f.close()

        self.fewrel_100 = {}
        f = open("data/rel100.txt", "r")
        content = f.readlines()
        for i in content:
            self.fewrel_100[i.strip()] = 1
        f.close()

        self.fewrel_200 = {}
        f = open("data/rel200.txt", "r")
        content = f.readlines()
        for i in content:
            self.fewrel_200[i.strip()] = 1
        f.close()
    @property
    def mode(self):
        return self._mode