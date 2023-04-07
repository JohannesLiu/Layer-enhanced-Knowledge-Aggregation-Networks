import pickle

import numpy as np
import pandas as pd
import torch

class HierDataLoader(object):
    """
    default: load the hatt solutions
    ag-0: load the aggregation solutions
    """
    def __init__(self, workdir, pattern = "default", device = "cuda:0"):
        super(HierDataLoader, self).__init__()
        self._pattern = pattern
        self._init_vec = pickle.load(open(str(workdir+'/data/initial_vectors/init_vec').replace("\\", "/"), 'rb'))
        self.device = device

        if self._pattern == "default":
            self.relation_levels_Tensor = torch.LongTensor(self._init_vec['relation_levels'])
            self.relation_levels_pd = pd.DataFrame(self._init_vec['relation_levels'], columns=['p_index', 'index'])
            self.relation_levels_np = self.relation_levels_pd.to_numpy()
            self.relation_level_layer = (1 + np.max(self._init_vec['relation_levels'], 0)).astype(np.int32)
        elif self._pattern =="ag-0":
            self.relation_levels_pd = pd.read_csv("./raw_data/ag_dp/ag-0-relation2id.csv", usecols=['p_s_0', 'id'])
            self.relation_levels_np = self.relation_levels_pd.to_numpy()
            self.relation_levels_Tensor = torch.LongTensor(self.relation_levels_np)
            self.relation_level_layer = (1 + np.max(self.relation_levels_np, 0))
            self.bottom_weight_np = pd.read_csv("./raw_data/ag_dp/ag-0-bottom_weight.csv")[['weight_test']].to_numpy().reshape(
                -1)
            self.top_weight_np = pd.read_csv("./raw_data/ag_dp/ag-0-top_weight.csv")[['weight_test']].to_numpy().reshape(-1)
            self.bottom_weight_Tensor = torch.from_numpy(self.bottom_weight_np).float().to(self.device)
            self.top_weight_Tensor = torch.from_numpy(self.top_weight_np).float().to(self.device)
            self.attention_weight = []
            self.attention_weight.append(self.top_weight_Tensor)
            self.attention_weight.append(self.bottom_weight_Tensor)
            relation_matrix_0 = torch.nn.init.xavier_uniform_(torch.rand(self.relation_level_layer[0], 230))
            relation_matrix_1 = torch.nn.init.xavier_uniform_(torch.rand(self.relation_level_layer[1], 230))
            self.relation_matrixs = [relation_matrix_0, relation_matrix_1]
            self.long_tail = []
            self.normal_body = []
            self.short_head = []
        else:
            raise Exception("没有对应的范式!")

    @property
    def pattern(self):
        return self._pattern