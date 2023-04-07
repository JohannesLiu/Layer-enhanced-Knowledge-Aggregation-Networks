import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from lekankit.networks.models.BaselineModel import BaseAttentionNetwork

class LeKAN(BaseAttentionNetwork):
    def __init__(self, sentence_encoder, relation_levels, relation_level_layer, keep_prob,
                 train_batch_size=None, test_batch_size=262, num_classes=53, device="cuda:0"):
        super(LeKAN, self).__init__(sentence_encoder = sentence_encoder,
                                                   relation_levels = relation_levels,
                                                   relation_level_layer = relation_level_layer,
                                                   keep_prob = keep_prob,
                                                   train_batch_size=train_batch_size,
                                                   test_batch_size=test_batch_size,
                                                   num_classes=num_classes,
                                                   device = device)
        self.w_s_1 = nn.Parameter(torch.Tensor(1, self.hidden_size * 2))
        self.w_s_2 = nn.Parameter(torch.Tensor(1, self.hidden_size * 2))
        self.w_s = [self.w_s_1, self.w_s_2]
        self.b_s = nn.Parameter(torch.Tensor(self.hier, 1))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.hier):
            # self.relation_matrixs.append(nn.Embedding(self.layer[i], self.hidden_size,
            #                                           _weight=nn.init.xavier_uniform_(
            #                                               torch.Tensor(self.layer[i], self.hidden_size))).to(self.device))
            self.relation_matrixs.append(nn.Embedding(self.layer[i], self.hidden_size,
                                                      _weight=nn.init.xavier_uniform_(
                                                          torch.Tensor(self.layer[i], self.hidden_size))).to(self.device))
        nn.init.xavier_uniform_(self.discrimitive_matrix)
        nn.init.zeros_(self.bias)

    def forward(self, data):  # data 包含word, pos1, pos2, mask, label, scope
        x = self.sentence_encoder(data)
        label_layer = self.relation_levels[data['label_index']]
        s_k_q_r = x
        for i in range(self.hier):
            current_relation = self.relation_matrixs[i](label_layer[:, i])  # batch_size * 230
            if i ==0:
                s_k_q_r_0 = torch.cat((s_k_q_r, current_relation), dim=1) #batch_size * 460
            elif i == 1:
                s_k_q_r_1 = torch.cat((s_k_q_r, current_relation), dim=1) # batch_size *460
        attention_logits_stack = torch.cat(( torch.tanh(self.w_s[0] @ s_k_q_r_0.t()), torch.tanh(self.w_s[1] @  s_k_q_r_1.t()))) +self.b_s
        attention_score_hidden = torch.cat([
            F.softmax(attention_logits_stack[:, data['scope'][i]:data['scope'][i + 1]], dim = -1) for i in
            range(self.train_batch_size)], 1)

        tower_repre = []
        for i in range(self.train_batch_size):
            sen_matrix = x[data['scope'][i]:data['scope'][i + 1]]
            layer_score = attention_score_hidden[:,
                          data['scope'][i]:data['scope'][i + 1]]  # 查找Layer_score #(2 ,bag_size)
            layer_repre = torch.reshape(layer_score @ sen_matrix, [-1])  # 获得层次化表示表示  # bagsize*230
            tower_repre.append(layer_repre)  # 获得每个句子的表示 (batchsize, bagsize*230 )

        stack_repre = self.drop(torch.stack(tower_repre))  # 获得新的表示
        logits = stack_repre @ self.discrimitive_matrix.t() + self.bias  # sen_num * 230 matmul 230 * 53 + 53
        return logits

    def forward_infer(self, data):
        x = self.sentence_encoder(data)  # batch_size * 230

        s_k_q_r = torch.unsqueeze(x, 0).repeat(53, 1, 1)
        for i in range(self.hier):
            current_relation = self.relation_matrixs[i](self.relation_levels[:, i])  # 53 * 230
            current_relation_squeeze = torch.unsqueeze(current_relation, 1).repeat(1, x.shape[0], 1)
            if i == 0:
                s_k_q_r_0 = torch.cat((s_k_q_r, current_relation_squeeze), dim=2) # 53 * batch_size * 690
            elif i == 1:
                s_k_q_r_1 = torch.cat((s_k_q_r, current_relation_squeeze), dim=2) # 53 * batch_size * 690

        current_logit = torch.cat((torch.tanh(self.w_s[0] @ s_k_q_r_0.permute(0, 2, 1)), torch.tanh(self.w_s[1] @ s_k_q_r_1.permute(0, 2, 1))), dim = 1) + self.b_s # 2 * 690 @  53 * 690 * batch_size = 53 * 2 * batch_size

        h1_att = torch.cat([F.softmax(current_logit[:, 0, data['scope'][j]:data['scope'][j + 1]], dim = -1) for j in
                            range(self.test_batch_size)], dim=1)
        h2_att = torch.cat([F.softmax(current_logit[:, 1, data['scope'][j]:data['scope'][j + 1]], dim = -1) for j in
                            range(self.test_batch_size)], dim=1)

        test_attention_scores_stack = torch.stack((h1_att, h2_att), dim=1)

        test_tower_output = []
        for i in range(self.test_batch_size):
            test_sen_matrix = (torch.unsqueeze(x[data['scope'][i]:data['scope'][i + 1]], 0)).repeat(53, 1,
                                                                                                    1)  # 先将 x[data['scope']] 扩充维度形成  53 * bag_size * 230， 然后在第一维重复53次
            test_layer_score = test_attention_scores_stack[:, :,
                               data['scope'][i]:data['scope'][i + 1]]  # 查找Layer_score #(53, 2 ,bag_size)
            test_layer_repre = torch.reshape(test_layer_score @ test_sen_matrix, [self.num_classes,
                                                                                  -1])  # 获得层次化表示表示  # 53 * 2 * bag_size @ 53* bag_size *230, 53* 2 *230 ,reshape 53 *460
            test_logits = test_layer_repre @ self.discrimitive_matrix.t() + self.bias  # 53 * 460 @ 460 * 53 +   (53, )
            test_output = torch.diagonal(F.softmax(test_logits, dim = -1))  # 获得对角输出
            test_tower_output.append(test_output)

        test_stack_output = torch.reshape(torch.stack(test_tower_output), [self.test_batch_size, self.num_classes])
        return test_stack_output
