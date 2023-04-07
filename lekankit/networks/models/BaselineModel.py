import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

class BaseAttentionNetwork(nn.Module):
    def __init__(self, sentence_encoder, relation_levels, relation_level_layer, keep_prob,
                 train_batch_size=None, test_batch_size=262, num_classes=53, device = "cuda:0"):
        '''
        Pay Attention!
        relation_matrix:
        id = 0: virtual node
        id = 1: NA node
        '''
        super(BaseAttentionNetwork, self).__init__()
        self.keep_prob = keep_prob
        self.hidden_size = sentence_encoder.hidden_size
        self.relation_levels = relation_levels
        self.sentence_encoder = sentence_encoder

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.hier = relation_levels.shape[1]
        self.layer = relation_level_layer
        self.num_classes = num_classes
        self.relation_matrixs = []
        self.device = device

        self.discrimitive_matrix = nn.Parameter(torch.Tensor(num_classes, self.hidden_size * 2))
        self.bias = torch.nn.Parameter(torch.Tensor(53))
        self.drop = nn.Dropout(1 - self.keep_prob)
        self.long_tail = []
        self.normal_body = []
        self.short_head = []

    def reset_parameters(self):
        NotImplemented
    def forward(self):
        NotImplemented
    def forward_infer(self):
        NotImplemented
    def query_func(self, attention_weight, label):
        NotImplemented

class HAttentionNetwork(BaseAttentionNetwork):
    def __init__(self, sentence_encoder, relation_levels, relation_level_layer, keep_prob,
                 train_batch_size=None, test_batch_size=262, num_classes=53, device = "cuda:0"):
        '''
        Pay Attention!
        relation_matrix:
        id = 0: virtual node
        id = 1: NA node
        '''
        super(HAttentionNetwork, self).__init__(sentence_encoder = sentence_encoder,
                                                   relation_levels = relation_levels,
                                                   relation_level_layer = relation_level_layer,
                                                   keep_prob = keep_prob,
                                                   train_batch_size=train_batch_size,
                                                   test_batch_size=test_batch_size,
                                                   num_classes=num_classes,
                                                   device = device)
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.hier):
            self.relation_matrixs.append(nn.Embedding(self.layer[i], self.hidden_size, _weight=nn.init.xavier_uniform_(
                torch.Tensor(self.layer[i], self.hidden_size))).to(self.device))
        nn.init.xavier_uniform_(self.discrimitive_matrix)
        nn.init.zeros_(self.bias)

    def forward(self, data):  # data 包含word, pos1, pos2, mask, label, scope
        x = self.sentence_encoder(data)

        label_layer = self.relation_levels[data['label_index']]
        attention_logits = []
        for i in range(self.hier):
            current_relation = self.relation_matrixs[i](label_layer[:, i])  # batch * 230
            attention_logits.append(torch.sum(current_relation * x, 1))  # batch*230 x batch*230
        #             print("torch.sum(current_relation * x, 1):" ,torch.sum(current_relation * x, 1).shape)

        attention_logits_stack = torch.stack(attention_logits)  # 将一个batch的结果堆叠起来   2 * batch_size
        #         print("attention_logits stack shape:", attention_logits_stack.shape)

        attention_score_hidden = torch.cat([
            F.softmax(attention_logits_stack[:, data['scope'][i]:data['scope'][i + 1]], dim = -1) for i in
            range(self.train_batch_size)], 1)  ###这段出了问题

        tower_repre = []
        for i in range(self.train_batch_size):
            sen_matrix = x[data['scope'][i]:data['scope'][i + 1]]
            #             print("sen_matrix shape: ",sen_matrix.shape)
            layer_score = attention_score_hidden[:,
                          data['scope'][i]:data['scope'][i + 1]]  # 查找Layer_score #(2 ,bag_size)
            #             print("layer_score shape: ",layer_score.shape)
            layer_repre = torch.reshape(layer_score @ sen_matrix, [-1])  # 获得层次化表示表示  # 2 * batch_size @ batch_size *230
            #             print("layer_score @ sen_matrix shape: ", (layer_score @ sen_matrix).shape) #(2, 230)
            tower_repre.append(layer_repre)  # 获得每个句子的表示  append (-1, 460)
        #             print("layer_repre shape: ", (layer_repre).shape)

        stack_repre = self.drop(torch.stack(tower_repre))  # 获得新的表示
        logits = stack_repre @ self.discrimitive_matrix.t() + self.bias  # sen_num * 230 matmul 230 * 53 + 53
        return logits

    def forward_infer(self, data):
        x = self.sentence_encoder(data)  # batch_size * 230

        test_attention_scores = []
        for i in range(self.hier):
            current_relation = self.relation_matrixs[i](self.relation_levels[:, i])  # 53 * 230
            current_logit = current_relation @ x.t()  # 53 * batch_size
            current_score = torch.cat([F.softmax(current_logit[:, data['scope'][j]:data['scope'][j + 1]], dim = -1) for j in
                                       range(self.test_batch_size)], 1)  ##得到每一个袋子的attention_score
            test_attention_scores.append(
                current_score)  # curr_relation_num * batch_size   堆叠起来，形成两个高度不同的attention_score: 53 * batch_size
        #             print(current_relation)
        #             print("torch.sum(current_relation * x, 1):" ,torch.sum(current_relation * x, 1).shape)

        test_attention_scores_stack = torch.stack(test_attention_scores, 1)  # 将一个batch的结果堆叠起来: 53* 2* batch_size
        #         print("attention_logits stack shape:", attention_logits_stack.shape)

        test_tower_output = []
        for i in range(self.test_batch_size):
            test_sen_matrix = (torch.unsqueeze(x[data['scope'][i]:data['scope'][i + 1]], 0)).repeat(53, 1,
                                                                                                    1)  # 先将 x[data['scope']] 扩充维度形成  53 * bag_size * 230， 然后在第一维重复53次
            #             print("sen_matrix shape: ",sen_matrix.shape)
            test_layer_score = test_attention_scores_stack[:, :,
                               data['scope'][i]:data['scope'][i + 1]]  # 查找Layer_score #(53, 2 ,bag_size)
            #             print("layer_score shape: ",layer_score.shape)
            test_layer_repre = torch.reshape(test_layer_score @ test_sen_matrix, [self.num_classes,
                                                                                  -1])  # 获得层次化表示表示  # 53 * 2 * bag_size @ 53* bag_size *230, 53* 2 *230 ,reshape 53 *460
            #             print("layer_score @ sen_matrix shape: ", (layer_score @ sen_matrix).shape) #(2, 230) (r_(h, t)^1), (r_(h, t)^2)
            #             print("layer_repre shape: ", (layer_repre).shape)
            test_logits = test_layer_repre @ self.discrimitive_matrix.t() + self.bias  # 53 * 460 @ 460 * 53 +   (53, )
            test_output = torch.diagonal(F.softmax(test_logits, dim = -1))
            test_tower_output.append(test_output)

        test_stack_output = torch.reshape(torch.stack(test_tower_output), [self.test_batch_size, self.num_classes])
        return test_stack_output


class KAttentionNetwork(BaseAttentionNetwork):
    def __init__(self, sentence_encoder,  relation_levels, relation_level_layer, attention_weight, relation_matrixs, keep_prob,
                 train_batch_size=262, test_batch_size=None, num_classes=53, device = "cuda:0"):
        '''
        Pay Attention!
        relation_matrix:
        id = 0: virtual node
        id = 1: NA node
        '''
        super(KAttentionNetwork, self).__init__(sentence_encoder = sentence_encoder,
                                                   relation_levels = relation_levels,
                                                   relation_level_layer = relation_level_layer,
                                                   keep_prob = keep_prob,
                                                   train_batch_size=train_batch_size,
                                                   test_batch_size=test_batch_size,
                                                   num_classes=num_classes,
                                                   device = device)

        self.relation_matrixs = []
        self.attention_weight = attention_weight

        self.w_s = nn.Parameter(torch.Tensor(self.hier, self.hidden_size * 3))
        self.b_s = nn.Parameter(torch.Tensor(self.hier, 1))

        self.reset_parameters(relation_matrixs)

    def reset_parameters(self, relation_matrixs):
        for i in range(self.hier):
            self.relation_matrixs.append(nn.Embedding(self.layer[i], self.hidden_size, _weight=relation_matrixs[i]).to(self.device))
        nn.init.xavier_uniform_(self.discrimitive_matrix)
        nn.init.zeros_(self.bias)

    def forward(self, data):  # data 包含word, pos1, pos2, mask, label, scope
        x = self.sentence_encoder(data)
        label_layer = self.relation_levels[data['label_index']]
        s_k_q_r = x #total_size * 230
        vertical_attention_logits = []
        for i in range(self.hier):
            current_relation = self.relation_matrixs[i](label_layer[:, i])  # total_size * 230
            s_k_q_r = torch.cat((s_k_q_r, current_relation), dim=-1)  # total_size * 690
            current_ver_attention = self.attention_weight[i][label_layer[:, i]]
            vertical_attention_logits.append(current_ver_attention) # [total_size, total_size]

        attention_logits_stack = self.w_s @ torch.tanh(s_k_q_r).t() + self.b_s #  2* 690  @ 690 * total_sie = 2 * total_size
        attention_score_hidden = torch.cat([
            F.softmax(attention_logits_stack[:, data['scope'][i]:data['scope'][i + 1]], dim = -1) for i in
            range(self.train_batch_size)], 1)  ###这段出了问题 #

        vertical_attention_logits_stack = torch.stack(vertical_attention_logits)  # 2* toltal_size


        tower_repre = []
        for i in range(self.train_batch_size):
            sen_matrix = x[data['scope'][i]:data['scope'][i + 1]]
            #             print("sen_matrix shape: ",sen_matrix.shape)
            layer_score = attention_score_hidden[:,
                          data['scope'][i]:data['scope'][i + 1]]  # 查找Layer_score #(2 ,bag_size)
            layer_attention_logits = vertical_attention_logits_stack[:, data['scope'][i]:data['scope'][i + 1]].permute( 1, 0)  # (bag_size, 2)
            layer_attention_score = torch.mean(F.softmax(layer_attention_logits, dim=-1).permute(1, 0), dim=1).reshape(2,1) #(2 * 1)
            layer_repre = torch.reshape( layer_attention_score * (layer_score @ sen_matrix), [-1])  # 获得层次化表示表示  # 2* 690 = 2*bag_size @ bag_size* 690
            #             print("layer_score @ sen_matrix shape: ", (layer_score @ sen_matrix).shape) #(2, 230) (r_(h, t)^1), (r_(h, t)^2)
            tower_repre.append(layer_repre)  # 获得每个句子的表示 (batchsize, bagsize*230 )
        #             print("layer_repre shape: ", (layer_repre).shape)

        stack_repre = self.drop(torch.stack(tower_repre))  # 获得新的表示
        logits = stack_repre @ self.discrimitive_matrix.t() + self.bias  # sen_num * 230 matmul 230 * 53 + 53
        return logits

    def forward_infer(self, data):
        x = self.sentence_encoder(data)  # total_size * 230

        test_attention_scores = []
        s_k_q_r = x #total_size *30
        s_k_q_r = torch.unsqueeze(s_k_q_r, dim = 0).repeat(self.num_classes, 1, 1)
        test_vertical_attention_logits = []
        for i in range(self.hier):
            current_relation = self.relation_matrixs[i](self.relation_levels[:, i])  # 53 * 230
            current_relation = torch.unsqueeze(current_relation, 1).repeat(1, x.shape[0], 1) # 53 * total_size* 230
            s_k_q_r = torch.cat((s_k_q_r, current_relation), dim = -1) # 53 * total_size * 690

            test_current_ver_attention = self.attention_weight[i][self.relation_levels[:, i]] # 53 *1
            test_vertical_attention_logits.append(test_current_ver_attention) # [53, 53]

        test_attention_score_stack = self.w_s @ torch.tanh(s_k_q_r).permute(0, 2, 1) + self.b_s # 53 * 2 * total_size

        vertical_attention_logits_stack = torch.stack(test_vertical_attention_logits)  # 2* 53
        test_layer_attention_score = torch.unsqueeze(F.softmax(vertical_attention_logits_stack.t()), dim = -1) # 53* 2 * 1
        test_tower_output = []
        for i in range(self.test_batch_size):
            test_sen_matrix = (torch.unsqueeze(x[data['scope'][i]:data['scope'][i + 1]], 0)).repeat(53, 1,
                                                                                                    1)  # 先将 x[data['scope']] 扩充维度形成  53 * bag_size * 230， 然后在第一维重复53次
            test_layer_score = test_attention_score_stack[:, :,
                               data['scope'][i]:data['scope'][i + 1]]  # 查找Layer_score #(53, 2 ,bag_size)

            test_layer_repre = torch.reshape(test_layer_attention_score * (test_layer_score @ test_sen_matrix), [self.num_classes,
                                                                                  -1])  # 获得层次化表示表示  # 53 * 2 * bag_size @ 53* bag_size *230, 53* 2 *230 ,reshape 53 *460

            test_logits = test_layer_repre @ self.discrimitive_matrix.t() + self.bias  # 53 * 460 @ 460 * 53 +   (53, )
            test_output = torch.diagonal(F.softmax(test_logits))  # 获得对角输出
            test_tower_output.append(test_output)

        test_stack_output = torch.reshape(torch.stack(test_tower_output), [self.test_batch_size, self.num_classes])
        return test_stack_output



class KAGttentionNetwork(BaseAttentionNetwork):
    def __init__(self, sentence_encoder,  relation_levels, relation_level_layer, attention_weight, relation_matrixs, keep_prob,
                 train_batch_size=262, test_batch_size=None, num_classes=53, device = "cuda:0"):
        '''
        Pay Attention!
        relation_matrix:
        id = 0: virtual node
        id = 1: NA node
        '''
        super(KAGttentionNetwork, self).__init__(sentence_encoder = sentence_encoder,
                                                   relation_levels = relation_levels,
                                                   relation_level_layer = relation_level_layer,
                                                   keep_prob = keep_prob,
                                                   train_batch_size=train_batch_size,
                                                   test_batch_size=test_batch_size,
                                                   num_classes=num_classes,
                                                   device = device)

        self.relation_matrixs = []
        self.attention_weight = attention_weight

        self.w_s = nn.Parameter(torch.Tensor(self.hier, self.hidden_size * 3))
        self.b_s = nn.Parameter(torch.Tensor(self.hier, 1))

        self.reset_parameters(relation_matrixs)

    def reset_parameters(self, relation_matrixs):
        for i in range(self.hier):
            self.relation_matrixs.append(nn.Embedding(self.layer[i], self.hidden_size, _weight=relation_matrixs[i]).to(self.device))
        nn.init.xavier_uniform_(self.discrimitive_matrix)
        nn.init.zeros_(self.bias)

    def forward(self, data):  # data 包含word, pos1, pos2, mask, label, scope
        x = self.sentence_encoder(data)
        label_layer = self.relation_levels[data['label_index']]
        s_k_q_r = x #total_size * 230
        vertical_attention_logits = []
        for i in range(self.hier):
            current_relation = self.relation_matrixs[i](label_layer[:, i])  # total_size * 230
            s_k_q_r = torch.cat((s_k_q_r, current_relation), dim=-1)  # total_size * 690
            current_ver_attention = self.attention_weight[i][label_layer[:, i]]
            vertical_attention_logits.append(current_ver_attention) # [total_size, total_size]

        attention_logits_stack = self.w_s @ torch.tanh(s_k_q_r).t() + self.b_s #  2* 690  @ 690 * total_sie = 2 * total_size
        attention_score_hidden = torch.cat([
            F.softmax(attention_logits_stack[:, data['scope'][i]:data['scope'][i + 1]], dim = -1) for i in
            range(self.train_batch_size)], 1)  ###这段出了问题 #

        vertical_attention_logits_stack = torch.stack(vertical_attention_logits)  # 2* toltal_size


        tower_repre = []
        for i in range(self.train_batch_size):
            sen_matrix = x[data['scope'][i]:data['scope'][i + 1]]
            #             print("sen_matrix shape: ",sen_matrix.shape)
            layer_score = attention_score_hidden[:,
                          data['scope'][i]:data['scope'][i + 1]]  # 查找Layer_score #(2 ,bag_size)
            layer_attention_logits = vertical_attention_logits_stack[:, data['scope'][i]:data['scope'][i + 1]].permute( 1, 0)  # (bag_size, 2)
            layer_attention_score = torch.mean(F.softmax(layer_attention_logits, dim=-1).permute(1, 0), dim=1).reshape(2,1) #(2 * 1)
            layer_repre = torch.reshape( layer_attention_score * (layer_score @ sen_matrix), [-1])  # 获得层次化表示表示  # 2* 690 = 2*bag_size @ bag_size* 690
            #             print("layer_score @ sen_matrix shape: ", (layer_score @ sen_matrix).shape) #(2, 230) (r_(h, t)^1), (r_(h, t)^2)
            tower_repre.append(layer_repre)  # 获得每个句子的表示 (batchsize, bagsize*230 )
        #             print("layer_repre shape: ", (layer_repre).shape)

        stack_repre = self.drop(torch.stack(tower_repre))  # 获得新的表示
        logits = stack_repre @ self.discrimitive_matrix.t() + self.bias  # sen_num * 230 matmul 230 * 53 + 53
        return logits

    def forward_infer(self, data):
        x = self.sentence_encoder(data)  # total_size * 230

        test_attention_scores = []
        s_k_q_r = x #total_size *30
        s_k_q_r = torch.unsqueeze(s_k_q_r, dim = 0).repeat(self.num_classes, 1, 1)
        test_vertical_attention_logits = []
        for i in range(self.hier):
            current_relation = self.relation_matrixs[i](self.relation_levels[:, i])  # 53 * 230
            current_relation = torch.unsqueeze(current_relation, 1).repeat(1, x.shape[0], 1) # 53 * total_size* 230
            s_k_q_r = torch.cat((s_k_q_r, current_relation), dim = -1) # 53 * total_size * 690

            test_current_ver_attention = self.attention_weight[i][self.relation_levels[:, i]] # 53 *1
            test_vertical_attention_logits.append(test_current_ver_attention) # [53, 53]

        test_attention_score_stack = self.w_s @ torch.tanh(s_k_q_r).permute(0, 2, 1) + self.b_s # 53 * 2 * total_size

        vertical_attention_logits_stack = torch.stack(test_vertical_attention_logits)  # 2* 53
        test_layer_attention_score = torch.unsqueeze(F.softmax(vertical_attention_logits_stack.t()), dim = -1) # 53* 2 * 1
        test_tower_output = []
        for i in range(self.test_batch_size):
            test_sen_matrix = (torch.unsqueeze(x[data['scope'][i]:data['scope'][i + 1]], 0)).repeat(53, 1,
                                                                                                    1)  # 先将 x[data['scope']] 扩充维度形成  53 * bag_size * 230， 然后在第一维重复53次
            test_layer_score = test_attention_score_stack[:, :,
                               data['scope'][i]:data['scope'][i + 1]]  # 查找Layer_score #(53, 2 ,bag_size)

            test_layer_repre = torch.reshape(test_layer_attention_score * (test_layer_score @ test_sen_matrix), [self.num_classes,
                                                                                  -1])  # 获得层次化表示表示  # 53 * 2 * bag_size @ 53* bag_size *230, 53* 2 *230 ,reshape 53 *460

            test_logits = test_layer_repre @ self.discrimitive_matrix.t() + self.bias  # 53 * 460 @ 460 * 53 +   (53, )
            test_output = torch.diagonal(F.softmax(test_logits))  # 获得对角输出
            test_tower_output.append(test_output)

        test_stack_output = torch.reshape(torch.stack(test_tower_output), [self.test_batch_size, self.num_classes])
        return test_stack_output



