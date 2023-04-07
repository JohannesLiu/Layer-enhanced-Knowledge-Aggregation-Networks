import torch
import numpy as np
from sklearn.metrics import average_precision_score

class TrainNYTMetric(object):
    def __init__(self, correct_prediction, exclude_na_flatten_output, order):
        def accuracy(self, output, target):
            with torch.no_grad():
                pred = torch.argmax(output, dim=1)
                assert pred.shape[0] == len(target)
                correct = 0
                correct += torch.sum(pred == target).item()
            return correct / len(target)

        def accuracy_all(self):
            return torch.mean(self.correct_predictions.float())

        def accuracy_sep(self, correct_predictions, label, tot1, tot2, s1, s2):
            num = 0
            s = 0
            for num in correct_predictions:  # 在这160个batch中
                if label[s] == 0:  # 如果预测==0
                    tot1 += 1.0
                    if num:  ##如果==0且正确
                        s1 += 1.0
                else:
                    tot2 += 1.0
                    if num:  ##如果预测！=0且正确
                        s2 += 1.0
                s = s + 1
            return s1 / tot1, s2 / tot2


class EvalNYTMetric(object):
    def __init__(self, id2rel, fewrel_100,fewrel_200, exclude_na_flatten_label, exclude_na_label, index_non_zero):
        self.id2rel = id2rel
        self.fewrel_100 = fewrel_100
        self.fewrel_200 = fewrel_200
        self.exclude_na_flatten_label = exclude_na_flatten_label
        self.exclude_na_label = exclude_na_label

        self.index_non_zero = index_non_zero

    def mi_ma_100(self, exclude_na_output):
        ss = 0
        ss10 = 0
        ss15 = 0
        ss20 = 0

        ss_rel = {}
        ss10_rel = {}
        ss15_rel = {}
        ss20_rel = {}

        for j, label in zip(exclude_na_output, self.exclude_na_label):
            score = None
            num = None
            for ind, ll in enumerate(label):
                if ll > 0:
                    score = j[ind]
                    num = ind
                    break
            if num is None:
                continue
            if self.id2rel[num + 1] in self.fewrel_100:
                ss += 1.0
                mx = 0
                for sc in j:
                    if sc > score:
                        mx = mx + 1
                if not num in ss_rel:
                    ss_rel[num] = 0
                    ss10_rel[num] = 0
                    ss15_rel[num] = 0
                    ss20_rel[num] = 0
                ss_rel[num] += 1.0
                if mx < 10:
                    ss10 += 1.0
                    ss10_rel[num] += 1.0
                if mx < 15:
                    ss15 += 1.0
                    ss15_rel[num] += 1.0
                if mx < 20:
                    ss20 += 1.0
                    ss20_rel[num] += 1.0
        mi_10 = (ss10 / ss)
        mi_15 = (ss15 / ss)
        mi_20 = (ss20 / ss)
        ma_10 = (np.array([ss10_rel[i] / ss_rel[i] for i in ss_rel])).mean()
        ma_15 = (np.array([ss15_rel[i] / ss_rel[i] for i in ss_rel])).mean()
        ma_20 = (np.array([ss20_rel[i] / ss_rel[i] for i in ss_rel])).mean()

        return {"mi_10": mi_10, "mi_15": mi_15, "mi_20": mi_20, "ma_10": ma_10, "ma_15": ma_15, "ma_20": ma_20}

    def mi_ma_200(self, exclude_na_output):
        ss = 0
        ss10 = 0
        ss15 = 0
        ss20 = 0

        ss_rel = {}
        ss10_rel = {}
        ss15_rel = {}
        ss20_rel = {}

        for j, label in zip(exclude_na_output, self.exclude_na_label):
            score = None
            num = None
            for ind, ll in enumerate(label):
                if ll > 0:
                    score = j[ind]
                    num = ind
                    break
            if num is None:
                continue
            if self.id2rel[num + 1] in self.fewrel_200:
                ss += 1.0
                mx = 0
                for sc in j:
                    if sc > score:
                        mx = mx + 1
                if not num in ss_rel:
                    ss_rel[num] = 0
                    ss10_rel[num] = 0
                    ss15_rel[num] = 0
                    ss20_rel[num] = 0
                ss_rel[num] += 1.0
                if mx < 10:
                    ss10 += 1.0
                    ss10_rel[num] += 1.0
                if mx < 15:
                    ss15 += 1.0
                    ss15_rel[num] += 1.0
                if mx < 20:
                    ss20 += 1.0
                    ss20_rel[num] += 1.0
        mi_10 = (ss10 / ss)
        mi_15 = (ss15 / ss)
        mi_20 = (ss20 / ss)

        ma_10 = (np.array([ss10_rel[i] / ss_rel[i] for i in ss_rel])).mean()
        ma_15 = (np.array([ss15_rel[i] / ss_rel[i] for i in ss_rel])).mean()
        ma_20 = (np.array([ss20_rel[i] / ss_rel[i] for i in ss_rel])).mean()

        return {"mi_10": mi_10, "mi_15": mi_15, "mi_20": mi_20, "ma_10": ma_10, "ma_15": ma_15, "ma_20": ma_20}

    def pr(self, exclude_na_output, exclude_na_flatten_output):
        m = average_precision_score(self.exclude_na_flatten_label, exclude_na_flatten_output)
        M = average_precision_score(self.exclude_na_label[:, self.index_non_zero], exclude_na_output[:, self.index_non_zero],
                                    average='macro')
        return {"m":m, "M": M}

class EvalNYTMetricX(object):
    def __init__(self,  exclude_na_flatten_label):
        self.exclude_na_flatten_label = exclude_na_flatten_label

    def pone_two_all(self, exclude_na_flatten_output):
        order = np.argsort(-exclude_na_flatten_output)
        return np.mean(self.exclude_na_flatten_label[order[:100]]),  np.mean(self.exclude_na_flatten_label[order[:200]]), np.mean(self.exclude_na_flatten_label[order[:300]])
