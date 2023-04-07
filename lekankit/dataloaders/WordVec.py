import numpy as np


class SkipGram(object):
    def __init__(self, data_path = "./data/"):
        super(SkipGram, self).__init__()
        self._data_path = data_=data_path
        self._SkipGramVec = np.load(self._data_path + 'vec.npy')

    @property
    def SkipGramVec(self):
        return self._SkipGramVec

if __name__ == "__main__":
    vec = SkipGram()
    print(vec.SkipGramVec)