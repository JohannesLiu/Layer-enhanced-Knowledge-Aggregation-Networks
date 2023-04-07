import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch import optim
from lekankit.networks.embedders import WordVec
from lekankit.networks.encoders import SentenceEncoder

