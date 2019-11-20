import sys
sys.path.append("..")
import src._init_paths
import torch
from models.model import create_model
from torch.autograd import Variable

model = create_model('hourglassdrawing', {'hm': 5, 'wh': 2, 'reg': 2}, 64)
dummy_input = dummy_input = Variable(torch.rand(5, 1, 256, 256)) #假设输入13张1*28*28的图片
y = model(dummy_input)

