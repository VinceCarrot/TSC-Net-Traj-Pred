import torch
import sys
from easydict import EasyDict
import yaml
from models.processor import *


if len(sys.argv) > 1:
    epoch = int(sys.argv[1])
else:
    epoch = None

with open('configs/two_stage_ssd.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
args = EasyDict(config)

if epoch == None:    
    test_tsc_net(args)
else:
    test_tsc_net(args,epoch)