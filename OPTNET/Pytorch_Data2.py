import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import copy
from Data_Generated import *
import os
import random


def Pytorch_Data2(Train_Input_A2, Test_Input_A2, NUM_Sample_Train_A2, NUM_Sample_Test_A2, NUM_PDQD, NUM_PGQG):

    Train_In_New = Train_Input_A2.reshape(NUM_Sample_Train_A2, 1, NUM_PDQD)
    Train_In_New = torch.from_numpy(Train_In_New)
    Train_In_New = Train_In_New.to(torch.float32)

    Test_In_New = Test_Input_A2.reshape(NUM_Sample_Test_A2, 1, NUM_PDQD)
    Test_In_New = torch.from_numpy(Test_In_New)
    Test_In_New = Test_In_New.to(torch.float32)

    return Train_In_New, Test_In_New