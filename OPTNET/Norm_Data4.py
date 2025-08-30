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

def Norm_Data4(Input_A2, NUM_Sample_Train_A2, NUM_Sample_Test_A2):

    NUM_Sample_Weight_A2 = 0
    K_Step = 0

    Train_Input_A2 = Input_A2[:NUM_Sample_Train_A2, :]
    Train_Input_Weight_A2 = Input_A2[NUM_Sample_Train_A2 + K_Step:NUM_Sample_Train_A2 + NUM_Sample_Weight_A2 + K_Step,
                            :]
    Test_Input_A2 = Input_A2[NUM_Sample_Train_A2 + NUM_Sample_Weight_A2 + K_Step:NUM_Sample_Train_A2 + NUM_Sample_Weight_A2 + NUM_Sample_Test_A2 + K_Step, :]

    Train_Min_Input_A2 = np.min(Train_Input_A2, 0)
    Train_Max_Input_A2 = np.max(Train_Input_A2, 0)

    Train_Input_A2 = (Train_Input_A2 - Train_Min_Input_A2) / (Train_Max_Input_A2 - Train_Min_Input_A2)
    Test_Input_A2 = (Test_Input_A2 - Train_Min_Input_A2) / (Train_Max_Input_A2 - Train_Min_Input_A2)

    return Train_Input_A2, Test_Input_A2