import torch
import numpy as np


def Pytorch_Data(Train_Input_A2, Test_Input_A2, Train_PGQG_A2, Test_PGQG_A2,
                 NUM_Sample_Train_A2, NUM_Sample_Test_A2, NUM_PDQD, NUM_PGQG):
    """
    将 numpy 数组打包为 PyTorch 张量，并整理为 [N, 1, D] 的三维形状。

    返回：
        Train_In_New  : torch.FloatTensor [NUM_Sample_Train_A2, 1, NUM_PDQD]
        Test_In_New   : torch.FloatTensor [NUM_Sample_Test_A2,  1, NUM_PDQD]
        Train_Out_New : torch.FloatTensor [NUM_Sample_Train_A2, 1, NUM_PGQG]
        Test_Out_New  : torch.FloatTensor [NUM_Sample_Test_A2,  1, NUM_PGQG]
    """

    # 输入：numpy -> [N, 1, Din] -> torch.float32
    Train_In_New = Train_Input_A2.reshape(NUM_Sample_Train_A2, 1, NUM_PDQD)
    Train_In_New = torch.from_numpy(Train_In_New).to(torch.float32)

    Test_In_New = Test_Input_A2.reshape(NUM_Sample_Test_A2, 1, NUM_PDQD)
    Test_In_New = torch.from_numpy(Test_In_New).to(torch.float32)

    # 输出(标签)：numpy -> [N, 1, Dout] -> torch.float32
    Train_Out_New = Train_PGQG_A2.reshape(NUM_Sample_Train_A2, 1, NUM_PGQG)
    Train_Out_New = torch.from_numpy(Train_Out_New).to(torch.float32)

    Test_Out_New = Test_PGQG_A2.reshape(NUM_Sample_Test_A2, 1, NUM_PGQG)
    Test_Out_New = torch.from_numpy(Test_Out_New).to(torch.float32)

    return Train_In_New, Test_In_New, Train_Out_New, Test_Out_New
