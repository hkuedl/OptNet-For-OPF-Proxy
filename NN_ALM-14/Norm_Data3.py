import numpy as np

def Norm_Data3(Input_A2, Output_PGQG_A2, NUM_Sample_Train_A2, NUM_Sample_Test_A2):
    """
    顺序切分输入/输出为训练与测试两段（不归一化）。

    参数
    ----
    Input_A2 : np.ndarray [N, Din]
        全部样本的输入特征。
    Output_PGQG_A2 : np.ndarray [N, Dout]
        全部样本的输出/标签。
    NUM_Sample_Train_A2 : int
        用作训练的样本数量（从开头截取）。
    NUM_Sample_Test_A2 : int
        用作测试的样本数量（紧随训练段之后截取）。

    返回
    ----
    Train_Input_A2 : np.ndarray [NUM_Sample_Train_A2, Din]
    Test_Input_A2  : np.ndarray [NUM_Sample_Test_A2,  Din]
    Train_PGQG_A2  : np.ndarray [NUM_Sample_Train_A2, Dout]
    Test_PGQG_A2   : np.ndarray [NUM_Sample_Test_A2,  Dout]

    说明
    ----
    - 原代码中存在“权重段/偏移量”的概念（NUM_Sample_Weight_A2、K_Step），当前均设为 0，
      等价于直接顺序切分：先取训练段，再取测试段。
    """

    # “权重段/偏移量”占位（与原逻辑一致，取 0）
    num_weight = 0
    k_step = 0

    # 计算测试段的起止下标
    start_test = NUM_Sample_Train_A2 + num_weight + k_step
    end_test = start_test + NUM_Sample_Test_A2

    # 输入切分
    Train_Input_A2 = Input_A2[:NUM_Sample_Train_A2, :]
    Test_Input_A2  = Input_A2[start_test:end_test, :]

    # 输出切分
    Train_PGQG_A2 = Output_PGQG_A2[:NUM_Sample_Train_A2, :]
    Test_PGQG_A2  = Output_PGQG_A2[start_test:end_test, :]

    return Train_Input_A2, Test_Input_A2, Train_PGQG_A2, Test_PGQG_A2
