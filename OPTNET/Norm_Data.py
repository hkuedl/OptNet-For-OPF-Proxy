import numpy as np

def Norm_Data(Input_A2, Output_PGQG_A2, NUM_Sample_Train_A2, NUM_Sample_Test_A2):
    """
    将原始输入/输出按（训练 / 测试）切分，并用训练集的 min-max 对输入与输出分别做归一化。
    """
    # 中间权重段控制量（当前为 0，相当于直接顺序切分）
    NUM_Sample_Weight_A2 = 0
    K_Step = 0

    # —— 顺序切分：输入 —— #
    Train_Input_A2 = Input_A2[:NUM_Sample_Train_A2, :]
    _Train_Input_Weight_A2 = Input_A2[
        NUM_Sample_Train_A2 + K_Step : NUM_Sample_Train_A2 + NUM_Sample_Weight_A2 + K_Step, :
    ]
    Test_Input_A2 = Input_A2[
        NUM_Sample_Train_A2 + NUM_Sample_Weight_A2 + K_Step :
        NUM_Sample_Train_A2 + NUM_Sample_Weight_A2 + NUM_Sample_Test_A2 + K_Step, :
    ]

    # —— 顺序切分：输出 —— #
    Train_PGQG_A2 = Output_PGQG_A2[:NUM_Sample_Train_A2, :]
    _Train_PGQG_Weight_A2 = Output_PGQG_A2[
        NUM_Sample_Train_A2 + K_Step : NUM_Sample_Train_A2 + NUM_Sample_Weight_A2 + K_Step, :
    ]
    Test_PGQG_A2 = Output_PGQG_A2[
        NUM_Sample_Train_A2 + NUM_Sample_Weight_A2 + K_Step :
        NUM_Sample_Train_A2 + NUM_Sample_Weight_A2 + NUM_Sample_Test_A2 + K_Step, :
    ]

    # —— 输入归一化（基于训练集的 min/max）—— #
    Train_Min_Input_A2 = np.min(Train_Input_A2, axis=0)
    Train_Max_Input_A2 = np.max(Train_Input_A2, axis=0)
    Train_Input_A2 = (Train_Input_A2 - Train_Min_Input_A2) / (Train_Max_Input_A2 - Train_Min_Input_A2)
    Test_Input_A2  = (Test_Input_A2  - Train_Min_Input_A2) / (Train_Max_Input_A2 - Train_Min_Input_A2)

    # —— 输出归一化（基于训练集的 min/max）—— #
    Train_Min_PGQG_A2 = np.min(Train_PGQG_A2, axis=0)
    Train_Max_PGQG_A2 = np.max(Train_PGQG_A2, axis=0)
    Train_PGQG_A2 = (Train_PGQG_A2 - Train_Min_PGQG_A2) / (Train_Max_PGQG_A2 - Train_Min_PGQG_A2)
    Test_PGQG_A2  = (Test_PGQG_A2  - Train_Min_PGQG_A2) / (Train_Max_PGQG_A2 - Train_Min_PGQG_A2)

    return Train_Input_A2, Test_Input_A2, Train_PGQG_A2, Test_PGQG_A2
