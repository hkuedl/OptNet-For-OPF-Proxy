import numpy as np

def Norm_Data2(Input_A2, Output_PGQG_A2,
               NUM_Sample_Train_A2, NUM_Sample_Test_A2):
    """
    基于训练集样本，计算输入与输出在各特征维度上的 min/max（逐列），
    供后续反归一化使用。

    参数
    ----
    Input_A2 : np.ndarray [N, Din]
        全部样本的输入特征。
    Output_PGQG_A2 : np.ndarray [N, Dout]
        全部样本的输出标签（如 PG/QG 或 VM/VA）。
    NUM_Sample_Train_A2 : int
        用于训练的样本数（从开头截取）。
    NUM_Sample_Test_A2 : int
        用于测试的样本数（不参与统计，仅保留参数名以与原接口一致）。

    返回
    ----
    Train_Max_Input_A2 : np.ndarray [Din]
    Train_Min_Input_A2 : np.ndarray [Din]
    Train_Max_PGQG_A2  : np.ndarray [Dout]
    Train_Min_PGQG_A2  : np.ndarray [Dout]

    备注
    ----
    - 仅使用训练集区间（前 NUM_Sample_Train_A2 行）来统计 min/max，
      避免数据泄漏。
    - 原实现中对数据做了归一化计算但并未返回，这里删除该冗余操作，
      不影响外部对 min/max 的使用。
    """

    # 仅取训练段，用于统计 min/max（避免数据泄漏）
    Train_Input_A2 = Input_A2[:NUM_Sample_Train_A2, :]
    Train_PGQG_A2  = Output_PGQG_A2[:NUM_Sample_Train_A2, :]

    # 逐列统计
    Train_Min_Input_A2 = np.min(Train_Input_A2, axis=0)
    Train_Max_Input_A2 = np.max(Train_Input_A2, axis=0)

    Train_Min_PGQG_A2  = np.min(Train_PGQG_A2, axis=0)
    Train_Max_PGQG_A2  = np.max(Train_PGQG_A2, axis=0)

    return Train_Max_Input_A2, Train_Min_Input_A2, Train_Max_PGQG_A2, Train_Min_PGQG_A2
