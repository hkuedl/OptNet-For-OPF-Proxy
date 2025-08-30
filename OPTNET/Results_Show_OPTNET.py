import os
import copy
import random
import numpy as np
import torch

from Data_Generated import *
from Norm_Data import *
from Pytorch_Data import *
from Norm_Data2 import *
from Norm_Data3 import *
from sklearn.metrics import mean_squared_error

# generating sinusoidal data
def seed_torch(seed=114514):  # 114544
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# %%
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    seed_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 仅决定评估 heavy 还是 light；不会改变“后600=heavy，前600=light”的切分规则
    EVAL_MODE = "light"  # 可选 "heavy" 或 "light"
    SFX = "Heavy" if EVAL_MODE.lower() == "heavy" else "Light"

    Mag_loss_Area1 = np.zeros((1, 1))
    Te_loss_Area1 = np.zeros((1, 1))

    Input_A1, Output_VMVA_A1, Output_PGQG_A1, PDQD_A1, Mag_L1, Ang_L1 = Data_Generated("L", 1)
    Input_A2, Output_VMVA_A2, Output_PGQG_A2, PDQD_A2, Mag_L2, Ang_L2 = Data_Generated("H", 1)

    NUM_Sample_Train = 2400
    NUM_Sample_Test = 600
    NUM_Train = 4800
    NUM_Test = 1200
    HALF = int(NUM_Test / 2)  # = 600

    # 拼接：两份数据各取2400训练 + 600测试
    Input = np.vstack((
        Input_A1[:NUM_Sample_Train, :],
        Input_A2[:NUM_Sample_Train, :],
        Input_A1[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
        Input_A2[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :]))
    Output_VMVA = np.vstack((
        Output_VMVA_A1[:NUM_Sample_Train, :],
        Output_VMVA_A2[:NUM_Sample_Train, :],
        Output_VMVA_A1[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
        Output_VMVA_A2[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :]))
    Output_PGQG = np.vstack((
        Output_PGQG_A1[:NUM_Sample_Train, :],
        Output_PGQG_A2[:NUM_Sample_Train, :],
        Output_PGQG_A1[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
        Output_PGQG_A2[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :]))

    PDQD = np.vstack((
        PDQD_A1[:NUM_Sample_Train, :],
        PDQD_A2[:NUM_Sample_Train, :],
        PDQD_A1[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
        PDQD_A2[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :]))

    Mag_L = np.vstack((
        Mag_L1[:NUM_Sample_Train, :],
        Mag_L2[:NUM_Sample_Train, :],
        Mag_L1[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
        Mag_L2[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :]))
    Ang_L = np.vstack((
        Ang_L1[:NUM_Sample_Train, :],
        Ang_L2[:NUM_Sample_Train, :],
        Ang_L1[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
        Ang_L2[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :]))

    # 归一化划分
    Train_Input_A1, Test_Input_A1, Train_VMVA_A1, Test_VMVA_A1 = Norm_Data(Input, Output_VMVA, NUM_Train, NUM_Test)
    Train_Input_A1, Test_Input_A1, Train_PGQG_A1, Test_PGQG_A1 = Norm_Data(Input, Output_PGQG, NUM_Train, NUM_Test)

    Train_Input_A1, Test_Input_A1, Train_Mag_A1, Test_Mag_A1 = Norm_Data3(Input, Mag_L, NUM_Train, NUM_Test)
    Train_Input_A1, Test_Input_A1, Train_Ang_A1, Test_Ang_A1 = Norm_Data3(Input, Ang_L, NUM_Train, NUM_Test)

    Train_Input_A1, Test_Input_A1, Train_PDQD, Test_PDQD = Norm_Data3(Input, PDQD, NUM_Train, NUM_Test)

    # ---- 关键切分：前600=heavy，后600=light（严格按你给的逻辑）----
    Test_VMVA_Heavy = copy.deepcopy(Test_VMVA_A1[HALF:, :])
    Test_VMVA_Light = copy.deepcopy(Test_VMVA_A1[:HALF, :])
    Test_Input_Heavy = copy.deepcopy(Test_Input_A1[HALF:, :])
    Test_Input_Light = copy.deepcopy(Test_Input_A1[:HALF:, :])

    Test_Mag_Heavy = copy.deepcopy(Test_Mag_A1[HALF:, :])
    Test_Mag_Light = copy.deepcopy(Test_Mag_A1[:HALF, :])

    Test_Ang_Heavy = copy.deepcopy(Test_Ang_A1[HALF:, :])
    Test_Ang_Light = copy.deepcopy(Test_Ang_A1[:HALF, :])

    Test_PGQG_Heavy = copy.deepcopy(Test_PGQG_A1[HALF:, :])
    Test_PGQG_Light = copy.deepcopy(Test_PGQG_A1[:HALF, :])

    Test_PDQD_Heavy = copy.deepcopy(Test_PDQD[HALF:, :])
    Test_PDQD_Light = copy.deepcopy(Test_PDQD[:HALF, :])

    # 组合输出：PGQG + VMVA
    Test_Total_Heavy = np.hstack((Test_PGQG_Heavy, Test_VMVA_Heavy))
    Test_Total_Light = np.hstack((Test_PGQG_Light, Test_VMVA_Light))

    # 训练/测试的随机打散（保持与原代码一致，即便这里只做评估）
    shuffle_Train = np.random.permutation(np.arange(len(Train_Input_A1)))
    shuffle_Test = np.random.permutation(np.arange(len(Test_Input_A1)))

    Train_Input_A1 = Train_Input_A1[shuffle_Train, :]
    Train_VMVA_A1 = Train_VMVA_A1[shuffle_Train, :]
    Train_PGQG_A1 = Train_PGQG_A1[shuffle_Train, :]

    Test_Input_A1 = Test_Input_A1[shuffle_Test, :]
    Test_VMVA_A1 = Test_VMVA_A1[shuffle_Test, :]
    Test_PGQG_A1 = Test_PGQG_A1[shuffle_Test, :]

    Train_Total_A1 = np.hstack((Train_PGQG_A1, Train_VMVA_A1))
    Test_Total_A1 = np.hstack((Test_PGQG_A1, Test_VMVA_A1))

    # %%
    batch_size = 8
    NUM_PDQD = 28 - 6
    NUM_PGQG = 26 + 10  # 36 = PG(5)+QG(5)+VM(13)+VA(13)

    # 构造张量（与原逻辑一致）
    Train_In_A1, Test_In_A1, Train_Out_A1, Test_Out_A1 = Pytorch_Data(
        Train_Input_A1, Test_Input_A1, Train_Total_A1, Test_Total_A1, NUM_Train, NUM_Test,
        NUM_PDQD, NUM_PGQG
    )
    Test_In_Heavy, Test_In_Light, Test_Out_Heavy, Test_Out_Light = Pytorch_Data(
        Test_Input_Heavy, Test_Input_Light, Test_Total_Heavy, Test_Total_Light, NUM_Sample_Test, NUM_Sample_Test,
        NUM_PDQD, NUM_PGQG
    )

    # 送显卡（尽管这里不训练，保持一致）
    Test_In_Heavy = Test_In_Heavy.to(device)
    Test_Out_Heavy = Test_Out_Heavy.to(device)
    Test_In_Light = Test_In_Light.to(device)
    Test_Out_Light = Test_Out_Light.to(device)
    Train_In_A1 = Train_In_A1.to(device)
    Test_In_A1 = Test_In_A1.to(device)
    Train_Out_A1 = Train_Out_A1.to(device)
    Test_Out_A1 = Test_Out_A1.to(device)

    # 反归一化所需极值 —— 注意这里按“总输出”36列获取
    Output_Total = np.hstack((Output_PGQG, Output_VMVA))
    In_Max_A1, In_Min_A1, Out_Max_A1, Out_Min_A1 = Norm_Data2(Input, Output_Total, NUM_Train, NUM_Test)

    # 选择评估段：heavy 或 light
    if SFX == "Heavy":
        Out_Pre_Test = torch.load("Res_OPTNET\\Pre_Test_Heavy_Total_Prun1.pt",
                                  map_location=torch.device('cpu'), weights_only=True)
        Test_Out_CURR = Test_Out_Heavy.reshape(HALF, NUM_PGQG)
        # 下面潮流评估也使用 Heavy 的 PD/QD、角度/幅值
        PDQD_eval = Test_PDQD_Heavy
        Ang_eval = Test_Ang_Heavy
        Mag_eval = Test_Mag_Heavy
    else:
        Out_Pre_Test = torch.load("Res_OPTNET\\Pre_Test_Light_Total_Prun1.pt",
                                  map_location=torch.device('cpu'), weights_only=True)
        Test_Out_CURR = Test_Out_Light.reshape(HALF, NUM_PGQG)
        PDQD_eval = Test_PDQD_Light
        Ang_eval = Test_Ang_Light
        Mag_eval = Test_Mag_Light

    # 反归一化：预测与真值均回到物理量
    Num_Pre_Test = Out_Pre_Test.detach().numpy() * (Out_Max_A1 - Out_Min_A1) + Out_Min_A1
    Num_Test = copy.deepcopy(Test_Out_CURR).detach().numpy() * (Out_Max_A1 - Out_Min_A1) + Out_Min_A1

    # 指标分块
    NUM_PG = 5
    NUM_QG = 5
    NUM_VM = 13
    NUM_VA = 13
    # %%
    import numpy as np
    from pyomo.core import *
    from pypower.api import case30, ppoption, runopf, runpf, case118, case14, case9
    from makeYbus_my import *

    data_in = case14()
    # 调整 PMAX（按原代码）
    data_in['gen'][0, 8] = 100
    data_in['gen'][1, 8] = 100
    data_in['gen'][2, 8] = 80
    data_in['gen'][3, 8] = 60
    data_in['gen'][4, 8] = 60

    Y = makeYbus_my(data_in['baseMVA'], data_in['bus'], data_in['branch'])
    Gencost = data_in['gencost'][:, 4:7]

    a2 = np.transpose(Gencost[:, 0])[np.newaxis, :]
    a1 = np.transpose(Gencost[:, 1])[np.newaxis, :]

    # 取块
    Num_Pre_Test_PG = Num_Pre_Test[:, 0:NUM_PG]
    Num_Pre_Test_QG = Num_Pre_Test[:, NUM_PG:NUM_PG + NUM_QG]
    Num_Pre_Test_VM = Num_Pre_Test[:, NUM_PG + NUM_QG:NUM_PG + NUM_QG + NUM_VA]
    Num_Pre_Test_VA = Num_Pre_Test[:, NUM_PG + NUM_QG + NUM_VA:NUM_PG + NUM_QG + NUM_VA + NUM_VM]

    Num_Test_PG = Num_Test[:, 0:NUM_PG]
    Num_Test_QG = Num_Test[:, NUM_PG:NUM_PG + NUM_QG]
    Num_Test_VM = Num_Test[:, NUM_PG + NUM_QG:NUM_PG + NUM_QG + NUM_VA]
    Num_Test_VA = Num_Test[:, NUM_PG + NUM_QG + NUM_VA:NUM_PG + NUM_QG + NUM_VA + NUM_VM]

    # 成本
    Cost_Mat_Pre = a2 * Num_Pre_Test_PG * Num_Pre_Test_PG + a1 * Num_Pre_Test_PG
    Cost_Mat = a2 * Num_Test_PG * Num_Test_PG + a1 * Num_Test_PG

    Cost_Vec_Pre = np.sum(Cost_Mat_Pre, 1)
    Cost_Vec = np.sum(Cost_Mat, 1)

    Cost_Diff_Mean = np.abs(np.sum((Cost_Vec_Pre - Cost_Vec) / Cost_Vec) / HALF)  # 按原逻辑，分母=600

    # 越界率
    Fea_PG = np.zeros_like(Num_Pre_Test_PG)
    Fea_QG = np.zeros_like(Num_Pre_Test_QG)
    Fea_VM = np.zeros_like(Num_Pre_Test_VM)

    PG_Max = data_in['gen'][:, 8]
    PG_Min = data_in['gen'][:, 9]
    QG_Max = data_in['gen'][:, 3]
    QG_Min = data_in['gen'][:, 4]
    VM_Max = data_in['bus'][:, 11]
    VM_Min = data_in['bus'][:, 12]

    for i in range(Num_Pre_Test_PG.shape[0]):
        for j in range(NUM_PG):
            if Num_Pre_Test_PG[i, j] < PG_Min[j] or Num_Pre_Test_PG[i, j] > PG_Max[j]:
                Fea_PG[i, j] = 1
            if Num_Pre_Test_QG[i, j] < QG_Min[j] or Num_Pre_Test_QG[i, j] > QG_Max[j]:
                Fea_QG[i, j] = 1

    for i in range(Num_Pre_Test_VM.shape[0]):
        for j in range(NUM_VM):
            if Num_Pre_Test_VM[i, j] < VM_Min[j] or Num_Pre_Test_VM[i, j] > VM_Max[j]:
                Fea_VM[i, j] = 1

    Rate_InFea_PG = np.sum(Fea_PG) / Fea_PG.size
    Rate_InFea_QG = np.sum(Fea_QG) / Fea_QG.size
    Rate_InFea_VM = np.sum(Fea_VM) / Fea_VM.size

    # 潮流一致性（按所选 heavy/light 进行）
    PD = copy.deepcopy(PDQD_eval[:, 0:14])
    QD = copy.deepcopy(PDQD_eval[:, 14:])

    PG_In = np.zeros_like(PD)
    QG_In = np.zeros_like(QD)

    PG_Index = [1 - 1, 2 - 1, 3 - 1, 6 - 1, 8 - 1]
    PG_In[:, PG_Index] = copy.deepcopy(Num_Pre_Test_PG)
    QG_In[:, PG_Index] = copy.deepcopy(Num_Pre_Test_QG)

    Num_Pre_Test_VA_full = np.hstack((Ang_eval, Num_Pre_Test_VA))
    Num_Pre_Test_VM_full = np.hstack((Mag_eval, Num_Pre_Test_VM))

    Ang = Num_Pre_Test_VA_full * np.pi / 180
    Ang1 = np.exp(1j * Ang)
    Vol_Vec1 = Num_Pre_Test_VM_full * Ang1

    IIIII = Y @ np.transpose(Vol_Vec1)
    I1 = IIIII.conjugate()
    Right = Vol_Vec1 * np.transpose(I1)

    Pi = (PG_In - PD) / 100
    Qi = (QG_In - QD) / 100
    Left = Pi + 1j * Qi

    PF_Mat = np.abs(Right - Left)
    Max_PF_Mat = np.mean(PF_Mat)

    # 打印
    print(f"=== {SFX} 集结果（{HALF} 样本）===")
    print(f"成本相对差（均值）: {Cost_Diff_Mean:.4e}")
    print(f"PG 越界率: {Rate_InFea_PG:.3%}")
    print(f"QG 越界率: {Rate_InFea_QG:.3%}")
    print(f"VM 越界率: {Rate_InFea_VM:.3%}")
    print(f"潮流不匹配平均值: {Max_PF_Mat:.4e}")
