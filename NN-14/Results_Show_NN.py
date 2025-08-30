import os
import copy
import random
import numpy as np
import torch

from Data_Generated import *
from Norm_Data import *
from Norm_Data2 import *
from Norm_Data3 import *
from pypower.api import case14
from makeYbus_my import makeYbus_my


def seed_torch(seed: int = 114514) -> None:
    """固定随机种子，尽量保证可复现。"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ========= 主流程 =========
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    seed_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 一键切换：前600为 Light，后600为 Heavy
    EVAL_MODE = "light"                  # "light" 或 "heavy"
    IS_LIGHT = (EVAL_MODE.lower() == "light")
    SFX = "Light" if IS_LIGHT else "Heavy"

    # 生成并拼接两份数据
    NUM_Sample_Train = 2400
    NUM_Sample_Test = 600

    Input_A1, Output_VMVA_A1, Output_PGQG_A1, PDQD_A1, Mag_L1, Ang_L1 = Data_Generated("L", 1)
    Input_A2, Output_VMVA_A2, Output_PGQG_A2, PDQD_A2, Mag_L2, Ang_L2 = Data_Generated("H", 1)

    Input = np.vstack((
        Input_A1[:NUM_Sample_Train, :],
        Input_A2[:NUM_Sample_Train, :],
        Input_A1[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
        Input_A2[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
    ))
    Output_VMVA = np.vstack((
        Output_VMVA_A1[:NUM_Sample_Train, :],
        Output_VMVA_A2[:NUM_Sample_Train, :],
        Output_VMVA_A1[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
        Output_VMVA_A2[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
    ))
    Output_PGQG = np.vstack((
        Output_PGQG_A1[:NUM_Sample_Train, :],
        Output_PGQG_A2[:NUM_Sample_Train, :],
        Output_PGQG_A1[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
        Output_PGQG_A2[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
    ))
    PDQD = np.vstack((
        PDQD_A1[:NUM_Sample_Train, :],
        PDQD_A2[:NUM_Sample_Train, :],
        PDQD_A1[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
        PDQD_A2[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
    ))
    Mag_L = np.vstack((
        Mag_L1[:NUM_Sample_Train, :],
        Mag_L2[:NUM_Sample_Train, :],
        Mag_L1[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
        Mag_L2[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
    ))
    Ang_L = np.vstack((
        Ang_L1[:NUM_Sample_Train, :],
        Ang_L2[:NUM_Sample_Train, :],
        Ang_L1[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
        Ang_L2[NUM_Sample_Train:NUM_Sample_Train + NUM_Sample_Test, :],
    ))

    NUM_Train = 4800
    NUM_Test = 1200
    HALF = NUM_Test // 2  # 600

    # 仅用于评估（不训练）：归一化切分
    _, Test_Input_A1, _, Test_VMVA_A1 = Norm_Data(Input, Output_VMVA, NUM_Train, NUM_Test)
    _, _, _, Test_PGQG_A1 = Norm_Data(Input, Output_PGQG, NUM_Train, NUM_Test)

    # 原尺度顺序切分（PD/QD、Vmag、Vang）
    _, _, _, Test_Mag_A1 = Norm_Data3(Input, Mag_L, NUM_Train, NUM_Test)
    _, _, _, Test_Ang_A1 = Norm_Data3(Input, Ang_L, NUM_Train, NUM_Test)
    _, _, _, Test_PDQD = Norm_Data3(Input, PDQD, NUM_Train, NUM_Test)

    # 前600=Light，后600=Heavy
    if IS_LIGHT:
        Test_Input_CURR = copy.deepcopy(Test_Input_A1[:HALF, :])
        Test_VMVA_CURR = copy.deepcopy(Test_VMVA_A1[:HALF, :])
        Test_PGQG_CURR = copy.deepcopy(Test_PGQG_A1[:HALF, :])
        Test_Mag_CURR = copy.deepcopy(Test_Mag_A1[:HALF, :])
        Test_Ang_CURR = copy.deepcopy(Test_Ang_A1[:HALF, :])
        Test_PDQD_CURR = copy.deepcopy(Test_PDQD[:HALF, :])
    else:
        Test_Input_CURR = copy.deepcopy(Test_Input_A1[HALF:, :])
        Test_VMVA_CURR = copy.deepcopy(Test_VMVA_A1[HALF:, :])
        Test_PGQG_CURR = copy.deepcopy(Test_PGQG_A1[HALF:, :])
        Test_Mag_CURR = copy.deepcopy(Test_Mag_A1[HALF:, :])
        Test_Ang_CURR = copy.deepcopy(Test_Ang_A1[HALF:, :])
        Test_PDQD_CURR = copy.deepcopy(Test_PDQD[HALF:, :])

    # 反归一化所需 min/max（仅取输出的）
    _, _, Out_Max_PGQG, Out_Min_PGQG = Norm_Data2(Input, Output_PGQG, NUM_Train, NUM_Test)
    _, _, Out_Max_VMVA, Out_Min_VMVA = Norm_Data2(Input, Output_VMVA, NUM_Train, NUM_Test)

    # 反归一化得到真值（物理量）
    Num_Test_PGQG = Test_PGQG_CURR * (Out_Max_PGQG - Out_Min_PGQG) + Out_Min_PGQG
    Num_Test_VMVA = Test_VMVA_CURR * (Out_Max_VMVA - Out_Min_VMVA) + Out_Min_VMVA

    # 加载对应 Light/Heavy 的 NN 预测结果（保持你原有行为：不额外反归一化）
    Out_Pre_Test_PGQG = torch.load(f"Res_NN\\Pre_{SFX}_PG_NN.pt", map_location=torch.device("cpu"))
    Out_Pre_Test_VMVA = torch.load(f"Res_NN\\Pre_{SFX}_VM_NN.pt", map_location=torch.device("cpu"))

    Num_Pre_Test_PGQG = Out_Pre_Test_PGQG.detach().numpy()  # 可能是 [600,10] 或 [600,5]
    Num_Pre_Test_VMVA = Out_Pre_Test_VMVA.detach().numpy()  # 通常 [600,26]

    # 若只有 PG（5列），用真值 QG 拼接，保证后续评估流程一致
    NUM_PG, NUM_QG, NUM_VM, NUM_VA = 5, 5, 13, 13
    if Num_Pre_Test_PGQG.shape[1] == NUM_PG:
        print("[Notice] PG-only predictions detected (5 columns). "
              "Concatenating ground-truth QG for evaluation.")
        Num_Pre_Test_PGQG = np.hstack([Num_Pre_Test_PGQG, Num_Test_PGQG[:, NUM_PG:NUM_PG + NUM_QG]])

    # ========== 成本/越界/潮流评估 ==========
    data_in = case14()
    # 调整 PMAX（与原代码一致）
    data_in["gen"][0, 8] = 100
    data_in["gen"][1, 8] = 100
    data_in["gen"][2, 8] = 80
    data_in["gen"][3, 8] = 60
    data_in["gen"][4, 8] = 60

    Y = makeYbus_my(data_in["baseMVA"], data_in["bus"], data_in["branch"])
    Gencost = data_in["gencost"][:, 4:7]
    a2 = Gencost[:, 0][np.newaxis, :]
    a1 = Gencost[:, 1][np.newaxis, :]

    # 拆块
    Num_Pre_Test_PG = Num_Pre_Test_PGQG[:, :NUM_PG]
    Num_Pre_Test_QG = Num_Pre_Test_PGQG[:, NUM_PG:NUM_PG + NUM_QG]
    Num_Pre_Test_VM = Num_Pre_Test_VMVA[:, :NUM_VM]
    Num_Pre_Test_VA = Num_Pre_Test_VMVA[:, NUM_VA:NUM_VA + NUM_VM]

    Num_Test_PG = Num_Test_PGQG[:, :NUM_PG]
    Num_Test_QG = Num_Test_PGQG[:, NUM_PG:NUM_PG + NUM_QG]
    Num_Test_VM = Num_Test_VMVA[:, :NUM_VM]
    Num_Test_VA = Num_Test_VMVA[:, NUM_VA:NUM_VA + NUM_VM]

    # 成本相对误差
    Cost_Mat_Pre = a2[:, :NUM_PG] * Num_Pre_Test_PG**2 + a1[:, :NUM_PG] * Num_Pre_Test_PG
    Cost_Mat     = a2[:, :NUM_PG] * Num_Test_PG**2     + a1[:, :NUM_PG] * Num_Test_PG
    Cost_Vec_Pre = np.sum(Cost_Mat_Pre, axis=1)
    Cost_Vec     = np.sum(Cost_Mat, axis=1)
    Cost_Diff_Mean = np.sum(np.abs(Cost_Vec_Pre - Cost_Vec) / np.maximum(Cost_Vec, 1e-8)) / Num_Pre_Test_PG.shape[0]

    # 越界率与越界幅度
    PG_Max, PG_Min = data_in["gen"][:, 8][:NUM_PG], data_in["gen"][:, 9][:NUM_PG]
    QG_Max, QG_Min = data_in["gen"][:, 3][:NUM_QG], data_in["gen"][:, 4][:NUM_QG]
    VM_Max, VM_Min = data_in["bus"][:, 11][:NUM_VM], data_in["bus"][:, 12][:NUM_VM]

    Fea_PG = np.zeros_like(Num_Pre_Test_PG)
    Fea_QG = np.zeros_like(Num_Pre_Test_QG)
    Fea_VM = np.zeros_like(Num_Pre_Test_VM)

    Val_PG = np.zeros_like(Num_Pre_Test_PG)
    Val_QG = np.zeros_like(Num_Pre_Test_QG)
    Val_VM = np.zeros_like(Num_Pre_Test_VM)

    for i in range(Num_Pre_Test_PG.shape[0]):
        for j in range(NUM_PG):
            if Num_Pre_Test_PG[i, j] < PG_Min[j] or Num_Pre_Test_PG[i, j] > PG_Max[j]:
                Fea_PG[i, j] = 1
                Val_PG[i, j] = PG_Min[j] - Num_Pre_Test_PG[i, j] if Num_Pre_Test_PG[i, j] < PG_Min[j] else Num_Pre_Test_PG[i, j] - PG_Max[j]
            if Num_Pre_Test_QG[i, j] < QG_Min[j] or Num_Pre_Test_QG[i, j] > QG_Max[j]:
                Fea_QG[i, j] = 1
                Val_QG[i, j] = QG_Min[j] - Num_Pre_Test_QG[i, j] if Num_Pre_Test_QG[i, j] < QG_Min[j] else Num_Pre_Test_QG[i, j] - QG_Max[j]

    for i in range(Num_Pre_Test_VM.shape[0]):
        for j in range(NUM_VM):
            if Num_Pre_Test_VM[i, j] < VM_Min[j] or Num_Pre_Test_VM[i, j] > VM_Max[j]:
                Fea_VM[i, j] = 1
                Val_VM[i, j] = VM_Min[j] - Num_Pre_Test_VM[i, j] if Num_Pre_Test_VM[i, j] < VM_Min[j] else Num_Pre_Test_VM[i, j] - VM_Max[j]

    Rate_InFea_PG = Fea_PG.sum() / Fea_PG.size
    Rate_InFea_QG = Fea_QG.sum() / Fea_QG.size
    Rate_InFea_VM = Fea_VM.sum() / Fea_VM.size

    Val_InFea_PG = Val_PG.sum() / Val_PG.size
    Val_InFea_QG = Val_QG.sum() / Val_QG.size
    Val_InFea_VM = Val_VM.sum() / Val_VM.size

    # 潮流一致性
    nb = data_in["bus"].shape[0]
    GEN_BUS_INDEX = np.array([1, 2, 3, 6, 8]) - 1  # 0-based

    PD = copy.deepcopy(Test_PDQD_CURR[:, :nb])
    QD = copy.deepcopy(Test_PDQD_CURR[:, nb:])

    PG_In = np.zeros_like(PD)
    QG_In = np.zeros_like(QD)
    PG_In[:, GEN_BUS_INDEX] = Num_Pre_Test_PG
    QG_In[:, GEN_BUS_INDEX] = Num_Pre_Test_QG

    # 将预测 VM/VA 与基准首列拼接成 14 维
    Num_Pre_Test_VA_full = np.hstack((Test_Ang_CURR, Num_Pre_Test_VA))
    Num_Pre_Test_VM_full = np.hstack((Test_Mag_CURR, Num_Pre_Test_VM))

    Ang = Num_Pre_Test_VA_full * np.pi / 180.0
    Vol = Num_Pre_Test_VM_full * np.exp(1j * Ang)

    I = (Y @ Vol.T).T
    Right = Vol * np.conjugate(I)

    Sbase = 100.0
    Left = (PG_In - PD) / Sbase + 1j * (QG_In - QD) / Sbase

    PF_Mat = np.abs(Right - Left)
    Max_PF_Mat = PF_Mat.mean()

    # 输出
    print(f"=== {SFX} 集结果（{Num_Pre_Test_PG.shape[0]} 样本）===")
    print(f"成本相对偏差（均值）: {Cost_Diff_Mean:.4e}")
    print(f"PG 越界率: {Rate_InFea_PG:.3%}，平均越界幅度: {Val_InFea_PG:.4e}")
    print(f"QG 越界率: {Rate_InFea_QG:.3%}，平均越界幅度: {Val_InFea_QG:.4e}")
    print(f"VM 越界率: {Rate_InFea_VM:.3%}，平均越界幅度: {Val_InFea_VM:.4e}")
    print(f"潮流不匹配平均值: {Max_PF_Mat:.4e}")
