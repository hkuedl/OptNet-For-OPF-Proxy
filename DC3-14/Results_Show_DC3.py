# -*- coding: utf-8 -*-
import os
import copy
import random
import numpy as np
import torch

# 你已有的工具/数据函数
from Data_Generated import Data_Generated         # -> (Input, Output_VMVA, Output_PGQG, PDQD, Mag, Ang)
from Norm_Data import Norm_Data                   # 基于训练集统计做 [0,1] 归一化并划分 train/test
from Norm_Data2 import Norm_Data2                 # 提取 min/max（用于反归一化）
from Norm_Data3 import Norm_Data3                 # 仅切分，不归一化（用于物理量标签：PDQD/Mag/Ang）

# 电力系统评估所需
from pypower.api import case14
from makeYbus_my import makeYbus_my


# ========== 工具函数 ==========
def seed_torch(seed: int = 114514) -> None:
    """固定随机种子以提升复现性（注意：仍可能受算子/硬件影响）。"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ========== 主流程 ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    seed_torch()

    # --------------------------------------------
    # 评估开关：严格遵循 “前600 = light，后600 = heavy”
    # --------------------------------------------
    EVAL_MODE = "heavy"         # 可选 "light" 或 "heavy"
    IS_LIGHT = (EVAL_MODE.lower() == "light")
    HALF = 600                  # 每个子集的样本数
    LABEL = "Light" if IS_LIGHT else "Heavy"

    # --------------------------------------------
    # 1) 生成两份数据并按 2400(train)+600(test) 规则拼接
    # --------------------------------------------
    Input_A1, VMVA_A1, PGQG_A1, PDQD_A1, Mag_A1, Ang_A1 = Data_Generated("L", 1)
    Input_A2, VMVA_A2, PGQG_A2, PDQD_A2, Mag_A2, Ang_A2 = Data_Generated("H", 1)

    NUM_TRAIN_EACH = 2400
    NUM_TEST_EACH  = 600
    NUM_TRAIN = 4800
    NUM_TEST  = 1200

    # 训练 + 测试 的拼接（两份数据各取一段）
    Input = np.vstack((
        Input_A1[:NUM_TRAIN_EACH, :],
        Input_A2[:NUM_TRAIN_EACH, :],
        Input_A1[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
        Input_A2[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
    ))
    Output_VMVA = np.vstack((
        VMVA_A1[:NUM_TRAIN_EACH, :],
        VMVA_A2[:NUM_TRAIN_EACH, :],
        VMVA_A1[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
        VMVA_A2[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
    ))
    Output_PGQG = np.vstack((
        PGQG_A1[:NUM_TRAIN_EACH, :],
        PGQG_A2[:NUM_TRAIN_EACH, :],
        PGQG_A1[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
        PGQG_A2[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
    ))
    PDQD = np.vstack((
        PDQD_A1[:NUM_TRAIN_EACH, :],
        PDQD_A2[:NUM_TRAIN_EACH, :],
        PDQD_A1[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
        PDQD_A2[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
    ))
    Mag = np.vstack((
        Mag_A1[:NUM_TRAIN_EACH, :],
        Mag_A2[:NUM_TRAIN_EACH, :],
        Mag_A1[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
        Mag_A2[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
    ))
    Ang = np.vstack((
        Ang_A1[:NUM_TRAIN_EACH, :],
        Ang_A2[:NUM_TRAIN_EACH, :],
        Ang_A1[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
        Ang_A2[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
    ))

    # --------------------------------------------
    # 2) 归一化划分（基于训练集统计）
    # --------------------------------------------
    # 输出分两套：VMVA 与 PGQG，后续会把两者拼接为 36 维总输出以恢复物理量
    _, Test_Input, _, Test_VMVA = Norm_Data(Input, Output_VMVA, NUM_TRAIN, NUM_TEST)
    _, _, _, Test_PGQG = Norm_Data(Input, Output_PGQG, NUM_TRAIN, NUM_TEST)

    # 物理量标签：仅切分，不归一化（电压幅值/相角、负荷）
    _, _, _, Test_Mag = Norm_Data3(Input, Mag,  NUM_TRAIN, NUM_TEST)
    _, _, _, Test_Ang = Norm_Data3(Input, Ang,  NUM_TRAIN, NUM_TEST)
    _, _, _, Test_PDQD = Norm_Data3(Input, PDQD, NUM_TRAIN, NUM_TEST)

    # --------------------------------------------
    # 3) 严格切分：前 600 = light，后 600 = heavy
    # --------------------------------------------
    Test_VMVA_Light, Test_VMVA_Heavy = Test_VMVA[:HALF, :], Test_VMVA[HALF:, :]
    Test_PGQG_Light, Test_PGQG_Heavy = Test_PGQG[:HALF, :], Test_PGQG[HALF:, :]
    Test_Input_Light, Test_Input_Heavy = Test_Input[:HALF, :], Test_Input[HALF:, :]

    Test_Mag_Light, Test_Mag_Heavy = Test_Mag[:HALF, :], Test_Mag[HALF:, :]
    Test_Ang_Light, Test_Ang_Heavy = Test_Ang[:HALF, :], Test_Ang[HALF:, :]
    Test_PDQD_Light, Test_PDQD_Heavy = Test_PDQD[:HALF, :], Test_PDQD[HALF:, :]

    # “总输出” = PGQG(10) + VMVA(26) -> 36 维（用于统一反归一化）
    Test_Total_Light = np.hstack((Test_PGQG_Light, Test_VMVA_Light))
    Test_Total_Heavy = np.hstack((Test_PGQG_Heavy, Test_VMVA_Heavy))

    # --------------------------------------------
    # 4) 取反归一化所需的 min/max（对 36 维总输出）
    # --------------------------------------------
    Output_Total_All = np.hstack((Output_PGQG, Output_VMVA))
    _, _, Out_Max_All, Out_Min_All = Norm_Data2(Input, Output_Total_All, NUM_TRAIN, NUM_TEST)

    # --------------------------------------------
    # 5) 读取 DC3 模型在子集上的预测（文件名请与实际一致）
    #    DC3 预测向量约为：PG(5) + QG(5) + VM(14) + VA(14) = 38 维
    #    其中 PG/QG 需要 *100（MW/Mvar），VM 为 p.u.，VA 为弧度
    # --------------------------------------------
    if IS_LIGHT:
        pred_path = "Res_DC3\\Pre_Test_Light_DC3.pt"
        Test_Out_CURR = Test_Total_Light
        PDQD_eval, Mag_eval, Ang_eval = Test_PDQD_Light, Test_Mag_Light, Test_Ang_Light
    else:
        pred_path = "Res_DC3\\Pre_Test_Heavy_DC3.pt"
        Test_Out_CURR = Test_Total_Heavy
        PDQD_eval, Mag_eval, Ang_eval = Test_PDQD_Heavy, Test_Mag_Heavy, Test_Ang_Heavy

    pred_tensor = torch.load(pred_path, map_location="cpu")   # 形状应为 [600, 38]
    Pred = np.asarray(pred_tensor.detach().cpu().numpy())

    # 预测分块（DC3 输出）
    NUM_PG, NUM_QG, NUM_VM, NUM_VA = 5, 5, 14, 14
    slc_PG = slice(0, NUM_PG)
    slc_QG = slice(NUM_PG, NUM_PG + NUM_QG)
    slc_VM = slice(NUM_PG + NUM_QG, NUM_PG + NUM_QG + NUM_VM)
    slc_VA = slice(NUM_PG + NUM_QG + NUM_VM, NUM_PG + NUM_QG + NUM_VM + NUM_VA)

    Pre_PG = Pred[:, slc_PG] * 100.0       # MW
    Pre_QG = Pred[:, slc_QG] * 100.0       # Mvar
    Pre_VM = Pred[:, slc_VM]               # p.u.
    Pre_VA = Pred[:, slc_VA]               # rad

    # 反归一化 ground truth（36 维：PGQG(10)+VMVA(26)）
    Num_GT = Test_Out_CURR * (Out_Max_All - Out_Min_All) + Out_Min_All
    # ground truth 分块（训练口径：VM/VA 仅 13 个）
    NUM_VM_GT, NUM_VA_GT = 13, 13
    GT_PG = Num_GT[:, 0:NUM_PG]                            # MW
    # GT_QG = Num_GT[:, NUM_PG:NUM_PG + NUM_QG]            # 若需可启用
    # GT_VM = Num_GT[:, NUM_PG + NUM_QG:NUM_PG + NUM_QG + NUM_VM_GT]
    # GT_VA = Num_GT[:, NUM_PG + NUM_QG + NUM_VM_GT:NUM_PG + NUM_QG + NUM_VM_GT + NUM_VA_GT]

    # --------------------------------------------
    # 6) 指标计算
    # --------------------------------------------
    # 6.1 成本相对误差（a2*PG^2 + a1*PG）
    case = case14()
    # 与原脚本一致的 PMAX 调整
    case['gen'][0, 8] = 100
    case['gen'][1, 8] = 100
    case['gen'][2, 8] = 80
    case['gen'][3, 8] = 60
    case['gen'][4, 8] = 60

    Gencost = case['gencost'][:, 4:7]
    a2 = Gencost[:, 0][np.newaxis, :]
    a1 = Gencost[:, 1][np.newaxis, :]

    Cost_Pre = (a2[:, :NUM_PG] * Pre_PG**2 + a1[:, :NUM_PG] * Pre_PG).sum(axis=1)
    Cost_GT  = (a2[:, :NUM_PG] * GT_PG**2  + a1[:, :NUM_PG] * GT_PG ).sum(axis=1)
    Cost_rel_err = np.mean(np.abs(Cost_Pre - Cost_GT) / np.maximum(np.abs(Cost_GT), 1e-8))

    # 6.2 物理越界率（预测与上下限比较）
    PG_Max, PG_Min = case['gen'][:, 8][:NUM_PG], case['gen'][:, 9][:NUM_PG]
    QG_Max, QG_Min = case['gen'][:, 3][:NUM_QG], case['gen'][:, 4][:NUM_QG]
    VM_Max, VM_Min = case['bus'][:, 11][:NUM_VM], case['bus'][:, 12][:NUM_VM]

    def viol_rate(x, xmin, xmax):
        """越界比例（逐元素判断）。"""
        return np.mean((x < xmin) | (x > xmax))

    Rate_PG = viol_rate(Pre_PG, PG_Min, PG_Max)
    Rate_QG = viol_rate(Pre_QG, QG_Min, QG_Max)
    Rate_VM = viol_rate(Pre_VM, VM_Min, VM_Max)

    # 6.3 潮流一致性（功率注入平衡误差）
    Ybus = makeYbus_my(case['baseMVA'], case['bus'], case['branch'])
    nb = case['bus'].shape[0]
    GEN_BUS_INDEX = np.array([1, 2, 3, 6, 8]) - 1  # 0-based

    # 负荷（物理量）
    PD = copy.deepcopy(PDQD_eval[:, :nb])   # MW
    QD = copy.deepcopy(PDQD_eval[:, nb:])   # Mvar

    # 将预测 PG/QG 填入相应发电机母线，其它母线为 0
    PG_full = np.zeros((Pre_PG.shape[0], nb));  PG_full[:, GEN_BUS_INDEX] = Pre_PG
    QG_full = np.zeros((Pre_QG.shape[0], nb));  QG_full[:, GEN_BUS_INDEX] = Pre_QG

    # 复电压（VA 为弧度，VM 为 p.u.）
    V = Pre_VM * np.exp(1j * Pre_VA)
    I = (Ybus @ V.T).T
    S_right = V * np.conjugate(I)           # V * conj(I)

    # 左端注入（Sbase=100，转标幺）
    Sbase = 100.0
    S_left = (PG_full - PD) / Sbase + 1j * (QG_full - QD) / Sbase

    PF_gap = np.abs(S_right - S_left)       # 节点复功率不平衡
    PF_gap_mean = PF_gap.mean()

    # --------------------------------------------
    # 7) 打印结果
    # --------------------------------------------
    print(f"=== {LABEL} 集结果（{HALF} 样本）===")
    print(f"成本相对误差（均值） : {Cost_rel_err:.4e}")
    print(f"PG 越界率           : {Rate_PG:.3%}")
    print(f"QG 越界率           : {Rate_QG:.3%}")
    print(f"VM 越界率           : {Rate_VM:.3%}")
    print(f"潮流不匹配平均值     : {PF_gap_mean:.4e}")
