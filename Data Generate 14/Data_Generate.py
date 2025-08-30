import matplotlib.pyplot as plt
import numpy as np
from pypower.api import case30,ppoption, runopf, runpf, case118, case14, case9
import copy

# === 载入测试系统 ===
# 这里使用 PYPOWER 的 IEEE 14 节点算例（结构与 MATPOWER 一致）
# data_in 是一个包含 'bus', 'branch', 'gen', 'gencost' 等键的字典
data_in = case14()

# === 调整发电机出力上限（PMAX） ===
# MATPOWER/ PYPOWER 中 gen 矩阵列从 1 开始编号：col9=PMAX，col10=PMIN
# Python 0 基索引下，PMAX 对应 gen[:, 8]
data_in['gen'][0,8] = 100
data_in['gen'][1,8] = 100
data_in['gen'][2,8] = 80
data_in['gen'][3,8] = 60
data_in['gen'][4,8] = 60

# === 读取原始负荷（PD/QD）作为基准谱 ===
# bus 矩阵（1 基）col3=PD, col4=QD；Python 0 基对应 col=2/3
PD = copy.deepcopy(data_in['bus'][:,2])
QD = copy.deepcopy(data_in['bus'][:,3])

# === 预分配保存数组（首行是占位的 0，后面会逐次 vstack 追加）===
PD_Save = np.zeros((np.size(PD,0),))             # 每次保存一整行 PD（长度=bus 数）
QD_Save = np.zeros((np.size(QD,0),))
PGEN_Save = np.zeros((np.size(data_in['gen'][:,1],0)))  # gen（1 基）col2=PG -> 0 基 col=1
QGEN_Save = np.zeros((np.size(data_in['gen'][:,2],0)))  # gen（1 基）col3=QG -> 0 基 col=2
VM_Save = np.zeros((np.size(data_in['bus'][:,2],0)))    # bus（1 基）col8=VM -> 0 基 col=7
VA_Save = np.zeros((np.size(data_in['bus'][:,2],0)))    # bus（1 基）col9=VA -> 0 基 col=8

num_bus = np.size(PD,0)  # 节点数
NUM = 3000               # 采样次数（要生成的场景数量）
Index_Delete = np.zeros((1,))  # 记录 OPF 失败的样本索引（首行占位）

for i in range(NUM):

    # === 构造负荷缩放因子 Con（三部分相加）===
    # alpha：主缩放（场景强度）——
    #   - 轻负荷：np.random.uniform(0.85, 1.150, (num_bus))
    #   - 重负荷：np.random.uniform(0.550, 1.650, (num_bus))
    # beta：较小的随机扰动（±2.5%）
    # gama：更小的随机扰动（±0.25%）
    # 最终 Con = alpha + beta + gama（逐节点独立缩放）
    # alpha = np.random.uniform(0.85, 1.150, (num_bus)) # light scenario
    alpha = np.random.uniform(0.550, 1.650, (num_bus)) # heavy scenario
    beta = np.random.uniform(-0.025, 0.025, (num_bus))
    gama = np.random.uniform(-0.0025, 0.0025, (num_bus))
    Con = alpha + beta + gama

    # === 生成该场景下的负荷（单位通常为系统标么值/ MW/Mvar，取决于 case 数据）===
    P_sample = Con * PD
    Q_sample = Con * QD

    # 将该场景负荷写回到 case
    data_in['bus'][:,2] = copy.deepcopy(P_sample)  # PD
    data_in['bus'][:,3] = copy.deepcopy(Q_sample)  # QD

    # === 运行交流最优潮流 OPF ===
    # 返回 ResRes，其中：
    #   - ResRes['success'] 布尔量，是否求解成功
    #   - ResRes['gen']     更新后的发电机出力矩阵
    #   - ResRes['bus']     更新后的节点状态（电压幅值/相角等）
    ResRes = runopf(data_in)

    if ResRes['success'] == True:
        # 成功：将该场景的各类量追加保存
        PD_Save   = np.vstack((PD_Save,   data_in['bus'][:,2]))
        QD_Save   = np.vstack((QD_Save,   data_in['bus'][:,3]))
        PGEN_Save = np.vstack((PGEN_Save, ResRes['gen'][:,1]))  # PG（0 基 col=1）
        QGEN_Save = np.vstack((QGEN_Save, ResRes['gen'][:,2]))  # QG（0 基 col=2）
        VM_Save   = np.vstack((VM_Save,   ResRes['bus'][:,7]))  # VM（0 基 col=7）
        VA_Save   = np.vstack((VA_Save,   ResRes['bus'][:,8]))  # VA（0 基 col=8）
    else:
        # 失败：记录该样本索引，便于后续剔除或分析
        Index_Delete = np.vstack((Index_Delete, i))

# === 保存数据到本地 .npy 文件 ===
# S 用于区分场景（例如 'L' 代表 light；当前脚本用的是 heavy，但这里仍以 'L' 为文件后缀）
S = "L"
data1 = "C:\\Users\\EEE\\OneDrive - The University of Hong Kong\\Desktop\\PY-Projects\\OPF_OPTNET_WD\\Data\\" + "PD_save_WD_" + S + ".npy"
data2 = "C:\\Users\\EEE\\OneDrive - The University of Hong Kong\\Desktop\\PY-Projects\\OPF_OPTNET_WD\\Data\\" + "QD_save_WD_" + S + ".npy"
data3 = "C:\\Users\\EEE\\OneDrive - The University of Hong Kong\\Desktop\\PY-Projects\\OPF_OPTNET_WD\\Data\\" + "P_Gen_save_WD_" + S + ".npy"
data4 = "C:\\Users\\EEE\\OneDrive - The University of Hong Kong\\Desktop\\PY-Projects\\OPF_OPTNET_WD\\Data\\" + "Q_Gen_save_WD_" + S + ".npy"
data5 = "C:\\Users\\EEE\\OneDrive - The University of Hong Kong\\Desktop\\PY-Projects\\OPF_OPTNET_WD\\Data\\" + "Vol_mag_save_WD_" + S + ".npy"
data6 = "C:\\Users\\EEE\\OneDrive - The University of Hong Kong\\Desktop\\PY-Projects\\OPF_OPTNET_WD\\Data\\" + "Vol_ang_save_WD_" + S + ".npy"
data7 = "C:\\Users\\EEE\\OneDrive - The University of Hong Kong\\Desktop\\PY-Projects\\OPF_OPTNET_WD\\Data\\" + "Index_Delete_WD" + S + ".npy"

np.save(data1, PD_Save)     # 形状约为 [1+Nsucc, nb]，首行是占位 0
np.save(data2, QD_Save)
np.save(data3, PGEN_Save)   # 形状约为 [1+Nsucc, ngen]
np.save(data4, QGEN_Save)
np.save(data5, VM_Save)     # 形状约为 [1+Nsucc, nb]
np.save(data6, VA_Save)
np.save(data7, Index_Delete)  # 记录失败样本索引（首行占位 0）
