# -*- coding: utf-8 -*-
"""
脚本功能概述
-----------
- 以 PYPOWER 的 case4gs 为基础，调整机组上下限/成本等参数；
- 通过对负荷进行随机缩放，循环调用 AC-OPF（runopf）求解；
- 对求解成功的样本，收集：节点负荷(P/Q)、机组出力(Pg/Qg)、电压幅值/相角，并保存为 .npy 文件。

矩阵列含义速查（MATPOWER/PYPOWER）
---------------------------------
bus（至少 13 列）:
 0 BUS_I, 1 BUS_TYPE, 2 PD, 3 QD, 4 GS, 5 BS, 6 BUS_AREA,
 7 VM, 8 VA, 9 BASE_KV, 10 ZONE, 11 VMAX, 12 VMIN

gen（常见前 10 列，后续可能扩展到 21 列以上）:
 0 GEN_BUS, 1 PG, 2 QG, 3 QMAX, 4 QMIN, 5 VG, 6 MBASE,
 7 STATUS, 8 PMAX, 9 PMIN, ...

branch（至少 13 列）:
 0 F_BUS, 1 T_BUS, 2 BR_R, 3 BR_X, 4 BR_B, 5 RATE_A, 6 RATE_B,
 7 RATE_C, 8 TAP, 9 SHIFT, 10 BR_STATUS, 11 ANGMIN, 12 ANGMAX
"""

import numpy as np
from pypower.api import case30,ppoption, runopf, runpf, case118, case14, case9, case4gs
import copy

# 加载基础算例
data_in = case4gs()                          # 4 节点系统（含 2 台机组）
data_in_2 = case14()                         # 14 节点系统（用于提取 gencost）

# ---------- 机组参数调整（索引按上面的 gen 列含义） ----------
data_in['gen'][1,8] = 300                    # 第2台机组 PMAX = 300
data_in['gen'][0,9] = data_in['gen'][0,8]*0.2# 第1台机组 PMIN = 0.2*PMAX
data_in['gen'][1,9] = data_in['gen'][1,8]*0.2# 第2台机组 PMIN = 0.2*PMAX
data_in['gen'][0,1] = 200                    # 第1台机组初始 PG = 200（初值，不是限值）

# 备份原始负荷（bus 的 PD/QD）
PD = copy.deepcopy(data_in['bus'][:,2])      # 有功负荷
QD = copy.deepcopy(data_in['bus'][:,3])      # 无功负荷

# 预分配保存容器（先放一行占位，后续逐次 vstack）
PD_Save = np.zeros((np.size(PD,0),))
QD_Save = np.zeros((np.size(QD,0),))
PGEN_Save = np.zeros((np.size(data_in['gen'][:,1],0)))
QGEN_Save = np.zeros((np.size(data_in['gen'][:,2],0)))
VM_Save = np.zeros((np.size(data_in['bus'][:,2],0)))
VA_Save = np.zeros((np.size(data_in['bus'][:,2],0)))

# 采用 case14 的前两行 gencost 作为本算例的机组成本（对应 2 台机组）
data_in['gencost'] = copy.deepcopy(data_in_2['gencost'][0:2,:])

# ---------- 将外部编号统一 +1（非必须，改变外部号但不影响内部映射） ----------
data_in['bus'][:,0] = data_in['bus'][:,0] + 1
data_in['gen'][:,0] = data_in['gen'][:,0] + 1
data_in['branch'][:,0] = data_in['branch'][:,0] + 1
data_in['branch'][:,1] = data_in['branch'][:,1] + 1

# 调整机组无功上下限（QMAX/QMIN）
data_in['gen'][0,3] = 200
data_in['gen'][0,4] = -150
data_in['gen'][1,3] = 200
data_in['gen'][1,4] = -150

# 裁剪到标准列数，避免多余列干扰
data_in['bus'] = data_in['bus'][:,0:13]      # bus 至少 13 列
data_in['gen'] = data_in['gen'][:,0:21]      # gen 常见到 21 列（含成本等）
data_in['branch'] = data_in['branch'][:,0:13]# branch 至少 13 列

# 直接修改两个节点的基础负荷（示例）
data_in['bus'][2,2] = 100                    # 第3个节点 PD=100
data_in['bus'][3,2] = 50                     # 第4个节点 PD=50
data_in['bus'][2,3] = 95                     # 第3个节点 QD=95
data_in['bus'][3,3] = 48                     # 第4个节点 QD=48

# ---------- 采样与 OPF 执行 ----------
num_bus = np.size(PD,0)                      # 节点数
NUM = 5000                                   # 采样次数
Index_Delete = np.zeros((1,))                # 记录 OPF 失败的样本索引
Obj = np.zeros((1,1))                        # 记录目标函数值（系统成本）

for i in range(NUM):
    # 为每个节点生成随机缩放因子（主因子 alpha + 小扰动 beta/gama）
    alpha = np.random.uniform(0.450, 1.225, (num_bus))
    beta = np.random.uniform(-0.025, 0.025, (num_bus))
    gama = np.random.uniform(-0.0025, 0.0025, (num_bus))
    Con = alpha + beta + gama                # 合成缩放因子，约在 [0.4225, 1.2525] 之间

    # 按缩放因子生成一组负荷样本
    P_sample = Con * PD
    Q_sample = Con * QD

    # ⚠️ 仅替换前两个节点的有功负荷（bus[0], bus[1] 的 PD）
    data_in['bus'][0:2,2] = copy.deepcopy(P_sample[0:2])
    # 这里未同步写回 QD（如下行是可能的做法）

    # 运行 AC-OPF（返回字典，含求解状态/机组/节点结果/目标值等）
    ResRes = runopf(data_in)

    if ResRes['success'] == True:
        # 成功则累计保存输入/输出
        PD_Save = np.vstack((PD_Save, data_in['bus'][:,2]))   # 本次使用的 PD
        QD_Save = np.vstack((QD_Save, data_in['bus'][:,3]))   # 本次使用的 QD（注意可能未改动）
        PGEN_Save = np.vstack((PGEN_Save, ResRes['gen'][:,1]))# 机组有功 Pg
        QGEN_Save = np.vstack((QGEN_Save, ResRes['gen'][:,2]))# 机组无功 Qg
        VM_Save = np.vstack((VM_Save, ResRes['bus'][:,7]))    # 节点电压幅值 VM
        VA_Save = np.vstack((VA_Save, ResRes['bus'][:,8]))    # 节点电压相角 VA
        Obj = np.vstack((Obj, ResRes['f']))                   # 目标函数值（总成本）
    else:
        # 记录失败样本编号
        Index_Delete = np.vstack((Index_Delete, i))

# ---------- 结果保存 ----------
Path = "C:\\Users\\EEE\\OneDrive - The University of Hong Kong\\Desktop\\PY-Projects\\ADMM_OPF\\GP_NN_OPF\\Data\\"

data1 = Path + "PD_save" + ".npy"
data2 = Path + "QD_save" + ".npy"
data3 = Path + "P_Gen_save" + ".npy"
data4 = Path + "Q_Gen_save" + ".npy"
data5 = Path + "Vol_mag_save" + ".npy"
data6 = Path + "Vol_ang_save" + ".npy"

np.save(data1, PD_Save)
np.save(data2, QD_Save)
np.save(data3, PGEN_Save)
np.save(data4, QGEN_Save)
np.save(data5, VM_Save)
np.save(data6, VA_Save)
