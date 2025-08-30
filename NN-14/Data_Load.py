import numpy as np


def Data_Load(S, flag):
    """
    读取并整理 OPF 数据集（负荷、发电、节点电压），生成输入与两个输出张量（矩阵）。

    参数
    ----
    S : str
        数据后缀（如 'L' / 'H'，用于区分轻载/重载等不同场景的文件）。
    flag : int
        控制分支；当前仅在 flag == 1 时执行加载逻辑，其他情况直接 pass（将导致未定义变量）。

    返回
    ----
    Input : ndarray, shape = [N-1, ?]
        作为模型输入的 PD/QD 组合（经过列删除）。
    Output_VMVA : ndarray, shape = [N-1, ?]
        作为模型输出之一的电压（幅值 + 相角）拼接矩阵（经过列删除）。
    Output_PGQG : ndarray, shape = [N-1, ?]
        作为模型输出之二的发电（有功 + 无功）拼接矩阵。

    说明
    ----
    - 本函数假设 .npy 文件的首行是占位/无效行，因此统一使用 [1:, :] 去除首行。
    - `np.delete(..., axis=1)` 会删除指定的列索引（0 基），常用于剔除平衡节点或无效量测。
    - 若 flag != 1，则不会执行任何赋值操作，直接 return 会报错；此处保持原逻辑不动。
    """

    if flag == 1:
        # -------- 构造文件路径（Windows 绝对路径） --------
        base = "E:\\OneDrive - The University of Hong Kong - Connect\\GAN_Project\\OPTNET\\14 Bus System\\Model\\NN\\Data\\"
        data11 = base + "PD_save_WD_" + S + ".npy"  # 负荷有功 PD
        data21 = base + "QD_save_WD_" + S + ".npy"  # 负荷无功 QD
        data31 = base + "P_Gen_save_WD_" + S + ".npy"  # 发电有功 PGen
        data41 = base + "Q_Gen_save_WD_" + S + ".npy"  # 发电无功 QGen
        data51 = base + "Vol_mag_save_WD_" + S + ".npy"  # 节点电压幅值 Vmag
        data61 = base + "Vol_ang_save_WD_" + S + ".npy"  # 节点电压相角 Vang

        # -------- 读取 .npy 文件 --------
        PD = np.load(data11)  # shape: [N, nb]
        QD = np.load(data21)  # shape: [N, nb]
        PGen = np.load(data31)  # shape: [N, ng]
        QGen = np.load(data41)  # shape: [N, ng]
        Vol_Mag = np.load(data51)  # shape: [N, nb]
        Vol_Ang = np.load(data61)  # shape: [N, nb]

        # -------- 去掉首行（占位/无效） --------
        PD = PD[1:, :]
        QD = QD[1:, :]
        PGen = PGen[1:, :]
        QGen = QGen[1:, :]
        Vol_Mag = Vol_Mag[1:, :]
        Vol_Ang = Vol_Ang[1:, :]

        # -------- 按列拼接：电压、发电、负荷 --------
        # VMVA: [Vmag, Vang]；PGQG: [PGen, QGen]；Input: [PD, QD]
        X = np.concatenate((Vol_Mag, Vol_Ang), axis=1)  # shape: [N-1, nb*2]
        X1 = np.concatenate((PGen, QGen), axis=1)  # shape: [N-1, ng*2]
        PDQD = np.concatenate((PD, QD), axis=1)  # shape: [N-1, nb*2]

        Output_VMVA = X[:, :]  # 电压（幅值+相角）输出
        Output_PGQG = X1[:, :]  # 发电（有功+无功）输出
        Input = PDQD  # 模型输入：负荷（有功+无功）

        # -------- 删除特定列（0 基索引）--------
        # 典型用法：剔除平衡节点或无效/冗余量测列。请确保索引在有效范围内。
        Dele_IDX = [0, 14]  # 对 VMVA（电压）删除的列
        Output_VMVA = np.delete(Output_VMVA, Dele_IDX, axis=1)

        Input_IDX = [0, 6, 7, 14, 20, 21]  # 对输入（PDQD）删除的列
        Input = np.delete(Input, Input_IDX, axis=1)

        # -------- 可选：单独取电压的第 1 列（索引 0）做别处用途 --------
        Vol1 = Vol_Mag[:, 0]  # shape: [N-1,]
        Vol2 = Vol_Ang[:, 0]  # shape: [N-1,]
        Vol1 = Vol1[:, np.newaxis]  # shape: [N-1, 1]
        Vol2 = Vol2[:, np.newaxis]  # shape: [N-1, 1]
    else:
        # 保持与原逻辑一致（不做任何处理），但注意：直接 return 会因变量未定义而报错
        pass

    return Input, Output_VMVA, Output_PGQG
