import numpy as np


def Data_Generated(S, flag):
    """
    按场景标记 S 读取并整理 OPF 相关数据（负荷、发电、节点电压），
    生成用于建模的输入与输出矩阵，并额外返回原始 PD/QD 拼接与
    第一列电压幅值/相角（便于作为参考/基准）。

    参数
    ----
    S : str
        文件名后缀，用于区分不同场景（例如 'L' 轻载、'H' 重载等）。
    flag : int
        控制分支；当前仅在 flag == 1 时执行加载逻辑，其他值直接 pass
        （注意：这会导致变量未定义，调用方需自行保证 flag==1）。

    返回
    ----
    Input        : np.ndarray, 形状 [N-1, Din]
        作为模型输入的负荷（PD/QD）组合，经列删除处理。
    Output_VMVA  : np.ndarray, 形状 [N-1, Dvmva]
        作为模型输出之一的电压（幅值+相角）拼接矩阵，经列删除处理。
    Output_PGQG  : np.ndarray, 形状 [N-1, Dpgqg]
        作为模型输出之二的发电（有功+无功）拼接矩阵。
    PDQD         : np.ndarray, 形状 [N-1, 2*nbus]
        原始 PD/QD 按列拼接（未做删列），常用于分析/反归一化。
    Vol1         : np.ndarray, 形状 [N-1, 1]
        电压幅值矩阵的第 1 列（索引 0），列向量形式。
    Vol2         : np.ndarray, 形状 [N-1, 1]
        电压相角矩阵的第 1 列（索引 0），列向量形式。

    说明
    ----
    - 本函数假设每个 .npy 的首行是占位/无效行，因此统一用 [1:, :] 去除。
    - 对电压输出和输入分别删除固定列（见 Dele_IDX 与 Input_IDX），常用于剔除平衡节点或无效量测。
    - 路径为写死的 Windows 绝对路径，如在其他环境运行请自行调整。
    - 若 flag != 1，则不会给返回变量赋值，直接 return 会报错；保持原逻辑不改动。
    """

    if flag == 1:
        # ---- 1) 构造文件路径（Windows 绝对路径） ----
        base = "E:\\OneDrive - The University of Hong Kong - Connect\\GAN_Project\\OPTNET\\14 Bus System\\Model\\DC3\\Data\\"
        data11 = base + "PD_save_WD_"   + S + ".npy"   # 负荷有功 PD
        data21 = base + "QD_save_WD_"   + S + ".npy"   # 负荷无功 QD
        data31 = base + "P_Gen_save_WD_"+ S + ".npy"   # 发电有功 PGen
        data41 = base + "Q_Gen_save_WD_"+ S + ".npy"   # 发电无功 QGen
        data51 = base + "Vol_mag_save_WD_" + S + ".npy"  # 节点电压幅值 Vmag
        data61 = base + "Vol_ang_save_WD_" + S + ".npy"  # 节点电压相角 Vang

        # ---- 2) 读取 .npy 文件 ----
        PD      = np.load(data11)   # [N, nbus]
        QD      = np.load(data21)   # [N, nbus]
        PGen    = np.load(data31)   # [N, ngen]
        QGen    = np.load(data41)   # [N, ngen]
        Vol_Mag = np.load(data51)   # [N, nbus]
        Vol_Ang = np.load(data61)   # [N, nbus]

        # ---- 3) 去掉首行（占位/无效） ----
        PD      = PD[1:, :]
        QD      = QD[1:, :]
        PGen    = PGen[1:, :]
        QGen    = QGen[1:, :]
        Vol_Mag = Vol_Mag[1:, :]
        Vol_Ang = Vol_Ang[1:, :]

        # ---- 4) 按列拼接得到三组矩阵 ----
        # VMVA: [Vmag, Vang]；PGQG: [PGen, QGen]；PDQD: [PD, QD]
        X    = np.concatenate((Vol_Mag, Vol_Ang), axis=1)
        X1   = np.concatenate((PGen, QGen),     axis=1)
        PDQD = np.concatenate((PD,   QD),       axis=1)

        # 便于阅读：重命名为“输出电压/输出发电”，以及“输入负荷”
        Output_VMVA = X
        Output_PGQG = X1
        Input       = PDQD

        # ---- 5) 删除指定列 ----
        # 对电压输出删除列（0 基索引）：如剔除平衡节点/无效量测
        Dele_IDX = [0, 14]
        Output_VMVA = np.delete(Output_VMVA, Dele_IDX, axis=1)

        # 对输入（PDQD）删除列：同理剔除不需要的节点量
        Input_IDX = [0, 6, 7, 14, 20, 21]
        Input = np.delete(Input, Input_IDX, axis=1)

        # ---- 6) 额外返回：电压幅值/相角的第一列（列向量） ----
        Vol1 = Vol_Mag[:, 0][:, np.newaxis]  # 幅值第 1 列 -> [N-1, 1]
        Vol2 = Vol_Ang[:, 0][:, np.newaxis]  # 相角第 1 列 -> [N-1, 1]
    else:
        # 保持原逻辑：什么都不做，但这会导致下方变量未定义。
        # 调用端必须保证 flag == 1。
        pass

    return Input, Output_VMVA, Output_PGQG, PDQD, Vol1, Vol2
