import numpy as np


def Data_Load_WD():
    """
    功能：
        从固定目录加载 6 个 .npy 数据文件（负荷、发电、节点电压），
        进行基础预处理（去掉首行、拼接、删列），并返回输入与两个输出矩阵。

    返回：
        Input         : np.ndarray，形状约为 [N-1, PD列数 + QD列数]，输入特征（负荷：有功+无功）
        Output_VMVA   : np.ndarray，形状约为 [N-1, (Vol_Mag列数 + Vol_Ang列数 - 被删除列数)]，
                        输出1（电压幅值+相角，删除指定列后）
        Output_PGQG   : np.ndarray，形状约为 [N-1, (PGen列数 + QGen列数)]，
                        输出2（发电有功+无功）
    说明：
        - 代码默认所有 .npy 的样本数与节点维度一一对应，且行数一致。
        - 通过切片 [1:, :] 去掉每个数组的首行（常见用途：去除表头或占位行）。
        - 通过 np.concatenate 在列维度上拼接。
        - 通过 np.delete(Output_VMVA, [0, 4], axis=1) 删除第 1 与第 5 列（示例：去除平衡节点等）。
    """

    # —— 构造 6 个数据文件的绝对路径 —— #
    data1 = "C:\\Users\\EEE\\" + "PD_save" + ".npy"       # 负荷有功（PD）
    data2 = "C:\\Users\\EEE\\" + "QD_save" + ".npy"       # 负荷无功（QD）
    data3 = "C:\\Users\\EEE\\" + "P_Gen_save" + ".npy"    # 发电有功（PGen）
    data4 = "C:\\Users\\EEE\\" + "Q_Gen_save" + ".npy"    # 发电无功（QGen）
    data5 = "C:\\Users\\EEE\\" + "Vol_mag_save" + ".npy"  # 节点电压幅值（Vol_Mag）
    data6 = "C:\\Users\\EEE\\" + "Vol_ang_save" + ".npy"  # 节点电压相角（Vol_Ang）

    # —— 读取 .npy 文件到内存 —— #
    PD = np.load(data1)        # 形状示例：[N, n_bus]，以下同理
    QD = np.load(data2)
    PGen = np.load(data3)
    QGen = np.load(data4)
    Vol_Mag = np.load(data5)
    Vol_Ang = np.load(data6)

    # —— 去掉首行（常见场景：首行为占位/无效/时间戳等） —— #
    PD = PD[1:, :]
    QD = QD[1:, :]
    PGen = PGen[1:, :]
    QGen = QGen[1:, :]
    Vol_Mag = Vol_Mag[1:, :]
    Vol_Ang = Vol_Ang[1:, :]

    # —— 按列拼接：电压（幅值+相角）、发电（有功+无功）、负荷（有功+无功） —— #
    X = np.concatenate((Vol_Mag, Vol_Ang), axis=1)  # [N-1, n_bus*2]
    X1 = np.concatenate((PGen, QGen), axis=1)       # [N-1, n_gen*2]
    PDQD = np.concatenate((PD, QD), axis=1)         # [N-1, n_bus*2]

    # —— 输出矩阵（电压、发电）与输入矩阵（负荷） —— #
    Output_VMVA = X[:, :]     # 电压幅值+相角
    Output_PGQG = X1[:, :]    # 发电有功+无功
    Input = PDQD              # 输入：负荷有功+无功

    # —— 删除电压输出里的指定列（例如：删除平衡节点或量测缺失节点） —— #
    Dele_IDX = [0, 4]         # 注意：基于 0 起始的列索引；请确保索引在有效范围内
    Output_VMVA = np.delete(Output_VMVA, Dele_IDX, 1)

    # —— 返回：输入与两个输出 —— #
    return Input, Output_VMVA, Output_PGQG
