import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import copy
from Data_Load_WD import *          # 加载数据
import os
import random
from Model_OPT import OptNetLatent   # 可微 QP 模型（OptNet 风格）
import time

# ========== 固定随机种子，保证实验可复现 ==========
def seed_torch(seed=114514):  # 114544
    """
    设定 Python、NumPy、PyTorch 的随机数种子，并控制 cuDNN 的确定性，
    以尽可能保证结果可复现（注意：某些算子或不同硬件仍可能导致轻微差异）。
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止 Python 的哈希随机化（影响 dict/set 顺序）
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # 多 GPU 场景
    torch.backends.cudnn.benchmark = False    # 关闭算法自动搜索（避免非确定性）
    torch.backends.cudnn.deterministic = True # 使用确定性算法

# ========== 训练一步（一个 epoch 内的遍历） ==========
def train_fn(dataTrain, data_real_Train, model, crit, opt):
    """
    参数:
        dataTrain        : Tensor，训练输入，形状 [N, T, Din] 或 [N, 1, Din]
        data_real_Train  : Tensor，训练标签，形状 [N, T, Dout] 或 [N, 1, Dout]
        model            : nn.Module，可前向得到预测
        crit             : 损失函数（如 nn.MSELoss）
        opt              : 优化器（如 Adam/SGD）
    说明:
        - 使用随机打乱索引 per 后，按全局变量 batch_size 分批训练。
        - 将标签 out_my reshape 为 [batch_size, size1] 与模型输出对齐。
        - 注意: 该函数依赖外部的全局 batch_size 变量，请在外部提前设置。
    返回:
        float，当前 epoch 的平均 loss（按 batch 均值）
    """
    model.train()
    loss_total_final = 0

    per = torch.randperm(dataTrain.size(dim=0))              # 打乱样本索引
    NUM = int((per.size(dim=0) / batch_size))               # 可整除的 batch 数（不足一批的尾部被丢弃）

    for idx in range(1, NUM + 1):
        Label_SUM = per[(idx - 1) * batch_size:idx * batch_size, ]  # 当前 batch 的索引

        in_my = dataTrain[Label_SUM, :, :]                 # [B, T, Din]
        out_my = data_real_Train[Label_SUM, :, :]          # [B, T, Dout]

        out_pre = model(in_my)                             # 模型前向，期望输出形状 [B, Dout]

        size1 = out_my.size(2)                             # Dout
        out_my = out_my.reshape((batch_size, size1))       # [B, Dout]，与 out_pre 对齐
        loss = crit(out_pre, out_my)                       # 计算损失

        loss_total_final += loss.item()

        # 反向传播与参数更新
        opt.zero_grad()
        loss.backward()
        opt.step()

        del out_pre                                        # 及时释放显存

    return loss_total_final / NUM

# ========== 测试/验证一步 ==========
def test_fn(dataTest, data_real_Test, model, crit, opt):
    """
    参数:
        dataTest        : Tensor，测试输入，形状 [N, T, Din] 或 [N, 1, Din]
        data_real_Test  : Tensor，测试标签，形状 [N, T, Dout] 或 [N, 1, Dout]
        model, crit, opt: 与训练同义（opt 在此未用，仅为接口统一）
    说明:
        - eval 模式下不启用 dropout/bn 更新，但此处未加 no_grad()，仍会构图。
          若希望更省显存，建议在调用时用 with torch.no_grad(): 包裹（此处保持原逻辑不改）。
    返回:
        float，平均测试损失
    """
    model.eval()
    loss_total_final = 0

    per = torch.randperm(dataTest.size(dim = 0))           # 测试集同样随机取 batch
    NUM = int((per.size(dim = 0)/batch_size))

    for idx in range(1, NUM + 1):
        Label_SUM = per[(idx - 1) * batch_size:idx * batch_size, ]

        in_my = dataTest[Label_SUM, :, :]                  # [B, T, Din]
        out_my = data_real_Test[Label_SUM, :, :]           # [B, T, Dout]

        out_pre = model(in_my)                             # [B, Dout]

        size1 = out_my.size(2)                             # Dout
        out_my = out_my.reshape((batch_size, size1))       # [B, Dout]

        loss1 = crit(out_pre, out_my)

        del out_pre

        loss_total_final += loss1.item()
    return loss_total_final / NUM

# ========== 环境设置（GPU/内存策略） ==========
os.environ["CUDA_VISIBLE_DEVICES"] = '3'                   # 指定使用第 4 张 GPU（索引从 0 开始）
torch.backends.cudnn.benchmark = True                      # （训练阶段）允许 cuDNN 自动选择最快算法
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 调整 CUDA 内存碎片化策略

if __name__ == "__main__":
    seed_torch()                                           # 设定随机种子
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 预分配（示例/占位）：记录训练&测试损失（Area1）
    Mag_loss_Area1 = np.zeros((1, 1))
    Te_loss_Area1 = np.zeros((1, 1))

    # 载入原始数据（来自自定义 Data_Load_WD 模块）
    # 返回示例：
    #   Input_A1         : [N, 8]      （PD/QD 等输入特征）
    #   Output_VMVA_A1   : [N, 4]      （电压幅值/相角等）
    #   Output_PGQG_A1   : [N, 4]      （发电有功/无功）
    Input_A1, Output_VMVA_A1, Output_PGQG_A1 = Data_Load_WD()

    # 训练/测试样本数及“间隔”K（从训练尾部错开抽取测试片段）
    NUM_Sample_Train = 5000
    K = 4800
    NUM_Sample_Test = 200

    # 构造训练+测试集合（将训练前 5000 与测试段 200 按行拼接）
    Input = np.vstack((Input_A1[:NUM_Sample_Train, :],
                       Input_A1[NUM_Sample_Train + K:NUM_Sample_Train + K + NUM_Sample_Test, :]))
    Output_VMVA = np.vstack((Output_VMVA_A1[:NUM_Sample_Train, :],
                             Output_VMVA_A1[NUM_Sample_Train + K:NUM_Sample_Train + K + NUM_Sample_Test, :]))
    Output_PGQG = np.vstack((Output_PGQG_A1[:NUM_Sample_Train, :],
                             Output_PGQG_A1[NUM_Sample_Train + K:NUM_Sample_Train + K + NUM_Sample_Test, :]))

    # 仅保留“真实测试段”的输入副本（后续用于构造 200x200 网格）
    Input_Test = copy.deepcopy(Input_A1[NUM_Sample_Train + K:NUM_Sample_Train + K + NUM_Sample_Test, :])

    # ========== 构造一个 200x200 的输入网格 ==========
    # 思路：固定除第 0/1 列外的其余输入为某基准样本值；第 0 列取第 i 行的值，第 1 列取第 j 行的值，
    # 从而扫描两个关键输入维度的二维平面，对模型输出进行可视化/评估。
    Input_Full = np.zeros((200*200, 8))    # 假定输入维度为 8（PD/QD 等）

    for i in range(200):
        for j in range(200):
            Input_Full[i*200 + j, :] = copy.deepcopy(Input_Test[0, :])  # 先拷贝一份基准向量
            Input_Full[i*200 + j, 0] = copy.deepcopy(Input_Test[i, 0])  # 替换第 0 列（例如 PD_0）
            Input_Full[i*200 + j, 1] = copy.deepcopy(Input_Test[j, 1])  # 替换第 1 列（例如 QD_0）

    # 训练/测试样本数（此处仅进行模型推理，不做训练；变量名沿用）
    NUM_Train = 5000
    NUM_Test = 200*200

    # 简单缩放（与训练时保持一致的归一化/标准化非常重要；此处按原逻辑除以 100）
    Input_Full = (Input_Full) / 100

    # 转成 [N, T=1, Din=8] 的三维张量，并放到设备上
    NUM_PDQD = 8
    NUM_PGQG = 4
    Train_In_New = Input_Full.reshape(NUM_Test, 1, NUM_PDQD)
    Train_In_New = torch.from_numpy(Train_In_New)
    Train_In_New = Train_In_New.to(torch.float32).to(device)

    # ========== 构建/加载 OptNet 模型 ==========
    num_output = 4       # 期望输出维度（例如 PG/QG 共 4 个量）
    num_input = 8        # 输入维度
    num_latent = 40      # QP 潜在维度（决策变量维度）
    num_ineq = 30        # 不等式约束数量
    num_eq = 30          # 等式约束数量
    Qpenalty = 0.1       # 预留的正则/惩罚项（由模型内部自行处理或外部 loss 使用）

    net = OptNetLatent(num_input, num_output, num_latent, Qpenalty, num_eq, num_ineq, 1e-5).to(device)
    net.load_state_dict(torch.load('Model_Total_BUS4.pth'))  # 载入已训练权重

    net.eval()                       # 推理模式（不更新 BN/Dropout）
    batch_size = 64                  # 注意：train_fn/test_fn 同样依赖此全局变量
    per = torch.randperm(Train_In_New.size(dim = 0))   # 打乱 200*200 个“网格点”
    NUM = int((per.size(dim = 0)/batch_size))          # 批次数

    # 预分配输出张量（第一行占位，后续拼接后再去掉）
    Out_Pre_Test1 = torch.zeros(1, num_output).to(device)

    # ========== 分批推理，得到所有网格点的预测 ==========
    for idx in range(1, NUM + 1):
        torch.cuda.empty_cache()  # 主动清空部分缓存（通常不必频繁调用，此处保持原逻辑）
        Label_SUM = per[(idx - 1) * batch_size:idx * batch_size, ]

        in_my = Train_In_New[Label_SUM, :, :]   # [B, 1, 8]

        out_pre = net(in_my)                    # [B, 4]
        Out_Pre_Test1 = torch.concatenate((Out_Pre_Test1, out_pre), 0)  # 纵向拼接
        torch.cuda.empty_cache()

    Out_Pre_Test1 = Out_Pre_Test1[1:, :]        # 去掉第一行占位
    Out_Pre_Test_Heavy = Out_Pre_Test1.reshape(NUM_Test, NUM_PGQG)  # [200*200, 4]

    # 保存预测结果（.pt 文件）
    torch.save(Out_Pre_Test_Heavy, 'Pre_Test_Total_BUS4.pt')
