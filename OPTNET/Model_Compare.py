import os
import copy
import random
import numpy as np
import torch

from Data_Load import Data_Load                   # -> (Input, Output_VMVA, Output_PGQG)
from Norm_Data import Norm_Data                   # 基于训练集 min-max 的归一化 & 划分
from Norm_Data2 import Norm_Data2                 # 提取归一化所需的输入/输出 min/max（用于反归一化）
from Pytorch_Data import Pytorch_Data             # 将 numpy [N,D] -> torch [N,1,D]

# QP 层（已训练/剪枝后的三个变体）
from Model_OPT import OptNetLatent, OptNetLatent_Q, OptNetLatent_Q_Train

import time


# ========== 工具函数 ==========
def seed_torch(seed: int = 114514) -> None:
    """固定随机数，尽量保证可复现（仍可能受算子/硬件影响）"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prune_small_(tensor: torch.Tensor, thresh: float) -> None:
    """原地将 |x| < thresh 的条目置零（保持 dtype/device 不变）"""
    with torch.no_grad():
        mask = tensor.abs() >= thresh
        tensor.mul_(mask)


def make_spd(Q: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """将 Q 投影为对称正定：对称化 + (LLᵀ + eps·I) 的安全策略"""
    Q = 0.5 * (Q + Q.transpose(-1, -2))
    # 数值稳定起见，再做一次最简单的抖动
    eye = torch.eye(Q.size(-1), device=Q.device, dtype=Q.dtype)
    return Q + eps * eye


# ========== 主流程 ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

if __name__ == "__main__":
    seed_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------
    # 1) 读取 L/H 两份数据并拼接：前 2400 训练 + 后 600 测试（每份数据）
    # -------------------------------
    Input_A1, Output_VMVA_A1, Output_PGQG_A1 = Data_Load("L", 1)
    Input_A2, Output_VMVA_A2, Output_PGQG_A2 = Data_Load("H", 1)

    NUM_Sample_Train = 2400
    NUM_Sample_Test = 600
    NUM_Train = 4800
    NUM_Test = 1200
    HALF = NUM_Test // 2  # 600 —— 约定：前600=light，后600=heavy

    # 拼接（两份数据各取 2400 训练 + 600 测试）
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

    # -------------------------------
    # 2) 归一化划分（基于训练集统计量）
    # -------------------------------
    Train_In, Test_In, Train_VMVA, Test_VMVA = Norm_Data(Input, Output_VMVA, NUM_Train, NUM_Test)
    Train_In, Test_In, Train_PGQG, Test_PGQG = Norm_Data(Input, Output_PGQG, NUM_Train, NUM_Test)

    # 组合输出（顺序保持：PGQG 在前，VMVA 在后）
    Train_Total = np.hstack((Train_PGQG, Train_VMVA))
    Test_Total = np.hstack((Test_PGQG, Test_VMVA))

    # 反归一化用的极值（注意取“总输出”）
    Output_Total = np.hstack((Output_PGQG, Output_VMVA))
    _, _, Out_Max, Out_Min = Norm_Data2(Input, Output_Total, NUM_Train, NUM_Test)

    # -------------------------------
    # 3) 约定切分：前600=light，后600=heavy
    # -------------------------------
    Test_In_Light,  Test_In_Heavy  = Test_In[:HALF, :],     Test_In[HALF:, :]
    Test_VMVA_Light, Test_VMVA_Heavy = Test_VMVA[:HALF, :],  Test_VMVA[HALF:, :]
    Test_PGQG_Light, Test_PGQG_Heavy = Test_PGQG[:HALF, :],  Test_PGQG[HALF:, :]

    Test_Total_Light = np.hstack((Test_PGQG_Light, Test_VMVA_Light))
    Test_Total_Heavy = np.hstack((Test_PGQG_Heavy, Test_VMVA_Heavy))

    # -------------------------------
    # 4) 转为 torch 张量，送指定 device
    # -------------------------------
    NUM_PDQD = 28 - 6        # 输入维度（与原工程保持一致）
    NUM_OUT  = 26 + 10       # 输出维度：VM(13)+VA(13)+PG(5)+QG(5)=36

    Train_In_t, Test_In_t, Train_Out_t, Test_Out_t = Pytorch_Data(
        Train_In, Test_In, Train_Total, Test_Total, NUM_Train, NUM_Test, NUM_PDQD, NUM_OUT
    )
    Test_In_Heavy_t, Test_In_Light_t, Test_Out_Heavy_t, Test_Out_Light_t = Pytorch_Data(
        Test_In_Heavy, Test_In_Light, Test_Total_Heavy, Test_Total_Light, NUM_Sample_Test, NUM_Sample_Test, NUM_PDQD, NUM_OUT
    )

    # 设备放置
    Train_In_t  = Train_In_t.to(device)
    Train_Out_t = Train_Out_t.to(device)
    Test_In_t   = Test_In_t.to(device)
    Test_Out_t  = Test_Out_t.to(device)

    Test_In_Heavy_t  = Test_In_Heavy_t.to(device)
    Test_Out_Heavy_t = Test_Out_Heavy_t.to(device)
    Test_In_Light_t  = Test_In_Light_t.to(device)
    Test_Out_Light_t = Test_Out_Light_t.to(device)

    # -------------------------------
    # 5) 构建/加载 3 个网络
    #    - net:            完整 OptNetLatent（推理参考）
    #    - net_train_1:    用于提取 G/h/M/L/A/b（从同一权重加载）
    #    - net_train:      稀疏/剪枝后的 QP 变体（不训练，仅推理）
    # -------------------------------
    num_input   = NUM_PDQD
    num_output  = NUM_OUT
    num_latent  = 60        # 原脚本设定
    num_ineq    = 20
    num_eq      = 20
    Qpenalty    = 0.1
    q_reg_eps   = 1e-4      # 让 Q 更稳的抖动

    # 完整模型（参考推理）
    net = OptNetLatent(num_input, num_output, num_latent, Qpenalty, num_eq, num_ineq, q_reg_eps).to(device)
    net.load_state_dict(torch.load('Res_OPTNET\\Model_Total.pth', map_location='cpu'))
    net.to(device).eval()

    # 训练版（用于提取参数）
    net_train_1 = OptNetLatent_Q_Train(num_input, num_output, num_latent, Qpenalty, num_eq, num_ineq, q_reg_eps).to(device)
    net_train_1.load_state_dict(torch.load('Res_OPTNET\\Model_Total.pth', map_location='cpu'))
    net_train_1.to(device).eval()

    # 剪枝后的轻量模型（不训练，仅推理）
    # 注意：该版本的不等式个数是 11（与原脚本一致）
    net_train = OptNetLatent_Q(num_input, num_output, num_latent, Qpenalty, num_eq, 11, q_reg_eps).to(device)
    net_train.eval()

    # 前向一次，拿到 h（按你原来的写法）
    with torch.no_grad():
        _, _, h_train = net_train_1(Train_In_t)

    # 把 G/h/M/L/A/b 等从 net_train_1 复制/裁剪到 net_train
    G_full = net_train_1.G.data.detach().cpu().numpy()          # (num_ineq, num_latent)
    h_full = h_train.detach().cpu().numpy()[:1, :]               # 取一行（与原脚本一致）

    # 删除的行索引（将 20 个不等式裁剪到 11 个）
    index = [0, 1, 2, 7, 8, 9, 11, 12, 16]
    G_corr  = np.delete(G_full, index, 0)                        # -> (11, num_latent)
    h_corr  = np.delete(h_full, index, 1)                        # -> (1, 11)

    # 复制结构参数
    net_train.G.data = torch.tensor(G_corr, dtype=torch.float32, device=device)
    net_train.M.data = copy.deepcopy(net_train_1.M.data.to(device))
    net_train.L.data = copy.deepcopy(net_train_1.L.data.to(device))
    net_train.A.data = copy.deepcopy(net_train_1.A.data.to(device))
    net_train.b.data = copy.deepcopy(net_train_1.b.data.to(device))

    net_train.fc_in.weight.data = copy.deepcopy(net_train_1.fc_in.weight.data.to(device))
    net_train.fc_in.bias.data   = copy.deepcopy(net_train_1.fc_in.bias.data.to(device))
    net_train.fc_out.weight.data = copy.deepcopy(net_train_1.fc_out.weight.data.to(device))
    net_train.fc_out.bias.data   = copy.deepcopy(net_train_1.fc_out.bias.data.to(device))

    h_corr_t = torch.tensor(h_corr, dtype=torch.float32, device=device)

    # -------------------------------
    # 6) 稀疏化（阈值剪枝）+ 构造 SPD 的 Q
    # -------------------------------
    THRESH = 1e-4
    prune_small_(net_train.L.data, THRESH)
    prune_small_(net_train.G.data, THRESH)
    prune_small_(net_train.A.data, THRESH)
    prune_small_(net_train.b.data, THRESH)

    with torch.no_grad():
        # Q = L Lᵀ + eps·I （并对称化，进一步提高数值稳定性）
        Q = net_train.L.data @ net_train.L.data.transpose(-1, -2)
        Q = make_spd(Q, eps=q_reg_eps)

    # -------------------------------
    # 7) 计时：原模型 heavy 推理 vs 剪枝模型 heavy 推理（逐样本）
    # -------------------------------
    with torch.no_grad():
        t1 = time.time()
        for i in range(HALF):   # heavy 一共 600
            _ = net(Test_In_Heavy_t[i, :, :])
        t2 = time.time()
        print(f"原模型 heavy 600 条推理用时: {t2 - t1:.3f}s")

        t1 = time.time()
        for i in range(HALF):
            _ = net_train(Test_In_Heavy_t[i, :, :], h_corr_t, Q, net_train.G.data, net_train.A.data, net_train.b.data)
        t2 = time.time()
        print(f"剪枝模型 heavy 600 条推理用时: {t2 - t1:.3f}s")

    # -------------------------------
    # 8) 生成 light 集的预测并保存
    # -------------------------------
    with torch.no_grad():
        # 你的 OptNetLatent_Q 支持批处理则可直接喂 batch，
        # 否则可以循环；这里保留与你原脚本一致的批接口
        NUM_Pre = net_train(Test_In_Light_t, h_corr_t, Q, net_train.G.data, net_train.A.data, net_train.b.data)

    torch.save(NUM_Pre, 'Res_OPTNET\\Pre_Test_Light_Total_Prun1.pt')
    print("保存完成：Res_OPTNET\\Pre_Test_Light_Total_Prun1.pt")
