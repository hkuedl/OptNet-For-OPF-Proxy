# -*- coding: utf-8 -*-
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn

from Data_Load import *          # 读取数据
from Norm_Data import Norm_Data                    # 归一化 + 划分 train/test
from Pytorch_Data import Pytorch_Data              # numpy -> torch [N,1,D]
from Model_OPT import NN_VM, NN_PG                 # 两个子网络结构
from pypower.api import case14                     # 取物理上下限（IEEE 14-bus）
import time

# ========== 工具函数 ==========
def seed_torch(seed: int = 114514) -> None:
    """固定随机种子，尽量保证可复现（仍可能受算子/硬件影响）。"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_fn(
    data_train: torch.Tensor,
    target_train: torch.Tensor,
    model: torch.nn.Module,
    crit: torch.nn.Module,
    opt: torch.optim.Optimizer,
    flag: int,
    down_t: torch.Tensor,
    up_t: torch.Tensor,
    pen1: float,
    pen2: float,
    batch_size: int,
) -> float:
    """
    单轮训练，支持对预测值施加“超/欠界”惩罚。
    参数：
      - flag: 1=PG, 2=QG, 3=VM, 4=VA（仅在 flag==1/2/3/4 时决定是否加界限惩罚）
      - down_t/up_t: 形状 [1,1,output_dim] 的上下限张量（已在 device 上）
      - pen1/pen2: 上/下越界惩罚系数
    """
    model.train()
    total = 0.0

    # 随机打乱索引（在 GPU 上生成，可避免拷贝）
    perm = torch.randperm(data_train.size(0), device=data_train.device)
    num_batches = perm.size(0) // batch_size

    for i in range(num_batches):
        idx = perm[i * batch_size:(i + 1) * batch_size]
        x = data_train[idx]
        y = target_train[idx]

        opt.zero_grad()
        y_pred = model(x)

        loss = crit(y_pred, y)
        # 仅当需要时加入界限惩罚（(z + |z|)/2 = ReLU(z)）
        if flag in (1, 2, 3, 4):
            loss += pen1 * torch.relu(y_pred - up_t).sum()   # 超上界
            loss += pen2 * torch.relu(down_t - y_pred).sum() # 低下界

        loss.backward()
        opt.step()

        total += float(loss.item())

    return total / max(num_batches, 1)


def test_fn(
    data_test: torch.Tensor,
    target_test: torch.Tensor,
    model: torch.nn.Module,
    crit: torch.nn.Module,
    batch_size: int,
) -> float:
    """验证集评估（无惩罚项）。"""
    model.eval()
    total = 0.0

    with torch.no_grad():
        perm = torch.randperm(data_test.size(0), device=data_test.device)
        num_batches = perm.size(0) // batch_size

        for i in range(num_batches):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            x = data_test[idx]
            y = target_test[idx]
            y_pred = model(x)
            loss = crit(y_pred, y)
            total += float(loss.item())

    return total / max(num_batches, 1)


# ========== 主流程 ==========
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

if __name__ == "__main__":
    seed_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # 数据载入与拼接：两份数据各取 2400 训练 + 600 测试
    # -------------------------------
    Input_A1, Output_VMVA_A1, Output_PGQG_A1 = Data_Load("L", 1)
    Input_A2, Output_VMVA_A2, Output_PGQG_A2 = Data_Load("H", 1)

    NUM_TRAIN_EACH, NUM_TEST_EACH = 2400, 600
    NUM_TRAIN, NUM_TEST = 4800, 1200

    Input = np.vstack((
        Input_A1[:NUM_TRAIN_EACH, :],
        Input_A2[:NUM_TRAIN_EACH, :],
        Input_A1[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
        Input_A2[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
    ))
    Output_VMVA = np.vstack((
        Output_VMVA_A1[:NUM_TRAIN_EACH, :],
        Output_VMVA_A2[:NUM_TRAIN_EACH, :],
        Output_VMVA_A1[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
        Output_VMVA_A2[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
    ))
    Output_PGQG = np.vstack((
        Output_PGQG_A1[:NUM_TRAIN_EACH, :],
        Output_PGQG_A2[:NUM_TRAIN_EACH, :],
        Output_PGQG_A1[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
        Output_PGQG_A2[NUM_TRAIN_EACH:NUM_TRAIN_EACH + NUM_TEST_EACH, :],
    ))

    # -------------------------------
    # 归一化 + 划分
    # -------------------------------
    Train_In, Test_In, Train_VMVA, Test_VMVA = Norm_Data(Input, Output_VMVA, NUM_TRAIN, NUM_TEST)
    _,        _,        Train_PGQG, Test_PGQG = Norm_Data(Input, Output_PGQG, NUM_TRAIN, NUM_TEST)

    # -------------------------------
    # 任务选择（输出维度 & 训练目标）
    # Flag: 1=PG, 2=QG, 3=VM, 4=VA
    # -------------------------------
    Flag = 4
    if Flag in (1, 2):
        num_half = Output_PGQG.shape[1] // 2
        # PG(0:5) / QG(5:10)
        Train_Out = Train_PGQG[:, :num_half] if Flag == 1 else Train_PGQG[:, num_half:]
        Test_Out  = Test_PGQG[:, :num_half]  if Flag == 1 else Test_PGQG[:,  num_half:]
        out_dim = 5
    else:
        num_half = Output_VMVA.shape[1] // 2
        # VM(0:13) / VA(13:26)
        Train_Out = Train_VMVA[:, :num_half] if Flag == 3 else Train_VMVA[:, num_half:]
        Test_Out  = Test_VMVA[:, :num_half]  if Flag == 3 else Test_VMVA[:,  num_half:]
        out_dim = 13

    # -------------------------------
    # 切分 Heavy/Light（与原逻辑一致：前600=Light；后600=Heavy）
    # -------------------------------
    HALF = 600
    Test_In_Heavy,  Test_In_Light  = copy.deepcopy(Test_In[HALF:, :]),  copy.deepcopy(Test_In[:HALF, :])
    Test_Out_Heavy, Test_Out_Light = copy.deepcopy(Test_Out[HALF:, :]), copy.deepcopy(Test_Out[:HALF, :])

    # -------------------------------
    # 打乱训练/测试（保持原风格）
    # -------------------------------
    shuffle_train = np.random.permutation(np.arange(len(Train_In)))
    shuffle_test  = np.random.permutation(np.arange(len(Test_In)))

    Train_In  = Train_In[shuffle_train, :]
    Train_Out = Train_Out[shuffle_train, :]
    Test_In   = Test_In[shuffle_test, :]
    Test_Out  = Test_Out[shuffle_test, :]

    # -------------------------------
    # 组装为 torch 张量 [N,1,D]
    # -------------------------------
    NUM_PDQD = 28 - 6
    Train_X, Test_X, Train_Y, Test_Y = Pytorch_Data(
        Train_In, Test_In, Train_Out, Test_Out, NUM_TRAIN, NUM_TEST, NUM_PDQD, out_dim
    )
    Test_X_H, Test_X_L, Test_Y_H, Test_Y_L = Pytorch_Data(
        Test_In_Heavy, Test_In_Light, Test_Out_Heavy, Test_Out_Light, NUM_TEST_EACH, NUM_TEST_EACH, NUM_PDQD, out_dim
    )

    # 送往 device
    Train_X = Train_X.to(device);  Train_Y = Train_Y.to(device)
    Test_X  = Test_X.to(device);   Test_Y  = Test_Y.to(device)
    Test_X_H = Test_X_H.to(device); Test_Y_H = Test_Y_H.to(device)
    Test_X_L = Test_X_L.to(device); Test_Y_L = Test_Y_L.to(device)

    # -------------------------------
    # 网络与训练配置
    # -------------------------------
    net = (NN_PG() if Flag in (1, 2) else NN_VM()).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    batch_size = 16
    epochs = 300

    # -------------------------------
    # 物理上下限（用于惩罚）
    # -------------------------------
    data_in = case14()
    # 调整 PMAX（按你原逻辑）
    data_in['gen'][0, 8] = 100
    data_in['gen'][1, 8] = 100
    data_in['gen'][2, 8] = 80
    data_in['gen'][3, 8] = 60
    data_in['gen'][4, 8] = 60

    vol_up   = data_in['bus'][1:, 11]  # 去掉平衡母线 => 13
    vol_down = data_in['bus'][1:, 12]
    p_up,  p_down  = data_in['gen'][:, 8], data_in['gen'][:, 9]  # 5
    q_up,  q_down  = data_in['gen'][:, 3], data_in['gen'][:, 4]  # 5

    # 选择对应上下限并做成 [1,1,out_dim]，一次性搬到 device
    if Flag == 1:
        up_arr, down_arr = p_up, p_down
        pen1, pen2 = 1e-4, 1e-4
    elif Flag == 2:
        up_arr, down_arr = q_up, q_down
        pen1, pen2 = 1e-4, 1e-4
    elif Flag == 3:
        up_arr, down_arr = vol_up, vol_down
        pen1, pen2 = 1e-3, 1e-3
    else:  # Flag == 4（原代码同样使用电压限）
        up_arr, down_arr = vol_up, vol_down
        pen1, pen2 = 1e-3, 1e-3

    up_t   = torch.tensor(up_arr,   dtype=torch.float32, device=device).view(1, 1, out_dim)
    down_t = torch.tensor(down_arr, dtype=torch.float32, device=device).view(1, 1, out_dim)

    # -------------------------------
    # 训练
    # -------------------------------
    tr_losses, te_losses = [], []
    for epoch in range(epochs):
        tr = train_fn(Train_X, Train_Y, net, loss_fn, optimizer, Flag, down_t, up_t, pen1, pen2, batch_size)
        te = test_fn(Test_X, Test_Y, net, loss_fn, batch_size)
        tr_losses.append(tr); te_losses.append(te)
        print(f"Epoch {epoch:03d} | Train: {tr:.8f} | Test: {te:.8f}")

    # -------------------------------
    # 推理（批量）
    # -------------------------------
    net.eval()
    with torch.no_grad():
        out_h = net(Test_X_H).reshape(NUM_TEST_EACH, out_dim)  # Heavy
        out_l = net(Test_X_L).reshape(NUM_TEST_EACH, out_dim)  # Light

    # ========== 单样本逐次推理的耗时（不保留输出，仅测时）==========
    NUM_Sample = Test_In_Heavy.size(0)
    t1 = time.time()
    with torch.no_grad():
        for i in range(NUM_Sample):
            _ = net(Test_In_Heavy[i, :, :])  # 单样本前向
    t2 = time.time()
    print("Single-sample forward loop time:", t2 - t1)
    # -------------------------------
    # 保存结果
    # -------------------------------
    if Flag == 1:
        torch.save(out_h, 'Pre_Heavy_PG_NN_ALM.pt')
        torch.save(out_l, 'Pre_Light_PG_NN_ALM.pt')
        torch.save(net.state_dict(), 'Model_PG_NN_ALM.pth')
        torch.save(np.array(tr_losses), 'Loss_PG_NN_ALM.pt')
        torch.save(np.array(te_losses), 'Test_Loss_PG_NN_ALM.pt')
    elif Flag == 2:
        torch.save(out_h, 'Pre_Heavy_QG_NN_ALM.pt')
        torch.save(out_l, 'Pre_Light_QG_NN_ALM.pt')
        torch.save(net.state_dict(), 'Model_QG_NN_ALM.pth')
        torch.save(np.array(tr_losses), 'Loss_QG_NN_ALM.pt')
        torch.save(np.array(te_losses), 'Test_Loss_QG_NN_ALM.pt')
    elif Flag == 3:
        torch.save(out_h, 'Pre_Heavy_VM_NN_ALM.pt')
        torch.save(out_l, 'Pre_Light_VM_NN_ALM.pt')
        torch.save(net.state_dict(), 'Model_VM_NN_ALM.pth')
        torch.save(np.array(tr_losses), 'Loss_VM_NN_ALM.pt')
        torch.save(np.array(te_losses), 'Test_Loss_VM_NN_ALM.pt')
    else:
        torch.save(out_h, 'Pre_Heavy_VA_NN_ALM.pt')
        torch.save(out_l, 'Pre_Light_VA_NN_ALM.pt')
        torch.save(net.state_dict(), 'Model_VA_NN_ALM.pth')
        torch.save(np.array(tr_losses), 'Loss_VA_NN_ALM.pt')
        torch.save(np.array(te_losses), 'Test_Loss_VA_NN_ALM.pt')
