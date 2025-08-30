import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn

from Data_Load import Data_Load       # -> (Input, Output_VMVA, Output_PGQG) 当 flag=1
from Norm_Data import Norm_Data                 # 归一化+划分（基于训练集统计量）
from Pytorch_Data import Pytorch_Data           # 将 numpy reshape 为 [N,1,D] 并转 torch
from Norm_Data2 import Norm_Data2               # 仅提取 min/max（用于反归一化范围）
from Model_OPT import OptNetLatent              # OptNet 主干模型
import time

# 固定随机种子，尽量保证可复现（仍可能受算子/硬件影响）
def seed_torch(seed: int = 114544) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_fn(dataTrain, data_real_Train, model, crit, opt):
    """单 epoch 训练：随机打乱后按 batch 迭代，计算 MSE 并反传更新"""
    model.train()
    loss_total_final = 0.0

    per = torch.randperm(dataTrain.size(0))              # 打乱样本顺序
    NUM = int(per.size(0) / batch_size)                  # 批次数

    for idx in range(1, NUM + 1):
        label_idx = per[(idx - 1) * batch_size: idx * batch_size]

        in_my = dataTrain[label_idx, :, :]               # [B,1,Din]
        out_my = data_real_Train[label_idx, :, :]        # [B,1,Dout]

        out_pre = model(in_my)                           # [B,Dout]

        # 标签维度拍平到 [B, Dout]
        size1 = out_my.size(2)
        out_my = out_my.reshape((batch_size, size1))

        loss = crit(out_pre, out_my)
        loss_total_final += loss.item()

        # 反向与更新
        opt.zero_grad()
        loss.backward()
        opt.step()

        del out_pre

    return loss_total_final / NUM


def test_fn(dataTest, data_real_Test, model, crit, opt):
    """验证集评估：仅前向与累计损失"""
    model.eval()
    loss_total_final = 0.0

    per = torch.randperm(dataTest.size(0))
    NUM = int(per.size(0) / batch_size)

    for idx in range(1, NUM + 1):
        label_idx = per[(idx - 1) * batch_size: idx * batch_size]

        in_my = dataTest[label_idx, :, :]
        out_my = data_real_Test[label_idx, :, :]

        out_pre = model(in_my)

        size1 = out_my.size(2)
        out_my = out_my.reshape((batch_size, size1))

        loss1 = crit(out_pre, out_my)
        loss_total_final += loss1.item()

        del out_pre

    return loss_total_final / NUM


# ========= 主程序 =========
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

if __name__ == "__main__":
    seed_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 任务规模开关：
    # FLAG=1 -> 14 节点（每个数据集 2400 训练 + 600 测试）
    FLAG = 1

    # 生成两份数据（不同随机源），随后拼接成总数据
    Input_A1, Output_VMVA_A1, Output_PGQG_A1 = Data_Load("L", FLAG)
    Input_A2, Output_VMVA_A2, Output_PGQG_A2 = Data_Load("H", FLAG)

    NUM_Sample_Train = 2400
    NUM_Sample_Test  = 600


    # 拼接：两份数据各取训练段与测试段
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

    # 总训练/测试量（两份数据拼接）
    NUM_Train = 4800
    NUM_Test  = 1200

    # 以 VMVA 与 PGQG 作为标签分别做一次归一化划分（均使用同一份输入统计）
    Train_Input, Test_Input, Train_VMVA, Test_VMVA = Norm_Data(Input, Output_VMVA, NUM_Train, NUM_Test)
    Train_Input, Test_Input, Train_PGQG, Test_PGQG = Norm_Data(Input, Output_PGQG, NUM_Train, NUM_Test)

    # 用于反归一化的总体极值（这里合并标签再取 min/max，和你原逻辑一致）
    Output_Total = np.hstack((Output_PGQG, Output_VMVA))
    In_Max, In_Min, Out_Max, Out_Min = Norm_Data2(Input, Output_Total, NUM_Train, NUM_Test)

    # —— 关键切分：严格按“前600=light，后600=heavy”（若 FLAG!=1，则为“前 NUM_Sample_Test=light，后者=heavy”）——
    HALF = NUM_Sample_Test  # FLAG=1 时为 600
    Test_VMVA_Light = copy.deepcopy(Test_VMVA[:HALF, :])
    Test_VMVA_Heavy = copy.deepcopy(Test_VMVA[HALF:, :])
    Test_Input_Light = copy.deepcopy(Test_Input[:HALF, :])
    Test_Input_Heavy = copy.deepcopy(Test_Input[HALF:, :])
    Test_PGQG_Light = copy.deepcopy(Test_PGQG[:HALF, :])
    Test_PGQG_Heavy = copy.deepcopy(Test_PGQG[HALF:, :])

    # 组合输出（PGQG + VMVA）
    Test_Total_Light = np.hstack((Test_PGQG_Light, Test_VMVA_Light))
    Test_Total_Heavy = np.hstack((Test_PGQG_Heavy, Test_VMVA_Heavy))

    # 训练/测试随机打散（与原逻辑一致）
    shuffle_train = np.random.permutation(np.arange(len(Train_Input)))
    shuffle_test  = np.random.permutation(np.arange(len(Test_Input)))

    Train_Input = Train_Input[shuffle_train, :]
    Train_VMVA  = Train_VMVA[shuffle_train, :]
    Train_PGQG  = Train_PGQG[shuffle_train, :]

    Test_Input = Test_Input[shuffle_test, :]
    Test_VMVA  = Test_VMVA[shuffle_test, :]
    Test_PGQG  = Test_PGQG[shuffle_test, :]

    Train_Total = np.hstack((Train_PGQG, Train_VMVA))
    Test_Total  = np.hstack((Test_PGQG,  Test_VMVA))

    # 输入/输出维度
    batch_size = 16
    NUM_PDQD   = 28 - 6          # 有 6 个参考/无用量被删除
    NUM_PGQG   = 26 + 10         # VM(13)+VA(13)+PG(5)+QG(5) = 36

    # 构造张量
    Train_In, Test_In, Train_Out, Test_Out = Pytorch_Data(
        Train_Input, Test_Input, Train_Total, Test_Total, NUM_Train, NUM_Test, NUM_PDQD, NUM_PGQG
    )
    Test_In_Light, Test_In_Heavy, Test_Out_Light, Test_Out_Heavy = Pytorch_Data(
        Test_Input_Light, Test_Input_Heavy, Test_Total_Light, Test_Total_Heavy,
        NUM_Sample_Test, NUM_Sample_Test, NUM_PDQD, NUM_PGQG
    )

    # 送显卡
    Train_In = Train_In.to(device);   Train_Out = Train_Out.to(device)
    Test_In  = Test_In.to(device);    Test_Out  = Test_Out.to(device)
    Test_In_Light = Test_In_Light.to(device); Test_Out_Light = Test_Out_Light.to(device)
    Test_In_Heavy = Test_In_Heavy.to(device); Test_Out_Heavy = Test_Out_Heavy.to(device)

    # OptNet 架构与优化器设置（参数保持与原脚本一致）
    num_output = 26 + 10
    num_input  = 28 - 6
    num_latent = 80          # 隐变量维度
    num_ineq   = 20
    num_eq     = 20
    Qpenalty   = 0.1

    net = OptNetLatent(num_input, num_output, num_latent, Qpenalty, num_eq, num_ineq, 1e-5).to(device)

    loss_fn  = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-3)

    # 训练与验证
    epochs = 300
    train_losses = np.zeros((1, 1))
    test_losses  = np.zeros((1, 1))

    for epoch in range(epochs):
        torch.cuda.empty_cache()

        tr_loss = train_fn(Train_In, Train_Out, net, loss_fn, optimizer)
        te_loss = test_fn(Test_In,  Test_Out,  net, loss_fn, optimizer)

        print(f"Epoch [{epoch}]  Training Loss: {tr_loss:.8f}, Testing Loss: {te_loss:.8f}")

        train_losses = np.vstack((train_losses, tr_loss))
        test_losses  = np.vstack((test_losses,  te_loss))

    # 训练完成后保存最终模型与损失曲线
    torch.save(net.state_dict(), 'Model_Total.pth')
    torch.save(train_losses, 'Loss_Total.pt')
    torch.save(test_losses,  'Test_Loss_Total.pt')

    # ========== 单样本逐次推理的耗时（不保留输出，仅测时）==========
    NUM_Sample = Test_In_Heavy.size(0)
    t1 = time.time()
    with torch.no_grad():
        for i in range(NUM_Sample):
            _ = net(Test_In_Heavy[i, :, :])  # 单样本前向
    t2 = time.time()
    print("Single-sample forward loop time:", t2 - t1)

    # 推理并保存 Light / Heavy 的预测
    net.eval()
    Out_Pre_Light = net(Test_In_Light).reshape(NUM_Sample_Test, NUM_PGQG)
    Out_Pre_Heavy = net(Test_In_Heavy).reshape(NUM_Sample_Test, NUM_PGQG)

    torch.save(Out_Pre_Heavy, 'Pre_Test_Heavy_Total.pt')
    torch.save(Out_Pre_Light, 'Pre_Test_Light_Total.pt')
