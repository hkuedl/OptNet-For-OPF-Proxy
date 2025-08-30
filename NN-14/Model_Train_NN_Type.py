import torch
import torch.nn as nn
import numpy as np
import copy
from Data_Load import *          # 提供 Data_Load(S, flag) -> (Input, Output_VMVA, Output_PGQG)
import os
import random
from Norm_Data import *               # Norm_Data: 按训练/测试划分并做归一化（返回 Train_X/Test_X 与对应 y）
from Pytorch_Data import *            # Pytorch_Data: 将 numpy 转为 [N,1,Dim] 的 torch.Tensor
from Model_OPT import NN_V, NN_P      # 两个 MLP：NN_V 输出 26 维，NN_P 输出 10 维
import time

# ========== 固定随机种子，保证实验可复现 ==========
def seed_torch(seed=114514):
    """设定 Python、NumPy、PyTorch 的随机种子，并控制 cuDNN 的确定性。"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止 Python 哈希随机化
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)          # 多 GPU 场景
    torch.backends.cudnn.benchmark = False    # 关闭自动算法搜索（避免非确定性）
    torch.backends.cudnn.deterministic = True # 使用确定性算法

# ========== 单个 epoch 的训练 ==========
def train_fn(dataTrain, data_real_Train, model, crit, opt):
    """
    dataTrain       : [N, 1, Din] 的 Tensor
    data_real_Train : [N, 1, Dout] 的 Tensor
    model           : NN_V 或 NN_P
    crit            : 损失函数（MSE）
    opt             : 优化器（Adam/SGD）
    说明：依赖外部全局变量 batch_size
    """
    model.train()
    loss_total_final = 0.0

    per = torch.randperm(dataTrain.size(0))                # 打乱样本
    NUM = int(per.size(0) / batch_size)                    # 满批次数（尾部不足一批的样本丢弃）

    for idx in range(1, NUM + 1):
        Label_SUM = per[(idx - 1) * batch_size:idx * batch_size]

        in_my = dataTrain[Label_SUM, :, :]                 # [B, 1, Din]
        out_my = data_real_Train[Label_SUM, :, :]          # [B, 1, Dout]

        opt.zero_grad()
        out_pre = model(in_my)                             # [B, Dout]

        size1 = out_my.size(2)                             # Dout
        out_my = out_my.reshape(batch_size, size1)         # [B, Dout]
        loss = crit(out_pre, out_my)

        loss_total_final += loss.item()
        loss.backward()
        opt.step()

        del out_pre                                        # 及时释放显存

    return loss_total_final / NUM

# ========== 测试/验证（评估 loss） ==========
def test_fn(dataTest, data_real_Test, model, crit, opt):
    """与 train_fn 类似，但只前向评估 loss，不更新参数。"""
    model.eval()
    loss_total_final = 0.0

    per = torch.randperm(dataTest.size(0))
    NUM = int(per.size(0) / batch_size)

    for idx in range(1, NUM + 1):
        Label_SUM = per[(idx - 1) * batch_size:idx * batch_size]

        in_my = dataTest[Label_SUM, :, :]                  # [B, 1, Din]
        out_my = data_real_Test[Label_SUM, :, :]           # [B, 1, Dout]

        out_pre = model(in_my)                             # [B, Dout]
        size1 = out_my.size(2)
        out_my = out_my.reshape(batch_size, size1)         # [B, Dout]

        loss1 = crit(out_pre, out_my)
        del out_pre
        loss_total_final += loss1.item()

    return loss_total_final / NUM

# ========== 环境设置（GPU/内存策略） ==========
os.environ["CUDA_VISIBLE_DEVICES"] = '2'                   # 指定第 3 张 GPU（从 0 开始计数）
torch.backends.cudnn.benchmark = True                      # 训练阶段可打开以提速
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 调整碎片化策略

if __name__ == "__main__":
    seed_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 记录训练/测试损失（按 epoch 追加）
    Mag_loss_Area1 = np.zeros((1, 1))
    Te_loss_Area1 = np.zeros((1, 1))

    # ====== 生成两份合成数据（不同随机种子）======
    # Data_Generated(seed, flag) -> (Input, Output_VMVA, Output_PGQG)
    Input_A1, Output_VMVA_A1, Output_PGQG_A1 = Data_Load("L", 1)
    Input_A2, Output_VMVA_A2, Output_PGQG_A2 = Data_Load("H", 1)

    # 训练/测试划分
    NUM_Sample_Train = 2400
    NUM_Sample_Test = 600

    # 组装训练集(2*2400) + 测试集(2*600)
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

    # 训练/测试样本量（总训练 4800、测试 1200）
    NUM_Train = 4800
    NUM_Test = 1200

    # 任务开关：Flag=1 → 训练 NN_P（输出 PGQG=10 维）；Flag!=1 → 训练 NN_V（输出 VMVA=26 维）
    Flag = 2

    # === 两次归一化拆分 ===
    # 1) 以 Output_VMVA 为标签
    Train_Input_A1, Test_Input_A1, Train_VMVA_A1, Test_VMVA_A1 = Norm_Data(
        Input, Output_VMVA, NUM_Train, NUM_Test
    )
    # 2) 以 Output_PGQG 为标签
    Train_Input_A1, Test_Input_A1, Train_PGQG_A1, Test_PGQG_A1 = Norm_Data(
        Input, Output_PGQG, NUM_Train, NUM_Test
    )


    # 将测试集再分成 heavy / light 各 600
    Test_VMVA_Heavy = copy.deepcopy(Test_VMVA_A1[600:, :])
    Test_VMVA_Light = copy.deepcopy(Test_VMVA_A1[:600, :])
    Test_Input_Heavy = copy.deepcopy(Test_Input_A1[600:, :])
    Test_Input_Light = copy.deepcopy(Test_Input_A1[:600, :])

    Test_PGQG_Heavy = copy.deepcopy(Test_PGQG_A1[600:, :])
    Test_PGQG_Light = copy.deepcopy(Test_PGQG_A1[:600, :])

    # 针对当前任务，挑选 heavy/light 的标签
    if Flag == 1:
        Test_Total_Heavy = copy.deepcopy(Test_PGQG_Heavy)
        Test_Total_Light = copy.deepcopy(Test_PGQG_Light)
    else:
        Test_Total_Heavy = copy.deepcopy(Test_VMVA_Heavy)
        Test_Total_Light = copy.deepcopy(Test_VMVA_Light)

    # 打乱训练/测试样本顺序
    shuffle_Train = np.random.permutation(np.arange(len(Train_Input_A1)))
    shuffle_Test = np.random.permutation(np.arange(len(Test_Input_A1)))

    Train_Input_A1 = Train_Input_A1[shuffle_Train, :]
    Train_VMVA_A1 = Train_VMVA_A1[shuffle_Train, :]
    Train_PGQG_A1 = Train_PGQG_A1[shuffle_Train, :]

    Test_Input_A1 = Test_Input_A1[shuffle_Test, :]
    Test_VMVA_A1 = Test_VMVA_A1[shuffle_Test, :]
    Test_PGQG_A1 = Test_PGQG_A1[shuffle_Test, :]

    # 根据 Flag 选定最终用于训练/测试的标签
    if Flag == 1:
        Train_Total_A1 = copy.deepcopy(Train_PGQG_A1)
        Test_Total_A1 = copy.deepcopy(Test_PGQG_A1)
    else:
        Train_Total_A1 = copy.deepcopy(Train_VMVA_A1)
        Test_Total_A1 = copy.deepcopy(Test_VMVA_A1)

    # ========= 数据转 Tensor（[N,1,Dim]）=========
    batch_size = 16
    NUM_PDQD = 28 - 6                        # 输入维度（示例：22）
    NUM_PGQG = 10 if Flag == 1 else 26       # 输出维度（NN_P:10 / NN_V:26）

    Train_In_A1, Test_In_A1, Train_Out_A1, Test_Out_A1 = Pytorch_Data(
        Train_Input_A1, Test_Input_A1, Train_Total_A1, Test_Total_A1,
        NUM_Train, NUM_Test, NUM_PDQD, NUM_PGQG
    )

    # heavy/light 两个子测试集
    Test_In_Heavy, Test_In_Light, Test_Out_Heavy, Test_Out_Light = Pytorch_Data(
        Test_Input_Heavy, Test_Input_Light, Test_Total_Heavy, Test_Total_Light,
        NUM_Sample_Test, NUM_Sample_Test, NUM_PDQD, NUM_PGQG
    )

    # 放到设备上
    Test_In_Heavy, Test_Out_Heavy = Test_In_Heavy.to(device), Test_Out_Heavy.to(device)
    Test_In_Light, Test_Out_Light = Test_In_Light.to(device), Test_Out_Light.to(device)
    Train_In_A1, Test_In_A1 = Train_In_A1.to(device), Test_In_A1.to(device)
    Train_Out_A1, Test_Out_A1 = Train_Out_A1.to(device), Test_Out_A1.to(device)

    # ========= 构建模型/损失/优化器 =========
    num_output = 10 if Flag == 1 else 26
    net = (NN_P() if Flag == 1 else NN_V()).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    epochs = 300
    for epoch in range(epochs):
        torch.cuda.empty_cache()

        Loss_Mag_Area1 = train_fn(Train_In_A1, Train_Out_A1, net, loss_fn, optimizer)
        Loss_Area1 = test_fn(Test_In_A1, Test_Out_A1, net, loss_fn, optimizer)
        print(f"Epoch [{epoch}] Training Loss: {Loss_Mag_Area1:.8f}, Testing Loss: {Loss_Area1:.8f}")

        # 记录并累加到 numpy（首行是占位 0）
        Mag_loss_Area1 = np.vstack((Mag_loss_Area1, Loss_Mag_Area1))
        Te_loss_Area1 = np.vstack((Te_loss_Area1,  Loss_Area1))

        torch.cuda.empty_cache()

    net.eval()

    # ========== 单样本逐次推理的耗时（不保留输出，仅测时）==========
    NUM_Sample = Test_In_Heavy.size(0)
    t1 = time.time()
    with torch.no_grad():
        for i in range(NUM_Sample):
            _ = net(Test_In_Heavy[i, :, :])  # 单样本前向
    t2 = time.time()
    print("Single-sample forward loop time:", t2 - t1)

    # ========== 批量推理并保存 ==========
    with torch.no_grad():
        Out_Pre_Test_Heavy = net(Test_In_Heavy).reshape(NUM_Sample_Test, NUM_PGQG)
        Out_Pre_Test_Light = net(Test_In_Light).reshape(NUM_Sample_Test, NUM_PGQG)

    if Flag == 1:
        torch.save(Out_Pre_Test_Heavy, 'Pre_Heavy_PG_NN.pt')
        torch.save(Out_Pre_Test_Light, 'Pre_Light_PG_NN.pt')
        torch.save(net.state_dict(), 'Model_PG_NN.pth')
        torch.save(Mag_loss_Area1, 'Loss_PG_NN.pt')
        torch.save(Te_loss_Area1, 'Test_Loss_PG_NN.pt')
    else:
        torch.save(Out_Pre_Test_Heavy, 'Pre_Heavy_VM_NN.pt')
        torch.save(Out_Pre_Test_Light, 'Pre_Light_VM_NN.pt')
        torch.save(net.state_dict(), 'Model_VM_NN.pth')
        torch.save(Mag_loss_Area1, 'Loss_VM_NN.pt')
        torch.save(Te_loss_Area1, 'Test_Loss_VM_NN.pt')
