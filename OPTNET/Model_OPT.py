import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from qpth.qp import QPFunction  # 可微 QP 求解器（quadratic program）


class NN_V(nn.Module):

    # 实际实现为三层 MLP（全连接 + ReLU），输入维度 28、输出维度 26
    def __init__(self):
        super(NN_V, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(28, 2 * 14 + 3 * 14 + 28),  # 28 -> 98
            nn.ReLU(True),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(2 * 14 + 3 * 14 + 28, 2 * 14 + 28),  # 98 -> 56
            nn.ReLU(True),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(2 * 14 + 28, 26)  # 56 -> 26
        )

    def forward(self, x):
        # x: [batch, 28]
        x = self.layer1(x)  # [batch, 98]
        x = self.layer2(x)  # [batch, 56]
        x = self.layer3(x)  # [batch, 26]
        return x


class NN_P(nn.Module):

    # 三层 MLP：输入 28 -> 输出 10
    def __init__(self):
        super(NN_P, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(28, 2 * 14 + 3 * 14 + 28),  # 28 -> 98
            nn.ReLU(True),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(2 * 14 + 3 * 14 + 28, 2 * 14 + 28),  # 98 -> 56
            nn.ReLU(True),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(2 * 14 + 28, 10)  # 56 -> 10
        )

    def forward(self, x):
        # x: [batch, 28]
        x = self.layer1(x)  # [batch, 98]
        x = self.layer2(x)  # [batch, 56]
        x = self.layer3(x)  # [batch, 10]
        return x


class NN_VM(nn.Module):

    # 三层 MLP：输入 28 -> 输出 13
    def __init__(self):
        super(NN_VM, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(28, 2 * 14 + 3 * 14 + 28),  # 28 -> 98
            nn.ReLU(True),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(2 * 14 + 3 * 14 + 28, 2 * 14 + 28),  # 98 -> 56
            nn.ReLU(True),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(2 * 14 + 28, 13)  # 56 -> 13
        )

    def forward(self, x):
        # x: [batch, 28]
        x = self.layer1(x)  # [batch, 98]
        x = self.layer2(x)  # [batch, 56]
        x = self.layer3(x)  # [batch, 13]
        return x


class NN_PG(nn.Module):

    # 三层 MLP：输入 28 -> 输出 5（例如 5 台机组的有功出力等）
    def __init__(self):
        super(NN_PG, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(28, 2 * 14 + 3 * 14 + 28),  # 28 -> 98
            nn.ReLU(True),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(2 * 14 + 3 * 14 + 28, 2 * 14 + 28),  # 98 -> 56
            nn.ReLU(True),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(2 * 14 + 28, 5)  # 56 -> 5
        )

    def forward(self, x):
        # x: [batch, 28]
        x = self.layer1(x)  # [batch, 98]
        x = self.layer2(x)  # [batch, 56]
        x = self.layer3(x)  # [batch, 5]
        return x


class OptNetLatent(nn.Module):
    # 可微 QP 层（OptNet 风格）：
    #   解以下问题的最优解 x*，并将其再映射为网络输出：
    #       minimize   0.5 * x^T Q x + p^T x
    #       subject to G x <= h,   A x = b
    # - 其中 Q = L L^T + eps*I（由可学习下三角 L 构造，确保半正定）
    # - p 由前端线性层 fc_in(x_in) 得到（随 batch 变化）
    # - G, h, A, b 在此作为可学习参数（与输入无关）
    def __init__(self, Num_Input, Num_Output, Num_Latent, Qpenalty, neq, nineq, eps=1e-4):
        super().__init__()

        # 将输入映射到潜在维度，作为 QP 的线性项 p（形状 [batch, Num_Latent]）
        self.fc_in = nn.Linear(Num_Input, Num_Latent)

        # 用下三角掩码 M 强制 L 下三角结构：Q = (M ⊙ L) (M ⊙ L)^T + eps*I
        self.M = Variable(torch.tril(torch.ones(Num_Latent, Num_Latent)))  # 常量下三角掩码
        self.L = Parameter(torch.tril(torch.ones(Num_Latent, Num_Latent)))  # 可学习下三角参数

        # 不等式约束 G x <= h，其中 h = G z^T + s
        self.G = Parameter(torch.Tensor(nineq, Num_Latent).uniform_(-1, 1))  # [nineq, latent]
        self.z = Parameter(torch.zeros(Num_Latent))  # [latent]
        self.s = Parameter(torch.ones(nineq))  # [nineq]

        # 等式约束 A x = b
        self.A = Parameter(torch.Tensor(neq, Num_Latent).uniform_(-1, 1))  # [neq, latent]
        self.b = Parameter(torch.ones(neq))  # [neq]

        # QP 解 x 再线性映射到最终输出，并用 Sigmoid 压到 (0,1)
        self.fc_out = nn.Linear(Num_Latent, Num_Output)

        # 记录形参
        self.num_input = Num_Input
        self.num_output = Num_Output
        self.num_latent = Num_Latent
        self.neq = neq
        self.ineq = nineq
        self.eps = eps

    def forward(self, x):
        # x: [batch, Num_Input]
        nBatch = x.size(0)
        x = x.view(nBatch, -1)
        x = F.relu(self.fc_in(x))  # 作为 QP 的线性项 p（记为 inputs，下方传给 QPFunction）

        # 构造 Q、G、h、A、b 的 batch 版本
        L = self.M * self.L
        Q = L.mm(L.t()) + self.eps * Variable(torch.eye(self.num_latent))  # [latent, latent]
        Q = Q.unsqueeze(0).expand(nBatch, self.num_latent, self.num_latent)  # [batch, latent, latent]
        G = self.G.unsqueeze(0).expand(nBatch, self.ineq, self.num_latent)  # [batch, nineq, latent]
        z = self.z.unsqueeze(0).expand(nBatch, self.num_latent)  # [batch, latent]
        s = self.s.unsqueeze(0).expand(nBatch, self.ineq)  # [batch, nineq]

        # h = z * G^T + s；注意此处 h 与输入无关，仅由可学习参数生成
        h = z.mm(self.G.t()) + s  # [batch, nineq]
        A = self.A.unsqueeze(0).expand(nBatch, self.neq, self.num_latent)  # [batch, neq, latent]
        b = self.b.unsqueeze(0).expand(nBatch, self.neq)  # [batch, neq]

        inputs = x  # 命名对齐 QP 接口含义：inputs 即线性项 p

        # 调用 qpth 的 QPFunction：
        #   minimize 0.5 * x^T Q x + inputs^T x
        #   s.t.     G x <= h,  A x = b
        x = QPFunction(verbose=-1)(
            Q.float(), inputs.float(), G.float(), h.float(), A.float(), b.float()
        )  # 返回 QP 解：x ∈ R^{Num_Latent}，形状 [batch, latent]

        # QP 解 -> 线性层 -> Sigmoid，得到最终输出
        x = x.float()
        x = torch.sigmoid(self.fc_out(x))  # [batch, Num_Output]
        return x

class OptNetLatent_Q(nn.Module):
    def __init__(self, Num_Input, Num_Output, Num_Latent, Qpenalty, neq, nineq, eps=1e-4):
        super().__init__()

        self.fc_in = nn.Linear(Num_Input, Num_Latent)

        self.M = Variable(torch.tril(torch.ones(Num_Latent, Num_Latent)))
        self.L = Parameter(torch.tril(torch.ones(Num_Latent, Num_Latent)))

        self.G = Parameter(torch.Tensor(nineq, Num_Latent).uniform_(-1, 1))

        self.A = Parameter(torch.Tensor(neq, Num_Latent).uniform_(-1, 1))
        self.b = Parameter(torch.ones(neq))

        self.fc_out = nn.Linear(Num_Latent, Num_Output)

        self.num_input = Num_Input
        self.num_output = Num_Output
        self.num_latent = Num_Latent
        self.neq = neq
        self.ineq = nineq
        self.eps = eps

    def forward(self, x, h1, Q1, G1, A1, b1):
        nBatch = x.size(0)
        x = x.view(nBatch, -1)
        x = F.relu(self.fc_in(x))

        # L = self.M * self.L
        # Q = L.mm(L.t()) + self.eps * Variable(torch.eye(self.num_latent))
        # Q = Q.unsqueeze(0).expand(nBatch, self.num_latent, self.num_latent)
        # G = self.G.unsqueeze(0).expand(nBatch, self.ineq, self.num_latent)
        # z = self.z.unsqueeze(0).expand(nBatch, self.num_latent)
        # s = self.s.unsqueeze(0).expand(nBatch, self.ineq)

        # h = z.mm(self.G.t()) + s
        # A = self.A.unsqueeze(0).expand(nBatch, self.neq, self.num_latent)
        # b = self.b.unsqueeze(0).expand(nBatch, self.neq)

        x = QPFunction(verbose=-1, eps=1e-4)(
            Q1.double(), x.double(), G1.double(), h1.double(), A1.double(), b1.double()
        )
        # print(Q.dtype)
        # xx = Dequantize(x, Q_output, RQM_output)

        x1 = x.float()
        x2 = torch.sigmoid(self.fc_out(x1))

        return x2


class OptNetLatent_Q_Train(nn.Module):
    # 与上一个类几乎一致，但在 QPFunction 中使用 double 精度，
    # 并在 forward 返回 (x2, x1, h) 以便训练/监控：
    #   - x1: QP 原始解（float）
    #   - x2: x1 经过 fc_out+sigmoid 的网络输出
    #   - h : 当前 batch 的不等式右端（监控约束裕度/可视化）
    def __init__(self, Num_Input, Num_Output, Num_Latent, Qpenalty, neq, nineq, eps=1e-12):
        super().__init__()

        self.fc_in = nn.Linear(Num_Input, Num_Latent)

        self.M = Variable(torch.tril(torch.ones(Num_Latent, Num_Latent)))
        self.L = Parameter(torch.tril(torch.ones(Num_Latent, Num_Latent)))

        self.G = Parameter(torch.Tensor(nineq, Num_Latent).uniform_(-1, 1))
        self.z = Parameter(torch.zeros(Num_Latent))
        self.s = Parameter(torch.ones(nineq))

        self.A = Parameter(torch.Tensor(neq, Num_Latent).uniform_(-1, 1))
        self.b = Parameter(torch.ones(neq))

        self.fc_out = nn.Linear(Num_Latent, Num_Output)

        self.num_input = Num_Input
        self.num_output = Num_Output
        self.num_latent = Num_Latent
        self.neq = neq
        self.ineq = nineq
        self.eps = eps

    def forward(self, x):
        # x: [batch, Num_Input]
        nBatch = x.size(0)
        x = x.view(nBatch, -1)
        x = F.relu(self.fc_in(x))  # 作为 QP 的线性项 p

        # 构造 Q、G、h、A、b（与上一个类相同）
        L = self.M * self.L
        Q = L.mm(L.t()) + self.eps * Variable(torch.eye(self.num_latent))  # [latent, latent]
        Q = Q.unsqueeze(0).expand(nBatch, self.num_latent, self.num_latent)  # [batch, latent, latent]
        G = self.G.unsqueeze(0).expand(nBatch, self.ineq, self.num_latent)  # [batch, nineq, latent]
        z = self.z.unsqueeze(0).expand(nBatch, self.num_latent)  # [batch, latent]
        s = self.s.unsqueeze(0).expand(nBatch, self.ineq)  # [batch, nineq]

        h = z.mm(self.G.t()) + s  # [batch, nineq]
        A = self.A.unsqueeze(0).expand(nBatch, self.neq, self.num_latent)  # [batch, neq, latent]
        b = self.b.unsqueeze(0).expand(nBatch, self.neq)  # [batch, neq]

        inputs = x  # QP 的线性项 p

        # 使用 double 精度提升数值稳定性
        x = QPFunction(verbose=-1)(
            Q.double(), inputs.double(), G.double(), h.double(), A.double(), b.double()
        )  # [batch, latent]

        x1 = x.float()  # QP 原始解
        x2 = torch.sigmoid(self.fc_out(x1))  # 最终输出
        return x2, x1, h  # (网络输出, QP 解, 不等式右端)
