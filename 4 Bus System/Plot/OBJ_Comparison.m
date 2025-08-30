clear;
clc;

%% ====== 加载数据 ======
load('Input.mat');
load('Cost_ipopt.mat');
load('Cost_Vec1.mat');

xdata = Input(9801:end,1);
ydata = Input(9801:end,2);
zdata1 = Cost_ipopt;
zdata2 = Cost_Vec1;

createfigure2(Input(9801:end,1), Input(9801:end,2), Cost_ipopt, Cost_Vec1);