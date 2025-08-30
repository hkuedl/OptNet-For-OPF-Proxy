clear;
clc;

load('PGEN_ipopt.mat')
load('PGEN_Pre.mat')

%% 主图：PG1–PG4 对比（Optimizer vs NN-OptNet）
% PGEN_ipopt: [N x 2], 第1列=PG1, 第2列=PG4
% PGEN_Pre:   [N x 2], 第1列=PG1, 第2列=PG4

% --- 读取列 ---
PG1_opt = PGEN_ipopt(:,1);
PG4_opt = PGEN_ipopt(:,2);

PG1_nn  = PGEN_Pre(:,1);
PG4_nn  = PGEN_Pre(:,2);

% --- 图窗口 ---
figure('Color','w','Units','inches','Position',[1 1 4.2 3.3]);
hold on;

% --- 绘制散点（与示例风格接近：蓝色空心圆 vs 橙色加号）---
% 如果点太多导致显示慢，可把 'idx = 1:5:numel(PG1_opt);' 做抽样
plot(PG1_opt, PG4_opt, 'o', ...
    'MarkerSize', 3, ...
    'MarkerEdgeColor', [0 0.4470 0.7410], ...   % 蓝
    'MarkerFaceColor', 'none', ...
    'LineStyle', 'none');

plot(PG1_nn,  PG4_nn,  '+', ...
    'MarkerSize', 3, ...
    'Color', [0.8500 0.3250 0.0980], ...        % 橙
    'LineStyle', 'none');

% --- 竖线（上限）---
x_up = 318;                  
xline(x_up, '--', 'Up limit', ...
    'Color', [0.0 0.6 0.0], ...                 % 绿色虚线
    'LineWidth', 1.5, ...
    'LabelOrientation', 'horizontal', ...
    'LabelVerticalAlignment', 'bottom');

% --- 轴与标注 ---
set(gca,'FontName','Times New Roman','FontSize',12,'LineWidth',1);
xlabel('PG1');
ylabel('PG4');

% 轴范围（与示例相近；如希望自动，改为 axis tight）
xmin = min([PG1_opt; PG1_nn]); xmax = max([PG1_opt; PG1_nn]);
ymin = min([PG4_opt; PG4_nn]); ymax = max([PG4_opt; PG4_nn]);
xmargin = 0.02*(xmax-xmin); ymargin = 0.05*(ymax-ymin);
xlim([xmin-xmargin, xmax+xmargin]);
ylim([ymin-ymargin, ymax+ymargin]);

grid on; box on;

% --- 图例 ---
lg = legend('Optimizer','NN-OptNet','Up limit', ...
    'Location','southwest');
set(lg,'Box','off');

hold off;

% 如需保存：
% exportgraphics(gcf,'PG1_PG4_main_only.png','Resolution',300);
