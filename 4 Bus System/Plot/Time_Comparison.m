%% 前 k 个时间的累计总和柱状图
% t 为你的 1x600（或 600x1）时间向量，单位：秒

load('Time_Stage2.mat');
load('Time_Stage13.mat');

t = Time_Stage13; % Stage1和3总消耗时间，可以替换成Stage2画出对应Stage2消耗的时间
if ~exist('t','var')
    error('未找到变量 t，请先将你的 1x600 时间向量赋给 t。');
end
t = t(:).';  % 统一为行向量

idx = [10 50 100 300 600];     % 需要展示的样本数
if numel(t) < max(idx)
    error('t 长度为 %d，不足以取到最大 k=%d。', numel(t), max(idx));
end

cumt = cumsum(t);              % 前缀和
vals = cumt(idx);              % 取前 k 的累计时间（秒）

% 画图（与示例风格接近）
figure('Color','w','Units','inches','Position',[1 1 3.8 3.0]);
b = bar(vals, 'BarWidth', 0.7);
b.FaceColor = [0.85 0.33 0.10];   % 橙色
b.EdgeColor = 'k';                % 黑色边框
b.LineWidth = 1;

set(gca, 'XTick', 1:numel(idx), 'XTickLabel', string(idx), ...
         'FontName','Times New Roman', 'FontSize', 12, ...
         'LineWidth', 1);
xlabel('# of samples');
ylabel('Cumulative Time (s)');    % 累计时间（秒）
ylim([0, max(vals)*1.15]);
box on; grid off;

% 可选：保存图片
% exportgraphics(gcf, 'cumulative_time_bar.png', 'Resolution', 300);
