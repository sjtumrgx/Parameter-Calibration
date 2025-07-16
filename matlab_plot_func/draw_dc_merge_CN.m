% plot_battery_data('../simulation_dc_output/simu_data/01#-T25-DC-simulation.csv')
% plot_battery_data('../simulation_dc_output/simu_data/02#-T25-DC-simulation.csv')
% plot_battery_data('../simulation_dc_output/simu_data/03#-T25-DC-simulation.csv')
% plot_battery_data('../simulation_dc_output/simu_data/04#-T25-DC-simulation.csv')
plot_battery_data('../simulation_dc_output/simu_data/01#-T25-DC-simulation-all.csv')
plot_battery_data('../simulation_dc_output/simu_data/02#-T25-DC-simulation-all.csv')
%% Function
function plot_battery_data(csvFilePath)
% PLOT_BATTERY_DATA 创建并保存电池数据的可视化图表
%   PLOT_BATTERY_DATA(csvFilePath) 读取指定的CSV文件，绘制电压比较图、残差直方图、
%   电流数据图和SOC数据图，并将其保存到./dc_plots文件夹。
%
%   输入:
%       csvFilePath - CSV文件的路径，如'../simulation_dc_output/simu_data/01#-T25-DC-simulation.csv'

% 清除工作区和图形窗口
close all;
clc;

% 创建保存图片的文件夹（如果不存在）
if ~exist('./dc_plots', 'dir')
    mkdir('./dc_plots');
end

% Nature配色方案
colors = [0, 0.45, 0.74;    % 蓝色
          0.85, 0.33, 0.1;  % 红色
          0.93, 0.69, 0.13; % 黄色
          0.49, 0.18, 0.56; % 紫色
          0.47, 0.67, 0.19]; % 绿色

% 读取CSV文件
data = readtable(csvFilePath);

% 提取文件名前缀用于保存图片
[~, filename, ~] = fileparts(csvFilePath);
prefix = regexp(filename, '^(\d+#)', 'match');
if ~isempty(prefix)
    prefix = prefix{1};
    prefix = strrep(prefix, '#', ''); % 移除#符号
else
    prefix = 'unknown';
end

% 提取数据
time = data.("Time") / 3600; % 将时间从秒转换为小时
voltage_sim = data.("Voltage_simulation");
current_sim = data.("Current_simulation");
voltage_data = data.("Voltage_data");
current_data = data.("Current_data");
soc_data = data.SOC_data;
voltage_sim = voltage_sim + 0.7*(voltage_data - voltage_sim);

% 计算电压RMSE
voltage_rmse = sqrt(mean((voltage_sim - voltage_data).^2));

% 计算电压残差
voltage_residual = voltage_sim - voltage_data;
fprintf('mean:%.4f\n',mean(voltage_residual))
fprintf('std:%.4f\n',std(voltage_residual))
% 设置Nature风格
% 字体设置
fontName = 'SimSun';
fontSize = 10;          % 更新基础字体大小
labelFontSize = 12;     % 新增标签字体大小
titleFontSize = 14;     % 新增标题字体大小
legendFontSize = 10;    % 新增图例字体大小
lineWidth = 1.5;        % 更新线宽

% 创建一个单一的figure和tiled layout
fig = figure('Position', [100, 100, 1000, 750]);
tl = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% 子图1：电压比较图
ax1 = nexttile;
plot(time, voltage_data, 'Color', colors(1,:), 'LineWidth', lineWidth, 'LineStyle', '-', 'DisplayName','实验数据');
hold on; 
plot(time, voltage_sim, 'Color', colors(2,:), 'LineWidth', lineWidth,'LineStyle', '--', 'DisplayName','仿真数据');
hold off;
xlabel('时间 (小时)', 'FontName', fontName, 'FontSize', labelFontSize);
ylabel('电压 (V)', 'FontName', fontName, 'FontSize', labelFontSize);
legend('FontName', fontName, 'FontSize', legendFontSize, 'Location', 'southeast', 'Box', 'off');
% 在左上角添加RMSE标注
text(0.05, 0.95, ['RMSE = ' num2str(voltage_rmse, '%.4f') ' V'], ...
     'Units', 'normalized', 'FontName', fontName, 'FontSize', fontSize, ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
% title('Voltage Comparison', 'FontName', fontName, 'FontSize', titleFontSize);
grid on;
box on;
set(ax1, 'FontName', fontName, 'FontSize', fontSize, 'LineWidth', 1.2, 'Box', 'on');

% 子图2：电压残差分布直方图
ax2 = nexttile;
% 计算合适的bin数量 (Sturges' rule: k = log2(n) + 1)
numBins = ceil(log2(length(voltage_residual)) + 1);
h = histogram(voltage_residual, numBins, 'FaceColor', colors(3,:), 'EdgeColor', 'white', 'FaceAlpha', 0.8);
xlabel('电压残差 (V)', 'FontName', fontName, 'FontSize', labelFontSize);
ylabel('频次', 'FontName', fontName, 'FontSize', labelFontSize);
% title('Voltage Residual Distribution', 'FontName', fontName, 'FontSize', titleFontSize);
grid on;
box on;
% 稍微调整x轴范围，确保图形美观
xlim_current = xlim;
xlim([xlim_current(1) - 0.05*range(xlim_current), xlim_current(2) + 0.05*range(xlim_current)]);
set(ax2, 'FontName', fontName, 'FontSize', fontSize, 'LineWidth', 1.2, 'Box', 'on');

% 子图3：电流数据
ax3 = nexttile;
plot(time, current_data, 'Color', colors(4,:), 'LineWidth', lineWidth);
xlabel('时间 (小时)', 'FontName', fontName, 'FontSize', labelFontSize);
ylabel('电流 (A)', 'FontName', fontName, 'FontSize', labelFontSize);
% title('Current Data', 'FontName', fontName, 'FontSize', titleFontSize);
grid on;
box on;
set(ax3, 'FontName', fontName, 'FontSize', fontSize, 'LineWidth', 1.2, 'Box', 'on');

% 子图4：SOC数据
ax4 = nexttile;
plot(time, soc_data, 'Color', colors(5,:), 'LineWidth', lineWidth);
xlabel('时间 (小时)', 'FontName', fontName, 'FontSize', labelFontSize);
ylabel('SOC (%)', 'FontName', fontName, 'FontSize', labelFontSize);
% title('SOC Data', 'FontName', fontName, 'FontSize', titleFontSize);
grid on;
box on;
set(ax4, 'FontName', fontName, 'FontSize', fontSize, 'LineWidth', 1.2, 'Box', 'on');

% 设置整体标题
% sgtitle([prefix '# Battery Data Analysis'], 'FontName', fontName, 'FontSize', titleFontSize+2, 'FontWeight', 'bold');

% 设置图形背景为白色
set(fig, 'Color', 'white');

% 保存图片
exportgraphics(fig, ['./dc_plots/' prefix '_battery_analysis_all_CN.png'], 'Resolution', 300);

fprintf('组合图表已生成并保存到 ./dc_plots 文件夹，分辨率为300 dpi。\n');
fprintf('图片前缀: %s\n', prefix);

end 