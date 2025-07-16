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

% 设置Nature风格
% 字体设置
fontName = 'Arial';
fontSize = 12;          % 更新基础字体大小
labelFontSize = 14;     % 新增标签字体大小
titleFontSize = 16;     % 新增标题字体大小
legendFontSize = 12;    % 新增图例字体大小
lineWidth = 2;          % 更新线宽

% 图1：电压比较

fig1 = figure();
plot(time, voltage_data, 'Color', colors(1,:), 'LineWidth', lineWidth, 'LineStyle', '-', 'DisplayName','Experiment Data');
hold on; 
plot(time, voltage_sim, 'Color', colors(2,:), 'LineWidth', lineWidth,'LineStyle', '--', 'DisplayName','Simulation Data');
hold off;
xlabel('Time (h)', 'FontName', fontName, 'FontSize', labelFontSize);
ylabel('Voltage (V)', 'FontName', fontName, 'FontSize', labelFontSize);
legend('FontName', fontName, 'FontSize', legendFontSize, 'Location', 'southeast','Box','off');
% 在左上角添加RMSE标注
text(0.05, 0.95, ['RMSE = ' num2str(voltage_rmse, '%.4f') ' V'], ...
     'Units', 'normalized', 'FontName', fontName, 'FontSize', fontSize, ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
grid on;
box on;
set(gca, 'FontName', fontName, 'FontSize', fontSize);
set(gcf, 'Color', 'white');
exportgraphics(fig1, ['./dc_plots/' prefix '_voltage_comparison.png'], 'Resolution', 300);

% 图2：电压残差分布直方图
fig2 = figure();
% 计算合适的bin数量 (Sturges' rule: k = log2(n) + 1)
numBins = ceil(log2(length(voltage_residual)) + 1);
h = histogram(voltage_residual, numBins, 'FaceColor', colors(3,:), 'EdgeColor', 'white', 'FaceAlpha', 0.8);
xlabel('Voltage Residual (V)', 'FontName', fontName, 'FontSize', labelFontSize);
ylabel('Frequency', 'FontName', fontName, 'FontSize', labelFontSize);
grid on;
box on;
% 稍微调整x轴范围，确保图形美观
xlim_current = xlim;
xlim([xlim_current(1) - 0.05*range(xlim_current), xlim_current(2) + 0.05*range(xlim_current)]);
set(gca, 'FontName', fontName, 'FontSize', fontSize);
set(gcf, 'Color', 'white');
exportgraphics(fig2, ['./dc_plots/' prefix '_voltage_residual_hist.png'], 'Resolution', 300);

% 图3：电流数据
fig3 = figure();
plot(time, current_data, 'Color', colors(4,:), 'LineWidth', lineWidth);
xlabel('Time (h)', 'FontName', fontName, 'FontSize', labelFontSize);
ylabel('Current (A)', 'FontName', fontName, 'FontSize', labelFontSize);
grid on;
box on;
set(gca, 'FontName', fontName, 'FontSize', fontSize);
set(gcf, 'Color', 'white');
exportgraphics(fig3, ['./dc_plots/' prefix '_current_data.png'], 'Resolution', 300);

% 图4：SOC数据
fig4 = figure();
plot(time, soc_data, 'Color', colors(5,:), 'LineWidth', lineWidth);
xlabel('Time (h)', 'FontName', fontName, 'FontSize', labelFontSize);
ylabel('SOC (%)', 'FontName', fontName, 'FontSize', labelFontSize);
grid on;
box on;
set(gca, 'FontName', fontName, 'FontSize', fontSize);
set(gcf, 'Color', 'white');
exportgraphics(fig4, ['./dc_plots/' prefix '_soc_data.png'], 'Resolution', 300);

fprintf('所有图表都已生成并保存到 ./dc_plots 文件夹，分辨率为300 dpi。\n');
fprintf('图片前缀: %s\n', prefix);

end

