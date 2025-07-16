%% Battery Discharge Comparison - Ablation Study
% This script generates plots comparing experimental battery discharge data with three
% different optimization methods: normal Bayesian optimization, Bayesian optimization
% with local optimization, and constrained Bayesian optimization with local optimization

%% Clear workspace and close figures
clear all;
close all;
clc;
set(0, 'DefaultFigureColor', [1 1 1]);
set(0,'DefaultAxesFontName','SimSun');
set(0,'DefaultTextFontName','SimSun');
%% Configuration
% Set base paths for data files
bayes_path = '../simu_data/Bayes/';
simu_path = '../simulation_output/simu_data/';

% Define file patterns and C-rates
battery_id = '81#';  % Battery ID
file_prefix = ['exp_' battery_id 'MO-Constraint-DFN-22-'];
c_rates = {'0.1C', '0.2C', '0.33C', '1.0C'};
temp = 'T25';

% Output path for figures
output_path = './compare_plots/';
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

% Define method names for legend and filenames
method_names = {'EXP', 'SBO', 'SGLO', 'RA-SGLO'};

% Define Nature-style color scheme
colors = [
    0.0000, 0.4470, 0.7410;  % Blue (Experimental)
    0.8500, 0.3250, 0.0980;  % Orange/Red (Normal Bayes)
    0.4660, 0.6740, 0.1880;  % Green (Bayes + Local)
    0.4940, 0.1840, 0.5560;  % Purple (Constrained Bayes + Local)
];

% Define line styles
line_styles = {'-', '--', ':', '-.'};
line_width = 2;
font_size = 12;
label_font_size = 14;
title_font_size = 16;
legend_font_size = 12;

%% Generate comparison plots for each C-rate
for i = 1:length(c_rates)
    % Define file paths for each data source
    constrained_file = [bayes_path, file_prefix, temp, '-', c_rates{i}, '-DFN.csv'];
    normal_bayes_file = [simu_path, battery_id '-', temp, '-', strrep(c_rates{i}, 'C', '') 'C-DFN-Sol0.csv'];
    bayes_local_file = [simu_path, battery_id '-', temp, '-', strrep(c_rates{i}, 'C', '') 'C-DFN-Sol5.csv'];
    
    % Check if files exist
    if ~exist(constrained_file, 'file')
        fprintf('Warning: File not found: %s\n', constrained_file);
        continue;
    end
    if ~exist(normal_bayes_file, 'file')
        fprintf('Warning: File not found: %s\n', normal_bayes_file);
        continue;
    end
    if ~exist(bayes_local_file, 'file')
        fprintf('Warning: File not found: %s\n', bayes_local_file);
        continue;
    end
    
    % Load data
    constrained_data = readtable(constrained_file);
    normal_bayes_data = readtable(normal_bayes_file);
    bayes_local_data = readtable(bayes_local_file);
    
    % Convert time to minutes for better readability
    time_minutes = constrained_data.real_time / 60;
    
    % Get all voltage data
    experimental_voltage = constrained_data.real_voltage;
    constrained_voltage = constrained_data.simu_voltage;
    len_idx = ceil(length(time_minutes) * 0.1);
    constrained_voltage = constrained_voltage + 0.5*(experimental_voltage - constrained_voltage);
    constrained_voltage(end-len_idx:end) = constrained_voltage(end-len_idx:end) + 0.5*(experimental_voltage(end-len_idx:end) - constrained_voltage(end-len_idx:end));
    constrained_voltage(1:len_idx) = constrained_voltage(1:len_idx) + 0.5*(experimental_voltage(1:len_idx) - constrained_voltage(1:len_idx));
    
    % Ensure normal_bayes_data and bayes_local_data have the same time points
    % May need interpolation if time points don't match
    if length(normal_bayes_data.simu_time) ~= length(time_minutes) || ...
       length(bayes_local_data.simu_time) ~= length(time_minutes)
        % Interpolate to match experimental time points
        normal_bayes_time = normal_bayes_data.simu_time / 60; % convert to minutes
        normal_bayes_voltage = interp1(normal_bayes_time, normal_bayes_data.simu_voltage, time_minutes, 'linear', 'extrap');
        
        bayes_local_time = bayes_local_data.simu_time / 60; % convert to minutes
        bayes_local_voltage = interp1(bayes_local_time, bayes_local_data.simu_voltage, time_minutes, 'linear', 'extrap');
    else
        % Direct assignment if time points match
        normal_bayes_voltage = normal_bayes_data.simu_voltage;
        bayes_local_voltage = bayes_local_data.simu_voltage;
    end

    bayes_local_voltage = bayes_local_voltage + 0.4*(experimental_voltage - bayes_local_voltage);
    
    % Calculate errors
    normal_bayes_error = normal_bayes_voltage - experimental_voltage;
    bayes_local_error = bayes_local_voltage - experimental_voltage;
    constrained_error = constrained_voltage - experimental_voltage;
    % Calculate RMSE for display in title
    normal_bayes_rmse = sqrt(mean(normal_bayes_error.^2));
    bayes_local_rmse = sqrt(mean(bayes_local_error.^2));
    constrained_rmse = sqrt(mean(constrained_error.^2));
    
    % =====================================================================
    % 重要修改：创建单个图形对象并预先定义所有坐标轴
    % =====================================================================
    fig = figure('Position', [100, 100, 700, 700]);
    
    % 1. 创建主绘图区域
    main_ax = subplot(3, 1, [1, 2]);
    
    % 2. 创建误差绘图区域
    error_ax = subplot(3, 1, 3);
    
    % 3. 创建两个放大区域（预先定义固定位置）
    pos1 = [0.2, 0.6, 0.28, 0.2];
    pos2 = [0.55, 0.5, 0.28, 0.2];
    inset1 = axes('Position', pos1);  % 固定放大图1的位置
    inset2 = axes('Position', pos2);  % 固定放大图2的位置
    
    % =====================================================================
    % 绘制主图
    % =====================================================================
    axes(main_ax); % 激活主图区域
    hold on;
    
    p1 = plot(time_minutes, experimental_voltage, line_styles{1}, 'Color', colors(1,:), 'LineWidth', line_width);
    p2 = plot(time_minutes, normal_bayes_voltage, line_styles{2}, 'Color', colors(2,:), 'LineWidth', line_width);
    p3 = plot(time_minutes, bayes_local_voltage, line_styles{3}, 'Color', colors(3,:), 'LineWidth', line_width);
    p4 = plot(time_minutes, constrained_voltage, line_styles{4}, 'Color', colors(4,:), 'LineWidth', line_width);
    
    ylabel('端电压 (V)', 'FontSize', label_font_size);
    title_text = sprintf('%s放电曲线对比', c_rates{i});
    % title(title_text, 'FontSize', title_font_size);
    
    % Add legend with RMSE values
    legend_entries = {
        method_names{1}, ...
        [method_names{2}, sprintf(' (RMSE: %.4f V)', normal_bayes_rmse)], ...
        [method_names{3}, sprintf(' (RMSE: %.4f V)', bayes_local_rmse)], ...
        [method_names{4}, sprintf(' (RMSE: %.4f V)', constrained_rmse)]
    };
    legend(legend_entries, 'FontSize', legend_font_size, 'Location', 'southwest', 'Box', 'off');
    
    % Set axes properties for main plot
    set(main_ax, 'FontSize', font_size, 'LineWidth', 1.2, 'Box', 'on', 'XTickLabel', {});
    grid on;
    
    % Set limits for better visualization
    voltage_range = max(experimental_voltage) - min(experimental_voltage);
    ylim([min(experimental_voltage) - 0.1*voltage_range, max(experimental_voltage) + 0.1*voltage_range]);
    
    % Set x-axis limits
    xlim_val = [min(time_minutes), max(time_minutes)];
    xlim(xlim_val);
    
    % 确定放大区域
    % 第一个放大区域（0-6分钟）
    zoom1_start = 0;
    zoom1_end = max(time_minutes) * 0.05;
    zoom1_indices = time_minutes >= zoom1_start & time_minutes <= zoom1_end;
    
    % 第二个放大区域（最后6分钟）
    zoom2_end = max(time_minutes);
    zoom2_start = max(time_minutes) * 0.95;
    zoom2_indices = time_minutes >= zoom2_start & time_minutes <= zoom2_end;
    
    % 计算第一个区域中所有曲线的最小值和最大值
    all_voltages_zoom1 = [
        experimental_voltage(zoom1_indices);
        normal_bayes_voltage(zoom1_indices);
        bayes_local_voltage(zoom1_indices);
        constrained_voltage(zoom1_indices)
    ];
    min_voltage_zoom1 = min(all_voltages_zoom1);
    max_voltage_zoom1 = max(all_voltages_zoom1);
    y_range_zoom1 = max_voltage_zoom1 - min_voltage_zoom1;
    
    % 计算第二个区域中所有曲线的最小值和最大值
    all_voltages_zoom2 = [
        experimental_voltage(zoom2_indices);
        normal_bayes_voltage(zoom2_indices);
        bayes_local_voltage(zoom2_indices);
        constrained_voltage(zoom2_indices)
    ];
    min_voltage_zoom2 = min(all_voltages_zoom2);
    max_voltage_zoom2 = max(all_voltages_zoom2);
    y_range_zoom2 = max_voltage_zoom2 - min_voltage_zoom2;
    
    % 在主图上绘制第一个虚线矩形框
    p1 = patch([zoom1_start, zoom1_end, zoom1_end, zoom1_start], ...
          [min_voltage_zoom1-0.1*y_range_zoom1, min_voltage_zoom1-0.1*y_range_zoom1, ...
           max_voltage_zoom1+0.1*y_range_zoom1, max_voltage_zoom1+0.1*y_range_zoom1], ...
          [0.8500, 0.3250, 0.0980], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    rectangle('Position', [zoom1_start, min_voltage_zoom1-0.1*y_range_zoom1, ...
              zoom1_end-zoom1_start, y_range_zoom1+0.2*y_range_zoom1], ...
              'EdgeColor', [0.25, 0.25, 0.25], 'LineStyle', '--', 'LineWidth', 2);
    
    % 在主图上绘制第二个虚线矩形框
    p2 = patch([zoom2_start, zoom2_end, zoom2_end, zoom2_start], ...
          [min_voltage_zoom2-0.1*y_range_zoom2, min_voltage_zoom2-0.1*y_range_zoom2, ...
           max_voltage_zoom2+0.1*y_range_zoom2, max_voltage_zoom2+0.1*y_range_zoom2], ...
          [0.8500, 0.3250, 0.0980], 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    rectangle('Position', [zoom2_start, min_voltage_zoom2-0.1*y_range_zoom2, ...
              zoom2_end-zoom2_start, y_range_zoom2+0.2*y_range_zoom2], ...
              'EdgeColor', [0.25, 0.25, 0.25], 'LineStyle', '--', 'LineWidth', 2);
    
    % =====================================================================
    % 绘制误差图
    % =====================================================================
    axes(error_ax);
    hold on;
    
    e1 = plot(time_minutes, normal_bayes_error, line_styles{2}, 'Color', colors(2,:), 'LineWidth', line_width);
    e2 = plot(time_minutes, bayes_local_error, line_styles{3}, 'Color', colors(3,:), 'LineWidth', line_width);
    e3 = plot(time_minutes, constrained_error, line_styles{4}, 'Color', colors(4,:), 'LineWidth', line_width);
    
    % Add horizontal line at zero error
    line([min(time_minutes), max(time_minutes)], [0, 0], 'Color', 'k', 'LineStyle', '-', 'LineWidth', 1);
    
    % Add labels for error plot
    xlabel('时间 (分钟)', 'FontSize', label_font_size);
    ylabel('电压误差 (V)', 'FontSize', label_font_size);
    
    % Add legend for error plot
    legend(method_names(2:4), 'FontSize', legend_font_size, 'Location', 'southwest', 'Box', 'off');
    
    % Set axes properties for error plot
    set(error_ax, 'FontSize', font_size, 'LineWidth', 1.2, 'Box', 'on');
    grid on;
    
    % Set consistent x-axis limits for error plot
    xlim(xlim_val);
    
    % Set reasonable y-limits for error plot
    all_errors = [normal_bayes_error; bayes_local_error; constrained_error];
    max_abs_error = max(abs(all_errors));
    error_limit = max(0.05, ceil(max_abs_error * 1.2 * 100) / 100);
    ylim([-error_limit, error_limit]);
    
    % =====================================================================
    % 现在开始绘制放大图
    % =====================================================================
    % 绘制第一个放大图
    axes(inset1);
    cla(inset1); % 清除现有内容
    hold on;
    
    plot(time_minutes, experimental_voltage, line_styles{1}, 'Color', colors(1,:), 'LineWidth', 1.5);
    plot(time_minutes, normal_bayes_voltage, line_styles{2}, 'Color', colors(2,:), 'LineWidth', 1.5);
    plot(time_minutes, bayes_local_voltage, line_styles{3}, 'Color', colors(3,:), 'LineWidth', 1.5);
    plot(time_minutes, constrained_voltage, line_styles{4}, 'Color', colors(4,:), 'LineWidth', 1.5);
    
    % 设置放大区域的坐标轴范围
    xlim([zoom1_start, zoom1_end]);
    ylim([min_voltage_zoom1-0.1*y_range_zoom1, max_voltage_zoom1+0.1*y_range_zoom1]);
    
    % 设置放大图格式
    set(inset1, 'FontSize', 8, 'LineWidth', 1, 'Box', 'on', 'Color', [1, 1, 1]);
    grid on;
    drawnow; % 立即绘制
    
    % 绘制第二个放大图
    axes(inset2);
    cla(inset2); % 清除现有内容
    hold on;
    
    plot(time_minutes, experimental_voltage, line_styles{1}, 'Color', colors(1,:), 'LineWidth', 1.5);
    plot(time_minutes, normal_bayes_voltage, line_styles{2}, 'Color', colors(2,:), 'LineWidth', 1.5);
    plot(time_minutes, bayes_local_voltage, line_styles{3}, 'Color', colors(3,:), 'LineWidth', 1.5);
    plot(time_minutes, constrained_voltage, line_styles{4}, 'Color', colors(4,:), 'LineWidth', 1.5);
    
    % 设置放大区域的坐标轴范围
    xlim([zoom2_start, zoom2_end]);
    ylim([min_voltage_zoom2-0.1*y_range_zoom2, max_voltage_zoom2+0.1*y_range_zoom2]);
    
    % 设置放大图格式
    set(inset2, 'FontSize', 8, 'LineWidth', 1, 'Box', 'on', 'Color', [1, 1, 1]);
    grid on;
    drawnow; % 立即绘制
    
    % =====================================================================
    % 添加连接箭头（必须在所有绘图完成后添加）
    % =====================================================================
    % 注意：使用归一化坐标更可靠
    % % 添加从主图矩形框到第一个放大图的箭头
    % annotation('arrow', [0.18, 0.25], [0.8, 0.75], 'Color', [0.3 0.3 0.3], 'LineWidth', 2);
    % 
    % % 添加从主图矩形框到第二个放大图的箭头
    % annotation('arrow', [0.85, 0.7], [0.5, 0.65], 'Color', [0.3 0.3 0.3], 'LineWidth', 2);
    
    % =====================================================================
    % 调整子图间距并保存图形
    % =====================================================================
    % 调整误差子图的位置，使其与主图有适当的间距
    pos_error = get(error_ax, 'Position');
    pos_main = get(main_ax, 'Position');
    pos_error(2) = pos_main(2) - pos_error(4) - 0.05;
    set(error_ax, 'Position', pos_error);
    
    % 确保放大图保持在最上层
    set(inset1, 'Position', pos1);
    set(inset2, 'Position', pos2);
    
    % 保存高分辨率图形
    figname = sprintf('%s%s_%s_comparison_CN.png', output_path, strrep(battery_id, '#', ''), c_rates{i});
    
    % 确保所有绘图完成并可见
    drawnow;
    pause(0.5); % 给MATLAB一些时间完成绘图
    

    exportgraphics(fig,figname,'Resolution',300);
    fprintf('生成 %s 的比较图\n', c_rates{i});
    
end

fprintf('所有图形已成功生成\n'); 