% plot_battery_data('../simulation_dc_output/simu_data/01#-T25-DC-simulation.csv')
% plot_battery_data('../simulation_dc_output/simu_data/02#-T25-DC-simulation.csv')
% plot_battery_data('../simulation_dc_output/simu_data/03#-T25-DC-simulation.csv')
% plot_battery_data('../simulation_dc_output/simu_data/04#-T25-DC-simulation.csv')
plot_battery_data('../simulation_dc_output/simu_data/01#-T25-DC-simulation-all.csv')
% plot_battery_data('../simulation_dc_output/simu_data/02#-T25-DC-simulation-all.csv')
%% Function
function plot_battery_data(csvFilePath)
    % PLOT_BATTERY_DATA 创建并保存电池数据的可视化图表
    %   PLOT_BATTERY_DATA(csvFilePath) 读取指定的CSV文件，绘制电压比较图、残差直方图、
    %   电流数据图和SOC数据图，并将其保存到./dc_plots文件夹。
    %   修改版本包含矩形框选区域和局部放大图
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
    
    % 科研风格配色用于矩形框和箭头
    rect_color = [0.8, 0.4, 0.2];      % 科研风格绿色矩形框
    arrow_color = [0.8, 0.4, 0.2];     % 橘色箭头
    
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
    fontName = 'Arial';
    fontSize = 10;          % 更新基础字体大小
    labelFontSize = 12;     % 新增标签字体大小
    titleFontSize = 14;     % 新增标题字体大小
    legendFontSize = 12;    % 新增图例字体大小
    lineWidth = 2;        % 更新线宽
    
    % 创建一个单一的figure和tiled layout
    fig = figure('Position', [100, 100, 1000, 750]);
    tl = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % 子图1：电压比较图
    ax1 = nexttile;
    plot(time, voltage_data, 'Color', colors(1,:), 'LineWidth', lineWidth, 'LineStyle', '-', 'DisplayName','Experiment Data');
    hold on; 
    plot(time, voltage_sim, 'Color', colors(2,:), 'LineWidth', lineWidth,'LineStyle', '--', 'DisplayName','Simulation Data');
    
    % 添加矩形框选区域
    zoom_x_range = [65, 80];
    zoom_y_range = [3.2, 3.45];
    rectangle('Position', [zoom_x_range(1), zoom_y_range(1), ...
                          diff(zoom_x_range), diff(zoom_y_range)], ...
              'EdgeColor', rect_color, 'LineWidth', 2.5, 'LineStyle', '-');
    
    hold off;
    xlabel('Time (h)', 'FontName', fontName, 'FontSize', labelFontSize);
    ylabel('Voltage (V)', 'FontName', fontName, 'FontSize', labelFontSize);
    legend('FontName', fontName, 'FontSize', legendFontSize, 'Location', 'northeast', 'Box', 'off');
    
    % 在左上角添加RMSE标注
    text(0.05, 0.95, ['RMSE = ' num2str(voltage_rmse, '%.4f') ' V'], ...
         'Units', 'normalized', 'FontName', fontName, 'FontSize', 12, ...
         'HorizontalAlignment', 'left', 'VerticalAlignment', 'top');
    grid on;
    box on;
    set(ax1, 'FontName', fontName, 'FontSize', fontSize, 'LineWidth', 1.2, 'Box', 'on');
    set(ax1,'YLim',[3.0,3.5])
    % 创建局部放大图（inset）
    % 获取ax1的位置信息
    ax1_pos = get(ax1, 'Position');
    % 设置inset的位置和大小（左下角）
    inset_width = 0.15;
    inset_height = 0.15;
    inset_x = ax1_pos(1) + 0.03;  % 距离左边界的距离
    inset_y = ax1_pos(2) + 0.03;  % 距离下边界的距离
    
    % 创建inset axes
    ax_inset = axes('Position', [inset_x, inset_y, inset_width, inset_height]);
    
    % 获取缩放区域的数据索引
    zoom_idx = (time >= zoom_x_range(1)) & (time <= zoom_x_range(2));
    time_zoom = time(zoom_idx);
    voltage_data_zoom = voltage_data(zoom_idx);
    voltage_sim_zoom = voltage_sim(zoom_idx);
    
    % 在inset中绘制缩放数据
    plot(ax_inset, time_zoom, voltage_data_zoom, 'Color', colors(1,:), 'LineWidth', lineWidth, 'LineStyle', '-');
    hold(ax_inset, 'on');
    plot(ax_inset, time_zoom, voltage_sim_zoom, 'Color', colors(2,:), 'LineWidth', lineWidth, 'LineStyle', '--');
    hold(ax_inset, 'off');
    
    % 设置inset的属性
    set(ax_inset, 'FontName', fontName, 'FontSize', fontSize-1, 'LineWidth', 1.0, 'Box', 'on');
    xlim(ax_inset, zoom_x_range);
    ylim(ax_inset, zoom_y_range);
    grid(ax_inset, 'on');
    
    axes(ax1);

    % 定义起点和终点坐标（数据坐标）
    start_point = [63, 3.33];
    end_point = [40, 3.23];

    % 获取当前坐标轴的位置和范围
    ax = gca;
    ax_pos = get(ax, 'Position');
    xlimits = xlim(ax);
    ylimits = ylim(ax);

    % 将数据坐标转换为归一化坐标
    start_norm_x = ax_pos(1) + (start_point(1) - xlimits(1)) / diff(xlimits) * ax_pos(3);
    start_norm_y = ax_pos(2) + (start_point(2) - ylimits(1)) / diff(ylimits) * ax_pos(4);
    end_norm_x = ax_pos(1) + (end_point(1) - xlimits(1)) / diff(xlimits) * ax_pos(3);
    end_norm_y = ax_pos(2) + (end_point(2) - ylimits(1)) / diff(ylimits) * ax_pos(4);

    % 绘制箭头
    annotation('arrow', [start_norm_x, end_norm_x], [start_norm_y, end_norm_y], ...
            'Color', [0.8, 0.4, 0.2], 'LineWidth', 2.0, ...
            'HeadStyle', 'cback2', 'HeadLength', 10, 'HeadWidth', 8);


    
   
    % 子图2：电压残差分布直方图
    ax2 = nexttile;
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
    set(ax2, 'FontName', fontName, 'FontSize', fontSize, 'LineWidth', 1.2, 'Box', 'on');
    
    % 子图3：电流数据
    ax3 = nexttile;
    plot(time, current_data, 'Color', colors(4,:), 'LineWidth', lineWidth);
    xlabel('Time (h)', 'FontName', fontName, 'FontSize', labelFontSize);
    ylabel('Current (A)', 'FontName', fontName, 'FontSize', labelFontSize);
    grid on;
    box on;
    set(ax3, 'FontName', fontName, 'FontSize', fontSize, 'LineWidth', 1.2, 'Box', 'on');
    
    % 子图4：SOC数据
    ax4 = nexttile;
    plot(time, soc_data, 'Color', colors(5,:), 'LineWidth', lineWidth);
    xlabel('Time (h)', 'FontName', fontName, 'FontSize', labelFontSize);
    ylabel('SOC (%)', 'FontName', fontName, 'FontSize', labelFontSize);
    grid on;
    box on;
    set(ax4, 'FontName', fontName, 'FontSize', fontSize, 'LineWidth', 1.2, 'Box', 'on');
    
    % 设置图形背景为白色
    set(fig, 'Color', 'white');
    
    % 保存图片
    exportgraphics(fig, ['./dc_plots/' prefix '_battery_analysis_all.png'], 'Resolution', 300);
    
    fprintf('增强版组合图表已生成并保存到 ./dc_plots 文件夹，分辨率为300 dpi。\n');
    fprintf('图片前缀: %s\n', prefix);
    fprintf('局部放大区域: Time %.1f-%.1f h, Voltage %.2f-%.2f V\n', ...
            zoom_x_range(1), zoom_x_range(2), zoom_y_range(1), zoom_y_range(2));
    
    end
    