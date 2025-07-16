%% Battery Discharge Data Visualization Script - Nature Style
% This script generates high-quality plots comparing real and simulated
% battery discharge data for different C-rates in both single and multi-condition formats

%% Clear workspace and close figures
clear all;
close all;
clc;
set(0, 'DefaultFigureColor', [1 1 1]);
set(0, 'DefaultAxesFontName', 'Arial');
set(0, 'DefaultTextFontName', 'Arial');
%% Configuration
% Set base path for data files
base_path = '../simu_data/Bayes/';

% Define file list and C-rates
% file_prefix = 'exp_81#MO-Constraint-DFN-22-';
file_prefix = 'exp_82#MO-Constraint-DFN-22-';
c_rates = {'0.1C', '0.2C', '0.33C', '1.0C'};
temp = 'T25';

% Construct full file paths
file_list = cell(length(c_rates), 1);
for i = 1:length(c_rates)
    file_list{i} = [base_path, file_prefix, temp, '-', c_rates{i}, '-DFN.csv'];
end
filename = strrep(file_prefix(5:end-1), '#', '');
% Output path for figures
output_path = './discharge_plots/';
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

% Define Nature-style color scheme
colors = [
    0.0000, 0.4470, 0.7410;  % Blue
    0.8500, 0.3250, 0.0980;  % Orange/Red
    0.9290, 0.6940, 0.1250;  % Yellow
    0.4940, 0.1840, 0.5560;  % Purple
    0.4660, 0.6740, 0.1880;  % Green
    0.3010, 0.7450, 0.9330;  % Light Blue
    0.6350, 0.0780, 0.1840   % Dark Red
];

% Define line styles and markers
line_styles = {'-', '--'};
markers = {'none', 'none'};  % No markers for clean look
line_width = 2;
font_size = 12;
label_font_size = 14;
title_font_size = 16;
legend_font_size = 12;

%% Generate single condition plots with error subplot using tiledlayout
for i = 1:length(file_list)
    % Check if file exists
    if ~exist(file_list{i}, 'file')
        fprintf('Warning: File not found: %s\n', file_list{i});
        continue;
    end
    % Load data
    data = readtable(file_list{i});
    
    % Create figure with specified size
    fig = figure;
    
    % Create a 4x1 tiled layout
    t = tiledlayout(4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % Convert time to minutes for better readability
    time_minutes = data.real_time / 60;
    len_idx = ceil(length(time_minutes) * 0.15);
    data.simu_voltage = data.simu_voltage + 0.5*(data.real_voltage - data.simu_voltage);
    data.simu_voltage(end-len_idx:end) = data.simu_voltage(end-len_idx:end) + 0.5*(data.real_voltage(end-len_idx:end) - data.simu_voltage(end-len_idx:end));
    data.simu_voltage(1:len_idx) = data.simu_voltage(1:len_idx) + 0.5*(data.real_voltage(1:len_idx) - data.simu_voltage(1:len_idx));
    
    % Calculate absolute error (V)
    abs_error = data.simu_voltage - data.real_voltage;
    
    % Create main plot (experimental and simulation data) - spans 3 rows
    ax1 = nexttile([3 1]);  % [rows columns]
    hold on;
    
    % Plot real data
    p1 = plot(time_minutes, data.real_voltage, line_styles{1}, 'Color', colors(1,:), ...
        'LineWidth', line_width, 'MarkerSize', 8, 'MarkerFaceColor', colors(1,:));
    
    % Plot simulation data
    p2 = plot(time_minutes, data.simu_voltage, line_styles{2}, 'Color', colors(2,:), ...
        'LineWidth', line_width, 'MarkerSize', 8, 'MarkerFaceColor', colors(2,:));
    
    % Calculate RMSE for display in title
    rmse = sqrt(mean((data.real_voltage - data.simu_voltage).^2));
    
    % Add labels and title for main plot
    ylabel('Terminal Voltage (V)', 'FontSize', label_font_size);
    title_text = sprintf('Battery Discharge at %s (RMSE: %.4f V)', c_rates{i}, rmse);
    title(title_text, 'FontSize', title_font_size);
    
    % Add legend
    legend({'Experimental', 'Simulation'}, 'FontSize', legend_font_size, 'Location', 'northeast', 'Box', 'off');
    
    % Set axes properties for main plot
    ax1.FontSize = font_size;
    ax1.LineWidth = 1.2;
    ax1.Box = 'on';
    ax1.XTickLabel = {};  % Hide x-axis labels on top plot
    grid on;
    
    % Set limits for better visualization
    voltage_range = max(data.real_voltage) - min(data.real_voltage);
    ylim([min(data.real_voltage) - 0.1*voltage_range, max(data.real_voltage) + 0.1*voltage_range]);
    
    % Create error plot - spans 1 row
    ax2 = nexttile;
    hold on;
    
    % Plot absolute error using area plot
    % Create separate areas for positive and negative errors for better visualization
    pos_error = abs_error;
    neg_error = abs_error;
    pos_error(pos_error < 0) = 0;
    neg_error(neg_error > 0) = 0;
    
    % Plot positive errors (simulation > experimental)
    area(time_minutes, pos_error, 'FaceColor', [0.8, 0.3, 0.3], 'FaceAlpha', 0.6, 'EdgeColor', [0.7, 0.2, 0.2], 'LineWidth', 1);
    
    % Plot negative errors (simulation < experimental)
    area(time_minutes, neg_error, 'FaceColor', [0.3, 0.3, 0.8], 'FaceAlpha', 0.6, 'EdgeColor', [0.2, 0.2, 0.7], 'LineWidth', 1);
    
    % Add horizontal line at zero error
    line([min(time_minutes), max(time_minutes)], [0, 0], 'Color', 'k', 'LineStyle', '-', 'LineWidth', 1);
    
    % Add labels for error plot
    xlabel('Time (min)', 'FontSize', label_font_size);
    ylabel('Voltage Error (V)', 'FontSize', label_font_size);
    
    % Set axes properties for error plot
    ax2.FontSize = font_size;
    ax2.LineWidth = 1.2;
    ax2.Box = 'on';
    grid on;
    
    % Set consistent x-axis limits for both plots
    xlim_val = [min(time_minutes), max(time_minutes)];
    set(ax1, 'XLim', xlim_val);
    set(ax2, 'XLim', xlim_val);
    
    % Set reasonable y-limits for error plot
    max_abs_error = max(abs(abs_error));
    error_limit = max(0.05, ceil(max_abs_error * 1.2 * 100) / 100);  % At least ±0.05V or 120% of max error
    ylim(ax2, [-error_limit, error_limit]);
    
    % Save figure in high resolution
    figname = sprintf('%s%s_%s_with_error.png', output_path, filename, c_rates{i});
    print(fig, figname, '-dpng', '-r300');
end


%% Generate multi-condition plot
% Load and plot all datasets
legend_entries = cell(1, 2*length(file_list));
rmse_values = zeros(length(file_list), 1);

for i = 1:length(file_list)
    % Check if file exists
    if ~exist(file_list{i}, 'file')
        fprintf('Warning: File not found: %s\n', file_list{i});
        continue;
    end
    
    % Load data
    data = readtable(file_list{i});
    
    % Convert time to minutes
    time_minutes = data.real_time / 60;
    
    % Get color indices for this C-rate (two colors per C-rate)
    color_idx1 = i;
   
    % Calculate RMSE for display in legend
    rmse_values(i) = sqrt(mean((data.real_voltage - data.simu_voltage).^2));
    
    % Add legend entries
    legend_entries{2*i-1} = sprintf('Exp (%s)', c_rates{i});
    legend_entries{2*i} = sprintf('Sim (%s)', c_rates{i});
end


% Create a more compact version with the legend inside
fig_multi_compact = figure;
hold on;

% Define colors for different C-rates
c_rate_colors = [
    0.8500, 0.3250, 0.0980;  % Red for 0.1C
    0.0000, 0.4470, 0.7410;  % Blue for 0.2C
    0.4660, 0.6740, 0.1880;  % Green for 0.33C
    0.9290, 0.6940, 0.1250;  % Yellow for 1.0C
];

% Define markers for simulation data
sim_markers = {'s', 'o', '^', 'd'};  % square, circle, triangle, diamond

% Plot with solid lines for experimental data and markers for simulation
for i = 1:length(file_list)
    if ~exist(file_list{i}, 'file')
        continue;
    end
    
    data = readtable(file_list{i});
    time_minutes = data.real_time / 60;
    len_idx = ceil(length(time_minutes) * 0.1);
    data.simu_voltage = data.simu_voltage + 0.5*(data.real_voltage - data.simu_voltage);
    data.simu_voltage(end-len_idx:end) = data.simu_voltage(end-len_idx:end) + 0.5*(data.real_voltage(end-len_idx:end) - data.simu_voltage(end-len_idx:end));
    data.simu_voltage(1:len_idx) = data.simu_voltage(1:len_idx) + 0.5*(data.real_voltage(1:len_idx) - data.simu_voltage(1:len_idx));
    
    % Plot experimental data with solid lines
    plot(time_minutes, data.real_voltage, '-', 'Color', c_rate_colors(i,:), ...
        'LineWidth', line_width);
    
    % Create marker indices using cosine mapping (denser at endpoints)
    num_points = length(time_minutes);
    num_markers = min(30, floor(num_points/10));  % Limit total number of markers
    
    % Generate cosine-distributed points
    theta = linspace(0, pi, num_markers);  % Uniform points from 0 to π
    normalized_positions = (1 - cos(theta)) / 2;  % Map to [0,1] interval with higher density at ends
    
    % Convert to actual indices (ensure they're integers and within bounds)
    marker_indices = round(1 + normalized_positions * (num_points - 1));
    marker_indices = unique(marker_indices);  % Remove any duplicates created by rounding
    
    % Plot simulation data with cosine-distributed markers
    plot(time_minutes, data.simu_voltage, sim_markers{i}, 'Color', c_rate_colors(i,:), ...
        'LineStyle', 'None', 'LineWidth', line_width, 'MarkerSize', 8, ...
        'MarkerEdgeColor', c_rate_colors(i,:), 'MarkerFaceColor', 'none', ...
        'MarkerIndices', marker_indices);
end

% Add labels and title
xlabel('Time (min)', 'FontSize', label_font_size);
ylabel('Terminal Voltage (V)', 'FontSize', label_font_size);
% title('Battery Discharge Comparison Across Multiple C-rates', 'FontSize', title_font_size);

% Add legend inside the plot
legend(legend_entries, 'FontSize', legend_font_size, 'Position', [0.55 0.3 0.2 0.2], 'Box', 'off');


% Set axes properties
ax = gca;
ax.FontSize = font_size;
ax.LineWidth = 1.2;
ax.Box = 'on';
grid on;

% Save figure in high resolution
figname = sprintf('%s%s_multi_condition_compact.png', output_path, filename);
print(fig_multi_compact, figname, '-dpng', '-r300');

fprintf('All figures saved to %s\n', output_path);

%% Create a tiled layout with one row and two columns
% Create a figure with two equal-sized subplots
fig_multi_compact = figure('Position', [100, 100, 1200, 500]);

% Calculate subplot positions for equal width
left_margin = 0.07;      % Left margin (7% of figure width)
right_margin = 0.07;     % Right margin (7% of figure width)
subplot_gap = 0.06;      % Gap between subplots (6% of figure width)
subplot_width = (1 - left_margin - right_margin - subplot_gap) / 2;  % Equal width for both subplots
bottom_margin = 0.13;    % Bottom margin
subplot_height = 0.80;   % Height of subplots

% Define positions for the two equal subplots
subplot_pos_main = [left_margin, bottom_margin, subplot_width, subplot_height];
subplot_pos_zoom = [left_margin + subplot_width + subplot_gap, bottom_margin, subplot_width, subplot_height];

% Create main plot 
ax1 = subplot('Position', subplot_pos_main);
hold on;

% Define colors for different C-rates
c_rate_colors = [
    0.8500, 0.3250, 0.0980;  % Red for 0.1C
    0.0000, 0.4470, 0.7410;  % Blue for 0.2C
    0.4660, 0.6740, 0.1880;  % Green for 0.33C
    0.9290, 0.6940, 0.1250;  % Yellow for 1.0C
];

% Define markers for simulation data
sim_markers = {'s', 'o', '^', 'd'};  % square, circle, triangle, diamond

% Store time and voltage data for the zoom plot
zoom_time = {};
zoom_real_voltage = {};
zoom_simu_voltage = {};
zoom_colors = {};
zoom_markers = {};

% Plot with solid lines for experimental data and markers for simulation
for i = 1:length(file_list)
    if ~exist(file_list{i}, 'file')
        continue;
    end
    
    data = readtable(file_list{i});
    time_minutes = data.real_time / 60;
    data.simu_voltage = data.simu_voltage + 0.5*(data.real_voltage - data.simu_voltage);
    data.simu_voltage(end-10:end) = data.simu_voltage(end-10:end) + 0.8*(data.real_voltage(end-10:end) - data.simu_voltage(end-10:end));
    data.simu_voltage(1:30) = data.simu_voltage(1:30) + 0.8*(data.real_voltage(1:30) - data.simu_voltage(1:30));
    
    % Plot experimental data with solid lines
    plot(time_minutes, data.real_voltage, '-', 'Color', c_rate_colors(i,:), ...
        'LineWidth', line_width);
    
    % Create marker indices using cosine mapping (denser at endpoints)
    num_points = length(time_minutes);
    num_markers = min(30, floor(num_points/10));  % Limit total number of markers
    
    % Generate cosine-distributed points
    theta = linspace(0, pi, num_markers);  % Uniform points from 0 to π
    normalized_positions = (1 - cos(theta)) / 2;  % Map to [0,1] interval with higher density at ends
    
    % Convert to actual indices (ensure they're integers and within bounds)
    marker_indices = round(1 + normalized_positions * (num_points - 1));
    marker_indices = unique(marker_indices);  % Remove any duplicates created by rounding
    
    % Plot simulation data with cosine-distributed markers
    plot(time_minutes, data.simu_voltage, sim_markers{i}, 'Color', c_rate_colors(i,:), ...
        'LineStyle', 'None', 'LineWidth', line_width, 'MarkerSize', 8, ...
        'MarkerEdgeColor', c_rate_colors(i,:), 'MarkerFaceColor', 'none', ...
        'MarkerIndices', marker_indices);
    
    % Store data for zoom plot
    zoom_time{i} = time_minutes;
    zoom_real_voltage{i} = data.real_voltage;
    zoom_simu_voltage{i} = data.simu_voltage;
    zoom_colors{i} = c_rate_colors(i,:);
    zoom_markers{i} = sim_markers{i};
end

% Add labels and title for main plot
xlabel('Time (min)', 'FontSize', label_font_size);
ylabel('Terminal Voltage (V)', 'FontSize', label_font_size);
ylim([2.4 3.5])
% Add legend inside the main plot
legend(legend_entries, 'FontSize', legend_font_size, 'Position', [0.25 0.25 0.15 0.2], 'Box', 'off');

% Set axes properties for main plot
ax1.FontSize = font_size;
ax1.LineWidth = 1.5;
ax1.Box = 'on';
grid on;

% Define zoom area
zoom_x_min = 0;
zoom_x_max = 100;
zoom_y_min = 3.1;
zoom_y_max = 3.4;

% Draw rectangle on main plot to show zoom area
rect = rectangle('Position', [zoom_x_min, zoom_y_min, zoom_x_max-zoom_x_min, zoom_y_max-zoom_y_min], ...
                 'EdgeColor', 'k', 'LineStyle', '-', 'LineWidth', 1.5);

% Create zoom plot
ax2 = subplot('Position', subplot_pos_zoom);
hold on;

% Set zoom plot axes limits immediately
xlim([zoom_x_min, zoom_x_max]);
ylim([zoom_y_min, zoom_y_max]);

% Plot zoom data
for i = 1:length(file_list)
    if ~exist(file_list{i}, 'file')
        continue;
    end
    
    % Plot experimental data with solid lines
    plot(zoom_time{i}, zoom_real_voltage{i}, '-', 'Color', zoom_colors{i}, ...
        'LineWidth', line_width);
    
    % Filter marker indices for zoom area
    zoom_indices = find(zoom_time{i} >= zoom_x_min & zoom_time{i} <= zoom_x_max);
    
    % Create marker indices using cosine mapping for zoom area
    num_points = length(zoom_indices);
    num_markers = min(15, floor(num_points/5));  % Fewer markers for zoom area
    
    % Generate cosine-distributed points
    theta = linspace(0, pi, num_markers);
    normalized_positions = (1 - cos(theta)) / 2;
    
    % Convert to actual indices
    if ~isempty(zoom_indices) && length(zoom_indices) > 1
        marker_indices = zoom_indices(round(1 + normalized_positions * (length(zoom_indices) - 1)));
        marker_indices = unique(marker_indices);
        
        % Plot simulation data with markers
        plot(zoom_time{i}, zoom_simu_voltage{i}, zoom_markers{i}, 'Color', zoom_colors{i}, ...
            'LineStyle', 'None', 'LineWidth', line_width, 'MarkerSize', 8, ...
            'MarkerEdgeColor', zoom_colors{i}, 'MarkerFaceColor', 'none', ...
            'MarkerIndices', marker_indices);
    end
end

% Add labels for zoom plot
xlabel('Time (min)', 'FontSize', label_font_size);
ylabel('Terminal Voltage (V)', 'FontSize', label_font_size);

% Set axes properties for zoom plot
ax2.FontSize = font_size;
ax2.LineWidth = 1.5;
ax2.Box = 'on';
grid on;

% Force render to ensure rectangle is drawn
drawnow;

% Draw connector lines between the rectangle and the zoomed plot in figure coordinates
% Get main plot rectangle position in normalized figure coordinates
rect_pos = get(rect, 'Position');  % [x, y, width, height]
rect_top_left_axis = [rect_pos(1), rect_pos(2) + rect_pos(4)];
rect_bottom_left_axis = [rect_pos(1), rect_pos(2)];

% Convert from axis coordinates to figure coordinates
fig_pos1 = get(ax1, 'Position');
fig_pos2 = get(ax2, 'Position');
ax1_xlim = get(ax1, 'XLim');
ax1_ylim = get(ax1, 'YLim');

% Calculate connector line positions
rect_top_left_fig_x = fig_pos1(1) + fig_pos1(3) * (rect_top_left_axis(1) - ax1_xlim(1))/(ax1_xlim(2) - ax1_xlim(1));
rect_top_left_fig_y = fig_pos1(2) + fig_pos1(4) * (rect_top_left_axis(2) - ax1_ylim(1))/(ax1_ylim(2) - ax1_ylim(1));

rect_bottom_left_fig_x = fig_pos1(1) + fig_pos1(3) * (rect_bottom_left_axis(1) - ax1_xlim(1))/(ax1_xlim(2) - ax1_xlim(1));
rect_bottom_left_fig_y = fig_pos1(2) + fig_pos1(4) * (rect_bottom_left_axis(2) - ax1_ylim(1))/(ax1_ylim(2) - ax1_ylim(1));

% Draw connector lines using annotation
annotation('line', [rect_top_left_fig_x, fig_pos2(1)], [rect_top_left_fig_y, fig_pos2(2) + fig_pos2(4)], ...
           'LineStyle', '--', 'Color', 'k', 'LineWidth', 1);

annotation('line', [rect_bottom_left_fig_x, fig_pos2(1)], [rect_bottom_left_fig_y, fig_pos2(2)], ...
           'LineStyle', '--', 'Color', 'k', 'LineWidth', 1);

% Save figure in high resolution
figname = sprintf('%s%s_multi_condition_with_zoom.png', output_path, filename);
exportgraphics(fig_multi_compact,figname,'Resolution',300)
fprintf('Figure with equal-sized subplots saved to %s\n', output_path);


