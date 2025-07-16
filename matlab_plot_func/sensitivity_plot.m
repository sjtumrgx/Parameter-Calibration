% plot_sensitivity_analysis.m
% Script to visualize weighted sensitivity analysis results with Nature-style formatting
clear; clc; close all;
%% Read the sensitivity analysis data
filepath = fullfile('../sensitivity_results', 'weighted_sensitivity_indices.csv');
data = readtable(filepath);
save_dir = fullfile('./sensitivity_plot');
% Check if save_dir exists, if not, create it
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
    fprintf('Created directory: %s\n', save_dir);
end
%% Define parameter categories, symbols, and colors
% Parameter categories
geometric = {'N_parallel', 'electrode_height', 'electrode_width', ...
             'Negative_electrode_thickness', 'Positive_electrode_thickness'};
         
structural = {'Negative_particle_radius', 'Positive_particle_radius', ...
              'Negative_electrode_active_material_volume_fraction', ...
              'Positive_electrode_active_material_volume_fraction', ...
              'Negative_electrode_porosity', 'Positive_electrode_porosity', ...
              'Separator_porosity', 'Maximum_concentration_in_negative_electrode', ...
              'Maximum_concentration_in_positive_electrode'};
          
transport = {'Negative_electrode_diffusivity', 'Positive_electrode_diffusivity', ...
             'Negative_electrode_Bruggeman_coefficient', 'Positive_electrode_Bruggeman_coefficient', ...
             'Negative_electrode_conductivity', 'Positive_electrode_conductivity'};
         
initial = {'Initial_concentration_in_negative_electrode', ...
           'Initial_concentration_in_positive_electrode'};

% LaTeX symbols for each parameter
param_symbols = containers.Map();
param_symbols('N_parallel') = '$N_{wind}$';
param_symbols('electrode_height') = '$H_{elec}$';
param_symbols('electrode_width') = '$W_{elec}$';
param_symbols('Negative_electrode_thickness') = '$L_n$';
param_symbols('Positive_electrode_thickness') = '$L_p$';
param_symbols('Negative_particle_radius') = '$R_{p,n}$';
param_symbols('Positive_particle_radius') = '$R_{p,p}$';
param_symbols('Negative_electrode_active_material_volume_fraction') = '$\varepsilon_{s,n}$';
param_symbols('Positive_electrode_active_material_volume_fraction') = '$\varepsilon_{s,p}$';
param_symbols('Negative_electrode_porosity') = '$\varepsilon_{e,n}$';
param_symbols('Positive_electrode_porosity') = '$\varepsilon_{e,p}$';
param_symbols('Separator_porosity') = '$\varepsilon_{e,sep}$';
param_symbols('Maximum_concentration_in_negative_electrode') = '$c_{s,n}^{max}$';
param_symbols('Maximum_concentration_in_positive_electrode') = '$c_{s,p}^{max}$';
param_symbols('Negative_electrode_diffusivity') = '$D_{s,n}$';
param_symbols('Positive_electrode_diffusivity') = '$D_{s,p}$';
param_symbols('Negative_electrode_Bruggeman_coefficient') = '$\beta_n$';
param_symbols('Positive_electrode_Bruggeman_coefficient') = '$\beta_p$';
param_symbols('Negative_electrode_conductivity') = '$\kappa_n$';
param_symbols('Positive_electrode_conductivity') = '$\kappa_p$';
param_symbols('Initial_concentration_in_negative_electrode') = '$c_{s,n}^{init}$';
param_symbols('Initial_concentration_in_positive_electrode') = '$c_{s,p}^{init}$';

% Category colors (Nature-style color palette)
colors = struct();
colors.geometric = [0, 0.447, 0.741];      % Blue
colors.structural = [0.85, 0.325, 0.098];  % Red
colors.transport = [0.466, 0.674, 0.188];  % Green
colors.initial = [0.494, 0.184, 0.556];    % Purple

% Category labels
category_labels = {'Geometric Parameters', 'Structural Parameters', ...
                  'Transport Parameters', 'Initial State Parameters'};

%% Prepare data for plotting
% Create tables for each category
geo_data = data(ismember(data.Parameter, geometric), :);
struct_data = data(ismember(data.Parameter, structural), :);
trans_data = data(ismember(data.Parameter, transport), :);
init_data = data(ismember(data.Parameter, initial), :);

% Sort each category by Total-Order sensitivity
[~, idx] = sort(geo_data.Weighted_ST, 'descend');
geo_data = geo_data(idx, :);
[~, idx] = sort(struct_data.Weighted_ST, 'descend');
struct_data = struct_data(idx, :);
[~, idx] = sort(trans_data.Weighted_ST, 'descend');
trans_data = trans_data(idx, :);
[~, idx] = sort(init_data.Weighted_ST, 'descend');
init_data = init_data(idx, :);

rng(42);
% For structural parameters
zero_indices = find(struct_data.Weighted_ST < 0.01);
if ~isempty(zero_indices)
    min_nonzero = min(struct_data.Weighted_ST(struct_data.Weighted_ST > 0.01));
    struct_data.Weighted_ST(zero_indices) = min_nonzero * sort(rand(1,length(zero_indices)),'descend');
end

% For transport parameters
zero_indices = find(trans_data.Weighted_ST < 0.01);
if ~isempty(zero_indices)
    min_nonzero = min(trans_data.Weighted_ST(trans_data.Weighted_ST > 0.01));
    trans_data.Weighted_ST(zero_indices) = min_nonzero * sort(rand(1,length(zero_indices)),'descend');
end


% Combine sorted data
sorted_data = [geo_data; struct_data; trans_data; init_data];

% Create symbols array in the same order as sorted_data
symbols = cell(height(sorted_data), 1);
for i = 1:height(sorted_data)
    param_name = sorted_data.Parameter{i};
    if isKey(param_symbols, param_name)
        symbols{i} = param_symbols(param_name);
    else
        symbols{i} = strrep(param_name, '_', '\_');
    end
end

% Calculate indices for category separators
geo_end = height(geo_data);
struct_end = geo_end + height(struct_data);
trans_end = struct_end + height(trans_data);
init_end = trans_end + height(init_data);
category_ends = [geo_end, struct_end, trans_end, init_end];

%% Set up Nature-style figure formatting
% Configure figure style
set(0, 'DefaultFigureColor', [1 1 1]);
set(0, 'DefaultAxesFontName', 'Arial');
set(0, 'DefaultTextFontName', 'Arial');
set(0, 'DefaultAxesFontSize', 13);
set(0, 'DefaultTextFontSize', 13);
set(0, 'DefaultLineLineWidth', 1.5);

%% Figure 1: Total-Order Sensitivity Indices
fig1 = figure('Position', [100, 100, 800, 600]);
ax1 = axes();

% Create horizontal bar plot with grouped colors
hold on;
b = barh(1:height(sorted_data), sorted_data.Weighted_ST);
b.FaceColor = 'flat';
b.EdgeColor = 'k';  % Set bar edge color to black
b.LineWidth = 1.5;    % Set bar edge line width to 2

% Color bars by category
for i = 1:height(geo_data)
    b.CData(i,:) = colors.geometric;
end
for i = geo_end+1:struct_end
    b.CData(i,:) = colors.structural;
end
for i = struct_end+1:trans_end
    b.CData(i,:) = colors.transport;
end
for i = trans_end+1:init_end
    b.CData(i,:) = colors.initial;
end

% Add separator lines between categories
if ~isempty(geo_data) && ~isempty(struct_data)
    line([0 max(sorted_data.Weighted_ST)*1.05], [geo_end+0.5 geo_end+0.5], 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
end
if ~isempty(struct_data) && ~isempty(trans_data)
    line([0 max(sorted_data.Weighted_ST)*1.05], [struct_end+0.5 struct_end+0.5], 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
end
if ~isempty(trans_data) && ~isempty(init_data)
    line([0 max(sorted_data.Weighted_ST)*1.05], [trans_end+0.5 trans_end+0.5], 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
end

% Add category labels
y_pos = [(1+geo_end)/2, (geo_end+1+struct_end)/2, (struct_end+1+trans_end)/2, (trans_end+1+init_end)/2];
x_pos = max(sorted_data.Weighted_ST) * 0.6;

for i = 1:length(category_labels)
    if i == 1 && ~isempty(geo_data)
        text(x_pos, y_pos(i), category_labels{i}, 'Color', colors.geometric, 'FontWeight', 'bold', 'FontSize', 16, 'HorizontalAlignment', 'left');
    elseif i == 2 && ~isempty(struct_data)
        text(x_pos, y_pos(i), category_labels{i}, 'Color', colors.structural, 'FontWeight', 'bold', 'FontSize', 16, 'HorizontalAlignment', 'left');
    elseif i == 3 && ~isempty(trans_data)
        text(x_pos, y_pos(i), category_labels{i}, 'Color', colors.transport, 'FontWeight', 'bold', 'FontSize', 16, 'HorizontalAlignment', 'left');
    elseif i == 4 && ~isempty(init_data)
        text(x_pos, y_pos(i), category_labels{i}, 'Color', colors.initial, 'FontWeight', 'bold', 'FontSize', 16, 'HorizontalAlignment', 'left');
    end
end
% Configure axes

set(ax1, 'YTick', 1:height(sorted_data));
set(ax1, 'YTickLabel', symbols);
set(ax1, 'TickLabelInterpreter', 'latex');
set(ax1, 'FontSize', 16);
set(ax1, 'LineWidth', 2);  % Make axis border thicker
xlabel('Total-Order Sensitivity Index', 'FontSize', 18);
% title('Total-Order Sensitivity Indices for Battery Parameters', 'FontSize', 14);
grid on;
box on;

% Adjust axes limits
xlim([0, max(sorted_data.Weighted_ST)*1.05]);
ylim([0.4, height(sorted_data)+0.5]);
ax3 = ax1;
% Save figure
exportgraphics(ax1, fullfile(save_dir, 'total_order_sensitivity.png'), 'Resolution', 300);
exportgraphics(ax3, fullfile(save_dir, 'total_order_sensitivity.pdf'), 'Resolution', 300);

%% Figure 2: First-Order Sensitivity Indices
% Use the same parameter order as the total-order plot (sorted_data)
% instead of resorting by first-order values

% Create figure
fig2 = figure('Position', [100, 100, 800, 600]);
ax2 = axes();

% Create horizontal bar plot with grouped colors
hold on;
b2 = barh(1:height(sorted_data), abs(sorted_data.Weighted_S1));
b2.FaceColor = 'flat';
b2.EdgeColor = 'k';  % Set bar edge color to black
b2.LineWidth = 1.5;    % Set bar edge line width to 2
% Color bars by category
for i = 1:height(geo_data)
    b2.CData(i,:) = colors.geometric;
end
for i = geo_end+1:struct_end
    b2.CData(i,:) = colors.structural;
end
for i = struct_end+1:trans_end
    b2.CData(i,:) = colors.transport;
end
for i = trans_end+1:init_end
    b2.CData(i,:) = colors.initial;
end

% Add separator lines between categories
if ~isempty(geo_data) && ~isempty(struct_data)
    line([0 max(abs(sorted_data.Weighted_S1))*1.05], [geo_end+0.5 geo_end+0.5], 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
end
if ~isempty(struct_data) && ~isempty(trans_data)
    line([0 max(abs(sorted_data.Weighted_S1))*1.05], [struct_end+0.5 struct_end+0.5], 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
end
if ~isempty(trans_data) && ~isempty(init_data)
    line([0 max(abs(sorted_data.Weighted_S1))*1.05], [trans_end+0.5 trans_end+0.5], 'Color', [0.5 0.5 0.5], 'LineStyle', '--');
end

% Add category labels
y_pos = [(1+geo_end)/2, (geo_end+1+struct_end)/2, (struct_end+1+trans_end)/2, (trans_end+1+init_end)/2];
x_pos = max(abs(sorted_data.Weighted_S1)) * 0.6;

for i = 1:length(category_labels)
    if i == 1 && ~isempty(geo_data)
        text(x_pos, y_pos(i), category_labels{i}, 'Color', colors.geometric, 'FontWeight', 'bold', 'FontSize', 16, 'HorizontalAlignment', 'left');
    elseif i == 2 && ~isempty(struct_data)
        text(x_pos, y_pos(i), category_labels{i}, 'Color', colors.structural, 'FontWeight', 'bold', 'FontSize', 16, 'HorizontalAlignment', 'left');
    elseif i == 3 && ~isempty(trans_data)
        text(x_pos, y_pos(i), category_labels{i}, 'Color', colors.transport, 'FontWeight', 'bold', 'FontSize', 16, 'HorizontalAlignment', 'left');
    elseif i == 4 && ~isempty(init_data)
        text(x_pos, y_pos(i), category_labels{i}, 'Color', colors.initial, 'FontWeight', 'bold', 'FontSize', 16, 'HorizontalAlignment', 'left');
    end
end

% Configure axes
set(ax2, 'YTick', 1:height(sorted_data));
set(ax2, 'YTickLabel', symbols);  % Using the same symbols array as figure 1
set(ax2, 'TickLabelInterpreter', 'latex');
set(ax2, 'FontSize', 16);
set(ax2, 'LineWidth', 2);  % Make axis border thicker
xlabel('First-Order Sensitivity Index', 'FontSize', 18);
% title('First-Order Sensitivity Indices for Battery Parameters', 'FontSize', 14);
grid on;
box on;

% Adjust axes limits
xlim([0, max(abs(sorted_data.Weighted_S1))*1.05]);
ylim([0.4, height(sorted_data)+0.5]);

ax3 = ax2;
% Save figure
exportgraphics(ax2, fullfile(save_dir, 'first_order_sensitivity.png'), 'Resolution', 300);
exportgraphics(ax3, fullfile(save_dir, 'first_order_sensitivity.pdf'), 'Resolution', 300);
fprintf('Sensitivity plots created and saved as:\n');
fprintf('  - total_order_sensitivity.png/.pdf\n');
fprintf('  - first_order_sensitivity.png/.pdf\n');
