%% Electrochemical Parameter Identification Results Comparison
clear;clc;close all;
%% 1. Generate RMSE Cumulative Minimum Data
rng(42); % For reproducibility

iterations = 1:500;

% RA-SGLO Data (Battery 81#)
rmse_ra_sglo = ones(1, 500) * 80; % Initial RMSE
% Rapid drop then convergence
change_points_ra_sglo = sort(randperm(199, 30)); % Random points of decrease
for i = 1:length(change_points_ra_sglo)
    if change_points_ra_sglo(i) < 200
        start_idx = change_points_ra_sglo(i);
        decrease_amount = (rmse_ra_sglo(start_idx-1) - 10) * (0.1 + 0.2*rand()); % decrease more initially
        next_val = max(10, rmse_ra_sglo(start_idx-1) - decrease_amount);
        rmse_ra_sglo(start_idx:199) = next_val;
    end
end
rmse_ra_sglo(200:500) = 10 + rand()*0.5; % Converged value around 10mV
% Ensure cumulative minimum
for i = 2:500
    rmse_ra_sglo(i) = min(rmse_ra_sglo(i-1), rmse_ra_sglo(i));
    if i > 200 && rmse_ra_sglo(i) < 10 % ensure it does not dip below 10 after 200
        rmse_ra_sglo(i) = 10 + rand()*0.1;
    end
    if rmse_ra_sglo(i) < 10
         rmse_ra_sglo(i) = 10 + rand()*0.1; % final floor
    end
end
rmse_ra_sglo(1) = 80; % Ensure start is 80
for i = 200:500
    rmse_ra_sglo(i) = min(rmse_ra_sglo(i), rmse_ra_sglo(199));
    if rmse_ra_sglo(i) < 10
        rmse_ra_sglo(i) = 10;
    end
end
% Fixed line for RA-SGLO RMSE smoothing
r_start_ra = rand()*2;
r_end_ra = rand()*2+4;
sub_vector_ra = linspace(r_start_ra, r_end_ra, 5);
rmse_ra_sglo(200:end) = mean(rmse_ra_sglo(195:199) - sub_vector_ra); % smooth out convergence to 10
if rmse_ra_sglo(200) < 10
    rmse_ra_sglo(200:end) = 10 + rand(1, length(200:500))*0.1;
end
for i = 2:500 % Final pass for cumulative minimum
    rmse_ra_sglo(i) = min(rmse_ra_sglo(i-1), rmse_ra_sglo(i));
end


% BO Data (Battery 81#)
rmse_bo = ones(1, 500) * 80;
% Slower drop, converges around 400 iterations to ~30mV
change_points_bo = sort(randperm(399, 50));
for i = 1:length(change_points_bo)
    if change_points_bo(i) < 400
        start_idx = change_points_bo(i);
        decrease_amount = (rmse_bo(start_idx-1) - 30) * (0.05 + 0.1*rand());
        next_val = max(30, rmse_bo(start_idx-1) - decrease_amount);
        rmse_bo(start_idx:399) = next_val;
    end
end
rmse_bo(400:500) = 30 + rand()*1.5; % Converged value around 30mV
% Ensure cumulative minimum
for i = 2:500
    rmse_bo(i) = min(rmse_bo(i-1), rmse_bo(i));
     if i > 400 && rmse_bo(i) < 30
        rmse_bo(i) = 30 + rand()*0.1;
    end
    if rmse_bo(i) < 30
         rmse_bo(i) = 30 + rand()*0.1; % final floor
    end
end
rmse_bo(1) = 80;
for i = 400:500
    rmse_bo(i) = min(rmse_bo(i), rmse_bo(399));
     if rmse_bo(i) < 30
        rmse_bo(i) = 30;
    end
end
% Fixed line for BO RMSE smoothing
r_start_bo = rand()*2;
r_end_bo = rand()*2+4;
sub_vector_bo = linspace(r_start_bo, r_end_bo, 5);
rmse_bo(400:end) = mean(rmse_bo(395:399) - sub_vector_bo); % smooth out convergence to 30
if rmse_bo(400) < 30
    rmse_bo(400:end) = 30 + rand(1, length(400:500))*0.1;
end
for i = 2:500 % Final pass for cumulative minimum
    rmse_bo(i) = min(rmse_bo(i-1), rmse_bo(i));
end


% GA Data (Battery 81#) - Slightly different from BO
rmse_ga = ones(1, 500) * 80;
% Slower drop, also converges around 400 iterations to ~30-35mV
change_points_ga = sort(randperm(399, 45)); % Slightly fewer changes than BO
for i = 1:length(change_points_ga)
    if change_points_ga(i) < 400
        start_idx = change_points_ga(i);
        decrease_amount = (rmse_ga(start_idx-1) - 32) * (0.04 + 0.1*rand()); % Slightly higher target
        next_val = max(32, rmse_ga(start_idx-1) - decrease_amount);
        rmse_ga(start_idx:399) = next_val;
    end
end
rmse_ga(400:500) = 32 + rand()*1.0; % Converged value around 32mV
% Ensure cumulative minimum
for i = 2:500
    rmse_ga(i) = min(rmse_ga(i-1), rmse_ga(i));
    if i > 400 && rmse_ga(i) < 32
        rmse_ga(i) = 32 + rand()*0.1;
    end
     if rmse_ga(i) < 32
         rmse_ga(i) = 32 + rand()*0.1; % final floor
    end
end
rmse_ga(1) = 80;
for i = 400:500
    rmse_ga(i) = min(rmse_ga(i), rmse_ga(399));
    if rmse_ga(i) < 32
        rmse_ga(i) = 32;
    end
end
% Fixed line for GA RMSE smoothing
r_start_ga = rand()*2;
r_end_ga = rand()*2+4;
sub_vector_ga = linspace(r_start_ga, r_end_ga, 5);
rmse_ga(400:end) = mean(rmse_ga(395:399) - sub_vector_ga); % smooth out convergence to 32
if rmse_ga(400) < 32
    rmse_ga(400:end) = 32 + rand(1, length(400:500))*0.1;
end
for i = 2:500 % Final pass for cumulative minimum
    rmse_ga(i) = min(rmse_ga(i-1), rmse_ga(i));
end

% Refine convergence to be exactly at the target after the convergence point
rmse_ra_sglo(find(iterations>=200,1):end) = min(rmse_ra_sglo(find(iterations>=200,1)-1),10 + abs(randn(1, length(find(iterations>=200,1):length(iterations)))*0.05));
rmse_ra_sglo(find(iterations>=200,1):end) = cummin(rmse_ra_sglo(find(iterations>=200,1):end)); % Ensure still cumulative min
idx_ra_sglo_conv = find(iterations>=200,1);
rmse_ra_sglo(idx_ra_sglo_conv:end) = rmse_ra_sglo(idx_ra_sglo_conv);
min_val_ra_sglo = 10 + rand()*0.2;
rmse_ra_sglo(idx_ra_sglo_conv:end) = min_val_ra_sglo;
for k=idx_ra_sglo_conv-1:-1:1 % make sure it is decreasing towards the min_val
    if rmse_ra_sglo(k) < min_val_ra_sglo
       rmse_ra_sglo(k) = min_val_ra_sglo + (rmse_ra_sglo(k-1)-min_val_ra_sglo)*rand()*0.5 + (k/idx_ra_sglo_conv)^2*(80-min_val_ra_sglo)*0.1;
    end
    if k>1 && rmse_ra_sglo(k-1) < rmse_ra_sglo(k)
        rmse_ra_sglo(k-1) = rmse_ra_sglo(k) + rand()*(80-rmse_ra_sglo(k))/(k);
    end
end
rmse_ra_sglo(1)=80;
for k=2:500 rmse_ra_sglo(k) = min(rmse_ra_sglo(k-1), rmse_ra_sglo(k)); end


rmse_bo(find(iterations>=400,1):end) = min(rmse_bo(find(iterations>=400,1)-1),30 + abs(randn(1, length(find(iterations>=400,1):length(iterations)))*0.1));
rmse_bo(find(iterations>=400,1):end) = cummin(rmse_bo(find(iterations>=400,1):end));
idx_bo_conv = find(iterations>=400,1);
rmse_bo(idx_bo_conv:end) = rmse_bo(idx_bo_conv);
min_val_bo = 30 + rand()*0.5;
rmse_bo(idx_bo_conv:end) = min_val_bo;
for k=idx_bo_conv-1:-1:1
    if rmse_bo(k) < min_val_bo
        rmse_bo(k) = min_val_bo + (rmse_bo(k-1)-min_val_bo)*rand()*0.5 + (k/idx_bo_conv)^2*(80-min_val_bo)*0.1;
    end
     if k>1 && rmse_bo(k-1) < rmse_bo(k)
        rmse_bo(k-1) = rmse_bo(k) + rand()*(80-rmse_bo(k))/(k);
    end
end
rmse_bo(1)=80;
for k=2:500 rmse_bo(k) = min(rmse_bo(k-1), rmse_bo(k)); end


rmse_ga(find(iterations>=400,1):end) = min(rmse_ga(find(iterations>=400,1)-1),32 + abs(randn(1, length(find(iterations>=400,1):length(iterations)))*0.1));
rmse_ga(find(iterations>=400,1):end) = cummin(rmse_ga(find(iterations>=400,1):end));
idx_ga_conv = find(iterations>=400,1);
rmse_ga(idx_ga_conv:end) = rmse_ga(idx_ga_conv);
min_val_ga = 32 + rand()*0.5;
rmse_ga(idx_ga_conv:end) = min_val_ga;
for k=idx_ga_conv-1:-1:1
    if rmse_ga(k) < min_val_ga
       rmse_ga(k) = min_val_ga + (rmse_ga(k-1)-min_val_ga)*rand()*0.5 + (k/idx_ga_conv)^2*(80-min_val_ga)*0.1;
    end
    if k>1 && rmse_ga(k-1) < rmse_ga(k)
        rmse_ga(k-1) = rmse_ga(k) + rand()*(80-rmse_ga(k))/(k);
    end
end
rmse_ga(1)=80;
for k=2:500 rmse_ga(k) = min(rmse_ga(k-1), rmse_ga(k)); end


% Ensure GA and BO are different
rmse_ga = rmse_ga + randn(1,500)*0.1; % Add small noise to differentiate
rmse_ga(1)=80;
for k=2:500 rmse_ga(k) = min(rmse_ga(k-1), rmse_ga(k)); end
rmse_ga(idx_ga_conv:end) = min(rmse_ga(idx_ga_conv:end), min_val_ga + 0.2); % Ensure GA is slightly worse or different than BO after convergence


%% 2. Generate Iteration Count and Time Data
% Battery 81#
iter_conv_81 = [find(rmse_ra_sglo <= (10 + 0.5), 1), ...
                find(rmse_bo <= (30 + 0.5), 1), ...
                find(rmse_ga <= (32 + 0.5), 1)];
time_taken_81 = [30 + rand()*2, 60 + rand()*5, 60 + rand()*5 + 2]; % RA-SGLO, BO, GA (in minutes)

% Battery 82# (slight variations)
iter_conv_82 = [iter_conv_81(1) + round(randn()*10), ... % RA-SGLO around 200
                iter_conv_81(2) + round(randn()*15), ... % BO around 400
                iter_conv_81(3) + round(randn()*15)];   % GA around 400
iter_conv_82 = max(iter_conv_82, 50); % Ensure positive and reasonable
time_taken_82 = [time_taken_81(1) * (1 + (rand()-0.5)*0.1), ... % +-10% variation
                 time_taken_81(2) * (1 + (rand()-0.5)*0.1), ...
                 time_taken_81(3) * (1 + (rand()-0.5)*0.1)];

% Battery 83# (slight variations)
iter_conv_83 = [iter_conv_81(1) + round(randn()*12), ... % RA-SGLO
                iter_conv_81(2) + round(randn()*12), ... % BO
                iter_conv_81(3) + round(randn()*18)];   % GA
iter_conv_83 = max(iter_conv_83, 50); % Ensure positive and reasonable
time_taken_83 = [time_taken_81(1) * (1 + (rand()-0.5)*0.12), ... % +-12% variation
                 time_taken_81(2) * (1 + (rand()-0.5)*0.12), ...
                 time_taken_81(3) * (1 + (rand()-0.5)*0.12)];


%% 3. Plotting
fig = figure('Position', [100, 100, 1200, 450]); % Wide figure for two subplots

% Define Colors and Line Styles (aesthetic scientific colors)
colors = [  0.9290, 0.6940, 0.1250; % RA-SGLO - Gold/Yellow
            0, 0.4470, 0.7410;  % BO - Blue
            0.8500, 0.3250, 0.0980]; % GA - Red/Orange
line_styles = {'-', '--', ':'};

% ----- Subplot 1: RMSE Cumulative Minimum Data -----
subplot(1, 2, 1);
hold on;
plot(iterations, rmse_ra_sglo, 'LineWidth', 2, 'Color', colors(1,:), 'LineStyle', line_styles{1});
plot(iterations, rmse_bo, 'LineWidth', 2, 'Color', colors(2,:), 'LineStyle', line_styles{2});
plot(iterations, rmse_ga, 'LineWidth', 2, 'Color', colors(3,:), 'LineStyle', line_styles{3});
hold off;

% Labels and Title
xlabel('Iteration');
ylabel('Best RMSE (Multi C-rates) (mV)');
title('RMSE Convergence Comparison (Battery 81#)');
grid on;
box on;

% Axes properties
ax1 = gca;
ax1.LineWidth = 1.5;
ax1.FontSize = 12;
ax1.XLim = [0 500];
ax1.YLim = [0 85]; % Adjusted to fit data range

% Legend
lgd1 = legend('RA-SGLO', 'BO', 'GA', 'Location', 'NorthEast');
lgd1.Box = 'off'; % No border for the legend

% ----- Subplot 2: Bar Chart (Iterations) and Line Plot (Time) -----
subplot(1, 2, 2);

% Data for bar chart
iterations_data = [iter_conv_81; iter_conv_82; iter_conv_83]'; % Rows: RA-SGLO, BO, GA; Cols: Bat81, Bat82, Bat83
% Data for line plot (time)
time_data = [time_taken_81; time_taken_82; time_taken_83]';

battery_labels = {'Battery 81#', 'Battery 82#', 'Battery 83#'};
method_labels = {'RA-SGLO', 'BO', 'GA'};

% Bar chart for iterations
yyaxis left; % Activate left y-axis
bar_plot = bar(iterations_data, 'grouped');
ylabel('Iterations to Convergence');
ax2_left = gca;
ax2_left.YColor = 'k'; % Black color for left y-axis labels
ax2_left.LineWidth = 1.5;
ax2_left.FontSize = 12;

% Apply colors to bars and set border
for i = 1:length(bar_plot)
    bar_plot(i).FaceColor = colors(i,:);
    bar_plot(i).EdgeColor = 'k';
    bar_plot(i).LineWidth = 1.5;
end
set(ax2_left, 'XTickLabel', battery_labels);


% Line plot for time on right y-axis
yyaxis right; % Activate right y-axis
hold on;
plot(1:length(battery_labels), time_data(1,:), 'o-', 'LineWidth', 2, 'Color', colors(1,:), 'MarkerFaceColor', colors(1,:), 'MarkerSize', 6, 'LineStyle', line_styles{1});
plot(1:length(battery_labels), time_data(2,:), 's--', 'LineWidth', 2, 'Color', colors(2,:), 'MarkerFaceColor', colors(2,:), 'MarkerSize', 6, 'LineStyle', line_styles{2});
plot(1:length(battery_labels), time_data(3,:), '^:', 'LineWidth', 2, 'Color', colors(3,:), 'MarkerFaceColor', colors(3,:), 'MarkerSize', 6, 'LineStyle', line_styles{3});
hold off;
ylabel('Time Taken (minutes)');
ax2_right = gca;
ax2_right.YColor = 'k'; % Black color for right y-axis labels
ax2_right.LineWidth = 1.5; % This will be overwritten by yyaxis if not set after
ax2_right.FontSize = 12;
ax2_right.XLim = [0.5, length(battery_labels) + 0.5]; % Match bar plot x-axis
% Ensure right y-axis limits are appropriate
max_time = max(time_data(:));
ax2_right.YLim = [0, ceil(max_time / 10) * 10 + 10];


% Title and Grid for Subplot 2
title('Convergence Iterations and Time Comparison');
grid on;
box on;

% Legend for Subplot 2 (might need manual adjustment for clarity with dual y-axis)
% Create dummy artists for the legend to represent both bar and line
hold(ax2_left, 'on'); % Hold left axis to add dummy plots for legend
h_dummy = zeros(3, 1);
for i=1:3
    h_dummy(i) = bar(NaN,NaN,'FaceColor',colors(i,:),'EdgeColor','k','LineWidth',1.5);
end
hold(ax2_left,'off');

lgd2 = legend(h_dummy, method_labels, 'Location', 'NorthWest');
lgd2.Box = 'off';
lgd2.FontSize = 10;
exportgraphics(fig,'draw_time/Time_iter_comparison.png','Resolution',300);

% Common settings for the whole figure
% sgtitle('Algorithm Performance Comparison for DFN Model Parameter Estimation', 'FontSize', 16, 'FontWeight', 'bold');

% Adjust layout to prevent overlapping titles/labels if necessary
% annotation('textbox', [0 0.9 1 0.1], ...
%     'String', 'Algorithm Performance Comparison for DFN Model Parameter Estimation', ...
%     'EdgeColor', 'none', ...
%     'HorizontalAlignment', 'center', ...
%     'FontSize', 16, 'FontWeight', 'bold');

%% End of script
