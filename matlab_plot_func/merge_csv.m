% CSV文件合并处理脚本

% 定义文件路径
inputDir = '../simulation_dc_output/simu_data/';
outputDir = '../simulation_dc_output/simu_data/';

% 定义输入文件名
files = {
    '01#-T25-DC-simulation.csv',
    '02#-T25-DC-simulation.csv',
    '03#-T25-DC-simulation.csv',
    '04#-T25-DC-simulation.csv'
};

% 定义输出文件名
outputFiles = {
    '01#-T25-DC-simulation-all.csv',
    '02#-T25-DC-simulation-all.csv'
};

% 读取CSV文件
data = cell(1, 4);
for i = 1:4
    filePath = fullfile(inputDir, files{i});
    data{i} = readtable(filePath);
    fprintf('已读取: %s\n', files{i});
end

% 修改Time列并合并
% 第一组：合并01#和02#
fprintf('正在处理第一组数据...\n');
data01 = data{1};
data02 = data{2};

% 获取01#的长度，用于调整02#的Time列
len01 = height(data01);

% 调整02#的Time列，从len01+1开始
data02.Time = ((len01+1):(len01+height(data02)))';

% 合并两个表
mergedData1 = [data01; data02];
fprintf('第一组数据合并完成，共%d行\n', height(mergedData1));

% 第二组：合并03#和04#
fprintf('正在处理第二组数据...\n');
data03 = data{3};
data04 = data{4};

% 获取03#的长度，用于调整04#的Time列
len03 = height(data03);

% 调整04#的Time列，从len03+1开始
data04.Time = ((len03+1):(len03+height(data04)))';

% 合并两个表
mergedData2 = [data03; data04];
fprintf('第二组数据合并完成，共%d行\n', height(mergedData2));

% 保存合并后的CSV文件
writetable(mergedData1, fullfile(outputDir, outputFiles{1}));
fprintf('已保存: %s\n', outputFiles{1});

writetable(mergedData2, fullfile(outputDir, outputFiles{2}));
fprintf('已保存: %s\n', outputFiles{2});

fprintf('所有文件处理完成!\n');
