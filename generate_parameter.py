import pybamm
import csv
# import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import os
import shutil
import casadi
import pandas as pd
import pickle
from scipy.interpolate import interp1d

model_list = ["Ai2020", "Chen2020", "Ecker2015", "Marquis2019", "Mohtat2020", "NCA_Kim2011", "Prada2013", "Ramadass2004", "Xu2019"]
# for model_name in model_list:
#     parameter_values = pybamm.ParameterValues(model_name)
#     model = pybamm.lithium_ion.DFN()
#     write = True
#     if write:
#         with open(rf'./DFN_parameter_sets/{model_name}_variable.txt', 'w') as file:
#             file.write("DFN model variables:\n")  # 写入标题
#             for v in model.variables.keys():
#                 file.write("\t- " + str(v) + "\n")  # 写入每个变量名称，每个名称占一行

#         with open(rf'./DFN_parameter_sets/{model_name}_parameter.txt', 'w') as file:
#             file.write("DFN model parameters:\n")  # 写入标题
#             for key, value in parameter_values.items():
#                 file.write("\t- " + f"{key}: {value}\n")

for model_name in model_list:
    parameter_values = pybamm.ParameterValues(model_name)
    model = pybamm.lithium_ion.DFN()
    write = True
    if write:
        # 写入变量到CSV文件
        with open(f'./parameter_sets/DFN_parameter_sets/{model_name}_variables.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["DFN model variables:"])  # 写入标题
            for v in model.variables.keys():
                writer.writerow([v])  # 写入每个变量名称

        # 写入参数到CSV文件
        with open(f'./parameter_sets/DFN_parameter_sets/{model_name}_parameters.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["DFN model parameters:", model_name])  # 写入标题
            for key, value in parameter_values.items():
                writer.writerow([key, value])  # 写入每个参数及其值

# model = pybamm.lithium_ion.DFN()
# # 准备要写入的变量和参数
# parameter_values = pybamm.ParameterValues("Ai2020")
# all_parameters = {var: [] for var in parameter_values.keys()}

# for model_name in model_list:
#     temp_parameter_values = pybamm.ParameterValues(model_name)
#     # 填充每个变量对应的参数值
#     for key in all_parameters.keys():
#         if key in temp_parameter_values.keys():
#             all_parameters[key].append(temp_parameter_values[key])
#         else:
#             all_parameters[key].append('')  # 如果某个模型没有该变量的参数，则添加空字符串

# # 写入CSV文件
# csv_file_path = './DFN_parameter_sets/all_parameters.csv'
# with open(csv_file_path, 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)

#     # 写入标题行，第一列是变量名，后面是模型名称
#     header = ['Variable'] + model_list
#     csvwriter.writerow(header)

#     # 写入数据行
#     for var in all_parameters.keys():
#         row = [var] + all_parameters[var]
#         csvwriter.writerow(row)

# print(f"Data has been written to {csv_file_path}")
