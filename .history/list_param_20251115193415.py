# list_param.py
import os
import json
import numpy as np
import pandas as pd


def min_max_func(low, high, norm_value):
    """和原代码一致的归一化反变换: x = norm * (high - low) + low"""
    return norm_value * (high - low) + low


def read_first_solution(csv_path):
    """
    从 CSV 中读取第一行 Solution 列，并转换为 numpy 数组。
    形如: "[0.4197 0.3264 ... 0.2340]"
    """
    df = pd.read_csv(csv_path)
    solution_str = str(df.iloc[0]["Solution"])
    cleaned = solution_str.strip("[]")  # 去掉前后中括号
    numbers = cleaned.split()           # 按空格拆分
    values = list(map(float, numbers))
    solution = np.array(values, dtype=float)

    if solution.size != 26:
        raise ValueError(f"Expect 26 parameters, but got {solution.size}")

    return solution


def build_param_dict(norm_param):
    """
    按 goal_programming.py 里 pybamm_sim 的定义，将 26 维归一化参数反归一化，
    并生成 {参数名: 物理量} 的字典。
    """

    # 逐个反归一化，对应 goal_programming.py 中的 0~25
    electrode_height = min_max_func(0.6, 1, norm_param[0])
    electrode_width = min_max_func(25, 30, norm_param[1])
    negative_electrode_conductivity = min_max_func(14, 215, norm_param[2])
    positive_electrode_diffusivity = min_max_func(5.9e-18, 1e-14, norm_param[3])
    positive_particle_radius = min_max_func(1e-8, 1e-5, norm_param[4])
    initial_conc_pos = min_max_func(35.3766672, 31513, norm_param[5])
    initial_conc_neg = min_max_func(48.8682, 29866, norm_param[6])
    positive_electrode_conductivity = min_max_func(0.18, 100, norm_param[7])
    negative_particle_radius = min_max_func(0.0000005083, 0.0000137, norm_param[8])
    negative_electrode_thickness = min_max_func(0.000036, 0.0007, norm_param[9])
    total_heat_transfer_coeff = min_max_func(5, 35, norm_param[10])
    separator_density = min_max_func(397, 2470, norm_param[11])
    separator_thermal_conductivity = min_max_func(0.10672, 0.34, norm_param[12])
    positive_electrode_porosity = min_max_func(0.12728395, 0.4, norm_param[13])
    separator_specific_heat_capacity = min_max_func(700, 1978, norm_param[14])
    max_conc_pos = min_max_func(22806, 63104, norm_param[15])
    neg_electrode_brugg = min_max_func(1.5, 4, norm_param[16])
    pos_electrode_brugg = min_max_func(1.5, 4, norm_param[17])
    separator_porosity = min_max_func(0.39, 1, norm_param[18])
    neg_cc_thickness = min_max_func(0.00001, 0.000025, norm_param[19])
    pos_cc_thickness = min_max_func(0.00001, 0.000025, norm_param[20])
    positive_electrode_thickness = min_max_func(0.000042, 0.0001, norm_param[21])
    pos_active_material_vol_frac = min_max_func(0.28485556, 0.665, norm_param[22])
    neg_electrode_cp = min_max_func(700, 1437, norm_param[23])
    pos_electrode_thermal_cond = min_max_func(1.04, 2.1, norm_param[24])
    neg_active_material_vol_frac = min_max_func(0.372403, 0.75, norm_param[25])

    # 和 pybamm_sim 里的 param_dict 键完全对应
    param_dict = {
        "Electrode height [m]": electrode_height,
        "Electrode width [m]": electrode_width,
        "Negative electrode conductivity [S.m-1]": negative_electrode_conductivity,
        "Positive electrode diffusivity [m2.s-1]": positive_electrode_diffusivity,
        "Positive particle radius [m]": positive_particle_radius,
        "Initial concentration in positive electrode [mol.m-3]": initial_conc_pos,
        "Initial concentration in negative electrode [mol.m-3]": initial_conc_neg,
        "Positive electrode conductivity [S.m-1]": positive_electrode_conductivity,
        "Negative particle radius [m]": negative_particle_radius,
        "Negative electrode thickness [m]": negative_electrode_thickness,
        "Total heat transfer coefficient [W.m-2.K-1]": total_heat_transfer_coeff,
        "Separator density [kg.m-3]": separator_density,
        "Separator thermal conductivity [W.m-1.K-1]": separator_thermal_conductivity,
        "Positive electrode porosity": positive_electrode_porosity,
        "Separator specific heat capacity [J.kg-1.K-1]": separator_specific_heat_capacity,
        "Maximum concentration in positive electrode [mol.m-3]": max_conc_pos,
        "Negative electrode Bruggeman coefficient (electrolyte)": neg_electrode_brugg,
        "Positive electrode Bruggeman coefficient (electrolyte)": pos_electrode_brugg,
        "Separator porosity": separator_porosity,
        "Negative current collector thickness [m]": neg_cc_thickness,
        "Positive current collector thickness [m]": pos_cc_thickness,
        "Positive electrode thickness [m]": positive_electrode_thickness,
        "Positive electrode active material volume fraction": pos_active_material_vol_frac,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": neg_electrode_cp,
        "Positive electrode thermal conductivity [W.m-1.K-1]": pos_electrode_thermal_cond,
        "Negative electrode active material volume fraction": neg_active_material_vol_frac,
    }

    return param_dict


def main():
    # 1. 读取归一化解
    csv_path = os.path.join("solutions", "Bayes", "81#MO-Constraint-DFN-26.csv")
    norm_param = read_first_solution(csv_path)

    # 2. 反归一化并生成字典
    param_dict = build_param_dict(norm_param)

    # 3. 写入 solutions/param_dict.txt
    out_dir = "solutions"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "param_dict.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        # 写成 JSON 格式，方便后续直接读取
        json.dump(param_dict, f, indent=4)

    print(f"Saved parameter dictionary to: {out_path}")


if __name__ == "__main__":
    main()
