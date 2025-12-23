import pybamm
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import pygad
import csv
import json
import argparse
from scipy.interpolate import interp1d
import multiprocessing
import math
import torch
import warnings
from joblib import Parallel, delayed
from dataclasses import dataclass
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine
from botorch.fit import fit_gpytorch_mll
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize


def read_file(file_name):
    data = pd.read_csv(file_name)
    time_max = data['time'].values[-1]
    voltage_max = data['V'].values[0]
    voltage_min = data['V'].values[-1] - 1
    capacity = data['Ah'].values[-1]
    return time_max, voltage_max, voltage_min, capacity


def min_max_func(low, high, norm_value):
    return norm_value * (high - low) + low


def plot_time_discharge(time_resampled, voltage_simulation_resampled, voltage_real_resampled, rmse_value, discharge_cur):
    fig, ax = plt.subplots()
    # Plotting
    ax.plot(time_resampled, voltage_simulation_resampled, linestyle='-', label='Simulation')
    ax.plot(time_resampled, voltage_real_resampled, linestyle='-', label='Experiment')

    plt.xlabel('Time [s]')
    plt.ylabel('Terminal Voltage [V]')
    plt.title(f"{file_name}-{discharge_cur}C-RMSE:{rmse_value:.4f} V")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(f"./simu_fig/{subdir_name}/{file_name}_{discharge_cur}C.png")
    plt.show()


def compute_time_discharge(sol, file_path, soc_range=(0.9, 1)):
    time_simulation = sol["Time [s]"].entries
    voltage_simulation = sol["Voltage [V]"].entries

    data = pd.read_csv(file_path)
    time_real = data['time'].values
    voltage_real = data['V'].values

    soc_str = data['SOC'].str.replace('%', '', regex=False)  # 去掉 %
    soc_numeric = pd.to_numeric(soc_str) / 100.0  # 转换为数值并归一化（0~1）

    time_max_real = time_real[-1]
    time_max_sim = time_simulation[-1]
    real_time_max = min(time_max_real, time_max_sim)

    # 重采样时间，间隔可自行调整（此处每 10s 一点）
    time_resampled = np.arange(0, real_time_max + 1, 10)

    # 对仿真与实测电压做插值
    interp_func_sim = interp1d(time_simulation, voltage_simulation, kind='linear', fill_value="extrapolate")
    voltage_simulation_resampled = interp_func_sim(time_resampled)

    interp_func_real_volt = interp1d(time_real, voltage_real, kind='linear', fill_value="extrapolate")
    voltage_real_resampled = interp_func_real_volt(time_resampled)

    # 对实测 SOC 做插值
    interp_func_real_soc = interp1d(time_real, soc_numeric, kind='linear', fill_value="extrapolate")
    soc_resampled = interp_func_real_soc(time_resampled)

    # 根据 soc_range 进行筛选
    if soc_range == 'all':
        # 不做任何筛选，直接计算全区间 RMSE
        mask = np.ones_like(time_resampled, dtype=bool)
    else:
        # 以 soc_range = (low_soc, high_soc) 的形式筛选
        low_soc, high_soc = soc_range
        mask = (soc_resampled >= low_soc) & (soc_resampled <= high_soc)

    # 检查筛选后是否还有数据点
    if not np.any(mask):
        print("Warning: 在所给的 SOC 区间内没有数据，无法计算 RMSE。")
        # 根据业务需求，可返回 None 或者返回一个提示
        return None, None, None, None, None

    # 9. 取出筛选后的时间、电压、SOC
    time_resampled_out = time_resampled[mask]
    voltage_sim_filtered = voltage_simulation_resampled[mask]
    voltage_real_filtered = voltage_real_resampled[mask]
    soc_resampled_out = soc_resampled[mask]

    # 10. 计算 RMSE
    rmse_value = np.sqrt(mean_squared_error(voltage_real_filtered, voltage_sim_filtered))
    print(f"TIME DISCHARGE RMSE (SOC range={soc_range}): {rmse_value}")

    return time_resampled_out, voltage_sim_filtered, voltage_real_filtered, soc_resampled_out, rmse_value


def pybamm_sim(param, min_voltage, max_voltage, discharge_cur, time_max, capacity, temperature, file):
    electrode_height = min_max_func(0.6, 1, param[0])
    electrode_width = min_max_func(25, 30, param[1])
    Negative_electrode_conductivity = min_max_func(14, 215, param[2])
    Positive_electrode_diffusivity = min_max_func(5.9e-18, 1e-14, param[3])
    Positive_particle_radius = min_max_func(1e-8, 1e-5, param[4])
    Initial_concentration_in_positive_electrode = min_max_func(35.3766672, 31513, param[5])
    Initial_concentration_in_negative_electrode = min_max_func(48.8682, 29866, param[6])
    Positive_electrode_conductivity = min_max_func(0.18, 100, param[7])
    Negative_particle_radius = min_max_func(0.0000005083, 0.0000137, param[8])
    Negative_electrode_thickness = min_max_func(0.000036, 0.0007, param[9])
    Total_heat_transfer_coefficient = min_max_func(5, 35, param[10])
    Separator_density = min_max_func(397, 2470, param[11])
    Separator_thermal_conductivity = min_max_func(0.10672, 0.34, param[12])
    Positive_electrode_porosity = min_max_func(0.12728395, 0.4, param[13])
    Separator_specific_heat_capacity = min_max_func(700, 1978, param[14])
    Maximum_concentration_in_positive_electrode = min_max_func(22806, 63104, param[15])
    Negative_electrode_Bruggeman_coefficient = min_max_func(1.5, 4, param[16])
    Positive_electrode_Bruggeman_coefficient = min_max_func(1.5, 4, param[17])
    Separator_porosity = min_max_func(0.39, 1, param[18])
    Negative_current_collector_thickness = min_max_func(0.00001, 0.000025, param[19])
    Positive_current_collector_thickness = min_max_func(0.00001, 0.000025, param[20])
    Positive_electrode_thickness = min_max_func(0.000042, 0.0001, param[21])
    Positive_electrode_active_material_volume_fraction = min_max_func(0.28485556, 0.665, param[22])
    Negative_electrode_specific_heat_capacity = min_max_func(700, 1437, param[23])
    Positive_electrode_thermal_conductivity = min_max_func(1.04, 2.1, param[24])
    Negative_electrode_active_material_volume_fraction = min_max_func(0.372403, 0.75, param[25])
    # Negative_electrode_density = min_max_func(1555, 3100, param[26])
    # Positive_electrode_specific_heat_capacity = min_max_func(700, 1270, param[27])
    # Positive_electrode_density = min_max_func(2341, 4206, param[28])
    # Negative_electrode_thermal_conductivity = min_max_func(1.04, 1.7, param[29])
    # Cation_transference_number = min_max_func(0.25, 0.4, param[30])
    # Positive_current_collector_thermal_conductivity = min_max_func(158, 238, param[31])
    # Negative_current_collector_thermal_conductivity = min_max_func(267, 401, param[32])
    # Separator_Bruggeman_coefficient = min_max_func(1.5, 2, param[33])
    # Maximum_concentration_in_negative_electrode = min_max_func(24983, 33133, param[34])
    # Positive_current_collector_density = min_max_func(2700, 3490, param[35])
    # Negative_current_collector_density = min_max_func(8933, 11544, param[36])
    # Positive_current_collector_conductivity = min_max_func(35500000, 37800000, param[37])
    # Negative_current_collector_conductivity = min_max_func(58411000, 59600000, param[38])
    # Negative_electrode_porosity = min_max_func(0.25, 0.5, param[39])
    # min_voltage = min_max_func(min_voltage - 0.3, min_voltage + 0.3, param[40])
    # max_voltage = min_max_func(max_voltage - 0.3, max_voltage + 0.3, param[41])

    parameter_values = pybamm.ParameterValues("Prada2013")
    option = {"cell geometry": "arbitrary", "thermal": "lumped", "contact resistance": "false"}
    if model_type == "DFN":
        model = pybamm.lithium_ion.DFN()  # Doyle-Fuller-Newman model
    else:
        model = pybamm.lithium_ion.SPM()
    exp = pybamm.Experiment(
        [(
            f"Discharge at {discharge_cur} C for {time_max} seconds",  # ageing cycles
            # f"Discharge at 0.5 C until {min_voltage}V",  # ageing cycles
            # f"Charge at 0.5 C for 1830 seconds",  # ageing cycles
        )]
    )
    # data = pd.read_csv(file)
    # # 从 CSV 中提取数据列
    # time_data = data['time'].values  # 总时间数据（秒）
    # current_data = data['A'].values  # 电流数据（安培，负值表示放电）
    # current_interpolant = pybamm.Interpolant(time_data, current_data, pybamm.t)
    # parameter_values["Current function [A]"] = current_interpolant

    param_dict = {
        "Number of electrodes connected in parallel to make a cell": 1,
        "Nominal cell capacity [A.h]": capacity,
        "Lower voltage cut-off [V]": min_voltage - 0.5,
        "Upper voltage cut-off [V]": max_voltage + 0.5,
        "Ambient temperature [K]": 273.15 + 25,
        "Initial temperature [K]": 273.15 + temperature,
        # "Total heat transfer coefficient [W.m-2.K-1]": 10,
        # "Cell cooling surface area [m2]": 0.126,
        # "Cell volume [m3]": 0.00257839,
        # cell
        "Electrode height [m]": electrode_height,
        "Electrode width [m]": electrode_width,
        "Negative electrode conductivity [S.m-1]": Negative_electrode_conductivity,
        "Positive electrode diffusivity [m2.s-1]": Positive_electrode_diffusivity,
        "Positive particle radius [m]": Positive_particle_radius,
        "Initial concentration in positive electrode [mol.m-3]": Initial_concentration_in_positive_electrode,
        "Initial concentration in negative electrode [mol.m-3]": Initial_concentration_in_negative_electrode,
        "Positive electrode conductivity [S.m-1]": Positive_electrode_conductivity,
        "Negative particle radius [m]": Negative_particle_radius,
        "Negative electrode thickness [m]": Negative_electrode_thickness,
        "Total heat transfer coefficient [W.m-2.K-1]": Total_heat_transfer_coefficient,
        "Separator density [kg.m-3]": Separator_density,
        "Separator thermal conductivity [W.m-1.K-1]": Separator_thermal_conductivity,
        "Positive electrode porosity": Positive_electrode_porosity,
        "Separator specific heat capacity [J.kg-1.K-1]": Separator_specific_heat_capacity,
        "Maximum concentration in positive electrode [mol.m-3]": Maximum_concentration_in_positive_electrode,
        "Negative electrode Bruggeman coefficient (electrolyte)": Negative_electrode_Bruggeman_coefficient,
        "Positive electrode Bruggeman coefficient (electrolyte)": Positive_electrode_Bruggeman_coefficient,
        "Separator porosity": Separator_porosity,
        "Negative current collector thickness [m]": Negative_current_collector_thickness,
        "Positive current collector thickness [m]": Positive_current_collector_thickness,
        "Positive electrode thickness [m]": Positive_electrode_thickness,
        "Positive electrode active material volume fraction": Positive_electrode_active_material_volume_fraction,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": Negative_electrode_specific_heat_capacity,
        "Positive electrode thermal conductivity [W.m-1.K-1]": Positive_electrode_thermal_conductivity,
        "Negative electrode active material volume fraction": Negative_electrode_active_material_volume_fraction,
        # "Negative electrode density [kg.m-3]": Negative_electrode_density,
        # "Positive electrode specific heat capacity [J.kg-1.K-1]": Positive_electrode_specific_heat_capacity,
        # "Positive electrode density [kg.m-3]": Positive_electrode_density,
        # "Negative electrode thermal conductivity [W.m-1.K-1]": Negative_electrode_thermal_conductivity,
        # "Cation transference number": Cation_transference_number,
        # "Positive current collector thermal conductivity [W.m-1.K-1]": Positive_current_collector_thermal_conductivity,
        # "Negative current collector thermal conductivity [W.m-1.K-1]": Negative_current_collector_thermal_conductivity,
        # "Separator Bruggeman coefficient (electrolyte)": Separator_Bruggeman_coefficient,
        # "Maximum concentration in negative electrode [mol.m-3]": Maximum_concentration_in_negative_electrode,
        # "Positive current collector density [kg.m-3]": Positive_current_collector_density,
        # "Negative current collector density [kg.m-3]": Negative_current_collector_density,
        # "Positive current collector conductivity [S.m-1]": Positive_current_collector_conductivity,
        # "Negative current collector conductivity [S.m-1]": Negative_current_collector_conductivity,
        # "Negative electrode porosity": Negative_electrode_porosity,

    }
    # Update the parameter value
    parameter_values.update(param_dict, check_already_exists=False)
    # Define the parameter to vary
    safe_solver = pybamm.CasadiSolver(mode="safe", dt_max=120)
    # Create a simulation
    sim = pybamm.Simulation(model, parameter_values=parameter_values, solver=safe_solver, experiment=exp)
    # Run the simulation
    sim.solve(initial_soc=1)
    sol = sim.solution
    return parameter_values, sol


def main_simulationMO(param, soc_range, save=False, plot=False):
    param_list = ["Ai2020", "Chen2020", "Prada2013"]
    # pybamm.set_logging_level("NOTICE")
    names = name.split(",")
    file_list = [f"./bat_data/{single}.csv" for single in names]
    all_time_rmse = []
    for i, file in enumerate(file_list):
        discharge_cur = float(names[i].split("-")[-1].replace("C", ""))
        temperature = int(names[i].split("-")[1].replace("T", ""))
        time_max, max_voltage, min_voltage, capacity = read_file(file_name=file)
        parameter_values, sol = pybamm_sim(param, min_voltage, max_voltage, discharge_cur, time_max, capacity, temperature, file)
        # soc_resampled, soc_voltage_simulation_resampled, soc_voltage_resampled, soc_rmse_value = compute_soc_discharge(sol=sol, capacity=parameter_values["Nominal cell capacity [A.h]"],file_path=file)
        time_resampled_out, voltage_sim_filtered, voltage_real_filtered, soc_resampled_out, rmse_value = compute_time_discharge(sol=sol, file_path=file, soc_range=soc_range)
        all_time_rmse.append(rmse_value)
        if plot:
            # plot_soc_discharge(soc_resampled, soc_voltage_simulation_resampled, soc_voltage_resampled, soc_rmse_value)
            plot_time_discharge(time_resampled_out, voltage_sim_filtered, voltage_real_filtered, rmse_value, discharge_cur)
        if save:
            df = pd.DataFrame({"real_time": time_resampled_out, "real_voltage": voltage_real_filtered, "simu_time": time_resampled_out, "simu_voltage": voltage_sim_filtered})
            df.to_csv(f"./simu_data/{subdir_name}/exp_{file_name}-T{temperature}-{discharge_cur}C-{model_type}.csv", index=False, sep=",")
    return all_time_rmse


def catch_error_simulation(solution, soc_range, return_dict):
    try:
        all_time_rmse = main_simulationMO(solution, soc_range)
        return_dict['result'] = all_time_rmse  # 返回计算结果
        return_dict['reason'] = 'No Problem!'
    except Exception as e:
        print(f"Error occurred: {e}")
        return_dict['result'] = [1.5] * wc_num  # 发生错误时设置为 None
        return_dict['reason'] = e


# 定义包装函数以处理超时和错误
def run_with_timeout(param, soc_range, timeout=15):
    param = param.cpu().numpy()
    print('param', param)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()  # 用于在进程间共享数据
    process = multiprocessing.Process(target=catch_error_simulation, args=(param, soc_range, return_dict))
    process.start()
    process.join(timeout)  # 等待进程完成，最多等待 timeout 秒

    if process.is_alive():
        process.terminate()  # 超时，终止进程
        process.join()  # 等待进程真正结束
        reason = "Over time!"
        return [1.5] * wc_num, reason  # 超时返回 100
    # 检查返回值是否为 NaN
    result = return_dict.get('result')
    try:
        print("\033[31m result:\033[0m", result)
        if True in np.isnan(result):
            reason = "Nan!"
            return [1.5] * wc_num, reason  # 发生错误或返回 NaN，返回 100
        else:
            dict_reason = return_dict.get('reason')
            return result, dict_reason  # 返回正常结果
    except Exception as e:
        print(f"Error occurred: {e}")
        reason = e
        return [1.5] * wc_num, reason


def obj_func(solution, soc_range):
    all_time_rmse, reason = run_with_timeout(solution, soc_range)
    obj = max(all_time_rmse)
    print("\033[31m Norm Solution Value\033[0m", solution)
    print("\033[31m RMSE (V):\033[0m", [mv for mv in all_time_rmse])
    print("\033[31m Value (V):\033[0m", obj)
    print("\033[31m Error Reason:\033[0m", reason)
    return obj


# 评估函数
def eval_objective(x):
    """返回整体的RMSE"""
    soc_range = 'all'
    return obj_func(x, soc_range) 


def eval_c1(x):
    """低SOC段的约束条件: RMSE < 20mV"""
    soc_range = (0.1, 0.3)
    return obj_func(x, soc_range) - 0.02  # 转换为约束形式 c1(x) <= 0


def eval_c2(x):
    """高SOC段的约束条件: RMSE < 20mV"""
    soc_range = (0.7, 0.9)
    return obj_func(x, soc_range) - 0.02  # 转换为约束形式 c2(x) <= 0


@dataclass
class ScboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = float("inf")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    tkwargs = {"device": device, "dtype": dtype}
    best_constraint_values: Tensor = torch.ones(2, **tkwargs) * torch.inf
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))


def update_tr_length(state: ScboState):
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def get_best_index_for_batch(Y: Tensor, C: Tensor):
    is_feas = (C <= 0).all(dim=-1)
    if is_feas.any():  # 如果存在可行点
        score = Y.clone()
        score[~is_feas] = float("inf")  # 不满足约束的点设为无穷大
        return score.argmin()  # 选择目标函数值最小的可行点，即最小的RMSE
    return C.clamp(min=0).sum(dim=-1).argmin()


def update_state(state, Y_next, C_next):
    best_ind = get_best_index_for_batch(Y=Y_next, C=C_next)
    y_next, c_next = Y_next[best_ind], C_next[best_ind]

    if (c_next <= 0).all():
        improvement_threshold = state.best_value + 1e-3 * math.fabs(state.best_value)
        if y_next > improvement_threshold or (state.best_constraint_values > 0).any():
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1
    else:
        total_violation_next = c_next.clamp(min=0).sum(dim=-1)
        total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1)
        if total_violation_next < total_violation_center:
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1

    state = update_tr_length(state)
    return state


def get_initial_points(dim, n_pts, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    base_name = name.split(",")[0].split("-")[0]
    bayes_csv = f"./solutions/Bayes/{base_name}MO-Constraint-{model_type}.csv"
    for i in range(n_pts):
        print(f"Loading Bayesian optimization results {i+1} from: {bayes_csv}")
        X_init[i] = torch.tensor(read_csv_solution(bayes_csv, i), **tkwargs)
    return X_init


def generate_batch(
        state,
        model,
        X,
        Y,
        C,
        batch_size,
        n_candidates,
        constraint_model,
        sobol: SobolEngine,
):
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    best_ind = get_best_index_for_batch(Y=Y, C=C)
    x_center = X[best_ind, :].clone()
    tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)

    dim = X.shape[-1]
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
        model=model, constraint_model=constraint_model, replacement=False
    )
    with torch.no_grad():
        X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size)

    return X_next


def get_fitted_model(X, Y):
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(
        MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
    )
    model = SingleTaskGP(
        X,
        Y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    with gpytorch.settings.max_cholesky_size(float("inf")):
        fit_gpytorch_mll(mll)

    return model


# 定义保存数据的函数，使用 JSON 格式
def save_data(train_X, train_Y, C1, C2, filename='data.json'):
    # 确保 tensor 在 CPU 上并转换为 NumPy 数组
    train_X_np = train_X.cpu().numpy()
    train_Y_np = train_Y.cpu().numpy().flatten()  # 展平成一维
    C1_np = C1.cpu().numpy().flatten()
    C2_np = C2.cpu().numpy().flatten()

    # 创建字典列表
    data_list = []
    for i in range(train_X_np.shape[0]):
        data_list.append({
            "train_X": train_X_np[i].tolist(),  # 转换为列表
            "train_Y": float(train_Y_np[i]),  # 确保为浮点数
            "C1": float(C1_np[i]),
            "C2": float(C2_np[i])
        })

    # 按 train_Y 排序
    data_list.sort(key=lambda x: x["train_Y"])

    # 保存到 JSON 文件
    with open(filename, 'w') as f:
        json.dump(data_list, f, indent=4)


def read_csv_solution(csv_file, i):
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Get the first row solution (best solution)
        solution_str = df.iloc[i]['Solution']
        cleaned_str = solution_str.strip('[]')  # 去除可能的方括号和两端空白
        numbers = cleaned_str.split()  # 按空格分割成列表
        solution = np.array(list(map(float, numbers)))  # 转为浮点数后创建数组

        print(solution)
        print(f"Successfully loaded solution from {csv_file}")
        print(f"Solution shape: {solution.shape}")
        print(f"Solution: {solution}")

        return solution
    except Exception as e:
        print(f"Error reading CSV solution: {e}")
        raise


def optimize_battery_params():
    train_X = get_initial_points(dim, n_init)

    # 并行计算初始点的目标值和约束
    njobs = 32
    train_Y_list = Parallel(n_jobs=njobs, prefer="threads")(delayed(eval_objective)(x) for x in train_X)
    C1_list = Parallel(n_jobs=njobs, prefer="threads")(delayed(eval_c1)(x) for x in train_X)
    C2_list = Parallel(n_jobs=njobs, prefer="threads")(delayed(eval_c2)(x) for x in train_X)

    # 统计有效点数量
    useful_point = sum(1 for y in train_Y_list if y != 1.5)
    print("################################################")
    print("num of useful point:", useful_point)
    print("################################################")

    # 转换为 tensor
    train_Y = torch.tensor(train_Y_list, **tkwargs).unsqueeze(-1)
    C1 = torch.tensor(C1_list, **tkwargs).unsqueeze(-1)
    C2 = torch.tensor(C2_list, **tkwargs).unsqueeze(-1)

    # 初始化 SCBO 状态
    state = ScboState(dim, batch_size=batch_size)
    N_CANDIDATES = 50
    sobol = SobolEngine(dim, scramble=True, seed=1)
    stop_length = 100

    # 保存可行解及其 RMSE 和约束值列表
    feasible_solutions = []
    feasible_rmse = []
    feasible_c1 = []  # 添加存储C1约束值的列表
    feasible_c2 = []  # 添加存储C2约束值的列表

    # --- 步骤 2: 在 while 循环中添加错误处理并保存到 JSON ---
    while True:
        try:
            # 拟合 GP 模型
            model = get_fitted_model(train_X, train_Y)
            c1_model = get_fitted_model(train_X, C1)
            c2_model = get_fitted_model(train_X, C2)

            # 生成新的候选点
            with gpytorch.settings.max_cholesky_size(float("inf")):
                X_next = generate_batch(
                    state=state,
                    model=model,
                    X=train_X,
                    Y=train_Y,
                    C=torch.cat((C1, C2), dim=-1),
                    batch_size=batch_size,
                    n_candidates=N_CANDIDATES,
                    constraint_model=ModelListGP(c1_model, c2_model),
                    sobol=sobol,
                )

            # 并行评估新的候选点
            Y_next_list = Parallel(n_jobs=njobs, prefer="threads")(delayed(eval_objective)(x) for x in X_next)
            C1_next_list = Parallel(n_jobs=njobs, prefer="threads")(delayed(eval_c1)(x) for x in X_next)
            C2_next_list = Parallel(n_jobs=njobs, prefer="threads")(delayed(eval_c2)(x) for x in X_next)

            # 转换为 tensor
            Y_next = torch.tensor(Y_next_list, **tkwargs).unsqueeze(-1)
            C1_next = torch.tensor(C1_next_list, **tkwargs).unsqueeze(-1)
            C2_next = torch.tensor(C2_next_list, **tkwargs).unsqueeze(-1)

            for i in range(len(Y_next_list)):
                if Y_next_list[i] != 1.5:
                    feasible_solutions.append(X_next[i].cpu().numpy())
                    feasible_rmse.append(Y_next_list[i])
                    feasible_c1.append(C1_next_list[i])  # 保存C1约束值
                    feasible_c2.append(C2_next_list[i])  # 保存C2约束值
                    if len(feasible_solutions) > 100:
                        worst_idx = np.argmax(feasible_rmse)
                        del feasible_solutions[worst_idx]
                        del feasible_rmse[worst_idx]
                        del feasible_c1[worst_idx]  # 同时删除对应的C1约束值
                        del feasible_c2[worst_idx]  # 同时删除对应的C2约束值

            # 更新有效点数量
            useful_point += sum(1 for y in Y_next_list if y != 1.5)
            print("################################################")
            print("num of useful point:", useful_point)
            print("################################################")

            # 更新状态
            C_next = torch.cat([C1_next, C2_next], dim=-1)
            state = update_state(state=state, Y_next=Y_next, C_next=C_next)

            # 添加新数据
            train_X = torch.cat((train_X, X_next), dim=0)
            train_Y = torch.cat((train_Y, Y_next), dim=0)
            C1 = torch.cat((C1, C1_next), dim=0)
            C2 = torch.cat((C2, C2_next), dim=0)

            # 打印当前状态
            if (state.best_constraint_values <= 0).all():
                print(f"{len(train_X)}) Best RMSE: {-state.best_value:.2e}, TR length: {state.length:.2e}")
            else:
                violation = state.best_constraint_values.clamp(min=0).sum()
                print(
                    f"{len(train_X)}) No feasible point yet! Constraint violation: "
                    f"{violation:.2e}, TR length: {state.length:.2e}"
                )

            # 检查停止条件
            print("\033[31m################################################\033[0m")
            print("\033[31mLength of train Y:\033[0m", len(train_Y))
            print("\033[31m################################################\033[0m")
            if len(train_Y) >= stop_length:
                print(f"Stopping condition reached: train_Y length >= {stop_length}")
                break

        except Exception as e:
            print(f"An error occurred: {e}")
            # 保存当前数据到 JSON
            save_data(train_X, train_Y, C1, C2, filename='./solutions/final_data.json')
            # 返回最优结果
            best_ind = get_best_index_for_batch(Y=train_Y, C=torch.cat((C1, C2), dim=-1))
            best_params = train_X[best_ind]
            best_rmse = train_Y[best_ind].item()
            print('\033[31m best params:\033[0m ', best_params)
            print('\033[31m best rmse:\033[0m ', best_rmse)

            # 按 RMSE 排序（使用zip将四个列表打包在一起）
            sorted_data = sorted(zip(feasible_solutions, feasible_rmse, feasible_c1, feasible_c2), key=lambda x: x[1])

            # 写入 CSV 文件，增加C1和C2列
            with open(rf'./solutions/Bayes/{file_name}.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Solution', 'RMSE', 'C1', 'C2'])  # 增加C1和C2列标题
                for solution, rmse, c1, c2 in sorted_data:  # 解包四个值
                    writer.writerow([solution, rmse, c1, c2])  # 写入所有四个值

            return best_params, best_rmse

    # 正常结束时保存数据到 JSON 并返回最优结果
    save_data(train_X, train_Y, C1, C2, filename='./solutions/final_data.json')
    best_ind = get_best_index_for_batch(Y=train_Y, C=torch.cat((C1, C2), dim=-1))
    best_params = train_X[best_ind]
    best_rmse = main_simulationMO(best_params.cpu().numpy(), 'all', save=True, plot=False)
    print("minimum value of RMSE:", min(train_Y))
    print('\033[31m best params:\033[0m ', best_params)
    print('\033[31m best rmse:\033[0m ', best_rmse)

    # 绘图
    fig, ax = plt.subplots(figsize=(8, 6))
    score = train_Y.clone()
    fx = np.minimum.accumulate(score.cpu().numpy())
    plt.plot(fx, marker="", lw=3)
    plt.plot([0, len(train_Y)], [0.01, 0.01], "k--", lw=3)
    plt.ylabel("Function value", fontsize=18)
    plt.xlabel("Number of evaluations", fontsize=18)
    plt.xlim([0, len(train_Y)])
    plt.grid(True)
    fig.savefig('./all_plot/Bayes_Cons.png', dpi=600)

    # 按 RMSE 排序（现在包含约束值）
    sorted_data = sorted(zip(feasible_solutions, feasible_rmse, feasible_c1, feasible_c2), key=lambda x: x[1])

    # 写入 CSV 文件，增加C1和C2列
    with open(rf'./solutions/Bayes/{file_name}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Solution', 'RMSE', 'C1', 'C2'])  # 增加C1和C2列标题
        for solution, rmse, c1, c2 in sorted_data:  # 解包四个值
            writer.writerow([solution, rmse, c1, c2])  # 写入所有四个值

    return best_params, train_Y[best_ind].item()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    tkwargs = {"device": device, "dtype": dtype}
    dim = 26
    lb = torch.zeros(dim, **tkwargs)
    ub = torch.ones(dim, **tkwargs)
    bounds = torch.stack([lb, ub])

    batch_size = 10
    n_init = 10
    print("devce:", device)
    dtype = torch.double

    name_list = ["81#-T25-0.1C", "81#-T25-0.2C", "81#-T25-0.33C", "81#-T25-1C"]
    d
    parser = argparse.ArgumentParser(description="Run Bayes optimization or load solution.")
    # 设置默认参数值
    default_train = True
    default_filename = "81#-T25-0.1C,81#-T25-0.2C,81#-T25-0.33C,81#-T25-1C"  # 替换为实际要设置的默认文件名
    default_method = "Bayes"
    default_model = "DFN"
    parser.add_argument('--train', action='store_true', default=default_train, help='Train the model.')
    parser.add_argument('--filename', type=str, default=default_filename, help='Filename for the optimization or solution.')
    parser.add_argument('--method', type=str, choices=["GA", "Bayes", "Local"], default=default_method, help='Optimization Method.')
    parser.add_argument('--model', type=str, choices=["DFN", "SPM"], default=default_model, help='Model Type.')
    args = parser.parse_args()
    name = args.filename
    model_type = args.model
    file_name = name.split(",")[0].split("-")[0] + "MO-Constraint" + f"-{model_type}"
    subdir_name = args.method
    wc_num = len(args.filename.split(","))
    if args.train:
        print("Training the model...")
        print(f"Filename: {args.filename}")
        print(f"Method: {args.method}")
        print(f"Model: {args.model}")
        optimize_battery_params()
