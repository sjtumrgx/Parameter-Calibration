import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

import pybamm.expression_tree
import pybamm.expression_tree.binary_operators

# Constants
DEFAULT_CAPACITY = 304.5 

def plot_results(sim_time, sim_voltage, sim_current, 
                 time_data, voltage_data, current_data, soc_data, 
                 output_dir, file_num):
    """Plot the simulation results against experimental data."""
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Plot current
    ax[0].plot(sim_time, sim_current, 'b-', label='Simulation')
    ax[0].plot(time_data, current_data, 'r--', label='Experimental Data')
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Current [A]")
    ax[0].set_title("Battery Current")
    ax[0].grid(True)
    ax[0].legend()

    # Plot voltage
    ax[1].plot(sim_time, sim_voltage, 'b-', label='Simulation')
    ax[1].plot(time_data, voltage_data, 'r--', label='Experimental Data')
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Voltage [V]")
    ax[1].set_title("Battery Voltage")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, "simu_fig", f"{file_num}#-T25-DC-simulation.png")
    plt.savefig(fig_path)
    print(f"Saved figure to {fig_path}")
    plt.show()

def save_simulation_data(sim_time, sim_voltage, sim_current,
                         time_data, voltage_data, current_data, soc_data,
                         output_dir, file_num):
    """Save simulation data to CSV file."""
    data = {
        "Time": sim_time,
        "Voltage_simulation": sim_voltage,
        "Current_simulation": sim_current,
        "Voltage_data": np.interp(sim_time, time_data, voltage_data),
        "Current_data": np.interp(sim_time, time_data, current_data),
        "SOC_data": np.interp(sim_time, time_data, soc_data)
    }

    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "simu_data", f"{file_num}#-T25-DC-simulation.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved simulation data to {csv_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run battery simulation with custom drive cycle")
    parser.add_argument('--drive_cycle', type=str, default='./bat_data/chargingdata.csv',
                        help='Path to the drive cycle CSV file (default: ./bat_data/chargingdata.csv)')
    parser.add_argument('--temperature', type=int, default=25,
                        help='Operating temperature in Celsius (default: 25)')
    parser.add_argument('--battery_id', type=str, default="PS",
                        help='Identifier for the battery (default: PS)')
    args = parser.parse_args()

    # Create output directories
    output_dir = "./simulation_dc_ps_output"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "simu_data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "simu_fig"), exist_ok=True)

    
    # Battery parameters optimized for 304.5Ah battery
    parameters_dict = {
        # --------------------------------------------------
        # 1. 结构与几何 —— 与原始相同（容量 304.5 Ah，对应 210 片并联极片）
        # --------------------------------------------------
        "Number of electrodes connected in parallel to make a cell": 210,
        "Electrode height [m]": 0.188,
        "Electrode width [m]": 0.175,
        "Negative electrode thickness [m]": 105e-6,
        "Positive electrode thickness [m]": 115e-6,
        "Negative current collector thickness [m]": 15e-6,
        "Positive current collector thickness [m]": 20e-6,

        # --------------------------------------------------
        # 2. 组分体积分数 & 孔隙率（轻调 Separator）
        # --------------------------------------------------
        "Positive electrode active material volume fraction": 0.54,
        "Negative electrode active material volume fraction": 0.57,
        "Positive electrode porosity": 0.37,
        "Negative electrode porosity": 0.37,
        "Separator porosity": 0.44,  # ↓ 由 0.48 ‒→ 0.44

        # --------------------------------------------------
        # 3. 传质 / 颗粒尺度参数
        # --------------------------------------------------
        "Positive electrode diffusivity [m2.s-1]": 6.0e-13,   # ↑ 3.5e-13 ⇒ 6.0e-13
        "Negative electrode diffusivity [m2.s-1]": 4.5e-13,   # ↑ 2.5e-13 ⇒ 4.5e-13
        "Positive particle radius [m]": 1.3e-6,               # ↓ 1.8e-6  ⇒ 1.3e-6
        "Negative particle radius [m]": 2.2e-6,               # ↓ 2.8e-6  ⇒ 2.2e-6

        # --------------------------------------------------
        # 4. 电导 & Bruggeman
        # --------------------------------------------------
        "Positive electrode conductivity [S.m-1]": 200,
        "Negative electrode conductivity [S.m-1]": 300,
        "Electrolyte conductivity [S.m-1]": 1.4,              # 乘以 1.4 的经验放大系数
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.45,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.45,

        # --------------------------------------------------
        # 5. 浓度 & SOC 初始化（★ 起始电压 ≈ 3.20 V，对应 SOC₀ ≈ 0.05）
        #    —— 建议根据你的 OCV 曲线反求后再覆盖，此处给出近似值
        # --------------------------------------------------
        "Maximum concentration in positive electrode [mol.m-3]": 55000.0,  # ↑
        "Maximum concentration in negative electrode [mol.m-3]": 31500.0,
        "Initial concentration in positive electrode [mol.m-3]":  55000.0 * 0.05,  # ≈ 2750
        "Initial concentration in negative electrode [mol.m-3]":  31500.0 * 0.95,  # ≈ 29925


        # --------------------------------------------------
        # 8. 其它运行边界
        # --------------------------------------------------
        "Nominal cell capacity [A.h]": DEFAULT_CAPACITY,
        "Lower voltage cut-off [V]": 2.0,
        "Upper voltage cut-off [V]": 3.6,          # LFP 常用上限 3.60 V
        "Open-circuit voltage at 0% SOC [V]": 3.18,
        "Open-circuit voltage at 100% SOC [V]": 3.49,

        # 温度相关（沿用命令行参数）
        "Initial temperature [K]": 273.15 + args.temperature,
        "Ambient temperature [K]": 273.15 + args.temperature,
    }




    # Import drive cycle data
    drive_cycle_file = args.drive_cycle
    file_num = args.battery_id
    drive_cycle_data = pd.read_csv(drive_cycle_file)
    print(f"Loaded drive cycle data from {drive_cycle_file}")
    print(f"Drive cycle data shape: {drive_cycle_data.shape}")


    # Columns: V, A, SOC, Time
    time_data = drive_cycle_data['Time'].to_numpy()
    current_data = drive_cycle_data['A'].to_numpy()
    voltage_data = drive_cycle_data['V'].to_numpy()
    soc_data = drive_cycle_data['SOC'].to_numpy()
    simu_time = time_data[-1]
    
    # Create a DFN model for the simulation
    model = pybamm.lithium_ion.SPM()

    # Create current interpolant
    current_interpolant = pybamm.Interpolant(time_data, current_data, pybamm.t)

    # Set up parameter values and assign current function
    parameter_values = pybamm.ParameterValues("Prada2013")
    parameter_values.update(parameters_dict, check_already_exists=False)
    parameter_values["Current function [A]"] = current_interpolant


    U_p_orig = parameter_values["Positive electrode OCP [V]"]
    U_n_orig = parameter_values["Negative electrode OCP [V]"]
    # dU = 0.1  # 根据实验差值粗估
    # parameter_values["Positive electrode OCP [V]"] = lambda sto: U_n_orig(sto) + dU
    # parameter_values["Negative electrode OCP [V]"] = lambda sto: U_n_orig(sto) - dU * 0

    # Set up simulation with CasadiSolver in "fast" mode for drive cycles
    solver = pybamm.CasadiSolver(mode="fast")
    simulation = pybamm.Simulation(model, parameter_values=parameter_values, solver=solver)

    # Create evaluation timepoints and solve
    t_eval = np.linspace(0, simu_time - 1, simu_time)
    print(f"Solving simulation for {simu_time} seconds...")
    solution = simulation.solve(initial_soc=soc_data[0])
    print("Simulation completed successfully.")

    # Extract simulation results
    sim_time = solution["Time [s]"].entries
    sim_voltage = solution["Terminal voltage [V]"].entries
    sim_current = solution["Current [A]"].entries

    # Plot the results
    plot_results(sim_time, sim_voltage, sim_current,
                time_data, voltage_data, current_data, soc_data,
                output_dir, file_num)

    # Save data
    save_simulation_data(sim_time, sim_voltage, sim_current,
                        time_data, voltage_data, current_data, soc_data,
                        output_dir, file_num)


if __name__ == "__main__":
    main() 