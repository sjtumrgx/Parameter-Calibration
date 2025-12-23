# simulate_discharge.py
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

# Constants for default values if experimental file reading fails
DEFAULT_CAPACITY = 280.0  # Ah
DEFAULT_MAX_VOLTAGE = 4.2  # V
DEFAULT_MIN_VOLTAGE = 2.5  # V
DEFAULT_TIME_MAX_ESTIMATE_FACTOR = 1.2  # Estimate max time = (1/C_rate) * 3600 * factor


def read_file_safe(file_name):
    """Safely reads experimental file details, providing defaults if file not found."""
    try:
        data = pd.read_csv(file_name)
        # Extract time and voltage data for plotting
        time_data = data['time'].values if not data.empty else np.array([])
        voltage_data = data['V'].values if not data.empty else np.array([])

        # Estimate time_max more robustly, handle empty files
        time_max = data['time'].values[-1] if not data.empty else None
        voltage_max = data['V'].values[0] if not data.empty else DEFAULT_MAX_VOLTAGE
        # Ensure voltage_min is calculated reasonably even if data is short
        voltage_min = data['V'].values[-1] if not data.empty else DEFAULT_MIN_VOLTAGE
        capacity = data['Ah'].values[-1] if not data.empty else DEFAULT_CAPACITY

        # If time_max couldn't be determined, estimate based on C-rate in filename
        if time_max is None:
            try:
                discharge_cur = float(file_name.split('-')[-1].replace('C.csv', ''))
                time_max = (1.0 / discharge_cur) * 3600 * DEFAULT_TIME_MAX_ESTIMATE_FACTOR  # Estimate in seconds
            except:
                time_max = 3600 * 10  # Default fallback (e.g., 10 hours for 0.1C)
            print(f"Warning: Could not read time_max from {file_name}. Estimated as {time_max:.0f}s.")

        print(f"Read parameters from {file_name}: Capacity={capacity:.2f}Ah, V_max={voltage_max:.2f}V, V_min={voltage_min:.2f}V, Time_max={time_max:.0f}s")
        return time_max, voltage_max, voltage_min, capacity, time_data, voltage_data
    except FileNotFoundError:
        print(f"Warning: File not found: {file_name}. Using default values.")
        try:
            # Attempt to get C-rate from filename for time estimation
            discharge_cur = float(file_name.split('-')[-1].replace('C.csv', ''))
            time_max_est = (1.0 / discharge_cur) * 3600 * DEFAULT_TIME_MAX_ESTIMATE_FACTOR
        except:
            time_max_est = 3600 * 10  # Default fallback
        print(f"Using defaults: Capacity={DEFAULT_CAPACITY}Ah, V_max={DEFAULT_MAX_VOLTAGE}V, V_min={DEFAULT_MIN_VOLTAGE}V, Estimated Time_max={time_max_est:.0f}s")
        return time_max_est, DEFAULT_MAX_VOLTAGE, DEFAULT_MIN_VOLTAGE, DEFAULT_CAPACITY, np.array([]), np.array([])
    except Exception as e:
        print(f"Error reading file {file_name}: {e}. Using default values.")
        try:
            discharge_cur = float(file_name.split('-')[-1].replace('C.csv', ''))
            time_max_est = (1.0 / discharge_cur) * 3600 * DEFAULT_TIME_MAX_ESTIMATE_FACTOR
        except:
            time_max_est = 3600 * 10
        print(f"Using defaults: Capacity={DEFAULT_CAPACITY}Ah, V_max={DEFAULT_MAX_VOLTAGE}V, V_min={DEFAULT_MIN_VOLTAGE}V, Estimated Time_max={time_max_est:.0f}s")
        return time_max_est, DEFAULT_MAX_VOLTAGE, DEFAULT_MIN_VOLTAGE, DEFAULT_CAPACITY, np.array([]), np.array([])


def plot_simulation_result(time_simulation, voltage_simulation, time_experimental, voltage_experimental,
                           discharge_cur, temperature, battery_id, model_type, output_dir):
    """Plots the simulation results and experimental data if available."""
    fig, ax = plt.subplots()

    # Plot simulation data
    ax.plot(time_simulation, voltage_simulation, linestyle='-', label=f'{discharge_cur}C Simulation', color='blue')

    # Plot experimental data if available
    if len(time_experimental) > 0 and len(voltage_experimental) > 0:
        ax.plot(time_experimental, voltage_experimental, linestyle='--', label=f'{discharge_cur}C Experimental', color='red')

    plt.xlabel('Time [s]')
    plt.ylabel('Terminal Voltage [V]')
    plt.title(f"{battery_id}-T{temperature}-{discharge_cur}C ({model_type})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Ensure output directory exists
    fig_dir = os.path.join(output_dir, "simu_fig")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"{battery_id}-T{temperature}-{discharge_cur}C-{model_type}.png")
    fig.savefig(fig_path)
    print(f"Saved plot to: {fig_path}")
    # plt.show() # Optionally display the plot immediately
    plt.close(fig)  # Close the figure to free memory


def pybamm_sim_fixed_params(fixed_params, min_voltage, max_voltage, discharge_cur, time_max, capacity, temperature, model_type):
    """Runs a PyBaMM simulation with a fixed set of parameters."""

    parameter_values = pybamm.ParameterValues("Prada2013")  # Or choose another base set if needed

    # --- Prepare the parameter dictionary for PyBaMM ---
    # Use the fixed physical values directly
    param_dict = {
        "Number of electrodes connected in parallel to make a cell": fixed_params["N_parallel"],
        "Nominal cell capacity [A.h]": capacity,
        "Lower voltage cut-off [V]": min_voltage - 0.1,  # Give some buffer
        "Upper voltage cut-off [V]": max_voltage + 0.1,  # Give some buffer
        "Ambient temperature [K]": 273.15 + 25,  # Assuming ambient is 25C
        "Initial temperature [K]": 273.15 + temperature,

        # Directly assign values from the fixed_params dictionary
        "Electrode height [m]": fixed_params["electrode_height"],
        "Electrode width [m]": fixed_params["electrode_width"],
        "Negative electrode conductivity [S.m-1]": fixed_params["Negative_electrode_conductivity"],
        "Positive electrode diffusivity [m2.s-1]": fixed_params["Positive_electrode_diffusivity"],
        "Positive particle radius [m]": fixed_params["Positive_particle_radius"],
        "Initial concentration in positive electrode [mol.m-3]": fixed_params["Initial_concentration_in_positive_electrode"],
        "Initial concentration in negative electrode [mol.m-3]": fixed_params["Initial_concentration_in_negative_electrode"],
        "Positive electrode conductivity [S.m-1]": fixed_params["Positive_electrode_conductivity"],
        "Negative particle radius [m]": fixed_params["Negative_particle_radius"],
        "Negative electrode thickness [m]": fixed_params["Negative_electrode_thickness"],

        "Positive electrode porosity": fixed_params["Positive_electrode_porosity"],
        "Maximum concentration in positive electrode [mol.m-3]": fixed_params["Maximum_concentration_in_positive_electrode"],
        "Negative electrode Bruggeman coefficient (electrolyte)": fixed_params["Negative_electrode_Bruggeman_coefficient"],
        "Positive electrode Bruggeman coefficient (electrolyte)": fixed_params["Positive_electrode_Bruggeman_coefficient"],
        "Separator porosity": fixed_params["Separator_porosity"],
        "Negative current collector thickness [m]": fixed_params["Negative_current_collector_thickness"],
        "Positive current collector thickness [m]": fixed_params["Positive_current_collector_thickness"],
        "Positive electrode thickness [m]": fixed_params["Positive_electrode_thickness"],
        "Positive electrode active material volume fraction": fixed_params["Positive_electrode_active_material_volume_fraction"],
        "Negative electrode active material volume fraction": fixed_params["Negative_electrode_active_material_volume_fraction"],
    }

    # --- Select Model ---
    option = {"cell geometry": "arbitrary", "thermal": "lumped"}  # Keep options simple
    if model_type == "DFN":
        model = pybamm.lithium_ion.DFN()
    elif model_type == "SPM":
        model = pybamm.lithium_ion.SPM()
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'DFN' or 'SPM'.")

    # --- Define Experiment ---
    # Discharge until min_voltage or for estimated time_max, whichever comes first
    # experiment_string = f"Discharge at {discharge_cur} C until {min_voltage} V or for {time_max:.0f} seconds"
    # exp = pybamm.Experiment([experiment_string])
    exp = pybamm.Experiment(
        [(
            f"Discharge at {discharge_cur} C for {time_max} seconds",  # ageing cycles
            # f"Discharge at 0.5 C until {min_voltage}V",  # ageing cycles
            # f"Charge at 0.5 C for 1830 seconds",  # ageing cycles
        )]
    )
    # Update parameter values
    # Use check_already_exists=False cautiously, ensure parameter names match PyBaMM's exactly
    parameter_values.update(param_dict, check_already_exists=False)

    # --- Setup and Run Simulation ---
    safe_solver = pybamm.CasadiSolver(mode="safe", dt_max=60)  # Adjust dt_max if needed
    sim = pybamm.Simulation(model, parameter_values=parameter_values, solver=safe_solver, experiment=exp)

    print(f"Starting simulation for {discharge_cur}C...")
    try:
        sol = sim.solve(initial_soc=1.0)  # Start from fully charged
        print(f"Simulation for {discharge_cur}C finished.")
        return sol
    except Exception as e:
        print(f"Error during simulation for {discharge_cur}C: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run PyBaMM discharge simulations with fixed parameters.")
    parser.add_argument('--battery_id', type=str, default="81#", help='Identifier for the battery (e.g., 81#).')
    parser.add_argument('--temperature', type=int, default=25, help='Operating temperature in Celsius.')
    parser.add_argument('--model', type=str, choices=["DFN", "SPM"], default="DFN", help='PyBaMM Model Type (DFN or SPM).')
    parser.add_argument('--data_dir', type=str, default="./bat_data", help='Directory containing experimental CSV files (used for capacity/voltage limits).')
    parser.add_argument('--output_dir', type=str, default="./simulation_output", help='Directory to save simulation plots and data.')
    parser.add_argument('--save_data', default=True, help='Save simulation time/voltage data to CSV.')

    args = parser.parse_args()

    fixed_parameters = {
        # 几何与结构参数 - 保持不变
        "N_parallel": 200,
        "electrode_height": 0.195,
        "electrode_width": 0.172,
        "Negative_electrode_thickness": 100e-6,
        "Positive_electrode_thickness": 110e-6,
        "Negative_current_collector_thickness": 15e-6,
        "Positive_current_collector_thickness": 20e-6,

        # 材料组成参数 - 微调孔隙率
        "Positive_electrode_active_material_volume_fraction": 0.52,
        "Negative_electrode_active_material_volume_fraction": 0.55,
        "Positive_electrode_porosity": 0.38,  
        "Negative_electrode_porosity": 0.38,  
        "Separator_porosity": 0.5,  

        # 传输特性参数 - 关键优化区域
        "Positive_electrode_diffusivity": 3e-13,  # 大幅提高(原8e-14)，解决放电后期扩散限制
        "Negative_electrode_diffusivity": 2e-13,  # 添加负极扩散系数
        "Positive_particle_radius": 2e-6,  # 进一步减小(原3e-6)
        "Negative_particle_radius": 3e-6,  # 进一步减小(原4e-6)
        "Negative_electrode_conductivity": 100.0,
        "Positive_electrode_conductivity": 50.0,  # 提高(原30.0)，减轻高倍率欧姆极化
        "Negative_electrode_Bruggeman_coefficient": 1.5,  # 进一步降低(原1.8)
        "Positive_electrode_Bruggeman_coefficient": 1.5,  # 进一步降低(原1.8)

        # 浓度参数 - 保持不变
        "Initial_concentration_in_positive_electrode": 28000.0,
        "Initial_concentration_in_negative_electrode": 5000.0,
        "Maximum_concentration_in_positive_electrode": 51500.0,
        "Maximum_concentration_in_negative_electrode": 30555.0,  # 添加负极最大浓度

    }

    # ##########################################################################

    # Define the C-rates to simulate
    c_rates = [0.1, 0.2, 0.33, 1]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "simu_data"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "simu_fig"), exist_ok=True)

    # Loop through each C-rate
    # Loop through each C-rate
    for c_rate in c_rates:
        print(f"\n--- Simulating {c_rate}C Discharge ---")
        # Construct the expected experimental filename to read capacity/limits
        exp_file_name = os.path.join(args.data_dir, f"{args.battery_id}-T{args.temperature}-{c_rate}C.csv")

        # Read experimental file details (or use defaults)
        time_max, max_voltage, min_voltage, capacity, time_exp, voltage_exp = read_file_safe(exp_file_name)

        # Run the simulation with fixed parameters
        solution = pybamm_sim_fixed_params(
            fixed_params=fixed_parameters,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
            discharge_cur=c_rate,
            time_max=time_max,  # Use estimated max time for experiment def
            capacity=capacity,
            temperature=args.temperature,
            model_type=args.model
        )

        if solution:
            # Extract simulation results
            time_sim = solution["Time [s]"].entries
            voltage_sim = solution["Voltage [V]"].entries

            # Plot the results (both simulation and experimental if available)
            plot_simulation_result(
                time_sim, voltage_sim,
                time_exp, voltage_exp,
                c_rate, args.temperature, args.battery_id, args.model, args.output_dir
            )

            # Save data if requested
            if args.save_data:
                data_dir = os.path.join(args.output_dir, "simu_data")
                data_path = os.path.join(data_dir, f"{args.battery_id}-T{args.temperature}-{c_rate}C-{args.model}.csv")

                # If we have experimental data, include it in the saved file
                if len(time_exp) > 0 and len(voltage_exp) > 0:
                    df_sim = pd.DataFrame({
                        "Time_Sim [s]": time_sim,
                        "Voltage_Sim [V]": voltage_sim,
                        "Time_Exp [s]": np.interp(np.linspace(0, 1, len(time_sim)),
                                                  np.linspace(0, 1, len(time_exp)),
                                                  time_exp) if len(time_exp) > 1 else np.nan,
                        "Voltage_Exp [V]": np.interp(np.linspace(0, 1, len(time_sim)),
                                                     np.linspace(0, 1, len(voltage_exp)),
                                                     voltage_exp) if len(voltage_exp) > 1 else np.nan
                    })
                else:
                    df_sim = pd.DataFrame({"Time [s]": time_sim, "Voltage [V]": voltage_sim})

                df_sim.to_csv(data_path, index=False)
                print(f"Saved simulation data to: {data_path}")
        else:
            print(f"Simulation failed for {c_rate}C.")

    print("\nAll simulations completed.")


if __name__ == '__main__':
    main()
