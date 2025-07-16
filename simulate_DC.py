import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse

# Constants
DEFAULT_CAPACITY = 280.0  # Ah


def min_max_func(min_val, max_val, norm_val):
    """Convert normalized value (0-1) to actual value in the given range."""
    return min_val + norm_val * (max_val - min_val)


def read_csv_solution(csv_file, i=0):
    """Read a specific solution (row) from a CSV file."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Get the specified row solution
        solution_str = df.iloc[i]['Solution']
        cleaned_str = solution_str.strip('[]')  # Remove possible brackets and spaces
        numbers = cleaned_str.split()  # Split by space
        solution = np.array(list(map(float, numbers)))  # Convert to array of floats

        print(f"Successfully loaded solution from {csv_file}")
        print(f"Solution shape: {solution.shape}")
        return solution
    except Exception as e:
        print(f"Error reading CSV solution: {e}")
        raise


def convert_normalized_params_to_real(param):
    """Convert normalized parameters (0-1) to their real physical values."""
    # Geometry and structure parameters
    N_parallel = min_max_func(180, 220, param[0])
    electrode_height = min_max_func(0.17, 0.22, param[1])
    electrode_width = min_max_func(0.15, 0.19, param[2])
    Negative_electrode_thickness = min_max_func(80e-6, 120e-6, param[3])
    Positive_electrode_thickness = min_max_func(90e-6, 130e-6, param[4])

    # Material composition parameters
    Positive_electrode_active_material_volume_fraction = min_max_func(0.45, 0.6, param[5])
    Negative_electrode_active_material_volume_fraction = min_max_func(0.48, 0.62, param[6])
    Positive_electrode_porosity = min_max_func(0.32, 0.45, param[7])
    Negative_electrode_porosity = min_max_func(0.32, 0.45, param[8])
    Separator_porosity = min_max_func(0.4, 0.6, param[9])

    # Transport properties
    Positive_electrode_diffusivity = min_max_func(1e-13, 1e-12, param[10])
    Negative_electrode_diffusivity = min_max_func(1e-13, 1e-12, param[11])
    Positive_particle_radius = min_max_func(1e-6, 4e-6, param[12])
    Negative_particle_radius = min_max_func(2e-6, 5e-6, param[13])
    Negative_electrode_conductivity = min_max_func(50.0, 150.0, param[14])
    Positive_electrode_conductivity = min_max_func(30.0, 80.0, param[15])
    Negative_electrode_Bruggeman_coefficient = min_max_func(1.2, 2.0, param[16])
    Positive_electrode_Bruggeman_coefficient = min_max_func(1.2, 2.0, param[17])

    # Concentration parameters
    Initial_concentration_in_positive_electrode = min_max_func(25000.0, 32000.0, param[18])
    Initial_concentration_in_negative_electrode = min_max_func(4000.0, 6000.0, param[19])
    Maximum_concentration_in_positive_electrode = min_max_func(45000.0, 58000.0, param[20])
    Maximum_concentration_in_negative_electrode = min_max_func(25000.0, 35000.0, param[21])

    # Print the denormalized parameters
    print("\nDenormalized Parameters:")
    print(f"N_parallel: {N_parallel:.2f}")
    print(f"electrode_height: {electrode_height:.4f} m")
    print(f"electrode_width: {electrode_width:.4f} m")
    print(f"Negative_electrode_thickness: {Negative_electrode_thickness * 1e6:.2f} μm")
    print(f"Positive_electrode_thickness: {Positive_electrode_thickness * 1e6:.2f} μm")
    print(f"Positive_electrode_active_material_volume_fraction: {Positive_electrode_active_material_volume_fraction:.4f}")
    print(f"Negative_electrode_active_material_volume_fraction: {Negative_electrode_active_material_volume_fraction:.4f}")
    print(f"Positive_electrode_porosity: {Positive_electrode_porosity:.4f}")
    print(f"Negative_electrode_porosity: {Negative_electrode_porosity:.4f}")
    print(f"Separator_porosity: {Separator_porosity:.4f}")
    print(f"Positive_electrode_diffusivity: {Positive_electrode_diffusivity:.2e} m²/s")
    print(f"Negative_electrode_diffusivity: {Negative_electrode_diffusivity:.2e} m²/s")
    print(f"Positive_particle_radius: {Positive_particle_radius * 1e6:.2f} μm")
    print(f"Negative_particle_radius: {Negative_particle_radius * 1e6:.2f} μm")
    print(f"Negative_electrode_conductivity: {Negative_electrode_conductivity:.2f} S/m")
    print(f"Positive_electrode_conductivity: {Positive_electrode_conductivity:.2f} S/m")
    print(f"Negative_electrode_Bruggeman_coefficient: {Negative_electrode_Bruggeman_coefficient:.2f}")
    print(f"Positive_electrode_Bruggeman_coefficient: {Positive_electrode_Bruggeman_coefficient:.2f}")
    print(f"Initial_concentration_in_positive_electrode: {Initial_concentration_in_positive_electrode:.2f} mol/m³")
    print(f"Initial_concentration_in_negative_electrode: {Initial_concentration_in_negative_electrode:.2f} mol/m³")
    print(f"Maximum_concentration_in_positive_electrode: {Maximum_concentration_in_positive_electrode:.2f} mol/m³")
    print(f"Maximum_concentration_in_negative_electrode: {Maximum_concentration_in_negative_electrode:.2f} mol/m³")

    # Create parameter dictionary
    param_dict = {
        "Number of electrodes connected in parallel to make a cell": N_parallel,
        "Electrode height [m]": electrode_height,
        "Electrode width [m]": electrode_width,
        "Negative electrode thickness [m]": Negative_electrode_thickness,
        "Positive electrode thickness [m]": Positive_electrode_thickness,
        "Negative current collector thickness [m]": 15e-6,  # Fixed value
        "Positive current collector thickness [m]": 20e-6,  # Fixed value

        # Material composition parameters
        "Positive electrode active material volume fraction": Positive_electrode_active_material_volume_fraction,
        "Negative electrode active material volume fraction": Negative_electrode_active_material_volume_fraction,
        "Positive electrode porosity": Positive_electrode_porosity,
        "Negative electrode porosity": Negative_electrode_porosity,
        "Separator porosity": Separator_porosity,

        # Transport properties
        "Positive electrode diffusivity [m2.s-1]": Positive_electrode_diffusivity,
        "Negative electrode diffusivity [m2.s-1]": Negative_electrode_diffusivity,
        "Positive particle radius [m]": Positive_particle_radius,
        "Negative particle radius [m]": Negative_particle_radius,
        "Negative electrode conductivity [S.m-1]": Negative_electrode_conductivity,
        "Positive electrode conductivity [S.m-1]": Positive_electrode_conductivity,
        "Negative electrode Bruggeman coefficient (electrolyte)": Negative_electrode_Bruggeman_coefficient,
        "Positive electrode Bruggeman coefficient (electrolyte)": Positive_electrode_Bruggeman_coefficient,

        # Concentration parameters
        "Initial concentration in positive electrode [mol.m-3]": Initial_concentration_in_positive_electrode,
        "Initial concentration in negative electrode [mol.m-3]": Initial_concentration_in_negative_electrode,
        "Maximum concentration in positive electrode [mol.m-3]": Maximum_concentration_in_positive_electrode,
        "Maximum concentration in negative electrode [mol.m-3]": Maximum_concentration_in_negative_electrode,
    }

    return param_dict


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run battery simulation with custom drive cycle")
    parser.add_argument('--drive_cycle', type=str, default='./bat_data/01#-T25-DC.csv',
                        help='Path to the drive cycle CSV file (default: ./bat_data/01#-T25-DC.csv)')
    parser.add_argument('--solution_file', type=str, default="solutions/Bayes/81#MO-Constraint-DFN-22.csv",
                        help='Path to the solution CSV file (default: solutions/Bayes/81#MO-Constraint-DFN-22.csv)')
    parser.add_argument('--solution_index', type=int, default=0,
                        help='Index of the solution row to use (default: 0)')
    args = parser.parse_args()

    # 1. Create output directories
    output_dir = "./simulation_dc_output"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "simu_data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "simu_fig"), exist_ok=True)

    # 2. Load battery parameters from solution file
    solution_file = args.solution_file
    solution_index = args.solution_index
    print(f"Reading solution from {solution_file}, using row {solution_index}")
    normalized_params = read_csv_solution(solution_file, solution_index)

    # 3. Convert normalized parameters to real physical values
    parameters_dict = convert_normalized_params_to_real(normalized_params)

    # 4. Add battery capacity and voltage limits
    parameters_dict.update({
        "Nominal cell capacity [A.h]": DEFAULT_CAPACITY,
        "Lower voltage cut-off [V]": 2.5,  # Generic lower voltage limit
        "Upper voltage cut-off [V]": 3.5,  # Generic upper voltage limit
        "Initial temperature [K]": 273.15 + 25,  # 25°C
        "Ambient temperature [K]": 273.15 + 25,  # 25°C
    })

    # 5. Create a DFN model for the simulation
    model = pybamm.lithium_ion.DFN()

    # 6. Import drive cycle data
    drive_cycle_file = args.drive_cycle
    file_num = drive_cycle_file.split('/')[-1].split('#')[0]
    print('File Number:', file_num)
    drive_cycle_data = pd.read_csv(drive_cycle_file)
    print(f"Loaded drive cycle data from {drive_cycle_file}")
    print(f"Drive cycle data shape: {drive_cycle_data.shape}")

    # Extract time and current data
    simu_time = drive_cycle_data.shape[0]
    # Columns: V, A, SOC, time
    time_data = drive_cycle_data.iloc[:simu_time, 3].to_numpy()  # time column (4th column)
    current_data = drive_cycle_data.iloc[:simu_time, 1].to_numpy()  # current column (2nd column)
    voltage_data = drive_cycle_data.iloc[:simu_time, 0].to_numpy()  # voltage column (1st column)
    soc_data = drive_cycle_data.iloc[:simu_time, 2].to_numpy()  # SOC column (3rd column)
    current_data = -current_data
    # 7. Create simulation time span
    simulation_time = len(time_data)  # Each time step is 1 second

    # 8. Create current interpolant
    current_interpolant = pybamm.Interpolant(time_data, current_data, pybamm.t)

    # 9. Set up parameter values and assign current function
    parameter_values = pybamm.ParameterValues("Prada2013")
    parameter_values.update(parameters_dict, check_already_exists=False)
    parameter_values["Current function [A]"] = current_interpolant

    # 10. Set up simulation with CasadiSolver in "fast" mode for drive cycles
    solver = pybamm.CasadiSolver(mode="fast")
    simulation = pybamm.Simulation(model, parameter_values=parameter_values, solver=solver)

    # 11. Create evaluation timepoints and solve
    t_eval = np.linspace(0, simu_time - 1, simu_time)
    print(f"Solving simulation for {simulation_time} seconds...")
    solution = simulation.solve(t_eval, initial_soc=soc_data[0] / 100)
    print("Simulation completed successfully.")

    # 12. Extract simulation results
    sim_time = solution["Time [s]"].entries
    sim_voltage = solution["Terminal voltage [V]"].entries
    sim_current = solution["Current [A]"].entries

    # 13. Plot the results
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))

    # Plot current
    ax[0].plot(sim_time, sim_current, 'b-', label='Simulation')
    ax[0].plot(time_data, current_data, 'r--', label='Drive Cycle Data')
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Current [A]")
    ax[0].set_title("Battery Current")
    ax[0].grid(True)
    ax[0].legend()

    # Plot voltage
    ax[1].plot(sim_time, sim_voltage, 'b-', label='Simulation')
    ax[1].plot(time_data, voltage_data, 'r--', label='Drive Cycle Data')
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Voltage [V]")
    ax[1].set_title("Battery Voltage")
    ax[1].grid(True)
    ax[1].legend()

    # Plot SOC (State of Charge)
    try:
        sim_soc = solution["State of Charge"].entries
        ax[2].plot(sim_time, sim_soc, 'b-', label='Simulation')
        ax[2].plot(time_data, soc_data, 'r--', label='Drive Cycle Data')
        ax[2].set_xlabel("Time [s]")
        ax[2].set_ylabel("State of Charge")
        ax[2].set_title("Battery SOC")
        ax[2].grid(True)
        ax[2].legend()
    except KeyError:
        print("State of Charge not available in simulation output")

    plt.tight_layout()

    # 14. Save figure
    fig_path = os.path.join(output_dir, "simu_fig", f"{file_num}#-T25-DC-simulation.png")
    plt.savefig(fig_path)
    print(f"Saved figure to {fig_path}")
    plt.show()

    # 15. Save data
    data = {
        "Time": sim_time,
        "Voltage_simulation": sim_voltage,
        "Current_simulation": sim_current,
        "Voltage_data": np.interp(sim_time, time_data, voltage_data),
        "Current_data": np.interp(sim_time, time_data, current_data),
        "SOC_data": np.interp(sim_time, time_data, soc_data)
    }

    # Add SOC from simulation if available
    if 'sim_soc' in locals():
        data["SOC_simulation"] = sim_soc

    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "simu_data", f"{file_num}#-T25-DC-simulation.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved simulation data to {csv_path}")


if __name__ == "__main__":
    main()
