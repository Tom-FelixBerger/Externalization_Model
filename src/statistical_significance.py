
import pandas as pd
import os
import numpy as np
from scipy.stats import mannwhitneyu

COLOR_DICT = {
    "alpha": "green",
    "beta": "red", 
    "gamma": "dodgerblue",
    "delta": "orange",
    "externalizers": "darkblue",
    "non-externalizers": "purple"
}

def read_data(csvpath="/../data/ABM_base_simulation.csv"):
    # Read the file
    file_dir = os.path.dirname(os.path.realpath(__file__))
    csv_file = file_dir+csvpath
    df = pd.read_csv(csv_file, dtype=str)

    # Convert to numeric
    df = df.loc[2:len(df)]
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass

    index_colnames = ["simulation", "generation", "learning_step"]
    df.rename(columns=dict(zip(df.columns[:3], index_colnames)), inplace=True)
    df.set_index(index_colnames, inplace=True)
    
    cols = []
    for i in range(int(len(df.columns)/2)):
        for attr in ["externalization", "behavior"]:
            cols.append((i, attr))
    df.columns = pd.MultiIndex.from_tuples(cols)

    return df

def test_for_significance(filename):
    df = read_data(csvpath="/../data/"+filename)
    
    max_sim = int(df.reset_index()["simulation"].max())
    simulation_results = []

    for simulation in range(max_sim+1):
        max_gen_in_sim = int(df.loc[pd.IndexSlice[simulation, :, :]].reset_index()["generation"].max())
        number_ext = df.loc[(simulation, max_gen_in_sim, 0), pd.IndexSlice[:, "externalization"]].sum()
        simulation_results.append(int(number_ext))


    print("Simulation results:", simulation_results)
    benchmark = [0]*96+[3,14,26,42]
    print("Benchmark results:", benchmark)

    # One-sided Mann-Whitney U test to test if the simulation results are greater than the benchmark
    statistic, p_value = mannwhitneyu(simulation_results, benchmark, alternative='greater')

    print("Mann-Whitney U statistic:", statistic)
    print("p-value:", p_value)

    



if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))
    test_for_significance("ABM_base_simulation.csv")
