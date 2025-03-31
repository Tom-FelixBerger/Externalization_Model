
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os
import model_2

plt.rcParams.update({
    'font.size': 16,  # Base font size
    'axes.titlesize': 16,  # Title font size
    'axes.labelsize': 16,  # Axis label font size
    'xtick.labelsize': 16,  # X tick label font size
    'ytick.labelsize': 16,  # Y tick label font size
    'legend.fontsize': 16,  # Legend font size
})

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

    # Reconstruct index and column index
    df = df.loc[2:len(df)]
    index_colnames = ["simulation", "generation", "learning_step", "game_round"]
    df.rename(columns=dict(zip(df.columns[:4], index_colnames)), inplace=True)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass
    df.set_index(index_colnames, inplace=True)
    df.columns = model_2.initialize_dataframe().columns

    return df

def visualize_robustness(filename):
    df = read_data(csvpath="/../data/"+filename)
    
    max_gen = int(df.reset_index()["generation"].max())
    max_sim = int(df.reset_index()["simulation"].max())
    natural_selection_df = pd.DataFrame(columns=["generation", "ext_trait", "mean", "97.5", "2.5"])

    for generation in range(max_gen+1):
        ext_list = []
        non_ext_list = []
        for simulation in range(max_sim+1):
            ext_list.append(df.loc[(simulation, generation, 0, 0), pd.IndexSlice[:, "externalization"]].sum())
            non_ext_list.append(len(df.columns)/5 - df.loc[(simulation, generation, 0, 0), pd.IndexSlice[:, "externalization"]].sum())
        natural_selection_df.loc[len(natural_selection_df)] = [
            generation, "externalizers", pd.Series(ext_list).mean(), pd.Series(ext_list).quantile(0.975), pd.Series(ext_list).quantile(0.025)
        ]
        natural_selection_df.loc[len(natural_selection_df)] = [
            generation, "non-externalizers", pd.Series(non_ext_list).mean(), pd.Series(non_ext_list).quantile(0.975), pd.Series(non_ext_list).quantile(0.025)
        ]

    max_lst = int(df.reset_index()["learning_step"].max())
    learning_process_df = pd.DataFrame(columns=["learning_step", "behavior", "mean", "97.5", "2.5"])
    behaviors = ["alpha", "beta", "gamma", "delta"]
    for behavior in behaviors:
        for learning_step in range(max_lst+1):
            list_dict = {key: [] for key in behaviors}
            for simulation in range(max_sim+1):
                for key in behaviors:
                    list_dict[key].append((df.loc[(simulation, 0, learning_step, 0), pd.IndexSlice[:, "behavior"]] == key).sum())
            for key in behaviors:
                learning_process_df.loc[len(learning_process_df)] = [
                    learning_step, key, pd.Series(list_dict[key]).mean(), pd.Series(list_dict[key]).quantile(0.975), pd.Series(list_dict[key]).quantile(0.025)
                ]

    result_ext_df = pd.DataFrame(columns=["simulation", "externalizers"])
    for simulation in range(max_sim+1):
        result_ext_df.loc[len(result_ext_df)] = [
            simulation, df.loc[(simulation, max_gen, 0, 0), pd.IndexSlice[:, "externalization"]].sum()
        ]


    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes = axes.flatten()

    axes[0].set_ylabel("Number of Agents")
    axes[0].set_xlabel("Learning Step of First Generation")
    axes[1].set_ylabel("Number of Agents")
    axes[1].set_xlabel("Generation")
    axes[2].set_ylabel("Number of Simulations")
    axes[2].set_xlabel("Final Externalizers")

    sns.lineplot(data=learning_process_df, x="learning_step", y="mean", hue="behavior", 
                        marker='o', palette=COLOR_DICT, markeredgewidth=0, ax=axes[0])
    sns.lineplot(data=learning_process_df, x="learning_step", y="97.5", hue="behavior",
                        linestyle='--', palette=COLOR_DICT, markeredgewidth=0, ax=axes[0])
    sns.lineplot(data=learning_process_df, x="learning_step", y="2.5", hue="behavior",
                        linestyle='--', palette=COLOR_DICT, markeredgewidth=0, ax=axes[0])
    sns.lineplot(data=natural_selection_df, x="generation", y="mean", hue="ext_trait", 
                        marker='o', palette=COLOR_DICT, markeredgewidth=0, ax=axes[1])  # Pass ax to seaborn
    sns.lineplot(data=natural_selection_df, x="generation", y="97.5", hue="ext_trait",
                        linestyle='--', palette=COLOR_DICT, markeredgewidth=0, ax=axes[1])
    sns.lineplot(data=natural_selection_df, x="generation", y="2.5", hue="ext_trait",
                        linestyle='--', palette=COLOR_DICT, markeredgewidth=0, ax=axes[1])
    sns.histplot(data=result_ext_df, x="externalizers", stat="count", bins=20, ax=axes[2], color="gray", kde=True)

    # Remove legends from individual plots
    for ax in axes:
        if hasattr(ax, 'get_legend') and ax.get_legend() is not None:
            ax.get_legend().remove()

    # Create custom legend elements
    handles = []
    for behavior in behaviors:
        # Line marker for line plots
        line = Line2D([0], [0], color=COLOR_DICT[behavior], marker='o', linestyle='-', linewidth=2, markersize=6)

        handles.append((line, behavior))
    
    for ext_trait in ["externalizers", "non-externalizers"]:
        line = Line2D([0], [0], color=COLOR_DICT[ext_trait], marker='o', linestyle='-', linewidth=2, markersize=6)
        handles.append((line, ext_trait))

    # Combine them into a single legend
    fig.legend(
        handles=[h[0] for h in handles], 
        labels=[h[1] for h in handles],
        ncol=3,
        bbox_to_anchor=(0.6, 0.2)
    )

    fig.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.3)

    return fig



if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))

    fig9 = visualize_robustness("ABM_base_simulation.csv")
    fig9.savefig(file_dir+"/../plots/fig9.png")
