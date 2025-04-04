
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

def visualize_robustness(filename):
    df = read_data(csvpath="/../data/"+filename)
    
    max_gen = int(df.reset_index()["generation"].max())
    max_sim = int(df.reset_index()["simulation"].max())
    final_ext = []

    for simulation in range(max_sim+1):
        max_gen_in_sim = int(df.loc[pd.IndexSlice[simulation, :, :]].reset_index()["generation"].max())
        number_ext = df.loc[(simulation, max_gen_in_sim, 0), pd.IndexSlice[:, "externalization"]].sum()
        final_ext.append(number_ext)

    max_lst = int(df.reset_index()["learning_step"].max())
    learning_process_df = pd.DataFrame(columns=["learning_step", "behavior", "mean", "97.5", "2.5"])
    behaviors = ["alpha", "beta", "gamma", "delta"]
    for behavior in behaviors:
        for learning_step in range(max_lst+1):
            list_dict = {key: [] for key in behaviors}
            for simulation in range(max_sim+1):
                for key in behaviors:
                    list_dict[key].append((df.loc[(simulation, 0, learning_step), pd.IndexSlice[:, "behavior"]] == key).sum())
            for key in behaviors:
                learning_process_df.loc[len(learning_process_df)] = [
                    learning_step, key, pd.Series(list_dict[key]).mean(), pd.Series(list_dict[key]).quantile(0.975), pd.Series(list_dict[key]).quantile(0.025)
                ]

    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5))
    axes = axes.flatten()

    axes[0].set_ylabel("Number of Agents")
    axes[0].set_xlabel("Learning Step of First Generation")
    axes[1].set_ylabel("Number of Simulations")
    axes[1].set_xlabel("Final Externalizers")

    sns.lineplot(data=learning_process_df, x="learning_step", y="mean", hue="behavior", 
                        marker='o', palette=COLOR_DICT, markeredgewidth=0, ax=axes[0])
    sns.lineplot(data=learning_process_df, x="learning_step", y="97.5", hue="behavior",
                        linestyle='--', palette=COLOR_DICT, markeredgewidth=0, ax=axes[0])
    sns.lineplot(data=learning_process_df, x="learning_step", y="2.5", hue="behavior",
                        linestyle='--', palette=COLOR_DICT, markeredgewidth=0, ax=axes[0])
    sns.histplot(x=pd.Series(final_ext), stat="count", bins=20, ax=axes[1], color="gray", kde=True)

    # Remove legends from individual plots
    for ax in axes:
        if hasattr(ax, 'get_legend') and ax.get_legend() is not None:
            ax.get_legend().remove()

    # Create custom legend elements
    handles = []
    for behavior in behaviors:
        # Line marker for line plots
        line1 = Line2D([0], [0], color=COLOR_DICT[behavior], marker='o', linestyle='-', linewidth=2, markersize=6)
        line2 = Line2D([0], [0], color=COLOR_DICT[behavior], marker=None, linestyle='--', linewidth=2, markersize=6)

        handles.append((line1, f"{behavior} (mean)"))
        handles.append((line2, f"{behavior} (95% CI)"))

    # Combine them into a single legend
    fig.legend(
        handles=[h[0] for h in handles], 
        labels=[h[1] for h in handles],
        ncol=2,
        # bbox_to_anchor=(0.6, 0.2)
        bbox_to_anchor=(0.7, 0.3)
    )

    fig.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.4)

    return fig



if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))

    fig9 = visualize_robustness("ABM_base_simulation.csv")
    fig9.savefig(file_dir+"/../plots/fig9.png")

    fig10 = visualize_robustness("ABM_popsize_8_simulation.csv")
    fig9.savefig(file_dir+"/../plots/fig10.png")
