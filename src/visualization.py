
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from model import initialize_dataframe

def read_data():
    # Read the file
    file_dir = os.path.dirname(os.path.realpath(__file__))
    csv_file = file_dir+"/../data/base_model_simulation.csv"
    df = pd.read_csv(csv_file, dtype=str)

    # Reconstruct index and column index
    df = df.loc[3:len(df)]
    index_colnames = ["generation", "learning_step", "game_round", "metric"]
    df.rename(columns=dict(zip(df.columns[:4], index_colnames)), inplace=True)
    for col in df.columns:
        if col != "metric":
            df[col] = pd.to_numeric(df[col])
    df.set_index(index_colnames, inplace=True)
    df.columns = initialize_dataframe().columns

    return df

def visualize_interaction_process(generation=0, learning_step=0):
    df = read_data()
    visualization_df = pd.DataFrame(columns=["game_round", "behavior", "matched_share", "payoff", "accumulated_payoff"])
    behaviors = ["alpha", "delta", "beta", "gamma"]
    visibility_offset = {"alpha": 0.008, "beta": 0, "gamma": -0.004, "delta": 0.004}
    for behavior in behaviors:
        total = df.loc[(generation, learning_step, 0, "shares"), pd.IndexSlice[:, behavior, :]].sum()
        accumulated_payoff = 0
        for game_round in range(15):
            if total > 0:
                matched_share = df.loc[(generation, learning_step, game_round, "shares"),
                                   pd.IndexSlice[:, behavior, "matched"]].sum()/total + visibility_offset[behavior]
                payoff_matched = df.loc[(generation, learning_step, game_round, "payoffs"),
                                 pd.IndexSlice["non-externalizing", behavior, "matched"]]
                payoff_unmatched = df.loc[(generation, learning_step, game_round, "payoffs"),
                                 pd.IndexSlice["non-externalizing", behavior, "unmatched"]]
                payoff = matched_share * payoff_matched + (1-matched_share) * payoff_unmatched
                accumulated_payoff += payoff
            else:
                print(behavior, game_round)
                payoff = 0
                matched_share = 0
            visualization_df.loc[len(visualization_df)] = [
                game_round, behavior, matched_share, payoff, accumulated_payoff
            ]

    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f'Development of Matched Share and Payoffs - Generation {generation}, Learning Step {learning_step}',
                fontsize=16)
    axes = axes.flatten()

    axes[0].set_title("Matched Share")
    axes[1].set_title("Payoff by Round")
    axes[2].set_title("Accumulated Payoff")

    # Method 1: Define a custom color palette with a dictionary
    # Replace these with your actual behavior names and desired colors
    color_dict = {
        "alpha": "green",
        "beta": "red", 
        "gamma": "dodgerblue",
        "delta": "orange"
    }

    # Create the plots using the custom palette
    line1 = sns.lineplot(data=visualization_df, x="game_round", y="matched_share", hue="behavior", 
                        marker='o', ax=axes[0], palette=color_dict, markeredgewidth=0)
    line2 = sns.lineplot(data=visualization_df, x="game_round", y="payoff", hue="behavior", 
                        marker='o', ax=axes[1], palette=color_dict, markeredgewidth=0)
    line3 = sns.lineplot(data=visualization_df, x="game_round", y="accumulated_payoff", hue="behavior", 
                        marker='o', ax=axes[2], palette=color_dict, markeredgewidth=0)

    # Remove the legends from the first three subplots
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    axes[2].get_legend().remove()

    # Turn off all spines, ticks, and labels for the fourth subplot
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    for spine in axes[3].spines.values():
        spine.set_visible(False)

    # Extract the handles and labels from one of the plots
    handles, labels = line1.get_legend_handles_labels()

    # Create a legend in the empty fourth subplot
    axes[3].legend(handles, labels, title="Behavior", loc='upper left', fontsize=12)

    return fig

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))
    fig = visualize_interaction_process(
        generation=0,
        learning_step=0
    )
    fig.savefig(file_dir+"/../plots/interaction_process_gen0_lst0.png")
