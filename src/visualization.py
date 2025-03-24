
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os
from model import initialize_dataframe

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
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    axes = axes.flatten()

    axes[0].set_ylabel("Matched Share")
    axes[0].set_xlabel("Game Round")
    axes[1].set_ylabel("Payoff by Round")
    axes[1].set_xlabel("Game Round")
    axes[2].set_ylabel("Accumulated Payoff")
    axes[2].set_xlabel("Game Round")

    # Create the plots using the custom palette
    line1 = sns.lineplot(data=visualization_df, x="game_round", y="matched_share", hue="behavior", 
                        marker='o', ax=axes[0], palette=COLOR_DICT, markeredgewidth=0)
    line2 = sns.lineplot(data=visualization_df, x="game_round", y="payoff", hue="behavior", 
                        marker='o', ax=axes[1], palette=COLOR_DICT, markeredgewidth=0)
    line3 = sns.lineplot(data=visualization_df, x="game_round", y="accumulated_payoff", hue="behavior", 
                        marker='o', ax=axes[2], palette=COLOR_DICT, markeredgewidth=0)

    # After creating all three subplots
    handles, labels = axes[0].get_legend_handles_labels()  # Get the legend handles and labels from any of the axes
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.03, 0.0), ncol=2)

    # Remove the individual legends as you're already doing
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    axes[2].get_legend().remove()

    # Adjust the figure layout to make room for the legend at the bottom
    fig.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.3)  # Add space at the bottom for the legend

    return fig

def visualize_learning_process(generation=0):
    df = read_data()

    payoff_ext_df = pd.DataFrame(columns=["learning_step", "ext_trait", "payoff_accumulated"])
    payoff_ext_df = payoff_ext_df.set_index(["learning_step","ext_trait"])
    for ext_trait in ["externalizers", "non-externalizers"]:
        for learning_step in range(15):
            payoff_ext_df.loc[(learning_step, ext_trait), "payoff_accumulated"] = 0
    
    visualization_df = pd.DataFrame(columns=["learning_step", "behavior", "share", "share_among_ext", "share_among_non_ext", "end_of_step_matched", "end_of_step_payoff"])
    behaviors = ["alpha", "beta", "gamma", "delta"]

    for behavior in behaviors:
        for learning_step in range(15):
            total_ext = df.loc[(generation, learning_step, 0, "shares"), pd.IndexSlice["externalizing", :, :]].sum()
            total_non_ext = df.loc[(generation, learning_step, 0, "shares"), pd.IndexSlice["non-externalizing", :, :]].sum()
            share = df.loc[(generation, learning_step, 14, "shares"),
                                pd.IndexSlice[:, behavior, :]].sum()
            share_among_ext = df.loc[
                        (generation, learning_step, 14, "shares"),
                        pd.IndexSlice["externalizing", behavior, :]
                    ].sum()/total_ext if behavior in ["alpha", "delta"] else 0
            share_among_non_ext = df.loc[(generation, learning_step, 14, "shares"),
                                pd.IndexSlice["non-externalizing", behavior, :]].sum()/total_non_ext
            end_of_step_matched = (
                df.loc[(generation, learning_step, 14, "shares"),
                pd.IndexSlice["non-externalizing", behavior, "matched"]]/
                df.loc[(generation, learning_step, 14, "shares"),
                pd.IndexSlice["non-externalizing", behavior, :]].sum()
            )
            end_of_step_payoff = (
                df.loc[
                    pd.IndexSlice[generation, learning_step, :, "payoffs"],
                    pd.IndexSlice["non-externalizing", behavior, "matched"]
                ].values * df.loc[
                    pd.IndexSlice[generation, learning_step, :, "shares"],
                    pd.IndexSlice["non-externalizing", behavior, "matched"]
                ].values + df.loc[
                    pd.IndexSlice[generation, learning_step, :, "payoffs"],
                    pd.IndexSlice["non-externalizing", behavior, "unmatched"]
                ].values * df.loc[
                    pd.IndexSlice[generation, learning_step, :, "shares"],
                    pd.IndexSlice["non-externalizing", behavior, "unmatched"]
                ].values
            ).sum()/df.loc[(generation, learning_step, 14, "shares"),
                                pd.IndexSlice["non-externalizing", behavior, :]].sum()
            payoff_ext_df.loc[(learning_step, "externalizers")] += end_of_step_payoff*share_among_ext
            payoff_ext_df.loc[(learning_step, "non-externalizers")] += end_of_step_payoff*share_among_non_ext

            visualization_df.loc[len(visualization_df)] = [
                learning_step, behavior, share, share_among_ext, share_among_non_ext, end_of_step_matched, end_of_step_payoff
            ]
    

    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    axes = axes.flatten()

    sns.lineplot(data=visualization_df, x="learning_step", y="share", hue="behavior", 
                        marker='o', ax=axes[0], palette=COLOR_DICT, markeredgewidth=0)
    sns.lineplot(data=visualization_df, x="learning_step", y="end_of_step_matched", hue="behavior",
                        marker='o', ax=axes[1], palette=COLOR_DICT, markeredgewidth=0)
    sns.lineplot(data=visualization_df, x="learning_step", y="end_of_step_payoff", hue="behavior",
                        marker='o', ax=axes[2], palette=COLOR_DICT, markeredgewidth=0)
    
    # Convert to wide format for stacked plots and reindex to match original behavior order
    pivot_share_ext = visualization_df.pivot(index='learning_step', columns='behavior', values='share_among_ext')[behaviors]
    pivot_share_non_ext = visualization_df.pivot(index='learning_step', columns='behavior', values='share_among_non_ext')[behaviors]

    # Create stacked area plots for the first three charts
    pivot_share_ext.plot.area(ax=axes[3], stacked=True, color=[COLOR_DICT[b] for b in behaviors], alpha=0.7)
    pivot_share_non_ext.plot.area(ax=axes[4], stacked=True, color=[COLOR_DICT[b] for b in behaviors], alpha=0.7)
    
    payoff_ext_df = payoff_ext_df.reset_index()
    sns.lineplot(data=payoff_ext_df, x="learning_step", y="payoff_accumulated", hue="ext_trait",
                 marker="o", ax=axes[5], palette=COLOR_DICT, markeredgewidth=0)

    for ax in axes:
        ax.set_xlabel("Learning Step")
    axes[0].set_ylabel("Share of Population")
    axes[1].set_ylabel("End of Step Matched Share")
    axes[2].set_ylabel("Accumulated Payoff Step")
    axes[3].set_ylabel("Share among Externalizers")
    axes[4].set_ylabel("Share among Non-Externalizers")
    axes[5].set_ylabel("Accumulated Payoff Learning Process")

    # Remove legends from individual plots
    for ax in axes:
        if hasattr(ax, 'get_legend') and ax.get_legend() is not None:
            ax.get_legend().remove()

    # Create custom legend elements
    handles = []
    for behavior in behaviors:
        # Line marker for line plots
        line = Line2D([0], [0], color=COLOR_DICT[behavior], marker='o', linestyle='-', linewidth=2, markersize=6)
        # Patch for stacked area plots
        patch = Patch(facecolor=COLOR_DICT[behavior], edgecolor='none', alpha=0.7)
        
        handles.append((line, behavior))
        handles.append((patch, behavior))
    
    for ext_trait in ["externalizers", "non-externalizers"]:
        line = Line2D([0], [0], color=COLOR_DICT[ext_trait], marker='o', linestyle='-', linewidth=2, markersize=6)
        handles.append((line, ext_trait))

    # Combine them into a single legend
    fig.legend(
        handles=[h[0] for h in handles], 
        labels=[h[1] for h in handles],
        ncol=5,
        bbox_to_anchor=(0.8, 0.125)
    )

    fig.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.2)

    return fig

def visualize_natural_selection_process():
    df = read_data()

    visualization_df = pd.DataFrame(columns=["generation", "ext_trait", "share"])
    ext_traits = ["externalizing", "non-externalizing"]

    for ext in ext_traits:
        for generation in range(100):
            share = df.loc[(generation, 0, 0, "shares"), pd.IndexSlice[ext, :, :]].sum()
            visualization_df.loc[len(visualization_df)] = [generation, ext.replace("ing", "ers"), share]

    sns.set_style("whitegrid")
    
    fig, ax = plt.subplots()  # Create a figure and an axes object
    sns.lineplot(data=visualization_df, x="generation", y="share", hue="ext_trait", 
                        marker='o', palette=COLOR_DICT, markeredgewidth=0, ax=ax)  # Pass ax to seaborn

    ax.set_xlabel("Learning Step")  # Set labels on the axes, not on the figure
    ax.set_ylabel("Share of Population")
    ax.legend(title=None)  # Remove legend title


    return fig


if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))

    fig1 = visualize_interaction_process(
        generation=0,
        learning_step=0
    )
    fig1.savefig(file_dir+"/../plots/fig1.png")

    fig2 = visualize_learning_process(
        generation=0
    )
    fig2.savefig(file_dir+"/../plots/fig2.png")

    fig3 = visualize_natural_selection_process()
    fig3.savefig(file_dir+"/../plots/fig3.png")
