
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from model import initialize_dataframe



def visualize_interaction_process(csv_file, generation=0, learning_step=0):
    # Read the CSV file
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

    # Filter data for the specified generation and learning step
    filtered_df = df.loc[pd.IndexSlice[generation, learning_step, :, :], :]
    filtered_df.index = filtered_df.index.remove_unused_levels()
    game_rounds = sorted(filtered_df.index.levels[2])

    # Define strategy names for easier reference
    strategies = [
        ("externalizing", "alpha"),
        ("externalizing", "delta"),
        ("non-externalizing", "alpha"),
        ("non-externalizing", "beta"),
        ("non-externalizing", "gamma"),
        ("non-externalizing", "delta")
    ]

    # Set up a consistent color palette for strategies
    strategy_colors = ["green", "orange", "green", "red", "blue", "orange"]
    strategy_grid_place = [4, 7, 0, 1, 2, 3]

    # Set the style for the plots
    sns.set_style("whitegrid")

    # Create figure 1: Matched share over game rounds
    fig1, axes1 = plt.subplots(2, 4, figsize=(15, 10))
    fig1.suptitle(f'Development of Matched Share - Generation {generation}, Learning Step {learning_step}', 
                    fontsize=16)
    axes1 = axes1.flatten()

    # Create figure 2: Payoffs over game rounds
    fig2, axes2 = plt.subplots(2, 4, figsize=(15, 10))
    fig2.suptitle(f'Development of Payoffs - Generation {generation}, Learning Step {learning_step}', 
                    fontsize=16)
    axes2 = axes2.flatten()

    # Plot data for each strategy
    for (strategy, color, i) in zip(strategies, strategy_colors, strategy_grid_place):
        # Extract data for this strategy
        matched_shares = []
        payoffs = []
        
        for round_num in game_rounds:
            # Get data for this round
            shares_data = filtered_df.loc[pd.IndexSlice[:, :, round_num, "shares"]]
            payoffs_data = filtered_df.loc[pd.IndexSlice[:, :, round_num, "payoffs"]]
            
            matched_share = shares_data[(strategy[0], strategy[1], "matched")].values[0]
            matched_shares.append(matched_share)
            matched_payoff = payoffs_data[(strategy[0], strategy[1], "matched")].values[0]
            
            unmatched_share = shares_data[(strategy[0], strategy[1], "unmatched")].values[0]
            unmatched_payoff = payoffs_data[(strategy[0], strategy[1], "unmatched")].values[0]
            
            # Calculate weighted average payoff
            total_share = matched_share + unmatched_share
            if total_share > 0:
                avg_payoff = (matched_share * matched_payoff + unmatched_share * unmatched_payoff) / total_share
            else:
                avg_payoff = 0
            
            payoffs.append(avg_payoff)
        
        # Plot matched share
        axes1[i].plot(game_rounds, matched_shares, 'o-', color=color)
        axes1[i].set_title(f"{strategy}")
        axes1[i].set_xlabel("Game Round")
        axes1[i].set_ylabel("Matched Share")

        # Plot payoffs
        axes2[i].plot(game_rounds, payoffs, 'o-', color=color)
        axes2[i].set_title(f"{strategy}")
        axes2[i].set_xlabel("Game Round")
        axes2[i].set_ylabel("Payoff")
        axes2[i].set_ylim(-0.1, 3.1)
        
    # Adjust layout and show plots
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.tight_layout(rect=[0, 0, 1, 0.95])

    return fig1, fig2

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))
    fig1, fig2 = visualize_interaction_process(
        file_dir+"/../data/base_model_simulation.csv",
        generation=0,
        learning_step=0
    )
    fig1.savefig(file_dir+"/../plots/interaction_process_matched_gen0_lst0.png")
    fig2.savefig(file_dir+"/../plots/interaction_process_payoffs_gen0_lst0.png")