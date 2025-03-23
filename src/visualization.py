
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model import initialize_dataframe

def visualize_interaction_process(csv_file, generation=1, learning_step=1):
    # Read the CSV file
    print(f"Reading data from {csv_file}...")
    df = pd.read_csv(csv_file)

    # Convert the flat column format back to multi-index
    df.columns = initialize_dataframe().columns
        
    # Convert index columns to numeric types
    df[("generation", "", "", "")] = pd.to_numeric(df[("generation", "", "", "")])
    df[("learning_step", "", "", "")] = pd.to_numeric(df[("learning_step", "", "", "")])
    df[("game_round", "", "", "")] = pd.to_numeric(df[("game_round", "", "", "")])
    
    # Make sure all data columns are numeric as well
    for col in df.columns:
        if col[2] in ["matched", "unmatched"] and col[3] in ["share", "payoff"]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter data for the specified generation and learning step
    filtered_df = df[(df[("generation", "", "", "")] == generation) & 
                     (df[("learning_step", "", "", "")] == learning_step)]
    
    if filtered_df.empty:
        print(f"No data found for generation {generation}, learning step {learning_step}")
        return
    
    print(f"Visualizing interaction process for generation {generation}, learning step {learning_step}...")
    
    # Get the game rounds
    game_rounds = sorted(filtered_df[("game_round", "", "", "")].unique())
    
    # Define strategy names for easier reference
    strategies = [
        ("externalizing", "alpha"),
        ("externalizing", "delta"),
        ("non-externalizing", "alpha"),
        ("non-externalizing", "beta"),
        ("non-externalizing", "gamma"),
        ("non-externalizing", "delta")
    ]
    
    strategy_labels = [
        "Ext-Alpha",
        "Ext-Delta",
        "Non-Ext-Alpha",
        "Non-Ext-Beta",
        "Non-Ext-Gamma",
        "Non-Ext-Delta"
    ]
    
    # Set up a consistent color palette for strategies
    strategy_colors = sns.color_palette("husl", len(strategies))
    
    # Set the style for the plots
    sns.set_style("whitegrid")
    
    # Create figure 1: Matched share over game rounds
    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))
    fig1.suptitle(f'Development of Matched Share - Generation {generation}, Learning Step {learning_step}', 
                  fontsize=16)
    axes1 = axes1.flatten()
    
    # Create figure 2: Payoffs over game rounds
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    fig2.suptitle(f'Development of Payoffs - Generation {generation}, Learning Step {learning_step}', 
                  fontsize=16)
    axes2 = axes2.flatten()
    
    # Plot data for each strategy
    for i, (strategy, label, color) in enumerate(zip(strategies, strategy_labels, strategy_colors)):
        # Extract data for this strategy
        matched_shares = []
        payoffs = []
        
        for round_num in game_rounds:
            # Get data for this round
            round_data = filtered_df[filtered_df[("game_round", "", "", "")] == round_num]
            
            if not round_data.empty:
                # Extract matched share for this strategy and round
                matched_share = round_data[(strategy[0], strategy[1], "matched", "share")].values[0]
                matched_shares.append(matched_share)
                
                # Extract payoff data
                matched_payoff = round_data[(strategy[0], strategy[1], "matched", "payoff")].values[0]
                unmatched_share = round_data[(strategy[0], strategy[1], "unmatched", "share")].values[0]
                unmatched_payoff = round_data[(strategy[0], strategy[1], "unmatched", "payoff")].values[0]
                
                # Calculate weighted average payoff
                total_share = matched_share + unmatched_share
                if total_share > 0:
                    avg_payoff = (matched_share * matched_payoff + unmatched_share * unmatched_payoff) / total_share
                else:
                    avg_payoff = 0
                
                payoffs.append(avg_payoff)
        
        # Plot matched share
        axes1[i].plot(game_rounds, matched_shares, 'o-', color=color, label=label)
        axes1[i].set_title(f"{label}")
        axes1[i].set_xlabel("Game Round")
        axes1[i].set_ylabel("Matched Share")
        if matched_shares:  # Check if list is not empty
            max_value = max(matched_shares)
            axes1[i].set_ylim(0, max(max_value * 1.1, 0.1))  # Set reasonable y-axis limits
        
        # Plot payoffs
        axes2[i].plot(game_rounds, payoffs, 'o-', color=color, label=label)
        axes2[i].set_title(f"{label}")
        axes2[i].set_xlabel("Game Round")
        axes2[i].set_ylabel("Average Payoff")
        if payoffs and max(payoffs) > 0:  # Check if list is not empty and has positive values
            axes2[i].set_ylim(0, max(payoffs) * 1.1)  # Set reasonable y-axis limits
        
    # Adjust layout and show plots
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.show()

visualize_interaction_process("../data/base_model_simulation.csv")