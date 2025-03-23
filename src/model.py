import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

MODEL_PARAMS = {
    'cc': 2,                        # payoff for c vs. c ("reward")
    'dc': 3,                        # payoff for d vs. c ("temptation")
    'cd': 0,                        # payoff for c vs. d ("sucker")
    'dd': 1,                        # payoff for d vs. d ("punishment")
    'replication_k': 10,            # selection strength
    'learning_steps': 15,           # number of learning steps
    'game_rounds': 15,              # number of game rounds
    'initial_externalizers': 0.01,  # initial share of externalizers
    'generations': 3                # number of generations
}

BEHAVIORS = {
    "externalizing": ["alpha", "delta"],
    "non-externalizing": ["alpha", "beta", "gamma", "delta"]
}

# Initialize the dataframe to store the simulation data in
def initialize_dataframe():
    init_ext = MODEL_PARAMS['initial_externalizers']
    initial_data = {
        ("generation", "", ""): [0],
        ("learning_step", "", ""): [0],
        ("game_round", "", ""): [0],
        ("metric", "", ""): ["shares"],
        ("externalizing", "alpha", "matched"): [0],
        ("externalizing", "alpha", "unmatched"): [init_ext/2],
        ("externalizing", "delta", "matched"): [0],
        ("externalizing", "delta", "unmatched"): [init_ext/2],
        ("non-externalizing", "alpha", "matched"): [0],
        ("non-externalizing", "alpha", "unmatched"): [(1-init_ext)/4],
        ("non-externalizing", "beta", "matched"): [0],
        ("non-externalizing", "beta", "unmatched"): [(1-init_ext)/4],
        ("non-externalizing", "gamma", "matched"): [0],
        ("non-externalizing", "gamma", "unmatched"): [(1-init_ext)/4],
        ("non-externalizing", "delta", "matched"): [0],
        ("non-externalizing", "delta", "unmatched"): [(1-init_ext)/4]
    }
    simulation_df = pd.DataFrame(initial_data)
    simulation_df = simulation_df.set_index(["generation", "learning_step", "game_round", "metric"])
    return simulation_df

def play_game(simulation_df, generation, learning_step, game_round):
    idx = (generation, learning_step, game_round, "payoffs")
    simulation_df.loc[idx, ("externalizing", "alpha", "matched")] = MODEL_PARAMS['cc']
    simulation_df.loc[idx, ("externalizing", "delta", "matched")] = MODEL_PARAMS['dd']
    simulation_df.loc[idx, ("non-externalizing", "alpha", "matched")] = MODEL_PARAMS['cc']
    simulation_df.loc[idx, ("non-externalizing", "beta", "matched")] = MODEL_PARAMS['dc']
    simulation_df.loc[idx, ("externalizing", "gamma", "matched")] = MODEL_PARAMS['cd']
    simulation_df.loc[idx, ("externalizing", "delta", "matched")] = MODEL_PARAMS['dd']

    c_unmatched = sum(
        simulation_df.loc[(generation, learning_step, game_round, "shares"), [
            ("externalizing", "alpha", "unmatched"),
            ("non-externalizing", "alpha", "unmatched"),
            ("non-externalizing", "gamma", "unmatched")
        ]])
    d_unmatched = sum(
        simulation_df.loc[(generation, learning_step, game_round, "shares"), [
            ("externalizing", "delta", "unmatched"),
            ("non-externalizing", "beta", "unmatched"),
            ("non-externalizing", "delta", "unmatched")
        ]])
    if c_unmatched + d_unmatched == 0:
        prob_c = 0
        prob_d = 0
    else:
        prob_c = c_unmatched/(c_unmatched + d_unmatched)
        prob_d = d_unmatched/(c_unmatched + d_unmatched)
    
    simulation_df.loc[idx, ("externalizing", "alpha", "unmatched")] = MODEL_PARAMS['cc']*prob_c + MODEL_PARAMS['cd']*prob_d
    simulation_df.loc[idx, ("externalizing", "delta", "unmatched")] = MODEL_PARAMS['dc']*prob_c + MODEL_PARAMS['dd']*prob_d
    simulation_df.loc[idx, ("non-externalizing", "alpha", "unmatched")] = MODEL_PARAMS['cc']*prob_c + MODEL_PARAMS['cd']*prob_d
    simulation_df.loc[idx, ("non-externalizing", "beta", "unmatched")] = MODEL_PARAMS['dc']*prob_c + MODEL_PARAMS['dd']*prob_d
    simulation_df.loc[idx, ("non-externalizing", "gamma", "unmatched")] = MODEL_PARAMS['cc']*prob_c + MODEL_PARAMS['cd']*prob_d
    simulation_df.loc[idx, ("non-externalizing", "delta", "unmatched")] = MODEL_PARAMS['dc']*prob_c + MODEL_PARAMS['dd']*prob_d

def update_matched(simulation_df, generation, learning_step, game_round):
    if game_round == MODEL_PARAMS['game_rounds']:
        return
    
    idx = (generation, learning_step, game_round, "shares")
    idx_next = (generation, learning_step, game_round+1, "shares")

    # Calculate probabilities among unmatched
    sum_unmatched = sum(simulation_df.loc[idx, [
        ("externalizing", "alpha", "unmatched"),
        ("non-externalizing", "alpha", "unmatched"),
        ("non-externalizing", "beta", "unmatched"),
        ("non-externalizing", "gamma", "unmatched"),
        ("non-externalizing", "delta", "unmatched")
    ]])
    if sum_unmatched > 0:
        prob = {
            "alpha": sum(simulation_df.loc[idx, [
                ("externalizing", "alpha", "unmatched"),
                ("non-externalizing", "alpha", "unmatched")
            ]])/sum_unmatched,
            "beta": simulation_df.loc[idx, ("non-externalizing", "beta", "unmatched")]/sum_unmatched,
            "gamma": simulation_df.loc[idx, ("non-externalizing", "gamma", "unmatched")]/sum_unmatched,
            "delta": sum(simulation_df.loc[idx, [
                ("externalizing", "delta", "unmatched"),
                ("non-externalizing", "delta", "unmatched")
            ]])/sum_unmatched
        }
    else:
        prob = {
            "alpha": 0,
            "beta": 0,
            "gamma": 0,
            "delta": 0
        }
    
    matching_pairs = [
        ("externalizing", "alpha", "alpha"),
        ("externalizing", "delta", "delta"),
        ("non-externalizing", "alpha", "alpha"),
        ("non-externalizing", "beta", "gamma"),
        ("non-externalizing", "gamma", "beta"),
        ("non-externalizing", "delta", "delta")
    ]
    
    # Update matched and unmatched values
    for ext_type, behavior, match_prob_key in matching_pairs:
        current_unmatched = simulation_df.loc[idx, (ext_type, behavior, "unmatched")]
        current_matched = simulation_df.loc[idx, (ext_type, behavior, "matched")]
        
        newly_matched = prob[match_prob_key] * current_unmatched

        simulation_df.loc[idx_next, (ext_type, behavior, "matched")] = current_matched + newly_matched
        simulation_df.loc[idx_next, (ext_type, behavior, "unmatched")] = current_unmatched - newly_matched

def update_behavior(simulation_df, generation, learning_step):
    if learning_step == MODEL_PARAMS['learning_steps']:
        return

    idx = (generation, learning_step, MODEL_PARAMS['game_rounds'], "shares")
    idx_next = (generation, learning_step+1, 0, "shares")

    # aggregate payoffs during last interaction process
    aggregated_payoffs = {}
    for ext_type in ["externalizing", "non-externalizing"]:
        for behavior in BEHAVIORS[ext_type]:
            aggregated_payoffs[(ext_type, behavior)] = (
                sum(
                    simulation_df.loc[pd.IndexSlice[generation, learning_step, :, "payoffs"], (ext_type, behavior, "matched")]*
                    simulation_df.loc[pd.IndexSlice[generation, learning_step, :, "shares"], (ext_type, behavior, "matched")]
                ) +
                sum(
                    simulation_df.loc[pd.IndexSlice[generation, learning_step, :, "payoffs"], (ext_type, behavior, "unmatched")]*
                    simulation_df.loc[pd.IndexSlice[generation, learning_step, :, "shares"], (ext_type, behavior, "unmatched")]
                )
            )
        aggregated_payoffs[(ext_type, "total")] = sum([aggregated_payoffs[(ext_type, behavior)] for behavior in BEHAVIORS[ext_type]])

    # update behavior success-based
    for ext_type in ["externalizing", "non-externalizing"]:
        for behavior in BEHAVIORS[ext_type]:
            simulation_df.loc[idx_next, (ext_type, behavior, "matched")] = 0

            simulation_df.loc[idx_next, (ext_type, behavior, "unmatched")] = (
                simulation_df.loc[idx, (ext_type, behavior, "unmatched")] +
                simulation_df.loc[idx, (ext_type, behavior, "matched")]
            ) * (
                aggregated_payoffs[(ext_type, behavior)]-aggregated_payoffs[(ext_type, "total")] /
                aggregated_payoffs[(ext_type, "total")]
            )

def update_externalization(simulation_df, generation):
    idx = (generation, MODEL_PARAMS['learning_steps'], MODEL_PARAMS['game_rounds'], "shares")
    idx_next = (generation+1, 0, 0, "shares")

    # aggregate payoffs during last interaction process
    aggregated_payoffs = {}
    for ext_type in ["externalizing", "non-externalizing"]:
        aggregated_payoffs[ext_type] = 0
        for behavior in BEHAVIORS[ext_type]:
            aggregated_payoffs[ext_type] += (sum(
                simulation_df.loc[pd.IndexSlice[generation, :, :, "payoffs"], (ext_type, behavior, "matched")] *
                simulation_df.loc[pd.IndexSlice[generation, :, :, "shares"], (ext_type, behavior, "matched")]
            ) + sum(
                simulation_df.loc[pd.IndexSlice[generation, :, :, "payoffs"], (ext_type, behavior, "unmatched")] *
                simulation_df.loc[pd.IndexSlice[generation, :, :, "shares"], (ext_type, behavior, "unmatched")]
            ))
    aggregated_payoffs["total"] = sum(aggregated_payoffs.values())

    # update externalization trait
    for ext_type in ["externalizing", "non-externalizing"]:
        old_share = simulation_df.loc[idx, pd.IndexSlice[ext_type, :, :]].sum()
        new_share = old_share * (1 + aggregated_payoffs[ext_type] - aggregated_payoffs["total"]) / aggregated_payoffs["total"]
        for behavior in BEHAVIORS[ext_type]:
            simulation_df.loc[idx_next, (ext_type, behavior, "unmatchded")] = new_share / len(BEHAVIORS[ext_type])
            simulation_df.loc[idx_next, (ext_type, behavior, "matched")] = 0

# Run the simulation and export data
def run_simulation():
    simulation_df = initialize_dataframe()

    # natural selection
    for generation in range(1, MODEL_PARAMS['generations'] + 1):
        # learning process
        for learning_step in range(1, MODEL_PARAMS['learning_steps'] + 1):

            # interaction process
            for game_round in range(1, MODEL_PARAMS['game_rounds'] + 1):
                play_game(simulation_df, generation, learning_step, game_round)
                update_matched(simulation_df)
            
            # learning of new interaction and partner search behavior
            update_behavior(simulation_df)

        # reproduction, that is update of externalization trait
        update_externalization(simulation_df)

    simulation_df.to_csv("../data/base_model_simulation.csv", index=False)

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




# To run the simulation:
run_simulation()
visualize_interaction_process("../data/base_model_simulation.csv")