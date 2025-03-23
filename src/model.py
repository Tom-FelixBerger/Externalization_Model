import pandas as pd
import numpy as np
import os

MODEL_PARAMS = {
    'cc': 2,                        # payoff for c vs. c ("reward")
    'dc': 3,                        # payoff for d vs. c ("temptation")
    'cd': 0,                        # payoff for c vs. d ("sucker")
    'dd': 1,                        # payoff for d vs. d ("punishment")
    'replication_k': 10,            # selection strength
    'learning_steps': 15,           # number of learning steps
    'game_rounds': 15,              # number of game rounds
    'initial_externalizers': 0.01,  # initial share of externalizers
    'generations': 100              # number of generations
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

# ensure that shares are non-negative and sum up to 1
def normalize_shares(simulation_df, generation, learning_step, game_round):
    idx = (generation, learning_step, game_round, "shares")
    
    # Get all values for the current shares row and ensure they are non-negative
    shares_values = simulation_df.loc[idx, :].values
    shares_values = np.maximum(shares_values, 0)

    # Calculate the sum of all shares
    total_shares = sum(shares_values)
    
    if total_shares != 0 and abs(total_shares - 1.0) > 1e-10:  # Only normalize if not already 1
        # Normalize all shares by dividing by the total
        normalized_shares = shares_values / total_shares
        
        # Update the dataframe with normalized values
        simulation_df.loc[idx, :] = normalized_shares
        
    return simulation_df

def play_game(simulation_df, generation, learning_step, game_round):
    idx = (generation, learning_step, game_round, "payoffs")
    simulation_df.loc[idx, ("externalizing", "alpha", "matched")] = MODEL_PARAMS['cc']
    simulation_df.loc[idx, ("externalizing", "delta", "matched")] = MODEL_PARAMS['dd']
    simulation_df.loc[idx, ("non-externalizing", "alpha", "matched")] = MODEL_PARAMS['cc']
    simulation_df.loc[idx, ("non-externalizing", "beta", "matched")] = MODEL_PARAMS['dc']
    simulation_df.loc[idx, ("non-externalizing", "gamma", "matched")] = MODEL_PARAMS['cd']
    simulation_df.loc[idx, ("non-externalizing", "delta", "matched")] = MODEL_PARAMS['dd']

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
    if game_round == MODEL_PARAMS['game_rounds']-1:
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

    normalize_shares(simulation_df, generation, learning_step, game_round+1)
        
def update_behavior(simulation_df, generation, learning_step):
    if learning_step == MODEL_PARAMS['learning_steps']-1:
        return

    idx = (generation, learning_step, MODEL_PARAMS['game_rounds']-1, "shares")
    idx_next = (generation, learning_step+1, 0, "shares")

    # aggregate payoffs during last interaction process
    aggregated_payoffs = {}
    shares = {}
    for ext_type in ["externalizing", "non-externalizing"]:
        for behavior in BEHAVIORS[ext_type]:
            shares[(ext_type, behavior)] = simulation_df.loc[idx, (ext_type, behavior, "matched")] + simulation_df.loc[idx, (ext_type, behavior, "unmatched")]
            aggregated_payoffs[(ext_type, behavior)] = (
                sum(
                    simulation_df.loc[pd.IndexSlice[generation, learning_step, :, "payoffs"], (ext_type, behavior, "matched")].values*
                    simulation_df.loc[pd.IndexSlice[generation, learning_step, :, "shares"], (ext_type, behavior, "matched")].values
                ) +
                sum(
                    simulation_df.loc[pd.IndexSlice[generation, learning_step, :, "payoffs"], (ext_type, behavior, "unmatched")].values*
                    simulation_df.loc[pd.IndexSlice[generation, learning_step, :, "shares"], (ext_type, behavior, "unmatched")].values
                )
            )/shares[(ext_type, behavior)] if shares[(ext_type, behavior)] > 0 else 0
        total_share = sum(shares[(ext_type, behavior)] for behavior in BEHAVIORS[ext_type])
        aggregated_payoffs[(ext_type, "mean")] = sum([aggregated_payoffs[(ext_type, behavior)]*shares[(ext_type, behavior)]
                                                      for behavior in BEHAVIORS[ext_type]])/total_share if total_share > 0 else 0

    # update behavior success-based
    for ext_type in ["externalizing", "non-externalizing"]:
        for behavior in BEHAVIORS[ext_type]:
            simulation_df.loc[idx_next, (ext_type, behavior, "matched")] = 0
            new_share =  shares[(ext_type, behavior)] * (1 +
                (aggregated_payoffs[(ext_type, behavior)]-aggregated_payoffs[(ext_type, "mean")]) /
                aggregated_payoffs[(ext_type, "mean")]
            ) if aggregated_payoffs[(ext_type, "mean")] > 0 else 0
            simulation_df.loc[idx_next, (ext_type, behavior, "unmatched")] = new_share
    
    normalize_shares(simulation_df, generation, learning_step+1, 0)

def update_externalization(simulation_df, generation):
    idx = (generation, MODEL_PARAMS['learning_steps']-1, MODEL_PARAMS['game_rounds']-1, "shares")
    idx_next = (generation+1, 0, 0, "shares")

    # aggregate payoffs during last interaction process
    aggregated_payoffs = {}
    shares = {}
    for ext_type in ["externalizing", "non-externalizing"]:
        shares[ext_type] = simulation_df.loc[idx, pd.IndexSlice[ext_type, :, :]].sum()
        aggregated_payoffs[ext_type] = 0
        for behavior in BEHAVIORS[ext_type]:
            aggregated_payoffs[ext_type] += (sum(
                simulation_df.loc[pd.IndexSlice[generation, :, :, "payoffs"], (ext_type, behavior, "matched")].values *
                simulation_df.loc[pd.IndexSlice[generation, :, :, "shares"], (ext_type, behavior, "matched")].values
            ) + sum(
                simulation_df.loc[pd.IndexSlice[generation, :, :, "payoffs"], (ext_type, behavior, "unmatched")].values *
                simulation_df.loc[pd.IndexSlice[generation, :, :, "shares"], (ext_type, behavior, "unmatched")].values
            ))/shares[ext_type] if shares[ext_type] > 0 else 0
    aggregated_payoffs["mean"] = aggregated_payoffs["externalizing"]*shares["externalizing"] + aggregated_payoffs["non-externalizing"]*shares["non-externalizing"]

    # update externalization trait
    for ext_type in ["externalizing", "non-externalizing"]:
        old_share = simulation_df.loc[idx, pd.IndexSlice[ext_type, :, :]].sum()
        new_share = old_share * (1 + MODEL_PARAMS["replication_k"]*(aggregated_payoffs[ext_type] - aggregated_payoffs["mean"]) / aggregated_payoffs["mean"])
        for behavior in BEHAVIORS[ext_type]:
            simulation_df.loc[idx_next, (ext_type, behavior, "unmatched")] = new_share / len(BEHAVIORS[ext_type])
            simulation_df.loc[idx_next, (ext_type, behavior, "matched")] = 0
    
    normalize_shares(simulation_df, generation+1, 0, 0)

# Run the simulation and export data
def run_simulation():
    simulation_df = initialize_dataframe()

    # natural selection
    for generation in range(MODEL_PARAMS['generations']):
        print(str(generation)+" of "+str(MODEL_PARAMS['generations']))
        # learning process
        for learning_step in range(MODEL_PARAMS['learning_steps']):

            # interaction process
            for game_round in range(MODEL_PARAMS['game_rounds']):
                play_game(simulation_df, generation, learning_step, game_round)
                update_matched(simulation_df, generation, learning_step, game_round)
            
            # learning of new interaction and partner search behavior
            ### AT SOME POINT NAN VALUES OCCUR - SET BREAK POINT AND DEBUG UPDATE BEHAVIOR
            update_behavior(simulation_df, generation, learning_step)

        # reproduction, that is update of externalization trait
        update_externalization(simulation_df, generation)
    return simulation_df

if __name__ == "__main__":
    simulation_df = run_simulation()
    file_dir = os.path.dirname(os.path.realpath(__file__))
    simulation_df.to_csv(file_dir + "/../data/base_model_simulation.csv")