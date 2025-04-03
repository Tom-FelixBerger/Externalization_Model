import pandas as pd
from numpy.random import choice
from random import sample, shuffle
import os

MODEL_PARAMS = {
    'cc': 2,                        # payoff for c vs. c ("reward")
    'dc': 3,                        # payoff for d vs. c ("temptation")
    'cd': 0,                        # payoff for c vs. d ("sucker")
    'dd': 1,                        # payoff for d vs. d ("punishment")
    'learning_steps': 15,            # number of learning steps
    "learning_mechanism": "success",# success-, frequency-, or source-based
    'game_rounds': 15,               # number of game rounds
    'pop_size': 100,                  # population size
    'initial_externalizers': 1,     # initial share of externalizers
    'generations': 50               # number of generations
}

def new_generation(ext, pop_size):
    row_values = []
    for i in range(pop_size):
        # Externalization
        row_values.append(1 if i < ext else 0)
        
        # Behavior
        if i < ext:
            row_values.append(choice(["alpha", "delta"], p=[0.5, 0.5]))
        else:
            row_values.append(choice(["alpha", "beta", "gamma", "delta"], p=[0.25]*4))
        
        # Other attributes
        row_values.append(0)  # payoff_interaction_process
        row_values.append(0)  # payoff_learning_process
        row_values.append("unpartnered")  # current_partner
    
    return row_values

def initialize_dataframe():
    pop_size = MODEL_PARAMS['pop_size']
    
    # Creating the multi-index columns with proper ordering
    cols = [("generation", ""), ("learning_step", ""), ("game_round", "")]
    
    # For each agent, add all its attributes in order
    for i in range(pop_size):
        for attr in ["externalization", "behavior", "payoff_interaction_process", 
                     "payoff_learning_process", "current_partner"]:
            cols.append((i, attr))
    
    multi_index = pd.MultiIndex.from_tuples(cols)
    simulation_df = pd.DataFrame(columns=multi_index)
    simulation_df = simulation_df.set_index(["generation", "learning_step", "game_round"])
    
    init_ext = MODEL_PARAMS['initial_externalizers']
    
    simulation_df.loc[(0, 0, 0)] = new_generation(init_ext, pop_size)
    
    return simulation_df

def find_partner(simulation_df, generation, learning_step, game_round):
    idx = (generation, learning_step, game_round)
    unpartnered = [i for i in range(MODEL_PARAMS['pop_size']) if simulation_df.loc[idx, (i, "current_partner")] == "unpartnered"]

    if unpartnered:
        shuffle(unpartnered)
        for pair in range(0, len(unpartnered), 2):
            simulation_df.loc[idx, (unpartnered[pair], "current_partner")] = unpartnered[pair + 1]
            simulation_df.loc[idx, (unpartnered[pair + 1], "current_partner")] = unpartnered[pair]

def play_game(simulation_df, generation, learning_step, game_round):
    idx = (generation, learning_step, game_round)
    next_idx = (generation, learning_step, game_round + 1)

    must_still_play = [i for i in range(MODEL_PARAMS['pop_size'])]
    while not must_still_play == []:

        agent = must_still_play[0]
        current_partner = simulation_df.loc[idx, (agent, "current_partner")]
        must_still_play.remove(agent)
        must_still_play.remove(current_partner)

        agent_behavior = simulation_df.loc[idx, (agent, "behavior")]
        partner_behavior = simulation_df.loc[idx, (current_partner, "behavior")]
        
        # Determine payoffs based on behaviors
        if agent_behavior == "alpha" or agent_behavior == "gamma": # agent cooperates
            if partner_behavior == "alpha" or partner_behavior == "gamma": # partner cooperates
                payoff = MODEL_PARAMS['cc']
                partner_payoff = MODEL_PARAMS['cc']
            else: # partner defects
                payoff = MODEL_PARAMS['cd']
                partner_payoff = MODEL_PARAMS['dc']
        else: # agent defects
            if partner_behavior == "alpha" or partner_behavior == "gamma": # partner cooperates
                payoff = MODEL_PARAMS['dc']
                partner_payoff = MODEL_PARAMS['cd']
            else: # partner defects
                payoff = MODEL_PARAMS['dd']
                partner_payoff = MODEL_PARAMS['dd']
        
        simulation_df.loc[idx, (agent, "payoff_interaction_process")] += payoff
        simulation_df.loc[idx, (current_partner, "payoff_interaction_process")] += partner_payoff

        # Determine whether partnership is stable
        if idx[2] != MODEL_PARAMS['game_rounds'] - 1:
            simulation_df.loc[next_idx, pd.IndexSlice[agent, :]] = simulation_df.loc[idx, pd.IndexSlice[agent, :]]
            simulation_df.loc[next_idx, pd.IndexSlice[current_partner, :]] = simulation_df.loc[idx, pd.IndexSlice[current_partner, :]]
            if not (
                (agent_behavior == "alpha" and partner_behavior == "alpha") or \
                (agent_behavior == "beta" and partner_behavior == "gamma") or \
                (agent_behavior == "gamma" and partner_behavior == "beta") or \
                (agent_behavior == "delta" and partner_behavior == "delta")
            ):
                simulation_df.loc[next_idx, (agent, "current_partner")] = "unpartnered"
                simulation_df.loc[next_idx, (current_partner, "current_partner")] = "unpartnered"

def update_behavior(simulation_df, generation, learning_step):
    idx = (generation, learning_step, MODEL_PARAMS['game_rounds'] - 1)
    next_idx = (generation, learning_step + 1, 0)
    pop_size = MODEL_PARAMS['pop_size']

    all_ranking = []
    ext_ranking = []
    for agent in range(pop_size):
        fitness = simulation_df.loc[idx, (agent, "payoff_interaction_process")]
        behavior = simulation_df.loc[idx, (agent, "behavior")]
        
        all_ranking.append((fitness, behavior))
        if behavior in ["alpha", "delta"]:
            ext_ranking.append((fitness, behavior))
        
    all_ranking.sort(reverse=True, key=lambda x: x[0])
    if MODEL_PARAMS["learning_mechanism"] == "success":
        all_probab = [len(all_ranking) - i for i in range(len(all_ranking))]
        all_probab = [i / sum(all_probab) for i in all_probab]
    elif MODEL_PARAMS["learning_mechanism"] == "frequency":
        all_probab = [1 / len(all_ranking) for _ in range(len(all_ranking))]
    elif MODEL_PARAMS["learning_mechanism"] == "source":
        all_probab = [0] * len(all_ranking)
        for sources in sample(range(len(all_ranking)), 4):
            all_probab[sources] = 1/4
    
    if ext_ranking:
        ext_ranking.sort(reverse=True, key=lambda x: x[0])
        if MODEL_PARAMS["learning_mechanism"] == "success":
            ext_probab = [len(ext_ranking) - i for i in range(len(ext_ranking))]
            ext_probab = [i / sum(ext_probab) for i in ext_probab]
        elif MODEL_PARAMS["learning_mechanism"] == "frequency":
            ext_probab = [1 / len(ext_ranking) for _ in range(len(ext_ranking))]
        elif MODEL_PARAMS["learning_mechanism"] == "source":
            ext_probab = [0] * len(ext_ranking)
            no_sources = min(len(ext_ranking), 4)
            for sources in sample(range(len(ext_ranking)), no_sources):
                ext_probab[sources] = 1/no_sources
    
    for agent in range(MODEL_PARAMS["pop_size"]):
        if simulation_df.loc[idx, (agent, "externalization")] == 1:
            new_behavior = choice([i[1] for i in ext_ranking], p=ext_probab)
        else:
            new_behavior = choice([i[1] for i in all_ranking], p=all_probab)
        
        simulation_df.loc[next_idx, (agent, "behavior")] = new_behavior
        simulation_df.loc[next_idx, (agent, "payoff_learning_process")] = (
            simulation_df.loc[idx, (agent, "payoff_learning_process")] +
            simulation_df.loc[idx, (agent, "payoff_interaction_process")]
        )
        simulation_df.loc[next_idx, (agent, "payoff_interaction_process")] = 0
        simulation_df.loc[next_idx, (agent, "current_partner")] = "unpartnered"
        simulation_df.loc[next_idx, (agent, "externalization")] = simulation_df.loc[idx, (agent, "externalization")]

def update_externalization(simulation_df, generation):
    idx = (generation, MODEL_PARAMS['learning_steps'] - 1, MODEL_PARAMS['game_rounds'] - 1)
    next_idx = (generation + 1, 0, 0)
    pop_size = MODEL_PARAMS['pop_size']
    
    ranking = [
        (simulation_df.loc[idx, (i, "payoff_learning_process")], 
         simulation_df.loc[idx, (i, "externalization")])
        for i in range(pop_size)
    ]
    ranking.sort(reverse=True, key=lambda x: x[0])
    top_half_ext = sum([i[1] for i in ranking[0:int(len(ranking)/2)]])
    new_ext = top_half_ext*2
    simulation_df.loc[next_idx] = new_generation(new_ext, pop_size)
    
    converged = ((simulation_df.loc[next_idx, pd.IndexSlice[:, "externalization"]].sum() == pop_size) or
                (simulation_df.loc[next_idx, pd.IndexSlice[:, "externalization"]].sum() == 0))
    return converged

# Run the simulation and export data
def run_simulation(simulation_number):
    simulation_df = initialize_dataframe()

    # natural selection
    for generation in range(MODEL_PARAMS['generations']):
        if generation%10 == 0:
            print(f"Generation {generation+1} of {MODEL_PARAMS['generations']}")
        # learning process
        for learning_step in range(MODEL_PARAMS['learning_steps']):

            # interaction process
            for game_round in range(MODEL_PARAMS['game_rounds']):
                find_partner(simulation_df, generation, learning_step, game_round)
                play_game(simulation_df, generation, learning_step, game_round)

            # learning of new interaction and partner search behavior
            update_behavior(simulation_df, generation, learning_step)

        # reproduction, that is update of externalization trait
        converged = update_externalization(simulation_df, generation)
        if converged:
            break
    
    # transform the simulation dataframe for saving of results
    results_df = simulation_df.copy()
    results_df = results_df.reset_index()
    results_df["simulation_number"] = simulation_number
    results_df = results_df.set_index(["simulation_number", "generation", "learning_step"])
    
    # drop all irrelevant columns and rows for space reasons
    results_df = results_df[results_df["game_round"] == 0]
    results_df = results_df.loc[
        # Either: generation 0 AND learning step between 0-14
        ((results_df.index.get_level_values('generation') == 0) & 
        (results_df.index.get_level_values('learning_step') < MODEL_PARAMS['learning_steps'])) |
        # Or: generation > 0 AND learning step = 0
        ((results_df.index.get_level_values('generation') > 0) & 
        (results_df.index.get_level_values('learning_step') == 0))
    ]
    results_df = results_df.drop(columns=["game_round"], level=0)
    results_df = results_df.drop(columns=[c for c in results_df.columns if not c[1] in ["externalization", "behavior"]])

    return results_df

if __name__ == "__main__":
    number_simulations = 100

    print(f"Running simulation 1 of {number_simulations}")
    results_df = run_simulation(0)
    for i in range(1, number_simulations):
        print(f"Running simulation {i+1} of {number_simulations}")
        results_df = pd.concat([results_df, run_simulation(i)], axis=0)

    file_dir = os.path.dirname(os.path.realpath(__file__))
    results_df.to_csv(file_dir + "/../data/ABM_base_simulation.csv")