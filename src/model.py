import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize the simulation dataframe with multi-index as specified
def initialize_dataframe():
    multi_index = pd.MultiIndex.from_tuples([
        ("generation", "", "", ""),
        ("learning_step", "", "", ""),
        ("game_round", "", "", ""),
        ("externalizing", "alpha", "matched", "share"),
        ("externalizing", "alpha", "matched", "payoff"),
        ("externalizing", "alpha", "unmatched", "share"),
        ("externalizing", "alpha", "unmatched", "payoff"),
        ("externalizing", "delta", "matched", "share"),
        ("externalizing", "delta", "matched", "payoff"),
        ("externalizing", "delta", "unmatched", "share"),
        ("externalizing", "delta", "unmatched", "payoff"),
        ("non-externalizing", "alpha", "matched", "share"),
        ("non-externalizing", "alpha", "matched", "payoff"),
        ("non-externalizing", "alpha", "unmatched", "share"),
        ("non-externalizing", "alpha", "unmatched", "payoff"),
        ("non-externalizing", "beta", "matched", "share"),
        ("non-externalizing", "beta", "matched", "payoff"),
        ("non-externalizing", "beta", "unmatched", "share"),
        ("non-externalizing", "beta", "unmatched", "payoff"),
        ("non-externalizing", "gamma", "matched", "share"),
        ("non-externalizing", "gamma", "matched", "payoff"),
        ("non-externalizing", "gamma", "unmatched", "share"),
        ("non-externalizing", "gamma", "unmatched", "payoff"),
        ("non-externalizing", "delta", "matched", "share"),
        ("non-externalizing", "delta", "matched", "payoff"),
        ("non-externalizing", "delta", "unmatched", "share"),
        ("non-externalizing", "delta", "unmatched", "payoff"),
    ])
    simulation_df = pd.DataFrame(columns=multi_index)
    return simulation_df

# initializes the dataframe holding the information about the interaction phase, i.e. the strategies,
# their proportions of partnered ('married') and available players, and the payoffs they can receive
def init_gameplay(strat_df, model_params):
    gameplay_df = strat_df.copy() 
    
    gameplay_df = gameplay_df.drop('payoff_learn_st', axis=1)
    gameplay_df['available'] = gameplay_df['total']
    gameplay_df['married'] = [0]*6
    
    gameplay_df['payoff_vs_c'] = (model_params['tem']*(gameplay_df['game_strat']=='d') +
                                  model_params['rew']*(gameplay_df['game_strat']=='c'))
    gameplay_df['payoff_vs_d'] = (model_params['pun']*(gameplay_df['game_strat']=='d') +
                                  model_params['suc']*(gameplay_df['game_strat']=='c'))
    
    return gameplay_df

# calculate probability for available strategies to play against cooperating or defecting player
def prob_c_d(gameplay_df, sum_av): # sum_av is the sum of available players, might be rounded to 0
    
    prob_coop, prob_defc = (0,0)
    if sum_av > 0:
        prob_coop = sum(gameplay_df[gameplay_df['game_strat']=='c']['available'])/sum_av
        prob_defc = sum(gameplay_df[gameplay_df['game_strat']=='d']['available'])/sum_av
        
    return (prob_coop, prob_defc)

# calculate expected payoff for the available share of strategies
def payoff_available(gameplay_df):
    
    sum_av = sum(gameplay_df['available'])
    prob_coop, prob_defc = prob_c_d(gameplay_df, sum_av)
    
    result = ((gameplay_df['payoff_vs_c']*prob_coop + 
                gameplay_df['payoff_vs_d']*prob_defc) *
                gameplay_df['available']/gameplay_df['total'])
    
    result = result.fillna(0) # handle NaN values due to total values of 0
    return result

# calculate expected payoff for the partnered share of strategies
def payoff_married(gameplay_df):
    result = ((gameplay_df['payoff_vs_c']*(gameplay_df['partner_strat']=='c') + 
                gameplay_df['payoff_vs_d']*(gameplay_df['partner_strat']=='d')) *
                gameplay_df['married']/gameplay_df['total'])
    
    result = result.fillna(0) # handle NaN values due to total values of 0
    return result

# calculate payoff for each strategy
def payoffs_game(gameplay_df):
    gameplay_df = gameplay_df.copy() 
    
    gameplay_df['payoff_married'] = payoff_married(gameplay_df)
    gameplay_df['payoff_available'] = payoff_available(gameplay_df)
    result = pd.DataFrame(gameplay_df['payoff_married'] + gameplay_df['payoff_available'],
                          columns=['payoffs_game'])
    return result['payoffs_game']

# function to calculate new proportions of partnered ('married') strategies in population
def marry(gameplay_df):
    gameplay_df = gameplay_df.copy() 
    
    sum_av = sum(gameplay_df['available']) # might be rounded to 0
    if sum_av > 0:
        
        # for each strategy, calculate probability to meet respective prefered partner
        strat_names = ['alpha', 'beta', 'gamma', 'delta']
        pref = {'alpha': 'alpha', 'beta': 'gamma', 'gamma': 'beta', 'delta': 'delta'}
        prob = dict((key, sum(gameplay_df.loc[pd.IndexSlice[:, key], 'available'])/sum_av)
                    for key in strat_names)

        # add expected partnering players ('just_ married') to each strategy's married share
        for key in strat_names:
            gameplay_df.loc[pd.IndexSlice[:, key], 'just_married'] = (
                gameplay_df.loc[pd.IndexSlice[:, key], 'available']*prob[pref[key]])
        gameplay_df['available'] += -gameplay_df['just_married']
        gameplay_df['married'] += gameplay_df['just_married']
    
    return gameplay_df[['available', 'married']]

# due to rounding and division by very small values, available and partnered players might
# not add up to the total strategy share and therefore need to be readjusted
def adjust_gameplay(gameplay_df):
    gameplay_df = gameplay_df.copy()
    
    gameplay_df['available'] = gameplay_df['available'] * (gameplay_df['available']>0)
    gameplay_df['married'] = gameplay_df['married'] * (gameplay_df['married'] > 0)
    scale_factor = gameplay_df['total']/(gameplay_df['available']+gameplay_df['married'])
    gameplay_df['available'] *= scale_factor
    gameplay_df['married'] *= scale_factor
    gameplay_df = gameplay_df.fillna(0) # handle NaN values due to total values of 0
    return gameplay_df

# strategies of learning process are initialized, either uniformly or according
# to the last generation's learning process result, if similarity parameter sim > 0
def init_strat(xn_df, sim, last_strat_df=None):
    strat_df = pd.DataFrame(columns=['externalization', 'game_strat', 'partner_strat',
                                      'name_strat', 'total', 'payoff_learn_st'])
    
    if (last_strat_df is None) or (sim == 0):
        # divide share of externalizers by 2 and non-externalizers by 4 for strategy shares
        strat_df.loc[0] = ['x', 'c', 'c', 'alpha', xn_df.loc['x', 'total']/2, 0]
        strat_df.loc[1] = ['x', 'd', 'd', 'delta', xn_df.loc['x', 'total']/2, 0]
        strat_df.loc[2] = ['n', 'c', 'c', 'alpha', xn_df.loc['n', 'total']/4, 0]
        strat_df.loc[3] = ['n', 'd', 'c', 'beta', xn_df.loc['n', 'total']/4, 0]
        strat_df.loc[4] = ['n', 'c', 'd', 'gamma', xn_df.loc['n', 'total']/4, 0]
        strat_df.loc[5] = ['n', 'd', 'd', 'delta', xn_df.loc['n', 'total']/4, 0]
    else:
        # calculate strat shares as sim*last_gen_share/sum_available_shares
                                    #  + (1-sim)*strat_share_uniformly
        
        # sum up last generation's strat shares (externalizing and non-externalizing)
        strat_names = ['alpha', 'beta', 'gamma', 'delta']
        last_strat_shares = [sum(last_strat_df.loc[pd.IndexSlice[:, strat], 'total'])
                             for strat in strat_names]
        
        # for externalizers only alpha and delta are available
        sum_alpha_delta = last_strat_shares[0]+last_strat_shares[3]
        strat_df.loc[0] = ['x', 'c', 'c', 'alpha', xn_df.loc['x', 'total']*(
            0.5*(1-sim)+ sim*last_strat_shares[0]/sum_alpha_delta), 0]
        strat_df.loc[1] = ['x', 'd', 'd', 'delta', xn_df.loc['x', 'total']*(
            0.5*(1-sim) + sim*last_strat_shares[3]/sum_alpha_delta), 0]
        
        # for non-externalizers all strategies are available, sum of available strategies is 1
        strat_df.loc[2] = ['n', 'c', 'c', 'alpha', xn_df.loc['n', 'total']*(
            0.25*(1-sim) + sim*last_strat_shares[0]), 0]
        strat_df.loc[3] = ['n', 'd', 'c', 'beta', xn_df.loc['n', 'total']*(
            0.25*(1-sim) + sim*last_strat_shares[1]), 0]
        strat_df.loc[4] = ['n', 'c', 'd', 'gamma', xn_df.loc['n', 'total']*(
            0.25*(1-sim) + sim*last_strat_shares[2]), 0]
        strat_df.loc[5] = ['n', 'd', 'd', 'delta', xn_df.loc['n', 'total']*(
            0.25*(1-sim) + sim*last_strat_shares[3]), 0]
        
    strat_df = strat_df.set_index(['externalization', 'name_strat'])
    
    return strat_df

# Extract data for the detailed simulation dataframe
def extract_simulation_data(gameplay_df, generation, learning_step, game_round):
    data = {}
    
    # Mapping for the externalization label
    ext_map = {'x': 'externalizing', 'n': 'non-externalizing'}
    
    for ext in ['x', 'n']:
        # Get all strategy names for this externalization type
        strat_names = ['alpha', 'delta'] if ext == 'x' else ['alpha', 'beta', 'gamma', 'delta']
        
        for strat in strat_names:
            # Get matched (married) and unmatched (available) data
            matched_share = gameplay_df.loc[(ext, strat), 'married']
            unmatched_share = gameplay_df.loc[(ext, strat), 'available']
            
            # Calculate payoffs
            matched_payoff = 0
            if matched_share > 0:
                partner_strat = gameplay_df.loc[(ext, strat), 'partner_strat']
                payoff_key = 'payoff_vs_c' if partner_strat == 'c' else 'payoff_vs_d'
                matched_payoff = gameplay_df.loc[(ext, strat), payoff_key]
            
            unmatched_payoff = 0
            if unmatched_share > 0:
                # Get the payoff for unmatched players
                sum_av = sum(gameplay_df['available'])
                if sum_av > 0:
                    prob_coop, prob_defc = prob_c_d(gameplay_df, sum_av)
                    payoff_vs_c = gameplay_df.loc[(ext, strat), 'payoff_vs_c']
                    payoff_vs_d = gameplay_df.loc[(ext, strat), 'payoff_vs_d']
                    unmatched_payoff = payoff_vs_c * prob_coop + payoff_vs_d * prob_defc
            
            # Add to data dictionary
            data[("generation", "", "", "")] = generation
            data[("learning_step", "", "", "")] = learning_step
            data[("game_round", "", "", "")] = game_round
            data[(ext_map[ext], strat, "matched", "share")] = matched_share
            data[(ext_map[ext], strat, "matched", "payoff")] = matched_payoff
            data[(ext_map[ext], strat, "unmatched", "share")] = unmatched_share
            data[(ext_map[ext], strat, "unmatched", "payoff")] = unmatched_payoff
    
    return data

# calculate expected payoff in this learning step for all the strategies
def payoff_learn_step(strat_df, model_params, simulation_df, generation, learning_step):
    strat_df = strat_df.copy()
    
    # reset payoff learn step and initialize gameplay df
    strat_df['payoff_learn_st'] = [0]*6
    gameplay_df = init_gameplay(strat_df, model_params)
    
    for game_ro in range(model_params['game_ti']):
        # play game and collect payoffs
        strat_df['payoff_learn_st'] += payoffs_game(gameplay_df)
        
        # Extract data for simulation dataframe before marrying
        data = extract_simulation_data(gameplay_df, generation, learning_step, game_ro)
        simulation_df.loc[len(simulation_df)] = data
        
        # Marry partners, adjust shares so sums are correct
        gameplay_df[['available', 'married']] = marry(gameplay_df)
        gameplay_df = adjust_gameplay(gameplay_df)
        
    return strat_df['payoff_learn_st']

# update strategies either success-based, frequency-based or mixed
def update_strats(strat_df, l_theta):
    strat_df = strat_df.copy()
    
    # calculate mean payoff for strategies on ext/next level
    mean_p = {
        ext: sum(strat_df.loc[(ext, ), 'total']*strat_df.loc[(ext, ), 'payoff_learn_st'])/sum(
            strat_df.loc[(ext, ), 'total']) for ext in ['x', 'n']
    }
    
    # calculate mean frequency for strategies on ext/next level
    mean_f = {
        ext: sum(strat_df.loc[(ext, ), 'total']*strat_df.loc[(ext, ), 'strat_freq'])/sum(
            strat_df.loc[(ext, ), 'total']) for ext in ['x', 'n']
    }

    # update strategies succes-based, frequency-based, or mixed depending on learn_factor
    strat_df.loc[('x', ['alpha', 'delta']), 'total'] *= (1 + (
        l_theta*(strat_df.loc[('x', ['alpha', 'delta']), 'payoff_learn_st']
                 - mean_p['x'])/mean_p['x'] +
        (1-l_theta)*(strat_df.loc[('x', ['alpha', 'delta']), 'strat_freq']
                     - mean_f['x'])/mean_f['x']
        ))
    strat_df.loc[('n', ['alpha', 'beta', 'gamma', 'delta']), 'total'] *= (1 + (
        l_theta*(strat_df.loc[('n', ['alpha', 'beta', 'gamma', 'delta']), 'payoff_learn_st']
                 - mean_p['n'])/mean_p['n'] +
        (1-l_theta)*(strat_df.loc[('n', ['alpha', 'beta', 'gamma', 'delta']), 'strat_freq']
                     - mean_f['n'])/mean_f['n']
        ))
    
    return strat_df

# preparations for update_strats
def learn_strats(strat_df, model_params):
    strat_df = strat_df.copy()
    strat_df = strat_df.sort_index()
    
    l_theta = model_params['l_theta']
     
    # add a column to the strat_df that displays the strategies frequency
    name_strat = pd.Series(strat_df.index.get_level_values('name_strat'))
    freq = {
        strat: sum(strat_df.loc[pd.IndexSlice[:, strat], 'total'])
        for strat in ['alpha', 'beta', 'gamma', 'delta']
    }
    strat_df['strat_freq'] = name_strat.map(freq).to_list()
        
    strat_df = update_strats(strat_df, l_theta)
    
    return strat_df['total']

# correct potential rounding mistakes to ensure correct sums for strat_df
def adjust_strats(xn_df, strat_df):
    strat_df = strat_df.copy()
    
    strat_df['total'] = strat_df['total'] * (strat_df['total'] > 0)
    scale_factor_x = xn_df.loc['x', 'total']/sum(strat_df.loc[('x', ['alpha', 'delta']), 'total'])
    scale_factor_n = xn_df.loc['n', 'total']/sum(strat_df.loc[('n', ['alpha', 'beta',
                                                                     'gamma', 'delta']), 'total'])
    strat_df.loc[('x', ['alpha', 'delta']), 'total'] *= scale_factor_x
    strat_df.loc[('n', ['alpha', 'beta', 'gamma', 'delta']), 'total'] *= scale_factor_n
    return strat_df['total']

# aggregate learn_step payoff on externalizing level
def aggregate_xn(strat_df):
    sum_x = sum(strat_df.loc[('x', 'total')])
    sum_n = sum(strat_df.loc[('n', 'total')])
    x_payoff = sum(strat_df.loc['x', 'payoff_learn_st'] * strat_df.loc['x', 'total']/sum_x)
    n_payoff = sum(strat_df.loc['n', 'payoff_learn_st'] * strat_df.loc['n', 'total']/sum_n)
    return [x_payoff, n_payoff]

# calculate payoffs in the generation's novel environment for externalizers vs. non-externalizers
def payoff_environment(xn_df, model_params, last_strat_df, simulation_df, generation):
    xn_df = xn_df.copy()
    
    # reset collected payoffs
    xn_df['payoff_environment'] = [0]*2
    strat_df = init_strat(xn_df, model_params['sim'], last_strat_df=last_strat_df)
    strat_df['total'] = adjust_strats(xn_df, strat_df)
    
    for learn_st in range(model_params['learn_ti']):
        # calculate expected payoff, aggregate on ext level, update and adjust strategies
        strat_df['payoff_learn_st'] = payoff_learn_step(strat_df, model_params, simulation_df, generation, learn_st)
        xn_df['payoff_environment'] += aggregate_xn(strat_df)
        strat_df['total'] = learn_strats(strat_df, model_params)
        strat_df['total'] = adjust_strats(xn_df, strat_df)
    
    return (xn_df['payoff_environment'], strat_df)

# calculate new porportions of ext/non according to replicator dynamics equation
def replicator_xn(xn_df, rep_k):
    # calculate mean payoff in population
    mean_payoff = sum([xn_df.loc[ext, 'total']*xn_df.loc[ext, 'payoff_environment']
                       for ext in ['x', 'n']])
    
    # apply replicator dynamics to each row of the dataframe
    xn_df['total'] *= (1 + rep_k*xn_df['total']*(xn_df['payoff_environment']
                                                 -mean_payoff)/mean_payoff)
    
    return xn_df['total']

# adjust possible rounding mistakes to ensure sums add up for xn_df
def adjust_xn(xn_df):
    xn_df = xn_df.copy()
    xn_df['total'] = xn_df['total']*(xn_df['total']>0)
    sum_xn = sum(xn_df['total'])
    xn_df['total'] *= 1/sum_xn
    return xn_df['total']

# initialize dataframe of externalization ('x') vs. non-externalization ('n')
# x_n = [share_x, share_n] are the initial shares
def init_xn(x_n=[0.5, 0.5]):
    xn_df = pd.DataFrame(columns=['externalization', 'total', 'payoff_environment'])
    xn_df = xn_df.set_index('externalization')
    xn_df.loc['x'] = [x_n[0], 0]
    xn_df.loc['n'] = [x_n[1], 0]
    return xn_df

# simulate natural selection of externalization with model parameters as given
# and write each generation's ext/non levels to simulation_df
def natural_selection(model_params):
    # initialize dataframes
    xn_df = init_xn([model_params['init_x'], 1-model_params['init_x']])
    last_strat_df = None # first generation
    
    # Initialize the dataframe to store generation-level data
    generation_df = pd.DataFrame(columns=['externalizers', 'non-externalizers', 'generation'])
    
    # Initialize the detailed simulation dataframe
    simulation_df = initialize_dataframe()
    
    for g in range(model_params['gen']):
        generation_df.loc[len(generation_df)] = [xn_df.loc['x', 'total'], xn_df.loc['n', 'total'], g]
        
        # calculate expected payoff, keep track of last generation's strategies
        # apply replicator dynamics and adjust shares due to rounding
        payoff_env, last_strat_df = payoff_environment(xn_df, model_params, last_strat_df, simulation_df, g)
        xn_df['payoff_environment'] = payoff_env
        xn_df['total'] = replicator_xn(xn_df, model_params['rep_k'])
        xn_df['total'] = adjust_xn(xn_df)
        if g%25 == 0:
            print('generation: ', g) # to display simulation progress
    
    return generation_df, simulation_df

# Example usage
def run_simulation(model_params):
    generation_df, simulation_df = natural_selection(model_params)
    return generation_df, simulation_df

# Example model parameters
base_model_params = {
    'tem': 3,     # temptation
    'rew': 2,     # reward
    'pun': 1,     # punishment
    'suc': 0,     # sucker
    'rep_k': 10, # selection strength
    'learn_ti': 15, # number of learning steps
    'game_ti': 15, # number of game rounds
    'sim': 0,   # similarity to previous generation
    'l_theta': 0.5, # learning factor (0=frequency, 1=success)
    'init_x': 0.01, # initial share of externalizers
    'gen': 3    # number of generations
}

# To run the simulation:
generation_df, simulation_df = run_simulation(base_model_params)
simulation_df.to_csv("../data/base_model_simulation.csv", index=False)