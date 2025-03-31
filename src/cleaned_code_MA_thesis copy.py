
#################################################################################################
#################################### Source Code Model 1 ########################################
#################################################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# initializes the dataframe holding the information about the interaction phase, i.e. the strategies,
# their proportions of partnered ('married') and available players, and the payoffs they can receive
def init_gameplay(strat_df, model_params):
    gameplay_df = strat_df.copy() 
    
    gameplay_df = gameplay_df.drop('payoff_learn_st', axis = 1)
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
def init_strat(xn_df, sim, last_strat_df = None):
    strat_df = pd.DataFrame(columns = ['externalization', 'game_strat', 'partner_strat',
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

# calculate expected payoff in this learning step for all the strategies
def payoff_learn_step(strat_df, model_params):
    strat_df = strat_df.copy()
    
    # reset payoff learn step and initialize gameplay df
    strat_df['payoff_learn_st'] = [0]*6
    gameplay_df = init_gameplay(strat_df, model_params)
    
    for game_ro in range(model_params['game_ti']):
        
        # play game and collect payoffs, marry partners, adjust shares so sums are correct
        strat_df['payoff_learn_st'] += payoffs_game(gameplay_df)
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
def payoff_environment(xn_df, model_params, last_strat_df):
    xn_df = xn_df.copy()
    
    # reset collected payoffs
    xn_df['payoff_environment'] = [0]*2
    strat_df = init_strat(xn_df, model_params['sim'], last_strat_df = last_strat_df)
    strat_df['total'] = adjust_strats(xn_df, strat_df)
    
    for learn_st in range(model_params['learn_ti']):
        
        # calculate expected payoff, aggregate on ext level, update and adjust strategies
        strat_df['payoff_learn_st'] = payoff_learn_step(strat_df, model_params)
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
def init_xn(x_n = [0.5, 0.5]):
    xn_df = pd.DataFrame(columns = ['externalization', 'total', 'payoff_environment'])
    xn_df = xn_df.set_index('externalization')
    xn_df.loc['x'] = [x_n[0], 0]
    xn_df.loc['n'] = [x_n[1], 0]
    return xn_df

# simulate natural selection of externalization with model parameters as given
# and write each generation's ext/non levels to simulation_df
def natural_selection(model_params, simulation_df):
    # initialize dataframes
    xn_df = init_xn([model_params['init_x'], 1-model_params['init_x']])
    last_strat_df = None # first generation
    
    for g in range(model_params['gen']):
        simulation_df.loc[len(simulation_df)] = [xn_df.loc['x', 'total'], xn_df.loc['n', 'total'], g]
        
        # calculate expected payoff, keep track of last generation's strategies
        # apply replicator dynamics and adjust shares due to rounding
        payoff_env, last_strat_df = payoff_environment(xn_df, model_params, last_strat_df)
        xn_df['payoff_environment'] = payoff_env
        xn_df['total'] = replicator_xn(xn_df, model_params['rep_k'])
        xn_df['total'] = adjust_xn(xn_df)
        if g%25 == 0:
            print('generation: ', g) # to display simulation progress

# simulate natural selection and prepare data to be visualized in a line chart
def simulate_nat_prep_plot(model_params):
    simulation_df = pd.DataFrame(columns = ['externalizing', 'non-externalizing', 'generation'])
    natural_selection(model_params, simulation_df)

    # transform dataframe to prepare visualization
    v_next = simulation_df[['non-externalizing', 'generation']].copy()
    v_next = v_next.rename(columns = {'non-externalizing': 'share'})
    v_ext = simulation_df[['externalizing', 'generation']].copy()
    v_ext = v_ext.rename(columns = {'externalizing': 'share'})
    v_next['strategy'] = ['non-externalizing']*len(v_next.index)
    v_ext['strategy'] = ['externalizing']*len(v_ext.index)

    return (simulation_df, pd.concat([v_next, v_ext]))

# simulate learning process for visualization
def simulate_learn_for_vis(model_params):
    xn_df = init_xn([model_params['init_x'], 1-model_params['init_x']])
    strat_df = init_strat(xn_df, 0)
    learning_df = pd.DataFrame(columns = ['learn_step', 'share', 'strategy'])
    
    # simulate learning process
    for learn_st in range(model_params['learn_ti']+1):
        
        # save strategy levels to result dataframe
        for name in ['alpha', 'beta', 'gamma', 'delta']:
            learning_df.loc[len(learning_df.index)] = [
                learn_st, sum(strat_df.loc[pd.IndexSlice[:, name], 'total']), name]
        
        strat_df['payoff_learn_st'] = payoff_learn_step(strat_df, model_params)
        # use the following to get insights to payoffs and aggregates for each learn step
        # print(strat_df)
        # print(aggregate_xn(strat_df))
        xn_df['payoff_environment'] += aggregate_xn(strat_df)
        strat_df['total'] = learn_strats(strat_df, model_params)
        strat_df['total'] = adjust_strats(xn_df, strat_df)

    return learning_df

# define model parameters
model_params = {'rew': 2, 'tem': 3, 'pun': 0, 'suc': 1, # game payoffs
                'gen': 200, 'learn_ti': 15, 'game_ti': 15, # generations, LS, IR
                'rep_k': 10, 'init_x': 0.01, # k for evolution speed, initial externalizers
                'sim': 0, 'l_theta': 0.5} # similarity parameter s, theta for learning mechanism

# plot natural selection
simulation_df, v = simulate_nat_prep_plot(model_params)
my_palette1 = ["orchid","midnightblue"]
my_plot = sns.lineplot(data = v, x="generation", y="share", hue="strategy", palette=my_palette1)
sns.move_legend(my_plot, "upper left",bbox_to_anchor=(0.5, 0.6))
my_plot.set_ylim(-0.05,1.05)
plt.savefig('my_natural_selection_plot.png')
plt.show()

# plot learning process
my_palette2 = ["green","red",'blue','orange']
learning_df = simulate_learn_for_vis(model_params)
my_plot = sns.lineplot(data = learning_df, x="learn_step", y="share",
                       hue="strategy", palette=my_palette2)
sns.move_legend(my_plot, "upper left",bbox_to_anchor=(0.65, 0.65))
my_plot.set_ylim(-0.05,1.05)
plt.savefig('my_learning_plot.png')
plt.show()


#################################################################################################
#################################### Source Code Model 2 ########################################
#################################################################################################

import pandas as pd
import numpy as np
from numpy.random import choice
from random import sample, random
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools

# agent class
class Agent:
    
    def __init__(self, id, x, population):
        self.id = id
        self.externalization = x
        
        if x:
            self.strategy = choice(['alpha', 'delta'], 1)[0]
        else:
            self.strategy = choice(['alpha', 'beta', 'gamma', 'delta'], 1)[0]
        
        self.ls_fitness = 0
        self.total_fitness = 0
        self.population = population
    
    def get_attributes(self):
        return {'id': self.id,
                'externalization': self.externalization,
                'strategy': self.strategy,
                'ls_fitness': self.ls_fitness,
                'total_fitness': self.total_fitness}
    
    def give_payoff(self, inc):
        self.ls_fitness += inc
        self.total_fitness += inc
    
    def set_strategy(self, strategy):
        self.strategy = strategy
    
    def reset_ls(self):
        self.ls_fitness = 0

# for iteration over agents of population in random order
class Population_iterator:
    
    def __init__(self, pop_size, agents, quasi_ext):
        self.agents = agents
        self.random_order = sample(list(range(pop_size)), pop_size) # shuffle agents
        self.quasi_ext = quasi_ext
        self.pop_size = pop_size
        self.current = 0
        
    def __iter__(self):
        self.current = 0
        return self
    
    def __next__(self):
        if self.current < self.pop_size:
            if self.quasi_ext:
                current_strat = self.agents[self.current].get_attributes()['strategy']
                while current_strat not in ['alpha', 'delta']:
                    self.current += 1
                    if self.current >= self.pop_size:
                        raise StopIteration
                    
                    current_strat = self.agents[self.current].get_attributes()['strategy']
                    
            return_pos = self.random_order[self.current]
            self.current += 1
            return self.agents[return_pos]
        else:
            raise StopIteration

# Population class contains dictionary of agents and source agents for source based learning
class Population:
    
    def __init__(self, size, externalizers, n_sources):
        self.size = size
        
        shuffled = sample(list(range(size)), size) # to assign random ids to externalizers
        self.agents = {id: Agent(id, shuffled[id]<externalizers, self)
                  for id in range(size)}
        
        self.source_ids = sample(range(size), n_sources)
    
    # iterate over population, if quasi_ext = True return only alpha and delta agents
    def iterate(self, quasi_ext = False):
        return Population_iterator(self.size, self.agents, quasi_ext)
    
    # return list of agents, if quasi_ext = True return only alpha and delta
    def get_agents(self, quasi_ext = False):
        if quasi_ext:
            return [self.agents[id] for id in range(self.size) if
                    self.agents[id].get_attributes()['strategy'] in ['alpha', 'delta']]
        else:
            return [self.agents[id] for id in range(self.size)]
    
    # return list of source agents, if quasi_ext = True return only alpha and delta
    def get_source_agents(self, quasi_ext = False):
        if quasi_ext:
            return [self.agents[id] for id in self.source_ids if
                    self.agents[id].get_attributes()['strategy'] in ['alpha', 'delta']]
        return [self.agents[id] for id in self.source_ids]
    
    def get_agent_by_id(self, id):
        return self.agents[id]
    
    def get_pop_size(self):
        return self.size
    
    # reset learn step attributes for all agents
    def reset_ls(self):
        for id in range(self.size):
            self.agents[id].reset_ls()
    
    # initialize new generation
    def new_generation(self, externalizers, n_sources):
        shuffled = sample(list(range(self.size)), self.size)
        self.agents = {id: Agent(id, shuffled[id]<externalizers, self)
                  for id in range(self.size)}
        self.source_ids = sample(range(self.size), n_sources)

# let agents with id_a and id_b interact and increase payoffs
def interact(model_params, population, id_a, id_b):
    ag_a = population.get_agent_by_id(id_a)
    ag_b = population.get_agent_by_id(id_b)
    action_a = 'c' if ag_a.get_attributes()['strategy'] in ['alpha', 'gamma'] else 'd'
    action_b = 'c' if ag_b.get_attributes()['strategy'] in ['alpha', 'gamma'] else 'd'
    
    payoff_matr = model_params['payoff_matr']
    game = choice(['PD', 'HD', 'SH'], 1, p=model_params['pi'])[0]
    
    if action_a == 'c':
        if action_b == 'c':
            ag_a.give_payoff(payoff_matr[game][0])
            ag_b.give_payoff(payoff_matr[game][0])
        else:
            ag_a.give_payoff(payoff_matr[game][1])
            ag_b.give_payoff(payoff_matr[game][2])
    elif action_b == 'c':
        ag_a.give_payoff(payoff_matr[game][2])
        ag_b.give_payoff(payoff_matr[game][1])
    else:
        ag_a.give_payoff(payoff_matr[game][3])
        ag_b.give_payoff(payoff_matr[game][3])
        
# form matchings and let all pairs of agents interact
def pair_and_play(population, model_params, return_value = 'None'):
    
    # reset payoff learn step
    population.reset_ls()
    
    if return_value == 'matchings_payoffs':
        return_df = pd.DataFrame(columns=['id_a', 'id_b', 'strat_a', 'strat_b',
                                          'ext_a', 'ext_b', 'payoff_a', 'payoff_b'])

    # put agents into buckets depending on whether their strategy is revealed
    # 'bucket_dict' to get bucket of by player id, 'buckets' contains list of players by bucket
    bucket_dict = {id: '' for id in range(model_params['pop_size'])} 
    buckets = {b: [] for b in ['alpha', 'beta', 'gamma', 'delta', 'unknown']}
    for agent in population.iterate():
        attr = agent.get_attributes()
        if random() < model_params['rho']: # reveal agent behavior with probability rho
            buckets[attr['strategy']].append(attr['id'])
            bucket_dict[attr['id']] = attr['strategy']
        else:
            buckets['unknown'].append(attr['id'])
            bucket_dict[attr['id']] = 'unknown'
    
    matches = [] # list of all formed matches
    # match known players among each other
    for strats in [('alpha','alpha'), ('beta', 'gamma'), ('delta', 'delta')]:
        # while two alphas / deltas OR at least one beta and one gamma are in the buckets
        while (((strats[0] in ['alpha', 'delta']) & (len(buckets[strats[0]])>1)) | 
                ((strats[0] == 'beta') & (len(buckets[strats[0]])*len(buckets[strats[1]])>0))):
            id_a = buckets[strats[0]].pop()
            id_b = buckets[strats[1]].pop()
            bucket_dict[id_a] = 'done'
            bucket_dict[id_b] = 'done'
            matches.append((id_a, id_b))

    # now we have a situation where there is one or less known alpha (just as delta),
    # and either zero known gammas or zero known betas.
    leftovers = [] # leftovers for players that couldn't find partner
    happy = {'alpha': ['alpha', 'gamma'], 'beta': ['alpha', 'gamma'],
             'gamma': ['beta', 'delta'], 'delta': ['beta', 'delta']}
    
    # for agents that have not found a partner yet, find partner
    for id_a in range(population.get_pop_size()):
        
        if bucket_dict[id_a] == 'done': # if already played skip current iteration
            continue 
        
        # remove agent from buckets (because he will have found a partner)
        buckets[bucket_dict[id_a]].remove(id_a)
        attr = population.get_agent_by_id(id_a).get_attributes()
        
        # if unknown, select from known if possible, else random from unknown
        if bucket_dict[id_a] == 'unknown':
            
            # find partner in preferred bucket or unknown and remove partner from buckets
            pref_buckets = sample(happy[attr['strategy']], 2)
            for b in pref_buckets + ['unknown']:
                if buckets[b]: # check for emptiness of bucket
                    id_b = buckets[b].pop()
                    bucket_dict[id_b] = 'done'
                    matches.append((id_a, id_b))
                    break                
            else:
                leftovers.append(id_a)
            
        else: # if known, other known players will reject, so search in unknown
            
            for id_b in buckets['unknown']:
                b_strat = population.get_agent_by_id(id_b).get_attributes()['strategy']
                if attr['strategy'] in happy[b_strat]:
                    buckets['unknown'].remove(id_b)
                    bucket_dict[id_b] = 'done'
                    matches.append((id_a, id_b))
                    break
            else:
                leftovers.append(id_a)
        
        bucket_dict[id_a] = 'done'
    
    while(len(leftovers)>1): # finally match leftovers with each other
        id_a = leftovers.pop()
        id_b = leftovers.pop()
        matches.append((id_a, id_b))
            
    for match in matches: # let matches interact and write to df
        interact(model_params, population, match[0], match[1])
        
        if return_value == 'matchings_payoffs':
            a = population.get_agent_by_id(match[0]).get_attributes()
            b = population.get_agent_by_id(match[1]).get_attributes()
            return_df.loc[len(return_df)] = [a['id'], b['id'], a['strategy'], b['strategy'],
                            a['externalization'], b['externalization'],
                            a['ls_fitness'], b['ls_fitness']]
            
    if return_value == 'population':
        return population
    elif return_value == 'matchings_payoffs':
        return return_df

# count externalizers in top half for natural selection
def count_ext_top_half(population):
    ranking = population.get_agents()
    ranking = [(a.get_attributes()['externalization'], a.get_attributes()['total_fitness'])
               for a in ranking]
    ranking.sort(key = lambda x: x[1], reverse = True)
    top_half = ranking[0:int(len(ranking)/2)]
    count_ext_top_half = sum([a[0] for a in top_half])
    return count_ext_top_half

# count all 6 strategies and other aggregates
def count_strats(learn_st, population):
    counts = {strat: 0 for strat in ['alpha', 'beta', 'gamma', 'delta',
                                     'externalizer', 'non-externalizer']}
    payoffs = {strat: 0 for strat in ['alpha', 'beta', 'gamma', 'delta',
                                      'externalizer', 'non-externalizer']}
    for agent in population.iterate():
        strat = agent.get_attributes()['strategy']
        counts[strat] += 1
        payoffs[strat] += agent.get_attributes()['ls_fitness']
        ext = 'externalizer' if agent.get_attributes()['externalization'] else 'non-externalizer'
        counts[ext] += 1
        payoffs[ext] += agent.get_attributes()['ls_fitness']
        ext_top50 = count_ext_top_half(population)
    result = pd.DataFrame({'learn_st': [learn_st]*6,
                           'strategy': ['alpha', 'beta', 'gamma', 'delta',
                                        'externalizer', 'non-externalizer'],
                           'count': [counts[strat]
                                     for strat in ['alpha', 'beta', 'gamma', 'delta',
                                                   'externalizer', 'non-externalizer']],
                           'payoff': [payoffs[strat]/counts[strat] if counts[strat]>0 else
                                      np.nan for strat in ['alpha', 'beta', 'gamma', 'delta',
                                                           'externalizer', 'non-externalizer']],
                           'ext_top_half': [ext_top50]*6})
    return result

# cultural learning of strategies (success-based, freq or source)
def learn_strats(population, model_params):
    pop_size = model_params['pop_size']
    new_strats = {id: '' for id in range(pop_size)}
    
    # lists of all potential demonstrators
    demonstrators = {'succ': None, 'freq': None, 'sour': None}
    demonstrators['succ'] = {True: population.get_agents(True),
                             False: population.get_agents(False)}
    demonstrators['succ'][True].sort(key = lambda agent:
                                    agent.get_attributes()['ls_fitness']) # sorts inplace
    demonstrators['succ'][False].sort(key = lambda agent:
                                    agent.get_attributes()['ls_fitness'])
    demonstrators['freq'] = {True: population.get_agents(True),
                             False: population.get_agents(False)}
    demonstrators['sour'] = {True: population.get_source_agents(True),
                             False: population.get_source_agents(False)}
    
    # list of respective demonstrators' selection biases
    biases = {'succ': None, 'freq': None, 'sour': None}
    biases['succ'] = {ext: list(range(1,len(demonstrators['succ'][ext])+1))
                      for ext in [True, False]}
    biases['succ'] = {ext: [b/sum(biases['succ'][ext]) for b in biases['succ'][ext]]
                      for ext in [True, False]}
    frequencies = count_strats(0, population)[['strategy', 'count']]
    frequencies = frequencies.set_index('strategy').squeeze().to_dict()    
    biases['freq'] = {ext: [frequencies[d.get_attributes()['strategy']]
                            for d in demonstrators['freq'][ext]] for ext in [True, False]}
    biases['freq'] = {ext: [b/sum(biases['freq'][ext]) for b in biases['freq'][ext]]
                      for ext in [True, False]}
    biases['sour'] = {ext: [1/len(demonstrators['sour'][ext])]*len(demonstrators['sour'][ext])
                      if len(demonstrators['sour'][ext])>0 else None for ext in [True, False]}
    
    # for each agent determine learning mechanism and copy demonstrator
    for agent in population.iterate():
        l_mech = choice(['succ', 'freq', 'sour'], 1, p=model_params['theta'])[0]
        ext = agent.get_attributes()['externalization']
        if bool(demonstrators[l_mech][ext]):
            demonstrator = choice(demonstrators[l_mech][ext], 1, p=biases[l_mech][ext])[0]
        else:
            demonstrator = agent
        new_strats[agent.get_attributes()['id']] = demonstrator.get_attributes()['strategy']
    
    # set new strategies
    for agent in population.iterate():
        agent.set_strategy(new_strats[agent.get_attributes()['id']])

# run learning process to determine lifetime payoff for each agent
def payoff_learning_process(population, model_params, return_value = 'None'):
    
    if return_value == 'process':
        process = pd.DataFrame(columns = ['learn_st', 'strategy', 'count',
                                          'payoff', 'ext_top_half'])
    
    for learn_st in range(model_params['learning_steps']):
        
        # calculate payoffs in this learning step for all the strategies and learn new strategy
        pair_and_play(population, model_params)
        if return_value == 'process': # add strategy aggregates to result dataframe
            process = count_strats(learn_st, population) if process.empty else pd.concat(
                [process, count_strats(learn_st, population)], axis = 0)
        learn_strats(population, model_params)
        
    if return_value == 'process':
        return process
    elif return_value == 'result':
        return count_strats(population)

# count externalizers in whole population
def count_externalizers(population):
    count_ext = 0
    for agent in population.iterate():
        if agent.get_attributes()['externalization']:
            count_ext += 1
    return count_ext

# simulate natural selection for the given model parameters and return final result, 
# ext/non levels for each generation, or nothing
def natural_selection(model_params, return_value = 'None'):

    # initialize population
    population = Population(model_params['pop_size'], model_params['init_x'],
                            model_params['n_sources'])
    
    if return_value == 'history':
        history = pd.DataFrame({'generation':[0],
                                'population_size':[model_params['pop_size']],
                                'externalizers': [model_params['init_x']]})
    elif return_value == 'final':
        final_gen = 0
        count_ext = model_params['init_x']
    
    for gen in range(model_params['gen']):
        if count_externalizers(population) in [0, model_params['pop_size']]:
            break
        
        # calculate expected payoff in a new environment for externalizers and non-externalizers
        payoff_learning_process(population, model_params)
        
        # update population externalization trait by replicator dynamics
        new_ext = count_ext_top_half(population)*2

        population.new_generation(new_ext, model_params['n_sources'])
        
        if return_value == 'history': # write current population state to the result dataframe
            count_ext = count_externalizers(population)
            history.loc[len(history.index)] = [gen+1, model_params['pop_size'], count_ext]
        elif return_value == 'final': # save current population state
            final_gen = gen + 1
            count_ext = count_externalizers(population)
        
    if return_value == 'history':
        return history
    elif return_value == 'final':
        return [count_ext, final_gen]

# simulate the interaction phase sample_size times 
def experiment_pair_and_play(model_params, sample_size):
    result = pd.DataFrame(columns=['test_interaction', 'id_a', 'id_b', 'strat_a', 'strat_b',
                                   'ext_a', 'ext_b', 'payoff_a', 'payoff_b'])
    for test_interaction in range(sample_size):
        test_population = Population(model_params['pop_size'], model_params['init_x'],
                                     model_params['n_sources'])
        run_result = pair_and_play(test_population, model_params,
                                   return_value='matchings_payoffs')
        run_result['test_interaction'] = [test_interaction]*len(run_result.index)
        result = pd.concat([result, run_result])
    return result.reset_index(drop=True)

# simulate learning process for sample_size times
def experiment_learning_process(model_params, sample_size):
    result = pd.DataFrame(columns=['test_run', 'learn_st', 'strategy',
                                   'count', 'payoff', 'ext_top_half'])
    for test_run in range(sample_size):
        print(test_run)
        test_population = Population(model_params['pop_size'], model_params['init_x'],
                                     model_params['n_sources'])
        run_result = payoff_learning_process(test_population, model_params,
                                             return_value='process')
        run_result['test_run'] = [test_run]*len(run_result.index)
        result = run_result if result.empty else pd.concat([result, run_result])
    return result.reset_index(drop=True)

# simulate random natural selection of fitness irrelevant trait A
def random_natural_selection(model_params):
    # initialize population
    history = pd.DataFrame(columns = ['generation', 'population_size', 'trait_A'])
    history.loc[0] = [0, model_params['pop_size'], model_params['init_x']]
    
    for gen in range(1, model_params['gen']):
        
        prior_A = int(history.loc[gen-1, 'trait_A'])
        pop_size = int(history.loc[gen-1, 'population_size'])
        new_trait_A = sum(sample([1]*prior_A + [0]*(pop_size-prior_A), int(pop_size/2)))*2
        history.loc[len(history.index)] = [gen, model_params['pop_size'], new_trait_A]
    
    return history

# simulate natural selection sample_size times (for fitness irrelevant trait A if random = True)
def experiment_natural_selection(model_params, sample_size, random=False):
    result_df = pd.DataFrame(columns=['simulation_no', 'generation',
                                      'population_size', 'externalizers'])
    for simulation_no in range(sample_size):
        if random:
            add_df = random_natural_selection(model_params)
        else:
            add_df = natural_selection(model_params, return_value = 'history')
        add_df['simulation_no'] = [simulation_no]*len(add_df.index)
        result_df = add_df.copy() if result_df.empty else pd.concat([result_df, add_df])
        if simulation_no%25==0: # to display simulation progress
            print(simulation_no)
    
    return result_df

# for robustness analysis combine varying parameter values to a list of parameter dictionaries
def combine_parameters(param_dict):
    # Get the parameter names and corresponding lists of values
    keys = param_dict.keys()
    values = param_dict.values()
    
    # Compute the Cartesian product of the values
    combinations = itertools.product(*values)
    
    # Create a list of dictionaries from the combinations
    result = [dict(zip(keys, combination)) for combination in combinations]
    
    return result

# conduct robustness analysis for all combinations of parameter values in param_dict, other
# parameteres are given by model_params. Each combination is simulated sample_size times
def natural_selection_robustness(model_params, sample_size, param_dict):
    result_df = pd.DataFrame(columns=['setup_name', 'simulation_no', 'generation',
                                      'population_size', 'externalizers'])
    
    parameter_setups = combine_parameters(param_dict)
        
    for parameter_setup in parameter_setups:
        setup_name = '_'.join([str(par)+'='+str(parameter_setup[par])
                               for par in parameter_setup])
        print(setup_name)
        
        for par in parameter_setup:
            model_params[par] = parameter_setup[par]
            
        for simulation_no in range(sample_size):
            ext, gen = natural_selection(model_params, return_value = 'final')
            result_df.loc[len(result_df)] = [setup_name, simulation_no, gen,
                                             model_params['pop_size'], ext]
            if simulation_no % 25 ==0:
                print(simulation_no)
    
    return result_df

# print for each setup of the passed ra_df the convergence results and statistical test results
# it is expected that benchmark df is a suited df for all ra setups
def print_ra_significance_results(ra_df, benchmark_df):
    for setup_name in pd.unique(ra_df['setup_name']):
        setup_df = ra_df[ra_df['setup_name']==setup_name].copy()
        setup_df = setup_df[['generation', 'population_size', 'externalizers', 'simulation_no']]
        setup_df = setup_df.reset_index(drop=True)
        
        final_gen_filt = pd.DataFrame(columns=['generation', 'population_size',
                                     'externalizers', 'simulation_no'])
        for sim_no in range(setup_df['simulation_no'].max()+1):
            final_gen = setup_df[setup_df['simulation_no']==sim_no]['generation'].max()
            final_gen_filt = final_gen_filt._append(setup_df[(setup_df['generation']==final_gen)
                                                    &(setup_df['simulation_no']==sim_no)].iloc[0])
        
        b_filt = benchmark_df[benchmark_df['generation']==49]
        
        print('---------- ' + setup_name + '----------')
        actual_simulation = final_gen_filt['externalizers'].to_list()
        benchmark_simulation = b_filt['trait_A'].to_list()
        pop_size = setup_df.iloc[0]['population_size']
        b_invasion = [c for c in benchmark_simulation if not c in [0, pop_size]]
        invasion = [c for c in actual_simulation if not c in [0, pop_size]]
        if len(invasion)>0:
            print('invasion_mean: '+str(np.mean(invasion)))
        else:
            print('invasion_mean: None')
        if len(b_invasion)>0:
            print('invasion_mean_benchmark: '+str(np.mean(b_invasion)))
        else:
            print('invasion_mean_benchmark: None')
        print('invasion_no: '+str(len(invasion)))
        print('invasion_no_benchmark: ' + str(len(b_invasion)))
        print('extinction_no: ' + str(len([c for c in actual_simulation if c==0])))
        print('extinction_no_benchmark: ' + str(len([c for c in benchmark_simulation if c==0])))
        print('takeover_no: ' + str(len([c for c in actual_simulation if c==pop_size])))
        print('takeover_no_benchmark: ' + str(len([c for c in benchmark_simulation
                                                   if c==pop_size])))

        # mann whitney u test for not normally distributed ordinal data
        stat, p_value = stats.mannwhitneyu(benchmark_simulation, actual_simulation,
                                           alternative='two-sided')
        print('WMW-test: '+ str(stat) +', '+str(p_value))
    

# define model parameters
model_params = {'pop_size': 100,
          'init_x': 1,
          'rho': 0.75,
          'theta': [1, 0, 0], # [succ, freq, sour]
          'n_sources': 4,
          'pi': [1, 0, 0], # [PD, HD, SH]
          'payoff_matr': {'PD': [2, 0, 3, 1], 'HD': [2, 1, 3, 0], 'SH': [3, 0, 2, 2]},
          'learning_steps': 15,
          'gen': 50
          }

# conduct robustness analysis and test statistical significance like this:
# to test only one setup, pass an empty dict {} for param_dict
ra_df = natural_selection_robustness(model_params, 100, {'rho': [0.5, 0.75],
                                                        'pi': [[1,0,0],[0,1,0],[0,0,1]],
                                                        'theta': [[0,1,0],[0,0,1]],
                                                     })
ra_df.to_csv('my_robustness_data.csv')
benchmark_df = experiment_natural_selection(model_params, 100, random = True)
print_ra_significance_results(ra_df, benchmark_df)

# visualize learning process like this
exp_ls_df = experiment_learning_process(model_params, 100)
vis_df = exp_ls_df[exp_ls_df['strategy'].isin(['alpha', 'beta', 'gamma', 'delta'])]
my_palette1 = ["green","red",'deepskyblue','orange']
fig, ax = plt.subplots()
sns.lineplot(x=vis_df['learn_st'], y=vis_df['count'], hue=vis_df['strategy'], palette=my_palette1,
             errorbar = ('pi', 95), err_style = 'band', err_kws={'linestyle': 'dashed', 'alpha': 0.03})
edge_df = vis_df.groupby(['strategy', 'learn_st']).agg(
    percentile_low=('count', lambda x: np.percentile(x, 2.5)),
    percentile_high=('count', lambda x: np.percentile(x, 97.5))
    ).reset_index()
order = {'alpha': 1, 'beta': 2, 'gamma': 3, 'delta': 4}
edge_df = edge_df.sort_values(by = 'strategy', key=lambda x: x.map(order))
sns.lineplot(x=edge_df['learn_st'], y=edge_df['percentile_low'], linestyle=':', hue=edge_df['strategy'], palette=my_palette1, alpha=0.5, legend=False)
sns.lineplot(x=edge_df['learn_st'], y=edge_df['percentile_high'], linestyle=':', hue=edge_df['strategy'], palette=my_palette1, alpha=0.5, legend=False)
ax.set_ylim(0, 100)
sns.move_legend(ax, (0.7, 0.4))
plt.savefig('my_learn_process.png')
plt.show()