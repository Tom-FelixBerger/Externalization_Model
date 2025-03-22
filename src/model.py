import pandas as pd

def initialize_dataframe():
    simulation_df = pd.DataFrame(columns = ["generation", "learning_step", "game_round", "externalizing", "non_externalizing"])
    return simulation_df

