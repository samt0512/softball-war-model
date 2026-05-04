import pandas as pd
import numpy as np
import glob
import os


def load_data():
    """Load and combine all seasons of games, boxscore, and pbp data."""
    
    def load_csv(path):
        df_list = []
        for filename in glob.glob(os.path.join(path, "*.csv")):
            df = pd.read_csv(filename)
            year = filename.split("_")[1].split(".")[0]
            df["season"] = int(year)
            df_list.append(df)
        return pd.concat(df_list, ignore_index=True)

    games_df    = load_csv(r'D:\softball-boxscore-data\data\games')
    boxscore_df = load_csv(r'D:\softball-boxscore-data\data\boxscore')
    pbp_df      = load_csv(r'D:\softball-boxscore-data\data\pbp')

    print(f"Games:     {games_df.shape}")
    print(f"Boxscore:  {boxscore_df.shape}")
    print(f"PBP:       {pbp_df.shape}\n")

    return games_df, boxscore_df, pbp_df


def build_run_expectancy(pbp_df):
    """
    Build a 3-state run expectancy matrix based on outs.
    
    Steps:
        1. Filter to play result rows only
        2. Determine the score for the batting team on each play
        3. Find the final score for the batting team at the end of each half inning
        4. Calculate runs scored from each play to end of half inning
        5. Group by outs_before and take the mean
        6. Return a dict or series with RE for 0, 1, 2 outs
    """
    filtered_pbp = pbp_df[pbp_df['play_type'] == "Play Result"].copy()
    filtered_pbp["batting_score"] = np.where(filtered_pbp["inning_half"] == "Top", filtered_pbp["away_score"], filtered_pbp["home_score"])
    filtered_pbp["inning_end_score"] = filtered_pbp.groupby(["game_id", "inning", "inning_half"])["batting_score"].transform("max")
    filtered_pbp["runs_to_end"] = filtered_pbp["inning_end_score"] - filtered_pbp["batting_score"]
    filtered_pbp = filtered_pbp[filtered_pbp["outs_before"] <= 2]
    re_matrix = filtered_pbp.groupby("outs_before")["runs_to_end"].mean()

    return re_matrix

def classify_event(description):
    # convert description to lowercase
    description = description.lower()
    # check for home run first (before single since some descriptions may contain both)
    if 'home run' in description or 'homered' in description:
        return "homerun"
    # check for triple
    if 'tripled' in description:
        return "triple"
    # check for double
    if 'doubled' in description:
        return "double"
    # check for single
    if 'singled' in description:
        return "single"
    # check for  walk
    if 'walked' in description:
        return "walk"
    # check for hit by pitch
    if 'hit by pitch' in description:
        return 'hit by pitch'
    # check for strikeout
    if 'struck out' in description:
        return 'strike out'
    # check for out (grounded out, flied out, lined out, popped up)
    if 'grounded out' in description or 'flied out' in description or 'lined out' in description or 'popped up' in description:
        return "out"
    # check for error
    if 'error' in description:
        return "error"
    # check for fielder's choice
    if "fielder's choice" in description:
        return "fielder's choice"
    # return "other" if nothing matches
    else:
        return "other"

def calculate_linear_weights(pbp_df, re_matrix):
    """
    Calculate linear weights for each event type using the RE matrix.
    
    Steps:
        1. Filter to play result rows
        2. Classify each play description into an event type
           (single, double, triple, HR, walk, strikeout, out, etc.)
        3. For each event calculate:
           linear_weight = score_value + RE_after - RE_before
        4. Average across all instances of each event
        5. Return a dict of event -> linear weight
    """
    filtered_pbp = pbp_df[pbp_df['play_type'] == "Play Result"].copy()
    filtered_pbp["event_type"] = filtered_pbp["description"].apply(classify_event)
    print(filtered_pbp["event_type"].value_counts())


def calculate_woba(boxscore_df, linear_weights):
    """
    Calculate wOBA for each player using linear weights.

    Steps:
        1. Convert boxscore stat columns to numeric
        2. Estimate hit types (1B, 2B, 3B) from total H and HR
        3. Calculate PA = AB + BB
        4. Apply linear weights to get raw wOBA numerator
        5. Divide by PA to get wOBA rate
        6. Return player season wOBA dataframe
    """
    pass


def calculate_war(boxscore_df, woba_df, games_df):
    """
    Calculate WAR for each player.

    Steps:
        1. Calculate league average wOBA
        2. Calculate wOBA scale
        3. Calculate wRAA (runs above average)
        4. Define replacement level from bottom 20% of qualified players
        5. Calculate batting runs above replacement
        6. Apply positional adjustment
        7. Calculate runs per win from games_df
        8. Divide total runs by RPW to get WAR
        9. Return player WAR dataframe
    """
    pass


if __name__ == "__main__":
    
    games_df, boxscore_df, pbp_df = load_data()
    
    re_matrix      = build_run_expectancy(pbp_df)
    linear_weights = calculate_linear_weights(pbp_df, re_matrix)
    """
    woba_df        = calculate_woba(boxscore_df, linear_weights)
    war_df         = calculate_war(boxscore_df, woba_df, games_df)
    """