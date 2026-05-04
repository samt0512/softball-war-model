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

    games_df    = load_csv(r'D:\softball-war-model\data\games')
    boxscore_df = load_csv(r'D:\softball-war-model\data\boxscore')
    boxscore_df = boxscore_df.drop_duplicates(subset=["athlete_id", "game_id"])
    pbp_df      = load_csv(r'D:\softball-war-model\data\pbp')

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
        6. Return a series with RE for 0, 1, 2 outs
    """
    filtered_pbp = pbp_df[pbp_df['play_type'] == "Play Result"].copy()
    filtered_pbp["batting_score"] = np.where(
        filtered_pbp["inning_half"] == "Top",
        filtered_pbp["away_score"],
        filtered_pbp["home_score"]
    )
    filtered_pbp["inning_end_score"] = filtered_pbp.groupby(
        ["game_id", "inning", "inning_half"]
    )["batting_score"].transform("max")
    filtered_pbp["runs_to_end"] = filtered_pbp["inning_end_score"] - filtered_pbp["batting_score"]
    filtered_pbp = filtered_pbp[filtered_pbp["outs_before"] <= 2]
    re_matrix = filtered_pbp.groupby("outs_before")["runs_to_end"].mean()

    return re_matrix


def classify_event(description):
    """Classify a play description into an event type."""
    description = description.lower()
    if 'home run' in description or 'homered' in description:
        return "homerun"
    if 'tripled' in description:
        return "triple"
    if 'doubled' in description:
        return "double"
    if 'singled' in description:
        return "single"
    if 'walked' in description:
        return "walk"
    if 'hit by pitch' in description:
        return 'hit by pitch'
    if 'struck out' in description:
        return 'strike out'
    if 'grounded out' in description or 'flied out' in description or 'lined out' in description or 'popped up' in description:
        return "out"
    if 'error' in description:
        return "error"
    if "fielder's choice" in description:
        return "fielder's choice"
    return "other"


def find_outs(description, event_type):
    """Infer the number of outs recorded on a play from its description and event type."""
    if 'double play' in description.lower():
        return 2
    elif event_type in ["single", "double", "triple", "homerun", "walk", "hit by pitch", "error"]:
        return 0
    elif event_type in ["out", "fielder's choice", "strike out"]:
        return 1
    else:
        return 0


def calculate_linear_weights(pbp_df, re_matrix):
    """
    Calculate linear weights for each event type using the RE matrix.

    Steps:
        1. Filter to play result rows
        2. Classify each play description into an event type
        3. For each event calculate: linear_weight = score_value + RE_after - RE_before
        4. Average across all instances of each event
        5. Return a dict of event -> linear weight
    """
    filtered_pbp = pbp_df[pbp_df['play_type'] == "Play Result"].copy()
    filtered_pbp["event_type"] = filtered_pbp["description"].apply(classify_event)
    filtered_pbp["outs_added"] = filtered_pbp.apply(
        lambda row: find_outs(row["description"], row["event_type"]), axis=1
    )
    filtered_pbp["outs_after"] = (filtered_pbp["outs_before"] + filtered_pbp["outs_added"]).clip(upper=3)
    filtered_pbp["RE_before"] = filtered_pbp["outs_before"].map(re_matrix)
    filtered_pbp["RE_after"]  = filtered_pbp["outs_after"].map(re_matrix)
    filtered_pbp["run_value"] = (filtered_pbp["score_value"] + filtered_pbp["RE_after"]) - filtered_pbp["RE_before"]
    linear_weights = filtered_pbp.groupby("event_type")["run_value"].mean().to_dict()

    return linear_weights


def get_hit_types(pbp_df):
    """
    Aggregate exact hit type counts per player per season from PBP data.
    Returns a dataframe with batter_id, season, and columns for each hit type.
    """
    filtered_pbp = pbp_df[pbp_df['play_type'] == "Play Result"].copy()
    filtered_pbp["event_type"] = filtered_pbp["description"].apply(classify_event)

    hit_types = ["single", "double", "triple", "homerun"]
    filtered_pbp = filtered_pbp[filtered_pbp["event_type"].isin(hit_types)]

    hit_counts = filtered_pbp.groupby(["batter_id", "season", "event_type"]).size().unstack(fill_value=0)
    hit_counts.columns.name = None
    hit_counts = hit_counts.reset_index()

    for col in hit_types:
        if col not in hit_counts.columns:
            hit_counts[col] = 0

    return hit_counts


def calculate_woba(boxscore_df, pbp_df, linear_weights):
    """
    Calculate wOBA for each player using linear weights.

    Steps:
        1. Convert boxscore stat columns to numeric
        2. Calculate PA = AB + BB
        3. Aggregate boxscore to season level
        4. Join exact hit type counts from PBP data
        5. Apply linear weights to get wOBA
        6. Return player season wOBA dataframe
    """
    boxscore_df["AB"] = pd.to_numeric(boxscore_df["AB"], errors="coerce").fillna(0)
    boxscore_df["BB"] = pd.to_numeric(boxscore_df["BB"], errors="coerce").fillna(0)
    boxscore_df["PA"] = boxscore_df["AB"] + boxscore_df["BB"]

    season_stats = boxscore_df.groupby(["athlete_id", "name", "season"]).agg(
        AB=("AB", "sum"),
        BB=("BB", "sum"),
        PA=("PA", "sum")
    ).reset_index()

    hit_types = get_hit_types(pbp_df)
    season_stats = season_stats.merge(
        hit_types,
        left_on=["athlete_id", "season"],
        right_on=["batter_id", "season"],
        how="left"
    ).fillna(0)

    season_stats["wOBA"] = (
        linear_weights.get("walk", 0)         * season_stats["BB"] +
        linear_weights.get("hit by pitch", 0) * season_stats.get("hit by pitch", 0) +
        linear_weights.get("single", 0)       * season_stats["single"] +
        linear_weights.get("double", 0)       * season_stats["double"] +
        linear_weights.get("triple", 0)       * season_stats["triple"] +
        linear_weights.get("homerun", 0)      * season_stats["homerun"]
    ) / season_stats["PA"].replace(0, np.nan)

    return season_stats


def calculate_war(boxscore_df, woba_df, games_df):
    """
    Calculate WAR for each player.

    Steps:
        1. Calculate league average wOBA
        2. Calculate wRAA (runs above average)
        3. Define replacement level from bottom 20% of qualified players
        4. Calculate batting runs above replacement
        5. Apply positional adjustment
        6. Calculate runs per win from games_df
        7. Divide total runs by RPW to get WAR
        8. Return player WAR dataframe
    """
    pass


if __name__ == "__main__":
    games_df, boxscore_df, pbp_df = load_data()

    re_matrix      = build_run_expectancy(pbp_df)
    linear_weights = calculate_linear_weights(pbp_df, re_matrix)
    woba_df        = calculate_woba(boxscore_df, pbp_df, linear_weights)
    war_df         = calculate_war(boxscore_df, woba_df, games_df)