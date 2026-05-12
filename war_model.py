import pandas as pd
import numpy as np
import glob
import os


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_data():
    """
    Load and combine all seasons of ESPN games, boxscore, and PBP data.

    Expects CSVs named {type}_{year}.csv inside each subdirectory.
    Adds a 'season' column derived from the filename year.
    Deduplicates boxscore on (athlete_id, game_id).
    """
    def load_csv(path):
        df_list = []
        for filename in glob.glob(os.path.join(path, "*.csv")):
            df   = pd.read_csv(filename)
            year = filename.split("_")[1].split(".")[0]
            df["season"] = int(year)
            df_list.append(df)
        return pd.concat(df_list, ignore_index=True)

    games_df    = load_csv(r"D:\softball-war-model\data\games")
    boxscore_df = load_csv(r"D:\softball-war-model\data\boxscore")
    boxscore_df = boxscore_df.drop_duplicates(subset=["athlete_id", "game_id"])
    pbp_df      = load_csv(r"D:\softball-war-model\data\pbp")

    print(f"Games:     {games_df.shape}")
    print(f"Boxscore:  {boxscore_df.shape}")
    print(f"PBP:       {pbp_df.shape}\n")

    return games_df, boxscore_df, pbp_df


# ── Run Expectancy ─────────────────────────────────────────────────────────────

def build_run_expectancy(pbp_df):
    """
    Build a 3-state run expectancy (RE) matrix based on outs only.

    Uses play-by-play data to compute the average runs scored from
    each plate appearance to the end of the half inning, grouped by
    outs before the play (0, 1, or 2).

    Known limitation: ignores base state. A full 24-state base-out
    RE matrix would be more accurate but requires base runner columns
    not available in the ESPN PBP feed.

    Returns a Series indexed by outs_before (0/1/2).
    """
    pbp = pbp_df[pbp_df["play_type"] == "Play Result"].copy()

    pbp["batting_score"] = np.where(
        pbp["inning_half"] == "Top",
        pbp["away_score"],
        pbp["home_score"]
    )
    pbp["inning_end_score"] = pbp.groupby(
        ["game_id", "inning", "inning_half"]
    )["batting_score"].transform("max")

    pbp["runs_to_end"] = pbp["inning_end_score"] - pbp["batting_score"]
    pbp = pbp[pbp["outs_before"] <= 2]

    return pbp.groupby("outs_before")["runs_to_end"].mean()


# ── Event Classification ───────────────────────────────────────────────────────

def classify_event(description):
    """Classify a PBP play description into a batting event type."""
    desc = description.lower()
    if "home run" in desc or "homered" in desc: return "homerun"
    if "tripled"    in desc:                    return "triple"
    if "doubled"    in desc:                    return "double"
    if "singled"    in desc:                    return "single"
    if "walked"     in desc:                    return "walk"
    if "hit by pitch" in desc:                  return "hit by pitch"
    if "struck out" in desc:                    return "strike out"
    if any(p in desc for p in ["grounded out", "flied out", "lined out", "popped up"]):
        return "out"
    if "error"           in desc: return "error"
    if "fielder's choice" in desc: return "fielder's choice"
    return "other"


def find_outs(description, event_type):
    """Infer outs recorded on a play from its description and event type."""
    if "double play" in description.lower():
        return 2
    if event_type in ["single", "double", "triple", "homerun", "walk", "hit by pitch", "error"]:
        return 0
    if event_type in ["out", "fielder's choice", "strike out"]:
        return 1
    return 0


# ── Linear Weights ─────────────────────────────────────────────────────────────

def calculate_linear_weights(pbp_df, re_matrix):
    """
    Derive linear weights for each event type from the RE matrix.

    For each play: run_value = score_value + RE_after - RE_before
    Linear weight for an event = average run_value across all instances.

    Returns a dict mapping event type -> linear weight (run value).
    """
    pbp = pbp_df[pbp_df["play_type"] == "Play Result"].copy()

    pbp["event_type"] = pbp["description"].apply(classify_event)
    pbp["outs_added"] = pbp.apply(
        lambda row: find_outs(row["description"], row["event_type"]), axis=1
    )
    pbp["outs_after"] = (pbp["outs_before"] + pbp["outs_added"]).clip(upper=3)
    pbp["RE_before"]  = pbp["outs_before"].map(re_matrix)
    pbp["RE_after"]   = pbp["outs_after"].map(re_matrix)
    pbp["run_value"]  = (pbp["score_value"] + pbp["RE_after"]) - pbp["RE_before"]

    return pbp.groupby("event_type")["run_value"].mean().to_dict()


# ── wOBA (ESPN Boxscore — Historical Seasons) ──────────────────────────────────

def get_hit_types(pbp_df):
    """
    Aggregate hit type counts (1B/2B/3B/HR) per player per season from PBP.
    Used by calculate_woba() for ESPN-based historical season calculations.
    Returns a DataFrame with columns: batter_id, season, single, double, triple, homerun.
    """
    pbp = pbp_df[pbp_df["play_type"] == "Play Result"].copy()
    pbp["event_type"] = pbp["description"].apply(classify_event)

    hit_types = ["single", "double", "triple", "homerun"]
    pbp = pbp[pbp["event_type"].isin(hit_types)]

    counts = pbp.groupby(["batter_id", "season", "event_type"]).size().unstack(fill_value=0)
    counts.columns.name = None
    counts = counts.reset_index()

    for col in hit_types:
        if col not in counts.columns:
            counts[col] = 0

    return counts


def calculate_woba(boxscore_df, pbp_df, linear_weights):
    """
    Calculate scaled wOBA from ESPN boxscore + PBP data (historical seasons).

    Note: ESPN boxscore has ~84% coverage and no HBP/SF columns.
    PA = AB + BB only. Hit types sourced from PBP (27% game coverage).
    For current-season WAR, use calculate_woba_ncaa() instead.

    Returns (season_stats_df, lg_wOBA, woba_scale).
    """
    for col in ["AB", "BB", "H", "IP"]:
        boxscore_df[col] = pd.to_numeric(boxscore_df[col], errors="coerce")

    # Drop pure pitching rows — their H = hits allowed, BB = walks issued
    boxscore_df = boxscore_df[
        ~(boxscore_df["IP"].notna() & (boxscore_df["AB"].fillna(0) == 0))
    ].copy()

    for col in ["AB", "BB", "H"]:
        boxscore_df[col] = boxscore_df[col].fillna(0)

    boxscore_df["PA"] = boxscore_df["AB"] + boxscore_df["BB"]

    season_stats = boxscore_df.groupby(["athlete_id", "name", "season"]).agg(
        AB=("AB", "sum"),
        BB=("BB", "sum"),
        H=("H",   "sum"),
        PA=("PA", "sum")
    ).reset_index()

    hit_types    = get_hit_types(pbp_df)
    season_stats = season_stats.merge(
        hit_types,
        left_on=["athlete_id", "season"],
        right_on=["batter_id",  "season"],
        how="left"
    ).fillna(0)

    season_stats["wOBA_num"] = (
        linear_weights.get("walk",        0) * season_stats["BB"]      +
        linear_weights.get("single",      0) * season_stats["single"]  +
        linear_weights.get("double",      0) * season_stats["double"]  +
        linear_weights.get("triple",      0) * season_stats["triple"]  +
        linear_weights.get("homerun",     0) * season_stats["homerun"]
    )

    season_stats["wOBA_raw"] = season_stats["wOBA_num"] / season_stats["PA"].replace(0, np.nan)

    qualified = season_stats[season_stats["PA"] > 0]
    lg_wOBA   = qualified["wOBA_num"].sum() / qualified["PA"].sum()
    lg_OBP    = (qualified["H"].sum() + qualified["BB"].sum()) / \
                (qualified["AB"].sum() + qualified["BB"].sum())

    woba_scale = lg_OBP / lg_wOBA if lg_wOBA != 0 else 1.0
    print(f"lg_wOBA: {lg_wOBA:.3f} | lg_OBP: {lg_OBP:.3f} | wOBA scale: {woba_scale:.3f}")

    season_stats["wOBA"] = season_stats["wOBA_raw"] * woba_scale

    return season_stats, lg_wOBA, woba_scale


# ── wOBA (NCAA Stats — Current Season) ────────────────────────────────────────

def calculate_woba_ncaa(ncaa_df, linear_weights):
    """
    Calculate scaled wOBA from official NCAA season stats (current season).

    Uses complete NCAA data with full PA counts and exact hit type breakdown
    (1B/2B/3B/HR). Preferred over calculate_woba() for current-season WAR.

    PA = AB + BB + HBP + SF
    wOBA scaled so league average wOBA == league average OBP.

    Returns (ncaa_df with wOBA column, lg_wOBA, woba_scale).
    """
    for col in ["AB", "H", "2B", "3B", "HR", "BB", "HBP", "SF"]:
        ncaa_df[col] = pd.to_numeric(ncaa_df[col], errors="coerce").fillna(0)

    ncaa_df["1B"] = ncaa_df["H"] - ncaa_df["2B"] - ncaa_df["3B"] - ncaa_df["HR"]
    ncaa_df["PA"] = ncaa_df["AB"] + ncaa_df["BB"] + ncaa_df["HBP"] + ncaa_df["SF"]

    ncaa_df["wOBA_num"] = (
        linear_weights.get("walk",         0) * ncaa_df["BB"]  +
        linear_weights.get("hit by pitch", 0) * ncaa_df["HBP"] +
        linear_weights.get("single",       0) * ncaa_df["1B"]  +
        linear_weights.get("double",       0) * ncaa_df["2B"]  +
        linear_weights.get("triple",       0) * ncaa_df["3B"]  +
        linear_weights.get("homerun",      0) * ncaa_df["HR"]
    )

    ncaa_df["wOBA_raw"] = ncaa_df["wOBA_num"] / ncaa_df["PA"].replace(0, np.nan)

    qualified  = ncaa_df[ncaa_df["PA"] > 0]
    lg_wOBA    = qualified["wOBA_num"].sum() / qualified["PA"].sum()
    lg_OBP     = (
        qualified["H"].sum() + qualified["BB"].sum() + qualified["HBP"].sum()
    ) / (
        qualified["AB"].sum() + qualified["BB"].sum() +
        qualified["HBP"].sum() + qualified["SF"].sum()
    )

    woba_scale = lg_OBP / lg_wOBA if lg_wOBA != 0 else 1.0
    print(f"lg_wOBA: {lg_wOBA:.3f} | lg_OBP: {lg_OBP:.3f} | wOBA scale: {woba_scale:.3f}")

    ncaa_df["wOBA"]  = ncaa_df["wOBA_raw"] * woba_scale
    ncaa_df["name"]  = ncaa_df["Player"]
    ncaa_df["season"] = 2026

    return ncaa_df, lg_wOBA, woba_scale


# ── WAR ────────────────────────────────────────────────────────────────────────

def calculate_war(woba_df, games_df, lg_wOBA, woba_scale, min_pa=100):
    """
    Calculate batting WAR for each player.

    RPW (runs per win) derived from average total runs per game across
    all games in games_df.

    Replacement level = bottom 20% wOBA among qualified players (PA >= min_pa).

    batting_runs = (wOBA - replacement_wOBA) / woba_scale * PA
    WAR          = batting_runs / RPW

    Note: batting WAR only — does not include pitching, fielding, or baserunning.

    Returns a DataFrame sorted by WAR descending, filtered to qualified players.
    """
    total_runs  = games_df["home_score"].sum() + games_df["away_score"].sum()
    total_games = len(games_df)
    rpw         = total_runs / total_games
    print(f"Total runs: {total_runs:.0f} | Total games: {total_games} | RPW: {rpw:.2f}")

    qualified        = woba_df[woba_df["PA"] >= min_pa].copy()
    replacement_wOBA = qualified["wOBA"].quantile(0.20)
    print(f"Qualified players: {len(qualified)} | Replacement wOBA: {replacement_wOBA:.3f}")

    woba_df["wRAA"]         = (woba_df["wOBA"] - lg_wOBA)          / woba_scale * woba_df["PA"]
    woba_df["batting_runs"] = (woba_df["wOBA"] - replacement_wOBA) / woba_scale * woba_df["PA"]
    woba_df["WAR"]          = woba_df["batting_runs"] / rpw

    war_df = woba_df[[
        "name", "season", "PA", "AB", "H", "BB",
        "wOBA", "wRAA", "batting_runs", "WAR"
    ]].copy()

    war_df = war_df[war_df["PA"] >= min_pa] \
                   .sort_values("WAR", ascending=False) \
                   .reset_index(drop=True)

    return war_df


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    games_df, boxscore_df, pbp_df = load_data()

    re_matrix      = build_run_expectancy(pbp_df)
    linear_weights = calculate_linear_weights(pbp_df, re_matrix)

    # Current season: use complete NCAA stats
    ncaa_df              = pd.read_csv(r"D:\softball-war-model\data\ncaa_stats_2026.csv")
    woba_df, lg_wOBA, woba_scale = calculate_woba_ncaa(ncaa_df, linear_weights)
    war_df               = calculate_war(woba_df, games_df, lg_wOBA, woba_scale, min_pa=100)

    print(war_df.head(25).to_string())
