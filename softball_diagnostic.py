"""
Softball Data Diagnostic
=========================
Audits the CSVs produced by softball_scraper.py and answers:
  1. Game coverage & PBP availability
  2. Unique players and PA qualification distribution
  3. PBP base-out state coverage and play type distribution
  4. Data quality flags (missing values, duplicate rows, etc.)

Usage:
  python softball_diagnostic.py --data-dir ./data/2025
  python softball_diagnostic.py --data-dir ./data/2025 --min-pa 100
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MIN_PA = 150   # Minimum PA to be considered a "qualified" hitter


# ── Helpers ───────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def subsection(title: str):
    print(f"\n  -- {title} --")


# ── Game Coverage ─────────────────────────────────────────────────────────────

def audit_games(games: pd.DataFrame):
    section("GAME COVERAGE")

    total        = len(games)
    with_pbp     = games["pbp_available"].sum()
    neutral      = games["neutral_site"].sum()
    conf_games   = games["conference_game"].sum()

    print(f"  Total games:            {total}")
    print(f"  Games with PBP:         {with_pbp} ({with_pbp/total*100:.1f}%)")
    print(f"  Conference games:       {conf_games} ({conf_games/total*100:.1f}%)")
    print(f"  Neutral site games:     {neutral}")

    subsection("Games by date (first and last)")
    games["date"] = pd.to_datetime(games["date"])
    print(f"  Season start:  {games['date'].min().date()}")
    print(f"  Season end:    {games['date'].max().date()}")

    subsection("Games per week")
    weekly = games.groupby(games["date"].dt.isocalendar().week).size()
    print(weekly.to_string())

    subsection("PBP availability by conference game")
    cross = pd.crosstab(
        games["conference_game"],
        games["pbp_available"],
        margins=True
    )
    index_map = {False: "Non-conf", True: "Conf", "All": "Total"}
    col_map   = {False: "No PBP",   True: "Has PBP", "All": "Total"}
    cross.index   = [index_map.get(i, str(i)) for i in cross.index]
    cross.columns = [col_map.get(c, str(c))   for c in cross.columns]
    print(cross.to_string())


# ── Player / Batting Audit ────────────────────────────────────────────────────

def audit_boxscore(box: pd.DataFrame, min_pa: int):
    section("BOXSCORE — PLAYER COVERAGE")

    # Keep batting rows (they have AB column populated)
    batting = box[pd.to_numeric(box["AB"], errors="coerce").notna()].copy()
    batting["AB"]  = pd.to_numeric(batting["AB"],  errors="coerce").fillna(0)
    batting["BB"]  = pd.to_numeric(batting["BB"],  errors="coerce").fillna(0)
    batting["H"]   = pd.to_numeric(batting["H"],   errors="coerce").fillna(0)
    batting["HR"]  = pd.to_numeric(batting["HR"],  errors="coerce").fillna(0)
    batting["R"]   = pd.to_numeric(batting["R"],   errors="coerce").fillna(0)
    batting["RBI"] = pd.to_numeric(batting["RBI"], errors="coerce").fillna(0)
    batting["K"]   = pd.to_numeric(batting["K"],   errors="coerce").fillna(0)
    batting["PA"]  = batting["AB"] + batting["BB"]

    subsection("Overall")
    print(f"  Total player-game rows:   {len(batting)}")
    print(f"  Unique athletes:          {batting['athlete_id'].nunique()}")
    print(f"  Unique teams:             {batting['team_name'].nunique()}")

    # Season aggregates per player
    player_season = batting.groupby(["athlete_id", "name", "team_name"]).agg(
        games = ("game_id", "nunique"),
        PA    = ("PA",  "sum"),
        AB    = ("AB",  "sum"),
        H     = ("H",   "sum"),
        HR    = ("HR",  "sum"),
        BB    = ("BB",  "sum"),
        K     = ("K",   "sum"),
        R     = ("R",   "sum"),
        RBI   = ("RBI", "sum"),
    ).reset_index()

    player_season["BA"]  = (player_season["H"]  / player_season["AB"].replace(0, np.nan)).round(3)
    player_season["KP"]  = (player_season["K"]  / player_season["PA"].replace(0, np.nan)).round(3)
    player_season["BBP"] = (player_season["BB"] / player_season["PA"].replace(0, np.nan)).round(3)

    subsection("PA distribution across all players")
    bins   = [0, 25, 50, 100, 150, 200, 300, 400, 9999]
    labels = ["1-25", "26-50", "51-100", "101-150", "151-200", "201-300", "301-400", "400+"]
    player_season["pa_bucket"] = pd.cut(player_season["PA"], bins=bins, labels=labels)
    dist = player_season["pa_bucket"].value_counts().sort_index()
    for bucket, count in dist.items():
        bar = "█" * (count // 5)
        print(f"    {bucket:>8} PA : {count:>4}  {bar}")

    qualified = player_season[player_season["PA"] >= min_pa]
    subsection(f"Qualified hitters (>= {min_pa} PA)")
    print(f"  Qualified players:  {len(qualified)}")
    print(f"  Avg PA:             {qualified['PA'].mean():.0f}")
    print(f"  Avg games played:   {qualified['games'].mean():.1f}")

    subsection(f"Top 15 by PA (qualified >= {min_pa})")
    top = qualified.sort_values("PA", ascending=False).head(15)
    print(top[["name", "team_name", "games", "PA", "AB", "H", "HR", "BB", "BA"]].to_string(index=False))

    subsection("Batting stat completeness (null rates)")
    for col in ["AB", "H", "HR", "BB", "K", "R", "RBI"]:
        null_pct = batting[col].isna().mean() * 100
        print(f"    {col:>4}: {null_pct:.1f}% missing")

    # Pitching rows (have IP)
    pitching = box[pd.to_numeric(box.get("IP", pd.Series(dtype=float)), errors="coerce").notna()].copy()
    if len(pitching) > 0:
        pitching["IP"] = pd.to_numeric(pitching["IP"], errors="coerce")
        subsection("Pitching coverage")
        print(f"  Pitcher-game rows:   {len(pitching)}")
        print(f"  Unique pitchers:     {pitching['athlete_id'].nunique()}")

    return player_season, qualified


# ── PBP Audit ─────────────────────────────────────────────────────────────────

def audit_pbp(pbp: pd.DataFrame):
    section("PLAY-BY-PLAY COVERAGE")

    total_plays = len(pbp)
    total_games = pbp["game_id"].nunique()

    print(f"  Total plays:            {total_plays}")
    print(f"  Games with PBP:         {total_games}")
    print(f"  Avg plays per game:     {total_plays/total_games:.0f}")

    subsection("Play type distribution")
    type_counts = pbp["play_type"].value_counts()
    for ptype, count in type_counts.items():
        pct = count / total_plays * 100
        print(f"    {ptype:<25} {count:>6}  ({pct:.1f}%)")

    # Filter to actual result plays for base-out analysis
    results = pbp[pbp["play_type"] == "Play Result"].copy()
    results["outs_before"] = pd.to_numeric(results["outs_before"], errors="coerce")
    results["inning"]      = pd.to_numeric(results["inning"],      errors="coerce")

    subsection("Play Result breakdown — inning half")
    print(results["inning_half"].value_counts().to_string())

    subsection("Outs distribution in Play Results")
    print(results["outs_before"].value_counts().sort_index().to_string())

    subsection("Scoring plays")
    scoring = results[results["scoring_play"] == True]
    print(f"  Scoring plays:          {len(scoring)} ({len(scoring)/len(results)*100:.1f}% of play results)")

    subsection("Batter/Pitcher ID coverage in Play Results")
    batter_null  = results["batter_id"].isna().mean()  * 100
    pitcher_null = results["pitcher_id"].isna().mean() * 100
    print(f"  Missing batter_id:   {batter_null:.1f}%")
    print(f"  Missing pitcher_id:  {pitcher_null:.1f}%")

    subsection("Play descriptions — action keyword frequency")
    desc = results["description"].dropna().str.lower()
    keywords = {
        "strikeout":  desc.str.contains("struck out|strikeout").sum(),
        "walk":       desc.str.contains("walked|walk").sum(),
        "single":     desc.str.contains("singled|single").sum(),
        "double":     desc.str.contains("doubled|double").sum(),
        "triple":     desc.str.contains("tripled|triple").sum(),
        "home run":   desc.str.contains("home run|homered").sum(),
        "ground out": desc.str.contains("grounded out").sum(),
        "fly out":    desc.str.contains("flied out|fly out").sum(),
        "line out":   desc.str.contains("lined out").sum(),
        "error":      desc.str.contains("reached on").sum(),
        "sac bunt":   desc.str.contains("sacrificed|sac bunt|bunted").sum(),
        "hit by pitch": desc.str.contains("hit by pitch").sum(),
    }
    for event, count in sorted(keywords.items(), key=lambda x: -x[1]):
        print(f"    {event:<18} {count:>6}")

    subsection("PBP inning coverage (how deep do games go?)")
    max_inning = results.groupby("game_id")["inning"].max()
    inning_dist = max_inning.value_counts().sort_index()
    print(inning_dist.to_string())


# ── Data Quality ──────────────────────────────────────────────────────────────

def audit_quality(games: pd.DataFrame, box: pd.DataFrame, pbp: pd.DataFrame):
    section("DATA QUALITY FLAGS")

    subsection("Duplicate game IDs in games.csv")
    dupes = games["game_id"].duplicated().sum()
    print(f"  Duplicate game rows: {dupes}" + (" ✓" if dupes == 0 else " ← investigate"))

    subsection("Games in boxscore not in games.csv")
    box_ids   = set(box["game_id"].unique())
    games_ids = set(games["game_id"].unique())
    missing   = box_ids - games_ids
    print(f"  Orphaned boxscore games: {len(missing)}" + (" ✓" if not missing else f" ← {list(missing)[:5]}"))

    subsection("Games with PBP flag but no plays in pbp.csv")
    pbp_ids      = set(pbp["game_id"].unique())
    flagged_pbp  = set(games[games["pbp_available"]]["game_id"].astype(str))
    pbp_ids_str  = {str(x) for x in pbp_ids}
    missing_pbp  = flagged_pbp - pbp_ids_str
    print(f"  Flagged but missing PBP: {len(missing_pbp)}" + (" ✓" if not missing_pbp else " ← investigate"))

    subsection("Score sanity check (games where both scores are 0)")
    zero_games = games[(games["home_score"].astype(str) == "0") & (games["away_score"].astype(str) == "0")]
    print(f"  0-0 final score games: {len(zero_games)}" + (" ✓" if len(zero_games) == 0 else " ← investigate"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Softball data diagnostic")
    parser.add_argument("--data-dir", required=True, help="Directory containing games.csv, pbp.csv, boxscore.csv")
    parser.add_argument("--min-pa",   type=int, default=DEFAULT_MIN_PA, help="Minimum PA to qualify (default: 150)")
    args = parser.parse_args()

    data_path = Path(args.data_dir)

    print(f"\nLoading data from {data_path.resolve()}...")

    games = pd.read_csv(data_path / "games.csv")
    box   = pd.read_csv(data_path / "boxscore.csv")
    pbp   = pd.read_csv(data_path / "pbp.csv")

    print(f"  games.csv:     {len(games):,} rows")
    print(f"  boxscore.csv:  {len(box):,} rows")
    print(f"  pbp.csv:       {len(pbp):,} rows")

    audit_games(games)
    player_season, qualified = audit_boxscore(box, args.min_pa)
    audit_pbp(pbp)
    audit_quality(games, box, pbp)

    print(f"\n{'='*60}")
    print("  DONE — review any ← flags above before proceeding")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
