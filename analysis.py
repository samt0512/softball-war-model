import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

sys.path.insert(0, r"D:\softball-war-model")
from war_model import (
    load_data,
    build_run_expectancy,
    calculate_linear_weights,
    calculate_woba_ncaa,
    calculate_war,
)


# ── Constants ──────────────────────────────────────────────────────────────────

NCAA_STATS_CSV = r"D:\softball-war-model\data\ncaa_stats_2026.csv"
TEAM_IDS_CSV   = r"D:\softball-war-model\data\ncaa_team_ids.csv"
MIN_PA         = 100  # Default qualifier for leaderboards


# ── Pipeline ───────────────────────────────────────────────────────────────────

def build_war_df(min_pa=1):
    """
    Run the full oWAR pipeline and return a war_df with team_id attached.

    Sets min_pa=1 by default so all players are included — callers can
    filter to qualified players using the MIN_PA constant or their own threshold.
    """
    ncaa_df                      = pd.read_csv(NCAA_STATS_CSV)
    games_df, boxscore_df, pbp_df = load_data()

    re_matrix      = build_run_expectancy(pbp_df)
    linear_weights = calculate_linear_weights(pbp_df, re_matrix)
    woba_df, lg_wOBA, woba_scale = calculate_woba_ncaa(ncaa_df, linear_weights)
    war_df                       = calculate_war(woba_df, games_df, lg_wOBA, woba_scale, min_pa=min_pa)

    # Merge team_id back using name + AB as a composite key
    # (name alone isn't unique — multiple players can share a name)
    lookup      = ncaa_df[["Player", "AB", "team_id"]].copy()
    lookup["AB"] = pd.to_numeric(lookup["AB"], errors="coerce")
    war_df      = war_df.merge(
        lookup,
        left_on=["name", "AB"],
        right_on=["Player", "AB"],
        how="left"
    ).drop(columns=["Player"])

    return war_df


# ── Query Functions ────────────────────────────────────────────────────────────

def get_team_leaderboard(team_id, war_df=None, min_pa=1):
    """
    Return a team's roster ranked by oWAR.

    Args:
        team_id: NCAA stats team ID (e.g. 613620 for Charlotte)
        war_df:  Pre-built war_df — pass one in to avoid rebuilding the pipeline.
                 If None, builds from scratch.
        min_pa:  Minimum PA to include. Defaults to 1 (all players).

    Returns a DataFrame with columns: name, PA, AB, H, BB, wOBA, oWAR.
    """
    if war_df is None:
        war_df = build_war_df(min_pa=min_pa)

    team_df = war_df[war_df["team_id"] == team_id].copy()
    team_df = team_df[team_df["PA"] >= min_pa]
    team_df = team_df.sort_values("oWAR", ascending=False).reset_index(drop=True)
    team_df["oWAR"] = team_df["oWAR"].round(2)
    team_df["wOBA"] = team_df["wOBA"].round(3)

    return team_df[["name", "PA", "AB", "H", "BB", "wOBA", "oWAR"]]


def get_overall_leaderboard(war_df=None, min_pa=MIN_PA, n=25):
    """
    Return the top N players across all D1 teams by oWAR.

    Args:
        war_df: Pre-built war_df — pass one in to avoid rebuilding the pipeline.
        min_pa: Minimum PA qualifier. Defaults to MIN_PA (100).
        n:      Number of players to return. Defaults to 25.

    Returns a DataFrame with columns: name, team_id, PA, AB, H, BB, wOBA, oWAR.
    """
    if war_df is None:
        war_df = build_war_df(min_pa=min_pa)

    leaderboard = war_df[war_df["PA"] >= min_pa] \
                        .sort_values("oWAR", ascending=False) \
                        .head(n) \
                        .reset_index(drop=True)
    leaderboard["oWAR"] = leaderboard["oWAR"].round(2)
    leaderboard["wOBA"] = leaderboard["wOBA"].round(3)

    return leaderboard[["name", "team_id", "PA", "AB", "H", "BB", "wOBA", "oWAR"]]


def get_player(name, war_df=None):
    """
    Look up a specific player by name.

    If multiple players share the same name, returns all matches
    with team_id so you can distinguish them.

    Args:
        name:   Player name (case-sensitive, e.g. 'Jenna Lord')
        war_df: Pre-built war_df — pass one in to avoid rebuilding the pipeline.

    Returns a DataFrame row(s) for the matching player(s).
    """
    if war_df is None:
        war_df = build_war_df()

    matches = war_df[war_df["name"] == name].reset_index(drop=True)

    if matches.empty:
        print(f"No player found named '{name}'")
        return None

    return matches[["name", "team_id", "PA", "AB", "H", "BB", "wOBA", "oWAR"]]


# ── Visualization ──────────────────────────────────────────────────────────────

def plot_team_leaderboard(team_id, team_name, war_df=None, min_pa=1,
                          save_path=None):
    """
    Save a clean table image of a team's oWAR leaderboard as a PNG.

    Args:
        team_id:   NCAA stats team ID
        team_name: Display name for the title (e.g. 'Charlotte 49ers')
        war_df:    Pre-built war_df — pass one in to avoid rebuilding the pipeline.
        min_pa:    Minimum PA to include. Defaults to 1.
        save_path: File path to save the PNG. If None, displays interactively.
    """
    df = get_team_leaderboard(team_id, war_df=war_df, min_pa=min_pa)

    col_labels = ["#", "Player", "PA", "AB", "H", "BB", "wOBA", "oWAR"]
    rows = []
    for i, row in df.iterrows():
        rows.append([
            i + 1,
            row["name"],
            int(row["PA"]),
            int(row["AB"]),
            int(row["H"]),
            int(row["BB"]),
            f'{row["wOBA"]:.3f}',
            f'{row["oWAR"]:.2f}',
        ])

    fig, ax = plt.subplots(figsize=(10, len(rows) * 0.38 + 1.4))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # Header row styling
    for col in range(len(col_labels)):
        cell = table[0, col]
        cell.set_facecolor("#00703C")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#ffffff")

    # Data row styling
    for row_idx in range(1, len(rows) + 1):
        bg = "#f7f7f7" if row_idx % 2 == 0 else "#ffffff"
        for col in range(len(col_labels)):
            cell = table[row_idx, col]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#e0e0e0")
            # Right-align numeric columns
            if col >= 2:
                cell.get_text().set_ha("right")
            # Left-align player name
            if col == 1:
                cell.get_text().set_ha("left")

    # Column widths
    col_widths = [0.04, 0.22, 0.07, 0.07, 0.07, 0.07, 0.09, 0.09]
    for col, width in enumerate(col_widths):
        for row_idx in range(len(rows) + 1):
            table[row_idx, col].set_width(width)

    fig.suptitle(
        f"{team_name} — oWAR Leaderboard (2025-26)",
        fontsize=13, fontweight="bold", y=0.98
    )
    fig.text(
        0.99, 0.01,
        "Batting only — excludes fielding, baserunning, and pitching",
        ha="right", va="bottom", fontsize=8, color="#999999", style="italic"
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build the pipeline once and reuse war_df for all queries
    war_df = build_war_df()

    print("=" * 60)
    print("D1 SOFTBALL oWAR LEADERBOARD — 2025-26")
    print("=" * 60)
    print(get_overall_leaderboard(war_df=war_df, n=25).to_string())

    print()
    print("=" * 60)
    print("CHARLOTTE 49ERS — 2025-26")
    print("=" * 60)
    print(get_team_leaderboard(613620, war_df=war_df).to_string())

    # Plot Charlotte leaderboard
    plot_team_leaderboard(
        team_id   = 613620,
        team_name = "Charlotte 49ers",
        war_df    = war_df,
        save_path = r"D:\softball-war-model\output\charlotte_owar_2026.png"
    )