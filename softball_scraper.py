"""
NCAA D1 Softball Play-by-Play Scraper
======================================
Pulls game data from ESPN's undocumented (but stable) API.
Outputs three CSVs per run:
  - pbp.csv        : play-by-play for every game
  - boxscore.csv   : per-player batting/pitching stats per game
  - games.csv      : game-level metadata (teams, scores, date, venue)

Usage:
  python softball_scraper.py --start 2025-02-01 --end 2025-05-15
  python softball_scraper.py --start 2025-02-01 --end 2025-05-15 --pbp-only
  python softball_scraper.py --start 2025-02-01 --end 2025-05-15 --output-dir ./data
"""

import requests
import pandas as pd
import time
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SCOREBOARD_URL = "http://site.api.espn.com/apis/site/v2/sports/baseball/college-softball/scoreboard"
SUMMARY_URL    = "http://site.api.espn.com/apis/site/v2/sports/baseball/college-softball/summary"
HEADERS        = {"User-Agent": "Mozilla/5.0 (research project)"}
DELAY          = 0.5   # seconds between API calls — be polite


# ── Helpers ───────────────────────────────────────────────────────────────────

def date_range(start: str, end: str):
    """Yield YYYYMMDD strings from start to end (inclusive)."""
    cur = datetime.strptime(start, "%Y-%m-%d")
    stop = datetime.strptime(end, "%Y-%m-%d")
    while cur <= stop:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def get_games_for_date(date_str: str) -> list[dict]:
    """
    Return a list of game metadata dicts for a given date.
    Only returns games that are STATUS_FINAL.
    """
    params = {"dates": date_str, "limit": 200}
    try:
        r = requests.get(SCOREBOARD_URL, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"Scoreboard request failed for {date_str}: {e}")
        return []

    games = []
    for event in r.json().get("events", []):
        comp = event.get("competitions", [{}])[0]
        status = event.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            continue

        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), {})
        away = next((c for c in competitors if c.get("homeAway") == "away"), {})

        games.append({
            "game_id":           event["id"],
            "date":              event["date"][:10],
            "home_team":         home.get("team", {}).get("displayName"),
            "home_team_id":      home.get("team", {}).get("id"),
            "home_score":        home.get("score"),
            "away_team":         away.get("team", {}).get("displayName"),
            "away_team_id":      away.get("team", {}).get("id"),
            "away_score":        away.get("score"),
            "neutral_site":      comp.get("neutralSite", False),
            "venue":             comp.get("venue", {}).get("fullName"),
            "attendance":        comp.get("attendance"),
            "pbp_available":     comp.get("playByPlayAvailable", False),
            "conference_game":   comp.get("conferenceCompetition", False),
        })
    return games


def get_game_summary(game_id: str) -> dict:
    """Fetch the full summary JSON for a game."""
    try:
        r = requests.get(SUMMARY_URL, params={"event": game_id}, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        log.warning(f"Summary request failed for game {game_id}: {e}")
        return {}


# ── PBP Parser ────────────────────────────────────────────────────────────────

def parse_plays(game_id: str, summary: dict) -> list[dict]:
    """
    Extract play-by-play rows from the summary.
    Each row is one discrete play/event in the game.
    """
    plays_raw = summary.get("plays", [])
    rows = []
    for play in plays_raw:
        play_type = play.get("type", {}).get("text", "")

        # Skip inning markers — keep actual play results and pitches
        if play_type in ("Start Inning", "End Inning", "Start Game", "End Game"):
            continue

        # Participants: identify pitcher and batter athlete IDs
        participants = play.get("participants", [])
        pitcher_id = next(
            (p["athlete"]["id"] for p in participants if p.get("type") == "pitcher"), None
        )
        batter_id = next(
            (p["athlete"]["id"] for p in participants if p.get("type") == "batter"), None
        )

        period = play.get("period", {})
        rows.append({
            "game_id":        game_id,
            "play_id":        play.get("id"),
            "sequence":       play.get("sequenceNumber"),
            "inning":         period.get("number"),
            "inning_half":    period.get("type"),        # "Top" / "Bottom"
            "play_type":      play_type,
            "description":    play.get("text"),
            "at_bat_id":      play.get("atBatId"),
            "bat_order":      play.get("batOrder"),
            "batter_id":      batter_id,
            "pitcher_id":     pitcher_id,
            "batter_hand":    play.get("bats", {}).get("abbreviation"),
            "outs_before":    play.get("outs"),
            "balls":          play.get("resultCount", {}).get("balls"),
            "strikes":        play.get("resultCount", {}).get("strikes"),
            "scoring_play":   play.get("scoringPlay", False),
            "score_value":    play.get("scoreValue", 0),
            "away_score":     play.get("awayScore"),
            "home_score":     play.get("homeScore"),
            "team_id":        play.get("team", {}).get("id"),
        })
    return rows


# ── Boxscore Parser ───────────────────────────────────────────────────────────

def parse_boxscore(game_id: str, summary: dict) -> list[dict]:
    """
    Extract per-player batting and pitching stat lines from the boxscore.
    Returns one row per player per stat group (batting / pitching).
    """
    rows = []
    teams = summary.get("boxscore", {}).get("players", [])

    for team_data in teams:
        team_info  = team_data.get("team", {})
        team_id    = team_info.get("id")
        team_name  = team_info.get("displayName")
        stat_groups = team_data.get("statistics", [])

        for group in stat_groups:
            stat_type = group.get("name") or group.get("abbreviation") or "unknown"
            labels    = group.get("labels", [])

            for athlete_entry in group.get("athletes", []):
                athlete = athlete_entry.get("athlete", {})
                stats   = athlete_entry.get("stats", [])

                row = {
                    "game_id":    game_id,
                    "team_id":    team_id,
                    "team_name":  team_name,
                    "athlete_id": athlete.get("id"),
                    "name":       athlete.get("displayName", "").strip(),
                    "position":   athlete_entry.get("position", {}).get("abbreviation"),
                    "starter":    athlete_entry.get("starter", False),
                    "bat_order":  athlete_entry.get("batOrder"),
                    "stat_type":  stat_type,
                }

                # Map label -> value (e.g. "AB" -> "3", "H" -> "1")
                for label, value in zip(labels, stats):
                    row[label] = value

                rows.append(row)

    return rows


# ── Roster Parser (athlete ID -> name lookup) ─────────────────────────────────

def parse_rosters(summary: dict) -> dict[str, str]:
    """Build a {athlete_id: name} dict from the rosters section."""
    lookup = {}
    for team_roster in summary.get("rosters", []):
        for entry in team_roster.get("roster", []):
            athlete = entry.get("athlete", {})
            aid = athlete.get("id")
            name = athlete.get("displayName", "").strip()
            if aid:
                lookup[aid] = name
    return lookup


# ── Main Scraper ──────────────────────────────────────────────────────────────

def scrape(start: str, end: str, pbp_only: bool = False, output_dir: str = "."):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_games   = []
    all_plays   = []
    all_boxscore = []

    dates = list(date_range(start, end))
    log.info(f"Scraping {len(dates)} dates from {start} to {end}")

    for date_str in dates:
        games = get_games_for_date(date_str)
        if not games:
            continue

        log.info(f"{date_str}: found {len(games)} completed games")
        all_games.extend(games)

        for game in games:
            game_id      = game["game_id"]
            pbp_flag     = game["pbp_available"]
            home         = game["home_team"]
            away         = game["away_team"]

            log.info(f"  Fetching {away} @ {home} (id={game_id}, pbp={pbp_flag})")

            summary = get_game_summary(game_id)
            if not summary:
                continue

            # Enrich game row with roster lookup for later joins
            roster_lookup = parse_rosters(summary)
            game["roster_count"] = len(roster_lookup)

            # Play-by-play
            plays = parse_plays(game_id, summary)
            all_plays.extend(plays)

            if not pbp_only:
                box = parse_boxscore(game_id, summary)
                all_boxscore.extend(box)

            time.sleep(DELAY)

    # ── Save outputs ──────────────────────────────────────────────────────────
    games_df = pd.DataFrame(all_games)
    games_path = output_path / "games.csv"
    games_df.to_csv(games_path, index=False)
    log.info(f"Saved {len(games_df)} games -> {games_path}")

    plays_df = pd.DataFrame(all_plays)
    pbp_path = output_path / "pbp.csv"
    plays_df.to_csv(pbp_path, index=False)
    log.info(f"Saved {len(plays_df)} plays -> {pbp_path}")

    if not pbp_only and all_boxscore:
        box_df = pd.DataFrame(all_boxscore)
        box_path = output_path / "boxscore.csv"
        box_df.to_csv(box_path, index=False)
        log.info(f"Saved {len(box_df)} player-game rows -> {box_path}")

    # ── Coverage summary ──────────────────────────────────────────────────────
    total_games    = len(games_df)
    games_with_pbp = len(plays_df["game_id"].unique()) if not plays_df.empty else 0
    log.info(
        f"\n{'='*50}\n"
        f"  Total games:       {total_games}\n"
        f"  Games with PBP:    {games_with_pbp} ({games_with_pbp/max(total_games,1)*100:.1f}%)\n"
        f"  Total plays:       {len(plays_df)}\n"
        f"  Output dir:        {output_path.resolve()}\n"
        f"{'='*50}"
    )

    return games_df, plays_df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCAA D1 Softball Scraper (ESPN API)")
    parser.add_argument("--start",      required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",        required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--pbp-only",   action="store_true", help="Skip boxscore parsing")
    parser.add_argument("--output-dir", default=".",  help="Directory to save CSVs (default: .)")
    args = parser.parse_args()

    scrape(
        start      = args.start,
        end        = args.end,
        pbp_only   = args.pbp_only,
        output_dir = args.output_dir,
    )
