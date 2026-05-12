# College Softball WAR Model

A Wins Above Replacement (WAR) model for NCAA D1 college softball — the first publicly available framework of its kind.

## Overview

This project builds a complete batting WAR leaderboard for D1 college softball. It combines ESPN play-by-play data (for run expectancy and linear weights) with official NCAA season stats (for complete player batting lines), producing a credible, publishable WAR leaderboard for the current D1 season.

## Methodology

### Linear Weights
Derived from ESPN play-by-play data using a run expectancy matrix. For each play:

```
run_value = score_value + RE_after - RE_before
```

Linear weight for each event type (single, double, triple, HR, walk, HBP) is the average run value across all instances in the dataset.

**Known limitation:** The RE matrix uses 3 states (outs only) rather than the full 24 base-out states. This compresses linear weight values, which the wOBA scaling step corrects for at the league level.

### wOBA
Weighted on-base average using linear weights applied to official NCAA season stats:

```
wOBA_num = lw_BB * BB + lw_HBP * HBP + lw_1B * 1B + lw_2B * 2B + lw_3B * 3B + lw_HR * HR
wOBA     = wOBA_num / PA * woba_scale
```

Scaled so that league average wOBA equals league average OBP.

### WAR
```
wRAA         = (wOBA - lg_wOBA) / woba_scale * PA
batting_runs = (wOBA - replacement_wOBA) / woba_scale * PA
WAR          = batting_runs / RPW
```

- **Replacement level:** bottom 20% wOBA among qualified players (min 100 PA)
- **RPW:** average total runs per game derived from all games in dataset

## Data Sources

### ESPN API (play-by-play and run expectancy)
- Undocumented but stable JSON endpoints (`site.api.espn.com`)
- 3 seasons scraped: 2023, 2024, 2025
- ~7,200 games, ~353,000 play-by-play rows
- ~27% of games have play-by-play available
- Used exclusively for building the RE matrix and linear weights

### NCAA Stats (`stats.ncaa.org`)
- Official source for complete D1 season batting stats
- Scraped via Playwright (headless=False required — Akamai bot protection)
- 308 D1 softball teams, ~5,400 players for 2025-26 season
- Provides: AB, H, 2B, 3B, HR, BB, HBP, SF — full PA and hit type breakdown
- Used for all player-level wOBA and WAR inputs

## Project Structure

```
softball-war-model/
├── softball_scraper.py      # Scrapes ESPN API → games/boxscore/pbp CSVs
├── softball_diagnostic.py   # Data quality audit tool
├── combine.py               # Merges split season data with deduplication
├── war_model.py             # Core WAR calculation pipeline
├── ncaa_scraper.py          # Scrapes NCAA stats → complete season batting stats
└── data/                    # Local data directory (not tracked in git)
    ├── games/
    ├── boxscore/
    ├── pbp/
    ├── ncaa_team_ids.csv    # D1 softball team IDs for current season
    └── ncaa_stats_2026.csv  # Complete 2025-26 batting stats (all D1 players)
```

## Stack

- Python, pandas, NumPy
- Playwright (NCAA stats scraping)
- BeautifulSoup (HTML parsing)
- Data stored locally as CSVs (not included in repo)

## Current Status

Pipeline is fully functional end-to-end for the 2025-26 season:

- ✅ ESPN data pipeline (scraping, combining, deduplication)
- ✅ Run expectancy matrix
- ✅ Linear weights from PBP
- ✅ NCAA stats scraper (308 D1 teams, ~5,400 players)
- ✅ wOBA with proper scaling (lg_wOBA = 0.060, lg_OBP = 0.376)
- ✅ WAR leaderboard (2,439 qualified players at 100 PA minimum)

**Known limitations:**
- Batting WAR only — pitching, fielding, and baserunning not yet included
- 3-state RE matrix (outs only) — full 24-state base-out matrix is v2
- Linear weights derived from 2023-25 ESPN PBP, applied to 2025-26 NCAA stats

## Usage

### 1. Scrape ESPN data (historical seasons)
```bash
python softball_scraper.py
```

### 2. Discover D1 softball team IDs (run once per season)
In `ncaa_scraper.py`, uncomment `find_d1_softball_teams()` and run:
```bash
python ncaa_scraper.py
```

### 3. Scrape NCAA season stats
```bash
python ncaa_scraper.py
```

### 4. Run WAR model
```bash
python war_model.py
```

## Author

Sam Turbeville
