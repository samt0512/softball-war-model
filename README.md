# College Softball WAR Model

A Wins Above Replacement (WAR) model for NCAA D1 college softball — the first publicly available framework of its kind.

## Overview

This project builds a WAR model for D1 college softball using play-by-play and boxscore data scraped from ESPN's API. The goal is to produce a credible, publishable WAR leaderboard for the current D1 season.

## Methodology

WAR is calculated using the following components:

- **Linear Weights** — derived from play-by-play data via run expectancy
- **wOBA** — weighted on-base average using linear weights
- **wRAA** — runs above average vs. league average wOBA
- **Replacement Level** — average wOBA of bottom 20% of qualified players
- **Positional Adjustment** — fixed run value per position scaled to season length
- **Runs Per Win** — derived from average runs per game

## Data Sources

- **ESPN API** — scoreboard and game summary endpoints (undocumented but stable)
- 3 seasons scraped: 2023, 2024, 2025
- ~7,200 games, ~64,000 boxscore rows, ~353,000 play-by-play rows
- ~27% of games have play-by-play available

## Project Structure
softball-war-model/
├── softball_scraper.py      # Scrapes ESPN API, outputs games/boxscore/pbp CSVs
├── softball_diagnostic.py   # Data quality audit tool
├── combine.py               # Merges split season data with deduplication
├── war_model.py             # Core WAR calculation pipeline
└── data/                    # Local data directory (not tracked in git)

## Stack

- Python, pandas, NumPy
- Data stored locally as CSVs (not included in repo)

## Status

Currently in active development. Pipeline is functional end-to-end with known limitations:

- Run expectancy matrix uses 3 states (outs only) — full 24-state base-out matrix planned
- wOBA scaling deferred until RE matrix is improved
- Positional adjustments in progress

## Usage

### Scrape data
```bash
python softball_scraper.py --start 2025-02-01 --end 2025-05-15 --output-dir ./data/2025
```

### Run WAR model
```bash
python war_model.py
```

## Author

Sam Turbeville
