from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
import time


# ── Constants ─────────────────────────────────────────────────────────────────

DATA_DIR        = r"D:\softball-war-model\data"
TEAM_IDS_CSV    = rf"{DATA_DIR}\ncaa_team_ids.csv"
NCAA_STATS_CSV  = rf"{DATA_DIR}\ncaa_stats_2026.csv"

# Known ID range for D1 softball teams, 2025-26 season.
# Discovered by scanning stats.ncaa.org/teams/{id}/season_to_date_stats
# and filtering for pages with OBPct (batting stats) + RPI Ranking (D1 only).
TEAM_ID_START = 613550
TEAM_ID_END   = 614100


# ── Step 1: Discover D1 Softball Team IDs ─────────────────────────────────────

def find_d1_softball_teams(id_start=TEAM_ID_START, id_end=TEAM_ID_END):
    """
    Scan a range of NCAA stats team IDs and collect those belonging to
    D1 softball programs.

    Detection logic:
        - Page must contain 'OBPct' in a stats table (softball/baseball batting stat)
        - Page must contain 'RPI Ranking' (D1-only — D2/D3 teams don't have RPI)

    Outputs ncaa_team_ids.csv with columns: team_id, name.
    Run once per season when team IDs change.
    """
    teams = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page    = browser.new_page()

        for team_id in range(id_start, id_end):
            try:
                page.goto(
                    f"https://stats.ncaa.org/teams/{team_id}/season_to_date_stats",
                    timeout=10000
                )
                page.wait_for_load_state("load", timeout=8000)

                html      = page.content()
                soup      = BeautifulSoup(html, "html.parser")
                page_text = soup.get_text()

                for table in soup.find_all("table"):
                    all_cells = [c.get_text(strip=True) for c in table.find_all(["th", "td"])]
                    if "OBPct" in all_cells and "RPI Ranking" in page_text:
                        name = str(team_id)  # NCAA stats page title is generic
                        print(f"✓ {team_id}")
                        teams.append({"team_id": team_id, "name": name})
                        break

            except Exception:
                pass

            time.sleep(0.3)

        browser.close()

    df = pd.DataFrame(teams)
    df.to_csv(TEAM_IDS_CSV, index=False)
    print(f"\nDone. Found {len(df)} D1 softball teams.")
    return df


# ── Step 2: Scrape Season Stats for All Teams ──────────────────────────────────

def scrape_all_teams():
    """
    Scrape season-to-date batting stats for every D1 softball team
    in ncaa_team_ids.csv.

    For each team, hits stats.ncaa.org/teams/{id}/season_to_date_stats,
    parses the hitting table, filters out pitchers (empty AB) and
    aggregate rows (Totals / Opponent Totals), and appends to a master list.

    Outputs ncaa_stats_2026.csv with one row per player.
    Columns include: Player, AB, H, 2B, 3B, HR, BB, HBP, SF, team_id, and more.
    """
    team_ids  = pd.read_csv(TEAM_IDS_CSV)["team_id"].tolist()
    all_stats = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page    = browser.new_page()

        for i, team_id in enumerate(team_ids):
            try:
                page.goto(
                    f"https://stats.ncaa.org/teams/{team_id}/season_to_date_stats",
                    timeout=10000
                )
                page.wait_for_selector("table", timeout=8000)

                html = page.content()
                soup = BeautifulSoup(html, "html.parser")

                for table in soup.find_all("table"):
                    cells = [c.get_text(strip=True) for c in table.find_all(["th", "td"])]
                    if "OBPct" not in cells:
                        continue

                    headers = [
                        c.get_text(strip=True)
                        for c in table.find_all("tr")[0].find_all(["th", "td"])
                    ]
                    rows = []
                    for row in table.find_all("tr")[1:]:
                        values = [td.get_text(strip=True) for td in row.find_all("td")]
                        if values:
                            rows.append(values)

                    # First table with OBPct sometimes has no rows — skip it
                    if not rows:
                        continue

                    df = pd.DataFrame(rows, columns=headers)
                    df["team_id"] = team_id

                    # Drop pitchers (no AB) and aggregate rows
                    df = df[df["AB"].str.strip() != ""].copy()
                    df = df[~df["Player"].isin(["Totals", "Opponent Totals"])].copy()

                    all_stats.append(df)
                    break

                if (i + 1) % 10 == 0:
                    print(f"Progress: {i + 1}/{len(team_ids)}")

            except Exception as e:
                print(f"Failed: {team_id} — {e}")

            time.sleep(0.3)

        browser.close()

    result = pd.concat(all_stats, ignore_index=True)
    result.to_csv(NCAA_STATS_CSV, index=False)
    print(f"\nDone. {len(result)} player rows saved to {NCAA_STATS_CSV}")
    return result


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1 — run once per season to discover team IDs:
    # find_d1_softball_teams()

    # Step 2 — scrape stats for all discovered teams:
    scrape_all_teams()
