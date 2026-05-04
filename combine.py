#Combine two csvs of softball data
import pandas as pd

# boxscore and pbp need different deduplication — drop on play_id/athlete+game, not just game_id
for filename, dedup_col in [("boxscore.csv", None), ("pbp.csv", "play_id")]:
    feb  = pd.read_csv(f"data/2025_feb/{filename}")
    main = pd.read_csv(f"data/2025_mar_may/{filename}")
    combined = pd.concat([feb, main])
    if dedup_col:
        combined = combined.drop_duplicates(subset=[dedup_col])
    else:
        combined = combined.drop_duplicates()
    combined.to_csv(f"data/2025/{filename}", index=False)
    print(f"{filename}: {len(combined)} rows")