import pandas as pd
box = pd.read_csv("data/2025/boxscore.csv")

# How many rows have no athlete_id?
print("Missing athlete_id:", box["athlete_id"].isna().sum())

# What's the max PA for any single athlete_id?
box["AB"] = pd.to_numeric(box["AB"], errors="coerce").fillna(0)
box["BB"] = pd.to_numeric(box["BB"], errors="coerce").fillna(0)
box["PA"] = box["AB"] + box["BB"]
print(box.groupby("athlete_id")["PA"].sum().sort_values(ascending=False).head(10))