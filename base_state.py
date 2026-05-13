"""
base_state.py

Infers base state (runners on 1st, 2nd, 3rd) for each play in the PBP dataset
by tracking runners sequentially through each half-inning.

This enables building a full 24-state (base × outs) run expectancy matrix,
which produces more accurate linear weights than the 3-state (outs only) version.

Pipeline:
    pbp_df = load_pbp()
    pbp_with_states = infer_base_states(pbp_df)
    # pbp_with_states now has: on_1b, on_2b, on_3b, base_state columns
"""

import pandas as pd
import numpy as np
import re
import glob


# ── Name Utilities ─────────────────────────────────────────────────────────────

def extract_batter_name(description):
    """
    Extract the batter's display name from the start of a play description.
    Handles formats: 'B. Wright', 'H. Wacaser', 'Edenfield', 'Mudge'.
    Returns name string or 'Unknown'.
    """
    match = re.match(
        r'^((?:[A-Z]\.\s+)?[A-Z][a-zA-Z\'\-]+(?:\s[A-Z][a-zA-Z\'\-]+)?)\s',
        description
    )
    return match.group(1) if match else 'Unknown'


def get_last_name(name):
    """Extract last name (lowercase) from any name format."""
    if not name:
        return ''
    return name.strip().split()[-1].lower()


# ── Event Classification ───────────────────────────────────────────────────────

def classify_batter_event(description):
    """
    Classify what happened to the batter from the play description.
    Returns event type string.
    """
    desc = description.lower()
    if 'home run' in desc or 'homered' in desc:  return 'homerun'
    if 'tripled'        in desc:                  return 'triple'
    if 'doubled'        in desc:                  return 'double'
    if 'singled'        in desc:                  return 'single'
    if 'hit by pitch'   in desc:                  return 'hbp'
    if 'walked'         in desc or 'intentional walk' in desc: return 'walk'
    if 'double play'    in desc:                  return 'double_play'
    if 'reached on an error' in desc or 'reached on error' in desc: return 'error'
    if "fielder's choice" in desc:                return 'fielders_choice'
    if 'struck out'     in desc or 'called out on strikes' in desc: return 'strikeout'
    if 'sac'            in desc:                  return 'sac'
    if any(p in desc for p in ['grounded out', 'flied out', 'lined out',
                                'popped up', 'fouled out']): return 'out'
    return 'other'


def is_non_batting_event(description):
    """
    Return True for substitution and lineup change rows that don't
    represent a plate appearance and shouldn't trigger a state update.
    """
    desc = description.lower()
    return any(phrase in desc for phrase in [
        'pinch ran for',
        ' to p for', ' to c for', ' to 1b for', ' to 2b for',
        ' to 3b for', ' to ss for', ' to lf for', ' to cf for',
        ' to rf for', ' to dh for', ' to dp for',
    ])


def is_stolen_base_event(description):
    """
    Return True for standalone stolen base / caught stealing events
    that don't involve a plate appearance.
    """
    desc = description.lower()
    has_steal = any(kw in desc for kw in [
        'stole second', 'stole third', 'stole home',
        'caught stealing', 'out at second', 'out at third', 'out at home',
    ])
    has_batting = any(kw in desc for kw in [
        'singled', 'doubled', 'tripled', 'home run', 'homered',
        'walked', 'hit by pitch', 'struck out', 'grounded out',
        'flied out', 'lined out', 'popped up', 'fouled out',
        'reached on', "fielder's choice",
    ])
    return has_steal and not has_batting


# ── Runner Movement Parsing ────────────────────────────────────────────────────

def parse_runner_movements(description):
    """
    Parse explicit runner movements from semicolon-separated clauses.

    Returns dict: {last_name_lower → destination}
    Destinations: 'scored' | '1b' | '2b' | '3b' | 'out'

    Example:
        "Edenfield doubled; H. Wacaser scored; Dack advanced to third"
        → {'wacaser': 'scored', 'dack': '3b'}
    """
    movements = {}
    clauses = description.split(';')

    for clause in clauses[1:]:
        clause = clause.strip()
        cl = clause.lower()

        if 'scored' in cl:
            name = re.sub(r'\s*scored.*', '', clause, flags=re.IGNORECASE).strip()
            if name:
                movements[get_last_name(name)] = 'scored'

        elif 'advanced to third' in cl:
            name = re.sub(r'\s*advanced.*', '', clause, flags=re.IGNORECASE).strip()
            if name:
                movements[get_last_name(name)] = '3b'

        elif 'advanced to second' in cl:
            name = re.sub(r'\s*advanced.*', '', clause, flags=re.IGNORECASE).strip()
            if name:
                movements[get_last_name(name)] = '2b'

        elif 'advanced to first' in cl:
            name = re.sub(r'\s*advanced.*', '', clause, flags=re.IGNORECASE).strip()
            if name:
                movements[get_last_name(name)] = '1b'

        elif 'out on the play' in cl:
            name = re.sub(r'\s*out on.*', '', clause, flags=re.IGNORECASE).strip()
            if name:
                movements[get_last_name(name)] = 'out'

        elif 'stole third' in cl:
            name = re.sub(r'\s*stole.*', '', clause, flags=re.IGNORECASE).strip()
            if name:
                movements[get_last_name(name)] = '3b'

        elif 'stole second' in cl:
            name = re.sub(r'\s*stole.*', '', clause, flags=re.IGNORECASE).strip()
            if name:
                movements[get_last_name(name)] = '2b'

    return movements


# ── State Update Logic ─────────────────────────────────────────────────────────

def update_base_state(state, description, event_type, batter_name):
    """
    Compute the new base state after a plate appearance.

    Args:
        state:       {1: name|None, 2: name|None, 3: name|None}
        description: Play description string
        event_type:  From classify_batter_event()
        batter_name: Display name of the batter

    Default advancement assumptions (applied when description doesn't specify):
        Single:  3b scores, 2b→3b, 1b→2b
        Double:  3b scores, 2b scores, 1b→3b
        Triple:  all runners score
        HR:      all runners score, bases empty
        Walk:    forced advancement only

    Returns: (new_state, runs_scored)
    """
    runs      = 0
    new_state = {1: None, 2: None, 3: None}
    explicit  = parse_runner_movements(description)

    def resolve(runner_name, default_dest):
        """Return explicit destination if known, else the default."""
        return explicit.get(get_last_name(runner_name), default_dest)

    def place(runner, dest):
        """Place a runner at a destination, incrementing runs if they score."""
        nonlocal runs
        if dest == 'scored':
            runs += 1
        elif dest == '3b' and new_state[3] is None:
            new_state[3] = runner
        elif dest == '2b' and new_state[2] is None:
            new_state[2] = runner
        elif dest == '1b' and new_state[1] is None:
            new_state[1] = runner
        # 'out' or already-occupied base: runner is removed

    # ── Home run ──────────────────────────────────────────────────────────────
    if event_type == 'homerun':
        for base in [1, 2, 3]:
            if state[base]:
                runs += 1
        runs += 1  # batter scores; bases stay empty

    # ── Triple ────────────────────────────────────────────────────────────────
    elif event_type == 'triple':
        for base in [1, 2, 3]:
            if state[base]:
                place(state[base], resolve(state[base], 'scored'))
        new_state[3] = batter_name

    # ── Double ────────────────────────────────────────────────────────────────
    elif event_type == 'double':
        for base, default in [(3, 'scored'), (2, 'scored'), (1, '3b')]:
            if state[base]:
                place(state[base], resolve(state[base], default))
        new_state[2] = batter_name

    # ── Single ────────────────────────────────────────────────────────────────
    elif event_type == 'single':
        for base, default in [(3, 'scored'), (2, '3b'), (1, '2b')]:
            if state[base]:
                place(state[base], resolve(state[base], default))
        new_state[1] = batter_name

    # ── Walk / HBP ────────────────────────────────────────────────────────────
    elif event_type in ('walk', 'hbp'):
        # Only forced advancement — unforced runners stay put
        on_1 = state[1] is not None
        on_2 = state[2] is not None

        if state[3]:
            default = 'scored' if (on_1 and on_2) else '3b'
            place(state[3], resolve(state[3], default))
        if state[2]:
            default = '3b' if on_1 else '2b'
            place(state[2], resolve(state[2], default))
        if state[1]:
            place(state[1], resolve(state[1], '2b'))

        new_state[1] = batter_name

    # ── Outs (batter retired, runners may advance via sac fly / FC) ───────────
    elif event_type in ('strikeout', 'out', 'sac', 'double_play'):
        for base in [1, 2, 3]:
            if state[base]:
                ln = get_last_name(state[base])
                if ln in explicit:
                    place(state[base], explicit[ln])
                else:
                    new_state[base] = state[base]  # stay

    # ── Reached on error / fielder's choice ───────────────────────────────────
    elif event_type in ('error', 'fielders_choice'):
        for base in [1, 2, 3]:
            if state[base]:
                ln = get_last_name(state[base])
                if ln in explicit:
                    place(state[base], explicit[ln])
                else:
                    new_state[base] = state[base]
        if new_state[1] is None:
            new_state[1] = batter_name

    # ── Other / unknown ───────────────────────────────────────────────────────
    else:
        new_state = dict(state)

    return new_state, runs


def handle_stolen_base(description, state):
    """
    Handle standalone stolen base and caught stealing events.
    Updates base state without changing outs (outs handled in main loop).

    Returns updated state.
    """
    desc      = description.lower()
    new_state = dict(state)

    if 'caught stealing' in desc or 'out at' in desc:
        if 'home' in desc and new_state[3]:
            new_state[3] = None
        elif 'third' in desc and new_state[2]:
            new_state[2] = None
        elif 'second' in desc and new_state[1]:
            new_state[1] = None

    elif 'stole home' in desc and new_state[3]:
        new_state[3] = None

    elif 'stole third' in desc and new_state[2]:
        runner      = new_state[2]
        new_state[2] = None
        new_state[3] = runner

    elif 'stole second' in desc and new_state[1]:
        runner      = new_state[1]
        new_state[1] = None
        new_state[2] = runner

    return new_state


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def infer_base_states(pbp_df):
    """
    Add base state columns to pbp_df by tracking runners through each game.

    Processes each half-inning sequentially. For each play, records the base
    state BEFORE the play occurs, then updates the state based on the outcome.

    Added columns:
        on_1b (bool):      Runner on first base before this play
        on_2b (bool):      Runner on second base before this play
        on_3b (bool):      Runner on third base before this play
        base_state (int):  0–7, binary encoding: on_1b + on_2b*2 + on_3b*4

    Returns enriched pbp_df with all original columns plus the four above.
    """
    plays = pbp_df[pbp_df['play_type'] == 'Play Result'].copy()
    plays = plays.sort_values(['game_id', 'inning', 'inning_half', 'sequence'])

    state_records = []

    for (game_id, inning, inning_half), group in plays.groupby(
        ['game_id', 'inning', 'inning_half'], sort=False
    ):
        state = {1: None, 2: None, 3: None}

        for _, row in group.iterrows():
            desc = str(row['description'])

            # Record state BEFORE this play
            on_1 = state[1] is not None
            on_2 = state[2] is not None
            on_3 = state[3] is not None

            state_records.append({
                'play_id':    row['play_id'],
                'on_1b':      on_1,
                'on_2b':      on_2,
                'on_3b':      on_3,
                'base_state': int(on_1) + int(on_2) * 2 + int(on_3) * 4,
            })

            # Skip lineup changes and substitutions
            if is_non_batting_event(desc):
                continue

            # Handle stolen base / caught stealing as runner-only events
            if is_stolen_base_event(desc):
                state = handle_stolen_base(desc, state)
                continue

            # Normal plate appearance — classify and update
            event_type  = classify_batter_event(desc)
            batter_name = extract_batter_name(desc)
            state, _    = update_base_state(state, desc, event_type, batter_name)

    state_df = pd.DataFrame(state_records)
    result   = pbp_df.merge(state_df, on='play_id', how='left')

    for col in ['on_1b', 'on_2b', 'on_3b']:
        result[col] = result[col].infer_objects(copy=False).fillna(False).astype(bool)
    result['base_state'] = result['base_state'].fillna(0).astype(int)

    return result


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_base_states(pbp_with_states):
    """
    Sanity check the inferred base states.

    Key check: for non-home-run scoring plays, at least one runner should
    have been on base before the play. Low coverage (< 80%) suggests parsing issues.
    """
    plays = pbp_with_states[pbp_with_states['play_type'] == 'Play Result']

    # Non-HR scoring plays must have had a runner on base
    scoring = plays[
        (plays['score_value'] > 0) &
        ~plays['description'].str.lower().str.contains('home run|homered')
    ]
    had_runner = (scoring['on_1b'] | scoring['on_2b'] | scoring['on_3b'])

    total      = len(scoring)
    covered    = had_runner.sum()

    print(f"Non-HR scoring plays:     {total}")
    print(f"Had runner on base:       {covered} ({covered / total * 100:.1f}%)")
    print(f"Missing runner (parsing): {total - covered} ({(total - covered) / total * 100:.1f}%)")
    print()
    print("Base state distribution across all plays:")
    state_labels = {
        0: 'Empty', 1: '1st', 2: '2nd', 3: '1st+2nd',
        4: '3rd', 5: '1st+3rd', 6: '2nd+3rd', 7: 'Loaded'
    }
    counts = plays['base_state'].value_counts().sort_index()
    for state, count in counts.items():
        pct = count / len(plays) * 100
        print(f"  {state_labels.get(state, state):12s}: {count:7,}  ({pct:.1f}%)")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    def load_pbp():
        df_list = []
        for filename in glob.glob(r"D:\softball-war-model\data\pbp\*.csv"):
            df   = pd.read_csv(filename)
            year = filename.split("_")[1].split(".")[0]
            df["season"] = int(year)
            df_list.append(df)
        return pd.concat(df_list, ignore_index=True)

    print("Loading PBP data...")
    pbp_df = load_pbp()
    print(f"Loaded {len(pbp_df):,} rows\n")

    print("Inferring base states...")
    pbp_with_states = infer_base_states(pbp_df)

    print("\nValidation:")
    validate_base_states(pbp_with_states)

    print("\nSample — first game, play results only:")
    sample = pbp_with_states[pbp_with_states['play_type'] == 'Play Result'].head(20)
    print(sample[[
        'sequence', 'inning', 'inning_half', 'outs_before',
        'on_1b', 'on_2b', 'on_3b', 'base_state', 'description'
    ]].to_string())