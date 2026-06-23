import pandas as pd
import numpy as np

LONG_BREAK_THRESH_MIN = 60  # gap >= 60 min => long break


def detect_sessions(df_sym):
    """
    For a single symbol dataframe (sorted by datetime),
    detect sessions and classify breaks.
    Adds: session_id, is_long_break_before, bars_to_next_long_break, gap_min
    """
    df = df_sym.copy().sort_values("datetime").reset_index(drop=True)
    n = len(df)
    dt = df["datetime"]
    gaps = dt.diff().dt.total_seconds().div(60).fillna(0).values

    session_id = np.zeros(n, dtype=int)
    is_long_break_before = np.zeros(n, dtype=bool)

    sid = 0
    for i in range(1, n):
        if gaps[i] >= LONG_BREAK_THRESH_MIN:
            sid += 1
            is_long_break_before[i] = True
        session_id[i] = sid

    df["session_id"] = session_id
    df["is_long_break_before"] = is_long_break_before
    df["gap_min"] = gaps

    # bars_to_next_long_break: bars remaining in current session
    bars_to_next = np.zeros(n, dtype=int)
    remaining = 0
    for i in range(n - 1, -1, -1):
        if i < n - 1 and is_long_break_before[i + 1]:
            remaining = 0
        bars_to_next[i] = remaining
        remaining += 1

    df["bars_to_next_long_break"] = bars_to_next
    return df


def add_sessions_all(data):
    results = []
    for sym, grp in data.groupby("symbol"):
        results.append(detect_sessions(grp))
    return pd.concat(results, ignore_index=True).sort_values(["symbol", "datetime"]).reset_index(drop=True)
