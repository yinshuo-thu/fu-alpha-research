import pandas as pd
import numpy as np

HORIZON = 30
NAN_BEFORE_LONG_BREAK = 30


def build_labels(df_sym, horizon=HORIZON):
    """
    Build horizon-bar forward return labels.
    - Count forward bars skipping nothing (short breaks are same session, just count through)
    - Stop at long break => NaN
    - Last NAN_BEFORE_LONG_BREAK bars before each long break => NaN
    - Last horizon bars at end of data => NaN
    """
    df = df_sym.copy().sort_values("datetime").reset_index(drop=True)
    n = len(df)
    labels = np.full(n, np.nan, dtype=np.float64)
    close = df["close"].values

    # Session ids reset only at long breaks. Within a session, row t+horizon is
    # exactly the forward bar after skipping short breaks, because short breaks
    # are absent rows but remain in the same session.
    session_id = df["session_id"].values
    for _, idx in pd.Series(np.arange(n)).groupby(session_id, sort=False):
        pos = idx.to_numpy()
        if len(pos) <= horizon:
            continue
        cur = close[pos[:-horizon]]
        fut = close[pos[horizon:]]
        ok = (cur != 0) & np.isfinite(cur) & np.isfinite(fut)
        vals = np.full(len(cur), np.nan, dtype=np.float64)
        vals[ok] = fut[ok] / cur[ok] - 1.0
        labels[pos[:-horizon]] = vals

    # Keep the explicit official rule in place even though horizon=session-tail
    # invalidation above already covers horizon=30.
    is_lb = df["is_long_break_before"].values
    for lb_idx in np.where(is_lb)[0]:
        labels[max(0, lb_idx - NAN_BEFORE_LONG_BREAK):lb_idx] = np.nan

    df["label"] = labels
    return df


def build_labels_all(data, horizon=HORIZON):
    results = []
    for sym, grp in data.groupby("symbol"):
        results.append(build_labels(grp, horizon))
    return pd.concat(results, ignore_index=True).sort_values(["symbol", "datetime"]).reset_index(drop=True)
