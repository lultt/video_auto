# night_grouper.py — assign each row to a "night" (18:00 ~ next day 06:00)

import pandas as pd
from .config import NIGHT_START_HOUR, NIGHT_END_HOUR


def assign_night(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'night' column like '2025-05-15_night'.

    Rule: rows between 18:00 on date D and 06:00 on date D+1 belong to 'D_night'.
    Rows outside 18:00–06:00 are labelled as 'daytime'.
    """
    df = df.copy()
    ts = df["abs_time"]

    hour = ts.dt.hour
    date = ts.dt.date

    # Default: night belongs to the date of the 18:00 start
    # If hour >= 18, night date = today
    # If hour < NIGHT_END_HOUR (6am), night date = yesterday (since it belongs to previous evening)
    night_date = date.where(hour >= NIGHT_START_HOUR, date - pd.Timedelta(days=1))
    night_date = night_date.where(
        (hour >= NIGHT_START_HOUR) | (hour < NIGHT_END_HOUR),
        pd.NaT
    )

    df["night"] = night_date.apply(
        lambda d: f"{d.strftime('%Y-%m-%d')}_night" if pd.notna(d) else "daytime"
    )
    return df


def get_night_list(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique night labels (excluding 'daytime')."""
    nights = sorted(n for n in df["night"].unique() if n != "daytime")
    return nights
