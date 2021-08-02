import pandas as pd
from datetime import timezone


################################################################################
def convert_to_datetime(
    datetime_series: pd.Series,
    timezone: timezone = timezone.utc
) -> pd.Series:
    datetime_series = pd.to_datetime(datetime_series, utc=False)
    datetime_series.index = datetime_series.to_numpy()
    datetime_series = datetime_series.index.tz_localize(timezone)

    return datetime_series
    
################################################################################