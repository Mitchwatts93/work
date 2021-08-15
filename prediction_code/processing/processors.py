"""functions for general processing of any data"""

import pandas as pd
from datetime import timezone


################################################################################
def convert_to_datetime(
    datetime_series: pd.Series,
    timezone: timezone = timezone.utc
) -> pd.Series:
    """convert a datetime series to pandas datetime objects, localize to the
    timezone passed.
    Args:
        datetime_series: a pd series with elements that can be cast as pd 
            datetimes using pd.to_datetime.
        timezone: a timezone which can be a tzinfo object (timezone from 
            datetime, pytz etc) or a string.
    Returns:
        datetime_series: a datetime series with the elements converted to pd
            datetime objects and localized to the passed timezone.
    """
    datetime_series = pd.to_datetime(datetime_series, utc=False)
    datetime_series.index = datetime_series.to_numpy()
    datetime_series = datetime_series.index.tz_localize(timezone)

    return datetime_series
    
################################################################################