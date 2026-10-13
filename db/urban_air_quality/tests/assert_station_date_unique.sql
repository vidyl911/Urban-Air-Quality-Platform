select
    station_id,
    observation_date,
    count(*) as record_count
from {{ ref('mart_station_daily_features') }}
group by
    station_id,
    observation_date
having count(*) > 1