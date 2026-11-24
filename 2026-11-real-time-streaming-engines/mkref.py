#!/usr/bin/env python3
"""Create the S4 reference table bench_users_ref (980k users, tier and region by hash)."""
from common import sql

sql("""CREATE TABLE IF NOT EXISTS bench_users_ref AS (
SELECT x AS user_id,
  CASE WHEN h % 4 = 0 THEN 'bronze' WHEN h % 4 = 1 THEN 'silver'
       WHEN h % 4 = 2 THEN 'gold' ELSE 'platinum' END::symbol AS tier,
  CASE WHEN (h / 256) % 8 = 0 THEN 'na' WHEN (h / 256) % 8 = 1 THEN 'latam'
       WHEN (h / 256) % 8 = 2 THEN 'emea' WHEN (h / 256) % 8 = 3 THEN 'apac'
       WHEN (h / 256) % 8 = 4 THEN 'anz' WHEN (h / 256) % 8 = 5 THEN 'mena'
       WHEN (h / 256) % 8 = 6 THEN 'ssa' ELSE 'cis' END::symbol AS region
FROM (SELECT x, (x * 2654435761) & 4294967295 AS h FROM long_sequence(1000000))
WHERE x % 50 != 0)""")
print(sql("SELECT count() FROM bench_users_ref")["dataset"][0][0], "rows in bench_users_ref")
