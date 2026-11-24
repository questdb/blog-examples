#!/usr/bin/env python3
"""
Recovery time with the article's definition (its bench/recovery_check.py), from QuestDB's
WAL telemetry instead of a consumer's receive series.

Per second of apply time (when rows become visible to queries): rows applied and their mean
delay from ts. Then, as the article:
  pre_rate   median rows/s over [kill - 40, kill - 5] (our runs kill at 45 s; the article
             uses [kill - 60, kill - 5])
  outage_s   seconds from the kill until rows first become visible again at >= 10% of
             pre_rate
  recover_s  seconds from the kill until the first second that starts 5 consecutive seconds
             each with >= 90% of pre_rate and mean delay < 1000 ms

With a matrix.py log (or --up <epoch_s>, when QuestDB answered queries again) it also
prints:
  outage_visible_s  seconds from the kill until every row sent during the outage is
                    visible: the latest apply time of the transactions whose newest row has
                    a ts between the kill and QuestDB answering again

Mean delay per transaction is taken from the midpoint of its [minTimestamp, maxTimestamp],
weighted by its rows. That is exact enough for runs without duplicates (the Kafka runs);
with late duplicates (the direct runs) a transaction's minTimestamp is up to 3 s old, so
the direct path uses each row's own ts instead (--rows: rows are written in ts order by one
producer, so each row belongs to the first transaction whose maxTimestamp reaches it).

Usage: recovery_article.py <table> <kill_epoch_s | runs/matrix/<scenario>_<rep>.log>
                           [--rows] [--up <epoch_s>]
"""
import statistics
import sys

from common import sql, table_id


def series(t, rows_mode):
    tid = table_id(t)
    tx = f"""
c0 AS (SELECT seqTxn, rowCount n, minTimestamp lo, maxTimestamp hi,
              max(maxTimestamp) OVER (ORDER BY seqTxn ROWS BETWEEN UNBOUNDED PRECEDING
                                      AND 1 PRECEDING) prev_hi
       FROM 'sys.telemetry_wal' WHERE tableId = {tid} AND event = 109),
a AS (SELECT seqTxn, created applied FROM 'sys.telemetry_wal'
      WHERE tableId = {tid} AND event = 105)"""
    if rows_mode:
        q = f"""
WITH {tx},
t AS ((SELECT c0.prev_hi, c0.hi, a.applied FROM c0 JOIN a ON a.seqTxn = c0.seqTxn
       WHERE c0.hi > c0.prev_hi ORDER BY c0.prev_hi) timestamp(prev_hi)),
r AS (SELECT e.ts, t.applied FROM {t} e ASOF JOIN t)
SELECT timestamp_floor('s', applied)::long / 1000000 sec, count() n,
       avg((applied - ts) / 1000.0) mean_ms
FROM r WHERE applied IS NOT NULL GROUP BY 1 ORDER BY 1"""
    else:
        q = f"""
WITH {tx},
j AS (SELECT a.applied, c0.n, (a.applied - (c0.lo + (c0.hi - c0.lo) / 2)) / 1000.0 lat
      FROM c0 JOIN a ON a.seqTxn = c0.seqTxn)
SELECT timestamp_floor('s', applied)::long / 1000000 sec, sum(n) n,
       sum(n * lat) / sum(n) mean_ms
FROM j GROUP BY 1 ORDER BY 1"""
    return {int(sec): (n, mean_ms) for sec, n, mean_ms in sql(q)["dataset"]}


def outage_visible(t, kill, up):
    """Seconds from the kill until the last transaction whose newest row was sent between the
    kill and QuestDB answering again is applied."""
    tid = table_id(t)
    last = sql(f"""
WITH c AS (SELECT seqTxn, maxTimestamp hi FROM 'sys.telemetry_wal'
           WHERE tableId = {tid} AND event = 109),
a AS (SELECT seqTxn, created applied FROM 'sys.telemetry_wal'
      WHERE tableId = {tid} AND event = 105)
SELECT max(a.applied)::long FROM c JOIN a ON a.seqTxn = c.seqTxn
WHERE c.hi >= {int(kill * 1e6)}::timestamp AND c.hi <= {int(up * 1e6)}::timestamp""")
    return last["dataset"][0][0] / 1e6 - kill


def times_from_log(path):
    """Kill time and the time QuestDB answered again (UTC epoch) from a matrix.py recovery
    log ('kill -9 <pid> at HH:MM:SS...', 'up at HH:MM:SS...', the date from the first line)."""
    import datetime as dt
    import re
    text = open(path).read()
    day = re.search(r"started (\d{4}-\d{2}-\d{2})T", text).group(1)
    epoch = lambda hms: dt.datetime.fromisoformat(f"{day}T{hms}+00:00").timestamp()
    kill = epoch(re.search(r"kill -9 \d+ at (\d+:\d+:\d+\.\d{6})", text).group(1))
    up = re.search(r"up at (\d+:\d+:\d+\.\d{6})", text)
    return kill, epoch(up.group(1)) if up else None


def main():
    t = sys.argv[1]
    if sys.argv[2].endswith(".log"):
        kill, up = times_from_log(sys.argv[2])
    else:
        kill, up = float(sys.argv[2]), None
    if "--up" in sys.argv:
        up = float(sys.argv[sys.argv.index("--up") + 1])
    rows_mode = "--rows" in sys.argv
    s = series(t, rows_mode)
    k = int(kill)
    pre = [s[x][0] for x in range(k - 40, k - 4) if x in s]
    pre_rate = statistics.median(pre)
    get = lambda x: s.get(x, (0, 0.0))
    after = [x for x in range(k + 1, k + 60)]
    outage = next((x - kill for x in after if get(x)[0] >= 0.1 * pre_rate), None)
    ok = lambda x: get(x)[0] >= 0.9 * pre_rate and get(x)[1] < 1000
    recover = next((x - kill for x in after if all(ok(x + i) for i in range(5))), None)
    print(f"{t}: pre_rate={pre_rate:.0f} rows/s outage_s={outage:.1f} recover_s={recover:.1f}")
    if up is not None:
        print(f"{t}: answering_s={up - kill:.2f} outage_visible_s={outage_visible(t, kill, up):.2f}")
    for x in range(k - 2, k + 16):
        n, m = get(x)
        print(f"  {x - kill:+6.1f} s  rows={n:>7}  mean_delay={m:9.1f} ms")


if __name__ == "__main__":
    main()
