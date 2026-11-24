#!/usr/bin/env python3
"""
S5 interval join, orders JOIN payments ON order_id with order.ts <= payment.ts <=
order.ts + 10 s: check the full join against the producer's ground truth, then time the
1 s poll a dashboard or downstream job would run ("pairs whose payment arrived in the
last second").

Two forms of the same join are checked and timed:
  interval  orders JOIN payments with the time bound in WHERE (a port of Flink's interval
            join)
  asof      payments ASOF JOIN orders ON (order_id) TOLERANCE 10s: each payment takes the
            latest order with its order_id at or up to 10 s before it, and order_id is
            unique. Each payment's order is up to ~250k order rows back, so the default
            (Fast) algorithm scans backward that far per payment; the asof_dense hint
            avoids it (see the SQL optimizer hints docs).

Usage: s5check.py <prefix> [end_ts, default: 5 s before the last payment] [runs, default 5]
Reads runs/<prefix>.json for the expected counts.
"""
import json
import os
import statistics
import sys

from common import RUNS_DIR, sql

BOUND_S = 10
COLS = "p.ts pay_ts, o.ts order_ts, o.order_id, o.user_id, o.amount order_amount, p.amount pay_amount"


def pairs(orders, payments, pay_window="", order_window=""):
    p = f"(SELECT * FROM {payments}{pay_window})" if pay_window else payments
    o = f"(SELECT * FROM {orders}{order_window})" if order_window else orders
    return f"""
SELECT {COLS}
FROM {p} p JOIN {o} o ON p.order_id = o.order_id
WHERE p.ts >= o.ts AND p.ts <= dateadd('s', {BOUND_S}, o.ts)"""


def asof(orders, payments, pay_filter=""):
    """pay_filter: a condition on p.ts, e.g. "p.ts IN '$now-1s..$now'"."""
    where = f"{pay_filter} AND o.ts IS NOT NULL" if pay_filter else "o.ts IS NOT NULL"
    return f"""
SELECT /*+ asof_dense(p o) */ {COLS}
FROM {payments} p ASOF JOIN {orders} o ON (order_id) TOLERANCE {BOUND_S}s
WHERE {where}"""


def polls(orders, payments, end):
    """The 1 s poll in both forms, for payments in (end - 1 s, end]."""
    pay = f"interval(dateadd('s', -1, {end}), {end})"
    return {
        "interval": pairs(orders, payments, pay_window=f" WHERE ts IN {pay}",
                          order_window=f" WHERE ts IN interval(dateadd('s', -{BOUND_S + 1}, "
                                       f"{end}), {end})"),
        "asof": asof(orders, payments, f"p.ts IN {pay}"),
    }


def checksum(q):
    return tuple(sql(f"SELECT count(), count_distinct(order_id), sum(order_id), "
                     f"sum(order_ts::long), sum(pay_ts::long) FROM ({q})")["dataset"][0])


def main():
    prefix = sys.argv[1]
    with open(os.path.join(RUNS_DIR, f"{prefix}.json")) as f:
        meta = json.load(f)
    orders, payments = meta["orders_table"], meta["payments_table"]
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    rows = [sql(f"SELECT count() FROM {t}")["dataset"][0][0] for t in (orders, payments)]
    print(f"orders {rows[0]} (sent {meta['orders']}), payments {rows[1]} (sent {meta['payments']})")
    full = {"interval": checksum(pairs(orders, payments)), "asof": checksum(asof(orders, payments))}
    for name, got in full.items():
        ok = (got[0] == meta["expected_pairs"] and got[1] == got[0]
              and rows == [meta["orders"], meta["payments"]])
        print(f"{name}: joined pairs {got[0]} (distinct orders {got[1]}), expected "
              f"{meta['expected_pairs']}; {meta['late_payments']} late payments must not join  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"asof and interval return the same pairs: "
          f"{'PASS' if full['asof'] == full['interval'] else 'FAIL'}")

    if len(sys.argv) > 2:
        end = sys.argv[2] if sys.argv[2] == "now()" else f"'{sys.argv[2]}'"
    else:
        last = sql(f"SELECT max(ts) FROM {payments}")["dataset"][0][0]
        end = f"dateadd('s', -5, '{last}')"
    p = polls(orders, payments, end)
    same = checksum(p["asof"]) == checksum(p["interval"])
    for name, q in p.items():
        wrapped = f"SELECT count() FROM ({q})"
        n = sql(wrapped)["dataset"][0][0]
        times = [sql(wrapped, timings=True)["timings"]["execute"] / 1e6 for _ in range(runs)]
        print(f"poll {name}: pairs with payment in the last 1s = {n}, execute min={min(times):.1f} ms "
              f"median={statistics.median(times):.1f} ms")
    print(f"poll asof and interval return the same pairs: {'PASS' if same else 'FAIL'}")


if __name__ == "__main__":
    main()
