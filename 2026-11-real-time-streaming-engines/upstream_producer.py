#!/usr/bin/env python3
"""
Run the original benchmark's producer (upstream/producer.py, see aws/fetch_upstream.sh)
unchanged, except that the event's ts is serialized as integer epoch microseconds instead
of float epoch seconds.

The QuestDB Kafka connector takes a designated timestamp from an integer (in
timestamp.units), a date or a string, but not from a JSON float. Everything else (rates,
user/type/country/amount distributions, the duplicate rule, keys, partitioning) is the
upstream code. Duplicates re-send the same serialized message, so they keep the same ts.

Usage: upstream_producer.py <upstream producer.py arguments>
"""
import json
import os
import runpy
import sys

_dumps = json.dumps


def _dumps_ts_micros(obj, *args, **kwargs):
    if isinstance(obj, dict) and isinstance(obj.get("ts"), float):
        obj = {**obj, "ts": round(obj["ts"] * 1_000_000)}
    return _dumps(obj, *args, **kwargs)


json.dumps = _dumps_ts_micros
upstream = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream", "producer.py")
sys.argv[0] = upstream
runpy.run_path(upstream, run_name="__main__")
