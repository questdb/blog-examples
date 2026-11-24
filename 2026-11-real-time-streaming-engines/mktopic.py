#!/usr/bin/env python3
"""
Create a Kafka topic for a run (refuses if it exists).

Usage: mktopic.py <topic> [--partitions 6] [--replication 1]
Settings: KAFKA_BOOTSTRAP (host:port), from the environment or .env.
"""
import argparse
import os
import sys

from confluent_kafka.admin import AdminClient, NewTopic

import common  # noqa: F401  (loads .env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topic")
    ap.add_argument("--partitions", type=int, default=6)
    ap.add_argument("--replication", type=int, default=1)
    args = ap.parse_args()
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP")
    if not bootstrap:
        sys.exit("set KAFKA_BOOTSTRAP to the broker host:port")
    admin = AdminClient({"bootstrap.servers": bootstrap})
    if args.topic in admin.list_topics(timeout=10).topics:
        sys.exit(f"topic {args.topic} already exists; pick a new name")
    futures = admin.create_topics([NewTopic(args.topic, args.partitions, args.replication)])
    futures[args.topic].result()
    print(f"created topic {args.topic} ({args.partitions} partitions, RF {args.replication})")


if __name__ == "__main__":
    main()
