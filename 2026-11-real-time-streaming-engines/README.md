# Streaming workloads on QuestDB, without a stream processor

This folder runs the workloads of
[fmorillo7694/spark-rtm-vs-flink](https://github.com/fmorillo7694/spark-rtm-vs-flink), a
Flink 2.3 vs Spark 4.3 Real-time Mode benchmark, on a single QuestDB server with no stream
processor, on two paths:

- **Direct**: a producer writes to QuestDB over QWP.
- **Through Kafka**: the article's own producer writes to Kafka, and the QuestDB Kafka
  connector writes to QuestDB.

Deduplication happens on write, transforms are views and materialized views, and the
stateful workloads are queries. `RESULTS.md` has the figures next to the article's; this
README covers how the harness works, how each number is measured, and what we learned.

## The workloads and how QuestDB runs them

| Workload | What it does | In QuestDB | Script |
|---|---|---|---|
| Stateless | Filter to purchases and add-to-cart, add tax, uppercase the country | A plain view over the base table (`mktable.py --clean-view`); a passthrough materialized view (`--clean-mv`) is also available | `mktable.py` |
| S1 | Count, sum and max of `amount` per user in 10 s windows | `SAMPLE BY 10s` materialized view (`--s1-mv`) | `mktable.py` |
| S2 | Deduplicate on `event_id` | `DEDUP SKIP KEYS(ts, event_id)` on the base table | `mktable.py` |
| S3 | Sessions per user, closed after a 5 s gap | Window functions over the last 30 s, for the sessions that closed in the last second | `s3check.py` |
| S4 | Enrich each event from a users table | `JOIN` with `bench_users_ref` | `mkref.py`, `qtime.py` |
| S5 | Join orders and payments, payment within 10 s of its order | `payments ASOF JOIN orders ON (order_id) TOLERANCE 10s` with the `asof_dense` hint | `s5producer.py`, `s5check.py` |

S5's query, as a consumer polls it:

```sql
SELECT /*+ asof_dense(p o) */ p.payment_id, p.ts pay_ts, o.ts order_ts,
       o.order_id, o.user_id, o.amount order_amount, p.amount pay_amount
FROM payments p
ASOF JOIN orders o ON (order_id) TOLERANCE 10s
WHERE p.ts IN '$now-1500T..$now' AND o.ts IS NOT NULL;
```

`TOLERANCE 10s` is the join rule. The window reads the last 1.5 s because a payment sent just
before a poll can become visible just after it; the consumer skips payment ids it has
already seen (`s5consumer.py`). Without `asof_dense`, the default ASOF algorithm scans back
from each payment to its order, up to ~250k rows, and a poll takes seconds. `s5check.py`
checks both this join and the equivalent interval join against the producer's ground truth.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env    # then set QDB_ADDR to your QuestDB host:port
```

Every script reads its settings from the environment, falling back to `.env`; a variable
already set in the environment wins, so a command on another host can override any value
(`ssh host env QDB_ADDR=10.0.0.5:9000 bench/aws/run.sh ...`).

## Running it on AWS

All results come from this setup, in eu-west-1, one subnet (same availability zone):

| | QuestDB server | Producer | Kafka (Kafka path only) |
|---|---|---|---|
| Instance | m5.2xlarge (the article's node type) | c7i.2xlarge | m5.2xlarge |
| Storage | 30 GiB root; 200 GiB gp3 data volume at 80,000 IOPS / 2,000 MiB/s (`aws/server-bdm.json`) | 30 GiB root | 100 GiB gp3 at 16,000 IOPS / 1,000 MiB/s (`aws/kafka-bdm.json`) |
| Software | QuestDB 10.2.0 (release tarball, bundled JRE), pinned to vCPUs 1-7 (`QDB_CPUS=1-7`, the article's 7-core budget); Kafka Connect with the QuestDB connector 0.25 on the same 7 vCPUs | Python producers | Kafka 4.3.2, one KRaft broker |

AMI: Ubuntu 24.04 LTS amd64 (SSM parameter
`/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id`).
Security group, inbound: TCP 22 from your IP, TCP 9000 from the producer and your IP (QWP
and REST), TCP 9092 from the producer and the QuestDB host (Kafka).

```bash
AMI=$(aws ssm get-parameter --region eu-west-1 --query Parameter.Value --output text \
  --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id)
aws ec2 run-instances --region eu-west-1 --image-id "$AMI" --instance-type m5.2xlarge \
  --key-name <key> --subnet-id <subnet> --security-group-ids <sg> \
  --block-device-mappings file://aws/server-bdm.json
aws ec2 run-instances --region eu-west-1 --image-id "$AMI" --instance-type c7i.2xlarge \
  --key-name <key> --subnet-id <subnet> --security-group-ids <sg> \
  --block-device-mappings file://aws/producer-bdm.json
aws ec2 run-instances --region eu-west-1 --image-id "$AMI" --instance-type m5.2xlarge \
  --key-name <key> --subnet-id <subnet> --security-group-ids <sg> \
  --block-device-mappings file://aws/kafka-bdm.json
```

Copy this folder to `~/bench` on each host (without `.venv`, `runs` and `.env`), then:

```bash
# QuestDB host
bash bench/aws/server_setup.sh                      # data volume, QuestDB, limits
bash bench/aws/connect_setup.sh                     # Kafka Connect + QuestDB connector
QDB_CPUS=1-7 bash bench/aws/start_qdb.sh pools_nap.conf

# Kafka host
KAFKA_ADVERTISED_HOST=<its private ip> bash bench/aws/kafka_setup.sh

# producer
bash bench/aws/producer_setup.sh
bash bench/aws/fetch_upstream.sh                    # the article's producer, pinned commit
QDB_ADDR=<server private ip>:9000 bench/.venv/bin/python bench/mkref.py
```

Then run scenarios from your machine with `aws/matrix.py`, which drives the hosts over ssh
and logs every command and result to `runs/matrix/<scenario>_<rep>.log`:

```bash
.venv/bin/python aws/matrix.py direct_dedup,kafka_dedup r1
```

Its docstring lists the scenarios (dedup, stateless, S1, S5, polling, recovery, scaling,
30-minute runs, on both paths). Set `SSH_KEY`, `SERVER_HOST`, `PRODUCER_HOST`, `KAFKA_HOST`,
`SERVER_PRIVATE`, `KAFKA_PRIVATE` and `QDB_ADDR` in `.env` (see `.env.example`). For
`RESULTS.md` every scenario ran twice on fresh tables, except the 30-minute S1 and S5 runs
and the S5 consumer measurements, which ran once. `archive.py` exports a run's telemetry and row counts
to `runs/` before the server goes away. Terminating the instances deletes their volumes.

## How each number is measured

- **Latency to visible**: from the event's `ts` (wall clock at send, as in the article) to
  the transaction that holds the row being applied and readable, from QuestDB's own WAL
  telemetry (`sys.telemetry_wal`, events 109 commit and 105 apply). Nothing polls for it.
  - `latency.py`: exact per row, for a producer writing its rows in `ts` order through one
    WAL writer (`--wal-id` when several producers write at once; `walids.py` lists them).
  - `cover_latency.py`: lower and upper bounds per row with no ordering assumption, for the
    Kafka path, where each connector task carries rows from many producers. On a direct run
    without duplicates both bounds match the exact numbers within 0.1 ms.
  - `view_latency.py`: bounds for an aggregating view (S1).
  - Through Kafka the first 10 s of each run are excluded (consumer-group start; the article
    excludes 60 s).
- **Query time**: QuestDB's own execute time from the REST timings. `poll.py` runs each
  query once a second on its own thread during ingest and reports p50 / p95 / p99 / max;
  `qtime.py` times them a few times at a given point.
- **End to end for a polling consumer (S5)**: `s5consumer.py` on the producer host polls
  every 100 ms or 1 s and records when it first receives each pair, minus the payment's `ts`
  (same host clock). It also reports completeness against the database.
- **CPU and memory**: `cpusample.py` on the server samples QuestDB (and the Connect worker)
  every second: cores and RSS. Where a run also polls, CPU is reported over a window with
  nothing reading. The Kafka broker is never counted, as in the article.
- **Correctness, every run**: rows stored = distinct ids = originals sent; each view holds
  exactly the rows it should and is `valid` at the latest base transaction; S5 joins equal
  the producer's ground truth; through Kafka, consumer lag 0 and no connector errors.
- **Recovery**: `aws/chaos.sh` kills QuestDB with `kill -9` 45 s into a run and restarts it
  after a set delay; `aws/matrix.py` reports when QuestDB answers queries again and when
  every row is visible again (see below).

Both hosts sync their clocks with chrony against the Amazon Time Sync Service
(169.254.169.123); the offset measured under 1 µs.

## Server and client configuration

Three server settings change from the defaults (`aws/pools_nap.conf`):

```ini
cairo.o3.partition.merge.append.enabled=true

shared.network.worker.count=2
shared.write.worker.count=2
shared.query.worker.count=3
wal.apply.worker.count=2
mat.view.refresh.worker.count=2

shared.worker.nap.threshold=100
shared.worker.sleep.threshold=1000
wal.apply.worker.nap.threshold=100
wal.apply.worker.sleep.threshold=1000
mat.view.refresh.worker.nap.threshold=100
mat.view.refresh.worker.sleep.threshold=1000
```

Measured one at a time (50k evt/s with 10% late duplicates, 2,000-row batches, base table
plus passthrough view; 300 s on defaults, 120 s for each change):

| Config | QuestDB CPU | Base p50 / p99 | View p50 / p99 |
|---|---|---|---|
| Defaults | 6.84 cores | 23.7 / 55.9 ms | 84.2 ms / 2.3 s (falls behind) |
| + merge-append | 6.33 cores | 23.1 / 43.4 ms | 52.0 / 84.2 ms |
| + dedicated pools | 4.53 cores | 23.2 / 43.3 ms | 46.9 / 75.2 ms |
| + idle workers nap and sleep sooner | 1.40 cores | 23.2 / 43.3 ms | 42.6 / 67.8 ms |

- [Merge-append partitions](https://questdb.com/docs/concepts/merge-append-partitions/)
  (opt-in since 10.1.0) are built for late rows and duplicates in a narrow recent range.
  Without them every commit carries out-of-order rows, the cost of applying one grows with
  the open hour partition, and the view falls seconds behind within 5 minutes.
- Dedicated pools keep queries from slowing ingestion (base p99 43.9 ms over a run, 44.3 ms
  while queries ran).
- With the default idle thresholds, workers spin between commits, so CPU follows the commit
  rate more than the row rate.

Client side, the QWP sender's batch size (`auto_flush_rows`) plays the role of Flink's buffer
timeout and sets most of QuestDB's median latency: 2,000 rows at 50k evt/s fill in 40 ms.
S5 at 8k orders/s uses 500 rows so a batch fills in about the same time. Base table at 50k
evt/s, tuned config:

| Batch | Visible p50 / p99 | QuestDB CPU |
|---|---|---|
| 2,000 | 22-23 / 43-46 ms | 0.40 cores (with dedup) |
| 5,000 | 52.7 / 94.7 ms | 0.20 cores |
| 10,000 | 95.7 / 187 ms | 0.14 cores |

Smaller batches buy median latency with CPU. In a 30 s run on default settings (base table
only), 1,000-row batches gave a visible p50 of 12.0 ms and p99 of 24.3 ms, at 6.12 cores of
the 7, against 22.2 / 42.2 ms at 1.98 cores with 2,000 rows in the same sweep.

The direct producer uses store-and-forward (`sf_dir`, `sender_id`): batches stay on local
disk until the server acknowledges them, and the client reconnects and replays them on its
own; dedup absorbs any batch replayed after the server already had it.

## Findings worth knowing

- **A materialized view costs more than a plain view.** At 50k evt/s the passthrough
  materialized view took QuestDB from 0.40 to 1.24 cores. A plain view stores nothing and
  costs only its query: about 3 ms for the last second at 50k evt/s, 5-7 ms at 100k.
- **S3 as a query**: polling for the sessions that closed in the last second, over a 30 s
  lookback, returns the same sessions as a full recompute (`s3check.py`) in 172 ms against
  1.2 s; recomputing every session grows with the lookback (9.7 s over 180 s).
- **Polling S5 every 100 ms** costs a roughly fixed amount per poll, most of it the join
  looking back over up to 10 s of orders: about 0.28 cores at 8k orders/s and 0.70 at 27k.
  The cost scales with how often you poll, not how much each poll reads; a two-tier
  consumer that mostly reads 300 ms windows saved nothing.
- **Long REST queries can stall ingestion on a small network pool.** REST and QWP share the
  port and the network pool (2 threads here). Two long analytical queries held both threads
  and payment commits stopped for 20.8 s until they finished. Size that pool for both, or run
  heavy analytics elsewhere.
- **The Kafka connector needs an integer timestamp.** The article's producer writes `ts` as
  a JSON float of epoch seconds, which the connector rejects
  (`Unsupported timestamp column type: VALUE_NUMBER_FLOAT`); `upstream_producer.py` runs it
  unchanged except for serializing `ts` as integer microseconds.
- **Connector tasks and batches matter.** Six tasks each flushing their own small batches
  cost QuestDB 1.38 cores at 50k evt/s; one task flushing 2,000 rows, 0.44 cores, about the
  same as the direct path.
- **CPU can differ between identical runs.** Stateless at 100k evt/s used 0.65 and 0.37
  cores in two runs with the same rows and latency, steadily for the whole run. Likely the
  two producers' commit timing against the idle-worker thresholds; not confirmed.

## Recovery after `kill -9`

The article kills a worker mid-run on S1 and measures how long results take to get back on
time. A single server has no worker to lose, so `aws/chaos.sh` kills the whole QuestDB
process with `kill -9` 45 s into a 150 s run at 50k evt/s (base table plus a materialized
view) and restarts it at once, as a supervisor (systemd, Kubernetes) would. On the direct
path the producer uses store-and-forward; through Kafka the broker holds the events. Both
the producer and the connector set `reconnect_max_backoff_millis=500`, so they reconnect
within half a second of the server being back (the client default backs off up to 5 s).

Two runs per path, every run exact: no row lost or duplicated, the view exact and `valid`,
and through Kafka consumer lag 0 and no connector errors.

| From the kill | Direct | Through Kafka |
|---|---|---|
| Every row sent during the outage visible | 3.0 / 2.7 s | 3.4 / 3.4 s |
| Recovered, as the article defines it | 3.2 / 2.8 s | 4.2 / 3.8 s |
| Every row visible within 100 ms again | 4.2 / 3.8 s | 5.2 / 4.8 s |

- "Every row sent during the outage visible" (`recovery_article.py`, `outage_visible_s`):
  the latest apply time of the transactions whose newest row was sent between the kill and
  the server answering again (WAL telemetry).
- "As the article defines it" (`recovery_article.py`, after the article's
  `bench/recovery_check.py`): per second of apply time, rows becoming visible and their
  mean delay from `ts`; recovered at the first second that starts 5 consecutive seconds each
  at 90% or more of the pre-kill rate with mean delay under 1 s.
- "Within 100 ms" (`aws/matrix.py`): per second of event time, the slowest row's delay;
  recovered when it stays under 100 ms for 3 consecutive seconds. Stricter, and it rounds
  up to whole seconds.
- QuestDB answered queries again 2.6 s after the kill direct and 2.9 s through Kafka
  (polled once a second, so an upper bound). The rest is applying what the client or Kafka
  held, about 2.5 s of events at 50k evt/s.
- A restart that waits adds that wait to every figure, and a client left at the default
  backoff can add up to a few seconds more, depending on where its next retry falls.

## Fairness notes and caveats

- The article's Flink and Spark arms are tuned (Flink buffer timeout 0 ms, Spark vendor
  settings) on dedicated EKS nodes, 3 cores / 6 GB per worker; its latency is the engine
  emitting the result, before any sink. QuestDB's latency includes the write to the
  database and the row becoming queryable.
- Kafka: one broker, replication factor 1 (the article: 3 brokers, RF 3).
- Duplicates: the article sends 10% late duplicates only for dedup (S2). The runs here
  carry them everywhere except stateless at 100k, S1 and S5, which have none as in the
  article; S3 and S4 query times were measured on data with duplicates.
- Durability is the WAL on EBS; the article checkpoints to S3.

## Files

- `producer.py`: port of the article's producer (same events, 1M keys, duplicates re-sent
  0-3 s later), writing over QWP; `--batch`, `--sf-dir`, `--dup-rate`, `--conf-extra`.
- `s5producer.py`: S5 orders and payments with the article's rules and the expected joins.
- `upstream_producer.py`, `mktopic.py`: the article's producer for the Kafka path; topics.
- `mktable.py`, `mkref.py`: tables and views; the S4 users table.
- `latency.py`, `cover_latency.py`, `view_latency.py`, `walids.py`, `applylag.py`:
  latency from WAL telemetry; `recovery_article.py`: recovery time with the article's
  definition, and when every row sent during the outage is visible.
- `qtime.py`, `poll.py`, `s3check.py`, `s5check.py`, `s5consumer.py`: query times, polling,
  correctness checks, the S5 consumer.
- `cpusample.py`, `cpuslice.py`, `summarize.py`: CPU and memory.
- `archive.py`: exports a run's telemetry, partitions and row counts.
- `common.py`: settings and the REST helper.
- `aws/`: host setup, `start_qdb.sh`, `start_connect.sh` / `stop_connect.sh`,
  `kafka_produce.sh`, `chaos.sh`, `run.sh` / `sweep.sh` for single runs, `matrix.py` for
  scenarios, server configs and block device mappings.

## What this shows

Not "QuestDB is faster than streaming engines": Flink and Spark emit each event with a
lower median and push their results. For many of these workloads you don't need a stream
processor: dedup, transforms, windowed aggregates, joins and enrichment run in the database
the results would have been written to anyway, which removes a component and the hop from
the engine to its sink.
