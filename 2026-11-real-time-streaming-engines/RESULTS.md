# Results: QuestDB with and without Kafka vs the article's Flink and Spark RTM

**Flink and Spark RTM** ([fmorillo7694/spark-rtm-vs-flink](https://github.com/fmorillo7694/spark-rtm-vs-flink)):
the tuned arms (Spark = vendor-tuned RTM) of `results/final30/RESULTS.md`, 30-minute runs at
50k evt/s (stateless 100k), mean of 2 runs per cell. **S5 is the exception**: it is not in
`final30`; its figures come from `results/final-run/b` (5-minute window after a 60 s warm-up,
2 runs) at 8k orders/s, and `results/final-run/a` at 16k orders/s (1 run). His
`FINAL_RUN_PLAN.md` sets S5 at 1,000 per producer pod, ~8k orders + ~6.8k payments/s.

**QuestDB**: this folder's AWS runs, QuestDB 10.2.0 with `aws/pools_nap.conf`, 7 pinned vCPUs
on an m5.2xlarge, producer on its own c7i.2xlarge, one Kafka broker on an m5.2xlarge; method
in `README.md`, scenarios in `aws/matrix.py`. Every scenario ran twice on fresh tables, except
the 30-minute S1 and S5 runs and the S5 consumer runs, which ran once. Cells show **run 1 /
run 2**. Every run on both paths was exact (row counts = distinct ids = originals sent; views
exact and `valid`; S5 joins = the producer's ground truth). Kafka-path latency excludes the
first 10 s of each run (consumer-group start; the article excludes 60 s) and is a lower ..
upper bound (`cover_latency.py`).

**What the latency numbers measure.** Flink and Spark: `out_ts - ts`, the engine emitting the
result (no Kafka output read, no sink). QuestDB: `ts` to the row being applied and readable.
For query-time workloads the result also needs the query (tables below) and, for a client
polling every second, up to 1 s of waiting for the next poll.

## Latency, p50 / p99

| Workload | Flink | Spark RTM | QuestDB direct (run 1 / run 2) | QuestDB via Kafka (run 1 / run 2) |
|---|---|---|---|---|
| Dedup (S2), 50k evt/s | 7.0 ms / 1.7 s | 13.1 ms / 4.3 s | 10% late duplicates. 30 min: 23.5 / 45.5 ms and 23.6 / 46.3 ms; 5 min: 22.9 / 42.8 and 23.0 / 43.0 ms | 5 min: 28-29 / 50-52 and 28-29 / 49-51 ms |
| Stateless, 100k evt/s | 6.1 / 28 ms | 6.3 / 30 ms | no duplicates, base table plus the plain view: base 22.0 / 35.4 and 21.6 / 35.3 ms; plain view query (last 1 s, ~32k rows) 6.0 / 49.9 and 4.6 / 10.0 ms | the same: base 19.4-37.5 / 45.0-50.6 and 18.7-38.1 / 44.5-51.7 ms; plain view query 7.4 / 40.6 and 6.0 / 15.2 ms |
| S1 10 s window per user | 6.9 ms / 1.6 s (per update) | 13.4 ms / 4.5 s | materialized view, no duplicates (as the article's S1). 30 min, 1 run: p50 64-155 ms, p99 206-333 ms; 5 min: p50 63-155 / 64-159 ms, p99 166-307 / 169-311 ms | materialized view, 5 min, no duplicates: p50 68-153 / 68-154 ms, p99 172-306 / 173-305 ms (base 28-30 / 51-56 and 28-30 / 51-55 ms) |
| S3 sessions | 8.3 ms / 4.3 s | falls behind (0.76x rate) | query, polled every 1 s: see below; keeps up | same |
| S4 enrichment | 6.4 / 20 ms | 6.7 / 287 ms | query, polled every 1 s: see below | same |
| S5 interval join | see the S5 section | not in RTM | see the S5 section | see the S5 section |

The article's S1 runs had no duplicates (`dups=0` in `final30/.../producers_done.txt`); only
its S2 used 10%. 30-minute S1 run, per 5 minutes: view p99 164-185 / 302-320 ms (lower /
upper bound) in minutes 0-25, then 1.07 / 1.17 s in minutes 25-30 (max 1.9-2.0 s, p95 also
up to 170 / 301 ms), with QuestDB CPU flat at 1.98 cores and the base table unaffected (p99
34.5-40.0 ms in every window). Cause not identified. Exact: 90,000,500 rows. With 10% late
duplicates, as part of the 30-minute run with every view and query, the S1 view's p99 was
0.31-0.51 / 0.33-0.54 s.

## Query-time workloads, each polled once a second during ingest

`poll.py`: each query on its own thread, once a second, for 230 s from 60 s into a 5-minute
run at 50k evt/s with 10% duplicates and all views (base, passthrough, plain view, S1 view),
all six queries running together. Execute time, QuestDB's REST timings, 230 polls per query
(so p99 is about the 3rd-slowest).

| Query | Direct p50 / p99 (run 1 / run 2) | Via Kafka p50 / p99 (run 1 / run 2) |
|---|---|---|
| S1 live, current 10 s window (~430k groups) | 86.8 / 200.0 and 90.9 / 153.7 ms | 88.4 / 154.9 and 93.8 / 149.1 ms |
| S3 sessions closed in the last 1 s, 30 s lookback (~39k sessions) | 274.1 / 401.0 and 271.1 / 339.3 ms | 300.0 / 380.8 and 335.1 / 412.7 ms |
| S4 enrich last 1 s (~47k rows) | 37.1 / 86.5 and 34.0 / 78.8 ms | 45.3 / 128.6 and 53.8 / 94.7 ms |
| S4 enrich last 100 ms | 17.0 / 82.0 and 20.2 / 58.7 ms | 26.6 / 81.5 and 29.7 / 80.2 ms |
| Stateless plain view, last 1 s | 3.6 / 15.4 and 3.3 / 5.8 ms | 3.9 / 26.7 and 4.1 / 25.6 ms |
| Stateless plain view, last 100 ms | 0.3 / 2.2 and 0.2 / 1.6 ms | 0.3 / 13.7 and 0.3 / 11.4 ms |

With the six pollers running, ingest latency of the base table rose to p50 23.7 / p99 94.9 and
23.7 / 89.1 ms direct (~46 ms p99 without them), and QuestDB used 2.73 / 2.67 cores direct,
3.05 + 0.51 / 3.09 + 0.48 (Connect) through Kafka (consumer lag 0, 0 connector errors). For
comparison, the 30-minute runs timed the same queries 5 times at 25 minutes (medians): S1
live 67 / 67 ms, S3 232 / 277 ms, S4 17 / 26 ms direct.

## S5: orders JOIN payments, payment within 10 s of its order

Data rules as the article: 80% of orders get one payment 0-9.5 s later (must join), 5% get a
late one 11-14 s later (must not), 15% none. The query, polled once a second:

```sql
SELECT /*+ asof_dense(p o) */ p.ts pay_ts, o.ts order_ts, o.order_id, o.user_id,
       o.amount order_amount, p.amount pay_amount
FROM payments p
ASOF JOIN orders o ON (order_id) TOLERANCE 10s
WHERE p.ts IN '$now-1s..$now' AND o.ts IS NOT NULL;
```

Without the `asof_dense` hint the default ASOF algorithm scans backward per payment (its
order is up to ~250k rows back) and the poll takes seconds. The interval-join form
(`JOIN ... WHERE p.ts BETWEEN o.ts AND o.ts + 10s`) returns the same pairs; both are checked
in every run against the producer's ground truth (8k: 1,920,018 direct / 1,920,091 Kafka
pairs; 27k: 6,480,691 / 6,528,173; no late payment joins).

| | Flink tuned | Flink default | Spark micro-batch | QuestDB direct | QuestDB via Kafka |
|---|---|---|---|---|---|
| **8k orders/s** (his S5 rate), p50 / p99 | 7.5 / 690 and 7.5 / 695 ms | 61 / 624 and 62 / 682 ms | 3.25 / 5.36 and 3.26 / 5.56 s | payment visible 21.6 / 52.2 ms (both runs); ASOF poll 12.4 / 22.4 and 11.8 / 20.1 ms | payment visible 25.4-26.0 / 45.2-45.7 and 24.1-27.0 / 43.7-47.7 ms; ASOF poll 16.0 / 33.7 and 14.8 / 33.7 ms |
| 16k orders/s (1 run) | 33 ms / 3.5 s | 88 ms / 3.9 s | | | |
| **27k orders/s** (~50k events/s, the other workloads' rate) | | | | payment visible 22.5 / 72.1 and 22.4 / 71.0 ms; ASOF poll 47.7 / 65.0 and 48.4 / 67.8 ms | payment visible 28.4-29.2 / 61.4-62.5 and 28.2-29.1 / 59.5-60.2 ms; ASOF poll 56.7 / 83.5 and 56.6 / 79.5 ms |

- QuestDB at 8k uses 500-row client batches (and connector batches), so a batch fills in
  ~35 ms as at the other rates. With the default 2,000 rows used everywhere else, a batch at
  8k takes ~135 ms to fill: payment visible p50 72.7 / 72.3 ms, p99 200.9 / 199.9 ms direct;
  ~72-77 / ~141 ms via Kafka.
- Interval-join poll for comparison: 8k direct 9.9 / 17.3 and 9.7 / 18.8 ms, Kafka 12.6 / 24.1
  and 11.7 / 27.1 ms; 27k direct 40.3 / 56.5 and 40.7 / 55.8 ms, Kafka 44.8 / 61.8 and
  45.3 / 66.7 ms. The ASOF form costs 2-4 ms (8k) to 7-12 ms (27k) more.
- Total time to a polling consumer = payment visible + wait for the next poll + poll. Flink
  pushes each pair as it is emitted.
- 27k ingest latency with nothing reading: p99 62.4 / 62.2 ms direct, 50-51 ms (both runs)
  via Kafka.
- Flink at ~27k orders/s: only an earlier run on an older image (`eks-session-c`, v43b, same
  tuned settings and resources) exists; its output fell from ~21k to ~9-10k pairs/s and p50
  reached 6.5 s. Not the article's measurement setup, so not used here.
- CPU, QuestDB, with the two once-a-second polls: 8k 0.34 / 0.33 cores direct, 0.81 / 0.81
  via Kafka (QuestDB 0.51 / 0.51 + Connect 0.30 / 0.29); 27k 0.36 / 0.36 direct, 0.95 / 0.97
  via Kafka. With nothing reading, 27k: 0.34 / 0.32 direct, 0.89 / 0.92 via Kafka.
- 30-minute S5 run at 8k orders/s (500-row batches), 1 run, with the 1 s polls and two
  polling consumers (`s5consumer.py`, overlapping 1.5 s window, skipping payment ids already
  seen; end to end = when the consumer got the pair minus the payment's ts, same host clock):

  | | p50 | p99 | max | complete |
  |---|---|---|---|---|
  | consumer polling every 100 ms, end to end | 89.0 ms | 145.1 ms | 164.9 ms | 10,868,955 / 10,868,955 |
  | consumer polling every 1 s, end to end | 541.4 ms | 1,031.9 ms | 1,067.6 ms | 10,863,109 / 10,863,109 |
  | ASOF poll (1 s window), execute | 11.5 ms | 19.3 ms | 22.6 ms | |
  | consumer query (1.5 s window), execute | 14.3-14.4 ms | 22.7-24.2 ms | | |
  | payment visible | 21.7 ms | 42.3 ms | | |

  Flat over the run: per 5 minutes the 100 ms consumer stays at p50 86.8-91.5 / p99
  144.1-149.6 ms and the ASOF poll at p50 11.4-11.7 / p99 18.8-19.7 ms; payment visible p50
  21.6-21.7 / p99 41.9-42.0 ms / max 70-81 ms in minutes 0-25. Joins exact: 11,522,092 pairs,
  719,377 late payments excluded, ASOF = interval join. QuestDB 0.63 cores with the 1 s polls
  and both consumers (0.34 with the 1 s polls only, 5-minute runs).
- 30-minute S5 run at 27k orders/s (2,000-row batches), 1 run, same setup:

  | | p50 | p99 | max | complete |
  |---|---|---|---|---|
  | consumer polling every 100 ms, end to end | 151.3 ms | 234.1 ms | 277.4 ms | 36,675,495 / 36,675,683 (see below) |
  | consumer polling every 1 s, end to end | 635.9 ms | 1,127.7 ms | 1,185.1 ms | 36,655,857 / 36,656,045 |
  | ASOF poll (1 s window), execute | 46.1 ms | 58.9 ms | 63.6 ms | |
  | consumer query (1.5 s window), execute | 54.7-55.6 ms | 68.5-71.0 ms | | |
  | payment visible, minutes 0-25 | 23.0-23.5 ms | 74.6-78.3 ms | 138-148 ms | |

  Flat over the run: per 5 minutes the 100 ms consumer stays at p50 149.2-152.2 / p99
  231.6-236.0 ms and the ASOF poll at p50 45.9-46.3 / p99 58.0-60.0 ms. Joins exact:
  38,879,826 pairs, 2,429,070 late payments excluded, ASOF = interval join. QuestDB 1.13
  cores with the 1 s polls and both consumers. The 188 pairs missing from both consumers'
  counts are a counting error in that run: payments newer than the consumer's start that
  its first poll already returned were counted as expected but not as seen (fixed in
  `s5consumer.py`). No pair arrived later than 277 ms.
- **100 ms consumer, 5-minute runs, one each** (`s5consumer.py`, end to end = pair received
  minus payment ts; completeness checked against the database):

  | | 8k direct | 8k via Kafka | 27k direct | 27k via Kafka |
  |---|---|---|---|---|
  | end to end p50 / p99 / max | 89.7 / 145.2 / 164.1 ms | 97.2 / 158.0 / 177.0 ms | 139.7 / 201.0 / 222.3 ms | 158.5 / 221.6 / 249.6 ms |
  | pairs seen / in the database | 1,266,688 / 1,266,688 | 1,266,979 / 1,266,979 | 4,276,260 / 4,276,260 | 4,306,762 / 4,306,762 |
  | consumer query p50 / p99 | 14.3 / 21.5 ms | 17.0 / 26.2 ms | 53.5 / 68.4 ms | 62.8 / 78.7 ms |

  Two-tier variant (every 100 ms the last 300 ms, every 1 s a 1.5 s sweep, all seven columns
  of the post's query): 8k direct 86.6 / 147.4 / 176.2 ms, 27k direct 135.9 / 239.0 / 280.4
  ms, both complete (1,266,686 and 4,276,064 pairs).
- **What polling every 100 ms costs** (QuestDB CPU averaged over 70-250 s, while the consumer
  polls, against a run with nothing reading):

  | | nothing reading | 1.5 s window, 2 columns | two-tier, 7 columns | cost of the consumer |
  |---|---|---|---|---|
  | 8k orders/s | 0.318 | 0.603 | 0.596 | ~0.28 cores |
  | 27k orders/s | 0.298 | 1.000 | 0.990 | ~0.70 cores |

  The window size and column count barely matter: each poll costs ~28 ms (8k) / ~70 ms
  (27k) of server CPU, mostly the ASOF JOIN's look-back over the orders (up to 10 s of them,
  to find each payment's order), so the cost scales with how often you poll. Ingest is
  unaffected (payment visible p99 52 ms at 8k, 62 ms at 27k, as without readers).
- **Long REST queries share the network pool with ingestion.** When the 30-minute runs'
  consumers stopped measuring, each ran a completeness check (one ASOF JOIN over the whole
  run, over REST) while the producer was still writing. HTTP and QWP share the port and the
  network pool, 2 threads here (`shared.network.worker.count=2`): the two long queries held
  both threads and payment commits stopped for 20.8 s at 27k (QuestDB at 2.0 cores, RSS 1.4
  -> 5.4 GiB) and up to 5.9 s at 8k. The per-5-minute figures above exclude that last
  window; `s5consumer.py` now runs its check after the producer stops. In a deployment, size
  that pool for both, or run heavy analytics elsewhere.

## Operations

| | Flink | Spark RTM | QuestDB direct (run 1 / run 2) | QuestDB via Kafka (run 1 / run 2) |
|---|---|---|---|---|
| Recovery after a kill | 62-70 s | not recovered in window | `kill -9`, restarted at once, client backoff capped at 500 ms: every outage row visible 3.0 / 2.7 s after the kill; recovered (article's definition) 3.2 / 2.8 s; every row within 100 ms again 4.2 / 3.8 s; zero loss/dups (store-and-forward + dedup) | same: 3.4 / 3.4 s; 4.2 / 3.8 s; 5.2 / 4.8 s; zero loss/dups (Kafka offsets + dedup) |
| Scaling 50k -> 100k -> 50k | p99 < 2 s sized for peak; autoscaler ~2.75 s | no autoscaling | S1 view p99 0.42-0.80 / 0.39-0.75 s at peak; base p99 56 / 60 ms at peak; nothing rescaled | S1 view p99 0.34-0.63 / 0.30-0.56 s at peak; base p99 47-62 / 50-65 ms at peak; nothing rescaled |
| 30-minute run | steady | S3 falls behind | flat every 5 min (both runs) | flat, consumer lag 0 at end (both runs) |

Recovery, every run exact (`recovery_article.py`, `aws/matrix.py`):

- "Every outage row visible": the latest apply time of the transactions whose newest row was
  sent between the kill and QuestDB answering again.
- "Article's definition" (after its `bench/recovery_check.py`): the first second after the
  kill that starts 5 seconds each at >= 90% of the pre-kill rate with mean delay < 1 s.
- "Within 100 ms": the slowest row per second of event time under 100 ms for 3 seconds.
- QuestDB answered queries again 2.6 s (direct) / 2.9 s (Kafka) after the kill.
- A restart that waits adds that wait to every figure, and the client's default reconnect
  backoff (up to 5 s) can add a few seconds more: with a 5 s wait before the restart and the
  default backoff, the article's definition gave 8.8-13.2 s direct and 10.2-12.3 s through
  Kafka, varying with where the client's next retry fell.

## Resources (engine side only; Kafka brokers excluded everywhere, as in the article)

Mean cores, then mean RSS, run 1 / run 2. Via Kafka: QuestDB + Connect. Where a run also
polls, CPU is over a window with nothing reading.

| Workload | Flink | Spark RTM | QuestDB direct | QuestDB via Kafka |
|---|---|---|---|---|
| Dedup only, 50k | 3.7 cores / 6.2 GiB | 3.6 / 9.2 | 0.40 / 0.41 cores, ~1.1 GiB | 0.74 / 0.80 cores (QuestDB 0.44 / 0.48), ~1.7 GiB |
| S1 only, 50k | 3.5 / 7.1 | 4.0 / 9.5 | 30 min: 1.98 cores, 0.94 GiB; 5 min: 2.07 / 1.95 cores, 1.57 / 1.11 GiB | 2.81 / 2.79 cores (QuestDB 2.45 / 2.43); 1.27 + 0.58 / 1.34 + 0.59 GiB |
| Dedup + S1 + stateless views + queries, 30 min | several jobs | | 2.31 / 2.34 cores; 1.30 / 1.18 GiB | 3.12 / 3.13 cores (QuestDB 2.74 / 2.73); 1.32 + 0.59 / 1.91 + 0.57 GiB |
| Stateless, 100k | 1.1 / 5.5 | 1.7 / 8.1 | base table plus the plain view: 0.65 / 0.37 cores; 0.85 / 1.28 GiB (stores every event) | the same: 1.72 / 1.54 cores (QuestDB 0.77 / 0.74 + Connect 0.94 / 0.81); 1.29 + 0.61 / 1.23 + 0.60 GiB |
| S5, 8k orders/s, two polls a second | | | 0.34 / 0.33 cores; 0.93 / 1.08 GiB | 0.81 / 0.81 cores; 1.05 + 0.60 / 1.14 + 0.61 GiB |
| S5, 27k orders/s, nothing reading | | | 0.34 / 0.32 cores, 1.1-1.4 GiB | 0.89 / 0.92 cores, 2.0-2.5 GiB |
| Scaling, 50k / 100k / 50k (base + S1 view) | | | 2.32 / 3.60 / 2.33 and 2.43 / 3.68 / 2.42 | 3.03 / 3.92 / 3.00 and 3.11 / 4.05 / 3.04 |

- Stateless at 100k: every run exact (30,001,000 rows direct, 29,760,992 through Kafka,
  consumer lag 0, no connector errors); CPU over 10-190 s, before the plain view is polled.
  The two direct runs differ steadily for the whole run (0.61-0.68 against 0.36-0.38 cores
  every 30 s) with the same rows and latency; likely the two producers' commit timing
  against the idle-worker nap/sleep thresholds, not confirmed.
- The 30-minute row also maintains a passthrough materialized view of the stateless
  transform, which costs more than the plain view (at 50k it took QuestDB from 0.40 to 1.24
  cores), so it overstates what the plain-view setup needs.
- Memory: Flink and Spark report cgroup working set; QuestDB is process RSS (memory-mapped
  data partly sits in page cache outside RSS), so the gap is real but smaller than shown.
  Run 2 of the Kafka 30-minute run started on the same QuestDB process right after the
  direct 30-minute run's analysis queries, so its RSS was 4.1 GiB in the first 5 minutes and
  1.3-1.4 GiB from minute 10 (mean 1.91). The direct 30-minute run 1 peaks at 7 GiB only
  after the run, during its analysis queries; within the run it stays 1.10-1.63 GiB.

## Reading it

- Medians: Flink and Spark emit each event in ~6-13 ms; QuestDB's 22-29 ms is the database
  write and visibility, mostly client batching.
- Tails: their p99s for S1 to S3, 1.6-4.5 s, come from checkpoints to S3; QuestDB stays at
  ~45-75 ms p99 for ingest, dedup and S5 payments, at 8k and at 27k orders/s. On S4 Flink's
  p99 (20 ms) is lower than QuestDB's query time (87 ms).
- S5: at his rate, a payment is stored and joinable in 22 ms p50 / 52 ms p99. Flink pushes
  each pair, so it wins the median; a consumer polling every 100 ms receives each pair in
  90 ms p50 / 145 ms p99, below Flink's 690 ms p99. Polling every second, the wait for the
  next poll dominates (541 / 1,032 ms).
- Query-time workloads run once a second with ingest: S4 34-54 ms p50, S1 live 87-94 ms, S3
  271-335 ms, all six together, with p99 up to ~410 ms (S3).
- Kafka is optional for QuestDB: client store-and-forward and dedup cover buffering,
  disconnects and restarts. Through Kafka it costs ~5 ms and ~0.3-0.5 cores for the connector
  at 50k evt/s, ~0.8-0.9 at 100k.
- The two runs of each cell agree within a few ms on latency; CPU agrees within a few
  hundredths of a core except stateless direct at 100k (0.65 / 0.37 cores).

## Caveats

- QuestDB: one Kafka broker (article: 3 brokers, RF 3); latency through Kafka is bounded, not
  exact per row; durability is the WAL on EBS, theirs is S3.
- QuestDB config changes from defaults: merge-append (without it the views fell seconds behind
  within 5 minutes), dedicated pools, earlier nap/sleep for idle workers.
- Client batch size sets QuestDB's median: 2,000 rows at 50k evt/s, 500 rows at 8k orders/s.
- Poll runs are 5 minutes (230 polls per query); his S3 and S4 figures are 30-minute runs.
