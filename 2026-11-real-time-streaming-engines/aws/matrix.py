#!/usr/bin/env python3
"""
Run one benchmark scenario end to end from your machine, over ssh to the hosts, and log
every command and result to runs/matrix/<scenario>_<rep>.log. Lines starting with RESULT
are the figures that go into RESULTS.md.

Usage: .venv/bin/python aws/matrix.py <scenario>[,<scenario>...] <rep>
       e.g. aws/matrix.py direct_dedup,kafka_dedup r1      (tables are m_<scenario>_<rep>)

All scenarios use the tuned server config (SERVER_CONF, default pools_nap.conf) and 2,000-row
batches unless noted. "Kafka" scenarios run the article's producer (8 processes, 50k evt/s
unless noted) into Kafka and the QuestDB connector into QuestDB.

Dedup and stateless
  direct_dedup          base table, 50k evt/s, 10% late duplicates, 300 s
  direct_passthrough    + passthrough materialized view
  direct_100k           two producers at 50k, 10% dups, base + passthrough + plain view
  direct_100k_plain     two producers at 50k, no dups, base + plain view only; CPU over
                        10-190 s with nothing reading, plain view polled from 200 s
  kafka_dedup           connector 1 task, 10% late duplicates, 300 s
  kafka_100k            16 producers (100k), connector 2 tasks, base + both stateless views
  kafka_100k_plain      the same with the plain view only

S1 windows
  direct_s1             50k evt/s, no dups, base + S1 view, 300 s
  direct_s1soak         the same for 30 minutes
  kafka_s1              the same through Kafka, connector 1 task, 300 s

Query-time workloads (each query once a second during ingest, poll.py)
  direct_poll           50k evt/s, 10% dups, all views; S1 live, S3, S4, plain view
  kafka_poll            the same through Kafka, connector 1 task

S5 join (s5producer.py rules; both join forms checked against the ground truth)
  direct_s5             27k orders/s, 300 s
  direct_s5poll         the same with both S5 polls once a second
  direct_s5poll8k       8k orders/s (the article's S5 rate), 2,000-row batches
  direct_s5poll8k_b500  8k orders/s, 500-row batches
  direct_s5soak         8k orders/s, 500-row batches, 30 minutes, polls plus s5consumer.py
                        every 100 ms and every 1 s
  direct_s5soak27k      the same at 27k orders/s, 2,000-row batches
  direct_s5base8k, direct_s5base27k
                        nothing reading (CPU baseline)
  direct_s5c100_8k, direct_s5c100_27k
                        only the 100 ms consumer (its CPU cost and end-to-end latency)
  direct_s5tier_8k, direct_s5tier_27k
                        the 100 ms consumer in two-tier mode (300 ms polls, 1.5 s sweep)
  kafka_s5, kafka_s5poll, kafka_s5poll8k, kafka_s5poll8k_b500, kafka_s5c100_8k,
  kafka_s5c100_27k      the same through Kafka (orders and payments topics, 1 task)

Recovery (kill -9 QuestDB 45 s into a 150 s run; reports startup time and when every row
is visible again, and copies the QuestDB log to runs/qdb_logs/<table>/)
  direct_recovery       restart 5 s after the kill, store-and-forward on
  direct_recovery0      restart at once
  direct_recovery_b500, direct_recovery0_b500
                        the same with the client reconnect backoff capped at 500 ms
  kafka_recovery, kafka_recovery0, kafka_recovery_b500, kafka_recovery0_b500
                        the same through Kafka (the connector's backoff for _b500)

Scaling and 30 minutes
  direct_scale          50k -> 100k -> 50k (second producer 100-200 s), base + S1 view
  kafka_scale           8 producers + 8 more at 100-200 s, connector 2 tasks
  direct_soak           30 minutes, 10% dups, all views, queries timed at 25 min
  kafka_soak            the same through Kafka

Settings (environment or .env): SSH_KEY, SSH_USER (default ubuntu); SERVER_HOST,
PRODUCER_HOST, KAFKA_HOST (addresses this machine reaches; QDB_ADDR must point at the
server too, for the analysis queries); SERVER_PRIVATE, KAFKA_PRIVATE (addresses the hosts
use between them); REMOTE_DIR (default bench), SF_DIR (default /home/ubuntu/sf),
QDB_CPUS (default 1-7), SERVER_CONF (default pools_nap.conf).
"""
import datetime as dt
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common import RUNS_DIR, sql  # noqa: E402  (also loads .env)

E = os.environ
KEY, USER = E["SSH_KEY"], E.get("SSH_USER", "ubuntu")
SERVER, PRODUCER, KAFKA = E["SERVER_HOST"], E["PRODUCER_HOST"], E.get("KAFKA_HOST", "")
SERVER_PRIV, KAFKA_PRIV = E["SERVER_PRIVATE"], E.get("KAFKA_PRIVATE", "")
RD = E.get("REMOTE_DIR", "bench")
SF = E.get("SF_DIR", "/home/ubuntu/sf")
CPUS = E.get("QDB_CPUS", "1-7")
CONF = E.get("SERVER_CONF", "pools_nap.conf")
PY = f"{RD}/.venv/bin/python"
QDB = f"QDB_ADDR={SERVER_PRIV}:9000"
KB = f"KAFKA_BOOTSTRAP={KAFKA_PRIV}:9092"
CLEAN = "event_type IN ('purchase', 'add_to_cart') AND amount > 0"
SETTLE_S = 15   # wait after the connector tasks start, before producing
WARMUP_S = 10   # Kafka-path latency excludes the first seconds (the article excludes 60 s)

_log = None
_connect = []   # Connect workers to stop if a scenario fails


def log(msg):
    print(msg, flush=True)
    _log.write(msg + "\n")
    _log.flush()


def result(**kv):
    log("RESULT " + " ".join(f"{k}={v}" for k, v in kv.items()))


def ssh(host, args):
    # Keepalives: a dropped connection fails within a minute instead of hanging.
    return ["ssh", "-i", KEY, "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=4", f"{USER}@{host}", shlex.join(args)]


def run(host, *args, check=True, quiet=False):
    log(f"$ [{host}] {shlex.join(args)}")
    p = subprocess.run(ssh(host, args), capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip()
    if out and not quiet:
        log(out)
    if check and p.returncode:
        raise SystemExit(f"command failed with exit code {p.returncode}")
    return p.stdout


def start(host, *args):
    log(f"$ [{host}] & {shlex.join(args)}")
    return subprocess.Popen(ssh(host, args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)


def finish(p):
    out = p.communicate()[0].strip()
    if out:
        log(out)
    if p.returncode:
        raise SystemExit(f"background command failed with exit code {p.returncode}")
    return out


def local(script, *args):
    log(f"$ {script} {shlex.join(args)}")
    p = subprocess.run([sys.executable, os.path.join(ROOT, script), *args], capture_output=True,
                       text=True)
    out = (p.stdout + p.stderr).strip()
    log(out)
    return out


def fetch(host, remote):
    os.makedirs(RUNS_DIR, exist_ok=True)
    subprocess.run(["scp", "-q", "-i", KEY, f"{USER}@{host}:{remote}", RUNS_DIR + "/"], check=True)


def iso(epoch):
    return dt.datetime.fromtimestamp(epoch, dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")


# ---- building blocks ---------------------------------------------------------------------

def mktable(t, *flags):
    run(PRODUCER, "env", QDB, PY, f"{RD}/mktable.py", t, "--dedup", "SKIP", *flags)


def sampler(run_id, seconds, connect=False):
    return start(SERVER, "python3", f"{RD}/cpusample.py", run_id, str(seconds),
                 *(["--extra", "ConnectStandalone"] if connect else []))


def producer(run_id, table, seconds, *extra, quiet=True):
    return start(PRODUCER, "env", QDB, PY, f"{RD}/producer.py", "--batch", "2000", "--table",
                 table, "--seconds", str(seconds), "--run-id", run_id,
                 *(["--quiet"] if quiet else []), *extra)


def producer_meta(run_id):
    fetch(PRODUCER, f"{RD}/runs/{run_id}.json")
    with open(os.path.join(RUNS_DIR, f"{run_id}.json")) as f:
        return json.load(f)


def cpu(run_id, windows, connect=False):
    """Mean CPU/RSS of QuestDB (and Connect) per (label, t0, t1) wall-clock window."""
    fetch(SERVER, f"{RD}/runs/{run_id}_cpu.json")
    with open(os.path.join(RUNS_DIR, f"{run_id}_cpu.json")) as f:
        c = json.load(f)
    for label, t0, t1 in windows:
        s = [x for x in c["samples"] if t0 <= c["started"] + x[0] <= t1]
        kv = {"window": label, "qdb_cpu": f"{statistics.mean(x[1] for x in s):.2f}",
              "qdb_rss_gib": f"{statistics.mean(x[2] for x in s):.2f}"}
        if connect:
            kv["connect_cpu"] = f"{statistics.mean(x[3] for x in s):.2f}"
            kv["connect_rss_gib"] = f"{statistics.mean(x[5] for x in s):.2f}"
            kv["total_cpu"] = f"{statistics.mean(x[1] + x[3] for x in s):.2f}"
        result(**kv)


def rows(t, expected):
    n, d = sql(f"SELECT count(), count_distinct(event_id) FROM {t}")["dataset"][0]
    result(table=t, rows=n, distinct=d, expected=expected, exact=(n == d == expected))


def view_rows(t):
    want = sql(f"SELECT count() FROM {t} WHERE {CLEAN}")["dataset"][0][0]
    got = sql(f"SELECT count() FROM {t}_clean_mv")["dataset"][0][0]
    result(view=f"{t}_clean_mv", rows=got, expected=want, exact=(got == want))


def view_status(t):
    for name, status, refreshed, base in sql(
            f"SELECT view_name, view_status, refresh_base_table_txn, base_table_txn "
            f"FROM materialized_views() WHERE view_name LIKE '{t}%'")["dataset"]:
        result(view=name, status=status, caught_up=(refreshed == base))


def wal_writers(t):
    tid = sql(f"SELECT id FROM tables() WHERE table_name = '{t}'")["dataset"][0][0]
    return [r[0] for r in sql(f"SELECT walId, min(created) f FROM 'sys.telemetry_wal' "
                              f"WHERE tableId = {tid} AND event = 109 GROUP BY walId "
                              f"ORDER BY f")["dataset"]]


def recovery_profile(t, kill):
    """Per second around the kill: max(apply time - newest row ts) of the transactions."""
    tid = sql(f"SELECT id FROM tables() WHERE table_name = '{t}'")["dataset"][0][0]
    lo = iso(kill - 6)
    res = sql(f"""
WITH c AS (SELECT seqTxn, maxTimestamp hi FROM 'sys.telemetry_wal' WHERE tableId = {tid} AND event = 109),
a AS (SELECT seqTxn, created applied FROM 'sys.telemetry_wal' WHERE tableId = {tid} AND event = 105),
j AS ((SELECT c.hi, (a.applied - c.hi) / 1000.0 lat_ms FROM c JOIN a ON a.seqTxn = c.seqTxn
       ORDER BY c.hi) timestamp(hi))
SELECT hi, round(max(lat_ms)) FROM j WHERE hi IN '{lo};45s' SAMPLE BY 1s""")["dataset"]
    for ts, mx in res:
        log(f"  {ts}  max_ms={mx}")
    worst = max(mx for _, mx in res)
    back = None
    for i, (ts, mx) in enumerate(res):
        t_s = dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
        if t_s > kill and all(m < 100 for _, m in res[i:i + 3]):
            back = t_s + 1 - kill
            break
    result(worst_latency_ms=worst, normal_after_kill_s=f"{back:.1f}" if back else "n/a")


def chaos(down=5):
    return start(SERVER, "env", f"QDB_CPUS={CPUS}", "bash", f"{RD}/aws/chaos.sh", "45", str(down),
                 CONF)


def chaos_epoch(chaos_out, label):
    """UTC epoch of a 'kill -9 <pid> at', 'restart at' or 'up at' line of chaos.sh."""
    hms = re.search(rf"{label} at (\d+:\d+:\d+\.\d{{6}})", chaos_out).group(1)
    day = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
    return dt.datetime.fromisoformat(f"{day}T{hms}+00:00").timestamp()


def kill_epoch(chaos_out):
    return chaos_epoch(chaos_out, r"kill -9 \d+")


def startup_times(chaos_out):
    """How long QuestDB took to answer a query after the restart, and after the kill."""
    kill, restart = kill_epoch(chaos_out), chaos_epoch(chaos_out, "restart")
    up = chaos_epoch(chaos_out, "up")
    result(restart_after_kill_s=f"{restart - kill:.2f}", startup_s=f"{up - restart:.2f}",
           answering_after_kill_s=f"{up - kill:.2f}")


def table_count():
    result(tables_at_restart=sql("SELECT count() FROM tables()")["dataset"][0][0],
           mat_views=sql("SELECT count() FROM materialized_views()")["dataset"][0][0])


def fetch_qdb_log(t):
    """Copy the server's QuestDB log directory (startup phases of the restart) to runs/."""
    dest = os.path.join(RUNS_DIR, "qdb_logs", t)
    os.makedirs(dest, exist_ok=True)
    subprocess.run(["scp", "-q", "-r", "-i", KEY, "-o", "ServerAliveInterval=15",
                    f"{USER}@{SERVER}:/data/qdb/log", dest], check=False)
    log(f"QuestDB log copied to {dest}")


def kafka_prep(name, tasks, topics=None, per_topic=False, batch=2000, conf_extra=""):
    for topic in (topics or [name]):
        run(PRODUCER, "env", KB, PY, f"{RD}/mktopic.py", topic)
    run(SERVER, "env", KB, f"QDB_CPUS={CPUS}", f"CONNECT_TASKS={tasks}", f"CONNECT_BATCH={batch}",
        f"CONNECT_CONF_EXTRA={conf_extra}",
        "bash", f"{RD}/aws/start_connect.sh", name, ",".join(topics or [name]),
        *(["--per-topic"] if per_topic else []))
    _connect.append(name)
    for _ in range(60):
        out = run(SERVER, "grep", "-c", "finished initialization and start",
                  f"/data/connect/{name}/connect.log", check=False, quiet=True).strip()
        if out and int(out) >= tasks:
            # The broker holds the first consumer-group assignment for its initial rebalance
            # delay (3 s by default) while more members join; start producing after that.
            log(f"{tasks} sink task(s) running; waiting {SETTLE_S} s for the consumer group")
            time.sleep(SETTLE_S)
            return
        time.sleep(2)
    raise SystemExit("connector tasks did not start")


def kafka_stop(name):
    run(SERVER, "bash", f"{RD}/aws/stop_connect.sh", name)
    _connect.remove(name)
    errors = run(SERVER, "grep", "-c", " ERROR ", f"/data/connect/{name}/connect.log",
                 check=False, quiet=True).strip()
    result(connector=name, error_lines=errors or 0)


def kafka_lag(name):
    out = run(KAFKA, "/opt/kafka/bin/kafka-consumer-groups.sh", "--bootstrap-server",
              "localhost:9092", "--describe", "--group", f"connect-questdb-{name}", quiet=True)
    lag = sum(int(line.split()[5]) for line in out.splitlines()[1:] if line.strip()
              and line.split()[5].isdigit())
    result(consumer_lag=lag)


def kafka_produce(topic, seconds, dup, producers=8, rate=50000, delay=0, offset=0, extra=()):
    return start(PRODUCER, "env", KB, f"RATE={rate}", f"PRODUCERS={producers}",
                 f"START_DELAY={delay}", f"SEED_OFFSET={offset}", "bash",
                 f"{RD}/aws/kafka_produce.sh", topic, str(seconds), str(dup), *extra)


def kafka_latency(table, t0, seconds, *extra):
    """cover_latency.py over the run after the warm-up, up to 2 s before the producers stop."""
    local("cover_latency.py", table, *extra, "--from", iso(t0 + WARMUP_S),
          "--to", iso(t0 + seconds - 2))


def originals(out):
    return sum(int(n) for n in re.findall(r"originals=(\d+)", out))


def qtime(t, delay):
    return start(PRODUCER, "env", QDB, PY, f"{RD}/qtime.py", t, "now()", "5", "30", str(delay))


def poller(kind, target, seconds, delay):
    return start(PRODUCER, "env", QDB, PY, f"{RD}/poll.py", kind, target, str(seconds), str(delay))


# ---- scenarios ----------------------------------------------------------------------------

def direct_base(t, flags, latency_tables):
    mktable(t, *flags)
    s = sampler(t, 312)
    finish(producer(t, t, 300))
    finish(s)
    m = producer_meta(t)
    rows(t, m["originals"])
    for lt in latency_tables:
        local("latency.py", lt, "1", str(m["last_event_id"]))
    if "--clean-mv" in flags:
        view_rows(t)
    cpu(t, [("run", m["started"] + 2, m["ended"])])


def direct_dedup(t):
    direct_base(t, [], [t])


def direct_passthrough(t):
    direct_base(t, ["--clean-mv"], [t, f"{t}_clean_mv"])


def direct_100k(t):
    mktable(t, "--clean-mv", "--clean-view")
    s = sampler(t, 312)
    a = producer(f"{t}_a", t, 300, "--seed", "42")
    b = producer(f"{t}_b", t, 300, "--seed", "43", "--id-base", "1000000000000", "--delay", "3")
    finish(a)
    finish(b)
    finish(s)
    ma, mb = producer_meta(f"{t}_a"), producer_meta(f"{t}_b")
    rows(t, ma["originals"] + mb["originals"])
    wa, wb = wal_writers(t)[:2]
    local("latency.py", t, "1", str(ma["last_event_id"]), "--wal-id", str(wa))
    local("latency.py", t, str(mb["first_event_id"]), str(mb["last_event_id"]), "--wal-id", str(wb))
    local("view_latency.py", t, f"{t}_clean_mv")
    view_rows(t)
    cpu(t, [("run", ma["started"] + 2, ma["ended"])])


def direct_100k_plain(t):
    """Stateless as the article ran it and as the post recommends: 100k evt/s, no duplicates,
    base table + plain view only (no materialized view). Nothing reads for the first 190 s
    (CPU window); the plain view is polled once a second from 200 s."""
    mktable(t, "--clean-view")
    s = sampler(t, 312)
    q = poller("view", t, 90, 200)
    a = producer(f"{t}_a", t, 300, "--seed", "42", "--dup-rate", "0")
    b = producer(f"{t}_b", t, 300, "--seed", "43", "--id-base", "1000000000000", "--delay", "3",
                 "--dup-rate", "0")
    finish(a)
    finish(b)
    finish(q)
    finish(s)
    ma, mb = producer_meta(f"{t}_a"), producer_meta(f"{t}_b")
    rows(t, ma["originals"] + mb["originals"])
    wa, wb = wal_writers(t)[:2]
    local("latency.py", t, "1", str(ma["last_event_id"]), "--wal-id", str(wa))
    local("latency.py", t, str(mb["first_event_id"]), str(mb["last_event_id"]), "--wal-id", str(wb))
    cpu(t, [("run", ma["started"] + 2, ma["ended"]), ("no_reader", ma["started"] + 10, ma["started"] + 190)])


def direct_scale(t):
    mktable(t, "--s1-mv")
    s = sampler(t, 312)
    a = producer(f"{t}_a", t, 300, "--seed", "42")
    b = producer(f"{t}_b", t, 100, "--seed", "43", "--id-base", "1000000000000", "--delay", "100")
    finish(a)
    finish(b)
    finish(s)
    ma, mb = producer_meta(f"{t}_a"), producer_meta(f"{t}_b")
    rows(t, ma["originals"] + mb["originals"])
    wa = wal_writers(t)[0]
    local("latency.py", t, "1", str(ma["last_event_id"]), "--wal-id", str(wa))
    p2_lo = int((mb["started"] - ma["started"]) * 50000) + 100000
    p2_hi = int((mb["ended"] - 3 - ma["started"]) * 50000)
    local("latency.py", t, str(p2_lo), str(p2_hi), "--wal-id", str(wa))
    phases = [("50k", ma["started"] + 2, mb["started"]), ("100k", mb["started"] + 2, mb["ended"]),
              ("50k_again", mb["ended"] + 2, ma["ended"])]
    for label, t0, t1 in phases:
        log(f"phase {label}")
        local("view_latency.py", t, f"{t}_s1_10s", "--from", iso(t0), "--to", iso(t1))
    cpu(t, phases)


def direct_s5(t, rate=27000, poll=False, batch=2000, seconds=300, consumers=(), tier=()):
    s = sampler(t, seconds + 25)
    q = poller("s5", t, seconds - 100, 60) if poll else None
    # The consumers poll from 60 s to seconds - 40 s, then wait 60 s (past the producer's end)
    # before their completeness query, so it does not compete with ingestion. tier: extra
    # s5consumer.py arguments (fast window ms, sweep every N polls) for two-tier polling.
    c = [start(PRODUCER, "env", QDB, PY, f"{RD}/s5consumer.py", t, str(ms), str(seconds - 100), "60",
               "1500", "60", *tier) for ms in consumers]
    run(PRODUCER, "env", QDB, PY, f"{RD}/s5producer.py", t, "--rate", str(rate), "--seconds",
        str(seconds), "--batch", str(batch))
    for p in c:
        finish(p)
    if q:
        finish(q)
    finish(s)
    m = producer_meta(t)
    mid = sql(f"SELECT dateadd('s', {seconds - 100}, min(ts)) FROM {t}_payments")["dataset"][0][0]
    local("s5check.py", t, mid)
    local("latency.py", f"{t}_payments", "1", str(m["orders"]), "--id-col", "payment_id")
    cpu(t, [("run", m["started"] + 2, m["ended"])])


BACKOFF = "reconnect_max_backoff_millis=500;"   # client default: 100 ms doubling up to 5000 ms


def direct_recovery(t, down=5, conf_extra=""):
    mktable(t, "--clean-mv")
    run(PRODUCER, "mkdir", "-p", SF)
    table_count()
    c = chaos(down)
    p = producer(t, t, 150, "--sf-dir", SF, *(["--conf-extra", conf_extra] if conf_extra else []),
                 quiet=False)
    finish(p)
    out = finish(c)
    kill = kill_epoch(out)
    startup_times(out)
    m = producer_meta(t)
    rows(t, m["originals"])
    view_rows(t)
    view_status(t)
    recovery_profile(t, kill)
    fetch_qdb_log(t)


def direct_recovery0(t):
    """direct_recovery with QuestDB restarted at once after the kill (no 5 s wait)."""
    direct_recovery(t, down=0)


def direct_recovery_b500(t):
    """direct_recovery with the client's reconnect backoff capped at 500 ms."""
    direct_recovery(t, conf_extra=BACKOFF)


def direct_recovery0_b500(t):
    direct_recovery(t, down=0, conf_extra=BACKOFF)


def direct_soak(t):
    mktable(t, "--clean-mv", "--clean-view", "--s1-mv")
    s = sampler(t, 1815)
    q = qtime(t, 1500)
    finish(producer(t, t, 1800))
    finish(q)
    finish(s)
    m = producer_meta(t)
    rows(t, m["originals"])
    local("latency.py", t, "1", str(m["last_event_id"]), "--windows", "6")
    local("latency.py", f"{t}_clean_mv", "1", str(m["last_event_id"]), "--windows", "6")
    local("view_latency.py", t, f"{t}_s1_10s")
    view_rows(t)
    view_status(t)
    t0 = m["started"]
    cpu(t, [("run", t0 + 2, m["ended"])] +
        [(f"{i * 5}-{i * 5 + 5}min", t0 + i * 300, t0 + i * 300 + 300) for i in range(6)])


def kafka_dedup(t):
    mktable(t)
    kafka_prep(t, 1)
    s = sampler(t, 310, connect=True)
    t0 = time.time()
    out = finish(kafka_produce(t, 300, 0.1))
    finish(s)
    kafka_lag(t)
    kafka_stop(t)
    rows(t, originals(out))
    kafka_latency(t, t0, 300)
    cpu(t, [("run", t0 + 2, t0 + 300)], connect=True)


def kafka_100k(t):
    mktable(t, "--clean-mv", "--clean-view")
    kafka_prep(t, 2)
    s = sampler(t, 310, connect=True)
    t0 = time.time()
    out = finish(kafka_produce(t, 300, 0, producers=16, rate=100000))
    finish(s)
    kafka_lag(t)
    kafka_stop(t)
    rows(t, originals(out))
    kafka_latency(t, t0, 300)
    kafka_latency(f"{t}_clean_mv", t0, 300)
    view_rows(t)
    cpu(t, [("run", t0 + 2, t0 + 300)], connect=True)


def kafka_100k_plain(t):
    """kafka_100k with the plain view only (no materialized view); no duplicates, as before."""
    mktable(t, "--clean-view")
    kafka_prep(t, 2)
    s = sampler(t, 310, connect=True)
    q = poller("view", t, 90, 200)
    t0 = time.time()
    out = finish(kafka_produce(t, 300, 0, producers=16, rate=100000))
    finish(q)
    finish(s)
    kafka_lag(t)
    kafka_stop(t)
    rows(t, originals(out))
    kafka_latency(t, t0, 300)
    cpu(t, [("run", t0 + 2, t0 + 300), ("no_reader", t0 + 10, t0 + 190)], connect=True)


def kafka_scale(t):
    mktable(t, "--s1-mv")
    kafka_prep(t, 2)
    s = sampler(t, 312, connect=True)
    t0 = time.time()
    a = kafka_produce(t, 300, 0)
    b = kafka_produce(t, 100, 0, delay=100, offset=8)
    out = finish(a) + "\n" + finish(b)
    finish(s)
    kafka_lag(t)
    kafka_stop(t)
    rows(t, originals(out))
    phases = [("50k", t0 + WARMUP_S, t0 + 100), ("100k", t0 + 105, t0 + 200),
              ("50k_again", t0 + 205, t0 + 298)]
    for label, p0, p1 in phases:
        log(f"phase {label}")
        local("cover_latency.py", t, "--from", iso(p0), "--to", iso(p1))
        local("view_latency.py", t, f"{t}_s1_10s", "--from", iso(p0), "--to", iso(p1))
    cpu(t, phases, connect=True)


def kafka_s5(t, rate=27200, poll=False, batch=2000, consumers=()):
    orders_t, payments_t = f"{t}_orders", f"{t}_payments"
    run(PRODUCER, "env", QDB, PY, f"{RD}/s5producer.py", t, "--create-only")
    kafka_prep(t, 1, topics=[orders_t, payments_t], per_topic=True, batch=batch)
    s = sampler(t, 330, connect=True)
    q = poller("s5", t, 200, 60) if poll else None
    c = [start(PRODUCER, "env", QDB, PY, f"{RD}/s5consumer.py", t, str(ms), "200", "60", "1500", "60")
         for ms in consumers]
    t0 = time.time()
    out = finish(kafka_produce(orders_t, 300, 0, rate=rate,
                               extra=("--payments-topic", payments_t)))
    for p in c:
        finish(p)
    if q:
        finish(q)
    finish(s)
    kafka_stop(t)
    pays = sum(int(n) for n in re.findall(r" payments=(\d+)", out))
    late = sum(int(n) for n in re.findall(r"late_payments=(\d+)", out))
    meta = {"run_id": t, "orders_table": orders_t, "payments_table": payments_t,
            "orders": originals(out), "payments": pays, "expected_pairs": pays - late,
            "late_payments": late}
    with open(os.path.join(RUNS_DIR, f"{t}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    mid = sql(f"SELECT dateadd('s', 200, min(ts)) FROM {payments_t}")["dataset"][0][0]
    local("s5check.py", t, mid)
    kafka_latency(payments_t, t0, 300, "--id-col", "payment_id")
    cpu(t, [("run", t0 + 2, t0 + 300)], connect=True)


def kafka_recovery(t, down=5, conf_extra=""):
    mktable(t, "--clean-mv")
    kafka_prep(t, 1, conf_extra=conf_extra)
    table_count()
    c = chaos(down)
    p = kafka_produce(t, 150, 0)
    out = finish(p)
    cout = finish(c)
    kill = kill_epoch(cout)
    startup_times(cout)
    time.sleep(10)   # let the connector drain what Kafka buffered during the outage
    kafka_lag(t)
    kafka_stop(t)
    rows(t, originals(out))
    view_rows(t)
    view_status(t)
    recovery_profile(t, kill)
    fetch_qdb_log(t)


def kafka_recovery0(t):
    """kafka_recovery with QuestDB restarted at once after the kill (no 5 s wait)."""
    kafka_recovery(t, down=0)


def kafka_recovery_b500(t):
    """kafka_recovery with the connector's reconnect backoff capped at 500 ms."""
    kafka_recovery(t, conf_extra=BACKOFF)


def kafka_recovery0_b500(t):
    kafka_recovery(t, down=0, conf_extra=BACKOFF)


def kafka_soak(t):
    mktable(t, "--clean-mv", "--clean-view", "--s1-mv")
    kafka_prep(t, 1)
    s = sampler(t, 1815, connect=True)
    q = qtime(t, 1500)
    t0 = time.time()
    out = finish(kafka_produce(t, 1800, 0.1))
    finish(q)
    finish(s)
    kafka_lag(t)
    kafka_stop(t)
    rows(t, originals(out))
    view_rows(t)
    view_status(t)
    cpu(t, [("run", t0 + 2, t0 + 1800)] +
        [(f"{i * 5}-{i * 5 + 5}min", t0 + i * 300, t0 + i * 300 + 300) for i in range(6)],
        connect=True)


def direct_s5poll(t):
    direct_s5(t, poll=True)


def direct_s5poll8k(t):
    direct_s5(t, rate=8000, poll=True)


def kafka_s5poll(t):
    kafka_s5(t, poll=True)


def kafka_s5poll8k(t):
    kafka_s5(t, rate=8000, poll=True)


# At 8k orders/s a 2,000-row batch takes ~135 ms to fill (orders + payments), against ~40 ms
# at 27k; 500 rows keeps the fill time about the same as the other runs.
def direct_s5poll8k_b500(t):
    direct_s5(t, rate=8000, poll=True, batch=500)


def kafka_s5poll8k_b500(t):
    kafka_s5(t, rate=8000, poll=True, batch=500)


def direct_s5soak(t):
    """30 minutes of S5 at 8k with both polls every second: does the ASOF poll stay flat?"""
    direct_s5(t, rate=8000, poll=True, batch=500, seconds=1800, consumers=(100, 1000))


def direct_s5base8k(t):
    """S5 at 8k, 500-row batches, nothing reading: the CPU baseline for the consumer runs."""
    direct_s5(t, rate=8000, batch=500)


def direct_s5c100_8k(t):
    """The same with only the 100 ms consumer, to measure its QuestDB CPU."""
    direct_s5(t, rate=8000, batch=500, consumers=(100,))


def direct_s5tier_8k(t):
    """The 100 ms consumer in two-tier mode: last 300 ms each poll, 1.5 s sweep every 1 s."""
    direct_s5(t, rate=8000, batch=500, consumers=(100,), tier=("300", "10", "full"))


def direct_s5tier_27k(t):
    direct_s5(t, rate=27000, batch=2000, consumers=(100,), tier=("300", "10", "full"))


def direct_s5base27k(t):
    """S5 at 27k, nothing reading: the same-session baseline for direct_s5c100_27k."""
    direct_s5(t, rate=27000, batch=2000)


def direct_s5c100_27k(t):
    """S5 at 27k with only the 100 ms consumer."""
    direct_s5(t, rate=27000, batch=2000, consumers=(100,))


def kafka_s5c100_8k(t):
    """Through Kafka at 8k (500-row connector batches) with only the 100 ms consumer."""
    kafka_s5(t, rate=8000, batch=500, consumers=(100,))


def kafka_s5c100_27k(t):
    """Through Kafka at 27k with only the 100 ms consumer."""
    kafka_s5(t, consumers=(100,))


def direct_s5soak27k(t):
    """The same at 27k orders/s (~50k events/s with payments, the other workloads' rate)."""
    direct_s5(t, rate=27000, poll=True, batch=2000, seconds=1800, consumers=(100, 1000))


def direct_poll(t):
    mktable(t, "--clean-mv", "--clean-view", "--s1-mv")
    s = sampler(t, 312)
    q = poller("events", t, 230, 60)
    finish(producer(t, t, 300))
    finish(q)
    finish(s)
    m = producer_meta(t)
    rows(t, m["originals"])
    local("latency.py", t, "1", str(m["last_event_id"]))
    local("view_latency.py", t, f"{t}_s1_10s")
    view_rows(t)
    cpu(t, [("run", m["started"] + 2, m["ended"])])


def kafka_poll(t):
    mktable(t, "--clean-mv", "--clean-view", "--s1-mv")
    kafka_prep(t, 1)
    s = sampler(t, 310, connect=True)
    q = poller("events", t, 230, 60)
    t0 = time.time()
    out = finish(kafka_produce(t, 300, 0.1))
    finish(q)
    finish(s)
    kafka_lag(t)
    kafka_stop(t)
    rows(t, originals(out))
    view_rows(t)
    cpu(t, [("run", t0 + 2, t0 + 300)], connect=True)


def direct_s1(t):
    """kafka_s1's direct counterpart: 50k evt/s, no dups, base + S1 view, 300 s."""
    mktable(t, "--s1-mv")
    s = sampler(t, 312)
    finish(producer(t, t, 300, "--dup-rate", "0"))
    finish(s)
    m = producer_meta(t)
    rows(t, m["originals"])
    local("latency.py", t, "1", str(m["last_event_id"]))
    local("view_latency.py", t, f"{t}_s1_10s", "--from", iso(m["started"] + WARMUP_S),
          "--to", iso(m["started"] + 298))
    cpu(t, [("run", m["started"] + 2, m["ended"])])


def direct_s1soak(t):
    """30 minutes of direct_s1 (no dups, base + S1 view), as the article's final30 S1 runs."""
    mktable(t, "--s1-mv")
    s = sampler(t, 1815)
    finish(producer(t, t, 1800, "--dup-rate", "0"))
    finish(s)
    m = producer_meta(t)
    rows(t, m["originals"])
    local("latency.py", t, "1", str(m["last_event_id"]), "--windows", "6")
    local("view_latency.py", t, f"{t}_s1_10s", "--from", iso(m["started"] + WARMUP_S),
          "--to", iso(m["started"] + 1798))
    for i in range(6):   # the view's bounds per 5 minutes, to show any drift
        local("view_latency.py", t, f"{t}_s1_10s", "--from", iso(m["started"] + max(WARMUP_S, i * 300)),
              "--to", iso(m["started"] + min(1798, i * 300 + 300)))
    t0 = m["started"]
    cpu(t, [("run", t0 + 2, m["ended"])] +
        [(f"{i * 5}-{i * 5 + 5}min", t0 + i * 300, t0 + i * 300 + 300) for i in range(6)])


def kafka_s1(t):
    mktable(t, "--s1-mv")
    kafka_prep(t, 1)
    s = sampler(t, 310, connect=True)
    t0 = time.time()
    out = finish(kafka_produce(t, 300, 0))
    finish(s)
    kafka_lag(t)
    kafka_stop(t)
    rows(t, originals(out))
    kafka_latency(t, t0, 300)
    local("view_latency.py", t, f"{t}_s1_10s", "--from", iso(t0 + WARMUP_S),
          "--to", iso(t0 + 298))
    cpu(t, [("run", t0 + 2, t0 + 300)], connect=True)


SCENARIOS = {f.__name__: f for f in (
    direct_dedup, direct_passthrough, direct_100k, direct_scale, direct_s5, direct_recovery,
    direct_soak, kafka_dedup, kafka_100k, kafka_scale, kafka_s5, kafka_recovery, kafka_soak,
    direct_s5poll, direct_s5poll8k, kafka_s5poll, kafka_s5poll8k, direct_poll, kafka_poll,
    kafka_s1, direct_s5poll8k_b500, kafka_s5poll8k_b500, direct_s5soak, direct_s5soak27k,
    direct_s1, direct_s5base8k, direct_s5c100_8k, direct_s5base27k, direct_s5c100_27k,
    kafka_s5c100_8k, kafka_s5c100_27k, direct_s1soak, direct_s5tier_8k, direct_s5tier_27k,
    direct_100k_plain, kafka_100k_plain, direct_recovery0, kafka_recovery0,
    direct_recovery_b500, direct_recovery0_b500, kafka_recovery_b500, kafka_recovery0_b500)}


def main():
    global _log
    scenarios = sys.argv[1].split(",") if len(sys.argv) == 3 else []
    if not scenarios or any(s not in SCENARIOS for s in scenarios):
        sys.exit(f"usage: matrix.py <scenario>[,<scenario>...] <rep>; "
                 f"scenarios: {', '.join(SCENARIOS)}")
    rep = sys.argv[2]
    os.makedirs(os.path.join(RUNS_DIR, "matrix"), exist_ok=True)
    for scenario in scenarios:   # in order; stops at the first failure
        t = f"m_{scenario}_{rep}"
        with open(os.path.join(RUNS_DIR, "matrix", f"{scenario}_{rep}.log"), "w") as _log:
            log(f"# {scenario} {rep} table {t} started {iso(time.time())}Z")
            try:
                SCENARIOS[scenario](t)
            finally:
                for name in list(_connect):
                    run(SERVER, "bash", f"{RD}/aws/stop_connect.sh", name, check=False)
            log(f"# done {iso(time.time())}Z")


if __name__ == "__main__":
    main()
