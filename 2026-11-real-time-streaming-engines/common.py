"""
Shared settings and a small QuestDB REST client for the scripts in this folder.

Settings come from the environment, with an optional .env file next to this file as a
fallback (see .env.example); a variable already set in the environment wins, so a remote
command can override any value.
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")


def load_env(path=os.path.join(HERE, ".env")):
    """Read KEY=value lines from path into os.environ, without overriding set variables."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


load_env()

# host:port of QuestDB; QWP ingestion and the REST API share port 9000 by default.
QDB_ADDR = os.environ.get("QDB_ADDR", "localhost:9000")
QDB_HTTP = os.environ.get("QDB_HTTP", f"http://{QDB_ADDR}")


def sql(query, timings=False):
    """Run one statement over REST and return the parsed JSON; exit with the error on failure."""
    params = {"query": query}
    if timings:
        params["timings"] = "true"
    url = f"{QDB_HTTP}/api/v1/sql/execute?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url) as r:
            res = json.load(r)
    except urllib.error.HTTPError as e:
        res = json.load(e)
    if "error" in res:
        sys.exit(f"query failed: {res['error']}\n{query}")
    return res


def table_id(name):
    return sql(f"SELECT id FROM tables() WHERE table_name = '{name}'")["dataset"][0][0]
