"""StatsBomb events -> per-match tables for the point-process models (see DESIGN.md).

Outputs (data/):
  matches.parquet   match_id, league, date, round, half (H1/H2), home/away team ids and names, scores
  windows.parquet   one row per team x period: match_id, period, team_id, opp_id, home, T (period length, s)
  events.parquet    one row per event: match_id, period, t (s), team_id, kind
                    kind in {chance, open_chance, box_entry, shot, goal}
  dead.parquet      dead time after goals: match_id, period, start, end (s)
  onpitch.parquet   player intervals on the pitch: match_id, team_id, player_id, period, start, end (s, within period)

Teams are always identified by id (names differ between StatsBomb files).
Usage: python src/build_events.py <statsbomb-open-data>/data
"""
import sys
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = {2: "Premier League", 7: "Ligue 1", 11: "La Liga", 12: "Serie A"}
SEASON = 27
BOX = dict(x=102.0, y_lo=18.0, y_hi=62.0)


def ts(s):
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def in_box(loc):
    return loc is not None and loc[0] >= BOX["x"] and BOX["y_lo"] <= loc[1] <= BOX["y_hi"]


def build_matches(raw):
    rows = []
    for cid, league in LEAGUES.items():
        ms = orjson.loads((raw / "matches" / str(cid) / f"{SEASON}.json").read_bytes())
        cut = min(m["match_date"] for m in ms if m["match_week"] == 20)
        for m in ms:
            rows.append(dict(match_id=m["match_id"], league=league, date=m["match_date"], round=m["match_week"],
                             half="H1" if m["match_date"] < cut else "H2",
                             home_id=m["home_team"]["home_team_id"], away_id=m["away_team"]["away_team_id"],
                             home_score=m["home_score"], away_score=m["away_score"]))
    return pd.DataFrame(rows)


def one_match(raw, m):
    ev = orjson.loads((raw / "events" / f"{m.match_id}.json").read_bytes())
    ev.sort(key=lambda e: e["index"])
    ev = [e for e in ev if e["period"] in (1, 2)]
    ids = {m.home_id, m.away_id}
    name = {e["team"]["id"]: e["team"]["name"] for e in ev if "team" in e}
    assert set(name) == ids, m.match_id

    def other(tid):
        return (ids - {tid}).pop()

    # period lengths: the Half End events (fallback: last event)
    T = {}
    for p in (1, 2):
        ends = [ts(e["timestamp"]) for e in ev if e["period"] == p and e["type"]["name"] == "Half End"]
        last = max(ts(e["timestamp"]) for e in ev if e["period"] == p)
        T[p] = max(ends) if ends else last
    # play pattern of each possession (taken from its first event)
    pattern = {}
    for e in ev:
        pattern.setdefault((e["period"], e["possession"]), e["play_pattern"]["name"])

    events, goals, kickoffs = [], [], []
    first_shot, first_box, open_flag = {}, {}, {}
    for e in ev:
        t, p, typ = ts(e["timestamp"]), e["period"], e["type"]["name"]
        tid = e.get("team", {}).get("id")
        key = (p, e["possession"], tid)
        if typ == "Shot":
            s = e["shot"]
            events.append(dict(period=p, t=t, team_id=tid, kind="shot"))
            if key not in first_shot:
                first_shot[key] = t
                open_flag[key] = (s.get("type", {}).get("name") == "Open Play"
                                  and pattern[(p, e["possession"])] not in ("From Corner", "From Free Kick"))
            if s["outcome"]["name"] == "Goal":
                goals.append(dict(period=p, t=t, team_id=tid))
            if in_box(e.get("location")):
                first_box.setdefault(key, t)
        elif typ == "Own Goal Against":
            goals.append(dict(period=p, t=t, team_id=other(tid)))
        elif typ == "Pass":
            ps = e["pass"]
            if ps.get("type", {}).get("name") == "Kick Off":
                kickoffs.append(dict(period=p, t=t))
            if "outcome" not in ps and in_box(ps.get("end_location")):
                first_box.setdefault(key, t)
        elif typ == "Carry":
            if in_box(e["carry"].get("end_location")):
                first_box.setdefault(key, t)
    for (p, _, tid), t in first_shot.items():
        events.append(dict(period=p, t=t, team_id=tid, kind="chance"))
        if open_flag[(p, _, tid)]:
            events.append(dict(period=p, t=t, team_id=tid, kind="open_chance"))
    for (p, _, tid), t in first_box.items():
        events.append(dict(period=p, t=t, team_id=tid, kind="box_entry"))
    for g in goals:
        events.append(dict(g, kind="goal"))
    # dead time after each goal: until the next kick-off pass in the same period
    dead = []
    for g in goals:
        nxt = [k["t"] for k in kickoffs if k["period"] == g["period"] and k["t"] > g["t"]]
        dead.append(dict(period=g["period"], start=g["t"], end=min(nxt) if nxt else T[g["period"]]))
    # players on the pitch (within-period clock): starting XI, substitutions, dismissals
    on, off, team_of = {}, {}, {}
    for e in ev:
        typ = e["type"]["name"]
        pt = (e["period"], ts(e["timestamp"]))
        if typ == "Starting XI":
            for pl in e["tactics"]["lineup"]:
                on[pl["player"]["id"]], team_of[pl["player"]["id"]] = (1, 0.0), e["team"]["id"]
        elif typ == "Substitution":
            off[e["player"]["id"]] = pt
            rid = e["substitution"]["replacement"]["id"]
            on[rid], team_of[rid] = pt, e["team"]["id"]
        else:
            card = ((e.get("foul_committed") or {}).get("card") or {}).get("name") or \
                   ((e.get("bad_behaviour") or {}).get("card") or {}).get("name")
            if card in ("Red Card", "Second Yellow") and "player" in e and e["player"]["id"] in on:
                off.setdefault(e["player"]["id"], pt)
                off[e["player"]["id"]] = min(off[e["player"]["id"]], pt)
    onpitch = []
    for pid, (p0, t0) in on.items():
        p1, t1 = off.get(pid, (2, T[2]))
        for p in (1, 2):
            a = t0 if p == p0 else (0.0 if p > p0 else None)
            b = t1 if p == p1 else (T[p] if p < p1 else None)
            if a is None or b is None or b <= a:
                continue
            onpitch.append(dict(team_id=team_of[pid], player_id=pid, period=p, start=a, end=min(b, T[p])))
    windows = [dict(period=p, team_id=tid, opp_id=other(tid), home=int(tid == m.home_id), T=T[p])
               for p in (1, 2) for tid in (m.home_id, m.away_id)]
    tag = dict(match_id=m.match_id)
    return ([dict(tag, **w) for w in windows], [dict(tag, **x) for x in events], [dict(tag, **d) for d in dead],
            [dict(tag, **o) for o in onpitch], name)


def main(raw):
    raw = Path(raw)
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    M = build_matches(raw)
    W, E, D, OP, names = [], [], [], [], {}
    for m in M.itertuples():
        w, e, d, o, nm = one_match(raw, m)
        W += w
        E += e
        D += d
        OP += o
        names.update(nm)
    M["home_name"], M["away_name"] = M.home_id.map(names), M.away_id.map(names)
    W, E, D, OP = pd.DataFrame(W), pd.DataFrame(E), pd.DataFrame(D), pd.DataFrame(OP)
    # integrity: goals per team equal the official score in every match
    g = E[E.kind == "goal"].groupby(["match_id", "team_id"]).size()
    bad = sum(int(g.get((r.match_id, r.home_id), 0)) != r.home_score or
              int(g.get((r.match_id, r.away_id), 0)) != r.away_score for r in M.itertuples())
    assert bad == 0, f"{bad} matches with goals != score"
    for name_, df in (("matches", M), ("windows", W), ("events", E), ("dead", D), ("onpitch", OP)):
        df.to_parquet(out / f"{name_}.parquet")
    print(M.groupby(["league", "half"]).size().unstack())
    print(E.kind.value_counts().to_dict())
    print("dead-time intervals", len(D), "median length (s)", round(float(np.median(D.end - D.start)), 1))


if __name__ == "__main__":
    main(sys.argv[1])
