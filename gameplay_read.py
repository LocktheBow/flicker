#!/usr/bin/env python3
"""
g-lab gameplay_read.py  --  the READ channel for during-game doses.

Runs on the machine you ACTUALLY play Marvel Rivals on (browsers can't see
keystrokes sent to a fullscreen game). It captures the TIMING of your key
presses and mouse clicks, PLUS your mouse-movement path (aim tempo), during
the entrainment window and turns them into state/tempo metrics (APM,
consistency, aim path length/speed/corrections) that pair with the
entrainment (write) and the pre/post Gf test (outcome) from dose.html.

PRIVACY BY DESIGN: it records only *timestamps* and whether an event was a
key, click, or move. It NEVER records which key you pressed or any text.
Mouse movement is aggregated into path length/speed/correction-count
on the fly -- the raw (x,y) trace itself is never written to disk, only the
aggregate numbers. Output is a single local JSON file. No network, ever.

INPUT LAG: pynput's listener callback runs synchronously in the OS event-
delivery path, so heavy per-event work (raw mouse-move handling especially,
which can fire hundreds of times/sec while aiming) can measurably delay
input -- noticeable on cloud-streamed games (GeForce NOW etc.) where the
latency budget is already tight. Mouse-move processing is throttled to
~66 Hz (MOVE_THROTTLE_S) so the expensive math only runs at that rate; raw
events still pass through pynput, but the callback does near-nothing for
the ones it skips.

HONEST SCIENCE: keystroke dynamics are a STATE / engagement / motor-tempo read,
NOT a measure of fluid intelligence. Their best use here is (a) a covariate and
(b) a confound check -- proving you played with similar intensity on real vs
sham sessions, so a Gf difference isn't just "I tried harder that day."

SHARED CLOCK: every snapshot row carries a unix-epoch timestamp (t0_epoch +
relative t), the same clock the flicker page stamps its HR/depth reads with --
so the input timeline overlays directly on the entrainment/arousal timeline.
Snapshots are written to the output JSON every --snap seconds (default 15),
not just at the end, so a crash or force-quit loses at most one window.

SETUP:
    pip install pynput
    # macOS: System Settings > Privacy & Security > Accessibility > allow your
    #        terminal app. (Antivirus may flag input hooks -- this is expected
    #        for any input monitor; the code is short, read it.)

AIM-ADAPTIVE (live): as well as logging, this serves a rolling motor-tempo
"arousal" at http://127.0.0.1:8788/aim -- how fast/busy your aiming and inputs
are right now versus your own session baseline (0.5 = at baseline, 1.0 = ~2x,
0 = idle). The flicker page fetches it and blends it with the webcam HR arousal
to set coupling depth, so behavioural engagement drives the dose too, not just
heart rate. This is a motor-engagement proxy, NOT physiological arousal -- still
timing/aggregate only, no key identity. Disable the server with --no-serve.

RUN (start this the moment you start the entrainment + game):
    python3 gameplay_read.py --minutes 30 --phase intervention
"""

import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

AIM_STATE = {"arousal": 0.5, "quality": 0.0, "apm": None, "aim_speed": None,
             "corr_rate": None, "epoch": 0.0, "src": "aim"}
AIM_LOCK = threading.Lock()


def set_aim(**kw):
    with AIM_LOCK:
        AIM_STATE.update(kw)


def rates_from_window(d_actions, d_path, d_corr, window_s):
    """Pure: per-minute / per-second rates over a rolling window. Testable."""
    w = window_s if window_s > 0 else 1.0
    return {"apm": d_actions / w * 60.0, "aim_speed": d_path / w,
            "corr_rate": d_corr / w * 60.0}


def aim_arousal(cur, base):
    """Pure: 0-1 motor arousal = current tempo vs baseline, averaged over the
    metrics that have a baseline. cur==base -> 0.5, cur==2*base -> 1.0,
    cur==0 -> 0.0. Returns 0.5 (neutral) until a baseline exists."""
    parts = []
    for k in ("apm", "aim_speed", "corr_rate"):
        b = base.get(k)
        if b and b > 1e-6:
            parts.append(max(0.0, min(1.0, (cur[k] / b - 1.0) * 0.5 + 0.5)))
    return round(sum(parts) / len(parts), 3) if parts else 0.5


class AimHandler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        if self.path.split("?")[0] not in ("/aim", "/"):
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        with AIM_LOCK:
            body = json.dumps(AIM_STATE).encode()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def window_metrics(key_ts, click_ts, mv_now, prev, t_now):
    """Pure logic (testable without real events): timing/tempo metrics for the
    window (prev['t'], t_now]. prev holds cumulative counts at the previous
    snapshot. Counts and aggregates only -- no key identity anywhere.
    Returns (row, new_prev)."""
    k, c = len(key_ts), len(click_ts)
    wk, wc = k - prev["keys"], c - prev["clicks"]
    ev = sorted(key_ts[prev["keys"]:k] + click_ts[prev["clicks"]:c])
    gaps = [(ev[i] - ev[i - 1]) * 1000.0 for i in range(1, len(ev))]
    dt = t_now - prev["t"]
    row = {
        "t": round(t_now, 2),
        "keys": wk,
        "clicks": wc,
        "apm": round((wk + wc) / (dt / 60.0), 1) if dt > 0 else None,
        "gap_med_ms": round(statistics.median(gaps), 1) if gaps else None,
        "path_px": round(mv_now["path_px"] - prev["path_px"], 1),
        "corrections": mv_now["corrections"] - prev["corrections"],
    }
    new_prev = {"t": t_now, "keys": k, "clicks": c,
                "path_px": mv_now["path_px"], "corrections": mv_now["corrections"]}
    return row, new_prev


def write_json(rec, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="g-lab gameplay read channel (timing only, no text).")
    ap.add_argument("--minutes", type=float, default=30.0, help="match the dose length")
    ap.add_argument("--phase", default="intervention", help="baseline|intervention|washout|retention")
    ap.add_argument("--condition", default="blind", help="leave as 'blind' to stay un-blinded")
    ap.add_argument("--out", default=None, help="output json path")
    ap.add_argument("--snap", type=float, default=15.0, help="seconds between timeline snapshots")
    ap.add_argument("--aim-port", type=int, default=8788, help="port for the live aim-arousal server")
    ap.add_argument("--no-serve", action="store_true", help="log only; don't serve live aim arousal")
    args = ap.parse_args()

    try:
        from pynput import keyboard, mouse
    except ImportError:
        sys.exit("Missing dependency. Run:  pip install pynput")

    key_ts = []    # timestamps of key presses (NO key identity stored)
    click_ts = []  # timestamps of mouse clicks

    # mouse-movement (aim) aggregator -- no raw path is kept, only running stats
    MOVE_MIN_DIST = 2.0       # px; ignores sub-pixel jitter
    MOVE_ANGLE_DEG = 100.0    # direction change beyond this = a "correction"
    MOVE_THROTTLE_S = 0.015   # ~66 Hz cap -- keeps the callback cheap to avoid input lag
    mv = {"samples": 0, "path_px": 0.0, "corrections": 0,
          "last_x": None, "last_y": None, "last_angle": None, "last_t": None}

    t0 = time.perf_counter()
    t0_epoch = time.time()  # same instant as t0: epoch = t0_epoch + relative t

    def on_press(_key):
        key_ts.append(time.perf_counter() - t0)

    def on_click(_x, _y, _button, pressed):
        if pressed:
            click_ts.append(time.perf_counter() - t0)

    def process_move(mv, x, y, now, min_dist=MOVE_MIN_DIST, angle_deg=MOVE_ANGLE_DEG, throttle_s=MOVE_THROTTLE_S):
        """Pure logic (testable without real time/events). Returns True if it
        did real work this call, False if it bailed early (throttled/init)."""
        if mv["last_t"] is not None and (now - mv["last_t"]) < throttle_s:
            return False  # throttled: skip all further work, this is the lag fix
        if mv["last_x"] is None:
            mv["last_x"], mv["last_y"], mv["last_t"] = x, y, now
            return False
        dx, dy = x - mv["last_x"], y - mv["last_y"]
        dist = math.hypot(dx, dy)
        mv["last_t"] = now
        if dist < min_dist:
            return False
        mv["samples"] += 1
        mv["path_px"] += dist
        angle = math.degrees(math.atan2(dy, dx))
        if mv["last_angle"] is not None:
            diff = abs(angle - mv["last_angle"]) % 360
            if diff > 180:
                diff = 360 - diff
            if diff >= angle_deg:
                mv["corrections"] += 1
        mv["last_angle"] = angle
        mv["last_x"], mv["last_y"] = x, y
        return True

    def on_move(x, y):
        process_move(mv, x, y, time.perf_counter() - t0)

    kl = keyboard.Listener(on_press=on_press)
    ml = mouse.Listener(on_click=on_click, on_move=on_move)
    kl.start()
    ml.start()

    started = datetime.now(timezone.utc).isoformat()
    dur = args.minutes * 60.0
    out = args.out or f"gameplay_{started.replace(':', '-').replace('.', '-')}.json"
    snaps = []
    prev = {"t": 0.0, "keys": 0, "clicks": 0, "path_px": 0.0, "corrections": 0}

    def snapshot(t_now):
        nonlocal prev
        mv_now = {"path_px": mv["path_px"], "corrections": mv["corrections"]}
        row, prev = window_metrics(key_ts, click_ts, mv_now, prev, t_now)
        row["epoch"] = round(t0_epoch + t_now, 2)
        snaps.append(row)
        write_json({"schema": "g-lab/1", "type": "gameplay_read", "ts": started,
                    "phase": args.phase, "condition": args.condition,
                    "minutes_planned": args.minutes, "privacy": "timestamps_only_no_text",
                    "t0_epoch": round(t0_epoch, 3), "timeseries": snaps,
                    "metrics": None}, out)  # metrics filled in at the end

    # live aim-arousal server (blended into the flicker dose alongside HR)
    if not args.no_serve:
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", args.aim_port), AimHandler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            print(f"[g-lab] aim arousal live at http://127.0.0.1:{args.aim_port}/aim")
        except OSError as e:
            print(f"[g-lab] aim server not started ({e}) -- logging still works.")

    WINDOW_S = 4          # rolling window for the live tempo read
    BASE_MIN = 15         # ~15 s of ticks before a baseline is trusted
    win = deque(maxlen=WINDOW_S)      # per-tick (d_actions, d_path, d_corr)
    hist = []                          # cur dicts, for the rolling baseline
    last = {"actions": 0, "path": 0.0, "corr": 0}

    def update_aim(now_rel):
        a_now, p_now, c_now = len(key_ts) + len(click_ts), mv["path_px"], mv["corrections"]
        win.append((a_now - last["actions"], p_now - last["path"], c_now - last["corr"]))
        last.update(actions=a_now, path=p_now, corr=c_now)
        cur = rates_from_window(sum(x[0] for x in win), sum(x[1] for x in win),
                                sum(x[2] for x in win), len(win))
        hist.append(cur)
        del hist[:-240]
        if len(hist) >= BASE_MIN:
            base = {k: statistics.median([h[k] for h in hist]) for k in cur}
            ar, q = aim_arousal(cur, base), 0.8
        else:
            ar, q = 0.5, 0.2  # warming up: neutral, below the page's gate
        set_aim(arousal=ar, quality=q, apm=round(cur["apm"], 1),
                aim_speed=round(cur["aim_speed"], 1), corr_rate=round(cur["corr_rate"], 1),
                epoch=round(t0_epoch + now_rel, 2))

    print(f"[g-lab] reading key/click TIMING only (no text) for {args.minutes:g} min.")
    print(f"[g-lab] snapshots every {args.snap:g}s -> {out}")
    print("[g-lab] go play. Ctrl-C to stop early.\n")
    try:
        while time.perf_counter() - t0 < dur:
            now_rel = time.perf_counter() - t0
            remaining = dur - now_rel
            update_aim(now_rel)
            if now_rel - prev["t"] >= args.snap:
                snapshot(now_rel)
            print(f"\r  {int(remaining)//60:02d}:{int(remaining)%60:02d} left  ·  "
                  f"actions: {len(key_ts)+len(click_ts)}  ·  aim {AIM_STATE['arousal']:.2f}   ",
                  end="", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[g-lab] stopped early.")
    finally:
        kl.stop()
        ml.stop()
        tail = time.perf_counter() - t0
        if tail - prev["t"] > 1.0:
            snapshot(tail)

    actual = time.perf_counter() - t0
    actions = sorted(key_ts + click_ts)
    n = len(actions)

    inter = [ (actions[i] - actions[i-1]) * 1000.0 for i in range(1, n) ]  # ms between actions
    minutes = max(1, math.ceil(actual / 60.0))
    per_minute = [0] * minutes
    for a in actions:
        idx = min(minutes - 1, int(a // 60))
        per_minute[idx] += 1

    metrics = {
        "actions_total": n,
        "keys": len(key_ts),
        "clicks": len(click_ts),
        "duration_s": round(actual, 1),
        "apm": round(n / (actual / 60.0), 1) if actual > 0 else None,
        "median_inter_action_ms": round(statistics.median(inter), 1) if inter else None,
        "inter_action_cv": round(statistics.pstdev(inter) / statistics.mean(inter), 3)
                           if len(inter) > 1 and statistics.mean(inter) > 0 else None,  # consistency
        "per_minute_actions": per_minute,
        "mouse_aim": {
            "samples": mv["samples"],
            "path_px": round(mv["path_px"], 1),
            "avg_speed_px_s": round(mv["path_px"] / actual, 1) if actual > 0 else None,
            "corrections": mv["corrections"],
            "corrections_per_min": round(mv["corrections"] / (actual / 60.0), 2) if actual > 0 else None,
        },
    }

    rec = {
        "schema": "g-lab/1",
        "type": "gameplay_read",
        "ts": started,
        "phase": args.phase,
        "condition": args.condition,
        "minutes_planned": args.minutes,
        "privacy": "timestamps_only_no_text",
        "t0_epoch": round(t0_epoch, 3),
        "timeseries": snaps,
        "metrics": metrics,
    }

    write_json(rec, out)

    print("\n[g-lab] saved:", out, f"({len(snaps)} timeline snapshots, epoch-stamped)")
    print(f"        APM {metrics['apm']}  ·  median gap {metrics['median_inter_action_ms']} ms"
          f"  ·  consistency CV {metrics['inter_action_cv']}")
    ma = metrics["mouse_aim"]
    print(f"        aim path {ma['path_px']} px  ·  avg speed {ma['avg_speed_px_s']} px/s"
          f"  ·  corrections/min {ma['corrections_per_min']}")
    print("        (state/tempo read, not a g measure. Pair with dose.html pre/post and aim.html.)")


if __name__ == "__main__":
    main()
