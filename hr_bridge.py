#!/usr/bin/env python3
"""hr_bridge.py — local rPPG -> HTTP bridge for the g-lab flicker page.

The hosted page (https://lockthebow.github.io/flicker/) normally reads your pulse
from the webcam *inside the browser*. Browsers without a camera API (Steam overlay)
can't do that read, so this script performs the identical POS rPPG read on this
machine and serves the derived numbers at http://127.0.0.1:8787/hr — the page
polls that endpoint automatically whenever its own camera is unavailable
(force it with ?nocam appended to the page URL).

Privacy: derived numbers only. No video is stored; nothing leaves 127.0.0.1.

Usage:
  python3 hr_bridge.py              # webcam POS rPPG (pip install opencv-python numpy)
  python3 hr_bridge.py --test       # synthetic 66 bpm signal, no camera — link check
  python3 hr_bridge.py --port 8787 --cam 0
"""
import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"hr": None, "hrv": None, "arousal": 0.5, "quality": 0.0, "src": None, "ts": 0.0}
LOCK = threading.Lock()


def set_state(**kw):
    with LOCK:
        STATE.update(kw)
        STATE["ts"] = round(time.time(), 2)


class Handler(BaseHTTPRequestHandler):
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
        if self.path.split("?")[0] not in ("/hr", "/"):
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        with LOCK:
            body = json.dumps(STATE).encode()
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_loop():
    t0 = time.time()
    while True:
        t = time.time() - t0
        hr = 66 + 4 * math.sin(2 * math.pi * t / 60)
        arousal = 0.55 + 0.15 * math.sin(2 * math.pi * t / 45)
        set_state(hr=round(hr, 1), hrv=45.0, arousal=round(arousal, 3), quality=0.9, src="test")
        time.sleep(0.25)


def cam_loop(cam_index):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(
            "camera %d failed to open — check System Settings > Privacy & Security > Camera "
            "(the terminal app needs access), or try --cam 1 / --test" % cam_index
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    buf = []          # (t, rMean, gMean, bMean) over skin-masked center-face ROI
    MAXS = 12.0
    hr_est = None
    hr_hist = []
    base_hr = None
    coverage = 0.0
    last_analyze = 0.0

    def analyze():
        nonlocal hr_est, base_hr
        n = len(buf)
        ts = np.array([x[0] for x in buf])
        dur = ts[-1] - ts[0]
        if dur < 3:
            return
        fps = n / dur
        R = np.array([x[1] for x in buf])
        G = np.array([x[2] for x in buf])
        B = np.array([x[3] for x in buf])
        mR, mG, mB = R.mean(), G.mean(), B.mean()
        if min(mR, mG, mB) < 1:
            set_state(quality=0.0, src="cam")
            return
        # POS (Wang 2017): temporally-normalized RGB projected orthogonal to skin tone —
        # cancels the shared luminance/motion drift green-only can't.
        Rn, Gn, Bn = R / mR - 1, G / mG - 1, B / mB - 1
        S1 = Gn - Bn
        S2 = Gn + Bn - 2 * Rn
        al = S1.std() / (S2.std() or 1e-6)
        p = S1 + al * S2
        # detrend (~1s centered moving-average high-pass) + 3-tap low-pass + normalize
        win = max(3, round(fps * 1.0))
        s = np.empty(n)
        for i in range(n):
            a, b = max(0, i - win), min(n - 1, i + win)
            s[i] = p[i] - p[a:b + 1].mean()
        s = np.convolve(np.pad(s, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
        s /= s.std() or 1e-6
        min_lag = int(fps * 60 / 240)
        max_lag = int(math.ceil(fps * 60 / 42))
        best, best_lag, acs = -2.0, -1, []
        for lag in range(max(1, min_lag), min(max_lag, n - 1) + 1):
            r = float(np.dot(s[:n - lag], s[lag:]) / (n - lag))
            acs.append(r)
            if r > best:
                best, best_lag = r, lag
        if not acs:
            return
        ac_abs = float(np.mean(np.abs(acs)))
        # quality = autocorr peak prominence (SNR-like) x face coverage: adapt only on a real pulse
        q = max(0.0, min(1.0, (best - ac_abs) / (abs(best) + 1e-6)))
        q *= max(0.4, min(1.0, coverage * 2.5))
        hrv = None
        if best_lag > 0 and q > 0.3:
            hr = 60 / (best_lag / fps)
            hr_est = hr if hr_est is None else 0.8 * hr_est + 0.2 * hr
            hr_hist.append(hr_est)
            del hr_hist[:-240]
            min_d = max(2, round(best_lag * 0.6))
            pk = []
            for i in range(1, n - 1):
                if s[i] > 0.5 and s[i] >= s[i - 1] and s[i] > s[i + 1]:
                    if not pk or i - pk[-1] >= min_d:
                        pk.append(i)
            if len(pk) > 3:
                ibi = [(buf[pk[i]][0] - buf[pk[i - 1]][0]) * 1000 for i in range(1, len(pk))]
                df = [ibi[i] - ibi[i - 1] for i in range(1, len(ibi))]
                if df:
                    hrv = math.sqrt(sum(d * d for d in df) / len(df))
            if base_hr is None and len(hr_hist) >= 60:  # ~30 s of estimates
                base_hr = float(np.median(hr_hist))
            hr_t = max(0.0, min(1.0, (hr_est - base_hr) / 12 + 0.5)) if base_hr else 0.5
            hrv_t = max(0.0, min(1.0, 1 - hrv / 80)) if hrv is not None else 0.5
            set_state(hr=round(hr_est, 1), hrv=round(hrv, 1) if hrv is not None else None,
                      arousal=round(0.6 * hr_t + 0.4 * hrv_t, 3), quality=round(q, 2), src="cam")
        else:
            set_state(quality=round(q, 2), src="cam")

    print("camera open — reading pulse (POS). Sit facing the camera, decent light.")
    while True:
        ok, frame = cap.read()
        t = time.time()
        if not ok:
            time.sleep(0.1)
            continue
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.22):int(h * 0.50), int(w * 0.30):int(w * 0.70)]
        small = cv2.resize(roi, (64, 48)).astype(np.int32)
        b, g, r = small[:, :, 0], small[:, :, 1], small[:, :, 2]
        # skin mask: reddish, mid-bright, not clipped -> excludes background/hair/shadow/glare
        mask = (r > 60) & (r < 250) & (r > g) & (g >= b) & ((r - b) > 12)
        cnt = int(mask.sum())
        coverage = cnt / mask.size
        if cnt < mask.size * 0.12:
            mask = np.ones_like(mask, dtype=bool)  # fallback: whole ROI
        buf.append((t, float(r[mask].mean()), float(g[mask].mean()), float(b[mask].mean())))
        while buf and t - buf[0][0] > MAXS:
            buf.pop(0)
        if len(buf) > 60 and t - last_analyze >= 0.5:
            last_analyze = t
            analyze()
        time.sleep(max(0.0, 1 / 30 - (time.time() - t)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--test", action="store_true", help="serve a synthetic 66 bpm signal (no camera)")
    args = ap.parse_args()

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("serving http://127.0.0.1:%d/hr  (page falls back to this when it has no camera)" % args.port)

    def status():
        while True:
            time.sleep(2)
            with LOCK:
                s = dict(STATE)
            print("\rHR %s  HRV %s  arousal %s  q %s   " % (
                s["hr"] or "—", s["hrv"] or "—", s["arousal"], s["quality"]), end="", flush=True)

    threading.Thread(target=status, daemon=True).start()
    try:
        if args.test:
            print("TEST MODE — synthetic 66 bpm signal, no camera")
            test_loop()
        else:
            cam_loop(args.cam)
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
