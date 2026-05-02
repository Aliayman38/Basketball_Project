"""
run_gui.py  —  Basketball Analytics Launcher
─────────────────────────────────────────────
python3 run_gui.py

Flow:
  1. Configure settings (left panel)
  2. Click "Calibrate Teams" — pick a frame, see player crops split into
     two groups by K-Means, click which group is Team A
  3. Click "Run Pipeline" — pipeline runs with locked team assignment
  4. Results tab shows heatmaps + CSV tables
"""

from __future__ import annotations
import os, queue, subprocess, sys, threading, tkinter as tk
from tkinter import filedialog, ttk

try:
    from PIL import Image, ImageTk
    _PIL = True
except ImportError:
    _PIL = False

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from analytics.homography import HomographyTransformer
from team_clustering.clusterer import TeamClusterer, CLASS_PLAYER, CLASS_REF
from detection.detector import BasketballDetector

BG   = "#0d0d0f"
SURF = "#141418"
BRD  = "#232329"
TXT  = "#e6e6f0"
MUT  = "#666680"
ACC  = "#f97316"
GRN  = "#22c55e"
RED  = "#ef4444"
MONO = ("Courier", 10)
HEAD = ("Helvetica", 9, "bold")

THUMB_W, THUMB_H = 80, 120   # player crop thumbnail size


# ─────────────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Basketball Analytics")
        self.configure(bg=BG)
        self.geometry("1280x760")
        self.minsize(1000, 600)
        self.log_q: queue.Queue[str | None] = queue.Queue()
        self._running        = False
        self._tk_images      = []
        self._team_a_is_g0   = None   # None = not calibrated yet
        self._clusterer      = None   # shared clusterer instance
        self._build()
        self.after(60, self._poll_log)

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build(self):
        # top bar
        bar = tk.Frame(self, bg=SURF, height=50)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)
        tk.Label(bar, text="BBALL", bg=SURF, fg=ACC,
                 font=("Helvetica", 14, "bold")).pack(side=tk.LEFT, padx=(16,0))
        tk.Label(bar, text="ANALYTICS", bg=SURF, fg=TXT,
                 font=("Helvetica", 14, "bold")).pack(side=tk.LEFT)
        tk.Label(bar, text="  ·  SAM2 + RT-DETR + TEAM CLUSTERING",
                 bg=SURF, fg=MUT, font=("Helvetica", 9)).pack(side=tk.LEFT)
        self.lbl_status = tk.Label(bar, text="Ready", bg=SURF, fg=MUT,
                                   font=("Helvetica", 10, "bold"))
        self.lbl_status.pack(side=tk.RIGHT, padx=16)

        pw = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BRD, sashwidth=4)
        pw.pack(fill=tk.BOTH, expand=True)

        # ── left sidebar ──────────────────────────────────────────────────────
        left = tk.Frame(pw, bg=SURF, width=290)
        left.pack_propagate(False)
        pw.add(left, minsize=230)

        def section(label):
            tk.Label(left, text=label, bg=SURF, fg=MUT,
                     font=HEAD).pack(anchor="w", padx=14, pady=(14,3))

        def entry_row(label, default="", browse=False):
            tk.Label(left, text=label, bg=SURF, fg=MUT,
                     font=("Helvetica", 9)).pack(anchor="w", padx=14)
            v = tk.StringVar(value=default)
            f = tk.Frame(left, bg=SURF)
            f.pack(fill=tk.X, padx=14, pady=(2,8))
            e = tk.Entry(f, textvariable=v, bg="#0d0d0f", fg=TXT,
                         insertbackground=TXT, relief=tk.FLAT, font=MONO,
                         highlightthickness=1, highlightbackground=BRD,
                         highlightcolor=ACC)
            e.pack(side=tk.LEFT, fill=tk.X, expand=True)
            if browse:
                def _b(var=v):
                    p = filedialog.askopenfilename()
                    if p: var.set(p)
                tk.Button(f, text="…", command=_b, bg=BRD, fg=TXT,
                          relief=tk.FLAT, font=("Helvetica",9), padx=6,
                          activebackground="#2e2e38").pack(side=tk.LEFT, padx=(4,0))
            return v

        section("INPUT")
        self.v_video  = entry_row("Video file", "data/test_3.mp4", True)

        section("MODEL")
        self.v_model  = entry_row("RT-DETR weights",
                                  "models/RT-DETR/RT-DETR.pt", True)
        self.v_sam2   = entry_row("SAM2 checkpoint",
                                  "models/sam2/sam2.1_hiera_small.pt", True)
        tk.Label(left, text="SAM2 size", bg=SURF, fg=MUT,
                 font=("Helvetica",9)).pack(anchor="w", padx=14)
        self.v_size = tk.StringVar(value="small")
        om = tk.OptionMenu(left, self.v_size, "small", "base", "large")
        om.config(bg="#0d0d0f", fg=TXT, activebackground=BRD,
                  highlightthickness=0, relief=tk.FLAT, font=MONO)
        om["menu"].config(bg="#0d0d0f", fg=TXT, activebackground=BRD)
        om.pack(fill=tk.X, padx=14, pady=(2,8))
        self.v_chunk = entry_row("Chunk size", "60")

        section("OUTPUT")
        self.v_output = entry_row("Output video", "runs/detect/output.mp4")

        section("ANALYTICS  (optional)")
        self.v_hom  = entry_row("Homography .npz", "", True)
        self.v_mpp  = entry_row("Metres / pixel", "")

        tk.Frame(left, bg=BRD, height=1).pack(fill=tk.X, pady=8)

        # Team calibration badge
        self.calib_badge = tk.Label(
            left, text="⚠  Teams not calibrated", bg="#2a1500",
            fg="#f97316", font=("Helvetica", 9, "bold"), pady=4
        )
        self.calib_badge.pack(fill=tk.X, padx=14, pady=(0,6))

        self.btn_calib = tk.Button(
            left, text="🎽  Calibrate Teams",
            command=self.open_calibration,
            bg="#1e3a5f", fg="#93c5fd", relief=tk.FLAT,
            font=("Helvetica", 10, "bold"), pady=8, cursor="hand2",
            activebackground="#1e4a7f", activeforeground="#fff"
        )
        self.btn_calib.pack(fill=tk.X, padx=14, pady=(0,8))

        self.btn_run = tk.Button(
            left, text="▶  Run Pipeline", command=self.run,
            bg=ACC, fg="#fff", relief=tk.FLAT,
            font=("Helvetica", 12, "bold"), pady=10, cursor="hand2",
            activebackground="#ea6a10", activeforeground="#fff"
        )
        self.btn_run.pack(fill=tk.X, padx=14, pady=(0,14))

        # ── right: notebook ───────────────────────────────────────────────────
        right = tk.Frame(pw, bg=BG)
        pw.add(right, minsize=600)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",     background=BG,   borderwidth=0)
        style.configure("TNotebook.Tab", background=SURF, foreground=MUT,
                        padding=[14,7],  font=("Helvetica",9,"bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", BG)],
                  foreground=[("selected", ACC)])

        self.nb = ttk.Notebook(right)
        self.nb.pack(fill=tk.BOTH, expand=True)

        # Log tab
        log_f = tk.Frame(self.nb, bg=BG)
        self.nb.add(log_f, text="  LOG  ")
        self.log_text = tk.Text(
            log_f, bg="#0a0a0c", fg="#aaaacc", font=MONO,
            relief=tk.FLAT, wrap=tk.WORD, state=tk.DISABLED,
            insertbackground=TXT, selectbackground="#2a2a3a"
        )
        sb = tk.Scrollbar(log_f, command=self.log_text.yview, bg=SURF)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.log_text.tag_config("phase", foreground=ACC,
                                 font=("Courier",10,"bold"))
        self.log_text.tag_config("ok",   foreground=GRN)
        self.log_text.tag_config("err",  foreground=RED)
        self.log_text.tag_config("warn", foreground="#facc15")
        self.log_text.tag_config("cmd",  foreground="#60a5fa")

        # Results tab
        res_outer = tk.Frame(self.nb, bg=BG)
        self.nb.add(res_outer, text="  RESULTS  ")
        self.res_canvas = tk.Canvas(res_outer, bg=BG, highlightthickness=0)
        res_vsb = tk.Scrollbar(res_outer, orient=tk.VERTICAL,
                               command=self.res_canvas.yview)
        self.res_canvas.configure(yscrollcommand=res_vsb.set)
        res_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.res_canvas.pack(fill=tk.BOTH, expand=True)
        self.res_frame = tk.Frame(self.res_canvas, bg=BG)
        self._res_win  = self.res_canvas.create_window(
            (0,0), window=self.res_frame, anchor="nw")
        self.res_frame.bind("<Configure>",
            lambda e: self.res_canvas.configure(
                scrollregion=self.res_canvas.bbox("all")))
        self.res_canvas.bind("<Configure>",
            lambda e: self.res_canvas.itemconfig(self._res_win, width=e.width))

        self._log("Configure settings and click  🎽 Calibrate Teams  before running.\n", "warn")

    # ── Team calibration window ───────────────────────────────────────────────
    def open_calibration(self):
        video = self.v_video.get().strip()
        model = self.v_model.get().strip()
        if not os.path.exists(video):
            self._log(f"[ERROR] Video not found: {video}", "err")
            return
        if not os.path.exists(model):
            self._log(f"[ERROR] Model not found: {model}", "err")
            return

        # Ask which frame to use
        top = tk.Toplevel(self, bg=BG)
        top.title("Team Calibration — Pick Frame")
        top.geometry("500x160")
        top.resizable(False, False)
        top.grab_set()

        tk.Label(top, text="Which frame to use for calibration?",
                 bg=BG, fg=TXT, font=("Helvetica",11)).pack(pady=(20,8))
        tk.Label(top, text="Pick a frame where all players are clearly visible.",
                 bg=BG, fg=MUT, font=("Helvetica",9)).pack()

        cap   = cv2.VideoCapture(video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        cap.release()

        frame_var = tk.IntVar(value=0)
        row = tk.Frame(top, bg=BG)
        row.pack(fill=tk.X, padx=30, pady=12)
        tk.Label(row, text="Frame:", bg=BG, fg=MUT,
                 font=MONO).pack(side=tk.LEFT)
        tk.Scale(row, from_=0, to=total, orient=tk.HORIZONTAL,
                 variable=frame_var, bg=BG, fg=TXT,
                 highlightthickness=0, troughcolor=BRD,
                 activebackground=ACC, length=300).pack(side=tk.LEFT, padx=8)
        tk.Label(row, textvariable=frame_var,
                 bg=BG, fg=TXT, font=MONO, width=5).pack(side=tk.LEFT)

        def _proceed():
            top.destroy()
            self._run_calibration(video, model, frame_var.get())

        tk.Button(top, text="Detect Players →", command=_proceed,
                  bg=ACC, fg="#fff", relief=tk.FLAT,
                  font=("Helvetica",11,"bold"), padx=20, pady=8,
                  cursor="hand2").pack(pady=4)

    def _run_calibration(self, video: str, model_path: str, frame_idx: int):
        self.lbl_status.config(text="Detecting…", fg=ACC)
        self.btn_calib.config(state=tk.DISABLED, text="Detecting…")
        self.update()

        def _thread():
            try:
                # Load frame
                cap = cv2.VideoCapture(video)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                cap.release()
                if not ok:
                    raise RuntimeError(f"Could not read frame {frame_idx}")

                # Detect
                detector = BasketballDetector(model_path=model_path,
                                              conf=0.30, iou=0.45, imgsz=640)
                result   = detector.detect(frame)
                dets     = detector.parse(result)

                # Give track_id placeholders so calibrate_from_frame works
                pid = 0
                for d in dets:
                    if d["class_id"] == CLASS_PLAYER:
                        d["track_id"] = pid
                        pid += 1
                    else:
                        d["track_id"] = -1

                # Cluster
                clusterer = TeamClusterer()
                g0, g1 = clusterer.calibrate_from_frame(frame, dets)
                self._clusterer = clusterer

                self.after(0, lambda: self._show_calibration_window(g0, g1))
            except Exception as exc:
                self.after(0, lambda: self._log(f"[ERROR] {exc}", "err"))
            finally:
                self.after(0, lambda: (
                    self.btn_calib.config(state=tk.NORMAL,
                                         text="🎽  Calibrate Teams"),
                    self.lbl_status.config(text="Ready", fg=MUT),
                ))

        threading.Thread(target=_thread, daemon=True).start()

    def _show_calibration_window(
        self,
        group0: list[np.ndarray],
        group1: list[np.ndarray],
    ):
        if not group0 and not group1:
            self._log("[ERROR] No players detected. Try a different frame.", "err")
            return

        win = tk.Toplevel(self, bg=BG)
        win.title("Team Calibration — Which group is Team A?")
        win.geometry("900x520")
        win.grab_set()
        self._calib_tkimages = []   # keep refs

        tk.Label(win,
                 text="Look at the jersey colours. Click  'This is Team A'  under the correct group.",
                 bg=BG, fg=TXT, font=("Helvetica",11)).pack(pady=(14,4))
        tk.Label(win,
                 text="The other group is automatically Team B.",
                 bg=BG, fg=MUT, font=("Helvetica",9)).pack(pady=(0,10))

        content = tk.Frame(win, bg=BG)
        content.pack(fill=tk.BOTH, expand=True, padx=20)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)

        def make_group_panel(parent, col, crops, group_name, color):
            frame = tk.Frame(parent, bg=SURF, padx=10, pady=10)
            frame.grid(row=0, column=col, sticky="nsew", padx=(0 if col else 0, 8))

            tk.Label(frame, text=group_name, bg=SURF, fg=color,
                     font=("Helvetica",12,"bold")).pack(pady=(0,8))

            # Crop grid — up to 10 thumbnails
            grid_f = tk.Frame(frame, bg=SURF)
            grid_f.pack()
            per_row = 5
            for i, crop in enumerate(crops[:10]):
                # Convert BGR → RGB → PIL → thumbnail
                rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil  = Image.fromarray(rgb)
                pil.thumbnail((THUMB_W, THUMB_H))
                tki  = ImageTk.PhotoImage(pil)
                self._calib_tkimages.append(tki)
                r, c = divmod(i, per_row)
                lbl  = tk.Label(grid_f, image=tki, bg=SURF,
                                 relief=tk.FLAT, bd=0)
                lbl.grid(row=r, column=c, padx=3, pady=3)

            if not crops:
                tk.Label(frame, text="(no players detected)",
                         bg=SURF, fg=MUT, font=MONO).pack(pady=20)

            count_lbl = tk.Label(frame,
                text=f"{len(crops)} player{'s' if len(crops)!=1 else ''}",
                bg=SURF, fg=MUT, font=("Helvetica",9))
            count_lbl.pack(pady=(6,8))

            return frame

        if _PIL:
            make_group_panel(content, 0, group0, "Group 1", "#60a5fa")
            make_group_panel(content, 1, group1, "Group 2", "#f87171")
        else:
            tk.Label(content,
                     text="Install Pillow to see crops:\n  pip install pillow",
                     bg=BG, fg=MUT, font=MONO).grid(row=0, column=0,
                     columnspan=2, pady=40)

        # Buttons
        btn_row = tk.Frame(win, bg=BG)
        btn_row.pack(pady=14)

        def _choose(g0_is_team_a: bool):
            if self._clusterer:
                self._clusterer.set_user_label_map(g0_is_team_a)
            self._team_a_is_g0 = g0_is_team_a
            self.calib_badge.config(
                text="✓  Teams calibrated",
                bg="#0f2a0f", fg=GRN
            )
            self._log(
                f"[Calibrate] Team A = Group {'1' if g0_is_team_a else '2'}, "
                f"Team B = Group {'2' if g0_is_team_a else '1'}", "ok"
            )
            win.destroy()

        tk.Button(btn_row,
                  text="✓  Group 1 is Team A",
                  command=lambda: _choose(True),
                  bg="#1e3a5f", fg="#93c5fd", relief=tk.FLAT,
                  font=("Helvetica",11,"bold"), padx=20, pady=9,
                  cursor="hand2",
                  activebackground="#1e4a7f").pack(side=tk.LEFT, padx=10)

        tk.Button(btn_row,
                  text="✓  Group 2 is Team A",
                  command=lambda: _choose(False),
                  bg="#3a1a1a", fg="#f87171", relief=tk.FLAT,
                  font=("Helvetica",11,"bold"), padx=20, pady=9,
                  cursor="hand2",
                  activebackground="#4a1a1a").pack(side=tk.LEFT, padx=10)

    # ── Log helpers ───────────────────────────────────────────────────────────
    def _log(self, text: str, tag: str = ""):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n", tag)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _log_clear(self):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _tag(self, line: str) -> str:
        l = line.lower()
        if "phase" in line or "═" in line:            return "phase"
        if "error" in l or "traceback" in l:           return "err"
        if "warning" in l:                             return "warn"
        if any(x in line for x in ("→","✓","saved","complete","done")): return "ok"
        if line.startswith("$"):                       return "cmd"
        return ""

    def _poll_log(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                if msg is None:
                    self._running = False
                    self.btn_run.config(state=tk.NORMAL, text="▶  Run Pipeline")
                    self.lbl_status.config(text="Done ✓", fg=GRN)
                    self._log("\n✓  Pipeline complete — loading results…", "ok")
                    self.after(500, self._load_results)
                else:
                    self._log(msg, self._tag(msg))
        except queue.Empty:
            pass
        self.after(60, self._poll_log)

    # ── Run ───────────────────────────────────────────────────────────────────
    def run(self):
        if self._running:
            return

        if self._team_a_is_g0 is None:
            # Warn but allow running without calibration
            self._log("[WARN] Running without team calibration — teams may be swapped.", "warn")

        self._running = True
        self._log_clear()
        self.nb.select(0)
        self.btn_run.config(state=tk.DISABLED, text="⏳  Running…")
        self.lbl_status.config(text="Running…", fg=ACC)

        cmd = [
            sys.executable, os.path.join(ROOT, "main.py"),
            "--video",      self.v_video.get().strip(),
            "--model",      self.v_model.get().strip(),
            "--sam2",       self.v_sam2.get().strip(),
            "--output",     self.v_output.get().strip() or "runs/detect/output.mp4",
            "--sam2-size",  self.v_size.get(),
            "--chunk-size", self.v_chunk.get().strip() or "60",
        ]
        hom = self.v_hom.get().strip()
        mpp = self.v_mpp.get().strip()
        if hom and os.path.exists(hom):
            cmd += ["--homography", hom]
        if mpp:
            cmd += ["--meters-per-pixel", mpp]
        if self._team_a_is_g0 is not None:
            cmd += ["--team-a-cluster", "0" if self._team_a_is_g0 else "1"]

        self._log(f"$ {' '.join(cmd)}\n", "cmd")

        def _thread():
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=ROOT,
                )
                for line in proc.stdout:
                    self.log_q.put(line.rstrip())
                proc.wait()
            except Exception as exc:
                self.log_q.put(f"[ERROR] {exc}")
            finally:
                self.log_q.put(None)

        threading.Thread(target=_thread, daemon=True).start()

    # ── Results ───────────────────────────────────────────────────────────────
    def _load_results(self):
        import csv as _csv
        for w in self.res_frame.winfo_children():
            w.destroy()
        self._tk_images.clear()

        out_dir = os.path.dirname(
            self.v_output.get().strip()) or "runs/detect"

        tk.Label(self.res_frame, text="ANALYTICS RESULTS",
                 bg=BG, fg=MUT, font=HEAD).pack(anchor="w", padx=20, pady=(16,10))

        # Heatmaps
        hm_row = tk.Frame(self.res_frame, bg=BG)
        hm_row.pack(fill=tk.X, padx=20, pady=(0,16))
        for fname, title in [("heatmap_team_a.png","Team A Heatmap"),
                              ("heatmap_team_b.png","Team B Heatmap")]:
            path = os.path.join(out_dir, fname)
            card = tk.Frame(hm_row, bg=SURF, padx=12, pady=12)
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))
            tk.Label(card, text=title, bg=SURF, fg=MUT,
                     font=HEAD).pack(anchor="w", pady=(0,8))
            if os.path.exists(path) and _PIL:
                img = Image.open(path)
                img.thumbnail((500, 280))
                tki = ImageTk.PhotoImage(img)
                self._tk_images.append(tki)
                tk.Label(card, image=tki, bg=SURF).pack()
            else:
                msg = ("Not available\n(needs --homography)"
                       if not os.path.exists(path) else
                       "pip install pillow  to view images")
                tk.Label(card, text=msg, bg=SURF, fg=MUT,
                         font=MONO, width=38, height=7).pack()

        # CSV tables
        for fname, title in [("distance_report.csv","Distance Report"),
                              ("speed_report.csv",   "Speed Report")]:
            path = os.path.join(out_dir, fname)
            if not os.path.exists(path):
                continue
            tk.Label(self.res_frame, text=title, bg=BG, fg=MUT,
                     font=HEAD).pack(anchor="w", padx=20, pady=(10,4))
            card = tk.Frame(self.res_frame, bg=SURF)
            card.pack(fill=tk.X, padx=20, pady=(0,14))
            with open(path) as f:
                all_rows = list(_csv.reader(f))
            if not all_rows:
                continue
            headers, data = all_rows[0], all_rows[1:]
            for ci, h in enumerate(headers):
                tk.Label(card, text=h, bg=BRD, fg=MUT,
                         font=("Courier",9,"bold"),
                         padx=10, pady=5).grid(
                    row=0, column=ci, sticky="ew", padx=1, pady=1)
            for ri, row in enumerate(data, 1):
                row_bg = SURF if ri%2==0 else "#1a1a20"
                for ci, val in enumerate(row):
                    tk.Label(card, text=val, bg=row_bg, fg=TXT,
                             font=MONO, padx=10, pady=4).grid(
                        row=ri, column=ci, sticky="ew", padx=1, pady=1)
            for ci in range(len(headers)):
                card.grid_columnconfigure(ci, weight=1)

        self.nb.select(1)


if __name__ == "__main__":
    App().mainloop()