"""
run_gui.py
───────────
Native desktop GUI — no browser needed.

Usage:
    python3 run_gui.py --video data/test_3.mp4

Step 1: Drag the slider to pick a good frame.
Step 2: Click the 4 court corners (TL → TR → BR → BL).
Step 3: Fill in settings, hit Run.
Step 4: Watch the live log.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from analytics.homography import HomographyTransformer

CORNER_COLORS = ["#ff8877", "#ffaa00", "#44ff44", "#4488ff"]
CORNER_NAMES  = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
HFILE         = os.path.join(ROOT, "config", "homography.npz")


# ─────────────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self, video_path: str):
        super().__init__()
        self.title("Basketball Analytics")
        self.configure(bg="#0d0d0f")
        self.geometry("1300x780")
        self.minsize(1000, 640)

        self.video_path  = video_path
        self.cap         = cv2.VideoCapture(video_path)
        self.total       = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps_vid     = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_idx   = 0
        self.raw_frame   = None   # current BGR frame
        self.tk_image    = None
        self.points: list[tuple[int, int]] = []
        self.canvas_scale = 1.0
        self.log_q: queue.Queue[str | None] = queue.Queue()
        self._running = False

        self._build_ui()
        # Delay first frame load until the window is fully rendered —
        # winfo_width/height return 1 in __init__ so _redraw would fail silently.
        self.after(150, lambda: self._load_frame(0))
        self.after(50, self._poll_log)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        BG   = "#0d0d0f"
        SURF = "#141418"
        BRD  = "#232329"
        TXT  = "#e6e6f0"
        MUT  = "#666680"
        ACC  = "#f97316"

        # ── top bar ──────────────────────────────────────────────────────────
        bar = tk.Frame(self, bg=SURF, height=48)
        bar.pack(fill=tk.X, side=tk.TOP)
        bar.pack_propagate(False)
        tk.Label(bar, text="BBALL", bg=SURF, fg=ACC,
                 font=("Helvetica", 15, "bold")).pack(side=tk.LEFT, padx=(18,0))
        tk.Label(bar, text="ANALYTICS", bg=SURF, fg=TXT,
                 font=("Helvetica", 15, "bold")).pack(side=tk.LEFT)
        tk.Label(bar, text=" · SAM2 + YOLO + TEAM CLUSTERING",
                 bg=SURF, fg=MUT, font=("Helvetica", 10)).pack(side=tk.LEFT, padx=8)

        self.lbl_step = tk.Label(bar, text="Step 1 — Click 4 court corners",
                                 bg=SURF, fg=ACC, font=("Helvetica", 11, "bold"))
        self.lbl_step.pack(side=tk.RIGHT, padx=18)

        # ── main pane ─────────────────────────────────────────────────────────
        pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BRD,
                              sashwidth=4, sashrelief=tk.FLAT)
        pane.pack(fill=tk.BOTH, expand=True)

        # ── left sidebar ──────────────────────────────────────────────────────
        left = tk.Frame(pane, bg=SURF, width=290)
        left.pack_propagate(False)
        pane.add(left, minsize=220)

        def section(parent, title):
            tk.Label(parent, text=title, bg=SURF, fg=MUT,
                     font=("Helvetica", 8, "bold")).pack(
                anchor="w", padx=16, pady=(14, 4))

        def field(parent, label, default="", width=28):
            tk.Label(parent, text=label, bg=SURF, fg=MUT,
                     font=("Helvetica", 9)).pack(anchor="w", padx=16)
            v = tk.StringVar(value=default)
            e = tk.Entry(parent, textvariable=v, width=width,
                         bg="#0d0d0f", fg=TXT, insertbackground=TXT,
                         relief=tk.FLAT, font=("Courier", 11),
                         highlightthickness=1, highlightbackground=BRD,
                         highlightcolor=ACC)
            e.pack(fill=tk.X, padx=16, pady=(2, 10))
            return v

        section(left, "VIDEO")
        self.v_video = field(left, "Path", self.video_path)

        section(left, "MODEL")
        self.v_sam2size = tk.StringVar(value="small")
        tk.Label(left, text="SAM2 size", bg=SURF, fg=MUT,
                 font=("Helvetica", 9)).pack(anchor="w", padx=16)
        opt = tk.OptionMenu(left, self.v_sam2size, "small", "base", "large")
        opt.config(bg="#0d0d0f", fg=TXT, activebackground=BRD,
                   highlightthickness=0, relief=tk.FLAT, font=("Helvetica", 11))
        opt["menu"].config(bg="#0d0d0f", fg=TXT, activebackground=BRD)
        opt.pack(fill=tk.X, padx=16, pady=(2, 10))
        self.v_chunk = field(left, "Chunk size", "60")

        section(left, "ANALYTICS")
        self.v_mpp = field(left, "Metres / pixel (optional)", "")

        # Corner status
        section(left, "CORNERS")
        self.corner_labels = []
        for i, name in enumerate(CORNER_NAMES):
            f = tk.Frame(left, bg=SURF)
            f.pack(fill=tk.X, padx=16, pady=1)
            dot = tk.Canvas(f, width=10, height=10, bg=SURF,
                            highlightthickness=0)
            dot.pack(side=tk.LEFT, padx=(0, 6))
            dot.create_oval(1, 1, 9, 9, fill=BRD, outline=BRD, tags="dot")
            lbl = tk.Label(f, text=f"{i+1}. {name}", bg=SURF, fg=MUT,
                           font=("Helvetica", 10))
            lbl.pack(side=tk.LEFT)
            self.corner_labels.append((dot, lbl))

        # Buttons
        btn_frame = tk.Frame(left, bg=SURF)
        btn_frame.pack(fill=tk.X, padx=16, pady=(12, 4))

        self.btn_reset = tk.Button(
            btn_frame, text="↩  Reset corners",
            command=self.reset_corners, bg="#2e2e38", fg=TXT,
            relief=tk.FLAT, font=("Helvetica", 10), padx=10, pady=6,
            activebackground="#3a3a4a", activeforeground=TXT, cursor="hand2"
        )
        self.btn_reset.pack(fill=tk.X, pady=(0, 6))

        self.btn_run = tk.Button(
            btn_frame, text="▶  Run with All Analytics",
            command=self.run_pipeline, bg="#555", fg="#999", state=tk.DISABLED,
            relief=tk.FLAT, font=("Helvetica", 11, "bold"), padx=10, pady=9,
            activebackground="#ea6a10", activeforeground="#fff", cursor="hand2",
            disabledforeground="#888"
        )
        self.btn_run.pack(fill=tk.X)

        # ── right area ────────────────────────────────────────────────────────
        right = tk.Frame(pane, bg=BG)
        pane.add(right, minsize=500)

        # Canvas
        self.canvas = tk.Canvas(right, bg="#000", cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        # Frame slider
        slider_bar = tk.Frame(right, bg=SURF, height=40)
        slider_bar.pack(fill=tk.X)
        slider_bar.pack_propagate(False)
        tk.Label(slider_bar, text="Frame", bg=SURF, fg=MUT,
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(12, 6))
        self.slider_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(slider_bar, from_=0, to=max(self.total - 1, 1),
                                orient=tk.HORIZONTAL, variable=self.slider_var,
                                command=self._on_slider)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.lbl_frame = tk.Label(slider_bar, text=f"0 / {self.total-1}",
                                  bg=SURF, fg=TXT, font=("Courier", 10),
                                  width=12)
        self.lbl_frame.pack(side=tk.RIGHT, padx=10)

        # Log area (hidden until run)
        self.log_frame = tk.Frame(right, bg=BG)
        self.log_text = tk.Text(
            self.log_frame, bg="#0a0a0c", fg="#aaaacc",
            font=("Courier", 10), relief=tk.FLAT,
            insertbackground="#fff", selectbackground="#2a2a3a",
            wrap=tk.WORD, state=tk.DISABLED
        )
        sb = tk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Tag styles in log
        self.log_text.tag_config("phase", foreground="#f97316", font=("Courier", 10, "bold"))
        self.log_text.tag_config("ok",    foreground="#22c55e")
        self.log_text.tag_config("err",   foreground="#ef4444")
        self.log_text.tag_config("warn",  foreground="#facc15")
        self.log_text.tag_config("cmd",   foreground="#60a5fa")
        self.log_text.tag_config("muted", foreground="#555577")

    # ── Frame loading & drawing ────────────────────────────────────────────────

    def _load_frame(self, idx: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        self.raw_frame = frame
        self.frame_idx = idx
        self.lbl_frame.config(text=f"{idx} / {self.total-1}")
        self._redraw()

    def _on_slider(self, val):
        idx = int(float(val))
        if idx != self.frame_idx:
            self._load_frame(idx)

    # Pillow resampling filter — compatible across all versions
    _RESAMPLE = getattr(Image, "LANCZOS",
                getattr(Image, "Resampling", Image).LANCZOS
                if hasattr(Image, "Resampling")
                else Image.ANTIALIAS)

    def _redraw(self):
        if self.raw_frame is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        fh, fw = self.raw_frame.shape[:2]
        scale = min(cw / fw, ch / fh)
        self.canvas_scale = scale
        nw, nh = int(fw * scale), int(fh * scale)
        ox = (cw - nw) // 2
        oy = (ch - nh) // 2
        self._ox, self._oy = ox, oy

        img_rgb = cv2.cvtColor(self.raw_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).resize((nw, nh), self._RESAMPLE)
        self.tk_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor=tk.NW, image=self.tk_image)

        # Polygon
        if len(self.points) > 1:
            pts_flat = []
            for px, py in self.points:
                pts_flat += [ox + px * scale, oy + py * scale]
            if len(self.points) == 4:
                pts_flat += pts_flat[:2]
            self.canvas.create_line(*pts_flat, fill="#ffffff55", width=1,
                                    dash=(5, 4))

        # Markers
        for i, (px, py) in enumerate(self.points):
            sx = ox + px * scale
            sy = oy + py * scale
            col = CORNER_COLORS[i]
            r = 9
            self.canvas.create_oval(sx-r, sy-r, sx+r, sy+r,
                                    fill=col+"44", outline=col, width=2.5)
            self.canvas.create_line(sx-14, sy, sx+14, sy, fill=col, width=1.5)
            self.canvas.create_line(sx, sy-14, sx, sy+14, fill=col, width=1.5)
            self.canvas.create_text(sx+14, sy-7, text=CORNER_NAMES[i][:2],
                                    fill="#ffffff", font=("Helvetica", 10, "bold"),
                                    anchor=tk.W)
            self.canvas.create_text(sx+14, sy+7,
                                    text=f"({px},{py})", fill=col,
                                    font=("Courier", 9), anchor=tk.W)

    # ── Corner picking ────────────────────────────────────────────────────────

    def _on_click(self, event):
        if len(self.points) >= 4 or self._running:
            return
        ox = getattr(self, "_ox", 0)
        oy = getattr(self, "_oy", 0)
        px = round((event.x - ox) / self.canvas_scale)
        py = round((event.y - oy) / self.canvas_scale)
        if px < 0 or py < 0:
            return
        self.points.append((px, py))
        i = len(self.points) - 1
        dot, lbl = self.corner_labels[i]
        dot.itemconfig("dot", fill=CORNER_COLORS[i], outline=CORNER_COLORS[i])
        lbl.config(fg="#e6e6f0")

        # Update button BEFORE _redraw so a redraw crash can't block it
        if len(self.points) == 4:
            self.btn_run.config(state=tk.NORMAL, bg="#f97316", fg="#fff")
            self.btn_run.update()   # force repaint on Linux/GTK
            self.lbl_step.config(text="Step 2 — Run the pipeline")
            self.canvas.config(cursor="arrow")

        self._redraw()

    def reset_corners(self):
        self.points.clear()
        self.btn_run.config(state=tk.DISABLED, bg="#555", fg="#999")
        self.canvas.config(cursor="crosshair")
        self.lbl_step.config(text="Step 1 — Click 4 court corners")
        for dot, lbl in self.corner_labels:
            dot.itemconfig("dot", fill="#2e2e38", outline="#2e2e38")
            lbl.config(fg="#666680")
        self._redraw()

    # ── Pipeline run ──────────────────────────────────────────────────────────

    def run_pipeline(self):
        if self._running:
            return

        # Save homography
        try:
            transformer = HomographyTransformer(
                src_points=list(self.points),
                court_width_px=1060, court_height_px=560,
            )
            os.makedirs(os.path.dirname(HFILE), exist_ok=True)
            transformer.save(HFILE)
        except Exception as e:
            self._append_log(f"[ERROR] Homography failed: {e}\n", "err")
            return

        mpp   = self.v_mpp.get().strip()
        cmd   = [
            sys.executable, os.path.join(ROOT, "main.py"),
            "--video",      self.v_video.get().strip(),
            "--homography", HFILE,
            "--sam2-size",  self.v_sam2size.get(),
            "--chunk-size", self.v_chunk.get().strip(),
        ]
        if mpp:
            cmd += ["--meters-per-pixel", mpp]

        # Show log panel
        self.canvas.pack_forget()
        self.slider.master.pack_forget()
        self.log_frame.pack(fill=tk.BOTH, expand=True)
        self.btn_run.config(state=tk.DISABLED, text="⏳  Running…")
        self.btn_reset.config(state=tk.DISABLED)
        self.lbl_step.config(text="Step 3 — Running pipeline…")
        self._running = True

        self._append_log(f"$ {' '.join(cmd)}\n\n", "cmd")

        def _thread():
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=ROOT,
                )
                for line in proc.stdout:
                    self.log_q.put(line.rstrip())
                proc.wait()
                self.log_q.put(None)   # sentinel
            except Exception as exc:
                self.log_q.put(f"[ERROR] {exc}")
                self.log_q.put(None)

        threading.Thread(target=_thread, daemon=True).start()

    def _poll_log(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                if msg is None:
                    # Done
                    self._running = False
                    self.btn_run.config(state=tk.NORMAL, text="▶  Run Again")
                    self.btn_reset.config(state=tk.NORMAL)
                    self.lbl_step.config(text="Done ✓")
                    self._append_log("\n✓ Pipeline complete.\n", "ok")
                else:
                    tag = "muted"
                    if "Phase" in msg or "═" in msg:    tag = "phase"
                    elif "error" in msg.lower():        tag = "err"
                    elif "warning" in msg.lower():      tag = "warn"
                    elif any(x in msg for x in ("done","saved","→","✓","complete")): tag = "ok"
                    self._append_log(msg + "\n", tag)
        except queue.Empty:
            pass
        self.after(60, self._poll_log)

    def _append_log(self, text: str, tag: str = ""):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, text, tag)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def on_close(self):
        self.cap.release()
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="data/test_3.mp4")
    args = p.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"Video not found: {args.video}")

    app = App(args.video)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()