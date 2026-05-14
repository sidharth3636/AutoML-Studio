import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import threading
import sys
import os

# Video animation support
try:
    import imageio
    from PIL import Image, ImageTk
    VIDEO_SUPPORT = True
except ImportError:
    VIDEO_SUPPORT = False

from coret import run_basic_automl
from nprepro import label_encode_large_dataset


# =========================
# Redirect terminal output
# =========================

class RedirectText:
    def __init__(self, widget):
        self.widget = widget

    def write(self, string):
        self.widget.insert("end", string)
        self.widget.see("end")

    def flush(self):
        pass


# =========================
# Main App
# =========================

class AutoMLApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("Light")
        ctk.set_default_color_theme("blue")

        self.title("AutoML Studio")
        self.geometry("1200x860")

        # shared variables
        self.selected_file    = tk.StringVar()
        self.problem_type_var = ctk.StringVar(value="Regression")
        self.cpu_var          = tk.IntVar(value=75)
        self.outlier_option        = ctk.StringVar(value="None")
        self.outlier_contamination = ctk.StringVar(value="0.05")
        self.skip_preprocess  = tk.BooleanVar()

        self.column_vars      = {}
        self.df               = None
        self.selected_targets = []

        # set after preprocessing finishes
        self.preprocessed_path = None

        # Video animation state
        self._video_frames   = []
        self._video_frame_idx = 0
        self._video_job      = None
        self._video_running  = False
        self._load_video_frames()

        self.build_layout()

        sys.stdout = RedirectText(self.terminal)

    # =========================
    # Video loader
    # =========================

    def _load_video_frames(self):
        """Load all PNG frames from the automl_frames/ folder."""
        if not VIDEO_SUPPORT:
            return
        script_dir   = os.path.dirname(os.path.abspath(__file__))
        frames_dir   = os.path.join(script_dir, "automl_frames")
        self._video_fps = 30  # original video fps

        if not os.path.isdir(frames_dir):
            print(f"[Video] Frames folder not found: {frames_dir}")
            return
        try:
            frame_files = sorted(
                f for f in os.listdir(frames_dir) if f.endswith(".png")
            )
            for fname in frame_files:
                img = Image.open(os.path.join(frames_dir, fname)) \
                           .resize((300, 140), Image.LANCZOS).convert("RGB")

                import numpy as np
                arr = np.array(img, dtype=np.float32)

                # Sample background color from the 4 corners (avg)
                corners = [
                    arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]
                ]
                bg_color = np.mean(corners, axis=0)  # e.g. near-black or grey

                # Replace pixels similar to bg_color (within tolerance 40)
                diff = np.abs(arr - bg_color)
                mask = np.all(diff < 40, axis=2)
                arr[mask] = [255, 255, 255]

                final = Image.fromarray(arr.astype(np.uint8), "RGB")
                self._video_frames.append(ImageTk.PhotoImage(final))
            print(f"[Video] Loaded {len(self._video_frames)} frames from automl_frames/")
        except Exception as e:
            print(f"[Video] Could not load frames: {e}")

    # =========================
    # Layout skeleton
    # =========================

    def build_layout(self):

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- sidebar ----
        sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)

        ctk.CTkLabel(
            sidebar,
            text="AutoML Studio",
            font=ctk.CTkFont(size=20, weight="bold")
        ).pack(pady=20)

        self.btn_preprocess = ctk.CTkButton(
            sidebar, text="Preprocessing", command=self.show_preprocess
        )
        self.btn_preprocess.pack(fill="x", padx=20, pady=10)

        self.btn_automl = ctk.CTkButton(
            sidebar, text="Run AutoML", command=self.show_automl
        )
        self.btn_automl.pack(fill="x", padx=20)

        self.status_label = ctk.CTkLabel(sidebar, text="Preprocessing Mode")
        self.status_label.pack(side="bottom", pady=20)

        # ---- main area ----
        self.main = ctk.CTkFrame(self)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(0, weight=0)   # dataset row
        self.main.grid_rowconfigure(1, weight=1)   # page content
        self.main.grid_rowconfigure(2, weight=0)   # terminal

        self._build_dataset_row()

        self.page_preprocess = ctk.CTkFrame(self.main)
        self.page_automl     = ctk.CTkFrame(self.main)

        self._build_preprocess_page()
        self._build_automl_page()

        self._build_terminal()

        self.show_preprocess()

    # =========================
    # Shared dataset row
    # =========================

    def _build_dataset_row(self):

        f = ctk.CTkFrame(self.main)
        f.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        f.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            f, text="Dataset",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=10)

        self.dataset_entry = ctk.CTkEntry(f, textvariable=self.selected_file)
        self.dataset_entry.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

        ctk.CTkButton(f, text="Browse", command=self.browse_file).grid(row=1, column=1, padx=10)

    # =========================
    # Shared terminal  (taller)
    # =========================

    def _build_terminal(self):

        f = ctk.CTkFrame(self.main)
        f.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 10))

        ctk.CTkLabel(f, text="Terminal",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=6, pady=(4, 0))

        # height increased to 220
        self.terminal = ctk.CTkTextbox(f, height=220)
        self.terminal.pack(fill="both", expand=True, padx=6, pady=6)

    # =========================
    # PAGE 1 — Preprocessing
    # =========================

    def _build_preprocess_page(self):

        p = self.page_preprocess
        p.grid_columnconfigure(0, weight=5)   # preprocessing options — much wider
        p.grid_columnconfigure(1, weight=1)   # preprocessing log — narrow
        p.grid_rowconfigure(0, weight=1)

        # ============================================================
        # LEFT PANEL — fixed layout: scrollable middle, pinned button
        # ============================================================
        left_outer = ctk.CTkFrame(p)
        left_outer.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        left_outer.grid_columnconfigure(0, weight=1)
        left_outer.grid_rowconfigure(0, weight=0)   # title
        left_outer.grid_rowconfigure(1, weight=1)   # scrollable content
        left_outer.grid_rowconfigure(2, weight=0)   # button — always pinned at bottom

        # Title
        ctk.CTkLabel(
            left_outer, text="Preprocessing Options",
            font=ctk.CTkFont(size=16, weight="bold")
        ).grid(row=0, column=0, pady=(12, 6), padx=10, sticky="w")

        # Scrollable content area — all options live here
        scroll = ctk.CTkScrollableFrame(left_outer, label_text="")
        scroll.grid(row=1, column=0, sticky="nsew", padx=6, pady=0)
        scroll.grid_columnconfigure(1, weight=1)

        # ── Target Columns ──────────────────────────────────────────
        ctk.CTkLabel(scroll, text="Target Columns",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=8, pady=(10, 2), sticky="w")

        self.target_listbox = tk.Listbox(
            scroll,
            selectmode=tk.MULTIPLE,
            height=10,
            relief="flat",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground="#cccccc",
            selectbackground="#3B8ED0",
            selectforeground="white",
            font=("Segoe UI", 11)
        )
        self.target_listbox.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))

        # ── Problem Type ─────────────────────────────────────────────
        ctk.CTkLabel(scroll, text="Problem Type").grid(row=2, column=0, padx=8, sticky="w")
        ctk.CTkOptionMenu(
            scroll, variable=self.problem_type_var,
            values=["Regression", "Classification"]
        ).grid(row=2, column=1, padx=8, sticky="ew", pady=4)

        # ── Outlier Handling ─────────────────────────────────────────
        ctk.CTkLabel(scroll, text="Outlier Handling").grid(row=3, column=0, padx=8, sticky="w")
        ctk.CTkOptionMenu(
            scroll, variable=self.outlier_option,
            values=["None", "IQR", "ZScore", "Isolation Forest"]
        ).grid(row=3, column=1, padx=8, sticky="ew", pady=4)

        # ── Contamination (shown only when Isolation Forest is selected) ──
        self._cont_label = ctk.CTkLabel(scroll, text="Contamination (0–0.5)")
        self._cont_entry = ctk.CTkEntry(scroll, textvariable=self.outlier_contamination, width=80)

        def _toggle_contamination(*_):
            if self.outlier_option.get() == "Isolation Forest":
                self._cont_label.grid(row=4, column=0, padx=8, sticky="w")
                self._cont_entry.grid(row=4, column=1, padx=8, sticky="ew", pady=2)
            else:
                self._cont_label.grid_remove()
                self._cont_entry.grid_remove()

        self.outlier_option.trace_add("write", _toggle_contamination)

        # ── Skip Preprocessing ───────────────────────────────────────
        ctk.CTkCheckBox(
            scroll, text="Skip Preprocessing",
            variable=self.skip_preprocess
        ).grid(row=5, column=0, columnspan=2, padx=8, pady=6, sticky="w")

        # ── Ignore Columns ───────────────────────────────────────────
        ctk.CTkLabel(scroll, text="Ignore Columns",
                     font=ctk.CTkFont(weight="bold")).grid(
            row=6, column=0, columnspan=2, padx=8, pady=(10, 2), sticky="w")

        # inner scrollable frame for column checkboxes
        self.columns_frame = ctk.CTkScrollableFrame(scroll, height=150)
        self.columns_frame.grid(row=7, column=0, columnspan=2, padx=8, pady=(0, 10), sticky="ew")

        ctk.CTkLabel(self.columns_frame,
                     text="Load a dataset to see columns",
                     text_color="gray").pack(anchor="w", padx=4)

        # ── Run Preprocessing button — PINNED at bottom, never hidden ──
        self.run_preprocess_btn = ctk.CTkButton(
            left_outer,
            text="▶  Run Preprocessing",
            height=44,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#2A9D5C",
            hover_color="#1F7A45",
            command=self.start_preprocessing
        )
        self.run_preprocess_btn.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        # ============================================================
        # RIGHT PANEL — log
        # ============================================================
        right = ctk.CTkFrame(p)
        right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(
            right, text="Preprocessing Log",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)

        self.preprocess_log = ctk.CTkTextbox(right)
        self.preprocess_log.pack(fill="both", expand=True, padx=10, pady=10)
        self.preprocess_log.insert("1.0", "Preprocessing output will appear here...\n")

    # =========================
    # PAGE 2 — Run AutoML
    # =========================

    def _build_automl_page(self):

        p = self.page_automl
        p.grid_columnconfigure(0, weight=3)   # settings — wider
        p.grid_columnconfigure(1, weight=2)   # results — narrower
        p.grid_rowconfigure(0, weight=1)

        # ---- left: settings ----
        settings = ctk.CTkFrame(p, fg_color="#FFFFFF")
        settings.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        settings.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            settings, text="AutoML Settings",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="#1A1A1A"
        ).grid(row=0, column=0, columnspan=2, pady=10)

        # ── Target verification (read-only badge, no listbox) ──
        ctk.CTkLabel(settings, text="Target Columns", text_color="#1A1A1A").grid(row=1, column=0, padx=10, sticky="w")
        self.target_verify_label = ctk.CTkLabel(
            settings,
            text="—",
            fg_color="#EAF4FF",
            corner_radius=6,
            wraplength=300,
            justify="left",
            anchor="w",
            padx=8, pady=6
        )
        self.target_verify_label.grid(row=1, column=1, padx=10, pady=6, sticky="ew")

        # ── Problem type ──
        ctk.CTkLabel(settings, text="Problem Type", text_color="#1A1A1A").grid(row=2, column=0, padx=10, sticky="w")
        ctk.CTkOptionMenu(
            settings, variable=self.problem_type_var,
            values=["Regression", "Classification"]
        ).grid(row=2, column=1, padx=10, sticky="ew", pady=4)

        # ── CPU slider with live % label ──
        ctk.CTkLabel(settings, text="CPU Usage", text_color="#1A1A1A").grid(row=3, column=0, padx=10, sticky="w")

        cpu_row = ctk.CTkFrame(settings, fg_color="transparent")
        cpu_row.grid(row=3, column=1, sticky="ew", padx=10, pady=4)
        cpu_row.grid_columnconfigure(0, weight=1)

        self.cpu_slider = ctk.CTkSlider(
            cpu_row, from_=10, to=100, variable=self.cpu_var
        )
        self.cpu_slider.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.cpu_pct_label = ctk.CTkLabel(
            cpu_row, text=f"{self.cpu_var.get()}%",
            width=42, anchor="e",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.cpu_pct_label.grid(row=0, column=1)

        # update label whenever slider moves
        self.cpu_var.trace_add("write", lambda *_: self.cpu_pct_label.configure(
            text=f"{int(self.cpu_var.get())}%"
        ))

        # ── Run AutoML button ──
        self.run_automl_btn = ctk.CTkButton(
            settings,
            text="▶  Run AutoML",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.start_automl
        )
        self.run_automl_btn.grid(row=4, column=0, columnspan=2, padx=10, pady=20, sticky="ew")

        # ── Video animation label (hidden until AutoML runs) ──
        self.video_label = tk.Label(
            settings,
            bg="#FFFFFF",
            bd=0,
            highlightthickness=0
        )
        # row=5, hidden by default — shown when animation starts

        # ---- right: results ----
        results = ctk.CTkFrame(p)
        results.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        ctk.CTkLabel(
            results, text="Results",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)

        self.results_box = ctk.CTkTextbox(results)
        self.results_box.pack(fill="both", expand=True, padx=10, pady=10)
        self.results_box.insert("1.0", "AutoML results will appear here...\n")

    # =========================
    # Sidebar navigation
    # =========================

    def show_preprocess(self):
        self.page_automl.grid_remove()
        self.page_preprocess.grid(row=1, column=0, sticky="nsew", padx=4)
        self.status_label.configure(text="Preprocessing Mode")
        self._highlight_btn(self.btn_preprocess)
        self.terminal.configure(height=80)   # smaller terminal for preprocessing

    def show_automl(self):
        self.page_preprocess.grid_remove()
        self.page_automl.grid(row=1, column=0, sticky="nsew", padx=4)
        self.status_label.configure(text="Run AutoML Mode")
        self._highlight_btn(self.btn_automl)
        self._update_target_verify()
        self.terminal.configure(height=220)  # restore full terminal for automl

    def _highlight_btn(self, active_btn):
        for btn in (self.btn_preprocess, self.btn_automl):
            btn.configure(
                fg_color=["#3B8ED0", "#1F6AA5"] if btn is active_btn else ["#2B2B2B", "#404040"]
            )

    def _update_target_verify(self):
        """Show confirmed targets as a green badge on the AutoML page."""
        if self.selected_targets:
            txt = "  ✅  " + ",  ".join(self.selected_targets)
            self.target_verify_label.configure(text=txt, fg_color="#DFF6E4")
        else:
            self.target_verify_label.configure(
                text="  ⚠  No targets selected — run Preprocessing first",
                fg_color="#FFF3CD"
            )

    # =========================
    # Load dataset
    # =========================

    def browse_file(self):

        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return

        self.selected_file.set(path)

        try:
            self.df = pd.read_csv(path)

            self.target_listbox.delete(0, tk.END)
            for col in self.df.columns:
                self.target_listbox.insert(tk.END, col)

            for widget in self.columns_frame.winfo_children():
                widget.destroy()
            self.column_vars.clear()
            for col in self.df.columns:
                var = tk.BooleanVar()
                self.column_vars[col] = var
                ctk.CTkCheckBox(
                    self.columns_frame, text=col, variable=var
                ).pack(anchor="w", padx=4, pady=1)

            print(f"Loaded dataset: {path}")
            print(f"Shape: {self.df.shape}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # =========================
    # Preprocessing pipeline
    # =========================

    def start_preprocessing(self):

        if not self.selected_file.get():
            messagebox.showerror("Error", "Please select a dataset first.")
            return

        indices = self.target_listbox.curselection()
        if not indices:
            messagebox.showerror("Error", "Please select at least one target column.")
            return

        self.selected_targets = [self.target_listbox.get(i) for i in indices]
        self.run_preprocess_btn.configure(state="disabled", text="Running...")
        threading.Thread(target=self._run_preprocessing, daemon=True).start()

    def _run_preprocessing(self):

        def log(msg):
            self.preprocess_log.insert("end", msg + "\n")
            self.preprocess_log.see("end")

        try:
            log("\n--- Starting Preprocessing ---\n")
            print("\nStarting Preprocessing...\n")

            df = pd.read_csv(self.selected_file.get())

            drop_cols = [c for c, v in self.column_vars.items() if v.get()]
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)
                log(f"Dropped columns: {drop_cols}")
                print(f"Dropping columns: {drop_cols}")

            temp_file = "temp_after_drop.csv"
            df.to_csv(temp_file, index=False)

            if self.skip_preprocess.get():
                self.preprocessed_path = temp_file
                log("Preprocessing skipped. Raw file will be used for AutoML.")
                print("Skipping preprocessing.")
            else:
                output_file = "preprocessed_dataset.csv"
                # Map UI labels → nprepro method keys
                _method_map = {
                    "None": "none",
                    "IQR": "iqr",
                    "ZScore": "zscore",
                    "Isolation Forest": "isolation_forest",
                }
                _method  = _method_map.get(self.outlier_option.get(), "none")
                try:
                    _contam = float(self.outlier_contamination.get())
                    _contam = max(0.001, min(0.5, _contam))   # clamp to valid range
                except ValueError:
                    _contam = 0.05

                label_encode_large_dataset(
                    input_csv=temp_file,
                    target_columns=self.selected_targets,
                    output_csv=output_file,
                    outlier_method=_method,
                    outlier_contamination=_contam,
                    problem_type=self.problem_type_var.get()
                )
                self.preprocessed_path = output_file
                log(f"\n✅ Preprocessing complete → {output_file}")

            log("\n🔀 Redirecting to Run AutoML page...")
            self.after(800, self.show_automl)

        except Exception as e:
            log(f"\n❌ Error: {e}")
            print(f"Preprocessing error: {e}")

        finally:
            self.run_preprocess_btn.configure(state="normal", text="▶  Run Preprocessing")

    # =========================
    # Video animation helpers
    # =========================

    def _start_animation(self):
        """Show the video label and begin looping frames."""
        if not self._video_frames:
            return
        self._video_running  = True
        self._video_frame_idx = 0
        # Place the label below the Run AutoML button (row 5)
        self.video_label.grid(
            in_=self.run_automl_btn.master,
            row=5, column=0, columnspan=2,
            padx=10, pady=(0, 14), sticky="ew"
        )
        self._tick_animation()

    def _tick_animation(self):
        if not self._video_running:
            return
        frame = self._video_frames[self._video_frame_idx]
        self.video_label.configure(image=frame)
        self.video_label.image = frame          # keep reference
        self._video_frame_idx = (self._video_frame_idx + 1) % len(self._video_frames)
        delay = int(1000 / getattr(self, "_video_fps", 30))
        self._video_job = self.after(delay, self._tick_animation)

    def _stop_animation(self):
        """Stop the animation and hide the label."""
        self._video_running = False
        if self._video_job:
            self.after_cancel(self._video_job)
            self._video_job = None
        self.video_label.grid_remove()

    # =========================
    # AutoML pipeline
    # =========================

    def start_automl(self):

        if not self.preprocessed_path:
            messagebox.showerror("Error", "Please run Preprocessing first.")
            return

        if not self.selected_targets:
            messagebox.showerror("Error", "No target columns found. Please run Preprocessing first.")
            return

        self.automl_input = self.preprocessed_path
        self.run_automl_btn.configure(state="disabled", text="Running...")
        self._start_animation()
        threading.Thread(target=self._run_automl, daemon=True).start()

    def _run_automl(self):

        def log(msg):
            self.results_box.insert("end", msg + "\n")
            self.results_box.see("end")

        try:
            self.results_box.delete("1.0", "end")
            log("--- AutoML Started ---")
            log(f"Input  : {self.automl_input}")
            log(f"Target : {self.selected_targets}")
            log(f"Type   : {self.problem_type_var.get()}\n")
            print(f"\nRunning AutoML on: {self.automl_input}\n")

            results = run_basic_automl(
                self.automl_input,
                self.selected_targets,
                self.problem_type_var.get(),
                self.cpu_var.get()
            )

            best_model, all_results, _, _, mae, rmse, r2, *_ = results

            log("\n========== RESULTS ==========\n")
            for name, score in all_results:
                log(f"  {name:<30} {round(score, 4)}")

            log(f"\n🏆 Best Model : {best_model[0]}")

            if self.problem_type_var.get() == "Regression":
                log(f"   MAE   : {mae, 4}")
                log(f"   RMSE  : {rmse, 4}")
                log(f"   R²    : {r2, 4}")

            print("\n========== RESULTS ==========")
            for name, score in all_results:
                print(f"{name:<25} {score}")
            print(f"\nBest Model: {best_model[0]}")
            if self.problem_type_var.get() == "Regression":
                print(f"MAE: {mae}  RMSE: {rmse}  R²: {r2}")

        except Exception as e:
            log(f"\n❌ Error: {e}")
            print(f"AutoML error: {e}")

        finally:
            self.run_automl_btn.configure(state="normal", text="▶  Run AutoML")
            self.after(0, self._stop_animation)


# =========================

if __name__ == "__main__":
    app = AutoMLApp()
    app.mainloop()