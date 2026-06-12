from __future__ import annotations

import ctypes
import tkinter as tk
from tkinter import ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .config import BARBELL_SPEED_CHART, CHART_OPTIONS, CHART_POINTS


class ChartWindow(tk.Toplevel):
    def __init__(self, master: tk.Misc, points_path: str, selected_point: str, palette: dict[str, str] | None = None) -> None:
        super().__init__(master)
        self.title("Time Charts")
        self.geometry("1000x700")
        self.minsize(860, 620)
        self.overrideredirect(True)
        self.configure(bg="#0a0d14")
        self.points_path = points_path
        default_chart = selected_point if selected_point in CHART_OPTIONS else CHART_OPTIONS[0]
        self.point_var = tk.StringVar(value=default_chart)
        self._palette = dict(palette or getattr(master, "_palette", {}))
        self._palette.setdefault("bg_root", "#0a0d14")
        self._palette.setdefault("bg_panel", "#121722")
        self._palette.setdefault("bg_panel_alt", "#171d2a")
        self._palette.setdefault("bg_input", "#20283a")
        self._palette.setdefault("bg_hover", "#2a354a")
        self._palette.setdefault("bg_titlebar", "#0c1018")
        self._palette.setdefault("fg_primary", "#edf2ff")
        self._palette.setdefault("fg_muted", "#8d98b3")
        self._palette.setdefault("accent", "#7aa2ff")
        self._palette.setdefault("accent_active", "#a9c0ff")
        self._palette.setdefault("border", "#283247")
        self._palette.setdefault("danger", "#d65a7a")
        self._palette.setdefault("danger_active", "#ea7e9b")
        self._is_maximized = False
        self._window_drag_origin: tuple[int, int] | None = None

        self._build_layout()
        self.after_idle(self._apply_window_chrome)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Map>", self._on_window_map)

    def _build_layout(self) -> None:
        outer = tk.Frame(self, bg=self._palette["border"], highlightthickness=0, bd=0)
        outer.pack(fill="both", expand=True)

        self._build_titlebar(outer)

        root = tk.Frame(outer, bg=self._palette["bg_root"], highlightthickness=0, bd=0)
        root.pack(fill="both", expand=True, padx=1, pady=(0, 1))

        shell = tk.Frame(root, bg=self._palette["border"], highlightthickness=0, bd=0)
        shell.pack(fill="both", expand=True, padx=12, pady=12)

        content = tk.Frame(shell, bg=self._palette["bg_panel"], highlightthickness=0, bd=0)
        content.pack(fill="both", expand=True, padx=1, pady=1)

        header = tk.Frame(content, bg=self._palette["bg_panel"], highlightthickness=0, bd=0)
        header.pack(fill="x", padx=16, pady=(16, 10))

        tk.Label(
            header,
            text="Time Charts",
            bg=self._palette["bg_panel"],
            fg=self._palette["fg_primary"],
            font=("Segoe UI Semibold", 11),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Tracked point movement across the recorded session",
            bg=self._palette["bg_panel"],
            fg=self._palette["fg_muted"],
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 0))

        controls_shell = tk.Frame(content, bg=self._palette["border"], highlightthickness=0, bd=0)
        controls_shell.pack(fill="x", padx=16, pady=(0, 12))
        controls = tk.Frame(controls_shell, bg=self._palette["bg_panel_alt"], highlightthickness=0, bd=0)
        controls.pack(fill="x", padx=1, pady=1)

        tk.Label(
            controls,
            text="Chart",
            bg=self._palette["bg_panel_alt"],
            fg=self._palette["fg_muted"],
            font=("Segoe UI Semibold", 9),
        ).pack(side="left", padx=(14, 8), pady=12)

        point_box = ttk.Combobox(controls, textvariable=self.point_var, values=CHART_OPTIONS, state="readonly", width=22)
        point_box.pack(side="left", padx=(0, 12), pady=10)
        point_box.bind("<<ComboboxSelected>>", lambda _event: self.render_chart())
        ttk.Button(controls, text="Refresh", command=self.render_chart).pack(side="left", pady=10)

        figure_shell = tk.Frame(content, bg=self._palette["border"], highlightthickness=0, bd=0)
        figure_shell.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.figure_host = tk.Frame(figure_shell, bg=self._palette["bg_panel_alt"], highlightthickness=0, bd=0)
        self.figure_host.pack(fill="both", expand=True, padx=1, pady=1)

        self.figure = Figure(figsize=(9, 6), dpi=100, facecolor=self._palette["bg_panel_alt"])
        self.axis = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.figure_host)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        self.render_chart()

    def _build_titlebar(self, parent: tk.Misc) -> None:
        titlebar = tk.Frame(parent, bg=self._palette["bg_titlebar"], height=56, highlightthickness=0, bd=0)
        titlebar.pack(fill="x", padx=1, pady=1)
        titlebar.pack_propagate(False)

        top_rule = tk.Frame(titlebar, bg=self._palette["accent"], height=1, highlightthickness=0, bd=0)
        top_rule.pack(side="top", fill="x")

        left = tk.Frame(titlebar, bg=self._palette["bg_titlebar"], highlightthickness=0, bd=0)
        left.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(2, 0))

        brand = tk.Frame(left, bg=self._palette["bg_panel_alt"], highlightthickness=1, highlightbackground=self._palette["border"], bd=0)
        brand.pack(side="left", pady=10)
        tk.Label(
            brand,
            text="CH",
            bg=self._palette["bg_input"],
            fg=self._palette["accent_active"],
            font=("Segoe UI Semibold", 9),
            width=3,
            pady=8,
        ).pack(side="left", padx=(10, 10), pady=8)
        text_wrap = tk.Frame(brand, bg=self._palette["bg_panel_alt"], highlightthickness=0, bd=0)
        text_wrap.pack(side="left", padx=(0, 16), pady=8)
        title_label = tk.Label(
            text_wrap,
            text="Chart Viewer",
            bg=self._palette["bg_panel_alt"],
            fg=self._palette["fg_primary"],
            font=("Segoe UI Semibold", 10),
        )
        title_label.pack(anchor="w")
        subtitle_label = tk.Label(
            text_wrap,
            text="point trajectories and time series",
            bg=self._palette["bg_panel_alt"],
            fg=self._palette["fg_muted"],
            font=("Segoe UI", 8),
        )
        subtitle_label.pack(anchor="w")

        controls = tk.Frame(titlebar, bg=self._palette["bg_panel_alt"], highlightthickness=1, highlightbackground=self._palette["border"], bd=0)
        controls.pack(side="right", fill="y", padx=(0, 12), pady=10)
        close_button = self._create_titlebar_button(
            controls,
            "✕",
            self._on_close,
            normal_bg=self._palette["bg_panel_alt"],
            hover_bg=self._palette["danger"],
            active_bg=self._palette["danger_active"],
        )
        close_button.pack(side="right", fill="y")
        self.maximize_button = self._create_titlebar_button(
            controls,
            "□",
            self._toggle_maximize,
            normal_bg=self._palette["bg_panel_alt"],
        )
        self.maximize_button.pack(side="right", fill="y")
        minimize_button = self._create_titlebar_button(
            controls,
            "_",
            self._minimize_window,
            normal_bg=self._palette["bg_panel_alt"],
        )
        minimize_button.pack(side="right", fill="y")

        drag_targets = (titlebar, left, brand, text_wrap, title_label, subtitle_label)
        for widget in drag_targets:
            widget.bind("<ButtonPress-1>", self._start_window_drag)
            widget.bind("<B1-Motion>", self._drag_window)
            widget.bind("<Double-Button-1>", lambda _event: self._toggle_maximize())

    def _create_titlebar_button(
        self,
        parent: tk.Misc,
        text: str,
        command,
        normal_bg: str,
        hover_bg: str | None = None,
        active_bg: str | None = None,
    ) -> tk.Label:
        button = tk.Label(
            parent,
            text=text,
            bg=normal_bg,
            fg=self._palette["fg_primary"],
            font=("Segoe UI Symbol", 10),
            width=4,
            padx=1,
            pady=1,
            cursor="hand2",
        )
        hover_color = hover_bg or self._palette["bg_hover"]
        active_color = active_bg or hover_color
        button.bind("<Enter>", lambda _event: button.configure(bg=hover_color))
        button.bind("<Leave>", lambda _event: button.configure(bg=normal_bg))
        button.bind("<ButtonPress-1>", lambda _event: button.configure(bg=active_color))
        button.bind("<ButtonRelease-1>", lambda _event: button.configure(bg=hover_color))
        button.bind("<Button-1>", lambda _event: command())
        return button

    def _start_window_drag(self, event) -> None:
        if self._is_maximized:
            return
        self._window_drag_origin = (event.x_root - self.winfo_x(), event.y_root - self.winfo_y())

    def _drag_window(self, event) -> None:
        if self._is_maximized or self._window_drag_origin is None:
            return
        offset_x, offset_y = self._window_drag_origin
        self.geometry(f"+{event.x_root - offset_x}+{event.y_root - offset_y}")

    def _toggle_maximize(self) -> None:
        if self._is_maximized:
            self.state("normal")
            self._is_maximized = False
            self.maximize_button.configure(text="□")
            return

        self.state("zoomed")
        self._is_maximized = True
        self.maximize_button.configure(text="❐")

    def _minimize_window(self) -> None:
        self.overrideredirect(False)
        self.iconify()

    def _apply_window_chrome(self) -> None:
        if self.tk.call("tk", "windowingsystem") != "win32":
            return

        try:
            hwnd = ctypes.c_void_p(self.winfo_id())
            dwmapi = ctypes.windll.dwmapi
            corner_preference = ctypes.c_int(2)
            dark_mode = ctypes.c_int(1)
            border_color = ctypes.c_int(0x00283247)

            dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(dark_mode), ctypes.sizeof(dark_mode))
            dwmapi.DwmSetWindowAttribute(hwnd, 33, ctypes.byref(corner_preference), ctypes.sizeof(corner_preference))
            dwmapi.DwmSetWindowAttribute(hwnd, 34, ctypes.byref(border_color), ctypes.sizeof(border_color))
        except Exception:
            return

    def _on_window_map(self, _event) -> None:
        if str(self.state()) != "iconic":
            self.overrideredirect(True)
            self.after_idle(self._apply_window_chrome)

    def _on_close(self) -> None:
        self.destroy()

    def render_chart(self) -> None:
        dataframe = pd.read_json(self.points_path, lines=True) if self.points_path else pd.DataFrame()
        point_name = self.point_var.get()

        self.axis.clear()
        self.figure.patch.set_facecolor(self._palette["bg_panel_alt"])
        self.axis.set_facecolor(self._palette["bg_panel_alt"])
        self.axis.tick_params(colors=self._palette["fg_muted"])
        for spine in self.axis.spines.values():
            spine.set_color(self._palette["border"])
        if point_name == BARBELL_SPEED_CHART:
            self._render_barbell_speed_chart(dataframe)
            self.canvas.draw_idle()
            return

        self._render_point_chart(dataframe, point_name)
        self.canvas.draw_idle()

    def _render_point_chart(self, dataframe: pd.DataFrame, point_name: str) -> None:
        x_column = f"{point_name}_x"
        y_column = f"{point_name}_y"
        if dataframe.empty or x_column not in dataframe.columns or y_column not in dataframe.columns:
            self.axis.set_title("No data for the selected point", color=self._palette["fg_primary"])
            return

        plot_frame = self._prepare_point_chart_frame(dataframe, point_name)
        if plot_frame.empty or plot_frame[x_column].dropna().empty or plot_frame[y_column].dropna().empty:
            self.axis.set_title("No data for the selected point", color=self._palette["fg_primary"])
            return

        self._annotate_rep_intervals(dataframe)
        self.axis.plot(plot_frame["time_s"], plot_frame[x_column], label="x", color="#7aa2ff", linewidth=2.2)
        self.axis.plot(plot_frame["time_s"], plot_frame[y_column], label="y", color="#86e0c2", linewidth=2.2)
        self.axis.set_xlabel("Time [s]", color=self._palette["fg_muted"])
        self.axis.set_ylabel("Position [px]", color=self._palette["fg_muted"])
        self.axis.set_title(f"Point time series: {point_name}", color=self._palette["fg_primary"])
        legend = self.axis.legend(facecolor=self._palette["bg_panel"], edgecolor=self._palette["border"])
        for text in legend.get_texts():
            text.set_color(self._palette["fg_primary"])
        self.axis.grid(True, alpha=0.18, color=self._palette["accent"])

    def _render_barbell_speed_chart(self, dataframe: pd.DataFrame) -> None:
        required_columns = {"time_s", "barbell_center_x", "barbell_center_y"}
        if dataframe.empty or not required_columns.issubset(dataframe.columns):
            self.axis.set_title("No data for barbell speed", color=self._palette["fg_primary"])
            return

        speed_frame = self._prepare_point_chart_frame(dataframe, "barbell_center")
        speed_frame = speed_frame[["time_s", "barbell_center_x", "barbell_center_y"]].dropna().copy()
        if len(speed_frame) < 2:
            self.axis.set_title("Not enough data to compute barbell speed", color=self._palette["fg_primary"])
            return

        delta_time = speed_frame["time_s"].diff()
        delta_x = speed_frame["barbell_center_x"].diff()
        delta_y = speed_frame["barbell_center_y"].diff()
        speed = ((delta_x.pow(2) + delta_y.pow(2)).pow(0.5) / delta_time.replace(0, pd.NA)).dropna()
        if speed.empty:
            self.axis.set_title("Not enough data to compute barbell speed", color=self._palette["fg_primary"])
            return

        speed_times = speed_frame.loc[speed.index, "time_s"]
        self._annotate_rep_intervals(dataframe)
        self.axis.plot(speed_times, speed, label="speed", color="#ffb86b", linewidth=2.4)
        self.axis.set_xlabel("Time [s]", color=self._palette["fg_muted"])
        self.axis.set_ylabel("Speed [px/s]", color=self._palette["fg_muted"])
        self.axis.set_title("Barbell speed over time", color=self._palette["fg_primary"])
        legend = self.axis.legend(facecolor=self._palette["bg_panel"], edgecolor=self._palette["border"])
        for text in legend.get_texts():
            text.set_color(self._palette["fg_primary"])
        self.axis.grid(True, alpha=0.18, color=self._palette["accent"])

    def _prepare_point_chart_frame(self, dataframe: pd.DataFrame, point_name: str) -> pd.DataFrame:
        x_column = f"{point_name}_x"
        y_column = f"{point_name}_y"
        if dataframe.empty or x_column not in dataframe.columns or y_column not in dataframe.columns:
            return pd.DataFrame()

        plot_frame = dataframe[["time_s", x_column, y_column]].copy()
        plot_frame = plot_frame.sort_values("time_s").drop_duplicates(subset=["time_s"], keep="last")
        if point_name == "barbell_center":
            plot_frame[[x_column, y_column]] = plot_frame[[x_column, y_column]].interpolate(
                method="linear",
                limit_area="inside",
            )
        return plot_frame

    def _annotate_rep_intervals(self, dataframe: pd.DataFrame) -> None:
        required_columns = {"time_s", "reps_completed"}
        if dataframe.empty or not required_columns.issubset(dataframe.columns):
            return

        rep_frame = dataframe[["time_s", "reps_completed"]].dropna().copy()
        if rep_frame.empty:
            return

        rep_frame = rep_frame.sort_values("time_s").drop_duplicates(subset=["time_s"], keep="last")
        rep_frame["reps_completed"] = pd.to_numeric(rep_frame["reps_completed"], errors="coerce")
        rep_frame = rep_frame.dropna(subset=["reps_completed"])
        if rep_frame.empty:
            return

        rep_frame["reps_completed"] = rep_frame["reps_completed"].astype(int)
        rep_values = rep_frame["reps_completed"].tolist()
        time_values = rep_frame["time_s"].tolist()
        change_indices = [0]
        for index in range(1, len(rep_values)):
            if rep_values[index] != rep_values[index - 1]:
                change_indices.append(index)

        interval_colors = ["#1b2a48", "#173326"]
        top_y = 0.97
        for segment_index, start_index in enumerate(change_indices):
            end_index = change_indices[segment_index + 1] if segment_index + 1 < len(change_indices) else len(time_values)
            start_time = float(time_values[start_index])
            end_time = float(time_values[end_index - 1])
            if end_index < len(time_values):
                end_time = float(time_values[end_index])
            if end_time <= start_time:
                continue

            rep_number = rep_values[start_index] + 1
            color = interval_colors[segment_index % len(interval_colors)]
            self.axis.axvspan(start_time, end_time, color=color, alpha=0.16, zorder=0)
            self.axis.axvline(end_time, color=self._palette["border"], alpha=0.45, linewidth=1.0, zorder=1)
            self.axis.text(
                (start_time + end_time) / 2.0,
                top_y,
                f"Rep {rep_number}",
                transform=self.axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
                color=self._palette["fg_primary"],
                bbox={"boxstyle": "round,pad=0.22", "facecolor": self._palette["bg_panel"], "edgecolor": self._palette["border"], "alpha": 0.82},
                zorder=2,
            )
