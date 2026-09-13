from __future__ import annotations

import ctypes
import math
import tkinter as tk
from tkinter import ttk

import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .config import BARBELL_POSITION_CHART, BARBELL_SPEED_CHART, CHART_OPTIONS


VIEW_MODE_TIME = "Time charts"
VIEW_MODE_SKELETON = "Skeleton overview"

FRONT_POINT_NAMES = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

FRONT_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

SPINAL_POINTS_9 = [
    "neck_start",
    "neck_end",
    "thoracic_start",
    "thoracic_1",
    "thoracic_2",
    "thoracic_end",
    "lumbar_start",
    "lumbar_1",
    "lumbar_end",
]

SPINAL_POINTS_4 = [
    "neck_end",
    "lumbar_start",
    "lumbar_end",
]

SELECTABLE_POINT_NAMES = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "neck_start",
    "neck_end",
    "thoracic_start",
    "thoracic_1",
    "thoracic_2",
    "thoracic_end",
    "lumbar_start",
    "lumbar_1",
    "lumbar_end",
]

FRONT_SELECTOR_LAYOUT = {
    "nose": (70, 18),
    "left_shoulder": (45, 40),
    "right_shoulder": (95, 40),
    "left_elbow": (35, 65),
    "right_elbow": (105, 65),
    "left_wrist": (30, 92),
    "right_wrist": (110, 92),
    "left_hip": (55, 82),
    "right_hip": (85, 82),
    "left_knee": (55, 114),
    "right_knee": (85, 114),
    "left_ankle": (55, 145),
    "right_ankle": (85, 145),
}

SPINAL_SELECTOR_LAYOUT = {
    "neck_start": (70, 20),
    "neck_end": (70, 35),
    "thoracic_start": (70, 52),
    "thoracic_1": (70, 68),
    "thoracic_2": (70, 84),
    "thoracic_end": (70, 100),
    "lumbar_start": (70, 116),
    "lumbar_1": (70, 132),
    "lumbar_end": (70, 148),
}


class ChartWindow(tk.Toplevel):
    def __init__(self, master: tk.Misc, points_path: str, selected_point: str, palette: dict[str, str] | None = None) -> None:
        super().__init__(master)
        self.title("Time Charts")
        self.geometry("1180x820")
        self.minsize(900, 660)
        self.overrideredirect(True)
        self.configure(bg="#0a0d14")
        self.points_path = points_path
        default_chart = selected_point if selected_point in CHART_OPTIONS else CHART_OPTIONS[0]
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
        self._cursor_time_s: float | None = None
        self._cursor_lines: list = []
        self._skeleton_click_targets: list[dict[str, object]] = []
        self.view_mode_var = tk.StringVar(value=VIEW_MODE_TIME)
        self.show_barbell_var = tk.BooleanVar(value=True)
        self.show_barbell_velocity_var = tk.BooleanVar(value=True)
        self.include_barbell_chart_var = tk.BooleanVar(value=True)
        self.include_barbell_velocity_chart_var = tk.BooleanVar(value=True)
        self._point_selector_expanded = True

        self._selected_points_set: set[str] = {default_chart} if default_chart in SELECTABLE_POINT_NAMES else {"nose"}
        self._selected_points: list[str] = [default_chart]
        self._two_column_threshold = 10
        self._build_layout(default_chart)
        self.after_idle(self._apply_window_chrome)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Map>", self._on_window_map)
        self.transient(master)
        self.lift(master)
        try:
            self.attributes("-topmost", True)
        except tk.TclError:
            pass
        self.focus_force()

    def _build_layout(self, default_chart: str) -> None:
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
            text="Select many charts and click a moment in time to align all plots with one marker.",
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
            text="View mode",
            bg=self._palette["bg_panel_alt"],
            fg=self._palette["fg_muted"],
            font=("Segoe UI Semibold", 9),
        ).grid(row=0, column=0, sticky="w", padx=(14, 8), pady=(12, 6))
        view_box = ttk.Combobox(
            controls,
            textvariable=self.view_mode_var,
            values=[VIEW_MODE_TIME, VIEW_MODE_SKELETON],
            state="readonly",
            width=28,
        )
        view_box.grid(row=0, column=1, sticky="w", padx=(0, 10), pady=(10, 6))
        view_box.bind("<<ComboboxSelected>>", self._on_view_mode_changed)

        self.chart_label = tk.Label(
            controls,
            text="Select points on skeleton",
            bg=self._palette["bg_panel_alt"],
            fg=self._palette["fg_muted"],
            font=("Segoe UI Semibold", 9),
        )
        self.chart_label.grid(row=1, column=0, sticky="nw", padx=(14, 8), pady=(4, 4))

        self.chart_list_shell = tk.Frame(controls, bg=self._palette["border"], highlightthickness=0, bd=0)
        self.chart_list_shell.grid(row=1, column=1, rowspan=3, sticky="ew", padx=(0, 10), pady=4)
        selector_host = tk.Frame(self.chart_list_shell, bg=self._palette["bg_input"], highlightthickness=0, bd=0)
        selector_host.pack(fill="x", padx=1, pady=1)
        selector_host.columnconfigure(0, weight=1)
        selector_host.columnconfigure(1, weight=1)

        self.front_selector_canvas = tk.Canvas(
            selector_host,
            width=160,
            height=176,
            bg=self._palette["bg_input"],
            highlightthickness=0,
            bd=0,
        )
        self.front_selector_canvas.grid(row=0, column=0, padx=(10, 8), pady=8, sticky="nsew")

        self.spinal_selector_canvas = tk.Canvas(
            selector_host,
            width=160,
            height=176,
            bg=self._palette["bg_input"],
            highlightthickness=0,
            bd=0,
        )
        self.spinal_selector_canvas.grid(row=0, column=1, padx=(8, 10), pady=8, sticky="nsew")

        barbell_options = tk.Frame(selector_host, bg=self._palette["bg_input"], highlightthickness=0, bd=0)
        barbell_options.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 8))
        ttk.Checkbutton(
            barbell_options,
            text="barbell",
            variable=self.include_barbell_chart_var,
            command=self.render_charts,
        ).pack(side="left", padx=(0, 18))
        ttk.Checkbutton(
            barbell_options,
            text="barbell velocity",
            variable=self.include_barbell_velocity_chart_var,
            command=self.render_charts,
        ).pack(side="left")

        self.chart_buttons = tk.Frame(controls, bg=self._palette["bg_panel_alt"], highlightthickness=0, bd=0)
        self.chart_buttons.grid(row=1, column=2, rowspan=3, sticky="ns", padx=(0, 12), pady=4)
        self.toggle_selector_button = ttk.Button(self.chart_buttons, text="Collapse", command=self._toggle_point_selector_panel)
        self.toggle_selector_button.pack(fill="x", pady=(0, 6))
        ttk.Button(self.chart_buttons, text="Apply", command=self.render_charts).pack(fill="x", pady=(0, 6))
        ttk.Button(self.chart_buttons, text="Select all", command=self._select_all_charts).pack(fill="x", pady=(0, 6))
        ttk.Button(self.chart_buttons, text="Clear selection", command=self._clear_point_selection).pack(fill="x", pady=(0, 6))
        ttk.Button(self.chart_buttons, text="Clear marker", command=self._clear_marker).pack(fill="x", pady=(0, 0))

        self.chart_hint = tk.Label(
            controls,
            text="Click any plot to place a vertical marker on all plots.",
            bg=self._palette["bg_panel_alt"],
            fg=self._palette["fg_muted"],
            font=("Segoe UI", 9),
            anchor="w",
        )
        self.chart_hint.grid(row=4, column=0, columnspan=3, sticky="ew", padx=14, pady=(4, 10))

        self.skeleton_options = tk.Frame(controls, bg=self._palette["bg_panel_alt"], highlightthickness=0, bd=0)
        self.skeleton_options.grid(row=1, column=0, columnspan=3, sticky="w", padx=14, pady=(8, 10))
        ttk.Checkbutton(
            self.skeleton_options,
            text="barbell",
            variable=self.show_barbell_var,
            command=self.render_charts,
        ).pack(side="left", padx=(0, 18))
        ttk.Checkbutton(
            self.skeleton_options,
            text="barbell velocity",
            variable=self.show_barbell_velocity_var,
            command=self.render_charts,
        ).pack(side="left")

        controls.columnconfigure(1, weight=1)
        self._draw_selection_skeletons()
        self._on_view_mode_changed()

        figure_shell = tk.Frame(content, bg=self._palette["border"], highlightthickness=0, bd=0)
        figure_shell.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self.figure_host = tk.Frame(figure_shell, bg=self._palette["bg_panel_alt"], highlightthickness=0, bd=0)
        self.figure_host.pack(fill="both", expand=True, padx=1, pady=1)

        self.figure = Figure(figsize=(10, 7), dpi=100, facecolor=self._palette["bg_panel_alt"])
        self.figure_scroll_canvas = tk.Canvas(
            self.figure_host,
            bg=self._palette["bg_panel_alt"],
            highlightthickness=0,
            bd=0,
            yscrollincrement=24,
        )
        self.figure_scroll_canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        figure_scrollbar = ttk.Scrollbar(self.figure_host, orient="vertical", command=self.figure_scroll_canvas.yview)
        figure_scrollbar.pack(side="right", fill="y", padx=(0, 10), pady=10)
        self.figure_scroll_canvas.configure(yscrollcommand=figure_scrollbar.set)

        self.figure_canvas_host = tk.Frame(self.figure_scroll_canvas, bg=self._palette["bg_panel_alt"], highlightthickness=0, bd=0)
        self.figure_canvas_window = self.figure_scroll_canvas.create_window((0, 0), window=self.figure_canvas_host, anchor="nw")
        self.figure_scroll_canvas.bind("<Configure>", self._on_figure_viewport_configure)
        self.figure_canvas_host.bind("<Configure>", self._on_figure_content_configure)

        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=self.figure_canvas_host)
        self.figure_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.figure_canvas.mpl_connect("button_press_event", self._on_chart_click)

        self.figure_scroll_canvas.bind("<Enter>", self._bind_figure_mousewheel)
        self.figure_scroll_canvas.bind("<Leave>", self._unbind_figure_mousewheel)
        self.render_charts()

    def _on_figure_viewport_configure(self, event) -> None:
        self.figure_scroll_canvas.itemconfigure(self.figure_canvas_window, width=event.width)

    def _on_figure_content_configure(self, _event=None) -> None:
        self.figure_scroll_canvas.configure(scrollregion=self.figure_scroll_canvas.bbox("all"))

    def _on_figure_mousewheel(self, event) -> None:
        self.figure_scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_figure_mousewheel(self, _event=None) -> None:
        self.figure_scroll_canvas.bind_all("<MouseWheel>", self._on_figure_mousewheel)

    def _unbind_figure_mousewheel(self, _event=None) -> None:
        self.figure_scroll_canvas.unbind_all("<MouseWheel>")

    def _on_view_mode_changed(self, _event=None) -> None:
        is_time_mode = self.view_mode_var.get() == VIEW_MODE_TIME
        if is_time_mode:
            self.chart_label.grid()
            if self._point_selector_expanded:
                self.chart_list_shell.grid()
                self.toggle_selector_button.configure(text="Collapse")
            else:
                self.chart_list_shell.grid_remove()
                self.toggle_selector_button.configure(text="Expand")
            self.chart_buttons.grid()
            self.chart_hint.grid()
            self.skeleton_options.grid_remove()
        else:
            self.chart_label.grid_remove()
            self.chart_list_shell.grid_remove()
            self.chart_buttons.grid_remove()
            self.chart_hint.grid_remove()
            self.skeleton_options.grid()
            self._cursor_time_s = None
        if not hasattr(self, "figure_canvas"):
            return
        self.render_charts()

    def _draw_selection_skeletons(self) -> None:
        self._draw_selector_canvas(
            canvas=self.front_selector_canvas,
            title="Front",
            point_layout=FRONT_SELECTOR_LAYOUT,
            connections=FRONT_CONNECTIONS,
        )
        spinal_connections = [(SPINAL_POINTS_9[index], SPINAL_POINTS_9[index + 1]) for index in range(len(SPINAL_POINTS_9) - 1)]
        self._draw_selector_canvas(
            canvas=self.spinal_selector_canvas,
            title="Spinal",
            point_layout=SPINAL_SELECTOR_LAYOUT,
            connections=spinal_connections,
        )

    def _toggle_point_selector_panel(self) -> None:
        self._point_selector_expanded = not self._point_selector_expanded
        if self._point_selector_expanded:
            self.chart_list_shell.grid()
            self.toggle_selector_button.configure(text="Collapse")
        else:
            self.chart_list_shell.grid_remove()
            self.toggle_selector_button.configure(text="Expand")

    def _draw_selector_canvas(
        self,
        *,
        canvas: tk.Canvas,
        title: str,
        point_layout: dict[str, tuple[int, int]],
        connections: list[tuple[str, str]],
    ) -> None:
        canvas.delete("all")
        canvas.create_text(80, 10, text=title, fill=self._palette["fg_muted"], font=("Segoe UI Semibold", 9))

        for start_name, end_name in connections:
            if start_name not in point_layout or end_name not in point_layout:
                continue
            start_x, start_y = point_layout[start_name]
            end_x, end_y = point_layout[end_name]
            canvas.create_line(start_x, start_y + 14, end_x, end_y + 14, fill="#5f789c", width=2)

        for point_name, (point_x, point_y) in point_layout.items():
            selected = point_name in self._selected_points_set
            fill = self._palette["accent"] if selected else "#425573"
            outline = "#b8d2ff" if selected else "#6a7e9a"
            y_offset = point_y + 14
            canvas.create_oval(point_x - 5, y_offset - 5, point_x + 5, y_offset + 5, fill=fill, outline=outline, width=1.5)
            canvas.create_text(point_x + 8, y_offset - 7, text=point_name, fill=self._palette["fg_primary"], anchor="w", font=("Segoe UI", 7))
            tag = f"pt::{point_name}"
            canvas.create_rectangle(point_x - 9, y_offset - 9, point_x + 9, y_offset + 9, outline="", fill="", tags=(tag, "point-hitbox"))
            canvas.tag_bind(tag, "<Button-1>", lambda _event, p=point_name: self._toggle_selected_point(p))

    def _toggle_selected_point(self, point_name: str) -> None:
        if point_name in self._selected_points_set:
            self._selected_points_set.remove(point_name)
        else:
            self._selected_points_set.add(point_name)
        self._draw_selection_skeletons()

    def _resolve_selected_chart_names(self) -> list[str]:
        selected: list[str] = [name for name in CHART_OPTIONS if name in self._selected_points_set]
        if self.include_barbell_chart_var.get():
            selected.append(BARBELL_POSITION_CHART)
        if self.include_barbell_velocity_chart_var.get():
            selected.append(BARBELL_SPEED_CHART)

        deduped: list[str] = []
        for name in selected:
            if name not in deduped:
                deduped.append(name)
        return deduped

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

    def _get_selected_chart_names(self) -> list[str]:
        return self._resolve_selected_chart_names()

    def _select_all_charts(self) -> None:
        self._selected_points_set = {name for name in SELECTABLE_POINT_NAMES}
        self.include_barbell_chart_var.set(True)
        self.include_barbell_velocity_chart_var.set(True)
        self._draw_selection_skeletons()
        self.render_charts()

    def _clear_point_selection(self) -> None:
        self._selected_points_set.clear()
        self.include_barbell_chart_var.set(False)
        self.include_barbell_velocity_chart_var.set(False)
        self._draw_selection_skeletons()
        self.render_charts()

    def _clear_marker(self) -> None:
        self._cursor_time_s = None
        self.render_charts()

    def _on_chart_click(self, event) -> None:
        if self.view_mode_var.get() == VIEW_MODE_SKELETON:
            self._handle_skeleton_click(event)
            return
        if self.view_mode_var.get() != VIEW_MODE_TIME:
            return
        if event.inaxes is None or event.xdata is None:
            return
        self._cursor_time_s = float(event.xdata)
        self._draw_cursor_lines()
        self.figure_canvas.draw_idle()

    def _handle_skeleton_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if not self._skeleton_click_targets:
            return

        nearest_target: dict[str, object] | None = None
        nearest_distance = float("inf")
        for target in self._skeleton_click_targets:
            target_axis = target.get("axis")
            if target_axis is not event.inaxes:
                continue
            point_x = float(target["x"])
            point_y = float(target["y"])
            distance = math.sqrt((float(event.xdata) - point_x) ** 2 + (float(event.ydata) - point_y) ** 2)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_target = target

        if nearest_target is None:
            return

        axis = event.inaxes
        x_limits = axis.get_xlim()
        y_limits = axis.get_ylim()
        axis_scale = max(abs(float(x_limits[1] - x_limits[0])), abs(float(y_limits[1] - y_limits[0])))
        click_threshold = max(8.0, axis_scale * 0.035)
        if nearest_distance > click_threshold:
            return

        point_name = str(nearest_target["point_name"])
        self._show_point_trajectory_from_skeleton(point_name)

    def _show_point_trajectory_from_skeleton(self, point_name: str) -> None:
        if point_name not in CHART_OPTIONS:
            return

        self.view_mode_var.set(VIEW_MODE_TIME)
        self._on_view_mode_changed()
        self._selected_points_set = {point_name}
        self.include_barbell_chart_var.set(False)
        self.include_barbell_velocity_chart_var.set(False)
        self._draw_selection_skeletons()
        self._selected_points = [point_name]
        self._cursor_time_s = None
        self.render_charts()

    def render_charts(self) -> None:
        dataframe = pd.read_json(self.points_path, lines=True) if self.points_path else pd.DataFrame()
        if self.view_mode_var.get() == VIEW_MODE_SKELETON:
            self._render_skeleton_overview(dataframe)
            return

        self._draw_selection_skeletons()
        chart_names = self._resolve_selected_chart_names()
        if chart_names:
            self._selected_points = chart_names
        elif not self._selected_points:
            self._selected_points = [CHART_OPTIONS[0]]

        self.figure.clear()
        self.figure.patch.set_facecolor(self._palette["bg_panel_alt"])

        if not self._selected_points:
            axis = self.figure.add_subplot(111)
            axis.set_facecolor(self._palette["bg_panel_alt"])
            axis.set_title("Select at least one chart", color=self._palette["fg_primary"])
            axis.tick_params(colors=self._palette["fg_muted"])
            self.figure_canvas.draw_idle()
            return

        chart_count = len(self._selected_points)
        column_count = 2 if chart_count >= self._two_column_threshold else 1
        row_count = int(math.ceil(chart_count / column_count))

        height_per_row = 2.6 if column_count == 1 else 2.4
        figure_height_inches = max(6.5, height_per_row * row_count)
        self.figure.set_size_inches(10, figure_height_inches, forward=True)
        pixel_height = int(figure_height_inches * self.figure.dpi)
        self.figure_canvas.get_tk_widget().configure(height=max(620, pixel_height))

        axes = self.figure.subplots(row_count, column_count, sharex=False)
        if isinstance(axes, (list, tuple)):
            axis_list = list(axes)
        else:
            axis_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

        for axis_index, axis in enumerate(axis_list):
            if axis_index >= chart_count:
                axis.set_visible(False)
                continue

            point_name = self._selected_points[axis_index]
            self._style_axis(axis)
            if point_name == BARBELL_SPEED_CHART:
                self._render_barbell_speed_chart(axis, dataframe)
            elif point_name == BARBELL_POSITION_CHART:
                self._render_barbell_position_chart(axis, dataframe)
            else:
                self._render_point_chart(axis, dataframe, point_name)
            axis.set_xlabel("Time [s]", color=self._palette["fg_muted"])

        self.figure.tight_layout(pad=2.0, h_pad=1.0)
        self._draw_cursor_lines()
        self.figure_canvas.draw_idle()
        self._on_figure_content_configure()

    def _render_skeleton_overview(self, dataframe: pd.DataFrame) -> None:
        self.figure.clear()
        self.figure.patch.set_facecolor(self._palette["bg_panel_alt"])
        self.figure.set_size_inches(10, 6.5, forward=True)
        self.figure_canvas.get_tk_widget().configure(height=680)
        self._skeleton_click_targets = []

        axes = self.figure.subplots(1, 2)
        axis_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

        self._style_axis(axis_list[0])
        self._style_axis(axis_list[1])
        self._render_front_skeleton(axis_list[0], dataframe)
        self._render_spinal_skeleton(axis_list[1], dataframe)

        self.figure.tight_layout(pad=2.0, w_pad=2.0)
        self.figure_canvas.draw_idle()
        self._on_figure_content_configure()

    def _render_front_skeleton(self, axis, dataframe: pd.DataFrame) -> None:
        points = self._collect_point_positions(dataframe, FRONT_POINT_NAMES)
        if not points:
            axis.set_title("Front skeleton: no point data", color=self._palette["fg_primary"])
            return

        for start_name, end_name in FRONT_CONNECTIONS:
            if start_name not in points or end_name not in points:
                continue
            start_x, start_y = points[start_name]
            end_x, end_y = points[end_name]
            axis.plot([start_x, end_x], [start_y, end_y], color="#86e0c2", linewidth=2.0, alpha=0.85)

        xs = []
        ys = []
        for point_name, (x_value, y_value) in points.items():
            axis.scatter([x_value], [y_value], s=40, color="#7aa2ff", zorder=3)
            axis.text(x_value, y_value - 6.0, point_name, fontsize=8, color=self._palette["fg_primary"], ha="center", va="bottom")
            self._register_skeleton_click_target(axis, point_name, x_value, y_value)
            xs.append(x_value)
            ys.append(y_value)

        if self.show_barbell_var.get():
            barbell_point = self._point_position(dataframe, "barbell_center")
            if barbell_point is not None:
                barbell_x, barbell_y = barbell_point
                axis.scatter([barbell_x], [barbell_y], s=70, color="#ffb86b", zorder=4)
                axis.text(barbell_x, barbell_y + 8.0, "barbell", fontsize=8, color="#ffb86b", ha="center", va="top")
                self._register_skeleton_click_target(axis, "barbell_center", barbell_x, barbell_y)
                xs.append(barbell_x)
                ys.append(barbell_y)

                if self.show_barbell_velocity_var.get():
                    velocity_vector = self._barbell_velocity_vector(dataframe)
                    if velocity_vector is not None:
                        vel_x, vel_y = velocity_vector
                        body_scale = self._axis_scale_from_points(xs, ys)
                        arrow_dx = vel_x * body_scale * 0.12
                        arrow_dy = vel_y * body_scale * 0.12
                        axis.arrow(
                            barbell_x,
                            barbell_y,
                            arrow_dx,
                            arrow_dy,
                            width=1.0,
                            head_width=8.0,
                            head_length=10.0,
                            color="#ff8f40",
                            alpha=0.9,
                            length_includes_head=True,
                            zorder=5,
                        )
                        axis.text(
                            barbell_x + arrow_dx,
                            barbell_y + arrow_dy,
                            "v",
                            fontsize=9,
                            color="#ff8f40",
                            ha="left",
                            va="bottom",
                        )

        self._set_axis_bounds(axis, xs, ys)
        axis.invert_yaxis()
        axis.set_title("Front skeleton with detected points", color=self._palette["fg_primary"])
        axis.set_xlabel("x [px]", color=self._palette["fg_muted"])
        axis.set_ylabel("y [px]", color=self._palette["fg_muted"])

    def _render_spinal_skeleton(self, axis, dataframe: pd.DataFrame) -> None:
        points9 = self._collect_point_positions(dataframe, SPINAL_POINTS_9)
        if len(points9) >= 3:
            spinal_order = [point_name for point_name in SPINAL_POINTS_9 if point_name in points9]
            points = points9
        else:
            points4 = self._collect_point_positions(dataframe, SPINAL_POINTS_4)
            spinal_order = [point_name for point_name in SPINAL_POINTS_4 if point_name in points4]
            points = points4

        if len(spinal_order) < 2:
            axis.set_title("Spinal skeleton: no point data", color=self._palette["fg_primary"])
            return

        xs = []
        ys = []
        for index in range(len(spinal_order) - 1):
            start_name = spinal_order[index]
            end_name = spinal_order[index + 1]
            start_x, start_y = points[start_name]
            end_x, end_y = points[end_name]
            axis.plot([start_x, end_x], [start_y, end_y], color="#7aa2ff", linewidth=2.2, alpha=0.9)

        for point_name in spinal_order:
            x_value, y_value = points[point_name]
            axis.scatter([x_value], [y_value], s=48, color="#86e0c2", zorder=3)
            axis.text(x_value + 6.0, y_value, point_name, fontsize=8, color=self._palette["fg_primary"], ha="left", va="center")
            self._register_skeleton_click_target(axis, point_name, x_value, y_value)
            xs.append(x_value)
            ys.append(y_value)

        self._set_axis_bounds(axis, xs, ys)
        axis.invert_yaxis()
        axis.set_title("Spinal skeleton (imitation)", color=self._palette["fg_primary"])
        axis.set_xlabel("x [px]", color=self._palette["fg_muted"])
        axis.set_ylabel("y [px]", color=self._palette["fg_muted"])

    def _collect_point_positions(self, dataframe: pd.DataFrame, point_names: list[str]) -> dict[str, tuple[float, float]]:
        positions: dict[str, tuple[float, float]] = {}
        for point_name in point_names:
            point = self._point_position(dataframe, point_name)
            if point is not None:
                positions[point_name] = point
        return positions

    def _point_position(self, dataframe: pd.DataFrame, point_name: str) -> tuple[float, float] | None:
        x_column = f"{point_name}_x"
        y_column = f"{point_name}_y"
        if dataframe.empty or x_column not in dataframe.columns or y_column not in dataframe.columns:
            return None

        point_frame = dataframe[[x_column, y_column]].copy().dropna()
        if point_frame.empty:
            return None

        x_value = pd.to_numeric(point_frame[x_column], errors="coerce").dropna()
        y_value = pd.to_numeric(point_frame[y_column], errors="coerce").dropna()
        if x_value.empty or y_value.empty:
            return None
        return float(x_value.median()), float(y_value.median())

    def _barbell_velocity_vector(self, dataframe: pd.DataFrame) -> tuple[float, float] | None:
        barbell_frame = self._prepare_barbell_chart_frame(dataframe)
        if len(barbell_frame) < 4:
            return None

        recent = barbell_frame.tail(5)
        start_x = float(recent.iloc[0]["barbell_center_x"])
        start_y = float(recent.iloc[0]["barbell_center_y"])
        end_x = float(recent.iloc[-1]["barbell_center_x"])
        end_y = float(recent.iloc[-1]["barbell_center_y"])
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        magnitude = math.sqrt(delta_x * delta_x + delta_y * delta_y)
        if magnitude <= 1e-6:
            return None
        return delta_x / magnitude, delta_y / magnitude

    def _axis_scale_from_points(self, xs: list[float], ys: list[float]) -> float:
        if not xs or not ys:
            return 50.0
        x_span = max(xs) - min(xs)
        y_span = max(ys) - min(ys)
        span = max(x_span, y_span)
        return max(30.0, float(span))

    def _set_axis_bounds(self, axis, xs: list[float], ys: list[float]) -> None:
        if not xs or not ys:
            return
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        x_span = max(1.0, x_max - x_min)
        y_span = max(1.0, y_max - y_min)
        x_margin = x_span * 0.25
        y_margin = y_span * 0.25
        axis.set_xlim(x_min - x_margin, x_max + x_margin)
        axis.set_ylim(y_min - y_margin, y_max + y_margin)
        axis.set_aspect("equal", adjustable="box")

    def _register_skeleton_click_target(self, axis, point_name: str, x_value: float, y_value: float) -> None:
        self._skeleton_click_targets.append(
            {
                "axis": axis,
                "point_name": point_name,
                "x": float(x_value),
                "y": float(y_value),
            }
        )

    def _style_axis(self, axis) -> None:
        axis.set_facecolor(self._palette["bg_panel_alt"])
        axis.tick_params(colors=self._palette["fg_muted"])
        for spine in axis.spines.values():
            spine.set_color(self._palette["border"])

    def _draw_cursor_lines(self) -> None:
        for line in self._cursor_lines:
            try:
                line.remove()
            except ValueError:
                pass
        self._cursor_lines = []
        if self._cursor_time_s is None:
            return

        marker_label = f"t={self._cursor_time_s:.2f}s"
        for axis in self.figure.axes:
            if not axis.get_visible():
                continue
            marker = axis.axvline(self._cursor_time_s, color="#f7768e", linewidth=1.4, alpha=0.9, zorder=6)
            self._cursor_lines.append(marker)
            label = axis.text(
                self._cursor_time_s,
                0.98,
                marker_label,
                transform=axis.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=8,
                color=self._palette["fg_primary"],
                bbox={"boxstyle": "round,pad=0.2", "facecolor": self._palette["bg_panel"], "edgecolor": self._palette["border"], "alpha": 0.9},
                zorder=7,
            )
            self._cursor_lines.append(label)

    def _render_point_chart(self, axis, dataframe: pd.DataFrame, point_name: str) -> None:
        x_column = f"{point_name}_x"
        y_column = f"{point_name}_y"
        if dataframe.empty or x_column not in dataframe.columns or y_column not in dataframe.columns:
            axis.set_title(f"No data for: {point_name}", color=self._palette["fg_primary"])
            return

        plot_frame = self._prepare_point_chart_frame(dataframe, point_name)
        if plot_frame.empty or plot_frame[x_column].dropna().empty or plot_frame[y_column].dropna().empty:
            axis.set_title(f"No data for: {point_name}", color=self._palette["fg_primary"])
            return

        self._annotate_rep_intervals(axis, dataframe)
        axis.plot(plot_frame["time_s"], plot_frame[x_column], label="x", color="#7aa2ff", linewidth=2.0)
        axis.plot(plot_frame["time_s"], plot_frame[y_column], label="y", color="#86e0c2", linewidth=2.0)
        axis.set_ylabel("Position [px]", color=self._palette["fg_muted"])
        axis.set_title(f"Point time series: {point_name}", color=self._palette["fg_primary"])
        legend = axis.legend(facecolor=self._palette["bg_panel"], edgecolor=self._palette["border"], loc="upper right")
        for text in legend.get_texts():
            text.set_color(self._palette["fg_primary"])
        axis.grid(True, alpha=0.18, color=self._palette["accent"])

    def _render_barbell_speed_chart(self, axis, dataframe: pd.DataFrame) -> None:
        required_columns = {"time_s", "barbell_center_x", "barbell_center_y"}
        if dataframe.empty or not required_columns.issubset(dataframe.columns):
            axis.set_title("No data for barbell speed", color=self._palette["fg_primary"])
            return

        speed_frame = self._prepare_barbell_chart_frame(dataframe)
        if len(speed_frame) < 2:
            axis.set_title("Not enough data to compute barbell speed", color=self._palette["fg_primary"])
            return

        delta_time = speed_frame["time_s"].diff()
        delta_x = speed_frame["barbell_center_x"].diff()
        delta_y = speed_frame["barbell_center_y"].diff()
        speed = ((delta_x.pow(2) + delta_y.pow(2)).pow(0.5) / delta_time.replace(0, pd.NA)).dropna()
        speed = speed.rolling(window=7, center=True, min_periods=1).mean()
        speed = speed.rolling(window=5, center=True, min_periods=1).median()
        speed = speed.bfill().ffill()
        if speed.empty:
            axis.set_title("Not enough data to compute barbell speed", color=self._palette["fg_primary"])
            return

        speed_times = speed_frame.loc[speed.index, "time_s"]
        self._annotate_rep_intervals(axis, dataframe)
        axis.plot(speed_times, speed, label="speed", color="#ffb86b", linewidth=2.2)
        axis.set_ylabel("Speed [px/s]", color=self._palette["fg_muted"])
        axis.set_title("Barbell speed over time", color=self._palette["fg_primary"])
        legend = axis.legend(facecolor=self._palette["bg_panel"], edgecolor=self._palette["border"], loc="upper right")
        for text in legend.get_texts():
            text.set_color(self._palette["fg_primary"])
        axis.grid(True, alpha=0.18, color=self._palette["accent"])

    def _render_barbell_position_chart(self, axis, dataframe: pd.DataFrame) -> None:
        required_columns = {"time_s", "barbell_center_x", "barbell_center_y"}
        if dataframe.empty or not required_columns.issubset(dataframe.columns):
            axis.set_title("No data for barbell position", color=self._palette["fg_primary"])
            return

        position_frame = self._prepare_barbell_chart_frame(dataframe)
        if position_frame.empty:
            axis.set_title("No data for barbell position", color=self._palette["fg_primary"])
            return

        self._annotate_rep_intervals(axis, dataframe)
        axis.plot(
            position_frame["time_s"],
            position_frame["barbell_center_x"],
            label="barbell x (approximated)",
            color="#ffb86b",
            linewidth=2.0,
        )
        axis.plot(
            position_frame["time_s"],
            position_frame["barbell_center_y"],
            label="barbell y (approximated)",
            color="#7aa2ff",
            linewidth=2.0,
        )
        axis.set_ylabel("Position [px]", color=self._palette["fg_muted"])
        axis.set_title("Barbell position over time", color=self._palette["fg_primary"])
        legend = axis.legend(facecolor=self._palette["bg_panel"], edgecolor=self._palette["border"], loc="upper right")
        for text in legend.get_texts():
            text.set_color(self._palette["fg_primary"])
        axis.grid(True, alpha=0.18, color=self._palette["accent"])

    def _prepare_barbell_chart_frame(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        frame = self._prepare_point_chart_frame(dataframe, "barbell_center")
        if frame.empty:
            return pd.DataFrame()

        frame = frame[["time_s", "barbell_center_x", "barbell_center_y"]].dropna(how="all").copy()
        if frame.empty:
            return pd.DataFrame()

        frame[["barbell_center_x", "barbell_center_y"]] = frame[["barbell_center_x", "barbell_center_y"]].interpolate(
            method="linear",
            limit_direction="both",
        )
        frame[["barbell_center_x", "barbell_center_y"]] = frame[["barbell_center_x", "barbell_center_y"]].rolling(
            window=5,
            center=True,
            min_periods=1,
        ).mean()
        frame[["barbell_center_x", "barbell_center_y"]] = frame[["barbell_center_x", "barbell_center_y"]].bfill().ffill()
        return frame

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

    def _annotate_rep_intervals(self, axis, dataframe: pd.DataFrame) -> None:
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
            axis.axvspan(start_time, end_time, color=color, alpha=0.16, zorder=0)
            axis.axvline(end_time, color=self._palette["border"], alpha=0.45, linewidth=1.0, zorder=1)
            axis.text(
                (start_time + end_time) / 2.0,
                top_y,
                f"Rep {rep_number}",
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color=self._palette["fg_primary"],
                bbox={"boxstyle": "round,pad=0.22", "facecolor": self._palette["bg_panel"], "edgecolor": self._palette["border"], "alpha": 0.82},
                zorder=2,
            )
