from __future__ import annotations

import ctypes
import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .charts import ChartWindow
from .config import (
    BARBELL_MODEL_OPTIONS,
    CHART_OPTIONS,
    CHART_POINTS,
    DEFAULT_AUX_YOLO_MODEL_PATH,
    DISPLAY_ASPECT_OPTIONS,
    DISPLAY_SCALE_OPTIONS,
    EXERCISES,
    POSE_MODEL_OPTIONS,
    discover_exercise_classifier_models,
    get_default_exercise_classifier_model,
)
from .models import DisplaySettings, resize_for_display
from .worker import AnalysisSettings, FramePacket, StatePacket, VideoAnalysisWorker


AUXILIARY_POINT_BASE_RADIUS = 4
AUXILIARY_POINT_SCALE_MIN = 100
AUXILIARY_POINT_SCALE_MAX = 1000


class StyledMessageDialog(tk.Toplevel):
    def __init__(self, master: tk.Misc, title: str, message: str, palette: dict[str, str], variant: str = "info") -> None:
        super().__init__(master)
        self.title(title)
        self.geometry("520x240")
        self.minsize(420, 220)
        self.overrideredirect(True)
        self.transient(master)
        self.configure(bg=palette["bg_root"])
        self._palette = palette
        self._variant = variant
        self._window_drag_origin: tuple[int, int] | None = None

        self._build_layout(title, message)
        self.after_idle(self._apply_window_chrome)
        self.bind("<Map>", self._on_window_map)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda _event: self._close())
        self._center_over_master(master)
        self.wait_visibility()
        self.lift()
        try:
            self.attributes("-topmost", True)
            self.after(150, lambda: self.attributes("-topmost", False))
        except tk.TclError:
            pass
        self.grab_set()
        self.focus_force()

    def _build_layout(self, title: str, message: str) -> None:
        outer = tk.Frame(self, bg=self._palette["border"], highlightthickness=0, bd=0)
        outer.pack(fill="both", expand=True)

        titlebar = tk.Frame(outer, bg=self._palette["bg_titlebar"], height=52, highlightthickness=0, bd=0)
        titlebar.pack(fill="x", padx=1, pady=1)
        titlebar.pack_propagate(False)

        accent_color = self._palette["danger"] if self._variant == "error" else self._palette["accent"]
        tk.Frame(titlebar, bg=accent_color, height=1, highlightthickness=0, bd=0).pack(side="top", fill="x")

        left = tk.Frame(titlebar, bg=self._palette["bg_titlebar"], highlightthickness=0, bd=0)
        left.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(2, 0))

        icon_text = "!" if self._variant == "error" else "i"
        icon = tk.Label(
            left,
            text=icon_text,
            bg=self._palette["bg_input"],
            fg=accent_color,
            font=("Segoe UI Semibold", 10),
            width=3,
            pady=8,
        )
        icon.pack(side="left", pady=9)

        text_wrap = tk.Frame(left, bg=self._palette["bg_titlebar"], highlightthickness=0, bd=0)
        text_wrap.pack(side="left", padx=(10, 0), pady=9)
        title_label = tk.Label(
            text_wrap,
            text=title,
            bg=self._palette["bg_titlebar"],
            fg=self._palette["fg_primary"],
            font=("Segoe UI Semibold", 10),
        )
        title_label.pack(anchor="w")
        subtitle = "Application message" if self._variant == "info" else "Processing error"
        subtitle_label = tk.Label(
            text_wrap,
            text=subtitle,
            bg=self._palette["bg_titlebar"],
            fg=self._palette["fg_muted"],
            font=("Segoe UI", 8),
        )
        subtitle_label.pack(anchor="w")

        close_button = tk.Label(
            titlebar,
            text="✕",
            bg=self._palette["bg_panel_alt"],
            fg=self._palette["fg_primary"],
            font=("Segoe UI Symbol", 10),
            width=4,
            cursor="hand2",
        )
        close_button.pack(side="right", padx=12, pady=10, fill="y")
        close_button.bind("<Enter>", lambda _event: close_button.configure(bg=self._palette["danger"]))
        close_button.bind("<Leave>", lambda _event: close_button.configure(bg=self._palette["bg_panel_alt"]))
        close_button.bind("<Button-1>", lambda _event: self._close())

        drag_targets = (titlebar, left, icon, text_wrap, title_label, subtitle_label)
        for widget in drag_targets:
            widget.bind("<ButtonPress-1>", self._start_window_drag)
            widget.bind("<B1-Motion>", self._drag_window)

        body = tk.Frame(outer, bg=self._palette["bg_root"], highlightthickness=0, bd=0)
        body.pack(fill="both", expand=True, padx=1, pady=(0, 1))

        panel_shell = tk.Frame(body, bg=self._palette["border"], highlightthickness=0, bd=0)
        panel_shell.pack(fill="both", expand=True, padx=16, pady=16)
        panel = tk.Frame(panel_shell, bg=self._palette["bg_panel"], highlightthickness=0, bd=0)
        panel.pack(fill="both", expand=True, padx=1, pady=1)

        message_label = tk.Label(
            panel,
            text=message,
            bg=self._palette["bg_panel"],
            fg=self._palette["fg_primary"],
            font=("Segoe UI", 10),
            justify="left",
            wraplength=440,
            anchor="w",
        )
        message_label.pack(fill="both", expand=True, padx=18, pady=(18, 12))

        footer = tk.Frame(panel, bg=self._palette["bg_panel"], highlightthickness=0, bd=0)
        footer.pack(fill="x", padx=18, pady=(0, 18))
        ok_button = tk.Button(
            footer,
            text="OK",
            command=self._close,
            bg=accent_color,
            fg=self._palette["bg_root"],
            activebackground=self._palette["accent_active"] if self._variant == "info" else self._palette["danger_active"],
            activeforeground=self._palette["bg_root"],
            relief="flat",
            bd=0,
            padx=22,
            pady=10,
            font=("Segoe UI Semibold", 10),
            cursor="hand2",
        )
        ok_button.pack(side="right")
        self.after_idle(ok_button.focus_set)

    def _center_over_master(self, master: tk.Misc) -> None:
        self.update_idletasks()
        master.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = master.winfo_x() + max((master.winfo_width() - width) // 2, 0)
        y = master.winfo_y() + max((master.winfo_height() - height) // 2, 0)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _start_window_drag(self, event) -> None:
        self._window_drag_origin = (event.x_root - self.winfo_x(), event.y_root - self.winfo_y())

    def _drag_window(self, event) -> None:
        if self._window_drag_origin is None:
            return
        offset_x, offset_y = self._window_drag_origin
        self.geometry(f"+{event.x_root - offset_x}+{event.y_root - offset_y}")

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

    def _close(self) -> None:
        try:
            self.grab_release()
        except tk.TclError:
            pass
        self.destroy()


class HoverTooltip:
    def __init__(self, widget: tk.Misc, text_getter, palette: dict[str, str], *, wraplength: int = 520, delay_ms: int = 250) -> None:
        self.widget = widget
        self.text_getter = text_getter
        self.palette = palette
        self.wraplength = wraplength
        self.delay_ms = delay_ms
        self._after_id: str | None = None
        self._tooltip_window: tk.Toplevel | None = None

        self.widget.bind("<Enter>", self._schedule_show, add="+")
        self.widget.bind("<Leave>", self._hide, add="+")
        self.widget.bind("<Motion>", self._follow_pointer, add="+")
        self.widget.bind("<ButtonPress>", self._hide, add="+")
        self.widget.bind("<FocusOut>", self._hide, add="+")
        self.widget.bind("<Destroy>", self._on_destroy, add="+")

    def _get_text(self) -> str:
        value = self.text_getter() if callable(self.text_getter) else self.text_getter
        return str(value).strip()

    def _schedule_show(self, _event=None) -> None:
        self._cancel_scheduled_show()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel_scheduled_show(self) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        self._after_id = None
        if not self.widget.winfo_exists():
            return

        text = self._get_text()
        if not text:
            return

        self._hide()
        self._tooltip_window = tk.Toplevel(self.widget)
        self._tooltip_window.overrideredirect(True)
        self._tooltip_window.transient(self.widget.winfo_toplevel())
        self._tooltip_window.configure(bg=self.palette["border"])

        body = tk.Label(
            self._tooltip_window,
            text=text,
            justify="left",
            anchor="w",
            wraplength=self.wraplength,
            bg=self.palette["bg_panel_alt"],
            fg=self.palette["fg_primary"],
            font=("Segoe UI", 9),
            padx=10,
            pady=8,
            relief="flat",
            bd=0,
        )
        body.pack(padx=1, pady=1)
        self._position_window()

    def _position_window(self) -> None:
        if self._tooltip_window is None or not self._tooltip_window.winfo_exists():
            return

        self._tooltip_window.update_idletasks()
        pointer_x = self.widget.winfo_pointerx() + 16
        pointer_y = self.widget.winfo_pointery() + 20
        width = self._tooltip_window.winfo_reqwidth()
        height = self._tooltip_window.winfo_reqheight()
        screen_width = self.widget.winfo_screenwidth()
        screen_height = self.widget.winfo_screenheight()

        x = min(pointer_x, screen_width - width - 12)
        y = min(pointer_y, screen_height - height - 12)
        self._tooltip_window.geometry(f"+{max(8, x)}+{max(8, y)}")

    def _follow_pointer(self, _event=None) -> None:
        if self._tooltip_window is not None:
            self._position_window()

    def _hide(self, _event=None) -> None:
        self._cancel_scheduled_show()
        if self._tooltip_window is not None:
            try:
                self._tooltip_window.destroy()
            except tk.TclError:
                pass
            self._tooltip_window = None

    def _on_destroy(self, _event=None) -> None:
        self._hide()


def show_styled_message(master: tk.Misc, title: str, message: str, palette: dict[str, str], variant: str = "info") -> None:
    if not bool(master.winfo_exists()):
        return

    try:
        dialog = StyledMessageDialog(master, title, message, palette, variant)
        master.wait_window(dialog)
        return
    except tk.TclError:
        pass

    if variant == "error":
        messagebox.showerror(title, message, parent=master)
    else:
        messagebox.showinfo(title, message, parent=master)


class PoseDesktopApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Pose Desktop App")
        self.geometry("1500x920")
        self.minsize(1280, 800)
        self.configure(bg="#081120")
        self.overrideredirect(True)
        self._palette = {
            "bg_root": "#0a0d14",
            "bg_panel": "#121722",
            "bg_panel_alt": "#171d2a",
            "bg_input": "#20283a",
            "bg_hover": "#2a354a",
            "bg_titlebar": "#0c1018",
            "bg_titlebar_chip": "#141b28",
            "fg_primary": "#edf2ff",
            "fg_muted": "#8d98b3",
            "accent": "#7aa2ff",
            "accent_active": "#a9c0ff",
            "border": "#283247",
            "danger": "#d65a7a",
            "danger_active": "#ea7e9b",
            "success": "#86e0c2",
        }
        self._is_maximized = False
        self._window_drag_origin: tuple[int, int] | None = None
        self._sidebar_min_width = 300
        self._sidebar_max_width = 560
        self._sidebar_width_step = 40
        self._configure_styles()

        self.frame_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
        self.state_queue: queue.Queue[StatePacket] = queue.Queue(maxsize=1)
        self.worker: VideoAnalysisWorker | None = None
        self.video_photo: ImageTk.PhotoImage | None = None
        self.preview_frame_bgr: np.ndarray | None = None
        self.points_path: str | None = None
        self.selected_video_path: str | None = None

        self.source_var = tk.StringVar(value="video")
        self.video_path_var = tk.StringVar(value="No file selected")
        self.camera_index_var = tk.IntVar(value=0)
        self.pose_model_var = tk.StringVar(value=list(POSE_MODEL_OPTIONS.keys())[0])
        self.show_pose_preview_var = tk.BooleanVar(value=True)
        self.use_barbell_var = tk.BooleanVar(value=True)
        self.barbell_model_var = tk.StringVar(value=list(BARBELL_MODEL_OPTIONS.keys())[0])
        self.reset_barbell_path_each_rep_var = tk.BooleanVar(value=False)
        self.use_aux_yolo_var = tk.BooleanVar(value=True)
        self.aux_yolo_path_var = tk.StringVar(value=str(DEFAULT_AUX_YOLO_MODEL_PATH))
        self.pose_conf_var = tk.DoubleVar(value=0.25)
        self.barbell_conf_var = tk.DoubleVar(value=0.25)
        self.aux_yolo_conf_var = tk.DoubleVar(value=0.25)
        self.aux_point_scale_var = tk.DoubleVar(value=100.0)
        self.aux_point_scale_hint_var = tk.StringVar(value="100%")
        self.exercise_var = tk.StringVar(value=EXERCISES[0])
        self.auto_model_var = tk.StringVar(value="")
        self.auto_model_hint_var = tk.StringVar(value="")
        self.sidebar_width_var = tk.IntVar(value=360)
        self.sidebar_width_hint_var = tk.StringVar(value="")
        self.aspect_ratio_var = tk.StringVar(value="16:9")
        self.scale_var = tk.StringVar(value="100%")
        self.chart_point_var = tk.StringVar(value=CHART_OPTIONS[0])
        self.status_var = tk.StringVar(value="Ready")
        self.frame_var = tk.StringVar(value="Frame: 0/0")
        self.rep_var = tk.StringVar(value="Reps: 0")
        self.active_exercise_var = tk.StringVar(value=f"Active exercise: {self.exercise_var.get()}")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.auto_model_options: dict[str, Path] = {}
        self.root_content = None
        self.sidebar_shell = None
        self.sidebar_wrap_labels: list[ttk.Label] = []
        self._hover_tooltips: list[HoverTooltip] = []

        self._build_layout()
        self._apply_sidebar_width()
        self._refresh_auto_model_options()
        self.exercise_var.trace_add("write", self._on_exercise_mode_changed)
        self.after_idle(self._apply_window_chrome)
        self.after(15, self._poll_worker)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Map>", self._on_window_map)

    def _update_controls_scrollregion(self, _event=None) -> None:
        self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox("all"))

    def _resize_controls_width(self, event) -> None:
        self.controls_canvas.itemconfigure(self.controls_window, width=event.width)

    def _get_sidebar_wraplength(self) -> int:
        return max(220, int(self.sidebar_width_var.get()) - 40)

    def _apply_sidebar_width(self) -> None:
        width = max(self._sidebar_min_width, min(self._sidebar_max_width, int(self.sidebar_width_var.get())))
        self.sidebar_width_var.set(width)
        self.sidebar_width_hint_var.set(f"{width}px")

        if self.root_content is not None:
            self.root_content.columnconfigure(0, minsize=width + 24)
        if self.sidebar_shell is not None:
            self.sidebar_shell.configure(width=width + 24)
        if hasattr(self, "controls_canvas"):
            self.controls_canvas.configure(width=width)
            if hasattr(self, "controls_window"):
                self.controls_canvas.itemconfigure(self.controls_window, width=width)

        wraplength = self._get_sidebar_wraplength()
        for label in self.sidebar_wrap_labels:
            label.configure(wraplength=wraplength)

    def _change_sidebar_width(self, delta: int) -> None:
        self.sidebar_width_var.set(int(self.sidebar_width_var.get()) + delta)
        self._apply_sidebar_width()

    def _on_controls_mousewheel(self, event) -> None:
        self.controls_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_controls_mousewheel(self, _event=None) -> None:
        self.controls_canvas.bind_all("<MouseWheel>", self._on_controls_mousewheel)

    def _unbind_controls_mousewheel(self, _event=None) -> None:
        self.controls_canvas.unbind_all("<MouseWheel>")

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")

        bg_root = self._palette["bg_root"]
        bg_panel = self._palette["bg_panel"]
        bg_panel_alt = self._palette["bg_panel_alt"]
        bg_input = self._palette["bg_input"]
        bg_hover = self._palette["bg_hover"]
        bg_titlebar = self._palette["bg_titlebar"]
        fg_primary = self._palette["fg_primary"]
        fg_muted = self._palette["fg_muted"]
        accent = self._palette["accent"]
        accent_active = self._palette["accent_active"]
        border = self._palette["border"]
        success = self._palette["success"]

        self.option_add("*Font", "{Segoe UI} 10")
        self.option_add("*TCombobox*Listbox.background", bg_input)
        self.option_add("*TCombobox*Listbox.foreground", fg_primary)
        self.option_add("*TCombobox*Listbox.selectBackground", accent)
        self.option_add("*TCombobox*Listbox.selectForeground", bg_root)

        style.configure(".", background=bg_root, foreground=fg_primary)
        style.configure("TFrame", background=bg_root)
        style.configure("Panel.TFrame", background=bg_panel)
        style.configure("PanelAlt.TFrame", background=bg_panel_alt)
        style.configure("TLabel", background=bg_root, foreground=fg_primary)
        style.configure("Panel.TLabel", background=bg_panel, foreground=fg_primary)
        style.configure("PanelMuted.TLabel", background=bg_panel, foreground=fg_muted)
        style.configure("PanelAlt.TLabel", background=bg_panel_alt, foreground=fg_primary)
        style.configure("PanelAltMuted.TLabel", background=bg_panel_alt, foreground=fg_muted)
        style.configure("Muted.TLabel", background=bg_root, foreground=fg_muted)
        style.configure("Titlebar.TFrame", background=bg_titlebar)
        style.configure(
            "TLabelframe",
            background=bg_panel,
            borderwidth=1,
            relief="solid",
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.configure(
            "TLabelframe.Label",
            background=bg_panel,
            foreground=accent_active,
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "TButton",
            background=bg_input,
            foreground=fg_primary,
            borderwidth=0,
            focusthickness=0,
            padding=(14, 10),
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "TButton",
            background=[("active", bg_hover), ("pressed", accent)],
            foreground=[("active", fg_primary), ("pressed", bg_root)],
        )
        style.configure(
            "Accent.TButton",
            background=accent,
            foreground=bg_root,
            borderwidth=0,
            focusthickness=0,
            padding=(14, 10),
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "Accent.TButton",
            background=[("active", accent_active), ("pressed", "#b7d9ff")],
            foreground=[("active", bg_root), ("pressed", bg_root)],
        )
        style.configure("TRadiobutton", background=bg_panel, foreground=fg_primary)
        style.map("TRadiobutton", background=[("active", bg_panel)], foreground=[("active", fg_primary)])
        style.configure("TCheckbutton", background=bg_panel, foreground=fg_primary)
        style.map("TCheckbutton", background=[("active", bg_panel)], foreground=[("active", fg_primary)])
        style.configure(
            "TCombobox",
            fieldbackground=bg_input,
            background=bg_input,
            foreground=fg_primary,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            arrowcolor=accent_active,
            padding=6,
        )
        style.map("TCombobox", fieldbackground=[("readonly", bg_input)], foreground=[("readonly", fg_primary)])
        style.configure(
            "TSpinbox",
            fieldbackground=bg_input,
            background=bg_input,
            foreground=fg_primary,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
            arrowsize=14,
        )
        style.configure(
            "Horizontal.TScale",
            background=bg_panel,
            troughcolor=bg_input,
            bordercolor=bg_input,
            lightcolor=accent,
            darkcolor=accent,
        )
        style.configure(
            "Horizontal.TProgressbar",
            background=success,
            troughcolor=bg_input,
            bordercolor=bg_input,
            lightcolor=success,
            darkcolor=success,
        )

    def _build_layout(self) -> None:
        outer = tk.Frame(self, bg=self._palette["border"], highlightthickness=0, bd=0)
        outer.pack(fill="both", expand=True)

        self._build_titlebar(outer)

        root = ttk.Frame(outer, padding=12)
        root.pack(fill="both", expand=True, padx=1, pady=(0, 1))
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        self.root_content = root

        sidebar_shell = tk.Frame(root, bg=self._palette["border"], highlightthickness=0, bd=0)
        sidebar_shell.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        self.sidebar_shell = sidebar_shell
        sidebar = ttk.Frame(sidebar_shell, style="Panel.TFrame", padding=10)
        sidebar.pack(fill="both", expand=True, padx=1, pady=1)
        sidebar.rowconfigure(1, weight=1)
        sidebar.columnconfigure(0, weight=1)

        sidebar_header = ttk.Frame(sidebar, style="Panel.TFrame")
        sidebar_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(sidebar_header, text="Control Center", style="Panel.TLabel", font=("Segoe UI Semibold", 11)).pack(anchor="w")
        ttk.Label(
            sidebar_header,
            text="Source, models and live display parameters",
            style="PanelMuted.TLabel",
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(2, 0))

        sidebar_width_row = ttk.Frame(sidebar_header, style="Panel.TFrame")
        sidebar_width_row.pack(fill="x", pady=(8, 0))
        ttk.Label(sidebar_width_row, text="Panel width", style="PanelMuted.TLabel", font=("Segoe UI", 9)).pack(side="left")
        ttk.Button(
            sidebar_width_row,
            text="+",
            width=3,
            command=lambda: self._change_sidebar_width(self._sidebar_width_step),
        ).pack(side="right")
        ttk.Button(
            sidebar_width_row,
            text="-",
            width=3,
            command=lambda: self._change_sidebar_width(-self._sidebar_width_step),
        ).pack(side="right", padx=(0, 6))
        ttk.Label(sidebar_width_row, textvariable=self.sidebar_width_hint_var, style="PanelMuted.TLabel").pack(side="right", padx=(0, 8))

        self.controls_canvas = tk.Canvas(
            sidebar,
            width=self.sidebar_width_var.get(),
            bg=self._palette["bg_panel"],
            highlightthickness=0,
            bd=0,
        )
        controls_scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=self.controls_canvas.yview)
        self.controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        self.controls_canvas.grid(row=1, column=0, sticky="nsew")
        controls_scrollbar.grid(row=1, column=1, sticky="ns")

        controls_host = ttk.Frame(self.controls_canvas, style="Panel.TFrame")
        self.controls_window = self.controls_canvas.create_window((0, 0), window=controls_host, anchor="nw")
        controls_host.bind("<Configure>", self._update_controls_scrollregion)
        self.controls_canvas.bind("<Configure>", self._resize_controls_width)
        self.controls_canvas.bind("<Enter>", self._bind_controls_mousewheel)
        self.controls_canvas.bind("<Leave>", self._unbind_controls_mousewheel)

        controls = ttk.LabelFrame(controls_host, text="Controls", padding=12)
        controls.pack(fill="both", expand=True)

        source_frame = ttk.LabelFrame(controls, text="Source", padding=8)
        source_frame.pack(fill="x", pady=(0, 10))
        ttk.Radiobutton(source_frame, text="Video", variable=self.source_var, value="video").pack(anchor="w")
        ttk.Radiobutton(source_frame, text="Camera", variable=self.source_var, value="camera").pack(anchor="w")
        ttk.Button(source_frame, text="Choose video file", command=self._choose_video, style="Accent.TButton").pack(fill="x", pady=(8, 4))
        self.video_path_label = ttk.Label(source_frame, textvariable=self.video_path_var, wraplength=self._get_sidebar_wraplength(), style="Muted.TLabel")
        self.video_path_label.pack(anchor="w")
        self.sidebar_wrap_labels.append(self.video_path_label)
        camera_row = ttk.Frame(source_frame)
        camera_row.pack(fill="x", pady=(8, 0))
        ttk.Label(camera_row, text="Camera index:").pack(side="left")
        ttk.Spinbox(camera_row, from_=0, to=10, textvariable=self.camera_index_var, width=6).pack(side="left", padx=(8, 0))

        model_frame = ttk.LabelFrame(controls, text="Models", padding=8)
        model_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(model_frame, text="Pose estimation").pack(anchor="w")
        self.pose_model_combobox = ttk.Combobox(
            model_frame,
            textvariable=self.pose_model_var,
            values=list(POSE_MODEL_OPTIONS.keys()),
            state="readonly",
        )
        self.pose_model_combobox.pack(fill="x", pady=(4, 8))
        ttk.Checkbutton(
            model_frame,
            text="Show pose estimation preview",
            variable=self.show_pose_preview_var,
            command=self._apply_display_changes,
        ).pack(anchor="w", pady=(0, 8))
        ttk.Checkbutton(model_frame, text="Barbell tracking", variable=self.use_barbell_var).pack(anchor="w")
        self.barbell_model_combobox = ttk.Combobox(
            model_frame,
            textvariable=self.barbell_model_var,
            values=list(BARBELL_MODEL_OPTIONS.keys()),
            state="readonly",
        )
        self.barbell_model_combobox.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(
            model_frame,
            text="Reset barbell path after each rep",
            variable=self.reset_barbell_path_each_rep_var,
        ).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(model_frame, text="Spinal curvature estimation", variable=self.use_aux_yolo_var).pack(anchor="w", pady=(10, 0))
        ttk.Button(model_frame, text="Choose curvature model file", command=self._choose_aux_yolo_model).pack(fill="x", pady=(4, 4))
        self.aux_yolo_path_label = ttk.Label(model_frame, textvariable=self.aux_yolo_path_var, wraplength=self._get_sidebar_wraplength(), style="Muted.TLabel")
        self.aux_yolo_path_label.pack(anchor="w")
        self.sidebar_wrap_labels.append(self.aux_yolo_path_label)

        confidence_frame = ttk.LabelFrame(controls, text="Confidence thresholds", padding=8)
        confidence_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(confidence_frame, text="Pose confidence").pack(anchor="w")
        ttk.Scale(confidence_frame, from_=0.05, to=0.95, variable=self.pose_conf_var, orient="horizontal").pack(fill="x")
        ttk.Label(confidence_frame, textvariable=tk.StringVar(value="")).pack_forget()
        ttk.Label(confidence_frame, text="Barbell confidence").pack(anchor="w", pady=(8, 0))
        ttk.Scale(confidence_frame, from_=0.05, to=0.95, variable=self.barbell_conf_var, orient="horizontal").pack(fill="x")
        ttk.Label(confidence_frame, text="Spinal curvature confidence").pack(anchor="w", pady=(8, 0))
        ttk.Scale(confidence_frame, from_=0.05, to=0.95, variable=self.aux_yolo_conf_var, orient="horizontal").pack(fill="x")

        workout_frame = ttk.LabelFrame(controls, text="Exercise and view", padding=8)
        workout_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(workout_frame, text="Exercise").pack(anchor="w")
        ttk.Combobox(workout_frame, textvariable=self.exercise_var, values=EXERCISES, state="readonly").pack(fill="x", pady=(4, 8))
        ttk.Label(workout_frame, text="Auto-classification model").pack(anchor="w")
        self.auto_model_combobox = ttk.Combobox(workout_frame, textvariable=self.auto_model_var, state="readonly")
        self.auto_model_combobox.pack(fill="x", pady=(4, 4))
        ttk.Button(workout_frame, text="Refresh models", command=self._refresh_auto_model_options).pack(fill="x", pady=(0, 4))
        self.auto_model_hint_label = ttk.Label(workout_frame, textvariable=self.auto_model_hint_var, wraplength=self._get_sidebar_wraplength(), style="Muted.TLabel")
        self.auto_model_hint_label.pack(anchor="w", pady=(0, 8))
        self.sidebar_wrap_labels.append(self.auto_model_hint_label)
        self._setup_hover_tooltips()
        ttk.Label(workout_frame, text="Aspect ratio").pack(anchor="w")
        aspect_box = ttk.Combobox(workout_frame, textvariable=self.aspect_ratio_var, values=list(DISPLAY_ASPECT_OPTIONS.keys()), state="readonly")
        aspect_box.pack(fill="x", pady=(4, 8))
        aspect_box.bind("<<ComboboxSelected>>", lambda _event: self._apply_display_changes())
        ttk.Label(workout_frame, text="Scale").pack(anchor="w")
        scale_box = ttk.Combobox(workout_frame, textvariable=self.scale_var, values=list(DISPLAY_SCALE_OPTIONS.keys()), state="readonly")
        scale_box.pack(fill="x", pady=(4, 0))
        scale_box.bind("<<ComboboxSelected>>", lambda _event: self._apply_display_changes())
        aux_point_scale_row = ttk.Frame(workout_frame, style="Panel.TFrame")
        aux_point_scale_row.pack(fill="x", pady=(8, 0))
        ttk.Label(aux_point_scale_row, text="Spinal point size").pack(side="left")
        ttk.Label(aux_point_scale_row, textvariable=self.aux_point_scale_hint_var, style="PanelMuted.TLabel").pack(side="right")
        ttk.Scale(
            workout_frame,
            from_=AUXILIARY_POINT_SCALE_MIN,
            to=AUXILIARY_POINT_SCALE_MAX,
            variable=self.aux_point_scale_var,
            orient="horizontal",
            command=self._on_auxiliary_point_scale_changed,
        ).pack(fill="x", pady=(4, 0))

        chart_frame = ttk.LabelFrame(controls, text="Charts", padding=8)
        chart_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(chart_frame, text="Point").pack(anchor="w")
        ttk.Combobox(chart_frame, textvariable=self.chart_point_var, values=CHART_OPTIONS, state="readonly").pack(fill="x", pady=(4, 8))

        viewer_shell = tk.Frame(root, bg=self._palette["border"], highlightthickness=0, bd=0)
        viewer_shell.grid(row=0, column=1, sticky="nsew")
        viewer = ttk.Frame(viewer_shell, style="PanelAlt.TFrame", padding=12)
        viewer.pack(fill="both", expand=True, padx=1, pady=1)
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(2, weight=1)

        viewer_header = ttk.Frame(viewer, style="PanelAlt.TFrame")
        viewer_header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        viewer_header.columnconfigure(0, weight=1)
        ttk.Label(viewer_header, text="Session Monitor", style="PanelAlt.TLabel", font=("Segoe UI Semibold", 11)).grid(row=0, column=0, sticky="w")
        ttk.Label(
            viewer_header,
            text="Annotated preview, processing status and quick actions",
            style="PanelAltMuted.TLabel",
            font=("Segoe UI", 9),
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        button_frame = ttk.Frame(viewer, style="PanelAlt.TFrame", padding=(0, 0, 0, 10))
        button_frame.grid(row=1, column=0, sticky="ew")
        button_frame.columnconfigure((0, 1, 2, 3, 4), weight=1)
        ttk.Button(button_frame, text="Start", command=self._start_processing, style="Accent.TButton").grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(button_frame, text="Pause", command=self._pause_processing).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(button_frame, text="Resume", command=self._resume_processing).grid(row=0, column=2, sticky="ew", padx=6)
        ttk.Button(button_frame, text="Cancel", command=self._cancel_processing).grid(row=0, column=3, sticky="ew", padx=6)
        ttk.Button(button_frame, text="Show charts", command=self._show_charts).grid(row=0, column=4, sticky="ew", padx=(6, 0))

        self.video_label = tk.Label(
            viewer,
            text="Preview will appear after start",
            anchor="center",
            bg=self._palette["bg_panel_alt"],
            fg=self._palette["fg_muted"],
            font=("Segoe UI Semibold", 13),
            padx=20,
            pady=20,
            relief="flat",
            highlightthickness=1,
            highlightbackground=self._palette["border"],
        )
        self.video_label.grid(row=2, column=0, sticky="nsew")

        info_frame = ttk.Frame(viewer, style="PanelAlt.TFrame", padding=(0, 12, 0, 0))
        info_frame.grid(row=3, column=0, sticky="ew")
        info_frame.columnconfigure(0, weight=1)
        info_frame.columnconfigure(1, weight=1)
        ttk.Label(info_frame, textvariable=self.status_var, style="PanelAltMuted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.frame_var, style="PanelAltMuted.TLabel").grid(row=0, column=1, sticky="e")
        ttk.Label(info_frame, textvariable=self.rep_var, style="PanelAlt.TLabel", font=("Segoe UI", 16, "bold")).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(info_frame, textvariable=self.active_exercise_var, style="PanelAlt.TLabel", font=("Segoe UI", 12, "bold")).grid(row=1, column=1, sticky="e", pady=(6, 0))
        self.progressbar = ttk.Progressbar(info_frame, variable=self.progress_var, maximum=1.0)
        self.progressbar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _setup_hover_tooltips(self) -> None:
        tooltip_specs = [
            (self.video_path_label, self.video_path_var.get),
            (self.pose_model_combobox, self.pose_model_var.get),
            (self.barbell_model_combobox, self.barbell_model_var.get),
            (self.aux_yolo_path_label, self.aux_yolo_path_var.get),
            (self.auto_model_combobox, self.auto_model_var.get),
            (self.auto_model_hint_label, self.auto_model_hint_var.get),
        ]
        for widget, text_getter in tooltip_specs:
            self._hover_tooltips.append(HoverTooltip(widget, text_getter, self._palette))

    def _build_titlebar(self, parent: tk.Misc) -> None:
        titlebar = tk.Frame(parent, bg=self._palette["bg_titlebar"], height=58, highlightthickness=0, bd=0)
        titlebar.pack(fill="x", padx=1, pady=1)
        titlebar.pack_propagate(False)

        top_rule = tk.Frame(titlebar, bg=self._palette["accent"], height=1, highlightthickness=0, bd=0)
        top_rule.pack(side="top", fill="x")

        left = tk.Frame(titlebar, bg=self._palette["bg_titlebar"], highlightthickness=0, bd=0)
        left.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(2, 0))

        brand_card = tk.Frame(left, bg=self._palette["bg_panel_alt"], highlightthickness=1, highlightbackground=self._palette["border"], bd=0)
        brand_card.pack(side="left", pady=10)

        badge = tk.Label(
            brand_card,
            text="WV",
            bg=self._palette["bg_input"],
            fg=self._palette["accent_active"],
            font=("Segoe UI Semibold", 9),
            width=3,
            pady=8,
        )
        badge.pack(side="left", padx=(10, 10), pady=8)

        text_wrap = tk.Frame(brand_card, bg=self._palette["bg_panel_alt"], highlightthickness=0, bd=0)
        text_wrap.pack(side="left", padx=(0, 16), pady=8)

        title_label = tk.Label(
            text_wrap,
            text="Pose Desktop App",
            bg=self._palette["bg_panel_alt"],
            fg=self._palette["fg_primary"],
            font=("Segoe UI Semibold", 10),
        )
        title_label.pack(anchor="w")

        subtitle_label = tk.Label(
            text_wrap,
            text="analysis workspace",
            bg=self._palette["bg_panel_alt"],
            fg=self._palette["fg_muted"],
            font=("Segoe UI", 8),
        )
        subtitle_label.pack(anchor="w")

        divider = tk.Frame(left, bg=self._palette["border"], width=1, highlightthickness=0, bd=0)
        divider.pack(side="left", fill="y", padx=14, pady=14)

        state_chip = tk.Label(
            left,
            text="SANDBOX",
            bg=self._palette["bg_titlebar_chip"],
            fg=self._palette["fg_muted"],
            font=("Segoe UI Semibold", 8),
            padx=10,
            pady=7,
        )
        state_chip.pack(side="left", pady=12)

        status_hint = tk.Label(
            left,
            text="Pose | Tracking | Charts",
            bg=self._palette["bg_titlebar"],
            fg=self._palette["fg_muted"],
            font=("Segoe UI", 8),
        )
        status_hint.pack(side="left", padx=(12, 0), pady=12)

        controls = tk.Frame(
            titlebar,
            bg=self._palette["bg_panel_alt"],
            highlightthickness=1,
            highlightbackground=self._palette["border"],
            bd=0,
        )
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

        drag_targets = (titlebar, left, brand_card, text_wrap, title_label, subtitle_label, state_chip, status_hint, divider)
        for widget in drag_targets:
            widget.bind("<ButtonPress-1>", self._start_window_drag)
            widget.bind("<B1-Motion>", self._drag_window)
            widget.bind("<Double-Button-1>", lambda _event: self._toggle_maximize())

    def _create_titlebar_button(
        self,
        parent: tk.Misc,
        text: str,
        command,
        normal_bg: str | None = None,
        hover_bg: str | None = None,
        active_bg: str | None = None,
    ) -> tk.Label:
        button = tk.Label(
            parent,
            text=text,
            bg=normal_bg or self._palette["bg_titlebar"],
            fg=self._palette["fg_primary"],
            font=("Segoe UI Symbol", 10),
            width=4,
            padx=1,
            pady=1,
            cursor="hand2",
        )
        default_bg = normal_bg or self._palette["bg_titlebar"]
        hover_color = hover_bg or self._palette["bg_hover"]
        active_color = active_bg or hover_color
        button.bind("<Enter>", lambda _event: button.configure(bg=hover_color))
        button.bind("<Leave>", lambda _event: button.configure(bg=default_bg))
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
        # Windows requires temporarily restoring native chrome before iconify.
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
            border_color = ctypes.c_int(0x00403D72)

            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            DWMWA_WINDOW_CORNER_PREFERENCE = 33
            DWMWA_BORDER_COLOR = 34

            dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(dark_mode),
                ctypes.sizeof(dark_mode),
            )
            dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_WINDOW_CORNER_PREFERENCE,
                ctypes.byref(corner_preference),
                ctypes.sizeof(corner_preference),
            )
            dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_BORDER_COLOR,
                ctypes.byref(border_color),
                ctypes.sizeof(border_color),
            )
        except Exception:
            return

    def _on_window_map(self, _event) -> None:
        if str(self.state()) != "iconic":
            self.overrideredirect(True)
            self.after_idle(self._apply_window_chrome)

    def _choose_video(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")],
        )
        if not selected:
            return
        self.selected_video_path = selected
        self.video_path_var.set(selected)
        self.source_var.set("video")
        self._load_video_preview(selected)

    def _choose_aux_yolo_model(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose a spinal curvature estimation model file",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if not selected:
            return
        self.aux_yolo_path_var.set(selected)

    def _is_auto_mode(self) -> bool:
        return self.exercise_var.get().startswith("auto")

    def _resolve_default_auto_model_label(self) -> str:
        default_model_path = get_default_exercise_classifier_model()
        if default_model_path is not None:
            for label, model_path in self.auto_model_options.items():
                if model_path == default_model_path:
                    return label
        return next(iter(self.auto_model_options), "")

    def _refresh_auto_model_options(self) -> None:
        self.auto_model_options = discover_exercise_classifier_models()
        model_labels = list(self.auto_model_options.keys())
        self.auto_model_combobox.configure(values=model_labels)

        current_selection = self.auto_model_var.get()
        if current_selection not in self.auto_model_options:
            self.auto_model_var.set(self._resolve_default_auto_model_label())

        if model_labels:
            self.auto_model_hint_var.set(f"Found {len(model_labels)} models in artifacts.")
        else:
            self.auto_model_var.set("")
            self.auto_model_hint_var.set("No auto-classification models found in the artifacts directory.")

        self._update_auto_model_controls()

    def _update_auto_model_controls(self) -> None:
        if self._is_auto_mode() and self.auto_model_options:
            self.auto_model_combobox.configure(state="readonly")
            return
        self.auto_model_combobox.configure(state="disabled")

    def _on_exercise_mode_changed(self, *_args) -> None:
        self._update_auto_model_controls()

    def _on_auxiliary_point_scale_changed(self, value: str) -> None:
        percentage = max(AUXILIARY_POINT_SCALE_MIN, min(AUXILIARY_POINT_SCALE_MAX, int(round(float(value)))))
        self.aux_point_scale_var.set(float(percentage))
        self.aux_point_scale_hint_var.set(f"{percentage}%")
        self._apply_display_changes()

    def _get_auxiliary_point_radius(self) -> int:
        percentage = max(
            AUXILIARY_POINT_SCALE_MIN,
            min(AUXILIARY_POINT_SCALE_MAX, int(round(self.aux_point_scale_var.get()))),
        )
        return max(1, int(round(AUXILIARY_POINT_BASE_RADIUS * percentage / 100.0)))

    def _build_settings(self) -> AnalysisSettings:
        if self.source_var.get() == "video":
            if not self.selected_video_path or not Path(self.selected_video_path).exists():
                raise ValueError("Select a video file first.")
        if self.use_aux_yolo_var.get() and not Path(self.aux_yolo_path_var.get()).exists():
            raise ValueError("Select a valid model file for spinal curvature estimation.")
        selected_auto_model_path = None
        if self._is_auto_mode():
            selected_label = self.auto_model_var.get()
            selected_auto_model_path = self.auto_model_options.get(selected_label)
            if selected_auto_model_path is None or not selected_auto_model_path.exists():
                raise ValueError("Select a valid model for exercise auto-classification.")
        return AnalysisSettings(
            source_type="camera" if self.source_var.get() == "camera" else "video",
            source_path=self.selected_video_path,
            camera_index=int(self.camera_index_var.get()),
            pose_selection=self.pose_model_var.get(),
            show_pose_preview=bool(self.show_pose_preview_var.get()),
            auxiliary_point_radius=self._get_auxiliary_point_radius(),
            use_barbell_tracking=bool(self.use_barbell_var.get()),
            barbell_selection=self.barbell_model_var.get(),
            use_auxiliary_yolo=bool(self.use_aux_yolo_var.get()),
            auxiliary_yolo_path=self.aux_yolo_path_var.get(),
            pose_confidence_threshold=float(self.pose_conf_var.get()),
            barbell_confidence_threshold=float(self.barbell_conf_var.get()),
            auxiliary_yolo_confidence_threshold=float(self.aux_yolo_conf_var.get()),
            exercise=self.exercise_var.get(),
            auto_exercise_model_path=None if selected_auto_model_path is None else str(selected_auto_model_path),
            aspect_ratio=self.aspect_ratio_var.get(),
            scale_label=self.scale_var.get(),
            reset_barbell_path_each_rep=bool(self.reset_barbell_path_each_rep_var.get()),
        )

    def _start_processing(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            settings = self._build_settings()
        except Exception as exc:
            show_styled_message(self, "Error", str(exc), self._palette, variant="error")
            return

        self.frame_var.set("Frame: 0/0")
        self.rep_var.set("Reps: 0")
        self.active_exercise_var.set(f"Active exercise: {self.exercise_var.get()}")
        self.status_var.set("Starting...")
        self.progress_var.set(0.0)
        self.points_path = None
        self.worker = VideoAnalysisWorker(settings, self.frame_queue, self.state_queue)
        self.worker.start()

    def _pause_processing(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.worker.pause_event.set()
            self.status_var.set("Paused")

    def _resume_processing(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.worker.pause_event.clear()
            self.status_var.set("Resumed")

    def _cancel_processing(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.worker.cancel_event.set()
            self.status_var.set("Cancelling...")

    def _apply_display_changes(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.worker.update_display_settings(
                self.aspect_ratio_var.get(),
                self.scale_var.get(),
                bool(self.show_pose_preview_var.get()),
                self._get_auxiliary_point_radius(),
            )
            return
        self._render_preview_frame()

    def _show_charts(self) -> None:
        if not self.points_path or not Path(self.points_path).exists():
            show_styled_message(
                self,
                "No data",
                "Charts will be available after saving points to a file.",
                self._palette,
                variant="info",
            )
            return
        ChartWindow(self, self.points_path, self.chart_point_var.get(), self._palette)

    def _poll_worker(self) -> None:
        try:
            while True:
                packet = self.frame_queue.get_nowait()
                self._update_frame(packet)
        except queue.Empty:
            pass

        try:
            while True:
                packet = self.state_queue.get_nowait()
                self._update_state(packet)
        except queue.Empty:
            pass

        self.after(15, self._poll_worker)

    def _update_frame(self, packet: FramePacket) -> None:
        self._display_frame(packet.frame_bgr)
        self.frame_var.set(f"Frame: {packet.frame_index}/{packet.total_frames}")
        self.rep_var.set(f"Reps: {packet.reps}")
        self.active_exercise_var.set(f"Active exercise: {packet.active_exercise}")
        self.status_var.set(packet.status_text)
        self.progress_var.set(packet.progress)
        self.points_path = packet.points_path

    def _load_video_preview(self, video_path: str) -> None:
        capture = cv2.VideoCapture(video_path)
        try:
            success, frame = capture.read()
        finally:
            capture.release()
        if not success or frame is None:
            self.preview_frame_bgr = None
            self.video_label.configure(image="", text="Failed to load the first frame")
            return
        self.preview_frame_bgr = frame
        self._render_preview_frame()
        self.status_var.set("Loaded the first frame preview.")

    def _render_preview_frame(self) -> None:
        if self.preview_frame_bgr is None or (self.worker is not None and self.worker.is_alive()):
            return
        display_settings = DisplaySettings(
            self.aspect_ratio_var.get(),
            self.scale_var.get(),
            bool(self.show_pose_preview_var.get()),
            self._get_auxiliary_point_radius(),
        )
        preview_frame = resize_for_display(self.preview_frame_bgr, display_settings)
        self._display_frame(preview_frame)

    def _display_frame(self, frame_bgr: np.ndarray) -> None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        self.video_photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.video_photo, text="")

    def _update_state(self, packet: StatePacket) -> None:
        self.status_var.set(packet.message)
        if packet.points_path:
            self.points_path = packet.points_path
        if packet.kind == "error":
            show_styled_message(self, "Processing error", packet.message, self._palette, variant="error")
        elif packet.kind == "done":
            self.progress_var.set(1.0)
        elif packet.kind == "status":
            self.progress_var.set(0.0)

    def _on_close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.worker.cancel_event.set()
            self.worker.join(timeout=1.0)
        self.destroy()


def main() -> None:
    app = PoseDesktopApp()
    app.mainloop()
