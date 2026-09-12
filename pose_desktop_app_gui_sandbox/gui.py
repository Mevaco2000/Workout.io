from __future__ import annotations

import ctypes
import json
import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .charts import ChartWindow
from .config import (
    AUXILIARY_KEYPOINT_MODEL_OPTIONS,
    BARBELL_MODEL_OPTIONS,
    CHART_OPTIONS,
    CHART_POINTS,
    DEFAULT_AUXILIARY_KEYPOINT_MODEL,
    DEFAULT_AUX_YOLO_MODEL_PATH,
    DISPLAY_ASPECT_OPTIONS,
    EXERCISES,
    POSE_MODEL_OPTIONS,
    RUNTIME_ROOT,
    discover_exercise_classifier_models,
    get_default_exercise_classifier_model,
)
from .models import DisplaySettings, resize_for_display
from .worker import AnalysisSettings, FramePacket, StatePacket, VideoAnalysisWorker


AUXILIARY_POINT_BASE_RADIUS = 4
AUXILIARY_POINT_SCALE_MIN = 100
AUXILIARY_POINT_SCALE_MAX = 1000
MODEL_CONFIG_PATH = RUNTIME_ROOT / "gui_last_models.json"
LEGACY_MODEL_CONFIG_PATH = RUNTIME_ROOT / "gui_sandbox_last_models.json"
VIDEO_ROTATION_OPTIONS = {
    "No rotation": "none",
    "Rotate 90 right": "cw90",
    "Rotate 90 left": "ccw90",
}


def _resolve_icon_assets_dir() -> Path:
    repo_local_assets = Path(__file__).resolve().parents[1] / "apka_treningowa"
    parent_assets = Path(__file__).resolve().parents[2] / "apka_treningowa"
    if repo_local_assets.exists():
        return repo_local_assets
    return parent_assets


ICON_ASSETS_DIR = _resolve_icon_assets_dir()
APP_DISPLAY_NAME = "Desktop System for Strength Training Monitoring and Analysis"
APP_AUTHOR = "Rafal Wysocki"
ICON_FILES = {
    "strength": "icons8-strength-50.png",
    "settings": "icons8-settings-50.png",
    "resume": "icons8-resume-button-50.png",
    "play": "icons8-play-50.png",
    "pause": "icons8-pause-50.png",
    "info": "icons8-info-50.png",
    "charts": "icons8-charts-50.png",
    "cancel": "icons8-cancel-50.png",
}
INFO_BANNER_CANDIDATES = [
    "cropped-Logo2-scaled-1-2048x204.jpg",
    "banner.png",
    "info_banner.png",
    "logos_banner.png",
    "politechnika_banner.png",
]
INFO_PROFILE_CANDIDATES = [
    "DSC_4520 1.jpg",
    "author.png",
    "author.jpg",
    "rafal_wysocki.png",
    "rafal_wysocki.jpg",
    "profile.png",
    "profile.jpg",
]

SPLASH_LOGO_CANDIDATES = [
    "logo.jpg",
    "logo.png",
    "logo.jpeg",
]

SPLASH_CROPPED_LOGO_CANDIDATES = [
    "cropped_logo.png",
    "cropped_logo.jpg",
    "cropped_logo.jpeg",
    "cropped-Logo2-scaled-1-2048x204.jpg",
]


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
        self.lift(master)
        try:
            self.attributes("-topmost", True)
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


class SplashScreen(tk.Toplevel):
    def __init__(self, master: tk.Misc, palette: dict[str, str], duration_ms: int = 2400) -> None:
        super().__init__(master)
        self._palette = palette
        self._duration_ms = max(1000, int(duration_ms))
        self._logo_photo: ImageTk.PhotoImage | None = None
        self._cropped_logo_photo: ImageTk.PhotoImage | None = None

        self.overrideredirect(True)
        self.configure(bg=self._palette["bg_root"])
        self.geometry("760x430")
        self.minsize(680, 380)
        self._build_layout()
        self._center_on_screen()
        self.after(self._duration_ms, self._close)

    def _build_layout(self) -> None:
        outer = tk.Frame(self, bg=self._palette["border"], highlightthickness=0, bd=0)
        outer.pack(fill="both", expand=True, padx=16, pady=16)

        body = tk.Frame(outer, bg=self._palette["bg_panel"], highlightthickness=0, bd=0)
        body.pack(fill="both", expand=True, padx=1, pady=1)

        title = tk.Label(
            body,
            text=APP_DISPLAY_NAME,
            bg=self._palette["bg_panel"],
            fg=self._palette["fg_primary"],
            font=("Segoe UI Semibold", 20),
            justify="center",
            wraplength=760,
        )
        title.pack(padx=24, pady=(34, 10))

        cropped_logo = self._load_logo_photo(SPLASH_CROPPED_LOGO_CANDIDATES, max_width=560, max_height=96)
        if cropped_logo is not None:
            self._cropped_logo_photo = cropped_logo
            tk.Label(body, image=self._cropped_logo_photo, bg=self._palette["bg_panel"]).pack(padx=20, pady=(0, 14))

        logo = self._load_logo_photo(SPLASH_LOGO_CANDIDATES, max_width=220, max_height=220)
        if logo is not None:
            self._logo_photo = logo
            tk.Label(body, image=self._logo_photo, bg=self._palette["bg_panel"]).pack(padx=20, pady=(0, 14))

        tk.Label(
            body,
            text=f"Author: {APP_AUTHOR}",
            bg=self._palette["bg_panel"],
            fg=self._palette["fg_muted"],
            font=("Segoe UI", 11),
            justify="center",
        ).pack(padx=24, pady=(0, 20))

    def _center_on_screen(self) -> None:
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = max((self.winfo_screenwidth() - width) // 2, 0)
        y = max((self.winfo_screenheight() - height) // 2, 0)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _load_logo_photo(self, candidates: list[str], *, max_width: int, max_height: int) -> ImageTk.PhotoImage | None:
        for candidate in candidates:
            candidate_path = ICON_ASSETS_DIR / candidate
            if not candidate_path.exists():
                continue
            try:
                image = Image.open(candidate_path).convert("RGBA")
                width, height = image.size
                if width <= 0 or height <= 0:
                    continue
                scale = min(1.0, float(max_width) / float(width), float(max_height) / float(height))
                target_width = max(1, int(round(width * scale)))
                target_height = max(1, int(round(height * scale)))
                resampling = getattr(Image, "Resampling", Image)
                resized = image.resize((target_width, target_height), resampling.LANCZOS)
                return ImageTk.PhotoImage(resized)
            except Exception:
                continue
        return None

    def _close(self) -> None:
        if self.winfo_exists():
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
        self.withdraw()
        self.title(APP_DISPLAY_NAME)
        self.geometry("1320x820")
        self.minsize(1120, 700)
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
        self._settings_panel_visible = False
        self._icon_images: dict[str, ImageTk.PhotoImage] = {}
        self._app_icon_photo: tk.PhotoImage | None = None
        self._info_banner_photo: ImageTk.PhotoImage | None = None
        self._info_profile_photo: ImageTk.PhotoImage | None = None
        self._configure_styles()

        self.frame_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
        self.state_queue: queue.Queue[StatePacket] = queue.Queue(maxsize=1)
        self.worker: VideoAnalysisWorker | None = None
        self.video_photo: ImageTk.PhotoImage | None = None
        self.preview_frame_bgr: np.ndarray | None = None
        self.points_path: str | None = None
        self.selected_video_path: str | None = None
        self.pose_target_point_norm: tuple[float, float] | None = None
        self.last_runtime_status_text: str = "Ready"

        self.source_var = tk.StringVar(value="video")
        self.video_path_var = tk.StringVar(value="No file selected")
        self.camera_index_var = tk.IntVar(value=0)
        self.pose_model_var = tk.StringVar(value=list(POSE_MODEL_OPTIONS.keys())[0])
        self.pose_person_index_var = tk.IntVar(value=1)
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
        self.video_scale_percent_var = tk.DoubleVar(value=100.0)
        self.video_scale_hint_var = tk.StringVar(value="100%")
        self.video_rotation_var = tk.StringVar(value="No rotation")
        self.chart_point_var = tk.StringVar(value=CHART_OPTIONS[0])
        self.aux_keypoint_model_var = tk.StringVar(value=DEFAULT_AUXILIARY_KEYPOINT_MODEL)
        self.enable_face_blur_var = tk.BooleanVar(value=False)
        self.always_on_top_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self.frame_var = tk.StringVar(value="Frame: 0/0")
        self.rep_var = tk.StringVar(value="Reps: 0")
        self.active_exercise_var = tk.StringVar(value=f"Active exercise: {self.exercise_var.get()}")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.auto_model_options: dict[str, Path] = {}
        self.root_content = None
        self.tools_sidebar_shell = None
        self.sidebar_shell = None
        self.sidebar_wrap_labels: list[ttk.Label] = []
        self._hover_tooltips: list[HoverTooltip] = []
        self._restore_geometry: str | None = None
        self._pending_auto_model_path: str | None = None

        self._load_icon_images()
        self._apply_app_icon()
        self._load_last_model_configuration()

        self._build_layout()
        self._apply_always_on_top()
        self._apply_sidebar_width()
        self._refresh_auto_model_options()
        self.exercise_var.trace_add("write", self._on_exercise_mode_changed)
        self.after_idle(self._apply_window_chrome)
        self.after(15, self._poll_worker)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Map>", self._on_window_map)

    def _load_icon_images(self) -> None:
        self._icon_images.clear()
        for icon_name, icon_file in ICON_FILES.items():
            icon_path = ICON_ASSETS_DIR / icon_file
            if not icon_path.exists():
                continue
            try:
                image = Image.open(icon_path).convert("RGBA")
                resampling = getattr(Image, "Resampling", Image)
                resized = image.resize((20, 20), resampling.LANCZOS)
                self._icon_images[icon_name] = ImageTk.PhotoImage(resized)
            except Exception:
                continue

    def _apply_app_icon(self) -> None:
        icon_path = ICON_ASSETS_DIR / ICON_FILES["strength"]
        if not icon_path.exists():
            return
        try:
            self._app_icon_photo = tk.PhotoImage(file=str(icon_path))
            self.iconphoto(True, self._app_icon_photo)
        except Exception:
            self._app_icon_photo = None

    def _center_child_window(self, window: tk.Toplevel) -> None:
        window.update_idletasks()
        self.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = self.winfo_x() + max((self.winfo_width() - width) // 2, 0)
        y = self.winfo_y() + max((self.winfo_height() - height) // 2, 0)
        window.geometry(f"{width}x{height}+{x}+{y}")

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
            self.root_content.columnconfigure(1, minsize=width + 24 if self._settings_panel_visible else 0)
        if self.sidebar_shell is not None:
            self.sidebar_shell.configure(width=width + 24)
            if self._settings_panel_visible:
                self.sidebar_shell.grid()
            else:
                self.sidebar_shell.grid_remove()
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

    def _create_icon_button(
        self,
        parent: tk.Misc,
        *,
        icon_key: str,
        command,
        fallback_text: str,
        tooltip_text: str,
        accent: bool = False,
    ) -> tk.Button:
        normal_bg = self._palette["accent"] if accent else self._palette["bg_input"]
        hover_bg = self._palette["accent_active"] if accent else self._palette["bg_hover"]
        active_bg = "#b7d9ff" if accent else self._palette["accent"]
        foreground = self._palette["bg_root"] if accent else self._palette["fg_primary"]
        icon_image = self._icon_images.get(icon_key)
        button = tk.Button(
            parent,
            image=icon_image,
            text="" if icon_image is not None else fallback_text,
            command=command,
            bg=normal_bg,
            fg=foreground,
            activebackground=hover_bg,
            activeforeground=foreground,
            relief="flat",
            bd=0,
            highlightthickness=0,
            width=34,
            height=34,
            padx=0,
            pady=0,
            cursor="hand2",
            font=("Segoe UI Semibold", 8),
        )
        button.bind("<Enter>", lambda _event: button.configure(bg=hover_bg))
        button.bind("<Leave>", lambda _event: button.configure(bg=normal_bg))
        button.bind("<ButtonPress-1>", lambda _event: button.configure(bg=active_bg))
        button.bind("<ButtonRelease-1>", lambda _event: button.configure(bg=hover_bg))
        self._hover_tooltips.append(HoverTooltip(button, tooltip_text, self._palette))
        return button

    def _toggle_settings_panel(self) -> None:
        self._settings_panel_visible = not self._settings_panel_visible
        self._apply_sidebar_width()
        if hasattr(self, "settings_toggle_button"):
            if self._settings_panel_visible:
                self.settings_toggle_button.configure(bg=self._palette["accent"])
            else:
                self.settings_toggle_button.configure(bg=self._palette["bg_input"])

    def _show_info_dialog(self) -> None:
        info_window = tk.Toplevel(self)
        info_window.overrideredirect(True)
        info_window.geometry("980x560")
        info_window.minsize(840, 460)
        info_window.transient(self)
        info_window.configure(bg=self._palette["bg_root"])
        info_window.bind("<Escape>", lambda _event: info_window.destroy())
        info_window.protocol("WM_DELETE_WINDOW", info_window.destroy)
        info_window.grab_set()
        self._center_child_window(info_window)

        outer = tk.Frame(info_window, bg=self._palette["border"], highlightthickness=0, bd=0)
        outer.pack(fill="both", expand=True)
        body = tk.Frame(outer, bg=self._palette["bg_panel"], highlightthickness=0, bd=0)
        body.pack(fill="both", expand=True, padx=1, pady=1)

        close_row = tk.Frame(body, bg=self._palette["bg_panel"], highlightthickness=0, bd=0)
        close_row.pack(fill="x", padx=10, pady=(10, 0))
        close_button = tk.Button(
            close_row,
            text="✕",
            command=info_window.destroy,
            bg=self._palette["bg_input"],
            fg=self._palette["fg_primary"],
            activebackground=self._palette["danger"],
            activeforeground=self._palette["bg_root"],
            relief="flat",
            bd=0,
            padx=10,
            pady=5,
            font=("Segoe UI Semibold", 10),
            cursor="hand2",
        )
        close_button.pack(side="right")

        tk.Label(
            body,
            text=APP_DISPLAY_NAME,
            bg=self._palette["bg_panel"],
            fg=self._palette["fg_primary"],
            font=("Segoe UI Semibold", 12),
            justify="center",
            anchor="center",
        ).pack(anchor="center", padx=16, pady=(8, 4))

        tk.Label(
            body,
            text=(
                "Division of Electronic Systems and Signal Processing\n"
                "Poznan University of Technology. Faculty of Automation, Robotics and Electrical Engineering. "
                "Institute of Automation and Robotics"
            ),
            bg=self._palette["bg_panel"],
            fg=self._palette["fg_muted"],
            font=("Segoe UI", 9),
            justify="center",
            anchor="center",
        ).pack(anchor="center", padx=16, pady=(0, 10))

        tk.Label(
            body,
            text=f"Author: {APP_AUTHOR}",
            bg=self._palette["bg_panel"],
            fg=self._palette["fg_primary"],
            font=("Segoe UI Semibold", 10),
            justify="center",
            anchor="center",
        ).pack(anchor="center", padx=16, pady=(0, 8))

        author_shell = tk.Frame(body, bg=self._palette["border"], highlightthickness=0, bd=0)
        author_shell.pack(fill="x", padx=16, pady=(0, 12))
        author_panel = tk.Frame(author_shell, bg=self._palette["bg_panel_alt"], highlightthickness=0, bd=0)
        author_panel.pack(fill="x", padx=1, pady=1)

        author_panel.grid_columnconfigure(1, weight=1)

        profile_photo = self._load_info_profile_image(max_height=150)
        if profile_photo is not None:
            image_label = tk.Label(author_panel, image=profile_photo, bg=self._palette["bg_panel_alt"])
            image_label.grid(row=0, column=0, sticky="nw", padx=(12, 12), pady=12)
        else:
            image_fallback = tk.Label(
                author_panel,
                text="Photo not found",
                bg=self._palette["bg_input"],
                fg=self._palette["fg_muted"],
                font=("Segoe UI", 9),
                width=18,
                height=8,
            )
            image_fallback.grid(row=0, column=0, sticky="nw", padx=(12, 12), pady=12)

        bio_label = tk.Label(
            author_panel,
            text=(
                "I am a graduate of the Military University of Technology and currently "
                "a researcher at Poznan University of Technology. My research interests "
                "focus on machine learning, with particular emphasis on computer vision "
                "and its potential applications in strength training and unmanned aerial "
                "vehicles (UAVs)."
            ),
            bg=self._palette["bg_panel_alt"],
            fg=self._palette["fg_primary"],
            font=("Segoe UI", 10),
            justify="center",
            wraplength=620,
            anchor="center",
        )
        bio_label.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)

        banner = self._load_info_banner_image(max_width=700)
        if banner is not None:
            banner_label = tk.Label(body, image=banner, bg=self._palette["bg_panel"])
            banner_label.pack(anchor="center", padx=16, pady=(2, 10))
        else:
            tk.Label(
                body,
                text="Nie znaleziono banera w folderze apka_treningowa.\n"
                "Dodaj plik banner.png albo info_banner.png, aby wyswietlac logotypy w oknie Info.",
                bg=self._palette["bg_panel"],
                fg=self._palette["fg_muted"],
                font=("Segoe UI", 9),
                justify="left",
                anchor="w",
            ).pack(fill="x", padx=16, pady=(4, 12))

        try:
            self._center_child_window(info_window)
            info_window.lift(self)
            info_window.attributes("-topmost", True)
        except tk.TclError:
            pass

    def _load_info_banner_image(self, max_width: int) -> ImageTk.PhotoImage | None:
        for candidate in INFO_BANNER_CANDIDATES:
            candidate_path = ICON_ASSETS_DIR / candidate
            if not candidate_path.exists():
                continue
            try:
                image = Image.open(candidate_path).convert("RGBA")
                width, height = image.size
                if width <= 0 or height <= 0:
                    continue
                scale = min(1.0, float(max_width) / float(width))
                target_width = max(1, int(round(width * scale)))
                target_height = max(1, int(round(height * scale)))
                resampling = getattr(Image, "Resampling", Image)
                resized = image.resize((target_width, target_height), resampling.LANCZOS)
                self._info_banner_photo = ImageTk.PhotoImage(resized)
                return self._info_banner_photo
            except Exception:
                continue
        self._info_banner_photo = None
        return None

    def _load_info_profile_image(self, max_height: int) -> ImageTk.PhotoImage | None:
        for candidate in INFO_PROFILE_CANDIDATES:
            candidate_path = ICON_ASSETS_DIR / candidate
            if not candidate_path.exists():
                continue
            try:
                image = Image.open(candidate_path).convert("RGBA")
                width, height = image.size
                if width <= 0 or height <= 0:
                    continue
                scale = min(1.0, float(max_height) / float(height))
                target_width = max(1, int(round(width * scale)))
                target_height = max(1, int(round(height * scale)))
                resampling = getattr(Image, "Resampling", Image)
                resized = image.resize((target_width, target_height), resampling.LANCZOS)
                self._info_profile_photo = ImageTk.PhotoImage(resized)
                return self._info_profile_photo
            except Exception:
                continue
        self._info_profile_photo = None
        return None

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
        root.columnconfigure(1, weight=0)
        root.columnconfigure(2, weight=1)
        root.rowconfigure(0, weight=1)
        self.root_content = root

        tools_sidebar_shell = tk.Frame(root, bg=self._palette["border"], highlightthickness=0, bd=0, width=64)
        tools_sidebar_shell.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        self.tools_sidebar_shell = tools_sidebar_shell
        tools_sidebar = ttk.Frame(tools_sidebar_shell, style="Panel.TFrame", padding=(8, 10))
        tools_sidebar.pack(fill="both", expand=True, padx=1, pady=1)

        tool_buttons = tk.Frame(tools_sidebar, bg=self._palette["bg_panel"], highlightthickness=0, bd=0)
        tool_buttons.pack(fill="y", expand=True)

        self.settings_toggle_button = self._create_icon_button(
            tool_buttons,
            icon_key="settings",
            command=self._toggle_settings_panel,
            fallback_text="S",
            tooltip_text="Settings",
        )
        self.settings_toggle_button.pack(pady=(4, 10))
        self.info_button = self._create_icon_button(
            tool_buttons,
            icon_key="info",
            command=self._show_info_dialog,
            fallback_text="i",
            tooltip_text="Info",
        )
        self.info_button.pack(pady=(0, 4))

        sidebar_shell = tk.Frame(root, bg=self._palette["border"], highlightthickness=0, bd=0)
        sidebar_shell.grid(row=0, column=1, sticky="ns", padx=(0, 12))
        self.sidebar_shell = sidebar_shell
        sidebar = ttk.Frame(sidebar_shell, style="Panel.TFrame", padding=10)
        sidebar.pack(fill="both", expand=True, padx=1, pady=1)
        sidebar.rowconfigure(1, weight=1)
        sidebar.columnconfigure(0, weight=1)

        sidebar_header = ttk.Frame(sidebar, style="Panel.TFrame")
        sidebar_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(sidebar_header, text="Settings", style="Panel.TLabel", font=("Segoe UI Semibold", 11)).pack(anchor="w")
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
        ttk.Checkbutton(
            source_frame,
            text="Always on top",
            variable=self.always_on_top_var,
            command=self._apply_always_on_top,
        ).pack(anchor="w", pady=(8, 0))

        model_frame = ttk.LabelFrame(controls, text="Models", padding=8)
        model_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(model_frame, text="Pose estimation").pack(anchor="w")
        self.pose_model_combobox = ttk.Combobox(
            model_frame,
            textvariable=self.pose_model_var,
            values=list(POSE_MODEL_OPTIONS.keys()),
            state="readonly",
        )
        self.pose_model_combobox.pack(fill="x", pady=(4, 4))
        ttk.Button(model_frame, text="Browse pose model file...", command=self._choose_pose_model_file).pack(fill="x", pady=(0, 8))
        person_row = ttk.Frame(model_frame, style="Panel.TFrame")
        person_row.pack(fill="x", pady=(0, 8))
        ttk.Label(person_row, text="Tracked person (YOLO, left->right):").pack(side="left")
        ttk.Spinbox(person_row, from_=1, to=6, textvariable=self.pose_person_index_var, width=6).pack(side="right")
        ttk.Label(
            model_frame,
            text="Tip: click person in preview to lock tracking by position.",
            style="PanelMuted.TLabel",
            wraplength=self._get_sidebar_wraplength(),
        ).pack(anchor="w", pady=(0, 8))
        ttk.Checkbutton(
            model_frame,
            text="Show pose estimation preview",
            variable=self.show_pose_preview_var,
            command=self._apply_display_changes,
        ).pack(anchor="w", pady=(0, 8))
        ttk.Checkbutton(
            model_frame,
            text="Face detection + blur",
            variable=self.enable_face_blur_var,
        ).pack(anchor="w", pady=(0, 8))
        ttk.Checkbutton(model_frame, text="Barbell tracking", variable=self.use_barbell_var).pack(anchor="w")
        self.barbell_model_combobox = ttk.Combobox(
            model_frame,
            textvariable=self.barbell_model_var,
            values=list(BARBELL_MODEL_OPTIONS.keys()),
            state="readonly",
        )
        self.barbell_model_combobox.pack(fill="x", pady=(4, 4))
        ttk.Button(model_frame, text="Browse barbell model file...", command=self._choose_barbell_model_file).pack(fill="x", pady=(0, 0))
        ttk.Checkbutton(
            model_frame,
            text="Reset barbell path after each rep",
            variable=self.reset_barbell_path_each_rep_var,
        ).pack(anchor="w", pady=(8, 0))
        ttk.Checkbutton(model_frame, text="Spinal curvature estimation", variable=self.use_aux_yolo_var).pack(anchor="w", pady=(10, 0))
        ttk.Label(model_frame, text="Spinal keypoint model").pack(anchor="w", pady=(8, 0))
        ttk.Combobox(
            model_frame,
            textvariable=self.aux_keypoint_model_var,
            values=AUXILIARY_KEYPOINT_MODEL_OPTIONS,
            state="readonly",
        ).pack(fill="x", pady=(4, 4))
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
        scale_row = ttk.Frame(workout_frame, style="Panel.TFrame")
        scale_row.pack(fill="x")
        ttk.Label(scale_row, text="Video scale").pack(side="left")
        ttk.Label(scale_row, textvariable=self.video_scale_hint_var, style="PanelMuted.TLabel").pack(side="right")
        ttk.Scale(
            workout_frame,
            from_=20,
            to=200,
            variable=self.video_scale_percent_var,
            orient="horizontal",
            command=self._on_video_scale_changed,
        ).pack(fill="x", pady=(4, 0))
        ttk.Label(workout_frame, text="Video rotation").pack(anchor="w", pady=(8, 0))
        rotation_box = ttk.Combobox(
            workout_frame,
            textvariable=self.video_rotation_var,
            values=list(VIDEO_ROTATION_OPTIONS.keys()),
            state="readonly",
        )
        rotation_box.pack(fill="x", pady=(4, 0))
        rotation_box.bind("<<ComboboxSelected>>", lambda _event: self._apply_display_changes())
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
        viewer_shell.grid(row=0, column=2, sticky="nsew")
        viewer = ttk.Frame(viewer_shell, style="PanelAlt.TFrame", padding=12)
        viewer.pack(fill="both", expand=True, padx=1, pady=1)
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(2, weight=1)

        viewer_header = ttk.Frame(viewer, style="PanelAlt.TFrame")
        viewer_header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        viewer_header.columnconfigure(0, weight=1)
        ttk.Label(viewer_header, text="Session Monitor", style="PanelAlt.TLabel", font=("Segoe UI Semibold", 11)).grid(row=0, column=0, sticky="w")

        button_frame = ttk.Frame(viewer, style="PanelAlt.TFrame", padding=(0, 0, 0, 10))
        button_frame.grid(row=1, column=0, sticky="ew")
        button_frame.columnconfigure(0, weight=1)
        action_buttons = tk.Frame(button_frame, bg=self._palette["bg_panel_alt"], highlightthickness=0, bd=0)
        action_buttons.grid(row=0, column=0)
        self.start_icon_button = self._create_icon_button(
            action_buttons,
            icon_key="play",
            command=self._start_processing,
            fallback_text="Play",
            tooltip_text="Play",
            accent=True,
        )
        self.start_icon_button.grid(row=0, column=0, padx=6)
        self.pause_icon_button = self._create_icon_button(
            action_buttons,
            icon_key="pause",
            command=self._pause_processing,
            fallback_text="Pause",
            tooltip_text="Pause",
        )
        self.pause_icon_button.grid(row=0, column=1, padx=6)
        self.resume_icon_button = self._create_icon_button(
            action_buttons,
            icon_key="resume",
            command=self._resume_processing,
            fallback_text="Resume",
            tooltip_text="Resume",
        )
        self.resume_icon_button.grid(row=0, column=2, padx=6)
        self.cancel_icon_button = self._create_icon_button(
            action_buttons,
            icon_key="cancel",
            command=self._cancel_processing,
            fallback_text="Cancel",
            tooltip_text="Cancel",
        )
        self.cancel_icon_button.grid(row=0, column=3, padx=6)
        self.charts_icon_button = self._create_icon_button(
            action_buttons,
            icon_key="charts",
            command=self._show_charts,
            fallback_text="Charts",
            tooltip_text="Charts",
        )
        self.charts_icon_button.grid(row=0, column=4, padx=6)

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
        self.video_label.bind("<Button-1>", self._on_video_click)

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
        brand_card.pack(pady=10)

        strength_icon = self._icon_images.get("strength")
        if strength_icon is not None:
            badge = tk.Label(
                brand_card,
                image=strength_icon,
                bg=self._palette["bg_input"],
                padx=10,
                pady=8,
            )
        else:
            badge = tk.Label(
                brand_card,
                text="STR",
                bg=self._palette["bg_input"],
                fg=self._palette["accent_active"],
                font=("Segoe UI Semibold", 9),
                width=4,
                pady=8,
            )
        badge.pack(side="left", padx=(10, 10), pady=8)

        text_wrap = tk.Frame(brand_card, bg=self._palette["bg_panel_alt"], highlightthickness=0, bd=0)
        text_wrap.pack(side="left", padx=(0, 16), pady=8)

        title_label = tk.Label(
            text_wrap,
            text=APP_DISPLAY_NAME,
            bg=self._palette["bg_panel_alt"],
            fg=self._palette["fg_primary"],
            font=("Segoe UI Semibold", 10),
            justify="center",
            anchor="center",
        )
        title_label.pack(anchor="center")

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

        drag_targets = (titlebar, left, brand_card, text_wrap, title_label)
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
            if self._restore_geometry:
                self.geometry(self._restore_geometry)
            else:
                self.state("normal")
            self._is_maximized = False
            self.maximize_button.configure(text="□")
            return

        self._restore_geometry = self.geometry()
        self._maximize_to_work_area()
        self._is_maximized = True
        self.maximize_button.configure(text="❐")

    def _maximize_to_work_area(self) -> None:
        if self.tk.call("tk", "windowingsystem") != "win32":
            self.state("zoomed")
            return

        try:
            rect = ctypes.wintypes.RECT()
            SPI_GETWORKAREA = 0x0030
            result = ctypes.windll.user32.SystemParametersInfoW(SPI_GETWORKAREA, 0, ctypes.byref(rect), 0)
            if not result:
                self.state("zoomed")
                return
            width = max(1, int(rect.right - rect.left))
            height = max(1, int(rect.bottom - rect.top))
            self.geometry(f"{width}x{height}+{int(rect.left)}+{int(rect.top)}")
        except Exception:
            self.state("zoomed")

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
            self.after_idle(self._apply_always_on_top)
            if self._is_maximized:
                self.after_idle(self._maximize_to_work_area)

    def _apply_always_on_top(self) -> None:
        try:
            self.attributes("-topmost", bool(self.always_on_top_var.get()))
        except tk.TclError:
            return

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
        self._save_last_model_configuration()

    def _make_unique_model_label(self, path: Path, existing_labels: dict) -> str:
        label = f"Custom: {path.name}"
        if label not in existing_labels:
            return label
        return f"Custom: {path}"

    def _choose_pose_model_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose a pose estimation model file",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if not selected:
            return
        path = Path(selected)
        label = self._register_custom_pose_model(path)
        self.pose_model_combobox.configure(values=list(POSE_MODEL_OPTIONS.keys()))
        self.pose_model_var.set(label)
        self._save_last_model_configuration()

    def _choose_barbell_model_file(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose a barbell tracking model file",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
        if not selected:
            return
        path = Path(selected)
        label = self._register_custom_barbell_model(path)
        self.barbell_model_combobox.configure(values=list(BARBELL_MODEL_OPTIONS.keys()))
        self.barbell_model_var.set(label)
        self._save_last_model_configuration()

    def _register_custom_pose_model(self, path: Path) -> str:
        normalized = path.resolve()
        for label, option in POSE_MODEL_OPTIONS.items():
            if Path(option["reference"]).resolve() == normalized:
                return label
        label = self._make_unique_model_label(path, POSE_MODEL_OPTIONS)
        POSE_MODEL_OPTIONS[label] = {"backend": "yolo", "reference": path}
        return label

    def _register_custom_barbell_model(self, path: Path) -> str:
        normalized = path.resolve()
        for label, existing_path in BARBELL_MODEL_OPTIONS.items():
            if Path(existing_path).resolve() == normalized:
                return label
        label = self._make_unique_model_label(path, BARBELL_MODEL_OPTIONS)
        BARBELL_MODEL_OPTIONS[label] = path
        return label

    def _resolve_pose_model_path(self) -> str | None:
        option = POSE_MODEL_OPTIONS.get(self.pose_model_var.get())
        if not option:
            return None
        return str(Path(option["reference"]))

    def _resolve_barbell_model_path(self) -> str | None:
        option = BARBELL_MODEL_OPTIONS.get(self.barbell_model_var.get())
        if option is None:
            return None
        return str(Path(option))

    def _load_last_model_configuration(self) -> None:
        self._pending_auto_model_path = None
        config_path = MODEL_CONFIG_PATH if MODEL_CONFIG_PATH.exists() else LEGACY_MODEL_CONFIG_PATH
        if not config_path.exists():
            return
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return

        pose_selection = data.get("pose_selection")
        if isinstance(pose_selection, str) and pose_selection in POSE_MODEL_OPTIONS:
            self.pose_model_var.set(pose_selection)
        pose_path = data.get("pose_model_path")
        if isinstance(pose_path, str) and pose_path:
            pose_model_file = Path(pose_path)
            if pose_model_file.exists():
                self.pose_model_var.set(self._register_custom_pose_model(pose_model_file))

        barbell_selection = data.get("barbell_selection")
        if isinstance(barbell_selection, str) and barbell_selection in BARBELL_MODEL_OPTIONS:
            self.barbell_model_var.set(barbell_selection)
        barbell_path = data.get("barbell_model_path")
        if isinstance(barbell_path, str) and barbell_path:
            barbell_model_file = Path(barbell_path)
            if barbell_model_file.exists():
                self.barbell_model_var.set(self._register_custom_barbell_model(barbell_model_file))

        aux_yolo_path = data.get("auxiliary_yolo_path")
        if isinstance(aux_yolo_path, str) and aux_yolo_path:
            self.aux_yolo_path_var.set(aux_yolo_path)

        use_barbell_tracking = data.get("use_barbell_tracking")
        if isinstance(use_barbell_tracking, bool):
            self.use_barbell_var.set(use_barbell_tracking)

        use_auxiliary_yolo = data.get("use_auxiliary_yolo")
        if isinstance(use_auxiliary_yolo, bool):
            self.use_aux_yolo_var.set(use_auxiliary_yolo)

        enable_face_blur = data.get("enable_face_blur")
        if isinstance(enable_face_blur, bool):
            self.enable_face_blur_var.set(enable_face_blur)

        exercise = data.get("exercise")
        if isinstance(exercise, str) and exercise in EXERCISES:
            self.exercise_var.set(exercise)

        auxiliary_keypoint_model = data.get("auxiliary_keypoint_model")
        if isinstance(auxiliary_keypoint_model, str) and auxiliary_keypoint_model in AUXILIARY_KEYPOINT_MODEL_OPTIONS:
            self.aux_keypoint_model_var.set(auxiliary_keypoint_model)

        video_scale_percent = data.get("video_scale_percent")
        if isinstance(video_scale_percent, (int, float)):
            self._set_video_scale_percent(float(video_scale_percent))
        else:
            scale_label = data.get("scale_label")
            if isinstance(scale_label, str) and scale_label.strip().endswith("%"):
                try:
                    self._set_video_scale_percent(float(scale_label.strip().replace("%", "")))
                except ValueError:
                    pass

        auto_model_path = data.get("auto_exercise_model_path")
        if isinstance(auto_model_path, str) and auto_model_path:
            self._pending_auto_model_path = auto_model_path

    def _save_last_model_configuration(self) -> None:
        selected_auto_model = self.auto_model_options.get(self.auto_model_var.get())
        payload = {
            "pose_selection": self.pose_model_var.get(),
            "pose_model_path": self._resolve_pose_model_path(),
            "barbell_selection": self.barbell_model_var.get(),
            "barbell_model_path": self._resolve_barbell_model_path(),
            "auxiliary_yolo_path": self.aux_yolo_path_var.get(),
            "auxiliary_keypoint_model": self.aux_keypoint_model_var.get(),
            "enable_face_blur": bool(self.enable_face_blur_var.get()),
            "use_barbell_tracking": bool(self.use_barbell_var.get()),
            "use_auxiliary_yolo": bool(self.use_aux_yolo_var.get()),
            "exercise": self.exercise_var.get(),
            "auto_exercise_model_path": None if selected_auto_model is None else str(selected_auto_model),
            "video_scale_percent": float(self.video_scale_percent_var.get()),
            "scale_label": self._get_scale_label(),
        }
        try:
            MODEL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            MODEL_CONFIG_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        except Exception:
            return

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

        if self._pending_auto_model_path:
            pending_path = Path(self._pending_auto_model_path)
            for label, model_path in self.auto_model_options.items():
                if model_path.resolve() == pending_path.resolve():
                    self.auto_model_var.set(label)
                    self._pending_auto_model_path = None
                    break

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

    def _set_video_scale_percent(self, value: float) -> None:
        percentage = max(20, min(200, int(round(float(value)))))
        self.video_scale_percent_var.set(float(percentage))
        self.video_scale_hint_var.set(f"{percentage}%")
        self.scale_var.set(f"{percentage}%")

    def _on_video_scale_changed(self, value: str) -> None:
        self._set_video_scale_percent(float(value))
        self._apply_display_changes()

    def _get_scale_label(self) -> str:
        return f"{int(round(self.video_scale_percent_var.get()))}%"

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
            pose_person_index=max(1, int(self.pose_person_index_var.get())),
            pose_target_x_norm=None if self.pose_target_point_norm is None else float(self.pose_target_point_norm[0]),
            pose_target_y_norm=None if self.pose_target_point_norm is None else float(self.pose_target_point_norm[1]),
            show_pose_preview=bool(self.show_pose_preview_var.get()),
            auxiliary_point_radius=self._get_auxiliary_point_radius(),
            use_barbell_tracking=bool(self.use_barbell_var.get()),
            barbell_selection=self.barbell_model_var.get(),
            use_auxiliary_yolo=bool(self.use_aux_yolo_var.get()),
            auxiliary_yolo_path=self.aux_yolo_path_var.get(),
            auxiliary_keypoint_model=self.aux_keypoint_model_var.get(),
            enable_face_blur=bool(self.enable_face_blur_var.get()),
            pose_confidence_threshold=float(self.pose_conf_var.get()),
            barbell_confidence_threshold=float(self.barbell_conf_var.get()),
            auxiliary_yolo_confidence_threshold=float(self.aux_yolo_conf_var.get()),
            exercise=self.exercise_var.get(),
            auto_exercise_model_path=None if selected_auto_model_path is None else str(selected_auto_model_path),
            aspect_ratio=self.aspect_ratio_var.get(),
            scale_label=self._get_scale_label(),
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

        self._save_last_model_configuration()

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
            self.status_var.set(f"{self.last_runtime_status_text} | Paused")

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
                self._get_scale_label(),
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
        self.last_runtime_status_text = packet.status_text
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
            self._get_scale_label(),
            bool(self.show_pose_preview_var.get()),
            self._get_auxiliary_point_radius(),
        )
        preview_frame = resize_for_display(self.preview_frame_bgr, display_settings)
        self._display_frame(preview_frame)

    def _display_frame(self, frame_bgr: np.ndarray) -> None:
        frame_bgr = self._apply_video_rotation(frame_bgr)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        self.video_photo = ImageTk.PhotoImage(image=image)
        self.video_label.configure(image=self.video_photo, text="")

    def _apply_video_rotation(self, frame_bgr: np.ndarray) -> np.ndarray:
        rotation_mode = VIDEO_ROTATION_OPTIONS.get(self.video_rotation_var.get(), "none")
        if rotation_mode == "cw90":
            return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
        if rotation_mode == "ccw90":
            return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame_bgr

    def _map_display_click_to_original(self, x_norm: float, y_norm: float) -> tuple[float, float]:
        rotation_mode = VIDEO_ROTATION_OPTIONS.get(self.video_rotation_var.get(), "none")
        if rotation_mode == "cw90":
            mapped_x = y_norm
            mapped_y = 1.0 - x_norm
        elif rotation_mode == "ccw90":
            mapped_x = 1.0 - y_norm
            mapped_y = x_norm
        else:
            mapped_x = x_norm
            mapped_y = y_norm
        return max(0.0, min(1.0, mapped_x)), max(0.0, min(1.0, mapped_y))

    def _on_video_click(self, event) -> None:
        if self.video_photo is None:
            return

        width = int(self.video_photo.width())
        height = int(self.video_photo.height())
        if width <= 0 or height <= 0:
            return
        if event.x < 0 or event.y < 0 or event.x >= width or event.y >= height:
            return

        x_norm = float(event.x / width)
        y_norm = float(event.y / height)
        x_norm, y_norm = self._map_display_click_to_original(x_norm, y_norm)
        self.pose_target_point_norm = (x_norm, y_norm)

        if self.worker is not None and self.worker.is_alive():
            self.worker.update_pose_target(x_norm, y_norm)
            self.status_var.set("Tracking target updated from click.")
        else:
            self.status_var.set("Tracking target set from click. Press Start.")

    def _update_state(self, packet: StatePacket) -> None:
        if packet.kind == "paused":
            self.status_var.set(f"{self.last_runtime_status_text} | Paused")
        else:
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
        self._save_last_model_configuration()
        if self.worker is not None and self.worker.is_alive():
            self.worker.cancel_event.set()
            self.worker.join(timeout=1.0)
        self.destroy()


def main() -> None:
    app = PoseDesktopApp()
    splash = SplashScreen(app, app._palette)
    app.wait_window(splash)
    app.deiconify()
    app.mainloop()
