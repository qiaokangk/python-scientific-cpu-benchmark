#!/usr/bin/env python3
"""PyQt front end for cpu_scientific_benchmark.py."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "python_scientific_cpu_benchmark_matplotlib")
)
if "--test-build" in sys.argv:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_requested_qt = os.environ.get("CPU_BENCH_QT_API", "").strip()
_qt_candidates = [_requested_qt] if _requested_qt else ["PyQt5", "PyQt6"]
_last_qt_error: Exception | None = None
for _candidate_qt in _qt_candidates:
    try:
        if _candidate_qt == "PyQt6":
            from PyQt6.QtCore import QEvent, QObject, QRectF, QSize, QThread, Qt, pyqtSignal
            from PyQt6.QtGui import QColor, QFont, QPainter, QTextCursor
            from PyQt6.QtWidgets import (
                QApplication,
                QCheckBox,
                QComboBox,
                QFileDialog,
                QFormLayout,
                QFrame,
                QGroupBox,
                QHBoxLayout,
                QLabel,
                QLineEdit,
                QMainWindow,
                QMessageBox,
                QPushButton,
                QPlainTextEdit,
                QProgressBar,
                QScrollArea,
                QSizePolicy,
                QSplitter,
                QSpinBox,
                QDoubleSpinBox,
                QTabBar,
                QTabWidget,
                QTableWidget,
                QTableWidgetItem,
                QHeaderView,
                QVBoxLayout,
                QWidget,
            )
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

            QT_API = "PyQt6"
            break
        if _candidate_qt == "PyQt5":
            from PyQt5.QtCore import QEvent, QObject, QRectF, QSize, QThread, Qt, pyqtSignal
            from PyQt5.QtGui import QColor, QFont, QPainter, QTextCursor
            from PyQt5.QtWidgets import (
                QApplication,
                QCheckBox,
                QComboBox,
                QFileDialog,
                QFormLayout,
                QFrame,
                QGroupBox,
                QHBoxLayout,
                QLabel,
                QLineEdit,
                QMainWindow,
                QMessageBox,
                QPushButton,
                QPlainTextEdit,
                QProgressBar,
                QScrollArea,
                QSizePolicy,
                QSplitter,
                QSpinBox,
                QDoubleSpinBox,
                QTabBar,
                QTabWidget,
                QTableWidget,
                QTableWidgetItem,
                QHeaderView,
                QVBoxLayout,
                QWidget,
            )
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

            QT_API = "PyQt5"
            break
    except Exception as exc:
        _last_qt_error = exc
else:
    raise RuntimeError("PyQt5 or PyQt6 is required for the UI.") from _last_qt_error

if QT_API == "PyQt6":
    QT_HORIZONTAL = Qt.Orientation.Horizontal
    QT_WHEEL_EVENT = QEvent.Type.Wheel
    QT_NO_FRAME = QFrame.Shape.NoFrame
    QT_EXPANDING = QSizePolicy.Policy.Expanding
    QT_NO_WRAP = QPlainTextEdit.LineWrapMode.NoWrap
    QT_MONO_HINT = QFont.StyleHint.Monospace
    QT_CURSOR_END = QTextCursor.MoveOperation.End
    QT_SCROLLBAR_ALWAYS_OFF = Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    QT_ANTIALIASING = QPainter.RenderHint.Antialiasing
    QT_NO_PEN = Qt.PenStyle.NoPen
    QT_ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
    QT_HEADER_STRETCH = QHeaderView.ResizeMode.Stretch
    QT_ITEM_IS_EDITABLE = Qt.ItemFlag.ItemIsEditable
else:
    QT_HORIZONTAL = Qt.Horizontal
    QT_WHEEL_EVENT = QEvent.Wheel
    QT_NO_FRAME = QFrame.NoFrame
    QT_EXPANDING = QSizePolicy.Expanding
    QT_NO_WRAP = QPlainTextEdit.NoWrap
    QT_MONO_HINT = QFont.Monospace
    QT_CURSOR_END = QTextCursor.End
    QT_SCROLLBAR_ALWAYS_OFF = Qt.ScrollBarAlwaysOff
    QT_ANTIALIASING = QPainter.Antialiasing
    QT_NO_PEN = Qt.NoPen
    QT_ALIGN_CENTER = Qt.AlignCenter
    QT_HEADER_STRETCH = QHeaderView.Stretch
    QT_ITEM_IS_EDITABLE = Qt.ItemIsEditable

from matplotlib.figure import Figure

import cpu_scientific_benchmark as bench


def _json_text(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _parse_literal(text: str) -> Any:
    stripped = text.strip()
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    if stripped.lower() == "null":
        return None
    try:
        return json.loads(stripped)
    except Exception:
        return stripped


def _safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event: Any) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event: Any) -> None:
        event.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event: Any) -> None:
        event.ignore()


class BadgeTabBar(QTabBar):
    """Draw notification badges over tabs without changing tab layout."""

    def __init__(self) -> None:
        super().__init__()
        self._badge_counts: dict[int, int] = {}

    def set_badge_count(self, index: int, count: int) -> None:
        if index < 0:
            return
        if count <= 0:
            self._badge_counts.pop(index, None)
        else:
            self._badge_counts[index] = int(count)
        self.update(self.tabRect(index))

    def badge_count(self, index: int) -> int:
        return self._badge_counts.get(index, 0)

    def paintEvent(self, event: Any) -> None:
        super().paintEvent(event)
        if not self._badge_counts:
            return
        painter = QPainter(self)
        painter.setRenderHint(QT_ANTIALIASING)
        font = QFont(self.font())
        font.setBold(True)
        font.setPixelSize(9)
        painter.setFont(font)
        for index, count in self._badge_counts.items():
            rect = self.tabRect(index)
            if not rect.isValid():
                continue
            text = "99+" if count > 99 else str(count)
            width = 15 if count < 10 else 21 if count < 100 else 25
            badge_rect = QRectF(rect.right() - width - 5, rect.top() + 3, width, 15)
            painter.setPen(QT_NO_PEN)
            painter.setBrush(QColor("#d93025"))
            painter.drawRoundedRect(badge_rect, 7.5, 7.5)
            painter.setPen(QColor("white"))
            painter.drawText(badge_rect, QT_ALIGN_CENTER, text)
        painter.end()


class ScrollAreaWheelRouter(QObject):
    """Route child wheel events to the outer scroll area unless a child scrolls."""

    def __init__(self, scroll_area: QScrollArea) -> None:
        super().__init__(scroll_area)
        self.scroll_area = scroll_area

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() != QT_WHEEL_EVENT:
            return False
        pixel_delta = event.pixelDelta().y() if hasattr(event, "pixelDelta") else 0
        angle_delta = event.angleDelta().y() if hasattr(event, "angleDelta") else 0
        delta = pixel_delta if pixel_delta else int(angle_delta / 2)
        inner_bar = self._preferred_inner_scroll_bar(obj)
        if delta and inner_bar is not None and self._bar_can_scroll(inner_bar, delta):
            return False
        bar = self.scroll_area.verticalScrollBar()
        if delta:
            bar.setValue(bar.value() - delta)
            event.accept()
            return True
        return False

    def _preferred_inner_scroll_bar(self, obj: QObject) -> Any:
        preferred = False
        current: QObject | None = obj
        while current is not None:
            try:
                preferred = preferred or bool(current.property("prefer_inner_wheel"))
            except Exception:
                pass
            if preferred and hasattr(current, "verticalScrollBar"):
                return current.verticalScrollBar()
            parent_func = getattr(current, "parent", None)
            current = parent_func() if callable(parent_func) else None
        return None

    @staticmethod
    def _bar_can_scroll(bar: Any, delta: int) -> bool:
        if delta > 0:
            return bar.value() > bar.minimum()
        return bar.value() < bar.maximum()


class HighResolutionCanvas(FigureCanvas):
    def __init__(self, figure: Figure) -> None:
        super().__init__(figure)
        self.setSizePolicy(QT_EXPANDING, QT_EXPANDING)
        self.setMinimumSize(QSize(660, 430))


class FigurePane(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(8.8, 5.8), dpi=120)
        self.figure.patch.set_facecolor("white")
        self.canvas = HighResolutionCanvas(self.figure)
        self.canvas_scroll = QScrollArea()
        self.canvas_scroll.setWidgetResizable(True)
        self.canvas_scroll.setFrameShape(QT_NO_FRAME)
        self.canvas_scroll.setAlignment(QT_ALIGN_CENTER)
        self.canvas_scroll.setWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas_scroll, 1)
        self.draw_empty(title)

    def set_row_count(self, row_count: int) -> None:
        _ = row_count
        self.canvas.setMinimumSize(QSize(660, 430))
        self.canvas.updateGeometry()

    def clear_figure(self) -> None:
        self.figure.clear()
        self.figure.patch.set_facecolor("white")

    def finish_draw(self) -> None:
        self.toolbar.update()
        self.canvas.draw()

    def draw_empty(self, message: str) -> None:
        self.set_row_count(8)
        self.clear_figure()
        ax = self.figure.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=16)
        self.finish_draw()


class BenchmarkWorker(QThread):
    progress = pyqtSignal(int, int, str)
    aggregate_ready = pyqtSignal(dict, str, int, int)
    completed = pyqtSignal(dict, str)
    failed = pyqtSignal(str)

    def __init__(self, config: dict[str, Any], working_dir: Path) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.working_dir = working_dir
        self._stop_requested = False

    def request_cancel(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            aggregate, message = self._run_impl()
            self.completed.emit(copy.deepcopy(aggregate), message)
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _run_impl(self) -> tuple[dict[str, Any], str]:
        output_cfg = self.config.get("output", {})
        output_dir = Path(str(output_cfg.get("directory", "benchmark_results"))).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = str(output_cfg.get("prefix", "cpu_python_scientific"))
        system_info = bench.collect_system_info()
        thread_modes = bench.resolve_thread_modes(self.config, system_info)
        names = bench.enabled_benchmark_names(self.config)
        total = len(thread_modes) * len(names)
        done = 0

        output_paths = bench.cli_output_paths(output_dir, prefix)
        effective_config_path = output_paths["effective_config"]
        bench.write_json(effective_config_path, self.config)

        aggregate: dict[str, Any] = {
            "system": system_info,
            "thread_modes": thread_modes,
            "config": self.config,
            "runs": [],
        }

        if not names:
            raise RuntimeError("No benchmark modules are enabled.")
        if not thread_modes:
            raise RuntimeError("No thread modes are enabled.")

        for mode in thread_modes:
            run: dict[str, Any] = {
                "status": "ok",
                "thread_label": str(mode["name"]),
                "thread_count": int(mode["threads"]),
                "elapsed_s": 0.0,
                "environment": {},
                "package_info": {},
                "threadpool_info": [],
                "monitoring": {"samples": []},
                "results": [],
            }
            aggregate["runs"].append(run)
        runs_by_label = {str(run["thread_label"]): run for run in aggregate["runs"]}

        script_path = Path(bench.__file__).resolve()
        execution_order = bench.normalize_execution_order(self.config)
        cases = bench.iter_benchmark_cases(names, thread_modes, execution_order)
        for case_index, (benchmark_index, name, _mode_index, mode) in enumerate(cases, start=1):
            if self._stop_requested:
                break
            label = str(mode["name"])
            threads = int(mode["threads"])
            run = runs_by_label[label]
            title = (
                f"{case_index}/{total} {name} / {label} "
                f"({bench.format_thread_count(threads)} thread(s))"
            )
            self.progress.emit(done, total, title)
            offset_s = float(run.get("elapsed_s", 0.0) or 0.0)
            worker_input = {
                "config": self.config,
                "thread_label": label,
                "thread_count": threads,
                "benchmark_name": name,
            }
            safe_label = _safe_filename(label)
            item_tag = f"{case_index:02d}_{benchmark_index:02d}_{_safe_filename(name)}_{safe_label}"
            worker_config_path = output_dir / f"{prefix}_ui_worker_{item_tag}_input.json"
            worker_output_path = output_dir / f"{prefix}_ui_worker_{item_tag}_results.json"
            worker_log_path = output_dir / f"{prefix}_ui_worker_{item_tag}.log"
            bench.write_json(worker_config_path, worker_input)
            env = bench.build_thread_env(threads, output_dir)
            command = [
                sys.executable,
                str(script_path),
                "--worker-one",
                "--worker-config",
                str(worker_config_path),
                "--worker-output",
                str(worker_output_path),
            ]
            started = time.perf_counter()
            proc = subprocess.Popen(
                command,
                cwd=str(self.working_dir),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            while proc.poll() is None:
                if self._stop_requested:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
                time.sleep(0.1)
            stdout, stderr = proc.communicate()
            elapsed = time.perf_counter() - started
            run["elapsed_s"] = offset_s + elapsed
            worker_log_path.write_text(
                "STDOUT\n"
                + (stdout or "")
                + "\nSTDERR\n"
                + (stderr or "")
                + f"\nRETURN_CODE {proc.returncode}\n",
                encoding="utf-8",
            )
            if self._stop_requested:
                break
            if proc.returncode != 0 or not worker_output_path.exists():
                run["results"].append(
                    {
                        "name": name,
                        "label": bench.BENCHMARK_LABELS.get(name, name),
                        "status": "error",
                        "reason": f"worker-one failed, see {worker_log_path}",
                        "error": stderr,
                        "sizes": copy.deepcopy(
                            self.config.get("modules", {}).get(name, {})
                        ),
                    }
                )
                run["status"] = "error"
            else:
                worker_data = bench.read_json(worker_output_path)
                if not run["package_info"]:
                    run["package_info"] = worker_data.get("package_info", {})
                if not run["threadpool_info"]:
                    run["threadpool_info"] = worker_data.get("threadpool_info", [])
                if not run["environment"]:
                    run["environment"] = worker_data.get("environment", {})
                for sample in worker_data.get("monitoring", {}).get("samples", []):
                    copied = copy.deepcopy(sample)
                    try:
                        copied["t_s"] = offset_s + float(copied.get("t_s", 0.0))
                    except Exception:
                        copied["t_s"] = offset_s
                    run["monitoring"]["samples"].append(copied)
                run["results"].extend(worker_data.get("results", []))
            done += 1
            message = self._last_result_message(label, threads, name, run["results"])
            self.aggregate_ready.emit(copy.deepcopy(aggregate), message, done, total)

        if self._stop_requested:
            aggregate["status"] = "cancelled"
            message = "Benchmark cancelled by user."
        else:
            aggregate["status"] = "ok"
            message = "Benchmark completed."

        bench.write_benchmark_outputs(aggregate, output_dir, prefix, output_cfg)
        return aggregate, message

    @staticmethod
    def _last_result_message(
        label: str, threads: int, name: str, results: list[dict[str, Any]]
    ) -> str:
        thread_text = bench.format_thread_count(threads)
        match = next((row for row in reversed(results) if row.get("name") == name), None)
        if not match:
            return f"{label} ({thread_text} threads) {name}: no result"
        if match.get("status") != "ok":
            return f"{label} ({thread_text} threads) {name}: {match.get('status')} {match.get('reason', '')}"
        metric = ""
        if match.get("metric_name") and match.get("metric_value") is not None:
            metric = f", {match['metric_value']:.3g} {match['metric_name']}"
        return (
            f"{label} ({thread_text} threads) {name}: "
            f"mean {bench.format_seconds(match.get('mean_s'))}, "
            f"timed {bench.format_seconds(match.get('timed_total_s'))}, "
            f"calls {int(match.get('total_calls', match.get('inner_loops', 1)) or 1)}{metric}"
        )


class BenchmarkWindow(QMainWindow):
    def __init__(self, config_path: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Python Scientific CPU Benchmark")
        self.resize(1450, 900)
        self.worker: BenchmarkWorker | None = None
        default_path = Path(config_path or "benchmark_config.json")
        self.config = bench.load_config(str(default_path) if default_path.exists() else None)
        self.latest_aggregate: dict[str, Any] | None = None
        self.module_checks: dict[str, QCheckBox] = {}
        self.modules_data: dict[str, dict[str, Any]] = {}
        self._loading_config = False
        self._updating_module_table = False
        self._updating_json = False
        self._build_ui(default_path)
        self.load_config_to_widgets(self.config, str(default_path))

    def _build_ui(self, default_path: Path) -> None:
        splitter = QSplitter(QT_HORIZONTAL)
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)

        left = self._build_control_panel(str(default_path))
        right = self._build_result_panel()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([470, 980])

    def _build_control_panel(self, default_path: str) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QT_NO_FRAME)
        scroll.setHorizontalScrollBarPolicy(QT_SCROLLBAR_ALWAYS_OFF)
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        config_group = QGroupBox("Configuration")
        config_form = QFormLayout(config_group)
        self.config_path_edit = QLineEdit(default_path)
        browse = QPushButton("Browse")
        browse.clicked.connect(self.browse_config)
        path_row = QHBoxLayout()
        path_row.addWidget(self.config_path_edit, 1)
        path_row.addWidget(browse)
        config_form.addRow("Config file", path_row)
        self.output_dir_edit = QLineEdit()
        self.prefix_edit = QLineEdit()
        config_form.addRow("Output dir", self.output_dir_edit)
        config_form.addRow("Output prefix", self.prefix_edit)
        buttons_row = QHBoxLayout()
        preset_row = QHBoxLayout()
        self.load_button = QPushButton("Load")
        self.save_button = QPushButton("Save")
        self.standard_button = QPushButton("Standard")
        self.smoke_button = QPushButton("Smoke")
        self.load_button.clicked.connect(self.load_config_from_file)
        self.save_button.clicked.connect(self.save_config_to_file)
        self.standard_button.clicked.connect(lambda: self.load_config_to_widgets(bench.default_config()))
        self.smoke_button.clicked.connect(lambda: self.load_config_to_widgets(bench.smoke_config()))
        buttons_row.addWidget(self.load_button)
        buttons_row.addWidget(self.save_button)
        preset_row.addWidget(self.standard_button)
        preset_row.addWidget(self.smoke_button)
        config_form.addRow(buttons_row)
        config_form.addRow(preset_row)
        layout.addWidget(config_group)

        benchmark_group = QGroupBox("Benchmark controls")
        form = QFormLayout(benchmark_group)
        self.repeats_spin = NoWheelSpinBox()
        self.repeats_spin.setRange(1, 100)
        self.warmups_spin = NoWheelSpinBox()
        self.warmups_spin.setRange(0, 100)
        self.seed_spin = NoWheelSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.memory_spin = NoWheelDoubleSpinBox()
        self.memory_spin.setRange(0.0, 1024.0)
        self.memory_spin.setDecimals(2)
        self.memory_spin.setSingleStep(0.5)
        self.target_repeat_spin = NoWheelDoubleSpinBox()
        self.target_repeat_spin.setRange(0.0, 3600.0)
        self.target_repeat_spin.setDecimals(2)
        self.target_repeat_spin.setSingleStep(0.5)
        self.enforce_threadpool_check = QCheckBox("threadpoolctl limits")
        self.gc_check = QCheckBox("GC between repeats")
        self.single_check = QCheckBox("single")
        self.multi_check = QCheckBox("multi")
        thread_row = QHBoxLayout()
        thread_row.addWidget(self.single_check)
        thread_row.addWidget(self.multi_check)
        thread_row.addStretch(1)
        self.multi_count_edit = QLineEdit()
        self.multi_count_edit.setPlaceholderText("auto, logical, physical, or integer")
        form.addRow("Repeats", self.repeats_spin)
        form.addRow("Warmups", self.warmups_spin)
        form.addRow("Random seed", self.seed_spin)
        form.addRow("Max memory GiB", self.memory_spin)
        form.addRow("Target case s", self.target_repeat_spin)
        form.addRow("Thread modes", thread_row)
        form.addRow("Multi threads", self.multi_count_edit)
        form.addRow(self.enforce_threadpool_check)
        form.addRow(self.gc_check)
        layout.addWidget(benchmark_group)

        monitor_group = QGroupBox("Runtime monitoring")
        monitor_form = QFormLayout(monitor_group)
        self.monitor_check = QCheckBox("record CPU, memory, frequency")
        self.monitor_interval_spin = NoWheelDoubleSpinBox()
        self.monitor_interval_spin.setRange(0.05, 10.0)
        self.monitor_interval_spin.setDecimals(2)
        self.monitor_interval_spin.setSingleStep(0.05)
        self.process_cpu_check = QCheckBox("process CPU")
        self.process_memory_check = QCheckBox("process memory")
        self.per_cpu_check = QCheckBox("per-core CPU")
        monitor_form.addRow(self.monitor_check)
        monitor_form.addRow("Interval s", self.monitor_interval_spin)
        monitor_form.addRow(self.process_cpu_check)
        monitor_form.addRow(self.process_memory_check)
        monitor_form.addRow(self.per_cpu_check)
        layout.addWidget(monitor_group)

        module_group = QGroupBox("Benchmark modules")
        module_layout = QVBoxLayout(module_group)
        for name in bench.BENCHMARK_ORDER:
            check = QCheckBox(name)
            check.setToolTip(bench.BENCHMARK_LABELS.get(name, name))
            check.stateChanged.connect(lambda _state, module=name: self._module_enabled_changed(module))
            self.module_checks[name] = check
            module_layout.addWidget(check)
        layout.addWidget(module_group)

        table_group = QGroupBox("Selected module parameters")
        table_layout = QVBoxLayout(table_group)
        self.module_select_combo = NoWheelComboBox()
        for name in bench.BENCHMARK_ORDER:
            self.module_select_combo.addItem(bench.BENCHMARK_LABELS.get(name, name), name)
        self.module_select_combo.currentIndexChanged.connect(lambda _index: self._load_selected_module_table())
        table_layout.addWidget(self.module_select_combo)
        self.module_param_table = QTableWidget(0, 2)
        self.module_param_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.module_param_table.horizontalHeader().setSectionResizeMode(QT_HEADER_STRETCH)
        self.module_param_table.verticalHeader().setVisible(False)
        self.module_param_table.setMinimumHeight(190)
        self.module_param_table.setStyleSheet("QTableWidget { gridline-color: #c8c8c8; background: white; }")
        self.module_param_table.itemChanged.connect(self._module_param_item_changed)
        table_layout.addWidget(self.module_param_table)
        layout.addWidget(table_group)

        json_group = QGroupBox("Module JSON preview / advanced edit")
        json_layout = QVBoxLayout(json_group)
        self.modules_editor = QPlainTextEdit()
        self.modules_editor.setProperty("prefer_inner_wheel", True)
        self.modules_editor.viewport().setProperty("prefer_inner_wheel", True)
        self.modules_editor.setLineWrapMode(QT_NO_WRAP)
        self.modules_editor.setMinimumHeight(220)
        mono = QFont("Menlo")
        mono.setStyleHint(QT_MONO_HINT)
        mono.setPointSize(11)
        self.modules_editor.setFont(mono)
        json_button_row = QHBoxLayout()
        self.apply_json_button = QPushButton("Apply JSON")
        self.refresh_json_button = QPushButton("Refresh JSON")
        self.apply_json_button.clicked.connect(self.apply_modules_json)
        self.refresh_json_button.clicked.connect(self.refresh_modules_json)
        json_button_row.addWidget(self.apply_json_button)
        json_button_row.addWidget(self.refresh_json_button)
        json_layout.addLayout(json_button_row)
        json_layout.addWidget(self.modules_editor)
        layout.addWidget(json_group)

        action_group = QGroupBox("Run")
        action_layout = QVBoxLayout(action_group)
        button_row = QHBoxLayout()
        save_row = QHBoxLayout()
        self.run_button = QPushButton("Run benchmark")
        self.cancel_button = QPushButton("Cancel calculation")
        self.cancel_button.setEnabled(False)
        self.save_results_button = QPushButton("Save results")
        self.save_results_button.setEnabled(False)
        self.run_button.clicked.connect(self.start_benchmark)
        self.cancel_button.clicked.connect(self.cancel_benchmark)
        self.save_results_button.clicked.connect(self.save_results_to_output_dir)
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.cancel_button)
        save_row.addWidget(self.save_results_button)
        action_layout.addLayout(button_row)
        action_layout.addLayout(save_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.status_label = QLabel("Idle")
        self.status_label.setWordWrap(True)
        action_layout.addWidget(self.progress)
        action_layout.addWidget(self.status_label)
        layout.addWidget(action_group)

        layout.addStretch(1)
        scroll.setWidget(holder)
        router = ScrollAreaWheelRouter(scroll)
        self._wheel_router = router
        for child in holder.findChildren(QWidget):
            child.installEventFilter(router)
        return scroll

    def _build_result_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        self.tabs = QTabWidget()
        self.badge_tab_bar = BadgeTabBar()
        self.tabs.setTabBar(self.badge_tab_bar)
        self.timing_pane = FigurePane("Timing plot will update after each test.")
        self.speedup_pane = FigurePane("Speedup plot needs single and multithread results.")
        self.metric_pane = FigurePane("Metric plot will update after each test.")
        self.monitor_pane = FigurePane("Runtime monitor plot will update while results arrive.")
        self.report_text = QPlainTextEdit()
        self.report_text.setReadOnly(True)
        mono = QFont("Menlo")
        mono.setStyleHint(QT_MONO_HINT)
        mono.setPointSize(11)
        self.report_text.setFont(mono)
        self.tabs.addTab(self.timing_pane, "Timing")
        self.tabs.addTab(self.speedup_pane, "Speedup")
        self.tabs.addTab(self.metric_pane, "Metrics")
        self.tabs.addTab(self.monitor_pane, "Monitor")
        self.tabs.addTab(self.report_text, "Text")
        self.tabs.currentChanged.connect(self._clear_tab_badge)
        layout.addWidget(self.tabs, 1)
        return panel

    def _mark_tab_updated(self, pane: QWidget) -> None:
        if not hasattr(self, "tabs") or not hasattr(self, "badge_tab_bar"):
            return
        index = self.tabs.indexOf(pane)
        if index < 0:
            return
        if index == self.tabs.currentIndex():
            self._clear_tab_badge(index)
            return
        count = self.badge_tab_bar.badge_count(index) + 1
        self.badge_tab_bar.set_badge_count(index, count)
        self.tabs.setTabToolTip(index, f"{self.tabs.tabText(index)} updated {count} time(s).")

    def _clear_tab_badge(self, index: int) -> None:
        if index < 0 or not hasattr(self, "badge_tab_bar"):
            return
        self.badge_tab_bar.set_badge_count(index, 0)
        if hasattr(self, "tabs"):
            self.tabs.setTabToolTip(index, "")

    def browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open benchmark config", str(Path.cwd()), "JSON files (*.json);;All files (*)"
        )
        if path:
            self.config_path_edit.setText(path)
            self.load_config_from_file()

    def load_config_from_file(self) -> None:
        path = self.config_path_edit.text().strip()
        try:
            cfg = bench.load_config(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self.load_config_to_widgets(cfg, path)

    def save_config_to_file(self) -> None:
        try:
            cfg = self.widgets_to_config()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid config", str(exc))
            return
        path = self.config_path_edit.text().strip()
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save benchmark config", str(Path.cwd() / "benchmark_config.json"), "JSON files (*.json)"
            )
        if not path:
            return
        bench.write_json(Path(path), cfg)
        self.config = cfg
        self.config_path_edit.setText(path)
        self.status_label.setText(f"Saved {path}")

    def load_config_to_widgets(self, cfg: dict[str, Any], path: str | None = None) -> None:
        self._loading_config = True
        self.config = copy.deepcopy(cfg)
        try:
            if path is not None:
                self.config_path_edit.setText(path)
            benchmark_cfg = self.config.get("benchmark", {})
            output_cfg = self.config.get("output", {})
            monitor_cfg = self.config.get("monitoring", {})
            self.output_dir_edit.setText(str(output_cfg.get("directory", "benchmark_results")))
            self.prefix_edit.setText(str(output_cfg.get("prefix", "cpu_python_scientific")))
            self.repeats_spin.setValue(int(benchmark_cfg.get("repeats", 1)))
            self.warmups_spin.setValue(int(benchmark_cfg.get("warmups", 1)))
            self.seed_spin.setValue(int(benchmark_cfg.get("random_seed", 20260606)))
            self.memory_spin.setValue(float(benchmark_cfg.get("max_memory_gb", 4.0) or 0.0))
            self.target_repeat_spin.setValue(float(benchmark_cfg.get("target_case_s", benchmark_cfg.get("target_repeat_s", 0.0)) or 0.0))
            modes = [str(item.get("name", "")) if isinstance(item, dict) else str(item) for item in benchmark_cfg.get("thread_modes", ["single", "multi"])]
            self.single_check.setChecked(any(mode.lower() == "single" for mode in modes))
            self.multi_check.setChecked(any(mode.lower() == "multi" for mode in modes))
            self.multi_count_edit.setText(str(benchmark_cfg.get("multi_thread_count", "auto")))
            self.enforce_threadpool_check.setChecked(bool(benchmark_cfg.get("enforce_threadpoolctl", True)))
            self.gc_check.setChecked(bool(benchmark_cfg.get("gc_between_repeats", True)))
            self.monitor_check.setChecked(bool(monitor_cfg.get("enabled", True)))
            self.monitor_interval_spin.setValue(float(monitor_cfg.get("interval_s", 0.25)))
            self.process_cpu_check.setChecked(bool(monitor_cfg.get("process_cpu", True)))
            self.process_memory_check.setChecked(bool(monitor_cfg.get("process_memory", True)))
            self.per_cpu_check.setChecked(bool(monitor_cfg.get("per_cpu", False)))

            self.modules_data = copy.deepcopy(self.config.get("modules", {}))
            for name, check in self.module_checks.items():
                check.setChecked(bool(self.modules_data.get(name, {}).get("enabled", False)))
            self.refresh_modules_json()
            self._load_selected_module_table()
            self.report_text.setPlainText(self._intro_text())
            self.latest_aggregate = None
            if hasattr(self, "save_results_button"):
                self.save_results_button.setEnabled(False)
            self.status_label.setText("Config loaded")
        finally:
            self._loading_config = False

    def refresh_modules_json(self) -> None:
        self._updating_json = True
        try:
            self.modules_editor.setPlainText(_json_text(self.modules_data))
        finally:
            self._updating_json = False

    def apply_modules_json(self) -> None:
        try:
            modules = json.loads(self.modules_editor.toPlainText())
        except json.JSONDecodeError as exc:
            QMessageBox.warning(self, "Invalid JSON", f"Module JSON is invalid: {exc}")
            return
        if not isinstance(modules, dict):
            QMessageBox.warning(self, "Invalid JSON", "Module JSON must be an object.")
            return
        self.modules_data = copy.deepcopy(modules)
        self._loading_config = True
        try:
            for name, check in self.module_checks.items():
                check.setChecked(bool(self.modules_data.get(name, {}).get("enabled", False)))
        finally:
            self._loading_config = False
        self._load_selected_module_table()
        self.status_label.setText("Applied module JSON")

    def _module_enabled_changed(self, name: str) -> None:
        if self._loading_config:
            return
        item = self.modules_data.setdefault(name, {})
        item["enabled"] = self.module_checks[name].isChecked()
        self.refresh_modules_json()

    def _load_selected_module_table(self) -> None:
        if not hasattr(self, "module_param_table"):
            return
        name = self.module_select_combo.currentData()
        if not name:
            return
        module_cfg = copy.deepcopy(self.modules_data.get(str(name), {}))
        keys = [key for key in module_cfg.keys() if key != "enabled"]
        preferred = [
            "n",
            "m",
            "d",
            "shape",
            "kernel_shape",
            "nrhs",
            "k",
            "tol",
            "maxiter",
            "loops",
            "nnz_per_row",
            "iterations",
            "a",
            "b",
            "c",
            "e",
            "sigma",
            "backend",
            "use_scipy_workers",
            "inner_loops",
            "target_case_s",
        ]
        keys.sort(key=lambda key: (preferred.index(key) if key in preferred else 999, key))
        self._updating_module_table = True
        try:
            self.module_param_table.setRowCount(len(keys))
            for row, key in enumerate(keys):
                key_item = QTableWidgetItem(str(key))
                key_item.setFlags(key_item.flags() & ~QT_ITEM_IS_EDITABLE)
                value = module_cfg.get(key)
                value_text = json.dumps(value) if isinstance(value, (list, dict, bool)) or value is None else str(value)
                value_item = QTableWidgetItem(value_text)
                self.module_param_table.setItem(row, 0, key_item)
                self.module_param_table.setItem(row, 1, value_item)
        finally:
            self._updating_module_table = False

    def _module_param_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_module_table or item.column() != 1:
            return
        name = self.module_select_combo.currentData()
        key_item = self.module_param_table.item(item.row(), 0)
        if not name or key_item is None:
            return
        module_cfg = self.modules_data.setdefault(str(name), {})
        module_cfg[str(key_item.text())] = _parse_literal(item.text())
        self.refresh_modules_json()

    def widgets_to_config(self) -> dict[str, Any]:
        cfg = copy.deepcopy(self.config)
        modules = copy.deepcopy(self.modules_data)
        for name, check in self.module_checks.items():
            item = modules.setdefault(name, {})
            if not isinstance(item, dict):
                raise ValueError(f"Module '{name}' must be a JSON object.")
            item["enabled"] = check.isChecked()
        modes = []
        if self.single_check.isChecked():
            modes.append("single")
        if self.multi_check.isChecked():
            modes.append("multi")
        if not modes:
            raise ValueError("Enable at least one thread mode.")
        cfg["modules"] = modules
        cfg.setdefault("benchmark", {})
        cfg["benchmark"].update(
            {
                "repeats": self.repeats_spin.value(),
                "warmups": self.warmups_spin.value(),
                "random_seed": self.seed_spin.value(),
                "max_memory_gb": self.memory_spin.value(),
                "target_case_s": self.target_repeat_spin.value(),
                "target_repeat_s": 0.0,
                "calibration_max_inner_loops": int(cfg.get("benchmark", {}).get("calibration_max_inner_loops", 100_000)),
                "thread_modes": modes,
                "multi_thread_count": self.multi_count_edit.text().strip() or "auto",
                "execution_order": "by_thread_mode",
                "enforce_threadpoolctl": self.enforce_threadpool_check.isChecked(),
                "gc_between_repeats": self.gc_check.isChecked(),
            }
        )
        cfg.setdefault("monitoring", {})
        cfg["monitoring"].update(
            {
                "enabled": self.monitor_check.isChecked(),
                "interval_s": self.monitor_interval_spin.value(),
                "process_cpu": self.process_cpu_check.isChecked(),
                "process_memory": self.process_memory_check.isChecked(),
                "per_cpu": self.per_cpu_check.isChecked(),
            }
        )
        cfg.setdefault("output", {})
        cfg["output"].update(
            {
                "directory": self.output_dir_edit.text().strip() or "benchmark_results",
                "prefix": self.prefix_edit.text().strip() or "cpu_python_scientific",
                "make_pdf": bool(cfg["output"].get("make_pdf", True)),
                "make_json": bool(cfg["output"].get("make_json", True)),
                "make_text": bool(cfg["output"].get("make_text", True)),
            }
        )
        return cfg

    def start_benchmark(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        try:
            cfg = self.widgets_to_config()
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid config", str(exc))
            return
        self.config = cfg
        self.latest_aggregate = None
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.status_label.setText("Starting...")
        self.report_text.setPlainText("Starting benchmark...\n")
        self.timing_pane.draw_empty("Waiting for first benchmark result.")
        self.speedup_pane.draw_empty("Waiting for single and multithread results.")
        self.metric_pane.draw_empty("Waiting for throughput metrics.")
        self.monitor_pane.draw_empty("Waiting for monitor samples.")
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.save_results_button.setEnabled(False)
        self.worker = BenchmarkWorker(cfg, Path.cwd())
        self.worker.progress.connect(self.on_progress)
        self.worker.aggregate_ready.connect(self.on_aggregate_ready)
        self.worker.completed.connect(self.on_completed)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def cancel_benchmark(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Cancelling current subprocess...")

    def on_progress(self, done: int, total: int, message: str) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.status_label.setText(message)

    def on_aggregate_ready(
        self, aggregate: dict[str, Any], message: str, done: int, total: int
    ) -> None:
        self.latest_aggregate = aggregate
        self.save_results_button.setEnabled(True)
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.status_label.setText(message)
        self.report_text.setPlainText(bench.format_plain_report(aggregate))
        self.report_text.moveCursor(QT_CURSOR_END)
        self._mark_tab_updated(self.report_text)
        self.draw_all_plots(aggregate)

    def on_completed(self, aggregate: dict[str, Any], message: str) -> None:
        self.latest_aggregate = aggregate
        self.progress.setValue(self.progress.maximum())
        output_files = aggregate.get("output_files", {})
        details = [message]
        for key in ("effective_config", "text", "json", "pdf"):
            value = output_files.get(key)
            if value:
                details.append(f"{key}: {value}")
        if output_files.get("pdf_error"):
            details.append(f"pdf_error: {output_files['pdf_error']}")
        self.status_label.setText(message)
        self.report_text.setPlainText(bench.format_plain_report(aggregate) + "\n" + "\n".join(details) + "\n")
        self._mark_tab_updated(self.report_text)
        self.draw_all_plots(aggregate)
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.save_results_button.setEnabled(True)

    def on_failed(self, error_text: str) -> None:
        self.status_label.setText("Benchmark failed")
        self.report_text.setPlainText(error_text)
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.save_results_button.setEnabled(self.latest_aggregate is not None)

    def save_results_to_output_dir(self) -> None:
        if not self.latest_aggregate:
            QMessageBox.information(self, "No results", "No benchmark results are available to save.")
            return
        aggregate = copy.deepcopy(self.latest_aggregate)
        cfg = copy.deepcopy(aggregate.get("config", self.config))
        cfg.setdefault("output", {})
        cfg["output"].update(
            {
                "directory": self.output_dir_edit.text().strip() or "benchmark_results",
                "prefix": self.prefix_edit.text().strip() or "cpu_python_scientific",
            }
        )
        aggregate["config"] = cfg
        output_cfg = cfg.get("output", {})
        output_dir = Path(str(output_cfg.get("directory", "benchmark_results"))).resolve()
        prefix = str(output_cfg.get("prefix", "cpu_python_scientific"))
        try:
            output_files, report_text = bench.write_benchmark_outputs(
                aggregate, output_dir, prefix, output_cfg
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.latest_aggregate = aggregate
        details = ["Saved results in CLI format."]
        for key in ("effective_config", "text", "json", "pdf"):
            value = output_files.get(key)
            if value:
                details.append(f"{key}: {value}")
        if output_files.get("pdf_error"):
            details.append(f"pdf_error: {output_files['pdf_error']}")
        self.status_label.setText("Results saved")
        self.report_text.setPlainText(report_text + "\n" + "\n".join(details) + "\n")
        self._mark_tab_updated(self.report_text)

    def draw_all_plots(self, aggregate: dict[str, Any]) -> None:
        self.draw_timing_plot(aggregate)
        self.draw_speedup_plot(aggregate)
        self.draw_metric_plot(aggregate)
        self.draw_monitor_plot(aggregate)

    def planned_benchmark_names(self, aggregate: dict[str, Any]) -> list[str]:
        names = bench.enabled_benchmark_names(aggregate.get("config", {}))
        if names:
            return names
        completed = [row.get("name") for row in bench.result_rows(aggregate)]
        return [name for name in bench.BENCHMARK_ORDER if name in completed]

    def planned_thread_labels(self, aggregate: dict[str, Any]) -> list[str]:
        labels = [str(mode.get("name", "")) for mode in aggregate.get("thread_modes", [])]
        labels = [label for label in labels if label]
        if labels:
            return labels
        rows = bench.result_rows(aggregate)
        return sorted(
            {str(row.get("thread_label", "")) for row in rows if row.get("thread_label")},
            key=bench.thread_label_sort_key,
        )

    def draw_timing_plot(self, aggregate: dict[str, Any]) -> None:
        rows = bench.result_rows(aggregate)
        fig = self.timing_pane.figure
        self.timing_pane.clear_figure()
        names = self.planned_benchmark_names(aggregate)
        modes = self.planned_thread_labels(aggregate)
        if not names or not modes or not any(row.get("status") == "ok" for row in rows):
            self.timing_pane.draw_empty("No successful timing rows yet.")
            return
        self.timing_pane.set_row_count(len(names))
        ax = fig.add_subplot(111)
        bench.plot_timing_bars(ax, rows, names=names, mode_labels=modes, label_width=30)
        bench.adjust_timing_figure(fig)
        self.timing_pane.finish_draw()
        self._mark_tab_updated(self.timing_pane)

    def draw_speedup_plot(self, aggregate: dict[str, Any]) -> None:
        rows = [
            row
            for row in bench.result_rows(aggregate)
            if row.get("status") == "ok"
            and row.get("speedup") is not None
            and int(row.get("thread_count", 0) or 0) != 1
        ]
        fig = self.speedup_pane.figure
        self.speedup_pane.clear_figure()
        names = self.planned_benchmark_names(aggregate)
        if not names:
            self.speedup_pane.draw_empty("Speedup appears after matching single and multithread results.")
            return
        self.speedup_pane.set_row_count(len(names))
        ax = fig.add_subplot(111)
        bench.plot_speedup_bars(ax, rows, names=names, label_width=30)
        bench.adjust_horizontal_bar_figure(fig)
        self.speedup_pane.finish_draw()
        self._mark_tab_updated(self.speedup_pane)

    def draw_metric_plot(self, aggregate: dict[str, Any]) -> None:
        rows = [
            row
            for row in bench.result_rows(aggregate)
            if row.get("status") == "ok"
            and row.get("metric_name")
            and row.get("metric_value") is not None
        ]
        fig = self.metric_pane.figure
        self.metric_pane.clear_figure()
        planned_names = self.planned_benchmark_names(aggregate)
        if not rows:
            if not planned_names:
                self.metric_pane.draw_empty("No throughput metrics yet.")
                return
            self.metric_pane.set_row_count(len(planned_names))
            ax = fig.add_subplot(111)
            bench.plot_metric_unit_bars(
                ax,
                [],
                names=planned_names,
                unit=None,
                label_width=30,
                title="Throughput metrics will fill in as tests finish",
            )
            bench.adjust_horizontal_bar_figure(fig)
            self.metric_pane.finish_draw()
            self._mark_tab_updated(self.metric_pane)
            return
        multi = [row for row in rows if str(row.get("thread_label", "")).lower() == "multi"]
        selected = multi or rows
        unit = str(selected[0].get("metric_name"))
        selected_by_name = {
            str(row.get("name")): row
            for row in selected
            if str(row.get("metric_name")) == unit
        }
        names = planned_names[:24] if planned_names else list(selected_by_name)[:24]
        self.metric_pane.set_row_count(len(names))
        ax = fig.add_subplot(111)
        bench.plot_metric_unit_bars(ax, selected, names=names, unit=unit, label_width=30)
        bench.adjust_horizontal_bar_figure(fig)
        self.metric_pane.finish_draw()
        self._mark_tab_updated(self.metric_pane)

    def draw_monitor_plot(self, aggregate: dict[str, Any]) -> None:
        fig = self.monitor_pane.figure
        self.monitor_pane.clear_figure()
        runs = [run for run in aggregate.get("runs", []) if run.get("monitoring", {}).get("samples")]
        if not runs:
            self.monitor_pane.draw_empty("No runtime monitor samples yet.")
            return
        run = runs[-1]
        logical_count = int(aggregate.get("system", {}).get("logical_cpu_count") or os.cpu_count() or 1)
        ax = fig.add_subplot(111)
        bench.plot_monitor_run(ax, run, logical_count)
        bench.adjust_monitor_figure(fig)
        self.monitor_pane.finish_draw()
        self._mark_tab_updated(self.monitor_pane)

    def _intro_text(self) -> str:
        return (
            "Python Scientific CPU Benchmark UI\n"
            "\n"
            "Left panel: edit parameters, thread modes, module enables, module parameter tables, and advanced JSON.\n"
            "Right panel: plots and text update after every completed benchmark test.\n"
            "Random inputs default to fixed PCG64 seeds derived from benchmark.random_seed.\n"
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyQt UI for Python scientific CPU benchmark.")
    parser.add_argument("--config", default=None, help="Optional JSON config to load.")
    parser.add_argument("--test-build", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.test_build:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = BenchmarkWindow(args.config)
    if args.test_build:
        window.resize(1400, 900)
        window.centralWidget().adjustSize()
        app.processEvents()
        return 0
    window.show()
    if hasattr(app, "exec"):
        return int(app.exec())
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
