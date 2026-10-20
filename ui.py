"""
ECG-XAI-Visualizer — Desktop Graphical Interface.

Application Layout:
-------------------
+-----------------------------------------------------------------------------+
| Header strip                                                                |
+------------------------------+----------------------------------------------+
| Settings pane (left, 440px+) | Results pane (right)                         |
|                              |                                              |
| • Input: file/folder, scan   | • Chips: processed / ischemia / skipped /    |
| • Output: folder, figures    |   failed            • Progress bar           |
| • Pre-processing: notch,     | • Tabs:                                      |
|   bandpass, wavelet, EMD,    |   - [Activity] live log                      |
|   derive missing leads       |   - [Viewer]   record ‹ › · Raw | Processed |
| • Sampling: input rate,      |                Model input | XAI · lead      |
|   resample                   |                toggles, or XAI method + lead |
| • Model: threshold, window   |                · plot + details              |
| • Patient: CSV, fallbacks    |   - [Summary]  sortable results table        |
| • Privacy: anonymize         |                                              |
| [ ▶ Start analysis ] [Cancel]| [ Open output folder ] [ Open summary CSV ]  |
+------------------------------+----------------------------------------------+

Keyboard Shortcuts:
-------------------
• Ctrl+O            : Browse input folder
• Ctrl+Shift+O      : Browse output folder
• Ctrl+R            : Start the analysis
• Esc               : Request graceful cancellation
• Alt+Left / Right  : Previous / next record in the viewer
• Ctrl+1 .. Ctrl+4  : Raw / Processed / Model input / XAI view
In the plots: scroll to zoom the time axis, drag to pan, double-click to reset.

Usage:
------
    python3 ui.py
"""
import html
import sys
from pathlib import Path

try:
    from PySide6.QtCore import (
        QPointF, QRect, QRegularExpression, Qt, QThread, QTimer, Signal, QUrl,
    )
    from PySide6.QtGui import (
        QColor, QDesktopServices, QFont, QFontMetrics, QIcon, QKeySequence, QPainter,
        QPainterPath, QPalette, QPen, QPixmap, QRegularExpressionValidator,
        QShortcut,
    )
    from PySide6.QtWidgets import (
        QAbstractSpinBox, QApplication, QButtonGroup, QCheckBox, QComboBox,
        QDoubleSpinBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, QHeaderView,
        QLabel, QListView, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
        QPushButton, QScrollArea, QSplitter, QStackedWidget, QStyle,
        QStyleOptionButton, QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser,
        QToolButton, QVBoxLayout, QWidget,
    )
except ImportError:
    print("The GUI needs PySide6. Install everything with:\n"
          "    pip install -r requirements.txt")
    raise SystemExit(1)

if not getattr(sys, "frozen", False):
    _root = str(Path(__file__).resolve().parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from ecgproc.deps import check_dependencies
    check_dependencies(import_modules=False)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ecgproc.constants import FINAL_LEADS, SUPPORTED_EXTENSIONS, lead_label
from ecgproc.files import find_files
from ecgproc.standardize import canonical_lead_name
from ecgxai import __version__
from ecgxai.explain import XAI_METHODS, lead_key
from ecgxai.plotting import draw_leads, draw_xai, placeholder, probability_text, xai_titles
from ecgxai.registry import MODELS, get_spec
from ecgxai.theme import (
    ACCENT, ACCENT_HOVER, ACCENT_PRESSED, BG, BORDER, BORDER_HOVER, DISABLED_TEXT, ERROR,
    FLAG, GRID_MAJOR, HOVER, MUTED, ON_ACCENT, SEGMENT_COLORS, SELECTED, SURFACE, SURFACE2,
    TEXT, WARN, XAI,
)
from runner.execution_runner import SHARE_SUFFIXES, SUMMARY_COLUMNS, SUMMARY_NAME, ExecutionRunner

APP = "ECG-XAI-Visualizer"

# ------
# Design
# ------

MONO = "'Consolas','Menlo','DejaVu Sans Mono',monospace"

QSS = f"""
QMainWindow {{ background: {BG}; }}
#Header {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; }}
#AppTitle {{ font-size: 17px; font-weight: 700; color: {TEXT}; }}
#AppSub {{ color: {MUTED}; font-size: 12px; }}
#Version {{ color: {MUTED}; font-family: {MONO}; font-size: 12px; }}

QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 8px; background: {SURFACE};
    margin-top: 12px; padding: 12px 10px 8px 10px; font-weight: 600; font-size: 12px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED}; }}

QPushButton {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 7px 12px; color: {TEXT};
}}
QPushButton:hover {{ background: {HOVER}; border-color: {BORDER_HOVER}; }}
QPushButton:pressed {{ background: #0E1520; }}
QPushButton:disabled {{ color: {DISABLED_TEXT}; border-color: #1A2531; }}

QPushButton#primary {{
    background: {ACCENT}; color: {ON_ACCENT}; border: none;
    padding: 11px 14px; font-size: 14px; font-weight: 700;
}}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#primary:pressed {{ background: {ACCENT_PRESSED}; }}
QPushButton#primary:disabled {{ background: #1E3A2F; color: #5E7A6C; }}

QPushButton#segL, QPushButton#segM, QPushButton#segR {{
    background: {SURFACE2}; color: {MUTED}; border: 1px solid {BORDER};
    padding: 6px 13px; border-radius: 0;
}}
QPushButton#segL {{ border-top-left-radius: 6px; border-bottom-left-radius: 6px; }}
QPushButton#segR {{ border-top-right-radius: 6px; border-bottom-right-radius: 6px; }}
QPushButton#segL:hover, QPushButton#segM:hover, QPushButton#segR:hover {{ color: {TEXT}; background: {HOVER}; }}
QPushButton#segL:checked, QPushButton#segM:checked, QPushButton#segR:checked {{
    background: {SELECTED}; color: {ACCENT}; border-color: #2E5C48;
}}
QPushButton#segL:disabled, QPushButton#segM:disabled, QPushButton#segR:disabled {{ color: {DISABLED_TEXT}; }}

QPushButton#lead {{
    background: {SURFACE2}; color: {MUTED}; border: 1px solid {BORDER};
    border-radius: 9px; padding: 2px 6px; font-family: {MONO}; font-size: 11px; min-width: 24px;
}}
QPushButton#lead:hover {{ border-color: {BORDER_HOVER}; color: {TEXT}; }}
QPushButton#lead:checked {{ color: {ACCENT}; background: #10241C; border-color: #2E5C48; }}

QToolButton {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 4px 8px; color: {MUTED};
}}
QToolButton:hover {{ color: {ERROR}; border-color: {BORDER_HOVER}; }}
QToolButton#nav {{ font-size: 15px; padding: 2px 9px; }}
QToolButton#nav:hover {{ color: {ACCENT}; }}
QToolButton:disabled {{ color: {DISABLED_TEXT}; border-color: #1A2531; }}

QProgressBar {{
    border: 1px solid {BORDER}; border-radius: 6px; background: {SURFACE2};
    text-align: center; color: {TEXT}; font-family: {MONO}; font-size: 12px; height: 22px;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}

QPlainTextEdit#log {{
    background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 6px;
    color: #C6D3E0; font-family: {MONO}; font-size: 12px; padding: 6px;
}}
QTextBrowser#details {{
    background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 6px;
    color: {TEXT}; font-size: 12px; padding: 6px;
}}

QLabel#chip_ok, QLabel#chip_warn, QLabel#chip_err, QLabel#chip_flag {{
    border-radius: 10px; padding: 3px 12px; font-family: {MONO}; font-size: 12px;
}}
QLabel#chip_ok   {{ color: {ACCENT}; background: #10241C; border: 1px solid #2E5C48; }}
QLabel#chip_flag {{ color: {FLAG};  background: #1D1530; border: 1px solid #4A3470; }}
QLabel#chip_warn {{ color: {WARN};  background: #241D10; border: 1px solid #5C4A22; }}
QLabel#chip_err  {{ color: {ERROR}; background: #241010; border: 1px solid #5C2E2E; }}

QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 6px; background: {SURFACE}; top: -1px; }}
QTabBar::tab {{
    background: transparent; color: {MUTED}; padding: 7px 16px;
    border: 1px solid transparent; border-bottom: none;
    border-top-left-radius: 6px; border-top-right-radius: 6px;
}}
QTabBar::tab:selected {{ color: {TEXT}; background: {SURFACE}; border-color: {BORDER}; }}
QTabBar::tab:hover:!selected {{ color: {TEXT}; }}

QTableWidget {{
    background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 6px;
    gridline-color: {GRID_MAJOR}; font-size: 12px;
    selection-background-color: {SELECTED}; selection-color: {TEXT};
}}
QHeaderView::section {{
    background: {SURFACE}; color: {MUTED}; border: none;
    border-bottom: 1px solid {BORDER}; border-right: 1px solid {GRID_MAJOR}; padding: 4px 8px;
}}
QTableCornerButton::section {{ background: {SURFACE}; border: none; }}

QCheckBox {{ spacing: 9px; color: {TEXT}; }}
QCheckBox:disabled {{ color: {DISABLED_TEXT}; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border: 1px solid #35485E; border-radius: 4px; background: {SURFACE2};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; background: #142031; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QCheckBox::indicator:checked:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}
QCheckBox::indicator:disabled {{ border-color: #1E2A38; background: #0E141C; }}
QCheckBox::indicator:checked:disabled {{ background: #1E3A2F; border-color: #2A4C3D; }}

QComboBox, QDoubleSpinBox {{
    background: {SURFACE2}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 5px 8px; color: {TEXT}; min-height: 20px;
    selection-background-color: {SELECTED}; selection-color: {ACCENT};
}}
QComboBox:hover, QDoubleSpinBox:hover {{ border-color: {BORDER_HOVER}; background: #101A26; }}
QComboBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QComboBox:on {{ border-color: {ACCENT}; background: #101A26; }}
QComboBox:disabled, QDoubleSpinBox:disabled {{
    color: {DISABLED_TEXT}; border-color: #1A2531; background: #0B1119;
}}
NumberBox::up-button, NumberBox::down-button {{ subcontrol-origin: border; }}
QComboBox {{ padding-right: 24px; }}
QComboBox::drop-down {{ border: none; background: transparent; width: 24px; }}
QComboBox::down-arrow {{ image: none; width: 0; height: 0; }}
/* Editable combo: the inner line edit draws its own frame and background,
   which would sit as a pale box inside the rounded field. */
QComboBox QLineEdit {{
    background: transparent; border: none; padding: 0; margin: 0; color: {TEXT};
    selection-background-color: {SELECTED}; selection-color: {ACCENT};
}}
QComboBox QLineEdit:disabled {{ color: {DISABLED_TEXT}; }}

/* Popup list: light text on an accent-tinted row — never the near-black
   that selection-color: {ON_ACCENT} produced on the closed control's palette. */
QComboBox QAbstractItemView {{
    background: {SURFACE}; border: 1px solid {BORDER_HOVER}; border-radius: 8px;
    padding: 4px; outline: 0; color: {TEXT};
    selection-background-color: {SELECTED}; selection-color: {ACCENT};
}}
QComboBox QAbstractItemView::item {{
    min-height: 26px; padding: 3px 8px; border-radius: 5px;
    color: {TEXT}; background: transparent;
}}
QComboBox QAbstractItemView::item:hover {{ background: {HOVER}; color: {TEXT}; }}
QComboBox QAbstractItemView::item:selected {{ background: {SELECTED}; color: {ACCENT}; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 12px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 4px; min-height: 28px; margin: 0 2px;
}}
QScrollBar::handle:vertical:hover {{ background: {BORDER_HOVER}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 0; }}
QScrollBar::handle:horizontal {{
    background: {BORDER}; border-radius: 4px; min-width: 28px; margin: 2px 0;
}}
QScrollBar::handle:horizontal:hover {{ background: {BORDER_HOVER}; }}

QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 8px; }}

QToolTip {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 6px; }}
QStatusBar {{ background: {SURFACE}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
"""

# One heartbeat
HEARTBEAT = [
    (0.00, 0.00), (0.14, 0.00), (0.19, 0.16), (0.24, 0.00), (0.30, 0.00),
    (0.33, -0.16), (0.365, 1.00), (0.40, -0.44), (0.44, 0.00), (0.55, 0.00),
    (0.62, 0.26), (0.70, 0.00), (1.00, 0.00),
]


def _pulse_path(x, y, w, h):
    path = QPainterPath()
    base = y + h * 0.62
    amp = h * 0.52
    pts = [(x + px * w, base - py * amp) for px, py in HEARTBEAT]
    path.moveTo(*pts[0])
    for p in pts[1:]:
        path.lineTo(*p)
    return path


def make_app_icon():
    pm = QPixmap(256, 256)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(BG))
    p.drawRoundedRect(8, 8, 240, 240, 52, 52)
    highlight = QColor(XAI)
    highlight.setAlpha(70)
    p.setBrush(highlight)
    p.drawRoundedRect(QRect(28 + int(0.30 * 200), 30, int(0.15 * 200), 196), 14, 14)
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QColor(GRID_MAJOR), 3))
    for i in range(1, 4):
        p.drawLine(8, 8 + i * 60, 248, 8 + i * 60)
        p.drawLine(8 + i * 60, 8, 8 + i * 60, 248)
    p.setPen(QPen(QColor(ACCENT), 16, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.drawPath(_pulse_path(28, 40, 200, 160))
    p.end()
    return QIcon(pm)


def _dot_icon(color):
    pm = QPixmap(14, 14)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    p.drawEllipse(3, 3, 8, 8)
    p.end()
    return QIcon(pm)


# --------------------
# Custom-drawn widgets
# --------------------


class PulseWidget(QWidget):

    def __init__(self):
        super().__init__()
        self.setFixedSize(56, 30)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(ACCENT), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        p.drawPath(_pulse_path(2, 2, self.width() - 4, self.height() - 6))
        p.end()


class CheckBox(QCheckBox):

    def _indicator_rect(self):
        opt = QStyleOptionButton()
        opt.initFrom(self)
        rect = self.style().subElementRect(
            QStyle.SubElement.SE_CheckBoxIndicator, opt, self)
        if rect.isEmpty():
            side = 18
            rect = QRect(0, (self.height() - side) // 2, side, side)
        return rect

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.checkState() != Qt.Checked:
            return
        r = self._indicator_rect()
        x, y, w, h = r.x(), r.y(), r.width(), r.height()
        tick = QPainterPath()
        tick.moveTo(x + w * 0.26, y + h * 0.52)
        tick.lineTo(x + w * 0.43, y + h * 0.70)
        tick.lineTo(x + w * 0.75, y + h * 0.30)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(ON_ACCENT if self.isEnabled() else "#5E7A6C"), 2.0,
                      Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawPath(tick)
        p.end()


class Combo(QComboBox):

    def __init__(self, parent=None):
        super().__init__(parent)
        view = QListView()
        view.setUniformItemSizes(True)
        self.setView(view)

    def paintEvent(self, event):
        super().paintEvent(event)
        edit = self.lineEdit()
        if not self.isEnabled():
            color = QColor(DISABLED_TEXT)
        elif (self.hasFocus() or self.view().isVisible()
                or (edit is not None and edit.hasFocus())):
            color = QColor(ACCENT)
        else:
            color = QColor(MUTED)
        cx = self.width() - 14.0
        cy = self.height() / 2.0
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(color, 1.7, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.drawLine(QPointF(cx - 4.0, cy - 2.0), QPointF(cx, cy + 2.2))
        p.drawLine(QPointF(cx, cy + 2.2), QPointF(cx + 4.0, cy - 2.0))
        p.end()


class RateCombo(Combo):

    PRESETS = (100, 128, 200, 250, 256, 360, 500, 512, 1000, 1024)
    MIN_HZ, MAX_HZ = 1, 10000

    def __init__(self, value=500, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        for hz in self.PRESETS:
            self.addItem(f"{hz} Hz", hz)
        self.lineEdit().setValidator(QRegularExpressionValidator(
            QRegularExpression(r"\d{0,5}\s*[Hh]?[Zz]?"), self))
        self.lineEdit().editingFinished.connect(self.commit)
        self._last_valid = 500
        self.set_value(value)

    def value(self):
        digits = "".join(ch for ch in self.currentText() if ch.isdigit())
        return int(digits) if digits else 0

    def rate(self):
        return min(max(self.value() or self._last_valid, self.MIN_HZ), self.MAX_HZ)

    def commit(self):
        hz = self.rate()
        self._last_valid = hz
        text = f"{hz} Hz"
        if text != self.currentText():
            self.lineEdit().setText(text)
        return hz

    def set_value(self, hz):
        hz = int(hz or 0)
        if hz:
            self._last_valid = hz
        index = self.findData(hz)
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            self.setCurrentIndex(-1)
            self.setEditText(f"{hz} Hz" if hz else "")


class NumberBox(QDoubleSpinBox):

    def __init__(self, lo, hi, step, decimals, value, suffix=""):
        super().__init__()
        self.setRange(lo, hi)
        self.setSingleStep(step)
        self.setDecimals(decimals)
        self.setValue(value)
        self.setSuffix(suffix)
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setKeyboardTracking(False)
        self.setMinimumWidth(92)


class ElidedLabel(QLabel):

    def __init__(self, placeholder=""):
        super().__init__(placeholder)
        self._full = ""
        self._placeholder = placeholder
        self.setMinimumWidth(60)

    def set_full_text(self, text):
        self._full = text or ""
        self.setToolTip(self._full)
        self._apply()

    def full_text(self):
        return self._full

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply()

    def _apply(self):
        if not self._full:
            super().setText(self._placeholder)
            return
        fm = self.fontMetrics()
        super().setText(fm.elidedText(self._full, Qt.ElideMiddle, max(60, self.width() - 4)))


class Segmented(QWidget):

    changed = Signal(str)

    def __init__(self, items):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = {}
        for i, (key, label) in enumerate(items):
            name = "segL" if i == 0 else ("segR" if i == len(items) - 1 else "segM")
            btn = QPushButton(label, objectName=name)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            self.group.addButton(btn)
            self.buttons[key] = btn
            lay.addWidget(btn)
            btn.toggled.connect(lambda on, k=key: on and self.changed.emit(k))

    def key(self):
        for k, b in self.buttons.items():
            if b.isChecked():
                return k
        return None

    def set_key(self, key):
        if key in self.buttons:
            self.buttons[key].setChecked(True)


class LeadChips(QWidget):

    changed = Signal()

    def __init__(self):
        super().__init__()
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(4)
        self._buttons = []

    def set_leads(self, names, visible=None):
        for b in self._buttons:
            self._lay.removeWidget(b)
            b.deleteLater()
        self._buttons = []
        for name in names:
            key = canonical_lead_name(name)
            text = lead_label(key) if key in FINAL_LEADS else name
            b = QPushButton(objectName="lead")
            b.setProperty("label", text)
            b.setProperty("standard", key in FINAL_LEADS)
            b.setProperty("lead", name)
            b.setCheckable(True)
            b.setChecked(visible is None or name in visible)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(f"{name}: show or hide this lead (right-click: show only this lead)")
            b.setContextMenuPolicy(Qt.CustomContextMenu)
            b.customContextMenuRequested.connect(lambda _pos, n=name: self.solo(n))
            b.toggled.connect(lambda _on: self.changed.emit())
            self._lay.addWidget(b)
            self._buttons.append(b)
        self._elide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def _elide(self):
        if not self._buttons:
            return
        room = (self.width() - self._lay.spacing() * (len(self._buttons) - 1)) // len(self._buttons)
        for b in self._buttons:
            label = b.property("label")
            if b.property("standard"):      # 'aVR', 'V6': short enough, never cut
                b.setText(label)
            else:
                b.setText(QFontMetrics(b.font()).elidedText(label, Qt.ElideRight,
                                                            max(16, min(90, room - 14))))

    def visible(self):
        return [b.property("lead") for b in self._buttons if b.isChecked()]

    def names(self):
        return [b.property("lead") for b in self._buttons]

    def set_all(self, on=True):
        for b in self._buttons:
            b.blockSignals(True)
            b.setChecked(on)
            b.blockSignals(False)
        self.changed.emit()

    def solo(self, name):
        for b in self._buttons:
            b.blockSignals(True)
            b.setChecked(b.property("lead") == name)
            b.blockSignals(False)
        self.changed.emit()


class ZoomPan:

    def __init__(self, canvas):
        self.canvas = canvas
        self._home = {}
        self._press = None
        canvas.mpl_connect("scroll_event", self._on_scroll)
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("button_release_event", self._on_release)
        canvas.mpl_connect("motion_notify_event", self._on_motion)

    def remember(self):
        self._home = {ax: ax.get_xlim() for ax in self.canvas.figure.axes
                      if getattr(ax, "ecg_time_axis", False)}

    def _set(self, ax, lo, width):
        home_lo, home_hi = self._home[ax]
        width = min(width, home_hi - home_lo)
        lo = max(home_lo, min(lo, home_hi - width))
        ax.set_xlim(lo, lo + width)          # x-shared axes follow
        self.canvas.draw_idle()

    def _on_scroll(self, ev):
        ax = ev.inaxes
        if ax not in self._home or ev.xdata is None:
            return
        lo, hi = ax.get_xlim()
        width = (hi - lo) * (0.8 if ev.button == "up" else 1.25)
        frac = (ev.xdata - lo) / (hi - lo) if hi > lo else 0.5
        self._set(ax, ev.xdata - frac * width, width)

    def _on_press(self, ev):
        ax = ev.inaxes
        if ax not in self._home:
            return
        if ev.dblclick:
            lo, hi = self._home[ax]
            self._set(ax, lo, hi - lo)
        elif ev.button == 1:
            self._press = (ax, ev.x, ax.get_xlim())

    def _on_motion(self, ev):
        if not self._press or ev.x is None:
            return
        ax, x0, (lo, hi) = self._press
        width = ax.get_window_extent().width or 1
        self._set(ax, lo - (ev.x - x0) / width * (hi - lo), hi - lo)

    def _on_release(self, _ev):
        self._press = None


# -------
# Workers
# -------


class ScanWorker(QThread):

    sig_done = Signal(int, object)  # generation, [Path]

    def __init__(self, generation, path):
        super().__init__()
        self.generation = generation
        self.path = path

    def run(self):
        files = []
        try:
            files = find_files(Path(self.path), SUPPORTED_EXTENSIONS)
        except Exception:
            pass
        self.sig_done.emit(self.generation, files)


class ProcessWorker(QThread):

    sig_progress = Signal(int, int, str)          # done, total, current record
    sig_counts = Signal(dict)                     # success/skipped/failed/positive
    sig_log = Signal(str)
    sig_finished = Signal(object, object, object, dict, bool, str, str)
    sig_error = Signal(str)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        runner = None
        try:
            s = self.settings
            runner = ExecutionRunner(path_source=s["input"], path_sink=s["output"],
                                     use_config=s.get("use_config", True))
            for key in ("pipeline", "notch", "default_fs", "resample", "resample_method",
                        "derive_leads", "model", "threshold", "window_start",
                        "allow_missing_leads", "metadata_csv", "fallback_age", "fallback_sex",
                        "xai_method", "xai_lead", "save_plots", "save_maps", "anonymize"):
                setattr(runner, key, s[key])
            if s.get("emd_method"):
                runner.emd_options["method"] = s["emd_method"]
            runner.keep_signals = True

            def progress(done, total, name):
                self.sig_progress.emit(done, total, name)
                self.sig_counts.emit(dict(runner.stats))

            records = runner.run(
                progress_cb=progress,
                log_cb=self.sig_log.emit,
                cancel_cb=lambda: self._cancelled,
            )
            self.sig_counts.emit(dict(runner.stats))
            summary_path = str(runner.summary_path) if runner.summary_path else None
            self.sig_finished.emit(records, list(runner.summary), summary_path,
                                   dict(runner.stats), runner.cancelled, runner.device,
                                   runner.summary_error or "")
        except Exception as exc:
            if runner is not None:
                self.sig_counts.emit(dict(runner.stats))
            self.sig_error.emit(str(exc))


class ExplainWorker(QThread):

    sig_done = Signal(str, str, str, object)      # record id, method, lead, result
    sig_failed = Signal(str, str, str, str)       # record id, method, lead, message

    def __init__(self, record, method, lead, device):
        super().__init__()
        self.record, self.method, self.lead, self.device = record, method, lead, device

    def run(self):
        rid = self.record["id"]
        try:
            from ecgxai.analyze import explain_record
            from ecgxai.registry import load_model
            loaded = load_model(self.record["model"], self.device)
            res = explain_record(loaded, self.record, self.method, self.lead)
            self.sig_done.emit(rid, self.method, self.lead, res)
        except Exception as exc:
            self.sig_failed.emit(rid, self.method, self.lead, str(exc))


# -----------
# Main window
# -----------

XAI_ITEMS = [(name, key) for key, name in XAI_METHODS.items()]
VIEWS = [("raw", "Raw"), ("processed", "Processed"), ("model", "Model input"), ("xai", "XAI")]
ATTR_COLUMNS = [f"{m}_{s}" for m in XAI_METHODS for s in SHARE_SUFFIXES]   # gradcam_p_wave ..
NUMERIC_COLUMNS = {"patient_age", "sampling_freq", "n_samples", "window_start_s", "probability",
                   "threshold", *ATTR_COLUMNS}


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP)
        self.setMinimumHeight(600)
        avail = self.screen().availableGeometry() if self.screen() is not None else None
        self.resize(min(1560, avail.width() - 16) if avail else 1560,
                    min(920, avail.height() - 40) if avail else 920)
        self.setAcceptDrops(True)

        self._records = {}
        self._summary = []
        self._summary_path = None
        self._output_dir = ""
        self._metadata_csv = ""
        self._worker = None
        self._scan_generation = 0
        self._scanners = []                 # in-flight scans; see _set_input
        self._counts = {}                   # files per format in the selected input
        self._scanned = ([], None)          # (find_files list, the input path it is for)
        self._run_device = "cpu"
        self._run_lead = "II"               # lead the last run explained with every method
        self._xai_worker = None
        self._xai_pending = None            # (record id, method, lead) waiting for the worker
        self._xai_errors = {}               # (record id, method, lead) -> message
        self._visible = {"raw": None, "std": None}   # lead visibility per lead set (None = all)
        self._close_requested = False       # window closes once the running batch ends
        self._closing = False               # closeEvent accepted: start no more workers

        self._config_error, startup_error = None, None
        try:
            defaults = ExecutionRunner()  # reads config.ini for initial values
            defaults.options()            # and its values must be usable, as the CLI checks
        except ValueError as exc:         # a broken config.ini must not keep the app from opening
            defaults = ExecutionRunner(use_config=False)
            self._config_error = str(exc)
        except ImportError as exc:
            startup_error = (f"A required package cannot be imported: {exc}\n\n"
                             "Reinstall them with: pip install -r requirements.txt")
        self._defaults = defaults

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(16, 14, 16, 12)
        body.setSpacing(16)
        outer.addLayout(body, 1)

        body.addWidget(self._build_left(defaults))
        body.addLayout(self._build_right(), 1)

        self.statusBar().showMessage("Ready")
        self._apply_defaults(defaults)
        self._wire_shortcuts()
        self._refresh_start_enabled()
        self._refresh_viewer_controls()
        self._redraw()
        if startup_error:
            self._append_log(f"Error: {startup_error}")
            QTimer.singleShot(0, lambda: QMessageBox.critical(self, APP, startup_error))
        if self._config_error:
            self._append_log(f"Error: {self._config_error}")
            QTimer.singleShot(0, lambda: QMessageBox.warning(
                self, APP, f"{self._config_error}\n\nThe app uses its built-in defaults until "
                           "config.ini is fixed and the app restarted."))

    # Header
    def _build_header(self):
        header = QWidget(objectName="Header")
        lay = QHBoxLayout(header)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(12)
        lay.addWidget(PulseWidget())
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(QLabel(APP, objectName="AppTitle"))
        titles.addWidget(QLabel("Ischemia prediction and explainability for 12-lead ECG recordings",
                                objectName="AppSub"))
        lay.addLayout(titles)
        lay.addStretch(1)
        lay.addWidget(QLabel(f"v{__version__}", objectName="Version"))
        return header

    # Left column
    def _group(self, title):
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lay.setSpacing(6)
        lay.setContentsMargins(12, 10, 12, 10)
        return box, lay

    def _hint(self, text=""):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        font = QFont(lbl.font())
        font.setPixelSize(11)
        lbl.setFont(font)
        lbl.setStyleSheet(f"color: {MUTED};")
        return lbl

    GUTTER = 12

    def _build_left(self, defaults):
        column = QWidget()
        col = QVBoxLayout(column)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self._scroll = scroll
        panel = QWidget()
        self._panel = panel
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, self.GUTTER, 0)
        lay.setSpacing(8)
        scroll.setWidget(panel)
        col.addWidget(scroll, 1)

        # Input
        box, g = self._group("Input")
        row = QHBoxLayout()
        self.btn_in_folder = QPushButton("Select folder…")
        self.btn_in_file = QPushButton("Select file…")
        row.addWidget(self.btn_in_folder)
        row.addWidget(self.btn_in_file)
        g.addLayout(row)
        self.lbl_input = ElidedLabel("No input selected")
        self.lbl_input.setStyleSheet(f"color: {MUTED};")
        g.addWidget(self.lbl_input)
        self.lbl_scan = QLabel("Drop a file/folder in this window")
        self.lbl_scan.setStyleSheet(f"color: {MUTED}; font-family: {MONO}; font-size: 11px;")
        self.lbl_scan.setWordWrap(True)
        g.addWidget(self.lbl_scan)
        lay.addWidget(box)

        # Output
        box, g = self._group("Output")
        self.btn_out_folder = QPushButton("Select output folder…")
        g.addWidget(self.btn_out_folder)
        self.lbl_output = ElidedLabel("No output folder selected")
        self.lbl_output.setStyleSheet(f"color: {MUTED};")
        g.addWidget(self.lbl_output)
        self.chk_save_plots = CheckBox("Save XAI figures (PNG)")
        self.chk_save_plots.setToolTip("One <record>_xai_<method>.png per record and XAI method "
                                       "(Grad-CAM, saliency map, SmoothGrad×SIGN, LIME), as shown "
                                       "in the XAI view")
        g.addWidget(self.chk_save_plots)
        self.chk_save_maps = CheckBox("Save attribution maps (.npz)")
        self.chk_save_maps.setToolTip("One <record>_xai_<method>.npz per record and XAI method: "
                                      "model input, attribution map and segment labels, for your "
                                      "own analysis")
        g.addWidget(self.chk_save_maps)
        lay.addWidget(box)

        # Pre-processing
        box, g = self._group("Pre-processing")
        g.addWidget(self._hint("Filters applied to the whole record, in this order"))
        row = QHBoxLayout()
        self.chk_notch = CheckBox("Notch filter")
        self.chk_notch.setToolTip("Removes mains interference at the selected frequency")
        self.cmb_notch = Combo()
        self.cmb_notch.addItem("50 Hz", 50)
        self.cmb_notch.addItem("60 Hz", 60)
        self.cmb_notch.setEnabled(False)
        row.addWidget(self.chk_notch, 1)
        row.addWidget(self.cmb_notch)
        g.addLayout(row)
        self.chk_bandpass = CheckBox("Bandpass 0.5-40 Hz")
        self.chk_bandpass.setToolTip("3rd-order Butterworth, zero-phase")
        g.addWidget(self.chk_bandpass)
        self.chk_wavelet = CheckBox("Wavelet baseline removal")
        self.chk_wavelet.setToolTip("db4 decomposition: the approximation below 0.5 Hz (level 9 at "
                                    "500 Hz, scaled with the sampling rate) is subtracted")
        g.addWidget(self.chk_wavelet)
        row = QHBoxLayout()
        self.chk_emd = CheckBox("EMD denoise")
        self.chk_emd.setToolTip(
            "Empirical Mode Decomposition. Thresholds the noise components, keeps the QRS\n"
            "complexes at full bandwidth and removes the baseline trend. Slow on long records.")
        self.cmb_emd = Combo()
        self.cmb_emd.addItem("Standard", "emd")
        self.cmb_emd.addItem("Ensemble", "eemd")
        self.cmb_emd.addItem("CEEMDAN", "ceemdan")
        self.cmb_emd.setToolTip("Ensemble and CEEMDAN separate the components better "
                                "but take far longer.")
        self.cmb_emd.setEnabled(False)
        row.addWidget(self.chk_emd, 1)
        row.addWidget(self.cmb_emd)
        g.addLayout(row)
        self.chk_derive = CheckBox("Derive missing leads")
        self.chk_derive.setToolTip("Einthoven / Goldberger: e.g. III, aVR, aVL, aVF from I and II.\n"
                                   "Derived leads are listed in the summary (calculated_leads).")
        g.addWidget(self.chk_derive)
        lay.addWidget(box)

        # Sampling
        box, g = self._group("Sampling")
        row = QHBoxLayout()
        self.chk_fs = CheckBox("Input sampling rate")
        self.chk_fs.setToolTip(
            "Sampling rate for files that store none (CSV, ASC, some MAT, XML/HL7 without a rate).\n"
            "Select a preset or type any value in Hz. Files that carry their own rate keep it.")
        self.cmb_fs = RateCombo(500)
        self.cmb_fs.setEnabled(False)
        self.cmb_fs.setToolTip("Pick a common rate or type any value in Hz.")
        row.addWidget(self.chk_fs, 1)
        row.addWidget(self.cmb_fs)
        g.addLayout(row)
        self.lbl_fs_hint = self._hint()
        self.lbl_fs_hint.setMinimumHeight(2 * QFontMetrics(self.lbl_fs_hint.font()).lineSpacing())
        self.lbl_fs_hint.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        g.addWidget(self.lbl_fs_hint)
        row = QHBoxLayout()
        self.chk_resample = CheckBox("Resample to")
        self.chk_resample.setToolTip("Changes the processed signal. The model input is brought "
                                     "to 500 Hz in any case.")
        self.cmb_resample = RateCombo(500)
        self.cmb_resample.setEnabled(False)
        self.cmb_resample.setToolTip("Pick a target rate or type any value in Hz.")
        self.cmb_resample_method = Combo()
        self.cmb_resample_method.addItem("FFT", "fft")
        self.cmb_resample_method.addItem("Interpolation", "interpolation")
        self.cmb_resample_method.setToolTip("Also used to bring the model input to 500 Hz.")
        row.addWidget(self.chk_resample)
        row.addWidget(self.cmb_resample)
        row.addWidget(self.cmb_resample_method, 1)
        g.addLayout(row)
        lay.addWidget(box)

        # Model
        box, g = self._group("Model")
        row = QHBoxLayout()
        row.addWidget(QLabel("Model"))
        self.cmb_model = Combo()
        for key, spec in MODELS.items():
            self.cmb_model.addItem(spec.name, key)
        row.addWidget(self.cmb_model, 1)
        g.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Decision threshold"))
        row.addStretch(1)
        self.num_threshold = NumberBox(0.0001, 0.9999, 0.01, 4, get_spec().threshold)
        self.num_threshold.setToolTip("P(ischemia) at or above this is labelled Ischemia.\n"
                                      f"The model's default is {get_spec().threshold:g}.")
        row.addWidget(self.num_threshold)
        g.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Window starts at"))
        row.addStretch(1)
        self.num_window = NumberBox(0.0, 3600.0, 0.5, 1, 0.0, " s")
        self.num_window.setToolTip("The model sees 3 s of the record from here (0 = the first 3 s,\n"
                                   "as in the original app).")
        row.addWidget(self.num_window)
        g.addLayout(row)
        self.chk_allow_missing = CheckBox("Run records with missing leads")
        self.chk_allow_missing.setToolTip(
            "Leads that are neither recorded nor derivable are zero-filled.\n"
            "Off: such records are skipped, because the model never saw zero-filled leads.")
        g.addWidget(self.chk_allow_missing)
        lay.addWidget(box)

        # Patient
        box, g = self._group("Age and sex")
        g.addWidget(self._hint("Taken from each file when it carries them; otherwise from the "
                               "metadata CSV; otherwise from the fallback values below."))
        row = QHBoxLayout()
        self.btn_metadata = QPushButton("Metadata CSV…")
        self.btn_metadata.setToolTip("Optional: an ID column plus age and/or sex columns. A row "
                                     "matches a record by file name,\nstored patient ID, or the "
                                     "leading number of the file name.")
        self.btn_metadata_clear = QToolButton()
        self.btn_metadata_clear.setText("✕")
        self.btn_metadata_clear.setToolTip("Clear the metadata CSV")
        self.btn_metadata_clear.setEnabled(False)
        row.addWidget(self.btn_metadata, 1)
        row.addWidget(self.btn_metadata_clear)
        g.addLayout(row)
        self.lbl_metadata = ElidedLabel("No metadata CSV")
        self.lbl_metadata.setStyleSheet(f"color: {MUTED};")
        g.addWidget(self.lbl_metadata)
        row = QHBoxLayout()
        self.chk_age = CheckBox("Fallback age")
        self.num_age = NumberBox(0, 125, 1, 1, 60, " y")
        self.num_age.setEnabled(False)
        row.addWidget(self.chk_age, 1)
        row.addWidget(self.num_age)
        g.addLayout(row)
        row = QHBoxLayout()
        self.chk_sex = CheckBox("Fallback sex")
        self.cmb_sex = Combo()
        self.cmb_sex.addItem("Female", "Female")
        self.cmb_sex.addItem("Male", "Male")
        self.cmb_sex.setEnabled(False)
        self.cmb_sex.setMinimumWidth(92)
        row.addWidget(self.chk_sex, 1)
        row.addWidget(self.cmb_sex)
        g.addLayout(row)
        self.lbl_patient_hint = self._hint()
        g.addWidget(self.lbl_patient_hint)
        lay.addWidget(box)

        # Privacy
        box, g = self._group("Privacy")
        self.chk_anonymize = CheckBox("Anonymize results")
        self.chk_anonymize.setToolTip("Leaves patient ID, names, birth date and exam date/time out "
                                      "of the summary, the figures and the viewer.\nFile names are "
                                      "kept: rename files that identify a patient.")
        g.addWidget(self.chk_anonymize)
        lay.addWidget(box)

        lay.addStretch(1)

        # Run
        bar = scroll.verticalScrollBar().sizeHint().width() or 12
        run = QVBoxLayout()
        run.setContentsMargins(0, 0, self.GUTTER + bar, 0)
        run.setSpacing(8)
        self.btn_start = QPushButton("▶  Start analysis", objectName="primary")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        run.addWidget(self.btn_start)
        run.addWidget(self.btn_cancel)
        col.addLayout(run)
        column.setFixedWidth(max(440, panel.minimumSizeHint().width() + self.GUTTER + bar))

        # Connections
        self.btn_in_folder.clicked.connect(self._pick_input_folder)
        self.btn_in_file.clicked.connect(self._pick_input_file)
        self.btn_out_folder.clicked.connect(self._pick_output_folder)
        self.btn_metadata.clicked.connect(self._pick_metadata)
        self.btn_metadata_clear.clicked.connect(self._clear_metadata)
        self.chk_notch.toggled.connect(self.cmb_notch.setEnabled)
        self.chk_emd.toggled.connect(self.cmb_emd.setEnabled)
        self.chk_fs.toggled.connect(self.cmb_fs.setEnabled)
        self.chk_fs.toggled.connect(self._refresh_hints)
        self.cmb_fs.currentTextChanged.connect(self._refresh_hints)
        self.chk_resample.toggled.connect(self.cmb_resample.setEnabled)
        self.chk_age.toggled.connect(self.num_age.setEnabled)
        self.chk_sex.toggled.connect(self.cmb_sex.setEnabled)
        self.chk_age.toggled.connect(self._refresh_hints)
        self.chk_sex.toggled.connect(self._refresh_hints)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_cancel.clicked.connect(self._on_cancel)
        return column

    # Right column

    def _build_right(self):
        """Results pane: count chips, progress bar, Activity / Viewer / Summary tabs."""
        right = QVBoxLayout()
        right.setSpacing(10)

        chips = QHBoxLayout()
        self.chip_ok = QLabel("0 analysed", objectName="chip_ok")
        self.chip_flag = QLabel("0 ischemia", objectName="chip_flag")
        self.chip_warn = QLabel("0 skipped", objectName="chip_warn")
        self.chip_err = QLabel("0 failed", objectName="chip_err")
        self.chip_flag.setToolTip("Records predicted Ischemia at the decision threshold")
        for chip in (self.chip_ok, self.chip_flag, self.chip_warn, self.chip_err):
            chips.addWidget(chip)
        chips.addStretch(1)
        right.addLayout(chips)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("idle")
        right.addWidget(self.progress)

        self.tabs = QTabWidget()
        right.addWidget(self.tabs, 1)

        # Activity
        self.log = QPlainTextEdit(objectName="log")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(8000)
        self.tabs.addTab(self.log, "Activity")

        # Viewer
        vw = QWidget()
        vl = QVBoxLayout(vw)
        vl.setContentsMargins(10, 10, 10, 10)
        vl.setSpacing(8)
        row = QHBoxLayout()
        row.addWidget(QLabel("Record"))
        self.cmb_records = Combo()
        self.cmb_records.setMinimumWidth(240)
        self.cmb_records.currentIndexChanged.connect(self._on_record_changed)
        row.addWidget(self.cmb_records, 1)
        self.btn_prev = QToolButton(objectName="nav")
        self.btn_prev.setText("‹")
        self.btn_prev.setToolTip("Previous record (Alt+Left)")
        self.btn_prev.clicked.connect(lambda: self._step_record(-1))
        self.btn_next = QToolButton(objectName="nav")
        self.btn_next.setText("›")
        self.btn_next.setToolTip("Next record (Alt+Right)")
        self.btn_next.clicked.connect(lambda: self._step_record(1))
        row.addWidget(self.btn_prev)
        row.addWidget(self.btn_next)
        row.addSpacing(12)
        self.seg_view = Segmented(VIEWS)
        self.seg_view.set_key("raw")
        self.seg_view.changed.connect(self._on_view_changed)
        row.addWidget(self.seg_view)
        vl.addLayout(row)

        row = QHBoxLayout()
        self.view_controls = QStackedWidget()
        page = QWidget()
        pl = QHBoxLayout(page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(6)
        pl.addWidget(QLabel("Leads"))
        self.chips_leads = LeadChips()
        self.chips_leads.changed.connect(self._on_leads_changed)
        pl.addWidget(self.chips_leads)
        self.btn_all_leads = QPushButton("All")
        self.btn_all_leads.setObjectName("lead")
        self.btn_all_leads.clicked.connect(lambda: self.chips_leads.set_all(True))
        pl.addWidget(self.btn_all_leads)
        pl.addStretch(1)
        self.view_controls.addWidget(page)
        page = QWidget()
        pl = QHBoxLayout(page)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(8)
        pl.addWidget(QLabel("Method"))
        self.cmb_view_method = Combo()
        for label, key in XAI_ITEMS:
            self.cmb_view_method.addItem(label, key)
        self.cmb_view_method.setToolTip("Grad-CAM: fast, one map over time for all leads.\n"
                                        "Saliency map: gradient per lead and sample.\n"
                                        f"SmoothGrad×SIGN: averaged over "
                                        f"{int(self._defaults.smoothgrad_samples)} noisy copies; "
                                        "slower, steadier.\n"
                                        f"LIME: {int(self._defaults.lime_samples)} copies with 50 ms "
                                        "pieces of the leads flattened; slowest,\none signed value "
                                        "per lead and 50 ms, explains the recorded signal.\n"
                                        "All four are computed during the run.")
        self.cmb_view_method.currentIndexChanged.connect(self._on_xai_choice_changed)
        pl.addWidget(self.cmb_view_method)
        pl.addSpacing(6)
        pl.addWidget(QLabel("Lead"))
        self.cmb_view_lead = Combo()
        for ld in FINAL_LEADS:
            self.cmb_view_lead.addItem(lead_label(ld), ld)
        self.cmb_view_lead.setToolTip("Lead whose attribution is split into P / QRS / ST / T.\n"
                                      "The run explains the lead set in config.ini (II by default) "
                                      "with every method;\nother leads are computed on demand.")
        self.cmb_view_lead.currentIndexChanged.connect(self._on_xai_choice_changed)
        pl.addWidget(self.cmb_view_lead)
        self.lbl_xai_status = QLabel("")
        self.lbl_xai_status.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        pl.addWidget(self.lbl_xai_status, 1)
        self.view_controls.addWidget(page)
        row.addWidget(self.view_controls, 1)
        self.btn_save_plot = QPushButton("Save plot…")
        self.btn_save_plot.setEnabled(False)
        self.btn_save_plot.clicked.connect(self._save_plot)
        row.addWidget(self.btn_save_plot)
        vl.addLayout(row)

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.fig.patch.set_facecolor(BG)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setStyleSheet(f"background-color: {BG};")
        self.canvas.setMinimumWidth(480)
        self.zoom = ZoomPan(self.canvas)
        split.addWidget(self.canvas)
        self.details = QTextBrowser(objectName="details")
        self.details.setOpenExternalLinks(False)
        self.details.setMinimumWidth(250)
        split.addWidget(self.details)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([900, 300])
        vl.addWidget(split, 1)
        self.tabs.addTab(vw, "Viewer")

        # Summary
        sm = QWidget()
        sl = QVBoxLayout(sm)
        sl.setContentsMargins(10, 10, 10, 10)
        sl.setSpacing(8)
        self.table = QTableWidget(0, len(SUMMARY_COLUMNS))
        self.table.setHorizontalHeaderLabels(SUMMARY_COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setDefaultSectionSize(110)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSortingEnabled(True)
        self.table.setToolTip("Double-click a row to open the record in the viewer;\n"
                              "in a method's columns (gradcam_…, saliency_…, smoothgrad_…, lime_…) "
                              "to see that explanation")
        self.table.cellDoubleClicked.connect(self._open_row)
        sl.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.btn_open_out = QPushButton("Open output folder")
        self.btn_open_out.setEnabled(False)
        self.btn_open_out.clicked.connect(self._open_output)
        self.btn_open_summary = QPushButton(f"Open {SUMMARY_NAME}")
        self.btn_open_summary.setEnabled(False)
        self.btn_open_summary.clicked.connect(self._open_summary)
        row.addWidget(self.btn_open_out)
        row.addWidget(self.btn_open_summary)
        row.addStretch(1)
        sl.addLayout(row)
        self.tabs.addTab(sm, "Summary")
        return right

    # Helpers

    def _apply_defaults(self, defaults):
        self.chk_save_plots.setChecked(defaults.save_plots)
        self.chk_save_maps.setChecked(defaults.save_maps)
        self.chk_notch.setChecked("notch" in defaults.pipeline)
        self.cmb_notch.setCurrentIndex(0 if int(defaults.notch) == 50 else 1)
        self.chk_bandpass.setChecked("bandpass" in defaults.pipeline)
        self.chk_wavelet.setChecked("wavelet" in defaults.pipeline)
        self.chk_emd.setChecked("emd" in defaults.pipeline)
        idx = self.cmb_emd.findData(defaults.emd_options.get("method", "emd"))
        if idx >= 0:
            self.cmb_emd.setCurrentIndex(idx)
        self.chk_derive.setChecked(defaults.derive_leads)
        if defaults.default_fs:
            self.chk_fs.setChecked(True)
            self.cmb_fs.set_value(int(defaults.default_fs))
        if defaults.resample:
            self.chk_resample.setChecked(True)
            self.cmb_resample.set_value(int(defaults.resample))
        self.cmb_resample_method.setCurrentIndex(0 if defaults.resample_method == "fft" else 1)
        try:
            spec = get_spec(defaults.model)
            self.cmb_model.setCurrentIndex(max(0, self.cmb_model.findData(spec.key)))
        except ValueError:
            spec = get_spec()
        self.num_threshold.setValue(float(defaults.threshold if defaults.threshold is not None
                                          else spec.threshold))
        self.num_window.setValue(float(defaults.window_start or 0))
        self.chk_allow_missing.setChecked(defaults.allow_missing_leads)
        if defaults.metadata_csv:
            self._metadata_csv = str(defaults.metadata_csv)
            self.lbl_metadata.set_full_text(self._metadata_csv)
            self.btn_metadata_clear.setEnabled(True)
        from ecgproc.patient import parse_age, parse_sex
        age = parse_age(defaults.fallback_age) if defaults.fallback_age not in (None, "") else None
        if age is not None:
            self.chk_age.setChecked(True)
            self.num_age.setValue(age)
        sex = parse_sex(defaults.fallback_sex) if defaults.fallback_sex not in (None, "") else None
        if sex is not None:
            self.chk_sex.setChecked(True)
            self.cmb_sex.setCurrentIndex(max(0, self.cmb_sex.findData(sex)))
        self.chk_anonymize.setChecked(defaults.anonymize)
        self._select_quietly(self.cmb_view_lead, self._config_lead())
        self._refresh_hints()
        self._report_adjusted_defaults(defaults)
        # Input and output stay empty on launch — the user picks both.

    def _report_adjusted_defaults(self, defaults):
        checks = [("threshold", defaults.threshold, self.num_threshold.value()),
                  ("window_start", defaults.window_start, self.num_window.value())]
        if defaults.default_fs:
            checks.append(("default_fs", defaults.default_fs, self.cmb_fs.rate()))
        if defaults.resample:
            checks.append(("resample", defaults.resample, self.cmb_resample.rate()))
        if self.chk_age.isChecked():
            from ecgproc.patient import parse_age
            checks.append(("fallback_age", parse_age(defaults.fallback_age), self.num_age.value()))
        for key, wanted, used in checks:
            try:
                differs = wanted is not None and abs(float(wanted) - float(used)) > 1e-9
            except (TypeError, ValueError):
                differs = False
            if differs:
                self._append_log(f"config.ini {key} = {wanted}: the app uses {used:g} "
                                 "(the closest value its control can hold).")

    def _config_lead(self):
        try:
            return lead_key(self._defaults.xai_lead)
        except (ValueError, TypeError):
            return "II"

    @staticmethod
    def _select_quietly(combo, data):
        i = combo.findData(data)
        if i < 0 or i == combo.currentIndex():
            return False
        combo.blockSignals(True)
        combo.setCurrentIndex(i)
        combo.blockSignals(False)
        return True

    def _refresh_hints(self, *_):
        rateless = self._counts.get("csv", 0) + self._counts.get("asc", 0)
        plural = "s" if rateless != 1 else ""
        if rateless and not self.chk_fs.isChecked():
            color = WARN
            text = (f"{rateless} selected file{plural} store{'' if rateless != 1 else 's'} no "
                    f"sampling rate and will be skipped: the model needs one. Set it here.")
        elif rateless:
            color = MUTED
            text = (f"{rateless} selected file{plural} store{'' if rateless != 1 else 's'} no "
                    f"sampling rate and will be read at {self.cmb_fs.rate()} Hz.")
        else:
            color, text = MUTED, ""
        self.lbl_fs_hint.setText(text)
        self.lbl_fs_hint.setStyleSheet(f"color: {color};")
        self.lbl_fs_hint.setVisible(bool(text))     # no empty gap when nothing needs saying

        bare = rateless   # CSV and ASC carry no demographics either
        covered = self._metadata_csv or (self.chk_age.isChecked() and self.chk_sex.isChecked())
        if bare and not covered:
            self.lbl_patient_hint.setText(
                f"{bare} selected CSV/ASC file{'s' if bare != 1 else ''} carr{'y' if bare != 1 else 'ies'} "
                "no age or sex: give a metadata CSV or both fallbacks, or "
                f"{'they' if bare != 1 else 'it'} will be skipped.")
            self.lbl_patient_hint.setStyleSheet(f"color: {WARN};")
        else:
            self.lbl_patient_hint.setText("Records with no age or sex from any source are skipped "
                                          "and logged.")
            self.lbl_patient_hint.setStyleSheet(f"color: {MUTED};")

    def _wire_shortcuts(self):
        binds = [("Ctrl+O", self._pick_input_folder),
                 ("Ctrl+Shift+O", self._pick_output_folder),
                 ("Ctrl+R", self._on_start),
                 ("Esc", self._on_cancel),
                 ("Alt+Left", lambda: self._step_record(-1)),
                 ("Alt+Right", lambda: self._step_record(1))]
        for i, (key, _label) in enumerate(VIEWS):
            binds.append((f"Ctrl+{i + 1}", lambda k=key: self._show_view(k)))
        for keys, fn in binds:
            sc = QShortcut(QKeySequence(keys), self)
            sc.activated.connect(fn)

    def _append_log(self, text):
        self.log.appendPlainText(str(text))

    def _set_input(self, path):
        self.lbl_input.set_full_text(path)
        self.lbl_scan.setText("Scanning…")
        self._scan_generation += 1
        self._counts = {}
        self._scanned = ([], None)
        self._refresh_hints()
        scanner = ScanWorker(self._scan_generation, path)
        scanner.sig_done.connect(self._on_scan_done)
        self._scanners.append(scanner)
        scanner.finished.connect(lambda s=scanner: self._retire_scanner(s))
        scanner.start()
        self._refresh_start_enabled()

    def _retire_scanner(self, scanner):
        if scanner in self._scanners:
            self._scanners.remove(scanner)
        scanner.deleteLater()

    def _set_output(self, path):
        self._output_dir = path
        self.lbl_output.set_full_text(path)
        self._recount()
        self._refresh_start_enabled()

    def _on_scan_done(self, generation, files):
        if generation != self._scan_generation:
            return
        self._scanned = (files, self.lbl_input.full_text())
        self._recount()

    def _recount(self):
        files, scanned_for = self._scanned
        if scanned_for is None or scanned_for != self.lbl_input.full_text():
            return                          # a scan of this input is still running
        try:
            runner = ExecutionRunner(path_source=scanned_for, path_sink=self._output_dir or None,
                                     use_config=False)
            runner.metadata_csv = self._metadata_csv or None
            files = runner.filter_inputs(files)
        except Exception:
            pass
        counts = {}
        for f in files:
            ext = f.suffix.lower().lstrip(".")
            kind = "wfdb" if ext in ("dat", "hea") else ext
            counts[kind] = counts.get(kind, 0) + 1
        total = len(files)
        if total == 0:
            self.lbl_scan.setText("No processable files found "
                                  "(mat, wfdb, xml, hl7, csv, asc, dcm).")
        else:
            parts = " · ".join(f"{k} {v}" for k, v in sorted(counts.items()))
            self.lbl_scan.setText(f"{total} file{'s' if total != 1 else ''} ready · {parts}")
        self._counts = counts
        self._refresh_hints()
        self._refresh_start_enabled()

    def _busy(self):
        return self._worker is not None and self._worker.isRunning()

    def _scanning(self):
        return bool(self.lbl_input.full_text()) and self._scanned[1] != self.lbl_input.full_text()

    def _refresh_start_enabled(self):
        """Start is live only with both paths chosen, the input scanned and no run in flight."""
        running = self._worker is not None and self._worker.isRunning()
        ready = bool(self.lbl_input.full_text()) and bool(self._output_dir)
        self.btn_start.setEnabled(ready and not running and not self._scanning())

    # Pickers

    def _pick_input_folder(self):
        if self._busy():
            return
        start = self.lbl_input.full_text() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select input folder", start)
        if path:
            self._set_input(path)

    def _pick_input_file(self):
        if self._busy():
            return
        start = self.lbl_input.full_text() or str(Path.home())
        exts = " ".join(f"*{e}" for e in SUPPORTED_EXTENSIONS)
        path, _ = QFileDialog.getOpenFileName(self, "Select ECG file", start,
                                              f"ECG files ({exts});;All files (*)")
        if path:
            self._set_input(path)

    def _pick_output_folder(self):
        if self._busy():
            return
        start = self._output_dir or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select output folder", start)
        if path:
            self._set_output(path)

    def _pick_metadata(self):
        if self._busy():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select metadata CSV",
                                              str(Path.home()), "CSV files (*.csv);;All files (*)")
        if path:
            self._metadata_csv = path
            self.lbl_metadata.set_full_text(path)
            self.btn_metadata_clear.setEnabled(True)
            self._recount()
            self._refresh_hints()

    def _clear_metadata(self):
        if self._busy():
            return
        self._metadata_csv = ""
        self.lbl_metadata.set_full_text("")
        self.btn_metadata_clear.setEnabled(False)
        self._recount()
        self._refresh_hints()

    # Drag-and-drop

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if self._busy():
            return
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self._set_input(path)

    # Run

    def _gather_settings(self):
        for box in (self.num_threshold, self.num_window, self.num_age):
            box.interpretText()
        pipeline = [name for name, box in (
            ("notch", self.chk_notch), ("bandpass", self.chk_bandpass),
            ("wavelet", self.chk_wavelet), ("emd", self.chk_emd),
        ) if box.isChecked()]
        return {
            "input": self.lbl_input.full_text(),
            "output": self._output_dir,
            "pipeline": pipeline,
            "notch": self.cmb_notch.currentData(),
            "emd_method": self.cmb_emd.currentData(),
            "default_fs": self.cmb_fs.commit() if self.chk_fs.isChecked() else None,
            "resample": self.cmb_resample.commit() if self.chk_resample.isChecked() else None,
            "resample_method": self.cmb_resample_method.currentData(),
            "derive_leads": self.chk_derive.isChecked(),
            "model": self.cmb_model.currentData(),
            "threshold": round(self.num_threshold.value(), 4),
            "window_start": self.num_window.value(),
            "allow_missing_leads": self.chk_allow_missing.isChecked(),
            "metadata_csv": self._metadata_csv or None,
            "fallback_age": self.num_age.value() if self.chk_age.isChecked() else None,
            "fallback_sex": self.cmb_sex.currentData() if self.chk_sex.isChecked() else None,
            "xai_method": "all",            # every method; the viewer switches between them
            "xai_lead": self._config_lead(),
            "save_plots": self.chk_save_plots.isChecked(),
            "save_maps": self.chk_save_maps.isChecked(),
            "anonymize": self.chk_anonymize.isChecked(),
            "use_config": self._config_error is None,
        }

    def _on_start(self):
        if self._worker is not None and self._worker.isRunning():
            return
        if self._scanning():
            self.statusBar().showMessage("Still scanning the input, start again in a moment")
            return
        s = self._gather_settings()
        if not s["input"] or not Path(s["input"]).exists():
            QMessageBox.warning(self, APP, "Select an input file or folder first.")
            return
        if not s["output"]:
            QMessageBox.warning(self, APP, "Select an output folder first.")
            return
        rateless = self._counts.get("csv", 0) + self._counts.get("asc", 0)
        if rateless and not s["default_fs"]:
            if QMessageBox.question(
                    self, APP,
                    f"{rateless} selected file{'s' if rateless != 1 else ''} (CSV/ASC) store no "
                    "sampling rate and will be skipped.\n\nTick \"Input sampling rate\" to give "
                    "them one.\n\nStart anyway?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
                return
        try:
            Path(s["output"]).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.warning(self, APP, f"Cannot use the output folder: {exc}")
            return

        self._records = {}
        self._summary = []
        self._summary_path = None
        self._run_lead = s["xai_lead"]
        self._xai_pending = None
        self._xai_errors = {}
        self.cmb_records.blockSignals(True)
        self.cmb_records.clear()
        self.cmb_records.blockSignals(False)
        self.table.setRowCount(0)
        self._set_counts({"success": 0, "skipped": 0, "failed": 0, "positive": 0})
        self._refresh_viewer_controls()
        self._redraw()

        pipe = " -> ".join(s["pipeline"]) if s["pipeline"] else "none"
        self._append_log("─" * 64)
        self._append_log(f"Run · {get_spec(s['model']).name} (threshold {s['threshold']:g}) · "
                         f"pipeline {pipe}"
                         + (f" · resample {s['resample']} Hz ({s['resample_method']})" if s["resample"] else "")
                         + f" · XAI {', '.join(XAI_METHODS.values())} on lead {lead_label(s['xai_lead'])}"
                         + (" · anonymize" if s["anonymize"] else ""))
        self._append_log(f"Input:  {s['input']}")
        self._append_log(f"Output: {s['output']}")

        self._panel.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("loading model…")
        self.statusBar().showMessage("Analysing…")
        self.tabs.setCurrentIndex(0)

        self._worker = ProcessWorker(s)
        self._worker.sig_progress.connect(self._on_progress)
        self._worker.sig_counts.connect(self._set_counts)
        self._worker.sig_log.connect(self._append_log)
        self._worker.sig_finished.connect(self._on_finished)
        self._worker.sig_error.connect(self._on_error)
        self._worker.start()

    def _on_cancel(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.statusBar().showMessage("Cancelling…")

    def _on_progress(self, done, total, name):
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        if name:
            self.progress.setFormat(f"{done + 1}/{total} · {name}")
        else:
            self.progress.setFormat(f"{done}/{total}")

    def _set_counts(self, counts):
        self.chip_ok.setText(f"{counts.get('success', 0)} analysed")
        self.chip_flag.setText(f"{counts.get('positive', 0)} ischemia")
        self.chip_warn.setText(f"{counts.get('skipped', 0)} skipped")
        self.chip_err.setText(f"{counts.get('failed', 0)} failed")

    def _on_finished(self, records, summary, summary_path, stats, cancelled, device,
                     summary_error=""):
        self._teardown_worker()
        self._records = records or {}
        self._summary = summary or []
        self._summary_path = summary_path
        self._run_device = device or "cpu"
        self._set_counts(stats)
        self._fill_summary_table()

        state = "Cancelled" if cancelled else "Done"
        line = (f"{state} · {stats['success']} analysed ({stats.get('positive', 0)} ischemia) · "
                f"{stats['skipped']} skipped · {stats['failed']} failed")
        self._append_log(line)
        if summary_path:
            self._append_log(f"Summary: {summary_path}")
        self.statusBar().showMessage(line)
        self.progress.setFormat(state.lower())

        self.btn_open_out.setEnabled(bool(self._output_dir))
        self.btn_open_summary.setEnabled(bool(summary_path))

        if self._records:
            first = next(iter(self._records.values()))
            self._select_quietly(self.cmb_view_lead, self._run_lead)   # the method stays as chosen
            self.cmb_records.blockSignals(True)
            for rid in sorted(self._records):
                rec = self._records[rid]
                self.cmb_records.addItem(_dot_icon(FLAG if self._is_positive(rec) else ACCENT), rid, rid)
            self.cmb_records.blockSignals(False)
            self._visible = {"raw": None, "std": None}
            self.cmb_records.setCurrentIndex(0)
            self._on_record_changed()
            self._show_view("xai" if first.get("xai") else "model")
            self.tabs.setCurrentIndex(1)
        elif self._summary:
            self.tabs.setCurrentIndex(2)
        self._refresh_viewer_controls()

        if self._close_requested:
            return
        if summary_error:
            where = (f"\n\nThe results were written to {Path(summary_path).name} instead."
                     if summary_path else "\n\nNo summary could be written; see the Activity tab.")
            QMessageBox.warning(self, APP, f"{summary_error}.{where}")
        if cancelled:
            return
        if stats["failed"] and not stats["success"]:
            QMessageBox.warning(self, APP, "Every file failed. See the Activity tab for details.")
        elif stats["skipped"] and not stats["success"]:
            QMessageBox.warning(self, APP, "Every file was skipped. The Activity tab says why "
                                           "(often: no sampling rate, or no age or sex).")

    def _on_error(self, message):
        self._teardown_worker()
        self._append_log(f"Error: {message}")
        self.statusBar().showMessage("Error")
        self.progress.setFormat("error")
        if not self._close_requested:
            QMessageBox.critical(self, APP, message)

    def _teardown_worker(self):
        if self._worker is not None:
            self._worker.wait(5000)
            self._worker.deleteLater()
            self._worker = None
        self._panel.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._refresh_start_enabled()
        if self._close_requested:
            QTimer.singleShot(0, self.close)

    # Summary table

    def _fill_summary_table(self):
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self._summary))
        pred_col = SUMMARY_COLUMNS.index("prediction")
        for r, entry in enumerate(self._summary):
            positive = self._row_positive(entry)
            for c, col in enumerate(SUMMARY_COLUMNS):
                val = entry.get(col, "")
                item = QTableWidgetItem()
                if col in NUMERIC_COLUMNS and val not in ("", None):
                    try:
                        item.setData(Qt.DisplayRole, val if isinstance(val, int)
                                     and not isinstance(val, bool) else float(val))
                    except (TypeError, ValueError):
                        item.setText(str(val))
                else:
                    item.setText("" if val is None else str(val))
                if c == pred_col:
                    item.setForeground(QColor(FLAG if positive else ACCENT))
                self.table.setItem(r, c, item)
        self.table.setSortingEnabled(True)
        self.table.sortItems(0, Qt.AscendingOrder)
        for col in ("filename", "xai_methods", *ATTR_COLUMNS):     # whole headers, e.g. smoothgrad_qrs
            self.table.resizeColumnToContents(SUMMARY_COLUMNS.index(col))
        header = self.table.horizontalHeader()
        for m in XAI_METHODS:                                       # file names stay elided
            c = SUMMARY_COLUMNS.index(f"{m}_plot")
            self.table.setColumnWidth(c, max(self.table.columnWidth(c), header.sectionSizeHint(c)))

    def _row_positive(self, row):
        try:
            spec = get_spec(self.cmb_model.currentData())
            return row.get("prediction") == spec.label(spec.positive_class)
        except ValueError:
            return False

    def _open_row(self, row, col):
        def cell(name):
            item = self.table.item(row, SUMMARY_COLUMNS.index(name))
            return item.text() if item else ""

        rid = cell("filename")
        idx = self.cmb_records.findData(rid)
        if idx < 0:
            self.statusBar().showMessage(f"{rid}: not kept for the viewer (only the first "
                                         "records of a large batch are)")
            return
        moved = []
        method = SUMMARY_COLUMNS[col].split("_", 1)[0]
        if method in XAI_METHODS:           # a method's column: show that explanation
            moved = [self._select_quietly(self.cmb_view_lead, canonical_lead_name(cell("xai_lead"))),
                     self._select_quietly(self.cmb_view_method, method)]
            if self.seg_view.key() != "xai":
                self.seg_view.blockSignals(True)
                self.seg_view.set_key("xai")
                self.seg_view.blockSignals(False)
                moved.append(True)
        if idx != self.cmb_records.currentIndex():
            self.cmb_records.setCurrentIndex(idx)       # redraws, with the row's method and lead
        elif any(moved):
            self._sync_view_controls()
            self._redraw()
        self.tabs.setCurrentIndex(1)

    def _open_output(self):
        if self._output_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._output_dir))

    def _open_summary(self):
        if self._summary_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._summary_path))

    # Viewer: state

    def _current(self):
        rid = self.cmb_records.currentData()
        return self._records.get(rid) if rid else None

    def _is_positive(self, rec):
        spec = get_spec(rec["model"])
        return rec["prediction"]["class"] == spec.positive_class

    def _refresh_viewer_controls(self):
        has = bool(self._records)
        n = self.cmb_records.count()
        i = self.cmb_records.currentIndex()
        self.cmb_records.setEnabled(has)
        self.btn_prev.setEnabled(has and i > 0)
        self.btn_next.setEnabled(has and 0 <= i < n - 1)
        for btn in self.seg_view.buttons.values():
            btn.setEnabled(has)
        self.view_controls.setEnabled(has)

    def _step_record(self, delta):
        n = self.cmb_records.count()
        if not n:
            return
        i = min(max(self.cmb_records.currentIndex() + delta, 0), n - 1)
        self.cmb_records.setCurrentIndex(i)

    def _show_view(self, key):
        if self._records:
            self.seg_view.set_key(key)

    def _on_record_changed(self, *_):
        rec = self._current()
        if rec is not None:
            names = rec["raw"]["names"]
            if self._visible["raw"] is not None and not set(self._visible["raw"]) & set(names):
                self._visible["raw"] = None
        self._refresh_viewer_controls()
        self._sync_view_controls()
        self._redraw()

    def _on_view_changed(self, _key):
        self._sync_view_controls()
        self._redraw()

    def _lead_set(self):
        return "raw" if self.seg_view.key() == "raw" else "std"

    def _sync_view_controls(self):
        view = self.seg_view.key()
        self.view_controls.setCurrentIndex(1 if view == "xai" else 0)
        rec = self._current()
        if rec is None or view == "xai":
            return
        names = (rec["raw"]["names"] if view == "raw" else [lead_label(x) for x in FINAL_LEADS])
        self.chips_leads.blockSignals(True)
        self.chips_leads.set_leads(names, self._visible[self._lead_set()])
        self.chips_leads.blockSignals(False)

    def _on_leads_changed(self):
        names, shown = self.chips_leads.names(), self.chips_leads.visible()
        self._visible[self._lead_set()] = None if len(shown) == len(names) else shown
        self._redraw()

    def _on_xai_choice_changed(self, *_):
        if self.seg_view.key() == "xai":
            self._redraw()

    # Viewer: drawing

    def _redraw(self):
        rec = self._current()
        view = self.seg_view.key()
        self.btn_save_plot.setEnabled(False)
        if rec is None:
            placeholder(self.fig, "Run an analysis to inspect its records here.")
            self.details.setHtml(self._empty_details())
            self._draw()
            return
        try:
            if view == "raw":
                raw = rec["raw"]
                draw_leads(self.fig, raw["signals"], raw["names"], raw["fs"],
                           visible=self._visible["raw"])
            elif view == "processed":
                proc = rec["processed"]
                w = rec["window"]
                draw_leads(self.fig, proc["signals"], [lead_label(x) for x in FINAL_LEADS],
                           proc["fs"], visible=self._visible["std"],
                           window=(w["start"] / w["fs"], w["stop"] / w["fs"]))
            elif view == "model":
                from ecgxai.analyze import model_window_of
                w = rec["window"]
                draw_leads(self.fig, model_window_of(rec), [lead_label(x) for x in FINAL_LEADS],
                           w["fs"], visible=self._visible["std"], t0=w["start"] / w["fs"])
            else:
                self._draw_xai(rec)
            self.btn_save_plot.setEnabled(bool(self.fig.axes))
        except Exception as exc:          # never let a drawing problem take the window down
            placeholder(self.fig, f"Could not draw this view: {exc}")
        self.details.setHtml(self._details_html(rec))
        self._draw()

    def _draw(self):
        self.canvas.draw_idle()
        self.zoom.remember()

    def _xai_choice(self):
        return self.cmb_view_method.currentData(), self.cmb_view_lead.currentData()

    def _draw_xai(self, rec):
        method, lead = self._xai_choice()
        res = rec.get("xai", {}).get((method, lead))
        key = (rec["id"], method, lead)
        if res is None:
            if key in self._xai_errors:
                placeholder(self.fig, f"{XAI_METHODS[method]} on lead {lead_label(lead)} failed: "
                                      f"{self._xai_errors[key]}")
                self.lbl_xai_status.setText("failed")
            else:
                placeholder(self.fig, f"Computing {XAI_METHODS[method]} for lead {lead_label(lead)}…")
                self.lbl_xai_status.setText("computing…")
                self._request_xai(rec, method, lead)
            return
        self.lbl_xai_status.setText(f"explains {res['target_label']}")
        from ecgxai.analyze import model_window_of
        idx = FINAL_LEADS.index(lead)
        window = model_window_of(rec)
        title, subtitle, color = xai_titles(rec["id"], lead_label(lead), res["method_name"],
                                            rec["prediction"], res["target_label"],
                                            self._is_positive(rec))
        w = rec["window"]
        draw_xai(self.fig, window[idx], w["fs"], res["map"][idx], res["labels"],
                 res["segments"], title, subtitle, color, res["segments"]["top_percentile"],
                 t0=w["start"] / w["fs"])

    def _request_xai(self, rec, method, lead):
        if self._closing:
            return
        if self._xai_worker is not None:
            self._xai_pending = (rec["id"], method, lead)
            return
        worker = ExplainWorker(rec, method, lead, self._run_device)
        worker.sig_done.connect(self._on_xai_done)
        worker.sig_failed.connect(self._on_xai_failed)
        worker.finished.connect(lambda w=worker: self._on_xai_worker_finished(w))
        self._xai_worker = worker
        worker.start()

    def _on_xai_done(self, rid, method, lead, _result):
        if self._current() is not None and self._current()["id"] == rid \
                and self.seg_view.key() == "xai" and self._xai_choice() == (method, lead):
            self._redraw()

    def _on_xai_failed(self, rid, method, lead, message):
        self._xai_errors[(rid, method, lead)] = message
        self._append_log(f"{rid}: {XAI_METHODS.get(method, method)} on lead "
                         f"{lead_label(lead)} failed: {message}")
        self._on_xai_done(rid, method, lead, None)

    def _on_xai_worker_finished(self, worker):
        if worker is self._xai_worker:
            self._xai_worker = None
        worker.wait()
        worker.deleteLater()
        pending, self._xai_pending = self._xai_pending, None
        if pending and not self._closing:
            rid, method, lead = pending
            rec = self._records.get(rid)
            if rec is not None and (method, lead) not in rec.get("xai", {}):
                self._request_xai(rec, method, lead)

    def _save_plot(self):
        rec = self._current()
        if rec is None:
            return
        view = self.seg_view.key()
        stem = Path(rec["id"]).stem
        suffix = view
        if view == "xai":
            method, lead = self._xai_choice()
            suffix = f"{method}_{lead_label(lead)}"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save plot", str(Path(self._output_dir or ".") / f"{stem}_{suffix}.png"),
            "PNG image (*.png)")
        if path:
            if not Path(path).suffix:
                path += ".png"
            try:
                self.fig.savefig(path, dpi=200, facecolor=self.fig.get_facecolor())
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, APP, f"Could not save the plot: {exc}")
                return
            self._append_log(f"Saved plot: {path}")

    # Viewer: details panel

    def _empty_details(self):
        return (f"<p style='color:{MUTED}'>Record details appear here: file, patient, signal, "
                "pre-processing, prediction and the attribution split.</p>")

    def _details_html(self, rec):
        esc = html.escape
        meta, pat, leads = rec["meta"], rec["patient"], rec["leads"]
        pred = rec["prediction"]
        spec = get_spec(rec["model"])
        anonymous = bool(rec.get("anonymized"))

        def section(title):
            return (f"<tr><td colspan='2' style='padding-top:10px; color:{MUTED}; "
                    f"font-size:11px; font-weight:600'>{esc(title.upper())}</td></tr>")

        def row(key, value, color=None):
            style = f" style='color:{color}'" if color else ""
            return (f"<tr><td style='color:{MUTED}; padding-right:10px'>{esc(key)}</td>"
                    f"<td{style}>{value}</td></tr>")

        def src(source, detail):
            return f"<span style='color:{MUTED}'> · {esc(source)}{(' ' + esc(detail)) if detail else ''}</span>"

        rows = ["<table cellspacing='0' cellpadding='1' width='100%'>",
                f"<tr><td colspan='2' style='font-size:13px; font-weight:700'>{esc(rec['id'])}</td></tr>",
                row("Format", esc(rec["format"])),
                row("Folder", esc(str(Path(rec["path"]).parent)))]
        rows.append(section("Patient"))
        if not anonymous:
            if meta.get("patient_id"):
                rows.append(row("ID", esc(str(meta["patient_id"]))))
            name = " ".join(str(meta.get(k) or "") for k in ("patient_firstname", "patient_lastname")).strip()
            if name:
                rows.append(row("Name", esc(name)))
            if meta.get("patient_birthdate"):
                rows.append(row("Birth date", esc(str(meta["patient_birthdate"]))))
        rows.append(row("Age", f"{pat['age']:g} y" + src(pat["age_source"], pat["age_detail"])))
        rows.append(row("Sex", esc(pat["sex"]) + src(pat["sex_source"], pat["sex_detail"])))
        exam = " ".join(str(meta.get(k) or "") for k in ("exam_date", "exam_time")).strip()
        if exam:
            rows.append(row("Exam", esc(exam)))

        raw = rec["raw"]
        n = raw["signals"].shape[1]
        rows.append(section("Signal"))
        rows.append(row("Sampling rate", f"{raw['fs']:g} Hz"
                        + src("file" if raw.get("fs_source") == "file" else "fallback", "")))
        rows.append(row("Samples", f"{n} ({n / raw['fs']:.1f} s)"))
        rows.append(row("Channels", str(len(raw["names"]))))
        if leads["calculated_leads"]:
            rows.append(row("Derived", esc(", ".join(lead_label(x) for x in leads["calculated_leads"]))))
        if leads["missing_leads"]:
            rows.append(row("Zero-filled", esc(", ".join(lead_label(x) for x in leads["missing_leads"])), WARN))
        if leads.get("empty_leads"):
            rows.append(row("Empty / flat", esc(", ".join(leads["empty_leads"])), WARN))
        if leads["ignored_leads"]:
            rows.append(row("Ignored", esc(", ".join(leads["ignored_leads"]))))

        w = rec["window"]
        rows.append(section("Pre-processing"))
        rows.append(row("Pipeline", esc(rec["pipeline"] or "none")))
        rows.append(row("Model input", f"{w['start'] / w['fs']:.2f}–{w['stop'] / w['fs']:.2f} s "
                                       f"at {w['fs']} Hz"))
        for note in rec.get("notes", []):
            rows.append(row("Note", esc(note), WARN))

        positive = self._is_positive(rec)
        rows.append(section("Prediction"))
        rows.append(row("Model", esc(spec.name)))
        rows.append(row("Result", f"<b>{esc(pred['label'])}</b>", FLAG if positive else ACCENT))
        rows.append(row(f"P({spec.labels[spec.positive_class].lower()})",
                        probability_text(pred["probability"], pred["threshold"])))
        rows.append(row("Threshold", f"{pred['threshold']:.10g}"))

        method, lead = self._xai_choice()
        res = rec.get("xai", {}).get((method, lead))
        if res is None and rec.get("xai"):
            res = next(iter(rec["xai"].values()))
        if res is not None:
            seg = res["segments"]
            rows.append(section("Explanation"))
            rows.append(row("Method", f"{esc(res['method_name'])} · lead {esc(lead_label(res['lead']))}"))
            rows.append(row("Explains", esc(res["target_label"])))
            if seg.get("method"):
                rows.append(row("Delineation", f"{esc(seg['method'])} ({seg['n_beats']} beats)"))
            if seg.get("shares"):
                rows.append(row("", f"<span style='color:{MUTED}'>share of the top "
                                    f"{100 - seg['top_percentile']:g} % attribution</span>"))
                from ecgxai.theme import SEGMENT_LABELS
                for key, label in SEGMENT_LABELS.items():
                    rows.append(row(label, f"{seg['shares'][key]:.1f} %", SEGMENT_COLORS[key]))
            if seg.get("note"):
                rows.append(row("Note", esc(seg["note"]), WARN))
        elif rec.get("xai_note"):
            rows.append(section("Explanation"))
            rows.append(row("Note", esc(rec["xai_note"]), WARN))
        rows.append("</table>")
        return "".join(rows)

    # Close

    def closeEvent(self, event):
        if self._worker is not None and self._worker.isRunning():
            self._close_requested = True
            self._worker.cancel()
            self.btn_cancel.setEnabled(False)
            self.statusBar().showMessage("Finishing the current file, then closing…")
            event.ignore()
            return
        self._closing = True
        self._xai_pending = None
        if self._xai_worker is not None:
            self._xai_worker.wait()
        if any(not s.isFinished() for s in self._scanners):
            app = QApplication.instance()
            app.setQuitOnLastWindowClosed(False)

            def quit_when_idle():
                if all(s.isFinished() for s in list(self._scanners)):
                    app.quit()
                else:
                    QTimer.singleShot(100, quit_when_idle)
            QTimer.singleShot(100, quit_when_idle)
        event.accept()


def build_palette():
    pal = QPalette()
    roles = {
        QPalette.Window: BG, QPalette.WindowText: TEXT,
        QPalette.Base: SURFACE2, QPalette.AlternateBase: SURFACE,
        QPalette.Text: TEXT, QPalette.Button: SURFACE,
        QPalette.ButtonText: TEXT, QPalette.Highlight: ACCENT,
        QPalette.HighlightedText: ON_ACCENT, QPalette.ToolTipBase: SURFACE,
        QPalette.ToolTipText: TEXT, QPalette.PlaceholderText: MUTED,
        QPalette.Link: "#6FB7FF", QPalette.BrightText: ERROR,
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(DISABLED_TEXT))
    return pal


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP)
    app.setStyle("Fusion")
    app.setPalette(build_palette())
    app.setStyleSheet(QSS)
    app.setWindowIcon(make_app_icon())
    window = MainWindow()
    window.show()
    rc = app.exec()
    for scanner in list(window._scanners):
        scanner.wait()
    return rc


if __name__ == "__main__":
    sys.exit(main())
