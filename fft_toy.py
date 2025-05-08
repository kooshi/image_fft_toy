# -*- coding: utf-8 -*-
import sys
import numpy as np
import cv2
import requests
from io import BytesIO
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QPushButton, QFileDialog, QLabel,
                            QComboBox, QCheckBox, QSlider, QSpinBox, QGroupBox,
                            QFormLayout, QDialog, QDialogButtonBox, QColorDialog,
                            QToolBar, QAction, QSizePolicy, QWidgetAction,
                            QRadioButton, QInputDialog)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QIcon, QCursor, QFont
from PyQt5.QtCore import Qt, QPoint, QSize, pyqtSignal, QRectF, QLine

# --- Helper Function for Color Conversion ---
def qimage_to_bgr_ndarray(qimg: QImage) -> np.ndarray | None:
    if qimg.isNull(): return None
    qimg = qimg.convertToFormat(QImage.Format_RGB888)
    width = qimg.width(); height = qimg.height()
    ptr = qimg.constBits();
    if not ptr: return None
    ptr.setsize(height * width * 3)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr_bgr.copy()

def bgr_ndarray_to_qimage(arr_bgr: np.ndarray) -> QImage | None:
    if arr_bgr is None: return None
    height, width, channel = arr_bgr.shape
    if channel != 3: return None
    bytes_per_line = 3 * width
    arr_rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    arr_rgb_contiguous = np.ascontiguousarray(arr_rgb)
    qimg = QImage(arr_rgb_contiguous.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return qimg.copy()
# --------------------------------------------

# --- Bresenham Line Algorithm ---
def bresenham_line(x0, y0, x1, y1):
    """Yields integer coordinates on the line from (x0, y0) to (x1, y1)."""
    dx = abs(x1 - x0); dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1; sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        yield (x0, y0)
        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 < dx: err += dx; y0 += sy
# --------------------------------

class DrawableCanvas(QWidget):
    drawingFinished = pyqtSignal(object) # Carries final BGR numpy array
    lineDrawn = pyqtSignal(QPoint, QPoint, QColor) # Carries start_point, end_point, pen_color (image coords)
    pointDrawn = pyqtSignal(QPoint, QColor)       # Carries point, pen_color (image coords)
    circleDrawn = pyqtSignal(QPoint, int, QColor) # Carries center_point, radius, pen_color (image coords)

    def __init__(self, parent=None, drawing_allowed=True):
        super().__init__(parent)
        self.display_image = None; self.drawing = False; self.last_point = QPoint(); self.last_image_point = QPoint()
        self.pen_color = QColor(Qt.red); self.pen_width = 5; self.drawing_enabled = False; self.drawing_allowed = drawing_allowed
        self.zoom_factor = 1.0; self.offset_x = 0; self.offset_y = 0; self._is_panning = False; self._initial_image_set = False
        self.setCursor(Qt.ArrowCursor); self.setMouseTracking(True)
        self.setAutoFillBackground(True); palette = self.palette(); palette.setColor(self.backgroundRole(), QColor(50, 50, 50)); self.setPalette(palette)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding); self.setMinimumSize(200, 150)

    def setImage(self, q_image: QImage | None):
        if q_image is None or q_image.isNull(): self.display_image = None; self._initial_image_set = False
        else:
            if q_image.format() != QImage.Format_RGB888: q_image = q_image.convertToFormat(QImage.Format_RGB888)
            self.display_image = q_image.copy()
            if not self._initial_image_set: self._initial_image_set = True
        self.update()
    def enableDrawing(self, enabled): self.drawing_enabled = enabled and self.drawing_allowed; self.updateCursor()
    def setPenColor(self, color): self.pen_color = color
    def setPenWidth(self, width): self.pen_width = width
    def updateCursor(self):
        if self.drawing_enabled: self.setCursor(Qt.CrossCursor)
        elif self._is_panning: self.setCursor(Qt.ClosedHandCursor)
        else: self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event):
        self.last_point = event.pos()
        if event.button() == Qt.LeftButton and self.drawing_enabled and self.drawing_allowed:
            if self.display_image and not self.display_image.isNull():
                self.drawing = True
                image_pos = self.mapWidgetToImage(event.pos())
                self.last_image_point = image_pos

                # --- Draw immediately on press ---
                effective_pen_width = max(1, int(round(self.pen_width / self.zoom_factor)))
                if effective_pen_width == 1:
                    self.pointDrawn.emit(image_pos, self.pen_color)
                else:
                    radius = effective_pen_width // 2
                    self.circleDrawn.emit(image_pos, radius, self.pen_color)
                # --- End immediate draw ---
            else: self.drawing = False
        elif event.button() == Qt.MiddleButton: self._is_panning = True; self.updateCursor()

    def mouseMoveEvent(self, event):
        current_pos = event.pos()
        if (event.buttons() & Qt.LeftButton) and self.drawing and self.drawing_enabled and self.drawing_allowed:
            if self.display_image and not self.display_image.isNull():
                current_image_pos = self.mapWidgetToImage(current_pos)
                if self.last_image_point != current_image_pos:
                    # Emit signal with line segment info for main app to handle
                    self.lineDrawn.emit(self.last_image_point, current_image_pos, self.pen_color)
                self.last_image_point = current_image_pos
                self.last_point = current_pos
        elif event.buttons() & Qt.MiddleButton and self._is_panning:
            diff = current_pos - self.last_point; self.offset_x += diff.x(); self.offset_y += diff.y(); self.last_point = current_pos; self.update()
        else: self.updateCursor()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing and self.drawing_allowed:
            self.drawing = False
            if self.display_image and not self.display_image.isNull():
                final_qimage = self.getQImage()
                if final_qimage:
                    numpy_data = qimage_to_bgr_ndarray(final_qimage)
                    if numpy_data is not None: self.drawingFinished.emit(numpy_data)
        elif event.button() == Qt.MiddleButton and self._is_panning: self._is_panning = False; self.updateCursor()

    def wheelEvent(self, event): # Identical
        if self.display_image is None or self.display_image.isNull(): return
        zoom_in_factor = 1.15; zoom_out_factor = 1 / 1.15; old_zoom = self.zoom_factor; delta = event.angleDelta().y()
        if delta > 0: self.zoom_factor *= zoom_in_factor
        elif delta < 0: self.zoom_factor *= zoom_out_factor
        else: return
        self.zoom_factor = max(0.05, min(100.0, self.zoom_factor)); mouse_pos = event.pos(); widget_width = self.width(); widget_height = self.height()
        mouse_x_rel_widget = mouse_pos.x(); mouse_y_rel_widget = mouse_pos.y(); img_w_old = self.display_image.width() * old_zoom; img_h_old = self.display_image.height() * old_zoom
        img_x_old_widget = (widget_width - img_w_old) / 2 + self.offset_x; img_y_old_widget = (widget_height - img_h_old) / 2 + self.offset_y
        if old_zoom > 1e-6: img_coord_x = (mouse_x_rel_widget - img_x_old_widget) / old_zoom; img_coord_y = (mouse_y_rel_widget - img_y_old_widget) / old_zoom
        else: img_coord_x = self.display_image.width() / 2; img_coord_y = self.display_image.height() / 2
        img_w_new = self.display_image.width() * self.zoom_factor; img_h_new = self.display_image.height() * self.zoom_factor
        img_x_new_target_widget = mouse_x_rel_widget - img_coord_x * self.zoom_factor; img_y_new_target_widget = mouse_y_rel_widget - img_coord_y * self.zoom_factor
        self.offset_x = img_x_new_target_widget - (widget_width - img_w_new) / 2; self.offset_y = img_y_new_target_widget - (widget_height - img_h_new) / 2; self.update()

    def paintEvent(self, event): # Nearest neighbor
        super().paintEvent(event); painter = QPainter(self)
        if self.display_image and not self.display_image.isNull():
            img_width_zoomed = self.display_image.width() * self.zoom_factor; img_height_zoomed = self.display_image.height() * self.zoom_factor
            draw_x = (self.width() - img_width_zoomed) / 2 + self.offset_x; draw_y = (self.height() - img_height_zoomed) / 2 + self.offset_y
            source_rect = QRectF(0, 0, self.display_image.width(), self.display_image.height()); target_rect = QRectF(draw_x, draw_y, img_width_zoomed, img_height_zoomed)
            painter.drawImage(target_rect, self.display_image, source_rect)
        else: painter.setPen(Qt.gray); painter.setFont(QFont("Arial", 10)); painter.drawText(self.rect(), Qt.AlignCenter, "No Data")

    def getQImage(self) -> QImage | None: return self.display_image.copy() if self.display_image and not self.display_image.isNull() else None
    def getImageData(self) -> np.ndarray | None: qimg = self.getQImage(); return qimage_to_bgr_ndarray(qimg) if qimg else None
    def mapWidgetToImage(self, widget_pos: QPoint) -> QPoint: # Identical
        if self.display_image is None or self.display_image.isNull() or self.zoom_factor < 1e-6: return QPoint(0, 0)
        img_width_zoomed = self.display_image.width() * self.zoom_factor; img_height_zoomed = self.display_image.height() * self.zoom_factor
        draw_x_widget = (self.width() - img_width_zoomed) / 2 + self.offset_x; draw_y_widget = (self.height() - img_height_zoomed) / 2 + self.offset_y
        relative_x = widget_pos.x() - draw_x_widget; relative_y = widget_pos.y() - draw_y_widget
        image_x = relative_x / self.zoom_factor; image_y = relative_y / self.zoom_factor
        return QPoint(int(round(image_x)), int(round(image_y)))
    def reset_view(self): self.zoom_factor = 1.0; self.offset_x = 0; self.offset_y = 0
    def scale_to_fit(self): # Identical
        if self.display_image is None or self.display_image.isNull() or self.width() <= 0 or self.height() <= 0: self.zoom_factor = 1.0; self.offset_x = 0; self.offset_y = 0; self.update(); return
        img_width = self.display_image.width(); img_height = self.display_image.height(); canvas_width = self.width(); canvas_height = self.height()
        if img_width <= 0 or img_height <= 0: self.zoom_factor = 1.0; self.offset_x = 0; self.offset_y = 0; self.update(); return
        margin = 0.95; width_ratio = (canvas_width * margin) / img_width; height_ratio = (canvas_height * margin) / img_height
        self.zoom_factor = max(0.05, min(width_ratio, height_ratio)); self.offset_x = 0; self.offset_y = 0; self.update()


class FourierTransformApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Fourier Transform Viewer (Color + Phase + Filters)")
        self.setGeometry(50, 50, 1600, 950)

        # Data storage (Identical)
        self.original_image = None; self.current_spatial_image = None; self.resulting_spatial_image = None
        self.frequency_image = None; self.magnitude_spectrum = None; self.normalized_spectrum = None
        self.global_log_mag_range = (0.0, 1.0)
        self.phase_vis_b = None; self.phase_vis_g = None; self.phase_vis_r = None
        self.display_normalized_spectrum = None; self.display_phase_vis_b = None; self.display_phase_vis_g = None; self.display_phase_vis_r = None

        # UI Setup
        self.central_widget = QWidget(); self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.create_toolbar(); self.create_control_panel(); self.create_drawing_panel(); self.setup_display_areas()
        self.statusBar().showMessage("Load a color image to begin."); self.update_tool_state(False)

    def create_toolbar(self): # Identical
        self.toolbar = QToolBar("Main Toolbar"); self.toolbar.setIconSize(QSize(24, 24)); self.addToolBar(Qt.LeftToolBarArea, self.toolbar)
        self.load_action = QAction(QIcon.fromTheme("document-open"), "Load Image", self); self.load_action.setStatusTip("Load image from disk"); self.load_action.triggered.connect(self.load_image); self.toolbar.addAction(self.load_action)
        self.random_action = QAction(QIcon.fromTheme("network-transmit-receive"), "Random Image", self); self.random_action.setStatusTip("Load random image from Lorem Picsum"); self.random_action.triggered.connect(self.open_random_image_dialog); self.toolbar.addAction(self.random_action)
        self.toolbar.addSeparator()
        self.reset_action = QAction(QIcon.fromTheme("edit-undo"), "Reset Freq", self); self.reset_action.setStatusTip("Reset frequency edits, view, use current spatial"); self.reset_action.triggered.connect(self.reset_frequency_domain); self.toolbar.addAction(self.reset_action)
        self.fit_view_action = QAction(QIcon.fromTheme("zoom-fit-best"), "Fit View", self); self.fit_view_action.setStatusTip("Fit images to view"); self.fit_view_action.triggered.connect(self.fit_views); self.toolbar.addAction(self.fit_view_action)
        self.toolbar.addSeparator()
        self.pencil_action = QAction(QIcon.fromTheme("accessories-graphics"), "Pencil", self); self.pencil_action.setStatusTip("Toggle drawing mode"); self.pencil_action.setCheckable(True); self.pencil_action.setChecked(True); self.pencil_action.triggered.connect(self.toggle_drawing_mode); self.toolbar.addAction(self.pencil_action)
        self.color_action = QAction(QIcon.fromTheme("preferences-color"),"Color", self); self.color_action.setStatusTip("Change drawing color"); self.color_action.triggered.connect(self.choose_color); self.toolbar.addAction(self.color_action)
        pen_width_widget = QWidget(); pen_layout = QHBoxLayout(pen_width_widget); pen_layout.setContentsMargins(5, 0, 5, 0); pen_layout.addWidget(QLabel("Width:"))
        self.pen_width_spin = QSpinBox(); self.pen_width_spin.setRange(1, 50); self.pen_width_spin.setValue(5); self.pen_width_spin.valueChanged.connect(self.set_pen_width); pen_layout.addWidget(self.pen_width_spin)
        self.pen_action = QWidgetAction(self); self.pen_action.setDefaultWidget(pen_width_widget); self.pen_action.setStatusTip("Set drawing pen width"); self.toolbar.addAction(self.pen_action)
        self.toolbar.addSeparator(); self.toolbar.addWidget(QLabel(" Filters:"))
        self.radial_filter_action = QAction(QIcon.fromTheme("media-record"), "Radial LP", self); self.radial_filter_action.setStatusTip("Apply Radial Low-Pass Filter (Keep Center)"); self.radial_filter_action.triggered.connect(self.prompt_radial_filter); self.toolbar.addAction(self.radial_filter_action)
        self.magnitude_filter_action = QAction(QIcon.fromTheme("view-sort-descending"), "Magnitude HP", self); self.magnitude_filter_action.setStatusTip("Apply Log Magnitude High-Pass Filter (Keep Strongest)"); self.magnitude_filter_action.triggered.connect(self.prompt_magnitude_filter); self.toolbar.addAction(self.magnitude_filter_action)
        self.gaussian_filter_action = QAction(QIcon.fromTheme("image-blur"), "Gaussian LP", self); self.gaussian_filter_action.setStatusTip("Apply Gaussian Low-Pass Filter"); self.gaussian_filter_action.triggered.connect(self.prompt_gaussian_filter); self.toolbar.addAction(self.gaussian_filter_action)
        self.statusBar().showMessage("Ready")

    def create_control_panel(self): # Identical
        pass
    def create_drawing_panel(self): # Identical
        drawing_layout = QHBoxLayout(); help_label = QLabel("Zoom: Wheel | Pan: Middle Drag | Draw: Left Drag (Pencil active)"); help_label.setStyleSheet("font-size: 9pt; color: gray;"); drawing_layout.addWidget(help_label); drawing_layout.addStretch(1); self.main_layout.addLayout(drawing_layout)

    def setup_display_areas(self): # Added connections for point/circle drawn
        top_display_layout = QHBoxLayout(); top_display_layout.setSpacing(10)
        self.spatial_widget = QGroupBox("Original Spatial (BGR)"); self.spatial_layout = QVBoxLayout(self.spatial_widget); self.spatial_layout.setContentsMargins(5, 20, 5, 5); self.spatial_canvas = DrawableCanvas(drawing_allowed=True); self.spatial_canvas.setObjectName("SpatialCanvas"); self.spatial_canvas.drawingFinished.connect(self.spatial_drawing_finished); self.spatial_canvas.lineDrawn.connect(self.handle_spatial_line_drawn); self.spatial_canvas.pointDrawn.connect(self.handle_spatial_point_drawn); self.spatial_canvas.circleDrawn.connect(self.handle_spatial_circle_drawn); self.spatial_layout.addWidget(self.spatial_canvas); top_display_layout.addWidget(self.spatial_widget)
        self.freq_widget = QGroupBox("Frequency Domain (RGB Log Mag)"); self.freq_outer_layout = QVBoxLayout(self.freq_widget); self.freq_outer_layout.setContentsMargins(5, 20, 5, 5); self.freq_outer_layout.setSpacing(5)
        lock_layout = QHBoxLayout(); lock_layout.addWidget(QLabel("Lock Channels:")); self.lock_b_check = QCheckBox("B"); self.lock_b_check.setToolTip("Prevent Blue channel edits"); self.lock_g_check = QCheckBox("G"); self.lock_g_check.setToolTip("Prevent Green channel edits"); self.lock_r_check = QCheckBox("R"); self.lock_r_check.setToolTip("Prevent Red channel edits"); lock_layout.addWidget(self.lock_b_check); lock_layout.addWidget(self.lock_g_check); lock_layout.addWidget(self.lock_r_check); lock_layout.addStretch(); self.freq_outer_layout.addLayout(lock_layout)
        self.freq_canvas = DrawableCanvas(drawing_allowed=True); self.freq_canvas.setObjectName("FrequencyCanvas"); self.freq_canvas.setPenColor(QColor(Qt.black)); self.freq_canvas.drawingFinished.connect(self.freq_drawing_finished); self.freq_canvas.lineDrawn.connect(self.handle_freq_line_drawn); self.freq_canvas.pointDrawn.connect(self.handle_freq_point_drawn); self.freq_canvas.circleDrawn.connect(self.handle_freq_circle_drawn); self.freq_outer_layout.addWidget(self.freq_canvas); top_display_layout.addWidget(self.freq_widget)
        self.result_outer_widget = QWidget(); self.result_outer_layout = QVBoxLayout(self.result_outer_widget); self.result_outer_layout.setContentsMargins(0,0,0,0); self.result_outer_layout.setSpacing(5)
        self.result_widget = QGroupBox("Resulting Spatial (iFFT)"); self.result_layout = QVBoxLayout(self.result_widget); self.result_layout.setContentsMargins(5, 20, 5, 5); self.result_canvas = DrawableCanvas(drawing_allowed=False); self.result_canvas.setObjectName("ResultCanvas"); self.result_layout.addWidget(self.result_canvas); self.result_outer_layout.addWidget(self.result_widget)
        self.apply_button = QPushButton("Apply Result to Original"); self.apply_button.setToolTip("Copy this image back to 'Original Spatial'"); self.apply_button.clicked.connect(self.apply_result_to_original); self.result_outer_layout.addWidget(self.apply_button); top_display_layout.addWidget(self.result_outer_widget)
        self.main_layout.addLayout(top_display_layout, 3)
        phase_display_layout = QHBoxLayout(); phase_display_layout.setSpacing(10)
        self.phase_b_widget = QGroupBox("Blue Phase (Hue) & Log Mag (Value)"); self.phase_b_layout = QVBoxLayout(self.phase_b_widget); self.phase_b_layout.setContentsMargins(5, 20, 5, 5); self.phase_b_canvas = DrawableCanvas(drawing_allowed=True); self.phase_b_canvas.setObjectName("PhaseBCanvas"); self.phase_b_canvas.drawingFinished.connect(self.phase_b_drawing_finished); self.phase_b_canvas.lineDrawn.connect(lambda p1, p2, c: self.handle_phase_line_drawn(p1, p2, c, self.phase_b_canvas, 'display_phase_vis_b', 0)); self.phase_b_canvas.pointDrawn.connect(lambda p, c: self.handle_phase_point_drawn(p, c, self.phase_b_canvas, 'display_phase_vis_b', 0)); self.phase_b_canvas.circleDrawn.connect(lambda p, r, c: self.handle_phase_circle_drawn(p, r, c, self.phase_b_canvas, 'display_phase_vis_b', 0)); self.phase_b_layout.addWidget(self.phase_b_canvas); phase_display_layout.addWidget(self.phase_b_widget)
        self.phase_g_widget = QGroupBox("Green Phase (Hue) & Log Mag (Value)"); self.phase_g_layout = QVBoxLayout(self.phase_g_widget); self.phase_g_layout.setContentsMargins(5, 20, 5, 5); self.phase_g_canvas = DrawableCanvas(drawing_allowed=True); self.phase_g_canvas.setObjectName("PhaseGCanvas"); self.phase_g_canvas.drawingFinished.connect(self.phase_g_drawing_finished); self.phase_g_canvas.lineDrawn.connect(lambda p1, p2, c: self.handle_phase_line_drawn(p1, p2, c, self.phase_g_canvas, 'display_phase_vis_g', 1)); self.phase_g_canvas.pointDrawn.connect(lambda p, c: self.handle_phase_point_drawn(p, c, self.phase_g_canvas, 'display_phase_vis_g', 1)); self.phase_g_canvas.circleDrawn.connect(lambda p, r, c: self.handle_phase_circle_drawn(p, r, c, self.phase_g_canvas, 'display_phase_vis_g', 1)); self.phase_g_layout.addWidget(self.phase_g_canvas); phase_display_layout.addWidget(self.phase_g_widget)
        self.phase_r_widget = QGroupBox("Red Phase (Hue) & Log Mag (Value)"); self.phase_r_layout = QVBoxLayout(self.phase_r_widget); self.phase_r_layout.setContentsMargins(5, 20, 5, 5); self.phase_r_canvas = DrawableCanvas(drawing_allowed=True); self.phase_r_canvas.setObjectName("PhaseRCanvas"); self.phase_r_canvas.drawingFinished.connect(self.phase_r_drawing_finished); self.phase_r_canvas.lineDrawn.connect(lambda p1, p2, c: self.handle_phase_line_drawn(p1, p2, c, self.phase_r_canvas, 'display_phase_vis_r', 2)); self.phase_r_canvas.pointDrawn.connect(lambda p, c: self.handle_phase_point_drawn(p, c, self.phase_r_canvas, 'display_phase_vis_r', 2)); self.phase_r_canvas.circleDrawn.connect(lambda p, r, c: self.handle_phase_circle_drawn(p, r, c, self.phase_r_canvas, 'display_phase_vis_r', 2)); self.phase_r_layout.addWidget(self.phase_r_canvas); phase_display_layout.addWidget(self.phase_r_widget)
        self.main_layout.addLayout(phase_display_layout, 2)

    def choose_color(self): color = QColorDialog.getColor(self.spatial_canvas.pen_color, self, "Choose Drawing Color"); self.spatial_canvas.setPenColor(color); self.freq_canvas.setPenColor(color); self.phase_b_canvas.setPenColor(color); self.phase_g_canvas.setPenColor(color); self.phase_r_canvas.setPenColor(color)
    def set_pen_width(self, width): self.spatial_canvas.setPenWidth(width); self.freq_canvas.setPenWidth(width); self.phase_b_canvas.setPenWidth(width); self.phase_g_canvas.setPenWidth(width); self.phase_r_canvas.setPenWidth(width)
    def toggle_drawing_mode(self, checked): # Identical
        self.spatial_canvas.enableDrawing(checked); self.freq_canvas.enableDrawing(checked); self.phase_b_canvas.enableDrawing(checked); self.phase_g_canvas.enableDrawing(checked); self.phase_r_canvas.enableDrawing(checked); self.update_tool_state(self.original_image is not None)
    def update_tool_state(self, image_loaded=True): # Identical
        self.reset_action.setEnabled(image_loaded); self.fit_view_action.setEnabled(image_loaded); self.apply_button.setEnabled(image_loaded); self.pencil_action.setEnabled(image_loaded)
        self.radial_filter_action.setEnabled(image_loaded); self.magnitude_filter_action.setEnabled(image_loaded); self.gaussian_filter_action.setEnabled(image_loaded)
        is_drawing_active = image_loaded and self.pencil_action.isChecked(); self.color_action.setEnabled(is_drawing_active); self.pen_action.setEnabled(is_drawing_active); self.pen_width_spin.setEnabled(is_drawing_active)

    def load_image(self): # Identical
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if file_name:
            try:
                self.statusBar().showMessage(f"Loading {file_name}..."); QApplication.processEvents(); img_bgr = cv2.imread(file_name, cv2.IMREAD_COLOR)
                if img_bgr is None: img_bgr = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None: raise ValueError(f"Failed to load/decode: {file_name}")
                if len(img_bgr.shape) != 3 or img_bgr.shape[2] != 3: raise ValueError("Not 3-channel color.")
                self.original_image = img_bgr.copy(); self.current_spatial_image = img_bgr.copy()
                self.process_and_display(); self.statusBar().showMessage(f"Loaded COLOR image: {file_name}"); self.update_tool_state(True); self.pencil_action.setChecked(True); self.toggle_drawing_mode(True)
            except Exception as e: error_msg = f"Error loading image: {e}"; print(error_msg); self.statusBar().showMessage(error_msg); self.update_tool_state(False)

    def reset_frequency_domain(self): # Renamed from reset_all
        """Resets frequency edits based on the current spatial image."""
        if self.current_spatial_image is not None:
            try:
                self.compute_frequency_domain() # Recalculate true freq/phase AND resets display states
                if self.frequency_image is None: raise ValueError("Freq computation failed.")
                self.resulting_spatial_image = self.inverse_transform(self.frequency_image) # Recompute result
                if self.resulting_spatial_image is None: raise ValueError("iFFT failed.")
                self.update_display_content() # Update all canvases with new states
                self.fit_views() # Also reset view
                self.statusBar().showMessage("Frequency domain edits reset.")
            except Exception as e: error_msg = f"Error resetting frequency domain: {e}"; print(error_msg); self.statusBar().showMessage(error_msg)
        else: self.statusBar().showMessage("No image loaded to reset frequency domain.")

    def fit_views(self): # Identical
        if self.original_image is not None:
            self.spatial_canvas.scale_to_fit(); self.freq_canvas.scale_to_fit(); self.result_canvas.scale_to_fit()
            self.phase_b_canvas.scale_to_fit(); self.phase_g_canvas.scale_to_fit(); self.phase_r_canvas.scale_to_fit()
            self.statusBar().showMessage("Views fitted.")
        else: self.statusBar().showMessage("No image loaded.")

    def process_and_display(self): # Identical
        if self.current_spatial_image is None: self._clear_all_canvases(); self.update_tool_state(False); return
        try:
            self.compute_frequency_domain()
            if self.frequency_image is None: raise ValueError("Frequency computation failed.")
            self.resulting_spatial_image = self.inverse_transform(self.frequency_image)
            if self.resulting_spatial_image is None: raise ValueError("Initial Inverse Transform failed.")
            self.update_display_content(); self.fit_views(); self.update_tool_state(True)
        except Exception as e: error_msg = f"Error processing/displaying image: {e}"; print(error_msg); self.statusBar().showMessage(error_msg); self._clear_all_canvases(); self.update_tool_state(False)

    def _clear_all_canvases(self): # Identical
        self.spatial_canvas.setImage(None); self.freq_canvas.setImage(None); self.result_canvas.setImage(None); self.phase_b_canvas.setImage(None); self.phase_g_canvas.setImage(None); self.phase_r_canvas.setImage(None)

    def compute_frequency_domain(self): # Identical (with phase value clamping)
        if self.current_spatial_image is None or len(self.current_spatial_image.shape) != 3: self.frequency_image=None; self.magnitude_spectrum=None; self.normalized_spectrum=None; self.phase_vis_b=None; self.phase_vis_g=None; self.phase_vis_r=None; self.display_normalized_spectrum=None; self.display_phase_vis_b=None; self.display_phase_vis_g=None; self.display_phase_vis_r=None; return
        try:
            height, width, _ = self.current_spatial_image.shape; self.frequency_image = np.zeros((height, width, 3), dtype=np.complex128); self.magnitude_spectrum = np.zeros((height, width, 3), dtype=np.float64); epsilon = 1e-9
            for c in range(3): img_channel_float = self.current_spatial_image[:, :, c].astype(np.float32); f_transform_channel = np.fft.fft2(img_channel_float); shifted_fft = np.fft.fftshift(f_transform_channel); self.frequency_image[:, :, c] = shifted_fft; magnitude_channel = np.abs(shifted_fft); log_mag_channel = np.log1p(magnitude_channel + epsilon); self.magnitude_spectrum[:, :, c] = log_mag_channel
            global_min_log_mag = np.min(self.magnitude_spectrum); global_max_log_mag = np.max(self.magnitude_spectrum)
            if global_max_log_mag - global_min_log_mag < epsilon: global_max_log_mag = global_min_log_mag + 1.0
            self.global_log_mag_range = (global_min_log_mag, global_max_log_mag)
            self.normalized_spectrum = np.zeros((height, width, 3), dtype=np.uint8); map_bgr_to_rgb_idx = {0: 2, 1: 1, 2: 0}; phase_vis_storage = {0: None, 1: None, 2: None}
            for c_bgr in range(3):
                norm_log_mag_uint8 = self.normalize_for_display(self.magnitude_spectrum[:, :, c_bgr], global_min_log_mag, global_max_log_mag)
                c_rgb = map_bgr_to_rgb_idx[c_bgr]; self.normalized_spectrum[:, :, c_rgb] = norm_log_mag_uint8
                phase = np.angle(self.frequency_image[:, :, c_bgr]); hue = ((phase + np.pi) / (2 * np.pi)) * 179.0; hue = hue.astype(np.uint8)
                saturation = np.full_like(hue, 255, dtype=np.uint8); value = np.maximum(1, norm_log_mag_uint8) # Clamp Value
                hsv_vis = cv2.merge([hue, saturation, value]); bgr_vis = cv2.cvtColor(hsv_vis, cv2.COLOR_HSV2BGR); phase_vis_storage[c_bgr] = bgr_vis
            self.phase_vis_b = phase_vis_storage[0]; self.phase_vis_g = phase_vis_storage[1]; self.phase_vis_r = phase_vis_storage[2]
            self.display_normalized_spectrum = self.normalized_spectrum.copy() if self.normalized_spectrum is not None else None; self.display_phase_vis_b = self.phase_vis_b.copy() if self.phase_vis_b is not None else None; self.display_phase_vis_g = self.phase_vis_g.copy() if self.phase_vis_g is not None else None; self.display_phase_vis_r = self.phase_vis_r.copy() if self.phase_vis_r is not None else None
        except Exception as e: print(f"Error FFT/Phase: {e}"); self.frequency_image=None; self.magnitude_spectrum=None; self.normalized_spectrum=None; self.phase_vis_b=None; self.phase_vis_g=None; self.phase_vis_r=None; self.display_normalized_spectrum=None; self.display_phase_vis_b=None; self.display_phase_vis_g=None; self.display_phase_vis_r=None; self.global_log_mag_range=(0,1)

    def normalize_for_display(self, img_channel: np.ndarray, min_val=None, max_val=None) -> np.ndarray: # Identical
        if img_channel is None: return np.zeros((10,10), dtype=np.uint8)
        if min_val is None: min_val = np.min(img_channel)
        if max_val is None: max_val = np.max(img_channel)
        range_val = max_val - min_val
        if range_val < 1e-9: return np.zeros(img_channel.shape, dtype=np.uint8)
        img_clipped = np.clip(img_channel, min_val, max_val); normalized = (img_clipped - min_val) / range_val; normalized_uint8 = (normalized * 255).astype(np.uint8)
        return normalized_uint8

    # --- Update Canvas Content Helpers (Identical) ---
    def _update_spatial_canvas_content(self, bgr_data: np.ndarray | None):
        if bgr_data is not None:
            try: spatial_qimg = bgr_ndarray_to_qimage(bgr_data); self.spatial_canvas.setImage(spatial_qimg if spatial_qimg and not spatial_qimg.isNull() else None)
            except Exception as e: print(f"Error spatial content: {e}"); self.spatial_canvas.setImage(None)
        else: self.spatial_canvas.setImage(None)
    def _update_freq_canvas_content(self, rgb_norm_spectrum: np.ndarray | None):
        if rgb_norm_spectrum is not None:
            try: height, width, _ = rgb_norm_spectrum.shape; bytes_per_line = 3 * width; freq_data_rgb = np.ascontiguousarray(rgb_norm_spectrum); freq_qimg = QImage(freq_data_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888); self.freq_canvas.setImage(freq_qimg if freq_qimg and not freq_qimg.isNull() else None)
            except Exception as e: print(f"Error freq content: {e}"); self.freq_canvas.setImage(None)
        else: self.freq_canvas.setImage(None)
    def _update_phase_canvas_content(self, bgr_phase_vis: np.ndarray | None, target_canvas: DrawableCanvas):
        if bgr_phase_vis is not None:
            try: phase_qimg = bgr_ndarray_to_qimage(bgr_phase_vis); target_canvas.setImage(phase_qimg if phase_qimg and not phase_qimg.isNull() else None)
            except Exception as e: print(f"Error phase content: {e}"); target_canvas.setImage(None)
        else: target_canvas.setImage(None)
    def _update_result_canvas_content(self, bgr_result_data: np.ndarray | None):
        if bgr_result_data is not None:
            try: result_qimg = bgr_ndarray_to_qimage(bgr_result_data); self.result_canvas.setImage(result_qimg if result_qimg and not result_qimg.isNull() else None)
            except Exception as e: print(f"Error result content: {e}"); self.result_canvas.setImage(None)
        else: self.result_canvas.setImage(None)
    # --- End Update Helpers ---

    def update_display_content(self): # Identical
        self._update_spatial_canvas_content(self.current_spatial_image); self._update_freq_canvas_content(self.display_normalized_spectrum); self._update_result_canvas_content(self.resulting_spatial_image)
        self._update_phase_canvas_content(self.display_phase_vis_b, self.phase_b_canvas); self._update_phase_canvas_content(self.display_phase_vis_g, self.phase_g_canvas); self._update_phase_canvas_content(self.display_phase_vis_r, self.phase_r_canvas)

    # --- Live Drawing Handlers ---
    def handle_spatial_line_drawn(self, p1: QPoint, p2: QPoint, color: QColor): # Uses Bresenham
        if self.current_spatial_image is None: return
        try:
            thickness = max(1, int(round(self.spatial_canvas.pen_width / self.spatial_canvas.zoom_factor)))
            bgr_color = (color.blue(), color.green(), color.red()); h, w, _ = self.current_spatial_image.shape
            if thickness == 1:
                for x, y in bresenham_line(p1.x(), p1.y(), p2.x(), p2.y()):
                    if 0 <= y < h and 0 <= x < w: self.current_spatial_image[y, x] = bgr_color
            else: cv2.line(self.current_spatial_image, (p1.x(), p1.y()), (p2.x(), p2.y()), bgr_color, thickness, cv2.LINE_AA)
            self._update_spatial_canvas_content(self.current_spatial_image)
        except Exception as e: print(f"Error handling spatial line draw: {e}")

    def handle_spatial_point_drawn(self, p: QPoint, color: QColor): # Identical
        if self.current_spatial_image is None: return
        try: bgr_color = (color.blue(), color.green(), color.red()); h, w, _ = self.current_spatial_image.shape; x, y = p.x(), p.y()
        except Exception as e: print(f"Error handling spatial point draw: {e}")
        if 0 <= y < h and 0 <= x < w: self.current_spatial_image[y, x] = bgr_color; self._update_spatial_canvas_content(self.current_spatial_image)

    def handle_spatial_circle_drawn(self, p: QPoint, radius: int, color: QColor): # Identical
        if self.current_spatial_image is None: return
        try: bgr_color = (color.blue(), color.green(), color.red()); cv2.circle(self.current_spatial_image, (p.x(), p.y()), radius, bgr_color, thickness=-1, lineType=cv2.LINE_AA); self._update_spatial_canvas_content(self.current_spatial_image)
        except Exception as e: print(f"Error handling spatial circle draw: {e}")

    def handle_freq_line_drawn(self, p1: QPoint, p2: QPoint, color: QColor): # Uses Bresenham
        if self.display_normalized_spectrum is None: return
        try:
            thickness = max(1, int(round(self.freq_canvas.pen_width / self.freq_canvas.zoom_factor)))
            pen_r, pen_g, pen_b, _ = color.getRgb(); is_black_pen = pen_r < 5 and pen_g < 5 and pen_b < 5
            lock_b = self.lock_b_check.isChecked(); lock_g = self.lock_g_check.isChecked(); lock_r = self.lock_r_check.isChecked()
            h, w, _ = self.display_normalized_spectrum.shape
            if thickness == 1:
                for x, y in bresenham_line(p1.x(), p1.y(), p2.x(), p2.y()):
                    if 0 <= y < h and 0 <= x < w:
                        underlying_rgb = self.display_normalized_spectrum[y, x]; target_r, target_g, target_b = underlying_rgb[0], underlying_rgb[1], underlying_rgb[2]
                        if is_black_pen:
                            if not lock_r: target_r = 0;
                            if not lock_g: target_g = 0;
                            if not lock_b: target_b = 0;
                        else:
                            if not lock_r: target_r = pen_r;
                            if not lock_g: target_g = pen_g;
                            if not lock_b: target_b = pen_b;
                        self.display_normalized_spectrum[y, x] = [target_r, target_g, target_b]
            else:
                line_mask = np.zeros(self.display_normalized_spectrum.shape[:2], dtype=np.uint8); cv2.line(line_mask, (p1.x(), p1.y()), (p2.x(), p2.y()), 255, thickness, cv2.LINE_AA); line_indices = np.where(line_mask > 0)
                if line_indices[0].size > 0:
                    underlying_rgb = self.display_normalized_spectrum[line_indices]; target_rgb = np.copy(underlying_rgb)
                    if is_black_pen:
                        if not lock_r: target_rgb[:, 0] = 0;
                        if not lock_g: target_rgb[:, 1] = 0;
                        if not lock_b: target_rgb[:, 2] = 0;
                    else:
                        if not lock_r: target_rgb[:, 0] = pen_r;
                        if not lock_g: target_rgb[:, 1] = pen_g;
                        if not lock_b: target_rgb[:, 2] = pen_b;
                    self.display_normalized_spectrum[line_indices] = target_rgb
            self._update_freq_canvas_content(self.display_normalized_spectrum)
        except Exception as e: print(f"Error handling freq line draw: {e}")

    def handle_freq_point_drawn(self, p: QPoint, color: QColor): # Identical
        if self.display_normalized_spectrum is None: return
        try:
            pen_r, pen_g, pen_b, _ = color.getRgb(); is_black_pen = pen_r < 5 and pen_g < 5 and pen_b < 5
            lock_b = self.lock_b_check.isChecked(); lock_g = self.lock_g_check.isChecked(); lock_r = self.lock_r_check.isChecked()
            h, w, _ = self.display_normalized_spectrum.shape; x, y = p.x(), p.y()
            if 0 <= y < h and 0 <= x < w:
                underlying_rgb = self.display_normalized_spectrum[y, x]; target_r, target_g, target_b = underlying_rgb[0], underlying_rgb[1], underlying_rgb[2]
                if is_black_pen:
                    if not lock_r: target_r = 0;
                    if not lock_g: target_g = 0;
                    if not lock_b: target_b = 0;
                else:
                    if not lock_r: target_r = pen_r;
                    if not lock_g: target_g = pen_g;
                    if not lock_b: target_b = pen_b;
                self.display_normalized_spectrum[y, x] = [target_r, target_g, target_b]; self._update_freq_canvas_content(self.display_normalized_spectrum)
        except Exception as e: print(f"Error handling freq point draw: {e}")

    def handle_freq_circle_drawn(self, p: QPoint, radius: int, color: QColor): # Identical
         if self.display_normalized_spectrum is None: return
         try:
            pen_r, pen_g, pen_b, _ = color.getRgb(); is_black_pen = pen_r < 5 and pen_g < 5 and pen_b < 5
            lock_b = self.lock_b_check.isChecked(); lock_g = self.lock_g_check.isChecked(); lock_r = self.lock_r_check.isChecked()
            circle_mask = np.zeros(self.display_normalized_spectrum.shape[:2], dtype=np.uint8); cv2.circle(circle_mask, (p.x(), p.y()), radius, 255, thickness=-1, lineType=cv2.LINE_AA); circle_indices = np.where(circle_mask > 0)
            if circle_indices[0].size == 0: return
            underlying_rgb = self.display_normalized_spectrum[circle_indices]; target_rgb = np.copy(underlying_rgb)
            if is_black_pen:
                if not lock_r: target_rgb[:, 0] = 0;
                if not lock_g: target_rgb[:, 1] = 0;
                if not lock_b: target_rgb[:, 2] = 0;
            else:
                if not lock_r: target_rgb[:, 0] = pen_r;
                if not lock_g: target_rgb[:, 1] = pen_g;
                if not lock_b: target_rgb[:, 2] = pen_b;
            self.display_normalized_spectrum[circle_indices] = target_rgb; self._update_freq_canvas_content(self.display_normalized_spectrum)
         except Exception as e: print(f"Error handling freq circle draw: {e}")

    def handle_phase_line_drawn(self, p1: QPoint, p2: QPoint, color: QColor, canvas: DrawableCanvas, display_attr_name: str, channel_index_bgr: int): # Uses Bresenham
        display_array_bgr = getattr(self, display_attr_name, None); norm_mag_display_rgb = self.display_normalized_spectrum
        if display_array_bgr is None or norm_mag_display_rgb is None: return
        try:
            thickness = max(1, int(round(canvas.pen_width / canvas.zoom_factor)))
            pen_hsv = color.getHsv(); target_hue = int(round(pen_hsv[0] / 2));
            if target_hue > 179: target_hue = 179
            h, w, _ = display_array_bgr.shape
            map_bgr_to_rgb_idx = {0: 2, 1: 1, 2: 0}; rgb_idx = map_bgr_to_rgb_idx[channel_index_bgr]
            if thickness == 1:
                for x, y in bresenham_line(p1.x(), p1.y(), p2.x(), p2.y()):
                    if 0 <= y < h and 0 <= x < w:
                        current_value_unclamped = norm_mag_display_rgb[y, x, rgb_idx]; current_value = np.maximum(1, current_value_unclamped)
                        target_hsv = np.array([target_hue, 255, current_value], dtype=np.uint8).reshape(1, 1, 3)
                        target_bgr = cv2.cvtColor(target_hsv, cv2.COLOR_HSV2BGR)[0, 0]; display_array_bgr[y, x] = target_bgr
            else:
                line_mask = np.zeros(display_array_bgr.shape[:2], dtype=np.uint8); cv2.line(line_mask, (p1.x(), p1.y()), (p2.x(), p2.y()), 255, thickness, cv2.LINE_AA); line_indices = np.where(line_mask > 0)
                if line_indices[0].size > 0:
                    current_value_unclamped = norm_mag_display_rgb[line_indices[0], line_indices[1], rgb_idx]; current_value = np.maximum(1, current_value_unclamped)
                    target_h = np.full_like(current_value, target_hue, dtype=np.uint8); target_s = np.full_like(current_value, 255, dtype=np.uint8); target_v = current_value
                    target_hsv_pixels = cv2.merge([target_h, target_s, target_v])
                    if target_hsv_pixels.ndim == 3 and target_hsv_pixels.shape[1] == 1: target_hsv_pixels_img = target_hsv_pixels
                    elif target_hsv_pixels.ndim == 2 and target_hsv_pixels.shape[1] == 3: target_hsv_pixels_img = target_hsv_pixels[:, np.newaxis, :]
                    else: target_hsv_pixels_img = target_hsv_pixels.reshape(-1, 1, 3)
                    target_bgr_pixels = cv2.cvtColor(target_hsv_pixels_img, cv2.COLOR_HSV2BGR); target_bgr_pixels = target_bgr_pixels.reshape(-1, 3)
                    display_array_bgr[line_indices] = target_bgr_pixels
            self._update_phase_canvas_content(display_array_bgr, canvas)
        except Exception as e: print(f"Error handling phase line draw: {e}")

    def handle_phase_point_drawn(self, p: QPoint, color: QColor, canvas: DrawableCanvas, display_attr_name: str, channel_index_bgr: int): # Identical
        display_array_bgr = getattr(self, display_attr_name, None); norm_mag_display_rgb = self.display_normalized_spectrum
        if display_array_bgr is None or norm_mag_display_rgb is None: return
        try:
            pen_hsv = color.getHsv(); target_hue = int(round(pen_hsv[0] / 2));
            if target_hue > 179: target_hue = 179
            h, w, _ = display_array_bgr.shape; x, y = p.x(), p.y(); map_bgr_to_rgb_idx = {0: 2, 1: 1, 2: 0}; rgb_idx = map_bgr_to_rgb_idx[channel_index_bgr]
            if 0 <= y < h and 0 <= x < w:
                current_value_unclamped = norm_mag_display_rgb[y, x, rgb_idx]; current_value = np.maximum(1, current_value_unclamped)
                target_hsv = np.array([target_hue, 255, current_value], dtype=np.uint8).reshape(1, 1, 3)
                target_bgr = cv2.cvtColor(target_hsv, cv2.COLOR_HSV2BGR)[0, 0]; display_array_bgr[y, x] = target_bgr
                self._update_phase_canvas_content(display_array_bgr, canvas)
        except Exception as e: print(f"Error handling phase point draw: {e}")

    def handle_phase_circle_drawn(self, p: QPoint, radius: int, color: QColor, canvas: DrawableCanvas, display_attr_name: str, channel_index_bgr: int): # Identical
        display_array_bgr = getattr(self, display_attr_name, None); norm_mag_display_rgb = self.display_normalized_spectrum
        if display_array_bgr is None or norm_mag_display_rgb is None: return
        try:
            pen_hsv = color.getHsv(); target_hue = int(round(pen_hsv[0] / 2));
            if target_hue > 179: target_hue = 179
            map_bgr_to_rgb_idx = {0: 2, 1: 1, 2: 0}; rgb_idx = map_bgr_to_rgb_idx[channel_index_bgr]
            circle_mask = np.zeros(display_array_bgr.shape[:2], dtype=np.uint8); cv2.circle(circle_mask, (p.x(), p.y()), radius, 255, thickness=-1, lineType=cv2.LINE_AA); circle_indices = np.where(circle_mask > 0)
            if circle_indices[0].size == 0: return
            current_value_unclamped = norm_mag_display_rgb[circle_indices[0], circle_indices[1], rgb_idx]; current_value = np.maximum(1, current_value_unclamped)
            target_h = np.full_like(current_value, target_hue, dtype=np.uint8); target_s = np.full_like(current_value, 255, dtype=np.uint8); target_v = current_value
            target_hsv_pixels = cv2.merge([target_h, target_s, target_v])
            if target_hsv_pixels.ndim == 3 and target_hsv_pixels.shape[1] == 1: target_hsv_pixels_img = target_hsv_pixels
            elif target_hsv_pixels.ndim == 2 and target_hsv_pixels.shape[1] == 3: target_hsv_pixels_img = target_hsv_pixels[:, np.newaxis, :]
            else: target_hsv_pixels_img = target_hsv_pixels.reshape(-1, 1, 3)
            target_bgr_pixels = cv2.cvtColor(target_hsv_pixels_img, cv2.COLOR_HSV2BGR); target_bgr_pixels = target_bgr_pixels.reshape(-1, 3)
            display_array_bgr[circle_indices] = target_bgr_pixels; self._update_phase_canvas_content(display_array_bgr, canvas)
        except Exception as e: print(f"Error handling phase circle draw: {e}")
    # --- End Live Drawing Handlers ---

    # --- drawingFinished Handlers (process final state) ---
    def spatial_drawing_finished(self, drawn_bgr_data: np.ndarray): # Identical
        if self.current_spatial_image is not None:
            try:
                self.compute_frequency_domain()
                if self.frequency_image is None: raise ValueError("Freq computation failed.")
                self.resulting_spatial_image = self.inverse_transform(self.frequency_image)
                if self.resulting_spatial_image is None: raise ValueError("iFFT failed.")
                self.update_display_content()
                self._update_spatial_canvas_content(self.current_spatial_image)
                self.statusBar().showMessage("Spatial drawing processed.")
            except Exception as e: error_msg = f"Error processing spatial drawing: {e}"; print(error_msg); self.statusBar().showMessage(error_msg)
        else: print("Spatial drawing finished, but no data."); self.statusBar().showMessage("Error processing spatial drawing.")

    def freq_drawing_finished(self, drawn_freq_bgr_data: np.ndarray): # Identical
        drawn_freq_rgb_data = cv2.cvtColor(drawn_freq_bgr_data, cv2.COLOR_BGR2RGB)
        if drawn_freq_rgb_data is None or self.frequency_image is None or self.display_normalized_spectrum is None or not self.global_log_mag_range: self.statusBar().showMessage("Error: Missing data for frequency modification finalization."); return
        try:
            original_calculated_norm_spectrum_rgb = self.normalized_spectrum; final_display_norm_spectrum_rgb = self.display_normalized_spectrum
            if final_display_norm_spectrum_rgb is None or original_calculated_norm_spectrum_rgb is None: self.statusBar().showMessage("Error: Missing spectrum data for final processing."); return
            if final_display_norm_spectrum_rgb.shape != original_calculated_norm_spectrum_rgb.shape: print(f"Shape mismatch"); self.statusBar().showMessage("Freq shape error."); self._update_freq_canvas_content(original_calculated_norm_spectrum_rgb); return
            change_mask_rgb = np.any(np.abs(final_display_norm_spectrum_rgb.astype(np.int16) - original_calculated_norm_spectrum_rgb.astype(np.int16)) > 10, axis=2)
            if not np.any(change_mask_rgb): self.statusBar().showMessage("No final changes detected."); return
            original_complex_bgr = self.frequency_image; modified_complex_bgr = original_complex_bgr.copy()
            global_min_log_mag, global_max_log_mag = self.global_log_mag_range; global_log_mag_range_width = global_max_log_mag - global_min_log_mag
            if global_log_mag_range_width < 1e-9: print("Warning: Global log mag range zero."); self.statusBar().showMessage("Cannot modify: Log range zero."); return
            changed_indices = np.where(change_mask_rgb); map_rgb_to_bgr_idx = {0: 2, 1: 1, 2: 0}; black_tolerance = 5
            lock_state = {0: self.lock_b_check.isChecked(), 1: self.lock_g_check.isChecked(), 2: self.lock_r_check.isChecked()}
            for r, c in zip(*changed_indices):
                final_rgb_pixel = final_display_norm_spectrum_rgb[r, c, :]; is_black = np.all(final_rgb_pixel < black_tolerance)
                for c_rgb in range(3):
                    c_bgr = map_rgb_to_bgr_idx[c_rgb]
                    if lock_state[c_bgr]: continue
                    if is_black: modified_complex_bgr[r, c, c_bgr] = 0 + 0j
                    else:
                        final_channel_val_uint8 = final_rgb_pixel[c_rgb]
                        target_log_mag = (final_channel_val_uint8 / 255.0) * global_log_mag_range_width + global_min_log_mag
                        target_magnitude = max(0, np.expm1(target_log_mag))
                        original_phase = np.angle(original_complex_bgr[r, c, c_bgr])
                        modified_complex_bgr[r, c, c_bgr] = target_magnitude * np.exp(1j * original_phase)
            self.frequency_image = modified_complex_bgr
            self.resulting_spatial_image = self.inverse_transform(self.frequency_image)
            self._update_result_canvas_content(self.resulting_spatial_image)
            self._update_phase_displays_from_magnitude(self.frequency_image)
            self.statusBar().showMessage(f"Frequency magnitude edit finalized.")
        except Exception as e: error_msg = f"Error finalizing frequency drawing: {e}"; print(error_msg); self.statusBar().showMessage(error_msg)

    def phase_b_drawing_finished(self, drawn_bgr_data): self._process_phase_drawing_finished(drawn_bgr_data, 'display_phase_vis_b', 0, "Blue")
    def phase_g_drawing_finished(self, drawn_bgr_data): self._process_phase_drawing_finished(drawn_bgr_data, 'display_phase_vis_g', 1, "Green")
    def phase_r_drawing_finished(self, drawn_bgr_data): self._process_phase_drawing_finished(drawn_bgr_data, 'display_phase_vis_r', 2, "Red")

    def _process_phase_drawing_finished(self, drawn_phase_bgr_data: np.ndarray, display_attr_name: str, channel_index_bgr: int, channel_name: str): # Identical
        final_display_phase_vis = getattr(self, display_attr_name, None)
        if drawn_phase_bgr_data is None or self.frequency_image is None or final_display_phase_vis is None: self.statusBar().showMessage(f"Error: Missing data for {channel_name} phase finalization."); return
        try:
            original_calculated_phase_vis = getattr(self, f"phase_vis_{'bgr'[channel_index_bgr]}", None)
            if original_calculated_phase_vis is None: self.statusBar().showMessage(f"Error: Missing original phase data for {channel_name}."); return
            if drawn_phase_bgr_data.shape != original_calculated_phase_vis.shape: print(f"Shape mismatch"); self.statusBar().showMessage(f"{channel_name} phase shape error."); self._update_phase_canvas_content(original_calculated_phase_vis, getattr(self, f"phase_{'bgr'[channel_index_bgr]}_canvas")); return
            change_mask = np.any(np.abs(drawn_phase_bgr_data.astype(np.int16) - original_calculated_phase_vis.astype(np.int16)) > 10, axis=2)
            if not np.any(change_mask): self.statusBar().showMessage(f"No final changes detected in {channel_name} phase."); return
            original_complex_bgr = self.frequency_image; modified_complex_bgr = original_complex_bgr.copy()
            changed_indices = np.where(change_mask)
            for r, c in zip(*changed_indices):
                final_bgr_pixel = drawn_phase_bgr_data[r, c, :].reshape(1, 1, 3)
                hsv_pixel = cv2.cvtColor(final_bgr_pixel, cv2.COLOR_BGR2HSV)[0, 0]
                final_hue = hsv_pixel[0]; new_phase = ((final_hue / 179.0) * (2 * np.pi)) - np.pi
                original_magnitude = np.abs(original_complex_bgr[r, c, channel_index_bgr])
                new_complex_value = original_magnitude * np.exp(1j * new_phase)
                modified_complex_bgr[r, c, channel_index_bgr] = new_complex_value
            self.frequency_image = modified_complex_bgr
            self.resulting_spatial_image = self.inverse_transform(self.frequency_image)
            self._update_result_canvas_content(self.resulting_spatial_image)
            self.statusBar().showMessage(f"{channel_name} phase edit finalized.")
        except Exception as e: error_msg = f"Error finalizing {channel_name} phase drawing: {e}"; print(error_msg); self.statusBar().showMessage(error_msg)
    # --- End drawingFinished Handlers ---

    def _update_phase_displays_from_magnitude(self, complex_bgr_data): # Identical (with clamping)
        if complex_bgr_data is None or self.display_phase_vis_b is None or self.display_phase_vis_g is None or self.display_phase_vis_r is None or not self.global_log_mag_range: print("Skipping phase display update due to missing data."); return
        try:
            height, width, _ = complex_bgr_data.shape; epsilon = 1e-9; global_min_log_mag, global_max_log_mag = self.global_log_mag_range
            new_value_bgr = np.zeros((height, width, 3), dtype=np.uint8)
            for c_bgr in range(3): new_mag = np.abs(complex_bgr_data[:, :, c_bgr]); new_log_mag = np.log1p(new_mag + epsilon); new_value_bgr[:, :, c_bgr] = self.normalize_for_display(new_log_mag, global_min_log_mag, global_max_log_mag)
            for c_bgr, (display_attr, canvas) in enumerate([('display_phase_vis_b', self.phase_b_canvas), ('display_phase_vis_g', self.phase_g_canvas), ('display_phase_vis_r', self.phase_r_canvas)]):
                current_display_bgr = getattr(self, display_attr, None)
                if current_display_bgr is None: continue
                try: current_hsv = cv2.cvtColor(current_display_bgr, cv2.COLOR_BGR2HSV); current_hue = current_hsv[:, :, 0]; current_sat = current_hsv[:, :, 1]
                except cv2.error as e: print(f"Warning: cvtColor failed for {display_attr}, possibly empty? Error: {e}"); current_hue = np.zeros(new_value_bgr.shape[:2], dtype=np.uint8); current_sat = np.full(new_value_bgr.shape[:2], 255, dtype=np.uint8)
                new_value_unclamped = new_value_bgr[:, :, c_bgr]; new_value = np.maximum(1, new_value_unclamped) # Clamp Value
                new_hsv = cv2.merge([current_hue, current_sat, new_value]); new_bgr = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2BGR)
                setattr(self, display_attr, new_bgr); self._update_phase_canvas_content(new_bgr, canvas)
            print("Phase displays updated based on new magnitudes.")
        except Exception as e: print(f"Error updating phase displays from magnitude: {e}")

    def apply_result_to_original(self): # Identical
        if self.resulting_spatial_image is not None:
            self.current_spatial_image = self.resulting_spatial_image.copy()
            try:
                self.compute_frequency_domain()
                if self.frequency_image is None: raise ValueError("Freq computation failed after apply.")
                self.resulting_spatial_image = self.inverse_transform(self.frequency_image)
                self.update_display_content()
                self.statusBar().showMessage("Result applied to original spatial image.")
            except Exception as e: error_msg = f"Error applying result: {e}"; print(error_msg); self.statusBar().showMessage(error_msg)
        else: self.statusBar().showMessage("No result image available to apply.")

    def inverse_transform(self, freq_data_bgr_complex: np.ndarray) -> np.ndarray | None: # Identical
        if freq_data_bgr_complex is None or len(freq_data_bgr_complex.shape) != 3: return None
        try: height, width, _ = freq_data_bgr_complex.shape; img_back_bgr_float = np.zeros((height, width, 3), dtype=np.float64)
        except Exception as e: print(f"Error during inverse FFT (Color): {e}"); return None
        for c in range(3): f_ishifted_channel = np.fft.ifftshift(freq_data_bgr_complex[:, :, c]); img_back_complex_channel = np.fft.ifft2(f_ishifted_channel); img_back_bgr_float[:, :, c] = np.abs(img_back_complex_channel)
        np.clip(img_back_bgr_float, 0, 255, out=img_back_bgr_float); img_back_bgr_uint8 = img_back_bgr_float.astype(np.uint8)
        return img_back_bgr_uint8

    # --- Filter Prompt Methods (Identical) ---
    def prompt_radial_filter(self):
        if self.frequency_image is None: self.statusBar().showMessage("Load an image first!"); return
        percent, ok = QInputDialog.getDouble(self, "Radial Low-Pass Filter", "Keep Radius (% of max):", 30.0, 0.1, 100.0, 1)
        if ok: self.apply_radial_filter(percent / 100.0)
    def prompt_magnitude_filter(self):
        if self.frequency_image is None: self.statusBar().showMessage("Load an image first!"); return
        percent, ok = QInputDialog.getDouble(self, "Log Magnitude Filter", "Keep Log Magnitude Above (% of range):", 5.0, 0.0, 100.0, 1)
        if ok: self.apply_magnitude_filter(percent / 100.0)
    def prompt_gaussian_filter(self):
        if self.frequency_image is None: self.statusBar().showMessage("Load an image first!"); return
        percent, ok = QInputDialog.getDouble(self, "Gaussian Low-Pass Filter", "Cutoff Sigma (% of max radius):", 10.0, 0.1, 100.0, 1)
        if ok: self.apply_gaussian_filter(percent / 100.0)
    # --- End Filter Prompt Methods ---

    # --- Filter Apply Methods (Identical) ---
    def apply_radial_filter(self, radius_fraction: float):
        if self.frequency_image is None: return
        try:
            h, w, _ = self.frequency_image.shape; center_row, center_col = h // 2, w // 2; max_radius = np.sqrt((h//2)**2 + (w//2)**2); radius_pixels = radius_fraction * max_radius
            y, x = np.ogrid[:h, :w]; dist_from_center = np.sqrt((y - center_row)**2 + (x - center_col)**2); mask = dist_from_center <= radius_pixels
            modified_frequency_image = self.frequency_image.copy() # Work on copy
            for c in range(3): modified_frequency_image[~mask, c] = 0 + 0j
            self.frequency_image = modified_frequency_image # Commit
            self._post_filter_update(); self.statusBar().showMessage(f"Applied Radial Filter (Radius: {radius_fraction*100:.1f}%)")
        except Exception as e: self.statusBar().showMessage(f"Error applying radial filter: {e}"); print(e)
    def apply_magnitude_filter(self, threshold_fraction: float): # Uses Log Mag
        if self.frequency_image is None or self.magnitude_spectrum is None: self.statusBar().showMessage("Frequency data not available."); return
        try:
            log_magnitudes = self.magnitude_spectrum; min_log_mag, max_log_mag = self.global_log_mag_range; log_mag_range = max_log_mag - min_log_mag
            if log_mag_range < 1e-9: self.statusBar().showMessage("Log magnitude range is zero."); return
            threshold_log_value = min_log_mag + (threshold_fraction * log_mag_range)
            mask_below_threshold = log_magnitudes < threshold_log_value
            modified_frequency_image = self.frequency_image.copy()
            for c in range(3): modified_frequency_image[mask_below_threshold[:,:,c], c] = 0 + 0j
            self.frequency_image = modified_frequency_image
            self._post_filter_update(); self.statusBar().showMessage(f"Applied Log Magnitude Filter (Keep above {threshold_fraction*100:.1f}%)")
        except Exception as e: self.statusBar().showMessage(f"Error applying log magnitude filter: {e}"); print(e)
    def apply_gaussian_filter(self, sigma_fraction: float):
        if self.frequency_image is None: return
        try:
            h, w, _ = self.frequency_image.shape; center_row, center_col = h // 2, w // 2; max_radius = np.sqrt((h//2)**2 + (w//2)**2); sigma_pixels = sigma_fraction * max_radius
            if sigma_pixels < 1e-6: self.statusBar().showMessage("Sigma too small."); return
            y, x = np.ogrid[:h, :w]; dist_sq = (y - center_row)**2 + (x - center_col)**2; gaussian_filter = np.exp(-dist_sq / (2 * sigma_pixels**2))
            modified_frequency_image = self.frequency_image.copy()
            for c in range(3): modified_frequency_image[:, :, c] *= gaussian_filter
            self.frequency_image = modified_frequency_image
            self._post_filter_update(); self.statusBar().showMessage(f"Applied Gaussian Filter (Sigma: {sigma_fraction*100:.1f}%)")
        except Exception as e: self.statusBar().showMessage(f"Error applying Gaussian filter: {e}"); print(e)
    # --- End Filter Apply Methods ---

    def _post_filter_update(self): # Identical
        if self.frequency_image is None: return
        try:
            self.resulting_spatial_image = self.inverse_transform(self.frequency_image)
            self._compute_visualizations_from_frequency_data() # Recalculate visualizations AND display states
            self.update_display_content() # Update all canvases from display states
        except Exception as e: print(f"Error during post-filter update: {e}"); self.statusBar().showMessage("Error updating displays after filter.")

    def _compute_visualizations_from_frequency_data(self): # Identical (recomputes everything from self.frequency_image)
        if self.frequency_image is None: return
        try:
            height, width, _ = self.frequency_image.shape; self.magnitude_spectrum = np.zeros((height, width, 3), dtype=np.float64); epsilon = 1e-9
            for c in range(3): magnitude_channel = np.abs(self.frequency_image[:, :, c]); log_mag_channel = np.log1p(magnitude_channel + epsilon); self.magnitude_spectrum[:, :, c] = log_mag_channel
            global_min_log_mag = np.min(self.magnitude_spectrum); global_max_log_mag = np.max(self.magnitude_spectrum)
            if global_max_log_mag - global_min_log_mag < epsilon: global_max_log_mag = global_min_log_mag + 1.0
            self.global_log_mag_range = (global_min_log_mag, global_max_log_mag)
            self.normalized_spectrum = np.zeros((height, width, 3), dtype=np.uint8); map_bgr_to_rgb_idx = {0: 2, 1: 1, 2: 0}; phase_vis_storage = {0: None, 1: None, 2: None}
            for c_bgr in range(3):
                norm_log_mag_uint8 = self.normalize_for_display(self.magnitude_spectrum[:, :, c_bgr], global_min_log_mag, global_max_log_mag)
                c_rgb = map_bgr_to_rgb_idx[c_bgr]; self.normalized_spectrum[:, :, c_rgb] = norm_log_mag_uint8
                phase = np.angle(self.frequency_image[:, :, c_bgr]); hue = ((phase + np.pi) / (2 * np.pi)) * 179.0; hue = hue.astype(np.uint8)
                saturation = np.full_like(hue, 255, dtype=np.uint8); value = np.maximum(1, norm_log_mag_uint8) # Clamp Value
                hsv_vis = cv2.merge([hue, saturation, value]); bgr_vis = cv2.cvtColor(hsv_vis, cv2.COLOR_HSV2BGR); phase_vis_storage[c_bgr] = bgr_vis
            self.phase_vis_b = phase_vis_storage[0]; self.phase_vis_g = phase_vis_storage[1]; self.phase_vis_r = phase_vis_storage[2]
            self.display_normalized_spectrum = self.normalized_spectrum.copy() if self.normalized_spectrum is not None else None; self.display_phase_vis_b = self.phase_vis_b.copy() if self.phase_vis_b is not None else None; self.display_phase_vis_g = self.phase_vis_g.copy() if self.phase_vis_g is not None else None; self.display_phase_vis_r = self.phase_vis_r.copy() if self.phase_vis_r is not None else None
        except Exception as e: print(f"Error recomputing visualizations: {e}")

    def open_random_image_dialog(self): # Identical
        dialog = RandomImageDialog(self)
        if dialog.exec_() == QDialog.Accepted: width, height = dialog.get_dimensions(); grayscale_request = dialog.get_grayscale(); blur_amount = dialog.get_blur(); source_type = dialog.get_source_type(); specific_id = dialog.get_specific_id() if source_type == 'id' else None; seed_value = dialog.get_seed_value() if source_type == 'seed' else None; self.load_random_image(width, height, grayscale_request, blur_amount, specific_id, seed_value)

    def load_random_image(self, width=400, height=300, grayscale_request=False, blur=0, specific_id=None, seed_value=None): # Identical
        try:
            self.statusBar().showMessage("Loading random image..."); QApplication.processEvents(); base_url = "https://picsum.photos/"; url_parts = [base_url];
            if specific_id: url_parts.append(f"id/{specific_id}/")
            elif seed_value: url_parts.append(f"seed/{seed_value}/")
            url_parts.append(f"{width}/{height}"); url = "".join(url_parts); params = {};
            if grayscale_request: params["grayscale"] = None
            if blur > 0: params["blur"] = blur
            print(f"Requesting Random Image URL: {url} with params {params}"); response = requests.get(url, params=params, timeout=20); response.raise_for_status(); img_array = np.frombuffer(response.content, np.uint8); img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_bgr is not None and len(img_bgr.shape) == 3: self.original_image = img_bgr.copy(); self.current_spatial_image = img_bgr.copy(); self.process_and_display(); self.statusBar().showMessage(f"Loaded random COLOR image ({width}x{height})"); self.update_tool_state(True); self.pencil_action.setChecked(True); self.toggle_drawing_mode(True)
            elif img_bgr is not None and len(img_bgr.shape) == 2: print("Info: Received grayscale, converting to color."); self.original_image = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR); self.current_spatial_image = self.original_image.copy(); self.process_and_display(); self.statusBar().showMessage(f"Loaded random image (processed as color) ({width}x{height})"); self.update_tool_state(True); self.pencil_action.setChecked(True); self.toggle_drawing_mode(True)
            else: raise ValueError("Failed to decode downloaded image.")
        except requests.exceptions.Timeout: error_msg = "Error: Timeout loading random image."; print(error_msg); self.statusBar().showMessage(error_msg); self.update_tool_state(False)
        except requests.exceptions.RequestException as e: error_msg = f"Network error: {e}"; print(error_msg); self.statusBar().showMessage(error_msg); self.update_tool_state(False)
        except Exception as e: error_msg = f"Error loading/processing random image: {e}"; print(error_msg); self.statusBar().showMessage(error_msg); self.update_tool_state(False)


class RandomImageDialog(QDialog): # Identical
    def __init__(self, parent=None):
        super().__init__(parent); self.setWindowTitle("Random Image Options"); self.setMinimumWidth(350); layout = QVBoxLayout()
        size_group = QGroupBox("Image Size"); size_layout = QFormLayout(); self.width_spin = QSpinBox(); self.width_spin.setRange(50, 2000); self.width_spin.setValue(400); self.width_spin.setSuffix(" px"); size_layout.addRow("Width:", self.width_spin); self.height_spin = QSpinBox(); self.height_spin.setRange(50, 2000); self.height_spin.setValue(300); self.height_spin.setSuffix(" px"); size_layout.addRow("Height:", self.height_spin); size_group.setLayout(size_layout); layout.addWidget(size_group)
        options_group = QGroupBox("Image Options"); options_layout = QFormLayout(); self.grayscale_check = QCheckBox(); self.grayscale_check.setChecked(False); self.grayscale_check.setToolTip("Request grayscale image (App processes color internally)"); options_layout.addRow("Request Grayscale:", self.grayscale_check); self.blur_spin = QSpinBox(); self.blur_spin.setRange(0, 10); self.blur_spin.setValue(0); options_layout.addRow("Blur (0-10):", self.blur_spin); options_group.setLayout(options_layout); layout.addWidget(options_group)
        source_group = QGroupBox("Image Source (Lorem Picsum)"); source_outer_layout = QVBoxLayout(); self.random_radio = QRadioButton("Completely Random"); self.id_radio = QRadioButton("Specific ID"); self.seed_radio = QRadioButton("Seed"); self.random_radio.setChecked(True); source_outer_layout.addWidget(self.random_radio); source_outer_layout.addWidget(self.id_radio); source_outer_layout.addWidget(self.seed_radio); source_form_layout = QFormLayout(); self.id_input = QComboBox(); self.id_input.setEditable(True); self.id_input.addItems(["", "0", "10", "100", "237", "1084", "870"]); self.id_input.setEnabled(False); source_form_layout.addRow("Specific ID:", self.id_input); self.seed_input = QComboBox(); self.seed_input.setEditable(True); self.seed_input.addItems(["picsum", "random", "nature", "technology", "people", ""]); self.seed_input.setEnabled(False); source_form_layout.addRow("Seed Value:", self.seed_input); source_outer_layout.addLayout(source_form_layout); source_group.setLayout(source_outer_layout); layout.addWidget(source_group); self.random_radio.toggled.connect(self.update_source_inputs); self.id_radio.toggled.connect(self.update_source_inputs); self.seed_radio.toggled.connect(self.update_source_inputs)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); button_box.accepted.connect(self.accept); button_box.rejected.connect(self.reject); layout.addWidget(button_box); self.setLayout(layout); self.adjustSize()
    def update_source_inputs(self): is_id = self.id_radio.isChecked(); is_seed = self.seed_radio.isChecked(); self.id_input.setEnabled(is_id); self.seed_input.setEnabled(is_seed); (is_id and self.seed_input.setCurrentIndex(self.seed_input.findText(""))); (is_seed and self.id_input.setCurrentIndex(self.id_input.findText("")))
    def get_dimensions(self): return self.width_spin.value(), self.height_spin.value()
    def get_grayscale(self): return self.grayscale_check.isChecked()
    def get_blur(self): return self.blur_spin.value()
    def get_source_type(self): return 'id' if self.id_radio.isChecked() else 'seed' if self.seed_radio.isChecked() else 'random'
    def get_specific_id(self): return self.id_input.currentText().strip() if self.id_radio.isChecked() else None
    def get_seed_value(self): return self.seed_input.currentText().strip() if self.seed_radio.isChecked() else None


if __name__ == "__main__":
    if hasattr(Qt, 'AA_EnableHighDpiScaling'): QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'): QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    window = FourierTransformApp()
    window.show()
    sys.exit(app.exec_())
