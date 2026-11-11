# -*- coding: utf-8 -*-
"""可复用界面组件：参数对话框、类别选择及标注画布。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QAbstractSpinBox, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QInputDialog, QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from .config import SCRIPT_DIR
from .data import parse_multiscale_sizes

BOX_COLORS = [
    QColor(0, 255, 0), QColor(255, 90, 90), QColor(80, 160, 255),
    QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255),
]
SECONDARY_BOX_COLORS = [
    QColor(255, 128, 0), QColor(128, 255, 0), QColor(0, 255, 128),
    QColor(0, 0, 255), QColor(128, 0, 255), QColor(255, 0, 128),
]

class BatchSamplingDialog(QDialog):
    """批量预标注参数；范围与采样策略仅在启动批量任务时配置。"""

    def __init__(
        self,
        start_frame: int,
        end_frame: int,
        total_frames: int,
        frame_step: int = 30,
        smart_enabled: bool = False,
        overlap_threshold: float = 70.0,
        orb_nfeatures: int = 2000,
        orb_max_side: int = 640,
        smart_base_step: int = 5,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("批量 AI 预标注")
        self.setModal(True)
        self.setMinimumWidth(400)

        self.total_frames = max(1, int(total_frames))
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(1, self.total_frames)
        self.start_frame_spin.setValue(min(max(1, int(start_frame)), self.total_frames))
        self.start_frame_spin.setToolTip("批量处理起始帧，按 1 开始计数。")

        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(1, self.total_frames)
        self.end_frame_spin.setValue(min(max(1, int(end_frame)), self.total_frames))
        self.end_frame_spin.setToolTip("批量处理结束帧，包含该帧。")

        self.range_summary = QLabel()
        self.range_summary.setStyleSheet("color: palette(mid);")
        self.start_frame_spin.valueChanged.connect(self._update_range_summary)
        self.end_frame_spin.valueChanged.connect(self._update_range_summary)

        range_form = QFormLayout()
        range_form.addRow("起始帧", self.start_frame_spin)
        range_form.addRow("结束帧（包含）", self.end_frame_spin)
        range_form.addRow(self.range_summary)
        self.range_group = QGroupBox("处理范围")
        self.range_group.setLayout(range_form)

        self.frame_step_spin = QSpinBox()
        self.frame_step_spin.setRange(1, 100000)
        self.frame_step_spin.setSingleStep(1)
        self.frame_step_spin.setValue(min(100000, max(1, int(frame_step))))
        self.frame_step_spin.setToolTip(
            "固定模式下作为采样间隔；智能模式下，作为配准失败后的兜底帧数。"
        )
        self.frame_step_label = QLabel()

        self.smart_overlap_check = QCheckBox("智能重叠率")
        self.smart_overlap_check.setChecked(bool(smart_enabled))

        self.overlap_threshold_spin = QDoubleSpinBox()
        self.overlap_threshold_spin.setRange(5.0, 99.0)
        self.overlap_threshold_spin.setDecimals(0)
        self.overlap_threshold_spin.setSingleStep(5.0)
        self.overlap_threshold_spin.setSuffix(" %")
        self.overlap_threshold_spin.setValue(min(99.0, max(5.0, float(overlap_threshold))))

        self.orb_nfeatures_spin = QSpinBox()
        self.orb_nfeatures_spin.setRange(100, 10000)
        self.orb_nfeatures_spin.setSingleStep(100)
        self.orb_nfeatures_spin.setValue(min(10000, max(100, int(orb_nfeatures))))
        self.orb_nfeatures_spin.setToolTip("每次检测最多提取的 ORB 特征点数量。")

        self.orb_max_side_spin = QSpinBox()
        self.orb_max_side_spin.setRange(320, 1920)
        self.orb_max_side_spin.setSingleStep(64)
        self.orb_max_side_spin.setSuffix(" px")
        self.orb_max_side_spin.setValue(min(1920, max(320, int(orb_max_side))))
        self.orb_max_side_spin.setToolTip("ORB 分析图的最长边尺寸。")

        self.smart_base_step_spin = QSpinBox()
        self.smart_base_step_spin.setRange(1, 30)
        self.smart_base_step_spin.setSingleStep(1)
        self.smart_base_step_spin.setSuffix(" 帧")
        self.smart_base_step_spin.setValue(min(30, max(1, int(smart_base_step))))
        self.smart_base_step_spin.setToolTip("自适应 ORB 检测间隔的基准值。")

        smart_form = QFormLayout()
        smart_form.addRow("重叠率阈值", self.overlap_threshold_spin)
        smart_form.addRow("ORB 特征点数", self.orb_nfeatures_spin)
        smart_form.addRow("分析图最长边", self.orb_max_side_spin)
        smart_form.addRow("检测步长基准", self.smart_base_step_spin)
        self.smart_options_group = QGroupBox("智能采样参数")
        self.smart_options_group.setLayout(smart_form)

        self.mode_hint = QLabel("低于阈值时标注。")
        self.cache_hint = QLabel("结果需手动保存。")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("开始批量预标注")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.range_group)
        layout.addWidget(self.smart_overlap_check)
        step_form = QFormLayout()
        step_form.addRow(self.frame_step_label, self.frame_step_spin)
        layout.addLayout(step_form)
        layout.addWidget(self.smart_options_group)
        layout.addWidget(self.mode_hint)
        layout.addWidget(self.cache_hint)
        layout.addWidget(buttons)
        self.smart_overlap_check.toggled.connect(self._update_sampling_controls)
        self._update_sampling_controls(bool(smart_enabled))
        self._update_range_summary()

    def _update_range_summary(self, *_args) -> None:
        start = self.start_frame_spin.value()
        end = self.end_frame_spin.value()
        count = max(0, end - start + 1)
        self.range_summary.setText(f"共 {count} / {self.total_frames} 帧")

    def _update_sampling_controls(self, smart_enabled: bool) -> None:
        enabled = bool(smart_enabled)
        self.frame_step_label.setText(
            "失败兜底间隔（帧）" if enabled else "固定采样间隔（帧）"
        )
        self.smart_options_group.setEnabled(enabled)
        self.mode_hint.setText("低于阈值时标注。" if enabled else "按固定间隔标注。")

    def accept(self) -> None:
        start = self.start_frame_spin.value()
        end = self.end_frame_spin.value()
        if not (1 <= start <= end <= self.total_frames):
            QMessageBox.warning(
                self, "批量处理范围无效", "结束帧不能早于起始帧或超过视频总帧数。"
            )
            self.end_frame_spin.setFocus()
            return
        super().accept()

    def values(self) -> dict:
        return {
            "smart_enabled": self.smart_overlap_check.isChecked(),
            "overlap_threshold": self.overlap_threshold_spin.value(),
            "start_frame": self.start_frame_spin.value(),
            "end_frame": self.end_frame_spin.value(),
            "frame_step": self.frame_step_spin.value(),
            "orb_nfeatures": self.orb_nfeatures_spin.value(),
            "orb_max_side": self.orb_max_side_spin.value(),
            "smart_base_step": self.smart_base_step_spin.value(),
        }


class LabelChooserDialog(QDialog):
    """选择已有类别或新增类别。类别采用追加方式，避免改变已有 YOLO 类别 ID。"""

    def __init__(
        self,
        classes: list[str],
        current_index: int,
        usage_counts: dict[str, int],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("选择目标类别")
        self.setModal(True)
        self.setMinimumWidth(360)
        self._classes = list(classes)
        self._usage = usage_counts
        self._current_index = (
            current_index if 0 <= current_index < len(self._classes) else 0
        )

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(lambda _item: self.accept())

        add_button = QPushButton("新增类别")
        add_button.clicked.connect(self._add_label)

        button_row = QHBoxLayout()
        button_row.addWidget(add_button)
        button_row.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.Ok).setText("确认（F）")

        layout = QVBoxLayout(self)
        layout.addWidget(self.list_widget, 1)
        layout.addLayout(button_row)
        layout.addWidget(buttons)
        self._refresh_list()

    def _refresh_list(self) -> None:
        self.list_widget.clear()
        for index, name in enumerate(self._classes):
            used = self._usage.get(name, 0)
            suffix = f"（已有 {used} 个标注框）" if used else ""
            self.list_widget.addItem(f"{index + 1}. {name}{suffix}")
        if self._classes:
            self.list_widget.setCurrentRow(
                min(self._current_index, len(self._classes) - 1)
            )

    def _add_label(self) -> None:
        name, ok = QInputDialog.getText(self, "新增类别", "类别名称：")
        name = name.strip()
        if not ok or not name:
            return
        if name not in self._classes:
            self._classes.append(name)
        self._current_index = self._classes.index(name)
        self._refresh_list()

    def accept(self) -> None:
        if not self._classes:
            QMessageBox.information(self, "没有类别", "请先添加类别。")
            return
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self._classes):
            QMessageBox.information(self, "未选择类别", "请先选择一个类别。")
            return
        self._current_index = row
        super().accept()

    def selected_name(self) -> str:
        return self._classes[self._current_index]

    def classes_result(self) -> list[str]:
        return list(self._classes)

    def keyPressEvent(self, event) -> None:
        if Qt.Key_1 <= event.key() <= Qt.Key_9:
            row = event.key() - Qt.Key_1
            if row < len(self._classes):
                self.list_widget.setCurrentRow(row)
                self.accept()
                return
        if event.key() == Qt.Key_F:
            self.accept()
            return
        super().keyPressEvent(event)


class ModelSettingsDialog(QDialog):
    """应用程序设置；参数由主窗口持久化并应用于后续任务。"""

    def __init__(
        self,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        multiscale_enabled: bool,
        multiscale_sizes: str,
        show_validation_imgsz: bool,
        min_box_side: int,
        manual_replace_overlap_enabled: bool,
        manual_replace_overlap_iou: float,
        camera_device_index: int,
        secondary_model_path: str,
        validation_model_mode: str,
        validation_overlap_iou: float,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setModal(True)
        self.setMinimumWidth(520)

        self.camera_device_combo = QComboBox()
        for device_index in range(10):
            label = f"摄像头 {device_index}（默认）" if device_index == 0 else f"摄像头 {device_index}"
            self.camera_device_combo.addItem(label, device_index)
        selected_camera_row = self.camera_device_combo.findData(int(camera_device_index))
        self.camera_device_combo.setCurrentIndex(max(0, selected_camera_row))
        self.camera_device_combo.setToolTip("选择摄像头设备。")
        camera_form = QFormLayout()
        camera_form.addRow("摄像头设备", self.camera_device_combo)
        self.camera_group = QGroupBox("摄像头")
        self.camera_group.setLayout(camera_form)

        self.secondary_model_edit = QLineEdit(str(secondary_model_path or ""))
        self.secondary_model_edit.setClearButtonEnabled(True)
        self.secondary_model_edit.setPlaceholderText("可选：选择第二个 YOLO .pt 权重文件")
        self.secondary_model_edit.setToolTip("选择用于实时验证的副模型。")
        secondary_browse_button = QPushButton("浏览…")
        secondary_browse_button.clicked.connect(self._browse_secondary_model)
        secondary_path_row = QWidget()
        secondary_path_layout = QHBoxLayout(secondary_path_row)
        secondary_path_layout.setContentsMargins(0, 0, 0, 0)
        secondary_path_layout.addWidget(self.secondary_model_edit, 1)
        secondary_path_layout.addWidget(secondary_browse_button)

        self.validation_model_mode_combo = QComboBox()
        self.validation_model_mode_combo.addItem("全量", "all")
        self.validation_model_mode_combo.addItem("叠加", "overlap")
        mode_row = self.validation_model_mode_combo.findData(str(validation_model_mode))
        self.validation_model_mode_combo.setCurrentIndex(max(0, mode_row))
        self.validation_model_mode_combo.setToolTip(
            "全量显示两个模型的检测框；叠加仅显示与主模型重叠的副模型检测框。"
        )

        self.validation_overlap_iou_spin = QDoubleSpinBox()
        self.validation_overlap_iou_spin.setRange(0.01, 1.00)
        self.validation_overlap_iou_spin.setDecimals(2)
        self.validation_overlap_iou_spin.setSingleStep(0.05)
        self.validation_overlap_iou_spin.setValue(validation_overlap_iou)
        self.validation_overlap_iou_spin.setToolTip("副模型框与主模型框的 IoU 达到此值时显示。")

        dual_model_form = QFormLayout()
        dual_model_form.addRow("副模型权重", secondary_path_row)
        dual_model_form.addRow("显示方式", self.validation_model_mode_combo)
        dual_model_form.addRow("重叠 IoU 阈值", self.validation_overlap_iou_spin)
        self.dual_model_group = QGroupBox("实时验证双模型")
        self.dual_model_group.setLayout(dual_model_form)
        self.secondary_model_edit.textChanged.connect(self._update_dual_model_controls)
        self.validation_model_mode_combo.currentIndexChanged.connect(
            self._update_dual_model_controls
        )
        self._update_dual_model_controls()

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.00)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(conf)
        self.conf_spin.setToolTip("低于此置信度的检测结果将被过滤。")

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 0.99)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(iou)
        self.iou_spin.setToolTip(
            "用于推理阶段的 NMS；多分辨率模式下也用于跨尺度去重。"
        )

        # 两种分辨率模式互斥：多分辨率模式启用时，单尺度 imgsz 不参与推理。
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 8192)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(imgsz)
        self.imgsz_spin.setToolTip(
            "单尺度推理尺寸；启用多分辨率后此项停用。"
        )
        self.imgsz_label = QLabel("单尺度输入尺寸（imgsz）")

        self.multiscale_check = QCheckBox("启用多分辨率推理与结果融合")
        self.multiscale_check.setChecked(multiscale_enabled)
        self.multiscale_check.setToolTip(
            "按列表中的多个尺寸分别推理，再融合检测结果。"
        )
        self.multiscale_sizes_edit = QLineEdit(multiscale_sizes)
        self.multiscale_sizes_edit.setPlaceholderText("例如：640, 1280, 1920")
        self.multiscale_sizes_edit.setToolTip(
            "用英文逗号分隔，至少填写两个不同尺寸（320–8192）。"
        )
        self.multiscale_sizes_label = QLabel("多分辨率尺寸列表")
        self.show_validation_imgsz_check = QCheckBox("实时验证框显示对应推理分辨率")
        self.show_validation_imgsz_check.setChecked(show_validation_imgsz)
        self.show_validation_imgsz_check.setToolTip(
            "仅在多分辨率融合模式下生效；在实时验证框标签中显示该检测框对应的来源分辨率数值。"
        )
        self.multiscale_check.toggled.connect(self._update_resolution_controls)

        resolution_form = QFormLayout()
        resolution_form.addRow(self.imgsz_label, self.imgsz_spin)
        resolution_form.addRow(self.multiscale_check)
        resolution_form.addRow(self.multiscale_sizes_label, self.multiscale_sizes_edit)
        resolution_form.addRow(self.show_validation_imgsz_check)
        self.resolution_group = QGroupBox("推理输入分辨率")
        self.resolution_group.setLayout(resolution_form)
        self._update_resolution_controls(multiscale_enabled)

        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 3000)
        self.max_det_spin.setSingleStep(50)
        self.max_det_spin.setValue(max_det)
        self.max_det_spin.setToolTip("每帧最多保留的检测框数量。")

        self.min_box_side_spin = QSpinBox()
        self.min_box_side_spin.setRange(0, 8192)
        self.min_box_side_spin.setSingleStep(5)
        self.min_box_side_spin.setSuffix(" px")
        self.min_box_side_spin.setValue(min_box_side)
        self.min_box_side_spin.setToolTip(
            "过滤宽或高小于此值的检测框；0 表示关闭。"
        )

        self.manual_replace_overlap_check = QCheckBox("标注智能修正功能：阈值")
        self.manual_replace_overlap_check.setChecked(manual_replace_overlap_enabled)
        self.manual_replace_overlap_iou_spin = QDoubleSpinBox()
        self.manual_replace_overlap_iou_spin.setRange(0.01, 1.00)
        self.manual_replace_overlap_iou_spin.setDecimals(2)
        self.manual_replace_overlap_iou_spin.setSingleStep(0.05)
        self.manual_replace_overlap_iou_spin.setValue(manual_replace_overlap_iou)
        self.manual_replace_overlap_iou_spin.setToolTip(
            "绘制新框时，自动移除与新框 IoU 达到此阈值的现有标注框，不区分类别。"
        )
        self.manual_replace_overlap_iou_spin.setEnabled(manual_replace_overlap_enabled)
        self.manual_replace_overlap_check.toggled.connect(
            self.manual_replace_overlap_iou_spin.setEnabled
        )


        form = QFormLayout()
        form.addRow("置信度阈值（conf）", self.conf_spin)
        form.addRow("NMS IoU 阈值", self.iou_spin)
        form.addRow(self.resolution_group)
        form.addRow("每帧最大检测数", self.max_det_spin)
        form.addRow("最小目标边长", self.min_box_side_spin)
        form.addRow(self.manual_replace_overlap_check, self.manual_replace_overlap_iou_spin)
        # 摄像头和副模型属于低频使用的扩展功能，放在设置末尾。
        form.addRow(self.camera_group)
        form.addRow(self.dual_model_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("应用")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        form_content = QWidget()
        form_content.setLayout(form)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(440)
        scroll_area.setMaximumHeight(680)
        scroll_area.setWidget(form_content)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll_area, 1)
        layout.addWidget(buttons)

    def _browse_secondary_model(self) -> None:
        current_path = self.secondary_model_edit.text().strip()
        start_dir = str(Path(current_path).expanduser().parent) if current_path else str(SCRIPT_DIR)
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择副模型 YOLO 权重", start_dir, "YOLO 权重 (*.pt);;所有文件 (*)"
        )
        if filename:
            self.secondary_model_edit.setText(filename)

    def _update_dual_model_controls(self, *_args) -> None:
        enabled = bool(self.secondary_model_edit.text().strip())
        self.validation_model_mode_combo.setEnabled(enabled)
        self.validation_overlap_iou_spin.setEnabled(
            enabled and self.validation_model_mode_combo.currentData() == "overlap"
        )

    def _update_resolution_controls(self, multiscale_enabled: bool) -> None:
        """同步启用状态，避免用户误以为两种分辨率设置会同时生效。"""
        single_scale_enabled = not bool(multiscale_enabled)
        self.imgsz_spin.setEnabled(single_scale_enabled)
        self.imgsz_label.setEnabled(single_scale_enabled)
        self.multiscale_sizes_edit.setEnabled(bool(multiscale_enabled))
        self.multiscale_sizes_label.setEnabled(bool(multiscale_enabled))
        self.show_validation_imgsz_check.setEnabled(bool(multiscale_enabled))

    def accept(self) -> None:
        if self.multiscale_check.isChecked():
            try:
                parse_multiscale_sizes(self.multiscale_sizes_edit.text())
            except ValueError as exc:
                QMessageBox.warning(self, "多分辨率参数无效", str(exc))
                self.multiscale_sizes_edit.setFocus()
                return
        super().accept()

    def values(self) -> tuple:
        enabled = self.multiscale_check.isChecked()
        sizes_text = self.multiscale_sizes_edit.text().strip()
        return (
            self.conf_spin.value(),
            self.iou_spin.value(),
            self.imgsz_spin.value(),
            self.max_det_spin.value(),
            self.min_box_side_spin.value(),
            self.manual_replace_overlap_check.isChecked(),
            self.manual_replace_overlap_iou_spin.value(),
            enabled,
            sizes_text,
            self.show_validation_imgsz_check.isChecked(),
            int(self.camera_device_combo.currentData()),
            self.secondary_model_edit.text().strip(),
            str(self.validation_model_mode_combo.currentData()),
            self.validation_overlap_iou_spin.value(),
        )


class Canvas(QWidget):
    """支持缩放、平移、手动画框和显示标注的画布。"""

    boxCreated = pyqtSignal(int, int, int, int)
    boxSelected = pyqtSignal(int)
    drawStarted = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 320)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.CrossCursor)
        self.setContextMenuPolicy(Qt.NoContextMenu)
        self.setMouseTracking(True)

        self._image: Optional[QImage] = None
        self._boxes: list[dict] = []
        self._validation_boxes: list[dict] = []
        self._selected_index = -1
        self._drag_start: Optional[QPoint] = None
        self._drag_end: Optional[QPoint] = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._is_panning = False
        self._last_mouse_pos: Optional[QPoint] = None
        self._validation_mode = False

    def set_validation_mode(self, enabled: bool) -> None:
        self._validation_mode = bool(enabled)
        self._selected_index = -1
        if enabled:
            self._drag_start = None
            self._drag_end = None
        self.update()

    def set_validation_boxes(self, boxes: list[dict]) -> None:
        """设置仅用于实时验证显示的检测框，不与标注框共享数据。"""
        self._validation_boxes = boxes
        self.update()

    def set_frame(self, frame_bgr: np.ndarray, reset_view: bool = False) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self._image = QImage(
            rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888
        ).copy()
        if reset_view:
            self.reset_view()
        self.update()

    def clear_frame(self) -> None:
        self._image = None
        self.update()

    def set_boxes(self, boxes: list[dict]) -> None:
        self._boxes = boxes
        # 每次切换帧/刷新列表都清除旧选择，避免同一索引误选到另一帧的目标。
        self._selected_index = -1
        self.update()

    def select_box(self, index: int) -> None:
        self._selected_index = index if 0 <= index < len(self._boxes) else -1
        self.update()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    def _layout(self):
        if self._image is None:
            return None
        iw, ih = self._image.width(), self._image.height()
        if iw <= 0 or ih <= 0 or self.width() <= 0 or self.height() <= 0:
            return None
        base_scale = min(self.width() / iw, self.height() / ih)
        scale = base_scale * self._zoom
        ox = (self.width() - iw * scale) / 2.0 + self._pan_x
        oy = (self.height() - ih * scale) / 2.0 + self._pan_y
        return scale, ox, oy

    def _to_image(self, pos: QPoint, clamp: bool = False) -> Optional[QPoint]:
        layout = self._layout()
        if layout is None or self._image is None:
            return None
        scale, ox, oy = layout
        image_right = ox + self._image.width() * scale
        image_bottom = oy + self._image.height() * scale
        if not clamp and not (
            ox <= pos.x() <= image_right and oy <= pos.y() <= image_bottom
        ):
            return None

        x = int(round((pos.x() - ox) / scale))
        y = int(round((pos.y() - oy) / scale))
        x = min(max(x, 0), self._image.width() - 1)
        y = min(max(y, 0), self._image.height() - 1)
        return QPoint(x, y)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        layout = self._layout()
        if layout is None or self._image is None:
            painter.end()
            return

        scale, ox, oy = layout
        iw, ih = self._image.width(), self._image.height()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawImage(QRectF(ox, oy, iw * scale, ih * scale), self._image)
        painter.setFont(QFont("Microsoft YaHei", 10))

        visible_boxes = self._validation_boxes if self._validation_mode else self._boxes
        for index, box in enumerate(visible_boxes):
            class_id = int(box["class_id"])
            if self._validation_mode and box.get("model_index") == 2:
                color_sequence = SECONDARY_BOX_COLORS
            else:
                color_sequence = BOX_COLORS
            color = color_sequence[class_id % len(color_sequence)]
            rect = QRectF(
                ox + float(box["x1"]) * scale,
                oy + float(box["y1"]) * scale,
                (float(box["x2"]) - float(box["x1"])) * scale,
                (float(box["y2"]) - float(box["y1"])) * scale,
            )
            if not self._validation_mode and index == self._selected_index:
                painter.setPen(QPen(QColor(255, 255, 255), 5))
                painter.drawRect(rect)
            painter.setPen(QPen(color, 2))
            painter.drawRect(rect)

            text_y = rect.top() - 6 if rect.top() > 18 else rect.bottom() + 16
            label = str(box.get("class_name", box["class_id"]))
            if self._validation_mode:
                model_index = box.get("model_index")
                if model_index in (1, 2):
                    label = f"{model_index}-{label}"
                label += f"  {float(box.get('confidence', 0.0)):.2f}"
                source_imgsz = box.get("source_imgsz")
                if source_imgsz is not None and box.get("show_source_imgsz", False):
                    label += f"  {int(source_imgsz)}"
            painter.setPen(QPen(QColor(0, 0, 0), 3))
            painter.drawText(QPoint(int(rect.left()) + 1, int(text_y) + 1), label)
            painter.setPen(QPen(color, 1))
            painter.drawText(QPoint(int(rect.left()), int(text_y)), label)

        if self._drag_start is not None and self._drag_end is not None:
            x1, x2 = sorted((self._drag_start.x(), self._drag_end.x()))
            y1, y2 = sorted((self._drag_start.y(), self._drag_end.y()))
            preview = QRectF(
                ox + x1 * scale,
                oy + y1 * scale,
                max(1, x2 - x1) * scale,
                max(1, y2 - y1) * scale,
            )
            painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
            painter.drawRect(preview)
        painter.end()

    def wheelEvent(self, event) -> None:
        if self._image is None:
            return
        img_pos = self._to_image(event.pos(), clamp=False)
        if img_pos is None:
            return

        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        new_zoom = min(max(self._zoom * zoom_factor, 0.1), 20.0)
        iw, ih = self._image.width(), self._image.height()
        base_scale = min(self.width() / iw, self.height() / ih)
        new_scale = base_scale * new_zoom
        self._pan_x = (
            event.pos().x()
            - (self.width() - iw * new_scale) / 2.0
            - img_pos.x() * new_scale
        )
        self._pan_y = (
            event.pos().y()
            - (self.height() - ih * new_scale) / 2.0
            - img_pos.y() * new_scale
        )
        self._zoom = new_zoom
        self.update()

    def mousePressEvent(self, event) -> None:
        if self._image is None:
            return
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._is_panning = True
            self._last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() == Qt.LeftButton:
            if self._validation_mode:
                return
            start = self._to_image(event.pos(), clamp=False)
            # 点击图片四周的留白不应被错误映射到图像边缘。
            if start is None:
                return
            self._drag_start = start
            self._drag_end = start
            self.drawStarted.emit()
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._is_panning and self._last_mouse_pos is not None:
            self._pan_x += event.pos().x() - self._last_mouse_pos.x()
            self._pan_y += event.pos().y() - self._last_mouse_pos.y()
            self._last_mouse_pos = event.pos()
            self.update()
            return
        if self._drag_start is not None:
            self._drag_end = self._to_image(event.pos(), clamp=True)
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self._is_panning = False
            self._last_mouse_pos = None
            self.setCursor(Qt.CrossCursor)
            return

        if event.button() == Qt.LeftButton and self._validation_mode:
            self._drag_start = None
            self._drag_end = None
            self.update()
            return

        if event.button() == Qt.LeftButton and self._drag_start is not None:
            end = self._to_image(event.pos(), clamp=True)
            start = self._drag_start
            self._drag_start = None
            self._drag_end = None
            self.update()
            if end is None:
                return
            x1, x2 = sorted((start.x(), end.x()))
            y1, y2 = sorted((start.y(), end.y()))
            if x2 - x1 > 3 and y2 - y1 > 3:
                self.boxCreated.emit(x1, y1, x2, y2)
            else:
                # 点击已有框即可选中；重叠时优先选中绘制顺序靠后的框。
                px, py = end.x(), end.y()
                hit_index = -1
                for index in range(len(self._boxes) - 1, -1, -1):
                    box = self._boxes[index]
                    if (
                        float(box["x1"]) <= px <= float(box["x2"])
                        and float(box["y1"]) <= py <= float(box["y2"])
                    ):
                        hit_index = index
                        break
                self.boxSelected.emit(hit_index)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.reset_view()
