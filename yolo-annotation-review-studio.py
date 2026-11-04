# -*- coding: utf-8 -*-
"""YOLO 图像/视频预标注与人工复核工具。

依赖：Python 3.9+、PyQt5、opencv-python、numpy；AI 预标注另需 ultralytics。
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QRectF, Qt, QThread, QSettings, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QCheckBox,
    QAbstractSpinBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

try:
    from ultralytics import YOLO

    HAS_YOLO = True
except ImportError:
    YOLO = None
    HAS_YOLO = False


VIDEO_SUFFIXES = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".webm"
}
IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}
ALL_SUFFIXES = VIDEO_SUFFIXES | IMAGE_SUFFIXES
SCRIPT_DIR = Path(__file__).resolve().parent

BOX_COLORS = [
    QColor(0, 255, 0),
    QColor(255, 90, 90),
    QColor(80, 160, 255),
    QColor(255, 255, 0),
    QColor(255, 0, 255),
    QColor(0, 255, 255),
]

# 可以直接修改这里，也可以设置环境变量 YOLO_MODEL_PATH，或在程序中点击“加载模型”。
AI_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    r"E:\code\Python\PythonProject1\runs\AC\yolo11n_AC2\weights\best.pt",
)


def load_image_unicode(path: Path) -> Optional[np.ndarray]:
    """使用 imdecode 读取图片，兼容 Windows 中文路径。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError, cv2.error):
        return None


def save_image_unicode(path: Path, image: np.ndarray) -> bool:
    """使用 imencode + tofile 保存图片，兼容 Windows 中文路径。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix or ".jpg"
        ok, buffer = cv2.imencode(suffix, image)
        if not ok:
            return False
        buffer.tofile(str(path))
        return True
    except (OSError, ValueError, cv2.error):
        return False


def write_yolo_label(
    label_path: Path,
    boxes: list[dict],
    img_w: int,
    img_h: int,
) -> None:
    """将像素坐标框写为 YOLO: class_id x_center y_center width height。"""
    if img_w <= 0 or img_h <= 0:
        raise ValueError("图片宽高必须大于 0，无法生成 YOLO 标签。")

    lines: list[str] = []
    for box in boxes:
        class_id = int(box["class_id"])
        if class_id < 0:
            continue

        x1 = max(0.0, min(float(box["x1"]), float(img_w)))
        y1 = max(0.0, min(float(box["y1"]), float(img_h)))
        x2 = max(0.0, min(float(box["x2"]), float(img_w)))
        y2 = max(0.0, min(float(box["y2"]), float(img_h)))
        if x2 <= x1 or y2 <= y1:
            continue

        x_center = ((x1 + x2) / 2.0) / img_w
        y_center = ((y1 + y2) / 2.0) / img_h
        box_w = (x2 - x1) / img_w
        box_h = (y2 - y1) / img_h
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{box_w:.6f} {box_h:.6f}"
        )

    label_path.parent.mkdir(parents=True, exist_ok=True)
    # 空列表也会写出空 txt，这是 YOLO 数据集合法的负样本标注。
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def box_iou(a: dict, b: dict) -> float:
    """计算两个像素坐标框的 IoU，用于避免重复运行 AI 时重复叠框。"""
    ix1 = max(float(a["x1"]), float(b["x1"]))
    iy1 = max(float(a["y1"]), float(b["y1"]))
    ix2 = min(float(a["x2"]), float(b["x2"]))
    iy2 = min(float(a["y2"]), float(b["y2"]))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, float(a["x2"]) - float(a["x1"])) * max(
        0.0, float(a["y2"]) - float(a["y1"])
    )
    area_b = max(0.0, float(b["x2"]) - float(b["x1"])) * max(
        0.0, float(b["y2"]) - float(b["y1"])
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def remove_overlapping_boxes(
    existing_boxes: list[dict], new_box: dict, iou_threshold: float
) -> tuple[list[dict], int]:
    """移除与新人工框 IoU 达到阈值的旧框，返回保留框与删除数量。

    不区分类别或来源：AI 框、已有人工框以及从标签文件加载的框都可被替换。
    该操作只应用于当前帧，不影响其他帧。
    """
    threshold = min(max(float(iou_threshold), 0.01), 1.0)
    kept_boxes: list[dict] = []
    removed_count = 0
    for old_box in existing_boxes:
        if box_iou(old_box, new_box) >= threshold:
            removed_count += 1
        else:
            kept_boxes.append(old_box)
    return kept_boxes, removed_count


def parse_multiscale_sizes(text: str, minimum: int = 320, maximum: int = 8192) -> tuple[int, ...]:
    """解析多分辨率尺寸列表，去重并保持用户输入顺序。"""
    values: list[int] = []
    for token in text.replace("，", ",").replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"无效的推理尺寸：{token!r}") from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"推理尺寸必须在 {minimum} 到 {maximum} 之间：{value}")
        if value not in values:
            values.append(value)
    if len(values) < 2:
        raise ValueError("启用多分辨率推理与结果融合时，至少需要填写两个不同的尺寸，例如：640, 1280, 1920。")
    return tuple(values)


def class_aware_nms(boxes: list[dict], iou_threshold: float, max_det: int) -> list[dict]:
    """对不同分辨率产生的框进行按类别 NMS，保留置信度较高的结果。"""
    if not boxes:
        return []

    ordered = sorted(
        boxes,
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    )
    kept: list[dict] = []
    while ordered and len(kept) < max_det:
        current = ordered.pop(0)
        kept.append(current)
        remaining: list[dict] = []
        current_class = int(current["model_class_id"])
        for candidate in ordered:
            if int(candidate["model_class_id"]) != current_class:
                remaining.append(candidate)
            elif box_iou(current, candidate) <= iou_threshold:
                remaining.append(candidate)
        ordered = remaining
    return kept


def infer_yolo_multiscale(
    model,
    image: np.ndarray,
    conf: float,
    iou: float,
    max_det: int,
    imgsz_values: tuple[int, ...],
    min_box_side: int = 0,
) -> list[dict]:
    """用多个输入尺寸推理同一原图，再按原图坐标融合检测框。

    Ultralytics Results.boxes.xyxy 返回映射回原图的像素坐标，因此不同
    imgsz 的结果可直接放在一起做跨尺度 NMS，无需自行缩放坐标。
    """
    all_boxes: list[dict] = []
    names = model.names

    for image_size in imgsz_values:
        results = model(
            image,
            conf=conf,
            iou=iou,
            imgsz=int(image_size),
            max_det=max_det,
            verbose=False,
        )
        if not results:
            continue
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        for coords, raw_cls_id, raw_conf in zip(xyxy, cls_ids, confidences):
            cls_id = int(raw_cls_id)
            if isinstance(names, dict):
                class_name = str(names.get(cls_id, cls_id))
            else:
                class_name = str(names[cls_id])
            all_boxes.append({
                "model_class_id": cls_id,
                "class_name": class_name,
                "confidence": float(raw_conf),
                "x1": float(coords[0]),
                "y1": float(coords[1]),
                "x2": float(coords[2]),
                "y2": float(coords[3]),
                "source_imgsz": int(image_size),
            })

    # 按映射回原图后的像素尺寸过滤小目标；宽度或高度任一不足阈值即过滤。
    # 0 表示关闭过滤。该逻辑同时作用于单尺度和多尺度推理。
    min_box_side = max(0, int(min_box_side))
    if min_box_side > 0:
        all_boxes = [
            box for box in all_boxes
            if min(float(box["x2"]) - float(box["x1"]),
                   float(box["y2"]) - float(box["y1"])) >= min_box_side
        ]

    if len(imgsz_values) <= 1:
        return all_boxes[:max_det]
    return class_aware_nms(all_boxes, iou, max_det)


class PredictThread(QThread):
    """在后台线程执行单帧 YOLO 推理，支持多分辨率融合。"""

    finished_signal = pyqtSignal(str, int, list)
    error_signal = pyqtSignal(str, int, str)

    def __init__(
        self,
        model,
        image: np.ndarray,
        file_path: str,
        frame_index: int,
        conf: float,
        iou: float,
        imgsz_values: tuple[int, ...],
        max_det: int,
        min_box_side: int = 0,
    ):
        super().__init__()
        self.model = model
        self.image = image.copy()
        self.file_path = file_path
        self.frame_index = frame_index
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz_values = tuple(int(v) for v in imgsz_values)
        self.max_det = int(max_det)
        self.min_box_side = max(0, int(min_box_side))

    def run(self) -> None:
        try:
            boxes = infer_yolo_multiscale(
                self.model,
                self.image,
                self.conf,
                self.iou,
                self.max_det,
                self.imgsz_values,
                self.min_box_side,
            )
            self.finished_signal.emit(self.file_path, self.frame_index, boxes)
        except Exception as exc:
            self.error_signal.emit(
                self.file_path, self.frame_index, f"{type(exc).__name__}: {exc}"
            )


class BatchPredictThread(QThread):
    """按固定帧步长扫描视频，支持多分辨率推理与跨尺度融合。"""

    frame_result = pyqtSignal(int, list, int, int)
    progress_signal = pyqtSignal(int, int)
    error_signal = pyqtSignal(int, str)
    completed_signal = pyqtSignal(int, int, bool, bool)

    def __init__(
        self,
        model,
        video_path: str,
        total_frames: int,
        frame_step: int,
        conf: float,
        iou: float,
        imgsz_values: tuple[int, ...],
        max_det: int,
        min_box_side: int = 0,
    ):
        super().__init__()
        self.model = model
        self.video_path = video_path
        self.total_frames = int(total_frames)
        self.frame_step = max(1, int(frame_step))
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz_values = tuple(int(v) for v in imgsz_values)
        self.max_det = int(max_det)
        self.min_box_side = max(0, int(min_box_side))
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def _infer(self, frame: np.ndarray) -> list[dict]:
        return infer_yolo_multiscale(
            self.model,
            frame,
            self.conf,
            self.iou,
            self.max_det,
            self.imgsz_values,
            self.min_box_side,
        )

    def run(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        sampled = 0
        failed = 0
        stopped = False
        fatal_error = False
        try:
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频：{self.video_path}")
            frame_index = 0
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if frame_index % self.frame_step == 0:
                    try:
                        boxes = self._infer(frame)
                        frame_h, frame_w = frame.shape[:2]
                        self.frame_result.emit(frame_index, boxes, frame_w, frame_h)
                    except Exception as exc:
                        failed += 1
                        error_text = f"{type(exc).__name__}: {exc}"
                        self.error_signal.emit(frame_index, error_text)
                        sampled += 1
                        self.progress_signal.emit(frame_index + 1, self.total_frames)
                        if "out of memory" in error_text.lower() or "cuda error: out of memory" in error_text.lower():
                            fatal_error = True
                            break
                    else:
                        sampled += 1
                        self.progress_signal.emit(frame_index + 1, self.total_frames)
                frame_index += 1
            stopped = self._stop_event.is_set()
        except Exception as exc:
            failed += 1
            self.error_signal.emit(-1, f"{type(exc).__name__}: {exc}")
        finally:
            cap.release()
            self.completed_signal.emit(sampled, failed, stopped, fatal_error)


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
            QMessageBox.information(self, "提示", "请先添加至少一个类别。")
            return
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self._classes):
            QMessageBox.information(self, "提示", "请先选择一个类别。")
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
    """AI 推理参数设置。参数由主窗口持久化，并应用于后续推理任务。"""

    def __init__(
        self,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        frame_step: int,
        multiscale_enabled: bool,
        multiscale_sizes: str,
        min_box_side: int,
        manual_replace_overlap_enabled: bool,
        manual_replace_overlap_iou: float,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("AI 推理设置")
        self.setModal(True)
        self.setMinimumWidth(420)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.00)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(conf)
        self.conf_spin.setToolTip("置信度低于该阈值的候选检测框将被过滤。")

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 0.99)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(iou)
        self.iou_spin.setToolTip(
            "用于 YOLO 推理阶段的 NMS；启用多分辨率推理时，也用于跨尺度检测框去重。"
        )

        # 两种分辨率模式互斥：多分辨率模式启用时，单尺度 imgsz 不参与推理。
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 8192)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(imgsz)
        self.imgsz_spin.setToolTip(
            "仅在未启用多分辨率推理时生效；启用多分辨率后将改用下方尺寸列表。"
        )
        self.imgsz_label = QLabel("单尺度输入尺寸（imgsz）")

        self.multiscale_check = QCheckBox("启用多分辨率推理与结果融合")
        self.multiscale_check.setChecked(multiscale_enabled)
        self.multiscale_check.setToolTip(
            "使用列表中的多个输入尺寸分别推理，再通过按类别 NMS 融合检测结果。"
        )
        self.multiscale_sizes_edit = QLineEdit(multiscale_sizes)
        self.multiscale_sizes_edit.setPlaceholderText("例如：640, 1280, 1920")
        self.multiscale_sizes_edit.setToolTip(
            "使用英文逗号分隔；至少填写两个不同的尺寸，取值范围为 320–8192。"
        )
        self.multiscale_sizes_label = QLabel("多分辨率尺寸列表")
        self.multiscale_check.toggled.connect(self._update_resolution_controls)

        resolution_form = QFormLayout()
        resolution_form.addRow(self.imgsz_label, self.imgsz_spin)
        resolution_form.addRow(self.multiscale_check)
        resolution_form.addRow(self.multiscale_sizes_label, self.multiscale_sizes_edit)
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
            "检测框的宽度或高度任一小于该值时，将过滤该目标。设置为 0 表示不启用过滤。"
        )

        self.manual_replace_overlap_check = QCheckBox("绘制新框时自动移除重叠标注框")
        self.manual_replace_overlap_check.setChecked(manual_replace_overlap_enabled)
        self.manual_replace_overlap_iou_spin = QDoubleSpinBox()
        self.manual_replace_overlap_iou_spin.setRange(0.01, 1.00)
        self.manual_replace_overlap_iou_spin.setDecimals(2)
        self.manual_replace_overlap_iou_spin.setSingleStep(0.05)
        self.manual_replace_overlap_iou_spin.setValue(manual_replace_overlap_iou)
        self.manual_replace_overlap_iou_spin.setToolTip(
            "新建人工标注框后，移除与其 IoU 大于或等于该阈值的当前帧旧框；不区分类别或来源。"
        )
        self.manual_replace_overlap_iou_spin.setEnabled(manual_replace_overlap_enabled)
        self.manual_replace_overlap_check.toggled.connect(
            self.manual_replace_overlap_iou_spin.setEnabled
        )

        self.frame_step_spin = QSpinBox()
        self.frame_step_spin.setRange(1, 100000)
        self.frame_step_spin.setSingleStep(1)
        self.frame_step_spin.setValue(frame_step)
        self.frame_step_spin.setToolTip(
            "批量视频预标注的采样间隔；例如设置为 30 时，处理第 1、31、61……帧。"
        )

        form = QFormLayout()
        form.addRow("置信度阈值（conf）", self.conf_spin)
        form.addRow("NMS IoU 阈值（IoU）", self.iou_spin)
        form.addRow(self.resolution_group)
        form.addRow("每帧最大检测数（max_det）", self.max_det_spin)
        form.addRow("最小目标边长", self.min_box_side_spin)
        form.addRow(self.manual_replace_overlap_check)
        form.addRow("重叠判定阈值（IoU）", self.manual_replace_overlap_iou_spin)
        form.addRow("批量视频采样间隔（帧）", self.frame_step_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("应用")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _update_resolution_controls(self, multiscale_enabled: bool) -> None:
        """同步启用状态，避免用户误以为两种分辨率设置会同时生效。"""
        single_scale_enabled = not bool(multiscale_enabled)
        self.imgsz_spin.setEnabled(single_scale_enabled)
        self.imgsz_label.setEnabled(single_scale_enabled)
        self.multiscale_sizes_edit.setEnabled(bool(multiscale_enabled))
        self.multiscale_sizes_label.setEnabled(bool(multiscale_enabled))

    def accept(self) -> None:
        if self.multiscale_check.isChecked():
            try:
                parse_multiscale_sizes(self.multiscale_sizes_edit.text())
            except ValueError as exc:
                QMessageBox.warning(self, "多分辨率参数无效", str(exc))
                self.multiscale_sizes_edit.setFocus()
                return
        super().accept()

    def values(self) -> tuple[float, float, int, int, int, bool, float, int, bool, str]:
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
            self.frame_step_spin.value(),
            enabled,
            sizes_text,
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
        self._selected_index = -1
        self._drag_start: Optional[QPoint] = None
        self._drag_end: Optional[QPoint] = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._is_panning = False
        self._last_mouse_pos: Optional[QPoint] = None

    def set_frame(self, frame_bgr: np.ndarray, reset_view: bool = False) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self._image = QImage(
            rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888
        ).copy()
        if reset_view:
            self.reset_view()
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

        for index, box in enumerate(self._boxes):
            color = BOX_COLORS[int(box["class_id"]) % len(BOX_COLORS)]
            rect = QRectF(
                ox + float(box["x1"]) * scale,
                oy + float(box["y1"]) * scale,
                (float(box["x2"]) - float(box["x1"])) * scale,
                (float(box["y2"]) - float(box["y1"])) * scale,
            )
            if index == self._selected_index:
                painter.setPen(QPen(QColor(255, 255, 255), 5))
                painter.drawRect(rect)
            painter.setPen(QPen(color, 2))
            painter.drawRect(rect)

            text_y = rect.top() - 6 if rect.top() > 18 else rect.bottom() + 16
            label = str(box.get("class_name", box["class_id"]))
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


class AnnotatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 图像与视频标注工具")
        self.resize(1450, 900)

        self.current_file: Optional[Path] = None
        self.is_video = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames = 0
        self.frame_idx = 0
        self.frame: Optional[np.ndarray] = None
        self.image_w = 0
        self.image_h = 0

        # key 为视频帧序号；图片标注统一使用 key=0。
        # 即使某帧没有框，也保留空列表，以便保存为 YOLO 负样本。
        self.annotations: dict[int, list[dict]] = {}
        self.dirty = False

        self.file_list: list[Path] = []
        self.file_index = -1
        self.media_dir: Optional[Path] = None
        self.output_root: Optional[Path] = None
        self.default_output_root: Optional[Path] = None
        self.labels_relative_dir = Path("labels")
        self.frame_images_relative_dir = Path("images")
        self.current_labels_dir: Optional[Path] = None
        self.frame_images_dir: Optional[Path] = None
        self.classes_file: Optional[Path] = None
        # 读取时按“当前输出目录 -> 原数据集目录 -> 曾用输出目录”的顺序查找，
        # 这样用户切换保存目录后，仍能读取之前已经存在的标注。
        self.label_search_dirs: list[Path] = []

        self.classes: list[str] = []
        self.last_class_name: Optional[str] = None

        self.ai_model = None
        self.model_path: Optional[Path] = None
        self.predict_thread: Optional[PredictThread] = None
        self.predicting = False

        # 使用 QSettings 持久化推理参数；首次启动采用常用默认值。
        self.app_settings = QSettings("YOLOAnnotator", "YOLOAnnotator")
        self.inference_conf = self._read_float_setting("inference/conf", 0.40, 0.01, 1.00)
        self.inference_iou = self._read_float_setting("inference/iou", 0.70, 0.01, 0.99)
        self.inference_imgsz = self._read_int_setting("inference/imgsz", 640, 320, 8192)
        self.inference_max_det = self._read_int_setting("inference/max_det", 300, 1, 3000)
        self.inference_min_box_side = self._read_int_setting(
            "inference/min_box_side", 0, 0, 8192
        )
        self.manual_replace_overlap_enabled = self.app_settings.value(
            "annotation/manual_replace_overlap_enabled", True, type=bool
        )
        self.manual_replace_overlap_iou = self._read_float_setting(
            "annotation/manual_replace_overlap_iou", 0.50, 0.01, 1.00
        )
        self.multiscale_enabled = self.app_settings.value("inference/multiscale_enabled", False, type=bool)
        self.multiscale_sizes = str(self.app_settings.value("inference/multiscale_sizes", "640, 1280, 1920"))
        self.frame_step = self._read_int_setting("video/frame_step", 30, 1, 100000)
        self.batch_thread: Optional[BatchPredictThread] = None
        self.batch_added_count = 0
        self.batch_error_count = 0
        self.batch_skipped_count = 0
        self._batch_stopped_by_user = False
        self._updating_file_list = False

        self._build_ui()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        model_path = Path(AI_MODEL_PATH).expanduser()
        if model_path.is_file():
            self._load_model(model_path, startup=True)
        elif not HAS_YOLO:
            self.status_lbl.setText(
                "就绪。未安装 ultralytics；仍可进行人工标注，但无法使用 AI 预标注。"
            )
        else:
            self.status_lbl.setText(
                "就绪。请打开图像/视频目录，并加载 YOLO 权重以启用 AI 预标注。"
            )
        self._sync_ui_state()

    def _build_ui(self) -> None:
        self.canvas = Canvas()
        self.canvas.boxCreated.connect(self._on_box_created)
        self.canvas.boxSelected.connect(self._on_canvas_box_selected)
        self.canvas.drawStarted.connect(self._stop_playback)

        # 使用一排可直接点击的一级操作项，不再将常用功能放进二级下拉菜单。
        self.menuBar().hide()
        toolbar = QToolBar("主操作栏", self)
        toolbar.setObjectName("mainActionToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        self.open_action = QAction("打开文件夹", self)
        self.open_action.triggered.connect(self._open_folder)
        toolbar.addAction(self.open_action)

        self.output_dir_action = QAction("设置标注输出目录", self)
        self.output_dir_action.triggered.connect(self._choose_output_dir)
        toolbar.addAction(self.output_dir_action)

        self.save_action = QAction("保存标注（S）", self)
        self.save_action.setShortcut("S")
        self.save_action.triggered.connect(self._save)
        toolbar.addAction(self.save_action)

        toolbar.addSeparator()
        self.model_action = QAction("加载模型", self)
        self.model_action.triggered.connect(self._choose_model)
        toolbar.addAction(self.model_action)

        self.model_settings_action = QAction("AI 推理设置", self)
        self.model_settings_action.triggered.connect(self._show_model_settings)
        toolbar.addAction(self.model_settings_action)

        toolbar.addSeparator()
        self.prev_frame_action = QAction("上一标注帧（A）", self)
        self.prev_frame_action.setShortcut("A")
        self.prev_frame_action.setToolTip(
            "跳转到当前帧之前最近一个包含有效标注框的视频帧；空标签帧会跳过。"
        )
        self.prev_frame_action.triggered.connect(lambda: self._step_frame(-1))
        toolbar.addAction(self.prev_frame_action)

        self.next_frame_action = QAction("下一标注帧（D）", self)
        self.next_frame_action.setShortcut("D")
        self.next_frame_action.setToolTip(
            "跳转到当前帧之后最近一个包含有效标注框的视频帧；空标签帧会跳过。"
        )
        self.next_frame_action.triggered.connect(lambda: self._step_frame(1))
        toolbar.addAction(self.next_frame_action)

        toolbar.addSeparator()
        self.ai_action = QAction("当前帧 AI 预标注", self)
        self.ai_action.triggered.connect(self._run_ai_predict)
        toolbar.addAction(self.ai_action)

        self.batch_ai_action = QAction("批量 AI 预标注", self)
        self.batch_ai_action.triggered.connect(self._run_batch_ai_predict)
        toolbar.addAction(self.batch_ai_action)

        self.batch_stop_action = QAction("停止批量预标注", self)
        self.batch_stop_action.triggered.connect(self._stop_batch_ai_predict)
        self.batch_stop_action.setEnabled(False)
        toolbar.addAction(self.batch_stop_action)

        toolbar.addSeparator()
        self.delete_action = QAction("删除选中框（Backspace）", self)
        self.delete_action.setToolTip("选中一个标注框后，按 Backspace 删除该框。")
        self.delete_action.triggered.connect(self._delete_selected_box)
        toolbar.addAction(self.delete_action)

        toolbar.addSeparator()
        self.clear_current_action = QAction("清空当前画面标注", self)
        self.clear_current_action.setToolTip(
            "清空当前图片或视频帧的全部标注；保存后会写出空 YOLO 标签文件。"
        )
        self.clear_current_action.triggered.connect(
            self._clear_current_frame_annotations
        )
        toolbar.addAction(self.clear_current_action)

        self.clear_cache_action = QAction("清空未保存标注缓存", self)
        self.clear_cache_action.setToolTip(
            "丢弃当前文件所有未保存的标注修改；不删除磁盘上已经保存的标签。"
        )
        self.clear_cache_action.triggered.connect(
            self._clear_unsaved_annotation_cache
        )
        toolbar.addAction(self.clear_cache_action)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider_label = QLabel("0 / 0")
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.slider, 1)
        slider_layout.addWidget(self.slider_label)

        central_layout = QVBoxLayout()
        central_layout.addWidget(self.canvas, 1)
        central_layout.addLayout(slider_layout)
        central_widget = QWidget()
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        # 左侧文件面板同时显示当前模型，避免模型信息隐藏在状态栏里。
        file_panel = QWidget()
        file_layout = QVBoxLayout(file_panel)
        file_layout.setContentsMargins(6, 6, 6, 6)
        self.model_info_lbl = QLabel("当前模型：未加载")
        self.model_info_lbl.setWordWrap(True)
        self.model_info_lbl.setToolTip("当前加载的 YOLO 权重文件")
        file_layout.addWidget(self.model_info_lbl)
        self.output_dir_lbl = QLabel("保存目录：尚未选择")
        self.output_dir_lbl.setWordWrap(True)
        self.output_dir_lbl.setToolTip("标注文件、视频抽帧图片和 classes.txt 的输出位置")
        file_layout.addWidget(self.output_dir_lbl)
        self.file_filter_label = QLabel("筛选文件：")
        file_layout.addWidget(self.file_filter_label)
        self.file_filter = QLineEdit()
        self.file_filter.setPlaceholderText("输入文件名或相对路径关键词")
        self.file_filter.setClearButtonEnabled(True)
        self.file_filter.setToolTip(
            "只筛选下方文件列表，不会删除或移动文件。\n"
            "输入文件名或相对路径中的关键词进行筛选；清空后显示全部文件。"
        )
        self.file_filter.textChanged.connect(self._filter_file_list)
        self.file_count_lbl = QLabel("尚未加载文件夹")
        self.file_list_widget = QListWidget()
        self.file_list_widget.currentItemChanged.connect(self._on_file_item_changed)
        file_layout.addWidget(self.file_filter)
        file_layout.addWidget(self.file_count_lbl)
        file_layout.addWidget(self.file_list_widget, 1)
        file_dock = QDockWidget("文件列表", self)
        file_dock.setObjectName("fileListDock")
        file_dock.setWidget(file_panel)
        file_dock.setMinimumWidth(230)
        self.addDockWidget(Qt.LeftDockWidgetArea, file_dock)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["目标类别", "框编号 / 坐标（px）"])
        self.tree.currentItemChanged.connect(self._on_tree_selection_changed)
        self.tree.itemDoubleClicked.connect(self._on_tree_item_double_clicked)
        dock = QDockWidget("当前帧标注列表", self)
        dock.setWidget(self.tree)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.status_lbl = QLabel("就绪。")
        self.statusBar().addWidget(self.status_lbl)

    # ---------- 设置 ----------

    def _read_float_setting(
        self, key: str, default: float, minimum: float, maximum: float
    ) -> float:
        try:
            value = float(self.app_settings.value(key, default))
        except (TypeError, ValueError):
            value = default
        return min(max(value, minimum), maximum)

    def _read_int_setting(
        self, key: str, default: int, minimum: int, maximum: int
    ) -> int:
        try:
            value = int(self.app_settings.value(key, default))
        except (TypeError, ValueError):
            value = default
        return min(max(value, minimum), maximum)

    def _show_model_settings(self) -> None:
        dialog = ModelSettingsDialog(
            self.inference_conf,
            self.inference_iou,
            self.inference_imgsz,
            self.inference_max_det,
            self.frame_step,
            self.multiscale_enabled,
            self.multiscale_sizes,
            self.inference_min_box_side,
            self.manual_replace_overlap_enabled,
            self.manual_replace_overlap_iou,
            self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        try:
            (
                self.inference_conf,
                self.inference_iou,
                self.inference_imgsz,
                self.inference_max_det,
                self.inference_min_box_side,
                self.manual_replace_overlap_enabled,
                self.manual_replace_overlap_iou,
                self.frame_step,
                self.multiscale_enabled,
                self.multiscale_sizes,
            ) = dialog.values()
        except ValueError as exc:
            QMessageBox.warning(self, "多分辨率参数无效", str(exc))
            return
        self.app_settings.setValue("inference/conf", self.inference_conf)
        self.app_settings.setValue("inference/iou", self.inference_iou)
        self.app_settings.setValue("inference/imgsz", self.inference_imgsz)
        self.app_settings.setValue("inference/max_det", self.inference_max_det)
        self.app_settings.setValue("inference/min_box_side", self.inference_min_box_side)
        self.app_settings.setValue(
            "annotation/manual_replace_overlap_enabled",
            self.manual_replace_overlap_enabled,
        )
        self.app_settings.setValue(
            "annotation/manual_replace_overlap_iou", self.manual_replace_overlap_iou
        )
        self.app_settings.setValue("inference/multiscale_enabled", self.multiscale_enabled)
        self.app_settings.setValue("inference/multiscale_sizes", self.multiscale_sizes)
        self.app_settings.setValue("video/frame_step", self.frame_step)
        self.app_settings.sync()
        self.status_lbl.setText(
            "AI 推理参数已保存："
            f"conf={self.inference_conf:.2f}, iou={self.inference_iou:.2f}, "
            f"单尺度尺寸={self.inference_imgsz}, max_det={self.inference_max_det}, "
            f"最小目标边长={self.inference_min_box_side}px, "
            f"新框重叠清理={self.manual_replace_overlap_enabled} "
            f"(IoU={self.manual_replace_overlap_iou:.2f}), "
            f"多分辨率推理={self.multiscale_enabled} ({self.multiscale_sizes}), "
            f"批量采样间隔={self.frame_step} 帧"
        )

    # ---------- 模型 ----------

    @staticmethod
    def _names_from_model(model) -> list[str]:
        names = model.names
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names)]
        return [str(name) for name in names]

    def _load_model(self, path: Path, startup: bool = False) -> bool:
        if not HAS_YOLO:
            if not startup:
                QMessageBox.warning(
                    self,
                    "缺少依赖",
                    "当前环境未安装 ultralytics。\n请执行：pip install ultralytics",
                )
            return False
        if not path.is_file():
            if not startup:
                QMessageBox.warning(self, "模型不存在", f"找不到模型文件：\n{path}")
            return False

        try:
            self.ai_model = YOLO(str(path))
            self.model_path = path.resolve()
            if not self.classes:
                self.classes = self._names_from_model(self.ai_model)
            self._update_model_info()
            self.status_lbl.setText(f"模型已加载：{path}")
            self._sync_ui_state()
            return True
        except Exception as exc:
            self.ai_model = None
            self.model_path = None
            self._update_model_info()
            message = f"模型加载失败：{type(exc).__name__}: {exc}"
            self.status_lbl.setText(message)
            if not startup:
                QMessageBox.critical(self, "模型加载失败", message)
            return False

    def _update_model_info(self) -> None:
        if self.model_path is None or self.ai_model is None:
            text = "当前模型：未加载"
            tooltip = text
        else:
            text = f"当前模型：{self.model_path.name}"
            tooltip = str(self.model_path)
        self.model_info_lbl.setText(text)
        self.model_info_lbl.setToolTip(tooltip)

    def _choose_model(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择 YOLO 权重文件",
            str(SCRIPT_DIR),
            "YOLO 权重 (*.pt);;所有文件 (*)",
        )
        if filename:
            self._load_model(Path(filename))

    # ---------- 数据集目录与文件 ----------

    def _configure_output_paths(self, output_root: Path) -> None:
        """根据用户选择的输出根目录，重新计算标签、抽帧和类别文件位置。"""
        self.output_root = Path(output_root).expanduser()
        self.current_labels_dir = self.output_root / self.labels_relative_dir
        self.frame_images_dir = self.output_root / self.frame_images_relative_dir
        self.classes_file = self.output_root / "classes.txt"
        if hasattr(self, "output_dir_lbl"):
            self.output_dir_lbl.setText(f"保存目录：{self.output_root}")
            self.output_dir_lbl.setToolTip(str(self.output_root))

    def _choose_output_dir(self) -> None:
        """允许用户独立指定标注结果的输出根目录。"""
        if self.media_dir is None or self.default_output_root is None:
            QMessageBox.information(self, "提示", "请先打开图片或视频文件夹。")
            return

        last_dir = self.app_settings.value("paths/last_output_dir", "")
        initial_dir = (
            Path(last_dir)
            if last_dir and Path(last_dir).is_dir()
            else self.output_root
        )
        if initial_dir is None or not initial_dir.is_dir():
            initial_dir = self.default_output_root

        folder = QFileDialog.getExistingDirectory(
            self,
            "选择标注结果保存目录",
            str(initial_dir),
            QFileDialog.ShowDirsOnly,
        )
        if not folder:
            return

        output_root = Path(folder).expanduser()
        destination_classes = output_root / "classes.txt"
        if destination_classes.is_file():
            try:
                existing_classes = [
                    line.strip()
                    for line in destination_classes.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip()
                ]
            except OSError as exc:
                QMessageBox.warning(
                    self, "无法检查类别文件", f"读取目标目录的 classes.txt 失败：\n{exc}"
                )
                return
            if not self.classes and existing_classes:
                # 当前数据集没有提供类别表时，可沿用目标输出目录已有的类别映射。
                self.classes = existing_classes
            elif self.classes and existing_classes != self.classes:
                answer = QMessageBox.warning(
                    self,
                    "类别文件不一致",
                    "所选保存目录中已经存在不同内容的 classes.txt。\n"
                    "继续保存可能覆盖目标目录原有的类别映射。是否仍然使用该目录？",
                    QMessageBox.Yes | QMessageBox.Cancel,
                    QMessageBox.Cancel,
                )
                if answer != QMessageBox.Yes:
                    return

        previous_label_dirs = list(self.label_search_dirs)
        self._configure_output_paths(output_root)
        if self.current_labels_dir is not None:
            ordered_dirs = [self.current_labels_dir, *previous_label_dirs]
            self.label_search_dirs = list(dict.fromkeys(ordered_dirs))
        self.app_settings.setValue("paths/last_output_dir", str(output_root))
        self.app_settings.sync()
        self.status_lbl.setText(
            f"保存目录已设置为：{output_root}。已有标注不会移动，之后保存时将写入新目录。"
        )

    def _open_folder(self) -> None:
        if not self._confirm_unsaved_changes():
            return

        folder = QFileDialog.getExistingDirectory(
            self,
            "选择包含图片/视频的文件夹",
            str(SCRIPT_DIR),
            QFileDialog.ShowDirsOnly,
        )
        if not folder:
            return
        selected_dir = Path(folder)

        # 识别常见数据集目录结构。输出根目录默认跟随数据集根目录，
        # 用户随后可以通过“设置保存目录”将结果写到其他位置。
        if selected_dir.name.lower() == "images":
            media_dir = selected_dir
            dataset_root = selected_dir.parent
            labels_relative_dir = Path("labels")
            frame_images_relative_dir = Path("images")
            source_classes_file = dataset_root / "classes.txt"
        elif selected_dir.parent.name.lower() == "images":
            media_dir = selected_dir
            dataset_root = selected_dir.parent.parent
            labels_relative_dir = Path("labels") / selected_dir.name
            frame_images_relative_dir = Path("images") / selected_dir.name
            source_classes_file = dataset_root / "classes.txt"
        elif (selected_dir / "images").is_dir() and not any(
            p.is_file() and p.suffix.lower() in ALL_SUFFIXES
            for p in selected_dir.iterdir()
        ):
            media_dir = selected_dir / "images"
            dataset_root = selected_dir
            labels_relative_dir = Path("labels")
            frame_images_relative_dir = Path("images")
            source_classes_file = dataset_root / "classes.txt"
        else:
            media_dir = selected_dir
            dataset_root = selected_dir
            labels_relative_dir = Path("labels")
            # 视频抽帧保存在输出根目录的 images 子目录，避免与原始媒体文件混在一起。
            frame_images_relative_dir = Path("images")
            source_classes_file = dataset_root / "classes.txt"

        if not media_dir.is_dir():
            QMessageBox.warning(self, "目录错误", f"媒体目录不存在：\n{media_dir}")
            return

        # 先用候选目录扫描，确认确实存在媒体文件后再更新窗口状态。
        candidate_labels_dir = dataset_root / labels_relative_dir
        candidate_frame_images_dir = dataset_root / frame_images_relative_dir

        def is_media_file(candidate: Path) -> bool:
            if not candidate.is_file() or candidate.suffix.lower() not in ALL_SUFFIXES:
                return False
            # 仅当输出目录是媒体目录的子目录时跳过，不能把 media_dir 本身全部排除。
            if candidate_frame_images_dir != media_dir:
                try:
                    candidate.relative_to(candidate_frame_images_dir)
                    return False
                except ValueError:
                    pass
            if candidate_labels_dir != media_dir:
                try:
                    candidate.relative_to(candidate_labels_dir)
                    return False
                except ValueError:
                    pass
            return True

        file_list = sorted(
            (p for p in media_dir.rglob("*") if is_media_file(p)),
            key=lambda p: str(p.relative_to(media_dir)).lower(),
        )
        if not file_list:
            QMessageBox.warning(
                self,
                "没有可处理的文件",
                f"目录中没有找到支持的图片或视频：\n{media_dir}",
            )
            return

        self.media_dir = media_dir
        self.default_output_root = dataset_root
        self.labels_relative_dir = labels_relative_dir
        self.frame_images_relative_dir = frame_images_relative_dir
        self._configure_output_paths(dataset_root)
        self.label_search_dirs = (
            [self.current_labels_dir] if self.current_labels_dir is not None else []
        )
        self.file_list = file_list
        self.file_index = 0
        self._refresh_file_list()

        self.classes = []
        self.last_class_name = None
        if source_classes_file.is_file():
            try:
                self.classes = [
                    line.strip()
                    for line in source_classes_file.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip()
                ]
            except OSError as exc:
                QMessageBox.warning(
                    self, "类别文件读取失败", f"无法读取 {source_classes_file}\n{exc}"
                )
        elif self.ai_model is not None:
            self.classes = self._names_from_model(self.ai_model)

        self._load_file(self.file_list[0])
        self._sync_ui_state()

    def _relative_media_path(self, path: Path) -> Path:
        if self.media_dir is None:
            return Path(path.name)
        try:
            return path.relative_to(self.media_dir)
        except ValueError:
            return Path(path.name)

    def _image_label_path(self, path: Path) -> Path:
        assert self.current_labels_dir is not None
        relative = self._relative_media_path(path)
        return (self.current_labels_dir / relative).with_suffix(".txt")

    def _video_frame_stem(self, frame_index: int) -> Path:
        """返回相对标签目录的帧文件名（不带扩展名）。"""
        assert self.current_file is not None
        relative = self._relative_media_path(self.current_file)
        return relative.parent / f"{relative.stem}_frame_{frame_index:06d}"

    def _video_frame_image_path(self, frame_index: int) -> Path:
        assert self.frame_images_dir is not None
        return self.frame_images_dir / self._video_frame_stem(frame_index).with_suffix(
            ".jpg"
        )

    def _video_frame_label_path(self, frame_index: int) -> Path:
        assert self.current_labels_dir is not None
        return self.current_labels_dir / self._video_frame_stem(frame_index).with_suffix(
            ".txt"
        )

    def _find_existing_label(self, relative_label_path: Path) -> Optional[Path]:
        """按当前输出目录优先、历史目录其次的顺序查找已有标签。"""
        for label_dir in self.label_search_dirs:
            candidate = label_dir / relative_label_path
            if candidate.is_file():
                return candidate
        # 容错：若目录列表尚未初始化，至少检查当前输出目录。
        if self.current_labels_dir is not None:
            candidate = self.current_labels_dir / relative_label_path
            if candidate.is_file():
                return candidate
        return None

    def _existing_image_label_path(self, path: Path) -> Optional[Path]:
        relative = self._relative_media_path(path).with_suffix(".txt")
        return self._find_existing_label(relative)

    def _existing_video_frame_label_path(self, frame_index: int) -> Optional[Path]:
        return self._find_existing_label(
            self._video_frame_stem(frame_index).with_suffix(".txt")
        )

    def _load_file(self, path: Path) -> bool:
        self._stop_playback()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.annotations = {}
        self.frame_idx = 0
        self.frame = None
        self.current_file = path
        self._set_dirty(False)
        self.canvas.reset_view()

        self.is_video = path.suffix.lower() in VIDEO_SUFFIXES
        if self.is_video:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                cap.release()
                QMessageBox.warning(self, "视频打开失败", f"无法打开视频：\n{path}")
                self.current_file = None
                self.total_frames = 0
                self._refresh_frame()
                self._sync_ui_state()
                return False

            self.cap = cap
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                # 部分视频容器无法提供帧数或不支持随机定位，至少允许标注首帧。
                ok, first_frame = cap.read()
                if not ok or first_frame is None:
                    cap.release()
                    self.cap = None
                    QMessageBox.warning(self, "视频读取失败", f"视频没有可读取的帧：\n{path}")
                    self.current_file = None
                    self.total_frames = 0
                    self._refresh_frame()
                    self._sync_ui_state()
                    return False
                self.total_frames = 1
                self.frame_idx = 0
                self.frame = first_frame
                self.image_h, self.image_w = first_frame.shape[:2]
                self.slider.setRange(0, 0)
                label_path = self._existing_video_frame_label_path(0)
                if label_path is not None:
                    self._load_yolo_txt(label_path, 0)
                self._refresh_frame(reset_view=True)
            else:
                self.total_frames = frame_count
                self.slider.setRange(0, self.total_frames - 1)
                self._seek(0, reset_view=True)
                if self.frame is None:
                    self.cap.release()
                    self.cap = None
                    QMessageBox.warning(self, "视频读取失败", f"无法读取视频首帧：\n{path}")
                    self.current_file = None
                    self.total_frames = 0
                    self._refresh_frame()
                    self._sync_ui_state()
                    return False
        else:
            image = load_image_unicode(path)
            if image is None:
                QMessageBox.warning(self, "图片读取失败", f"无法读取图片：\n{path}")
                self.current_file = None
                self.total_frames = 0
                self._refresh_frame()
                self._sync_ui_state()
                return False

            self.frame = image
            self.total_frames = 1
            self.frame_idx = 0
            self.image_h, self.image_w = image.shape[:2]
            self.slider.setRange(0, 0)
            label_path = self._existing_image_label_path(path)
            if label_path is not None:
                self._load_yolo_txt(label_path, 0)
            self._refresh_frame(reset_view=True)

        self._set_file_list_current()
        self._sync_ui_state()
        return True

    # ---------- YOLO 标签读取 ----------

    def _class_name_for_id(self, class_id: int) -> str:
        if class_id < len(self.classes):
            return self.classes[class_id]
        # 文件里出现超出 classes.txt 范围的 id 时，保留编号并补齐占位类名，
        # 防止后续人工新增类别时覆盖这个 ID。
        while len(self.classes) <= class_id:
            self.classes.append(f"class_{len(self.classes)}")
        return self.classes[class_id]

    def _load_yolo_txt(
        self,
        txt_path: Path,
        frame_index: int,
        image_w: Optional[int] = None,
        image_h: Optional[int] = None,
    ) -> None:
        image_w = int(image_w if image_w is not None else self.image_w)
        image_h = int(image_h if image_h is not None else self.image_h)
        boxes: list[dict] = []
        try:
            lines = txt_path.read_text(encoding="utf-8-sig").splitlines()
        except OSError as exc:
            self.status_lbl.setText(f"YOLO 标签文件读取失败：{txt_path}（{exc}）")
            return

        for line_number, line in enumerate(lines, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                self.status_lbl.setText(
                    f"YOLO 标签格式错误，已跳过：{txt_path.name} 第 {line_number} 行"
                )
                continue
            try:
                class_id = int(parts[0])
                cx, cy, width, height = map(float, parts[1:])
                if class_id < 0 or not np.isfinite([cx, cy, width, height]).all():
                    raise ValueError("类别编号或坐标无效")
                if width <= 0 or height <= 0:
                    continue
                x1 = (cx - width / 2.0) * image_w
                y1 = (cy - height / 2.0) * image_h
                x2 = (cx + width / 2.0) * image_w
                y2 = (cy + height / 2.0) * image_h
                x1 = max(0.0, min(x1, float(image_w)))
                y1 = max(0.0, min(y1, float(image_h)))
                x2 = max(0.0, min(x2, float(image_w)))
                y2 = max(0.0, min(y2, float(image_h)))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append(
                    {
                        "class_id": class_id,
                        "class_name": self._class_name_for_id(class_id),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "source": "loaded",
                    }
                )
            except (ValueError, OverflowError):
                self.status_lbl.setText(
                    f"YOLO 标签内容无效，已跳过：{txt_path.name} 第 {line_number} 行"
                )

        # 空 txt 同样需要记录，代表用户/数据集明确标注了负样本。
        self.annotations[frame_index] = boxes

    # ---------- 画面与帧切换 ----------

    def _refresh_frame(self, reset_view: bool = False) -> None:
        if self.frame is None:
            self.canvas.update()
            self.canvas.set_boxes([])
            self.tree.clear()
            self.slider_label.setText("0 / 0")
            return

        self.canvas.set_frame(self.frame, reset_view=reset_view)
        self.canvas.set_boxes(self.annotations.get(self.frame_idx, []))

        self.slider.blockSignals(True)
        self.slider.setValue(self.frame_idx)
        self.slider.blockSignals(False)
        self.slider_label.setText(f"{self.frame_idx + 1} / {self.total_frames}")
        self._refresh_tree()

    def _seek(self, index: int, reset_view: bool = False) -> None:
        if not self.is_video or self.cap is None or self.total_frames <= 0:
            return
        index = min(max(int(index), 0), self.total_frames - 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.status_lbl.setText(f"无法读取第 {index + 1} 帧。")
            return

        self.frame = frame
        self.frame_idx = index
        self.image_h, self.image_w = frame.shape[:2]

        if index not in self.annotations:
            label_path = self._existing_video_frame_label_path(index)
            if label_path is not None:
                self._load_yolo_txt(label_path, index)

        self._refresh_frame(reset_view=reset_view)
        self._sync_ui_state()

    @staticmethod
    def _label_file_has_boxes(label_path: Path) -> bool:
        """判断 YOLO 标签文件是否至少包含一条有效目标记录。

        空文件、空白行和格式错误的行不会被视为带标注帧。
        """
        try:
            lines = label_path.read_text(encoding="utf-8-sig").splitlines()
        except OSError:
            return False

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
                cx, cy, width, height = map(float, parts[1:])
            except (TypeError, ValueError, OverflowError):
                continue
            if (
                class_id >= 0
                and np.isfinite([cx, cy, width, height]).all()
                and width > 0
                and height > 0
            ):
                return True
        return False

    def _marked_video_frame_indices(self) -> list[int]:
        """返回当前视频所有包含标注框的帧序号（从 0 开始）。

        内存缓存优先于磁盘标签：缓存中存在的帧即使被清空，也不会被旧磁盘
        标签重新计入。磁盘标签按 label_search_dirs 的优先顺序查找。
        """
        if not self.is_video or self.current_file is None:
            return []

        marked: set[int] = set()

        # 当前会话缓存可能包含尚未保存的 AI / 人工标注，也可能显式清空了旧框。
        for frame_index, boxes in self.annotations.items():
            if 0 <= frame_index < self.total_frames and boxes:
                marked.add(frame_index)

        # 收集当前视频对应的所有标签文件编号。只扫描当前视频所在的相对目录，
        # 避免把同一数据集其他视频的帧标签混入导航列表。
        relative_video = self._relative_media_path(self.current_file)
        prefix = f"{relative_video.stem}_frame_"
        candidate_indices: set[int] = set()
        label_dirs = self.label_search_dirs or (
            [self.current_labels_dir] if self.current_labels_dir is not None else []
        )
        for label_root in label_dirs:
            if label_root is None:
                continue
            label_folder = label_root / relative_video.parent
            if not label_folder.is_dir():
                continue
            try:
                entries = label_folder.iterdir()
                for path in entries:
                    if path.suffix.lower() != ".txt" or not path.stem.startswith(prefix):
                        continue
                    frame_text = path.stem[len(prefix):]
                    if not frame_text.isdigit():
                        continue
                    frame_index = int(frame_text)
                    if 0 <= frame_index < self.total_frames:
                        candidate_indices.add(frame_index)
            except OSError:
                continue

        # 对每个候选编号重新按目录优先级解析一次，确保当前输出目录中的空标签
        # 能覆盖旧目录里的非空标签。
        for frame_index in candidate_indices:
            if frame_index in self.annotations:
                # 内存缓存对该帧拥有最终决定权（包含显式空列表）。
                continue
            label_path = self._existing_video_frame_label_path(frame_index)
            if label_path is not None and self._label_file_has_boxes(label_path):
                marked.add(frame_index)

        return sorted(marked)

    def _step_frame(self, delta: int) -> None:
        """跳转到前一个 / 后一个包含标注框的视频帧，而非固定步长帧。"""
        if not self.is_video or self.predicting:
            return

        marked_frames = self._marked_video_frame_indices()
        if delta < 0:
            candidates = [index for index in marked_frames if index < self.frame_idx]
            target = max(candidates) if candidates else None
            direction_text = "之前"
        else:
            candidates = [index for index in marked_frames if index > self.frame_idx]
            target = min(candidates) if candidates else None
            direction_text = "之后"

        if target is None:
            self.status_lbl.setText(
                f"当前帧{direction_text}没有其他包含有效标注框的帧。"
            )
            return

        self._seek(target)
        if self.frame_idx != target:
            return
        self.status_lbl.setText(
            f"已跳转到第 {target + 1} 帧（共 {self.total_frames} 帧）；"
            f"当前视频共找到 {len(marked_frames)} 个带标注框的帧。"
        )

    def _on_slider_changed(self, value: int) -> None:
        if self.is_video and value != self.frame_idx:
            self._seek(value)

    def _refresh_file_list(self) -> None:
        """重建左侧文件列表，并保持当前文件选中状态。"""
        if not hasattr(self, "file_list_widget"):
            return
        self._updating_file_list = True
        self.file_list_widget.blockSignals(True)
        self.file_list_widget.clear()
        for index, path in enumerate(self.file_list):
            if self.media_dir is not None:
                try:
                    display_name = str(path.relative_to(self.media_dir))
                except ValueError:
                    display_name = path.name
            else:
                display_name = path.name
            item = QListWidgetItem(display_name)
            item.setToolTip(str(path))
            item.setData(Qt.UserRole, index)
            self.file_list_widget.addItem(item)
        if 0 <= self.file_index < self.file_list_widget.count():
            self.file_list_widget.setCurrentRow(self.file_index)
        self.file_list_widget.blockSignals(False)
        self._updating_file_list = False
        self._filter_file_list(self.file_filter.text())

    def _set_file_list_current(self) -> None:
        if not hasattr(self, "file_list_widget"):
            return
        self._updating_file_list = True
        self.file_list_widget.blockSignals(True)
        if 0 <= self.file_index < self.file_list_widget.count():
            self.file_list_widget.setCurrentRow(self.file_index)
        else:
            self.file_list_widget.setCurrentRow(-1)
        self.file_list_widget.blockSignals(False)
        self._updating_file_list = False

    def _filter_file_list(self, text: str) -> None:
        if not hasattr(self, "file_list_widget"):
            return
        query = text.strip().casefold()
        visible = 0
        total = self.file_list_widget.count()
        for row in range(total):
            item = self.file_list_widget.item(row)
            matched = not query or query in item.text().casefold()
            item.setHidden(not matched)
            if matched:
                visible += 1
        if total:
            self.file_count_lbl.setText(f"显示 {visible} / {total} 个文件")
        else:
            self.file_count_lbl.setText("尚未加载文件夹")

    def _on_file_item_changed(self, current, _previous) -> None:
        if self._updating_file_list or current is None:
            return
        value = current.data(Qt.UserRole)
        if value is None:
            return
        target_index = int(value)
        if target_index == self.file_index:
            return
        if self.predicting:
            self.status_lbl.setText("AI 预标注正在处理，完成后再切换文件。")
            self._set_file_list_current()
            return
        if not self._confirm_unsaved_changes():
            self._set_file_list_current()
            return

        previous_index = self.file_index
        self.file_index = target_index
        if not self._load_file(self.file_list[target_index]):
            self.file_index = previous_index
            # 目标文件无法读取时恢复原文件，避免界面停留在空白状态。
            if 0 <= previous_index < len(self.file_list):
                self._load_file(self.file_list[previous_index])
            else:
                self._set_file_list_current()
            return
        self._set_file_list_current()

    # ---------- 人工标注编辑 ----------

    def _on_box_created(self, x1: int, y1: int, x2: int, y2: int) -> None:
        if self.current_file is None or self.frame is None:
            return

        usage: dict[str, int] = {}
        for frame_boxes in self.annotations.values():
            for box in frame_boxes:
                name = box.get("class_name", str(box.get("class_id", "?")))
                usage[name] = usage.get(name, 0) + 1

        default_index = (
            self.classes.index(self.last_class_name)
            if self.last_class_name in self.classes
            else 0
        )
        dialog = LabelChooserDialog(self.classes, default_index, usage, self)
        if dialog.exec_() != QDialog.Accepted:
            return

        self.classes = dialog.classes_result()
        name = dialog.selected_name()
        self.last_class_name = name
        class_id = self.classes.index(name)
        box = {
            "class_id": class_id,
            "class_name": name,
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "source": "manual",
        }
        current_boxes = self.annotations.setdefault(self.frame_idx, [])
        removed_count = 0
        if self.manual_replace_overlap_enabled:
            current_boxes, removed_count = remove_overlapping_boxes(
                current_boxes, box, self.manual_replace_overlap_iou
            )
        current_boxes.append(box)
        self.annotations[self.frame_idx] = current_boxes
        self._set_dirty(True)
        self._refresh_frame()
        if removed_count:
            self.status_lbl.setText(
                f"已添加人工标注框，并移除 {removed_count} 个重叠标注框 "
                f"（IoU ≥ {self.manual_replace_overlap_iou:.2f}）。"
            )
        else:
            self.status_lbl.setText("已添加人工标注框。")

    def _on_canvas_box_selected(self, index: int) -> None:
        """画布点击框时同步选中右侧列表项目。"""
        if index < 0:
            self.canvas.select_box(-1)
            self.tree.setCurrentItem(None)
            self._sync_ui_state()
            return
        root = self.tree.topLevelItem(0)
        if root is None or index >= root.childCount():
            self.canvas.select_box(-1)
            self._sync_ui_state()
            return
        item = root.child(index)
        self.tree.setCurrentItem(item)
        self.tree.scrollToItem(item)

    def _refresh_tree(self) -> None:
        self.tree.clear()
        boxes = self.annotations.get(self.frame_idx, [])
        root = QTreeWidgetItem([f"第 {self.frame_idx + 1} 帧", f"{len(boxes)} 个标注框"])
        self.tree.addTopLevelItem(root)

        for index, box in enumerate(boxes):
            coords = (
                f"#{index + 1}  "
                f"({box['x1']:.0f}, {box['y1']:.0f}) - "
                f"({box['x2']:.0f}, {box['y2']:.0f})"
            )
            child = QTreeWidgetItem([str(box["class_name"]), coords])
            child.setData(0, Qt.UserRole, index)
            root.addChild(child)
        root.setExpanded(True)

    def _on_tree_selection_changed(self, current, _previous) -> None:
        if current is None:
            self.canvas.select_box(-1)
        else:
            index = current.data(0, Qt.UserRole)
            self.canvas.select_box(int(index) if index is not None else -1)
        self._sync_ui_state()

    def _selected_box_index(self) -> int:
        item = self.tree.currentItem()
        if item is None:
            return -1
        value = item.data(0, Qt.UserRole)
        return int(value) if value is not None else -1

    def _on_tree_item_double_clicked(self, item, _column: int) -> None:
        index_value = item.data(0, Qt.UserRole)
        if index_value is None:
            return
        index = int(index_value)
        boxes = self.annotations.get(self.frame_idx, [])
        if not (0 <= index < len(boxes)):
            return

        current_id = int(boxes[index]["class_id"])
        usage: dict[str, int] = {}
        for frame_boxes in self.annotations.values():
            for box in frame_boxes:
                name = str(box["class_name"])
                usage[name] = usage.get(name, 0) + 1
        dialog = LabelChooserDialog(self.classes, current_id, usage, self)
        if dialog.exec_() != QDialog.Accepted:
            return

        self.classes = dialog.classes_result()
        name = dialog.selected_name()
        boxes[index]["class_id"] = self.classes.index(name)
        boxes[index]["class_name"] = name
        self.last_class_name = name
        self._set_dirty(True)
        self._refresh_frame()

    def eventFilter(self, watched, event) -> bool:
        """在主工作区响应 Backspace，同时保留文本输入控件的原生退格行为。"""
        if (
            event.type() == QEvent.KeyPress
            and event.key() == Qt.Key_Backspace
            and not event.isAutoRepeat()
            and self.isActiveWindow()
        ):
            focus_widget = QApplication.focusWidget()
            text_editing_widgets = (
                QLineEdit,
                QAbstractSpinBox,
                QComboBox,
                QTextEdit,
                QPlainTextEdit,
            )
            if not isinstance(focus_widget, text_editing_widgets):
                if self.delete_action.isEnabled():
                    self._delete_selected_box()
                    return True
        return super().eventFilter(watched, event)

    def _delete_selected_box(self) -> None:
        index = self._selected_box_index()
        boxes = self.annotations.get(self.frame_idx, [])
        if index < 0 or index >= len(boxes):
            return
        del boxes[index]
        # 保留空列表，保存时可输出空标签文件。
        self.annotations[self.frame_idx] = boxes
        self._set_dirty(True)
        self._refresh_frame()
        self.status_lbl.setText("已删除选中的标注框。")

    # ---------- AI 预标注 ----------

    def _active_imgsz_values(self) -> tuple[int, ...]:
        if self.multiscale_enabled:
            return parse_multiscale_sizes(self.multiscale_sizes)
        return (int(self.inference_imgsz),)

    def _run_ai_predict(self) -> None:
        if self.ai_model is None:
            QMessageBox.warning(
                self,
                "模型不可用",
                "尚未加载 YOLO 模型。\n请安装 ultralytics 并点击顶部“加载模型”选择 .pt 权重。",
            )
            return
        if self.frame is None or self.current_file is None:
            return
        if self.predicting:
            self.status_lbl.setText("AI 预标注正在执行，请等待当前任务完成。")
            return

        self.predicting = True
        self._sync_ui_state()
        self.status_lbl.setText(
            f"AI 预标注处理中：{self.current_file.name}，第 {self.frame_idx + 1} 帧……"
        )
        self.predict_thread = PredictThread(
            self.ai_model,
            self.frame,
            str(self.current_file),
            self.frame_idx,
            self.inference_conf,
            self.inference_iou,
            self._active_imgsz_values(),
            self.inference_max_det,
            self.inference_min_box_side,
        )
        self.predict_thread.finished_signal.connect(self._on_ai_finished)
        self.predict_thread.error_signal.connect(self._on_ai_error)
        self.predict_thread.finished.connect(self._on_ai_thread_stopped)
        self.predict_thread.start()

    def _apply_ai_boxes(
        self,
        frame_index: int,
        new_boxes: list[dict],
        ensure_empty_frame: bool = False,
        replace_existing: bool = False,
    ) -> tuple[int, int]:
        """映射模型类别；单帧预标注可替换旧框，批量预标注默认合并。"""
        if self.ai_model is None:
            return 0, len(new_boxes)
        if not self.classes:
            self.classes = self._names_from_model(self.ai_model)

        name_to_dataset_id = {name: index for index, name in enumerate(self.classes)}
        mapped_boxes: list[dict] = []
        skipped = 0
        for box in new_boxes:
            name = str(box["class_name"])
            if name not in name_to_dataset_id:
                skipped += 1
                continue
            mapped_boxes.append({
                "class_id": name_to_dataset_id[name],
                "class_name": name,
                "x1": box["x1"],
                "y1": box["y1"],
                "x2": box["x2"],
                "y2": box["y2"],
                "source": "ai",
            })

        # 如果模型确实返回了目标、但类别名称一个都无法映射，保留原标注，
        # 避免类别配置错误时误删人工结果。模型正常返回空结果则视为有效负样本。
        valid_result = not new_boxes or bool(mapped_boxes)
        if not valid_result:
            return 0, skipped

        old_boxes = self.annotations.get(frame_index)
        if replace_existing:
            # 单帧 AI 预标注采用“重新标注”语义：成功后整帧替换，空结果也会清空旧框。
            self.annotations[frame_index] = mapped_boxes
            changed = old_boxes != mapped_boxes
            added = len(mapped_boxes)
        else:
            # 批量预标注默认保留已有人工作业，只追加不重复的 AI 框。
            if ensure_empty_frame and frame_index not in self.annotations:
                self.annotations[frame_index] = []
            current_boxes = self.annotations.setdefault(frame_index, []) if (
                ensure_empty_frame or mapped_boxes or frame_index in self.annotations
            ) else []
            added = 0
            for candidate in mapped_boxes:
                duplicate = any(
                    int(existing["class_id"]) == int(candidate["class_id"])
                    and box_iou(existing, candidate) >= 0.85
                    for existing in current_boxes
                )
                if not duplicate:
                    current_boxes.append(candidate)
                    added += 1
            changed = added > 0 or (
                ensure_empty_frame and old_boxes is None
            )

        if changed:
            self._set_dirty(True)
        if self.frame_idx == frame_index and self.frame is not None:
            self._refresh_frame()
        return added, skipped

    def _on_ai_finished(self, file_path: str, frame_index: int, new_boxes: list) -> None:
        if self.current_file is None or str(self.current_file) != file_path:
            self.status_lbl.setText("AI 推理完成，但原文件已切换，结果未应用。")
            return

        added, skipped = self._apply_ai_boxes(
            frame_index, new_boxes, replace_existing=True
        )
        message = f"当前帧 AI 预标注完成：检测到 {added} 个目标。"
        if not new_boxes and not skipped:
            message = "当前帧未检测到目标，原有标注框已清除。"
        if skipped:
            message += f"另有 {skipped} 个框因模型类别与数据集类别名称不一致而跳过。"
        self.status_lbl.setText(message)

    def _run_batch_ai_predict(self) -> None:
        if self.ai_model is None:
            QMessageBox.warning(self, "模型不可用", "请先点击顶部“加载模型”选择 YOLO 权重。")
            return
        if not self.is_video or self.current_file is None:
            QMessageBox.information(self, "需要视频文件", "请先在文件列表中选择一个视频文件。")
            return
        if self.predicting:
            self.status_lbl.setText("当前已有 AI 推理任务正在执行。")
            return
        if self.total_frames <= 0:
            QMessageBox.warning(self, "无法批量标注", "当前视频无法获取有效帧数。")
            return

        step = max(1, int(self.frame_step))
        sample_count = (self.total_frames + step - 1) // step
        first_samples = ", ".join(str(i * step + 1) for i in range(min(5, sample_count)))
        suffix = "……" if sample_count > 5 else ""
        answer = QMessageBox.question(
            self,
            "批量 AI 预标注",
            f"将按 {step} 帧的采样间隔处理当前视频，预计处理 {sample_count} 帧。\n"
            f"采样帧序号（从 1 开始）：{first_samples}{suffix}\n\n"
            f"输入尺寸：{', '.join(map(str, self._active_imgsz_values()))}\n"
            f"多分辨率推理：{'启用' if self.multiscale_enabled else '未启用'}\n"
            f"最小目标边长：{self.inference_min_box_side}px（0 表示关闭）\n"
            f"检测结果将写入当前文件的未保存标注缓存，请复核后再保存。是否开始？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return

        self.batch_added_count = 0
        self.batch_error_count = 0
        self.batch_skipped_count = 0
        self._batch_stopped_by_user = False
        self.predicting = True
        self.status_lbl.setText(
            f"批量 AI 预标注已开始：采样间隔 {step} 帧，预计处理 {sample_count} 帧；处理期间请勿切换文件。"
        )

        self.batch_thread = BatchPredictThread(
            self.ai_model,
            str(self.current_file),
            self.total_frames,
            step,
            self.inference_conf,
            self.inference_iou,
            self._active_imgsz_values(),
            self.inference_max_det,
            self.inference_min_box_side,
        )
        self.batch_thread.frame_result.connect(self._on_batch_frame_result)
        self.batch_thread.progress_signal.connect(self._on_batch_progress)
        self.batch_thread.error_signal.connect(self._on_batch_error)
        self.batch_thread.completed_signal.connect(self._on_batch_completed)
        self.batch_thread.finished.connect(self._on_batch_thread_stopped)
        self.batch_thread.start()
        self._sync_ui_state()

    def _on_batch_frame_result(
        self, frame_index: int, new_boxes: list, image_w: int, image_h: int
    ) -> None:
        # 批量处理前先载入该帧磁盘上已有的标签，避免覆盖之前的人工标注。
        if frame_index not in self.annotations:
            existing_label = self._existing_video_frame_label_path(frame_index)
            if existing_label is not None:
                self._load_yolo_txt(
                    existing_label, frame_index, image_w=image_w, image_h=image_h
                )
        added, skipped = self._apply_ai_boxes(
            frame_index, new_boxes, ensure_empty_frame=True
        )
        self.batch_added_count += added
        self.batch_skipped_count += skipped

    def _on_batch_progress(self, current_frame: int, total_frames: int) -> None:
        self.status_lbl.setText(
            f"批量 AI 预标注进行中：已扫描至第 {current_frame} / {total_frames} 帧；"
            f"新增 {self.batch_added_count} 个检测框。"
        )

    def _on_batch_error(self, frame_index: int, error: str) -> None:
        self.batch_error_count += 1
        frame_text = f"第 {frame_index + 1} 帧" if frame_index >= 0 else "视频读取"
        self.status_lbl.setText(f"批量 AI 预标注遇到错误（{frame_text}）：{error}")

    def _on_batch_completed(
        self, sampled: int, failed: int, stopped: bool, fatal_error: bool
    ) -> None:
        self._batch_stopped_by_user = stopped
        self.batch_error_count = max(self.batch_error_count, failed)
        if fatal_error:
            state_text = "因推理错误提前终止（请检查显存；必要时降低推理输入尺寸）"
        else:
            state_text = "已停止" if stopped else "完成"
        result_text = (
            f"批量 AI 预标注{state_text}：处理 {sampled} 个采样帧，"
            f"新增 {self.batch_added_count} 个检测框，失败 {failed} 帧，"
            f"因类别名称不匹配跳过 {self.batch_skipped_count} 个检测框。"
        )
        if self.dirty:
            result_text += " 请检查结果并点击顶部“保存标注”。"
        self.status_lbl.setText(result_text)

    def _on_batch_thread_stopped(self) -> None:
        self.predicting = False
        self.batch_thread = None
        self.batch_stop_action.setEnabled(False)
        self._sync_ui_state()

    def _stop_batch_ai_predict(self) -> None:
        if self.batch_thread is not None and self.batch_thread.isRunning():
            self.batch_thread.request_stop()
            self.batch_stop_action.setEnabled(False)
            self.status_lbl.setText("正在停止批量 AI 预标注，等待当前帧推理结束……")

    def _on_ai_error(self, file_path: str, frame_index: int, error: str) -> None:
        if self.current_file is not None and str(self.current_file) == file_path:
            self.status_lbl.setText(f"AI 预标注失败：{error}")
            QMessageBox.critical(self, "AI 预标注失败", error)

    def _on_ai_thread_stopped(self) -> None:
        self.predicting = False
        self.predict_thread = None
        self._sync_ui_state()

    # ---------- 清理标注 ----------

    def _clear_current_frame_annotations(self) -> None:
        """清空当前图片/视频帧全部标注，并将其作为待保存的空标签记录。"""
        if self.current_file is None or self.frame is None or self.predicting:
            return

        frame_key = self.frame_idx if self.is_video else 0
        existing_boxes = self.annotations.get(frame_key, [])
        box_count = len(existing_boxes)
        frame_description = (
            f"视频第 {frame_key + 1} 帧" if self.is_video else self.current_file.name
        )
        answer = QMessageBox.question(
            self,
            "清空当前画面标注",
            f"确定清空 {frame_description} 中的全部 {box_count} 个标注框吗？\n\n"
            "此操作不会删除原始图片或视频。点击“保存标注”后，会将该画面的 YOLO TXT "
            "写成空文件，表示该画面没有目标。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        # 保留空列表作为显式负样本记录；保存时会写出空 txt。
        self.annotations[frame_key] = []
        self._set_dirty(True)
        self._refresh_frame()
        self._sync_ui_state()
        self.status_lbl.setText(
            f"已清空 {frame_description} 的全部标注。请点击“保存标注”写入空标签文件。"
        )

    def _clear_unsaved_annotation_cache(self) -> None:
        """丢弃当前文件全部未保存缓存，并从磁盘恢复当前画面的已保存标签。"""
        if self.current_file is None or self.frame is None or self.predicting:
            return
        if not self.dirty:
            self.status_lbl.setText("当前文件没有未保存的标注修改；磁盘上的标签保持不变。")
            return

        cached_frame_count = len(self.annotations)
        cached_box_count = sum(len(boxes) for boxes in self.annotations.values())
        answer = QMessageBox.question(
            self,
            "清空未保存标注缓存",
            f"当前文件的内存缓存中有 {cached_frame_count} 帧、{cached_box_count} 个标注框。\n\n"
            "继续后会丢弃当前文件全部未保存的修改，而不是只清除当前帧。"
            "磁盘上已经保存的标签文件不会删除；当前画面会重新读取磁盘上的已有标签，"
            "视频其他帧在再次访问时按需读取。\n\n是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        current_frame_index = self.frame_idx
        self.annotations.clear()
        self._set_dirty(False)

        # 缓存清空后，重新从磁盘读取当前帧的持久化标签；其他视频帧会在跳转时按需加载。
        if self.is_video:
            self._seek(current_frame_index)
        else:
            label_path = self._existing_image_label_path(self.current_file)
            if label_path is not None:
                self._load_yolo_txt(label_path, 0)
            self._refresh_frame()

        self._sync_ui_state()
        self.status_lbl.setText(
            f"已清空当前文件全部未保存缓存（原缓存 {cached_frame_count} 帧、"
            f"{cached_box_count} 个框）。磁盘上已保存的标签未删除。"
        )

    # ---------- 保存 ----------

    def _save(self) -> bool:
        if self.current_file is None or self.current_labels_dir is None:
            return False
        if self.frame is None:
            QMessageBox.warning(self, "无法保存", "当前没有有效画面。")
            return False
        if not self.annotations:
            self.status_lbl.setText(
                "当前没有标注缓存可保存。若要将当前画面保存为空标签，请先使用“清空当前画面标注”。"
            )
            return False
        if not self.is_video and 0 not in self.annotations:
            self.status_lbl.setText("当前图片没有标注缓存可保存。")
            return False

        try:
            if self.is_video:
                if self.cap is None:
                    raise RuntimeError("视频对象已关闭，请重新打开视频。")

                restore_index = self.frame_idx
                for frame_index in sorted(self.annotations):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"读取视频第 {frame_index + 1} 帧失败，保存中止。")

                    image_path = self._video_frame_image_path(frame_index)
                    label_path = self._video_frame_label_path(frame_index)
                    if not save_image_unicode(image_path, frame):
                        raise OSError(f"视频帧图片保存失败：{image_path}")
                    height, width = frame.shape[:2]
                    write_yolo_label(
                        label_path,
                        self.annotations[frame_index],
                        width,
                        height,
                    )

                self._seek(restore_index)
                saved_info = (
                    f"已保存 {len(self.annotations)} 帧，图片目录：{self.frame_images_dir}，"
                    f"标签目录：{self.current_labels_dir}"
                )
            else:
                boxes = self.annotations[0]
                label_path = self._image_label_path(self.current_file)
                write_yolo_label(label_path, boxes, self.image_w, self.image_h)
                saved_info = f"已保存：{label_path}"

            if self.classes and self.classes_file is not None:
                self.classes_file.parent.mkdir(parents=True, exist_ok=True)
                self.classes_file.write_text(
                    "\n".join(self.classes) + "\n", encoding="utf-8"
                )

            self._set_dirty(False)
            self.status_lbl.setText(saved_info)
            return True
        except Exception as exc:
            QMessageBox.critical(
                self,
                "保存失败",
                f"标注没有全部保存成功：\n{type(exc).__name__}: {exc}",
            )
            self.status_lbl.setText(f"保存失败：{exc}")
            return False

    # ---------- 窗口状态与退出 ----------

    def _set_dirty(self, value: bool) -> None:
        self.dirty = bool(value)
        title = f" - {self.current_file.name}" if self.current_file else ""
        marker = " *" if self.dirty else ""
        self.setWindowTitle(f"YOLO 图像与视频标注工具{title}{marker}")
        # 缓存清理按钮只在当前文件确实存在未保存修改时启用。
        if hasattr(self, "clear_cache_action"):
            has_file = self.current_file is not None and self.frame is not None
            self.clear_cache_action.setEnabled(
                has_file and self.dirty and not self.predicting
            )

    def _confirm_unsaved_changes(self) -> bool:
        if not self.dirty:
            return True

        box = QMessageBox(self)
        box.setWindowTitle("存在未保存的标注")
        box.setText("当前文件有未保存的修改。是否先保存？")
        save_button = box.addButton("保存", QMessageBox.AcceptRole)
        discard_button = box.addButton("放弃修改", QMessageBox.DestructiveRole)
        cancel_button = box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(save_button)
        box.exec_()

        clicked = box.clickedButton()
        if clicked is save_button:
            return self._save()
        if clicked is discard_button:
            self._set_dirty(False)
            return True
        return False

    def _stop_playback(self) -> None:
        # 当前版本采用逐帧手动查看，不启用自动播放；保留此接口供画布事件调用。
        return

    def _sync_ui_state(self) -> None:
        has_file = self.current_file is not None and self.frame is not None
        has_video = has_file and self.is_video
        self.open_action.setEnabled(not self.predicting)
        self.output_dir_action.setEnabled(
            self.media_dir is not None and not self.predicting
        )
        self.model_action.setEnabled(not self.predicting)
        self.model_settings_action.setEnabled(not self.predicting)
        self.save_action.setEnabled(has_file and not self.predicting)
        self.ai_action.setEnabled(has_file and self.ai_model is not None and not self.predicting)
        self.batch_ai_action.setEnabled(
            has_video and self.ai_model is not None and not self.predicting
        )
        self.batch_stop_action.setEnabled(
            self.batch_thread is not None and self.batch_thread.isRunning()
        )
        # 标注帧可能离当前帧很远，因此不能再按帧步长或相邻帧边界判断；
        # 是否存在目标由点击时的标注索引扫描决定。
        self.prev_frame_action.setEnabled(has_video and not self.predicting)
        self.next_frame_action.setEnabled(has_video and not self.predicting)
        self.delete_action.setEnabled(
            self._selected_box_index() >= 0 and not self.predicting
        )
        self.clear_current_action.setEnabled(has_file and not self.predicting)
        self.clear_cache_action.setEnabled(
            has_file and self.dirty and not self.predicting
        )
        self.file_list_widget.setEnabled(not self.predicting and bool(self.file_list))

    def closeEvent(self, event) -> None:
        if not self._confirm_unsaved_changes():
            event.ignore()
            return
        if self.batch_thread is not None and self.batch_thread.isRunning():
            self.batch_thread.request_stop()
            self.batch_thread.wait()
        if self.predict_thread is not None and self.predict_thread.isRunning():
            self.predict_thread.wait()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = AnnotatorWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
