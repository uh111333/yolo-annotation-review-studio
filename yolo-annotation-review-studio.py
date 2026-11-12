# -*- coding: utf-8 -*-
"""YOLO 图像与视频标注复核工具。

依赖：Python 3.9+、PyQt5、opencv-python、numpy；AI 预标注另需 ultralytics。
"""

from __future__ import annotations

import os
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QRectF, Qt, QThread, QSettings, QTimer, pyqtSignal
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
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QTextEdit,
    QToolBar,
    QToolButton,
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


def _natural_sort_key(value: str) -> tuple:
    """生成稳定的自然排序键，使文件名中的数字按数值排序。"""
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", str(value))
    )


def group_image_sequences(entries: list[Path]) -> tuple[list[Path], dict[Path, list[Path]]]:
    """按同目录、去除文件主体全部数字后的名称建立图片序列。

    扩展名不参与匹配；仅图片文件参与分组，视频和其他媒体保持独立。
    返回文件列表中的代表路径，以及每个组员路径到完整序列的映射。
    """
    groups: dict[tuple[Path, str], list[Path]] = {}
    for path in entries:
        if path.suffix.casefold() not in IMAGE_SUFFIXES:
            continue
        key = re.sub(r"\d", "", path.stem).casefold()
        groups.setdefault((path.parent.resolve(), key), []).append(path)

    sequence_lookup: dict[Path, list[Path]] = {}
    grouped_members: set[Path] = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        ordered = sorted(
            members,
            key=lambda item: (
                _natural_sort_key(item.stem),
                item.suffix.casefold(),
                item.name.casefold(),
            ),
        )
        for member in ordered:
            resolved = member.resolve()
            sequence_lookup[resolved] = ordered
            grouped_members.add(resolved)

    representatives = [
        path for path in entries
        if path.resolve() not in grouped_members
        or sequence_lookup[path.resolve()][0].resolve() == path.resolve()
    ]
    return representatives, sequence_lookup



def _directory_contains_label_txt(folder: Path) -> bool:
    """判断目录树中是否存在 TXT 标签文件；遇到嵌套数据集时不跨入其 images/。"""
    try:
        for current, directories, filenames in os.walk(folder, topdown=True, followlinks=False):
            current_path = Path(current)
            if (current_path / "images").is_dir():
                directories[:] = []
                continue
            directories[:] = [
                name for name in directories
                if name.casefold() not in {
                    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv",
                    "node_modules", ".idea", ".vscode",
                }
            ]
            if any(Path(name).suffix.lower() == ".txt" for name in filenames):
                return True
    except OSError:
        return False
    return False


def _annotation_subdirectories(dataset_root: Path) -> list[Path]:
    """列出数据集根目录中可能存放 YOLO TXT 标签的直接子目录。

    空的 labels/ 可作为数据集结构标识；其他名称的目录需至少包含一个 TXT 文件。
    这只用于识别数据集，不影响标签查找时的匹配顺序。
    """
    result = []
    try:
        children = sorted(
            (item for item in Path(dataset_root).iterdir() if item.is_dir()),
            key=lambda item: str(item).casefold(),
        )
    except OSError:
        return result

    for child in children:
        if child.name.casefold() == "images":
            continue
        if child.name.casefold() == "labels" or _directory_contains_label_txt(child):
            result.append(child)
    return result


def resolve_dataset_layout(
    selected_dir: Path,
) -> tuple[Path, Path, Path, Path, Path, str]:
    """根据 YOLO 数据集结构确定媒体扫描范围和默认标签/抽帧目录。

    支持两种明确的数据集状态：
    1. images/ + 标签目录，且数据集根目录直接包含视频：按视频复核模式打开。
       images/ 是程序保存的视频帧图像目录，不作为独立图片导入。
    2. images/ + 标签目录且没有根目录视频：按图片数据集模式打开 images/。
       图片和视频帧标签均只从当前输出根目录中查找。

    其余既有目录形式保持原有处理方式。
    返回：媒体扫描目录、数据集根目录、标签相对目录、视频帧图像相对目录、
    类别文件路径、布局模式。
    """
    selected_dir = Path(selected_dir)

    def has_direct_video_files(folder: Path) -> bool:
        try:
            return any(
                item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES
                for item in folder.iterdir()
            )
        except OSError:
            return False

    if selected_dir.name.lower() == "images":
        dataset_root = selected_dir.parent
        # 即使用户直接选择 images/，只要同级存在标签目录和根目录视频，
        # 也切换到视频复核模式，避免把视频抽帧图像当成独立图片导入。
        if _annotation_subdirectories(dataset_root) and has_direct_video_files(dataset_root):
            media_dir = dataset_root
            layout_mode = "video_dataset"
        else:
            media_dir = selected_dir
            layout_mode = "image_dataset"
        labels_relative_dir = Path("labels")
        frame_images_relative_dir = Path("images")
        source_classes_file = dataset_root / "classes.txt"
    elif selected_dir.parent.name.lower() == "images":
        media_dir = selected_dir
        dataset_root = selected_dir.parent.parent
        labels_relative_dir = Path("labels") / selected_dir.name
        frame_images_relative_dir = Path("images") / selected_dir.name
        source_classes_file = dataset_root / "classes.txt"
        layout_mode = "image_split"
    elif (selected_dir / "images").is_dir() and _annotation_subdirectories(selected_dir):
        dataset_root = selected_dir
        labels_relative_dir = Path("labels")
        frame_images_relative_dir = Path("images")
        source_classes_file = dataset_root / "classes.txt"
        if has_direct_video_files(selected_dir):
            # 视频与 images/、标签目录同处数据集根目录：只导入视频，
            # 每次定位视频帧时由 _seek() 读取对应的逐帧标签。
            media_dir = selected_dir
            layout_mode = "video_dataset"
        else:
            # 没有视频时，images/ 是待标注的原始图片目录。
            media_dir = selected_dir / "images"
            layout_mode = "image_dataset"
    elif (selected_dir / "images").is_dir() and not any(
        p.is_file() and p.suffix.lower() in ALL_SUFFIXES
        for p in selected_dir.iterdir()
    ):
        media_dir = selected_dir / "images"
        dataset_root = selected_dir
        labels_relative_dir = Path("labels")
        frame_images_relative_dir = Path("images")
        source_classes_file = dataset_root / "classes.txt"
        layout_mode = "image_dataset"
    else:
        media_dir = selected_dir
        dataset_root = selected_dir
        labels_relative_dir = Path("labels")
        # 视频抽帧继续保存在数据集根目录的 images/ 子目录。
        frame_images_relative_dir = Path("images")
        source_classes_file = dataset_root / "classes.txt"
        layout_mode = "generic"

    return (
        media_dir,
        dataset_root,
        labels_relative_dir,
        frame_images_relative_dir,
        source_classes_file,
        layout_mode,
    )


def discover_yolo_dataset_roots(selected_dir: Path) -> list[Path]:
    """递归查找包含 images/ 与 YOLO TXT 标签目录的数据集根目录。

    标签目录可以命名为 labels/ 或任意其他名称。扫描时不会进入已识别数据集的
    images/ 与标签目录，也不会把嵌套数据集的标签误识别为父数据集的标签。
    """
    selected_dir = Path(selected_dir).resolve()
    if selected_dir.name.lower() == "images":
        dataset_root = selected_dir.parent
        if (dataset_root / "images").is_dir() and _annotation_subdirectories(dataset_root):
            return [dataset_root]
        scan_root = dataset_root
    elif selected_dir.parent.name.lower() == "images":
        dataset_root = selected_dir.parent.parent
        if (dataset_root / "images").is_dir() and _annotation_subdirectories(dataset_root):
            return [dataset_root]
        scan_root = dataset_root
    else:
        scan_root = selected_dir

    if not scan_root.is_dir():
        return []

    roots: list[Path] = []
    for current, dirs, _files in os.walk(scan_root, topdown=True, followlinks=False):
        current_path = Path(current)
        if not (current_path / "images").is_dir():
            continue

        annotation_dirs = _annotation_subdirectories(current_path)
        if not annotation_dirs:
            continue

        roots.append(current_path)
        excluded = {"images"}
        excluded.update(item.name.casefold() for item in annotation_dirs)
        dirs[:] = [name for name in dirs if name.casefold() not in excluded]

    return sorted(set(roots), key=lambda item: (len(item.parts), str(item).casefold()))

# 主模型沿用原有颜色序列；副模型使用独立且不重复的颜色序列，
# 两组颜色互不重合，避免相同类别编号在双模型验证画面中难以区分。
BOX_COLORS = [
    QColor(0, 255, 0),
    QColor(255, 90, 90),
    QColor(80, 160, 255),
    QColor(255, 255, 0),
    QColor(255, 0, 255),
    QColor(0, 255, 255),
]
SECONDARY_BOX_COLORS = [
    QColor(255, 128, 0),
    QColor(128, 255, 0),
    QColor(0, 255, 128),
    QColor(0, 0, 255),
    QColor(128, 0, 255),
    QColor(255, 0, 128),
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
    # 此底层写入函数仍支持空标签；主界面的保存流程会在无有效框时直接删除标签文件。
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


def filter_boxes_overlapping_reference(
    candidate_boxes: list[dict], reference_boxes: list[dict], iou_threshold: float
) -> list[dict]:
    """仅保留与任意参考框达到指定 IoU 的候选框；不比较类别。"""
    if not candidate_boxes or not reference_boxes:
        return []
    threshold = min(max(float(iou_threshold), 0.01), 1.0)
    return [
        candidate
        for candidate in candidate_boxes
        if any(box_iou(candidate, reference) >= threshold for reference in reference_boxes)
    ]


def normalize_batch_frame_range(
    total_frames: int, start_frame: int, end_frame: int
) -> tuple[int, int]:
    """校验批量推理帧范围，使用从 1 开始的闭区间；结束帧 0 表示视频末尾。"""
    total = int(total_frames)
    start = int(start_frame)
    end = int(end_frame)
    if total <= 0:
        raise ValueError("当前视频没有可用帧。")
    if start < 1 or start > total:
        raise ValueError(f"起始帧必须在 1 到 {total} 之间。")
    if end == 0:
        end = total
    elif end < 1 or end > total:
        raise ValueError(f"结束帧必须在 1 到 {total} 之间；填写 0 表示视频末尾。")
    if end < start:
        raise ValueError("结束帧不能早于起始帧。")
    return start, end


def batch_sample_frame_numbers(start_frame: int, end_frame: int, frame_step: int) -> list[int]:
    """返回批量推理实际采样的帧序号（从 1 开始，起始帧作为首个采样点）。"""
    start = int(start_frame)
    end = int(end_frame)
    step = max(1, int(frame_step))
    if start < 1 or end < start:
        raise ValueError("批量处理帧范围无效。")
    return list(range(start, end + 1, step))


@dataclass
class ORBFeatureSet:
    keypoints: list
    descriptors: Optional[np.ndarray]
    texture_std: float


@dataclass
class ORBPairAnalysis:
    valid: bool
    overlap_percent: Optional[float] = None


def prepare_orb_frame(frame: np.ndarray, max_side: int = 640) -> np.ndarray:
    """缩小并转换为 uint8 灰度图，供 ORB 特征提取。"""
    if frame is None or frame.size == 0:
        raise ValueError("无法分析空视频帧。")
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("视频帧通道格式不受支持。")

    height, width = gray.shape[:2]
    if height < 2 or width < 2:
        raise ValueError("视频帧尺寸过小，无法提取 ORB 特征。")
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale < 1.0:
        target_size = (
            max(2, int(round(width * scale))),
            max(2, int(round(height * scale))),
        )
        gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(gray, dtype=np.uint8)


def extract_orb_features(gray: np.ndarray, orb: cv2.ORB) -> ORBFeatureSet:
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return ORBFeatureSet(
        keypoints=keypoints or [],
        descriptors=descriptors,
        texture_std=float(np.std(gray)),
    )


def analyze_orb_pair(
    reference: Optional[ORBFeatureSet],
    current: Optional[ORBFeatureSet],
    image_shape: tuple[int, int],
    matcher: cv2.BFMatcher,
) -> ORBPairAnalysis:
    """用 ORB 匹配和 RANSAC 单应性估算两帧视野交叠比例。"""
    if reference is None or current is None:
        return ORBPairAnalysis(False)

    ref_count = len(reference.keypoints)
    cur_count = len(current.keypoints)
    if reference.texture_std < 3.0 or current.texture_std < 3.0:
        return ORBPairAnalysis(False)
    if ref_count < 30 or cur_count < 30:
        return ORBPairAnalysis(False)
    if reference.descriptors is None or current.descriptors is None:
        return ORBPairAnalysis(False)

    try:
        pairs = matcher.knnMatch(reference.descriptors, current.descriptors, k=2)
        candidates = [
            pair[0]
            for pair in pairs
            if len(pair) >= 2 and pair[0].distance < 0.75 * pair[1].distance
        ]
        # 同一当前帧特征只保留距离最小的一条匹配，降低重复匹配对模型的干扰。
        candidates.sort(key=lambda item: item.distance)
        matches = []
        used_train_indices = set()
        for match in candidates:
            if match.trainIdx in used_train_indices:
                continue
            used_train_indices.add(match.trainIdx)
            matches.append(match)
    except (cv2.error, TypeError, ValueError):
        return ORBPairAnalysis(False)

    if len(matches) < 12:
        return ORBPairAnalysis(False)

    source_points = np.float32(
        [reference.keypoints[item.queryIdx].pt for item in matches]
    ).reshape(-1, 1, 2)
    target_points = np.float32(
        [current.keypoints[item.trainIdx].pt for item in matches]
    ).reshape(-1, 1, 2)
    try:
        homography, mask = cv2.findHomography(
            source_points, target_points, cv2.RANSAC, 3.0
        )
    except cv2.error:
        return ORBPairAnalysis(False)
    if homography is None or mask is None or not np.isfinite(homography).all():
        return ORBPairAnalysis(False)

    inlier_mask = mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_ratio = inlier_count / float(max(1, len(matches)))
    if inlier_count:
        inlier_source = source_points.reshape(-1, 2)[inlier_mask]
        inlier_target = target_points.reshape(-1, 2)[inlier_mask]
        try:
            projected = cv2.perspectiveTransform(
                inlier_source.reshape(-1, 1, 2), homography
            ).reshape(-1, 2)
        except cv2.error:
            return ORBPairAnalysis(False)
        median_error = float(np.median(np.linalg.norm(projected - inlier_target, axis=1)))
    else:
        median_error = None

    if inlier_count < 10 or inlier_ratio < 0.25:
        return ORBPairAnalysis(False)
    if median_error is None or not np.isfinite(median_error) or median_error > 4.0:
        return ORBPairAnalysis(False)

    height, width = image_shape
    reference_corners = np.float32([
        [0.0, 0.0], [float(width), 0.0],
        [float(width), float(height)], [0.0, float(height)],
    ]).reshape(-1, 1, 2)
    try:
        projected_corners = cv2.perspectiveTransform(
            reference_corners, homography
        ).reshape(-1, 2)
    except cv2.error:
        return ORBPairAnalysis(False)
    if not np.isfinite(projected_corners).all():
        return ORBPairAnalysis(False)
    if float(np.max(np.abs(projected_corners))) > max(width, height) * 20.0:
        return ORBPairAnalysis(False)

    projected_contour = projected_corners.astype(np.float32).reshape(-1, 1, 2)
    projected_area = abs(float(cv2.contourArea(projected_contour)))
    frame_area = float(width * height)
    if projected_area < frame_area * 0.02 or projected_area > frame_area * 100.0:
        return ORBPairAnalysis(False)
    if not cv2.isContourConvex(projected_contour):
        return ORBPairAnalysis(False)

    current_corners = np.float32([
        [0.0, 0.0], [float(width), 0.0],
        [float(width), float(height)], [0.0, float(height)],
    ])
    try:
        intersection_area, _ = cv2.intersectConvexConvex(
            projected_corners.astype(np.float32), current_corners
        )
    except cv2.error:
        return ORBPairAnalysis(False)
    if not np.isfinite(float(intersection_area)):
        return ORBPairAnalysis(False)

    overlap = min(100.0, max(0.0, float(intersection_area) / frame_area * 100.0))
    return ORBPairAnalysis(True, overlap_percent=overlap)


def adaptive_orb_detection_interval(
    overlap_percent: float,
    previous_overlap: Optional[float],
    previous_frame: Optional[int],
    current_frame: int,
    threshold: float,
    base_step: int,
) -> int:
    """根据重叠率下降速度调整检测间隔；默认基准 5 帧，对应 2～10 帧。"""
    base = max(1, int(base_step))
    minimum = max(1, base // 2)
    maximum = max(minimum, base * 2)
    if overlap_percent <= threshold + 5.0:
        return minimum
    if previous_overlap is None or previous_frame is None:
        return base

    elapsed = max(1, int(current_frame) - int(previous_frame))
    decline_rate = (float(previous_overlap) - float(overlap_percent)) / elapsed
    if decline_rate <= 0.02:
        return maximum
    frames_to_threshold = max(0.0, (float(overlap_percent) - float(threshold)) / decline_rate)
    predicted_interval = int(frames_to_threshold * 0.5)
    return min(maximum, max(minimum, predicted_interval))



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
        secondary_model=None,
        dual_model_mode: str = "all",
        dual_model_overlap_iou: float = 0.30,
    ):
        super().__init__()
        self.model = model
        self.secondary_model = secondary_model
        self.dual_model_mode = str(dual_model_mode)
        self.dual_model_overlap_iou = float(dual_model_overlap_iou)
        self.secondary_warning: Optional[str] = None
        self.image = image.copy()
        self.file_path = file_path
        self.frame_index = frame_index
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz_values = tuple(int(v) for v in imgsz_values)
        self.max_det = int(max_det)
        self.min_box_side = max(0, int(min_box_side))

    def _infer(self, model) -> list[dict]:
        return infer_yolo_multiscale(
            model,
            self.image,
            self.conf,
            self.iou,
            self.max_det,
            self.imgsz_values,
            self.min_box_side,
        )

    def run(self) -> None:
        try:
            boxes = self._infer(self.model)
            if self.secondary_model is not None:
                # 同一工作线程内顺序调用两个模型，避免 GPU 并发推理和额外任务队列。
                for detection in boxes:
                    detection["model_index"] = 1
                try:
                    secondary_boxes = self._infer(self.secondary_model)
                    for detection in secondary_boxes:
                        detection["model_index"] = 2
                    if self.dual_model_mode == "overlap":
                        secondary_boxes = filter_boxes_overlapping_reference(
                            secondary_boxes, boxes, self.dual_model_overlap_iou
                        )
                    boxes.extend(secondary_boxes)
                except Exception as exc:
                    # 副模型失败不影响主模型本轮结果。
                    self.secondary_warning = f"{type(exc).__name__}: {exc}"
            self.finished_signal.emit(self.file_path, self.frame_index, boxes)
        except Exception as exc:
            self.error_signal.emit(
                self.file_path, self.frame_index, f"{type(exc).__name__}: {exc}"
            )


class CameraCaptureThread(QThread):
    """独占摄像头设备并持续采集；只保留最新帧，避免 Qt 信号队列积压。"""

    error_signal = pyqtSignal(str)

    def __init__(self, device_index: int = 0):
        super().__init__()
        self.device_index = int(device_index)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._frame_event = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._sequence = 0

    def run(self) -> None:
        capture = cv2.VideoCapture(self.device_index)
        if not capture.isOpened():
            capture.release()
            self.error_signal.emit(
                f"无法打开本机摄像头设备 {self.device_index}。请检查设备连接、权限或占用情况。"
            )
            return
        try:
            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    if not self._stop_event.is_set():
                        self.error_signal.emit("摄像头读取失败或设备已断开。")
                    break
                with self._lock:
                    self._latest_frame = frame
                    self._sequence += 1
                self._frame_event.set()
        finally:
            # 唤醒等待中的批量处理线程，使其能够感知采集线程已经结束。
            self._frame_event.set()
            capture.release()

    def request_stop(self) -> None:
        self._stop_event.set()
        self._frame_event.set()

    def latest_frame(
        self, copy_frame: bool = True
    ) -> tuple[int, Optional[np.ndarray]]:
        with self._lock:
            if self._latest_frame is None:
                return self._sequence, None
            frame = self._latest_frame.copy() if copy_frame else self._latest_frame
            return self._sequence, frame

    def wait_for_new_frame(
        self, after_sequence: int, timeout: float = 0.2
    ) -> tuple[int, Optional[np.ndarray]]:
        """等待序号更新；中间帧允许被最新帧覆盖，不建立积压队列。"""
        self._frame_event.clear()
        sequence, frame = self.latest_frame(copy_frame=False)
        if frame is not None and sequence > int(after_sequence):
            return sequence, frame
        self._frame_event.wait(max(0.0, float(timeout)))
        return self.latest_frame(copy_frame=False)


class BatchPredictThread(QThread):
    """统一处理视频、图片组或摄像头；ORB 决策与 YOLO 推理共用。"""

    frame_result = pyqtSignal(int, object, int, int, object)
    progress_signal = pyqtSignal(int, int, object)
    error_signal = pyqtSignal(int, str)
    completed_signal = pyqtSignal(int, int, bool, bool, int, int)

    def __init__(
        self,
        model,
        video_path: Optional[str],
        total_frames: int,
        frame_step: int,
        conf: float,
        iou: float,
        imgsz_values: tuple[int, ...],
        max_det: int,
        min_box_side: int = 0,
        start_frame: int = 1,
        end_frame: Optional[int] = None,
        overlap_threshold: float = 70.0,
        orb_nfeatures: int = 2000,
        orb_max_side: int = 640,
        smart_base_step: int = 5,
        image_paths: Optional[list[Path]] = None,
        camera_thread: Optional[CameraCaptureThread] = None,
    ):
        super().__init__()
        self.model = model
        self.video_path = str(video_path) if video_path else None
        self.image_paths = [Path(item) for item in image_paths] if image_paths else None
        self.is_image_sequence = self.image_paths is not None
        self.camera_thread = camera_thread
        self.is_camera_source = camera_thread is not None
        self.total_frames = int(total_frames)
        if self.is_image_sequence and self.total_frames != len(self.image_paths):
            raise ValueError("图片组数量与批量处理总数不一致。")
        if self.is_camera_source and (self.is_image_sequence or self.video_path):
            raise ValueError("摄像头批量处理不能同时指定视频或图片组输入。")
        if not self.is_camera_source and not self.is_image_sequence and not self.video_path:
            raise ValueError("批量处理缺少有效的视频、图片组或摄像头输入。")
        self.start_frame = max(1, int(start_frame))
        self.end_frame = (
            int(end_frame) if end_frame is not None else self.total_frames
        )
        if not self.is_camera_source and (
            self.end_frame < self.start_frame or self.end_frame > self.total_frames
        ):
            raise ValueError("批量处理范围超出媒体长度或起止顺序无效。")
        self.range_total_frames = (
            0 if self.is_camera_source else self.end_frame - self.start_frame + 1
        )
        self.frame_step = max(1, int(frame_step))
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz_values = tuple(int(v) for v in imgsz_values)
        self.max_det = int(max_det)
        self.min_box_side = max(0, int(min_box_side))
        self.overlap_threshold = min(99.0, max(1.0, float(overlap_threshold)))
        self.orb_nfeatures = min(10000, max(0, int(orb_nfeatures)))
        self.orb_enabled = self.orb_nfeatures > 0
        self.orb_max_side = min(1920, max(320, int(orb_max_side)))
        # 图片组的每一张都进行 ORB 检测；视频仍采用自适应检测步长。
        self.smart_base_step = 1 if self.is_image_sequence else min(30, max(1, int(smart_base_step)))
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

    def _read_source_item(self, frame_index: int, cap) -> Optional[np.ndarray]:
        if self.is_image_sequence:
            return load_image_unicode(self.image_paths[frame_index])
        ok, frame = cap.read()
        return frame if ok and frame is not None else None

    def run(self) -> None:
        cap = None
        sampled = 0
        failed = 0
        stopped = False
        source_stopped = False        fatal_error = False
        scanned_frames = 0
        valid_comparisons = 0
        fallback_samples = 0
        latest_overlap_percent: Optional[float] = None
        try:
            if not self.is_image_sequence and not self.is_camera_source:
                if not self.video_path:
                    raise RuntimeError("未提供有效的视频路径。")
                cap = cv2.VideoCapture(self.video_path)
                if not cap.isOpened():
                    raise RuntimeError(f"无法打开视频：{self.video_path}")

            start_index = None if self.is_camera_source else self.start_frame - 1
            end_index = None if self.is_camera_source else self.end_frame - 1
            if not self.is_image_sequence and not self.is_camera_source and start_index > 0:
                # 起始帧只定位一次；进入范围后始终通过 read() 顺序解码。
                seek_ok = cap.set(cv2.CAP_PROP_POS_FRAMES, start_index)
                reported_position = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if (not seek_ok) or (
                    reported_position >= 0
                    and abs(reported_position - start_index) > 0.5
                ):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    skipped_frames = 0
                    while skipped_frames < start_index and not self._stop_event.is_set():
                        ok, _ = cap.read()
                        if not ok:
                            raise RuntimeError(
                                f"无法定位到起始帧 {self.start_frame}，视频可能提前结束。"
                            )
                        skipped_frames += 1
                    if self._stop_event.is_set():
                        stopped = True
                        return

            # 特征点数为 0 时完全关闭 ORB：视频/摄像头按固定间隔，图片组仍逐张推理。
            orb = cv2.ORB_create(nfeatures=self.orb_nfeatures) if self.orb_enabled else None
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) if self.orb_enabled else None
            reference_features: Optional[ORBFeatureSet] = None
            reference_analysis_shape: Optional[tuple[int, int]] = None
            next_detection_frame = start_index if start_index is not None else 0
            last_valid_overlap: Optional[float] = None
            last_valid_frame: Optional[int] = None
            fallback_anchor_frame = start_index if start_index is not None else 0
            min_detection_step = max(1, self.smart_base_step // 2)

            frame_index = (start_index - 1) if start_index is not None else -1
            last_camera_sequence = 0
            camera_start_index: Optional[int] = None
            while not self._stop_event.is_set():
                if self.is_camera_source:
                    assert self.camera_thread is not None
                    sequence, frame = self.camera_thread.wait_for_new_frame(
                        last_camera_sequence, timeout=0.2
                    )
                    if self._stop_event.is_set():
                        break
                    if sequence <= last_camera_sequence or frame is None:
                        if not self.camera_thread.isRunning():
                            source_stopped = True
                            if not self._stop_event.is_set():
                                failed += 1
                                self.error_signal.emit(-1, "摄像头采集已停止。")
                            break
                        continue
                    last_camera_sequence = sequence
                    frame_index = sequence - 1
                    if camera_start_index is None:
                        # 以任务启动后取得的第一张最新画面作为采样序列起点。
                        camera_start_index = frame_index
                        start_index = frame_index
                        next_detection_frame = frame_index
                        fallback_anchor_frame = frame_index
                    relative_index = frame_index - camera_start_index
                    # 摄像头进度显示实际采集序号；不制造虚假的总帧数。
                    scanned_frames = sequence
                else:
                    frame_index += 1
                    if frame_index > end_index:
                        break
                    frame = self._read_source_item(frame_index, cap)
                    relative_index = frame_index - start_index
                    scanned_frames = relative_index + 1

                if frame is None:
                    if not self.is_image_sequence and not self.is_camera_source:
                        # 视频读取失败通常代表到达实际结尾；保持原有行为，不计作推理错误。
                        break
                    if self.is_camera_source:
                        continue
                    failed += 1
                    source_name = self.image_paths[frame_index].name
                    self.error_signal.emit(frame_index, f"无法读取图片：{source_name}")
                    self.progress_signal.emit(
                        scanned_frames, self.range_total_frames, latest_overlap_percent
                    )
                    continue

                should_sample = False
                should_detect = False
                used_fallback = False
                yolo_success = False
                current_features: Optional[ORBFeatureSet] = None
                current_analysis_shape: Optional[tuple[int, int]] = None
                analysis: Optional[ORBPairAnalysis] = None

                if not self.orb_enabled:
                    # 图片组的图片均视为有效信息，不因固定间隔而跳过。
                    if self.is_image_sequence:
                        should_sample = True
                    elif self.is_camera_source:
                        should_sample = frame_index >= next_detection_frame
                    else:
                        should_sample = relative_index % self.frame_step == 0
                else:
                    should_detect = (
                        self.is_image_sequence or frame_index >= next_detection_frame
                    )
                    if should_detect:
                        try:
                            current_gray = prepare_orb_frame(frame, self.orb_max_side)
                            current_analysis_shape = current_gray.shape[:2]
                            current_features = extract_orb_features(current_gray, orb)
                        except (ValueError, cv2.error):
                            current_gray = None
                            current_features = None

                        if reference_features is None:
                            # 首张图片/视频帧没有比较基准，直接标注建立锚点。
                            should_sample = True
                        else:
                            if (
                                current_features is not None
                                and current_analysis_shape == reference_analysis_shape
                            ):
                                analysis = analyze_orb_pair(
                                    reference_features,
                                    current_features,
                                    current_analysis_shape,
                                    matcher,
                                )
                            else:
                                # 图片尺寸/比例变化时不强行估算重叠率，按无法配准处理。
                                analysis = ORBPairAnalysis(False)

                            if analysis.valid:
                                valid_comparisons += 1
                                fallback_anchor_frame = frame_index
                                latest_overlap_percent = analysis.overlap_percent
                                should_sample = analysis.overlap_percent < self.overlap_threshold
                                if self.is_image_sequence:
                                    next_interval = 1
                                else:
                                    next_interval = adaptive_orb_detection_interval(
                                        analysis.overlap_percent,
                                        last_valid_overlap,
                                        last_valid_frame,
                                        frame_index,
                                        self.overlap_threshold,
                                        self.smart_base_step,
                                    )
                                last_valid_overlap = analysis.overlap_percent
                                last_valid_frame = frame_index
                            elif self.is_image_sequence:
                                # 图片本身通常已经过抽帧/筛选；配准失败意味着无法证明它是重复画面，
                                # 因此当前图片立即进入 YOLO，不等待视频式的多帧兜底。
                                should_sample = True
                                used_fallback = True
                                next_interval = 1
                            else:
                                frames_since_valid = max(0, frame_index - fallback_anchor_frame)
                                if frames_since_valid >= self.frame_step:
                                    should_sample = True
                                    used_fallback = True
                                next_interval = min_detection_step

                        if should_sample:
                            next_interval = 1 if self.is_image_sequence else self.smart_base_step
                        else:
                            next_detection_frame = frame_index + max(
                                1, int(next_interval or self.smart_base_step)
                            )

                if should_sample:
                    if self._stop_event.is_set():
                        break
                    frame_h, frame_w = frame.shape[:2]
                    boxes = None
                    try:
                        boxes = self._infer(frame)
                        yolo_success = True
                    except Exception as exc:
                        failed += 1
                        error_text = f"{type(exc).__name__}: {exc}"
                        self.error_signal.emit(frame_index, error_text)
                        if "out of memory" in error_text.lower() or "cuda error: out of memory" in error_text.lower():
                            fatal_error = True
                        if self.orb_enabled:
                            # 推理失败时不切换参考帧，避免丢弃上一有效覆盖基准。
                            next_detection_frame = frame_index + (
                                1 if self.is_image_sequence else self.smart_base_step
                            )
                    # 摄像头只归档成功检测到目标的画面；空结果和推理失败均不生成图片/标签。
                    # 视频与图片组继续发送每次采样结果，维持原有预览及负样本处理行为。
                    if not self.is_camera_source or (yolo_success and bool(boxes)):
                        self.frame_result.emit(frame_index, boxes, frame_w, frame_h, frame)
                    sampled += 1

                    if self.orb_enabled:
                        if yolo_success and current_features is not None:
                            # 只在 YOLO 推理成功后更新参考帧；本帧 ORB 特征直接复用。
                            reference_features = current_features
                            reference_analysis_shape = current_analysis_shape
                            last_valid_overlap = None
                            last_valid_frame = None
                            fallback_anchor_frame = frame_index
                        if not yolo_success:
                            fallback_anchor_frame = frame_index
                        # 视频/摄像头采样后回到自适应步长；图片组下一张仍检测，不跳图。
                        next_detection_frame = frame_index + (
                            1 if self.is_image_sequence else self.smart_base_step
                        )
                        if used_fallback:
                            fallback_samples += 1
                    elif self.is_camera_source:
                        # 摄像头关闭 ORB 时，按实际采集序号计算固定间隔；推理期间跳过的帧不补算。
                        next_detection_frame = frame_index + self.frame_step

                if should_sample or (
                    self.orb_enabled
                    and (should_detect or (not self.is_camera_source and scanned_frames % 10 == 0))
                ):
                    self.progress_signal.emit(
                        frame_index + 1 if self.is_camera_source else scanned_frames,
                        self.range_total_frames,
                        latest_overlap_percent,
                    )
                elif self.is_image_sequence and scanned_frames % 10 == 0:
                    self.progress_signal.emit(
                        scanned_frames, self.range_total_frames, latest_overlap_percent
                    )
                if fatal_error:
                    break

            if scanned_frames:
                self.progress_signal.emit(
                    frame_index + 1 if self.is_camera_source else scanned_frames,
                    self.range_total_frames,
                    latest_overlap_percent,
                )
            stopped = self._stop_event.is_set() or source_stopped
        except Exception as exc:
            failed += 1
            self.error_signal.emit(-1, f"{type(exc).__name__}: {exc}")
        finally:
            if cap is not None:
                cap.release()
            self.completed_signal.emit(
                sampled, failed, stopped, fatal_error,
                valid_comparisons, fallback_samples,
            )


class BatchSamplingDialog(QDialog):
    """批量预标注参数；ORB 默认启用，特征点数为 0 时关闭。"""

    def __init__(
        self,
        start_frame: int,
        end_frame: int,
        total_frames: int,
        frame_step: int = 30,
        overlap_threshold: float = 70.0,
        orb_nfeatures: int = 2000,
        orb_max_side: int = 640,
        smart_base_step: int = 5,
        source_kind: str = "video",
        parent=None,
    ):
        super().__init__(parent)
        self.is_image_sequence = str(source_kind).lower() == "image_sequence"
        self.is_camera = str(source_kind).lower() == "camera"
        self.unit = "张" if self.is_image_sequence else "帧"
        self.setWindowTitle("摄像头自动预标注" if self.is_camera else "批量 AI 预标注")
        self.setModal(True)
        self.setMinimumWidth(400)

        self.total_frames = max(1, int(total_frames))
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(1, self.total_frames)
        self.start_frame_spin.setValue(min(max(1, int(start_frame)), self.total_frames))
        self.start_frame_spin.setToolTip(f"批量处理起始{self.unit}，按 1 开始计数。")

        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(1, self.total_frames)
        self.end_frame_spin.setValue(min(max(1, int(end_frame)), self.total_frames))
        self.end_frame_spin.setToolTip(f"批量处理结束{self.unit}，包含该{self.unit}。")

        self.range_summary = QLabel()
        self.range_summary.setStyleSheet("color: palette(mid);")
        self.start_frame_spin.valueChanged.connect(self._update_range_summary)
        self.end_frame_spin.valueChanged.connect(self._update_range_summary)

        range_form = QFormLayout()
        range_form.addRow(f"起始{self.unit}", self.start_frame_spin)
        range_form.addRow(f"结束{self.unit}（包含）", self.end_frame_spin)
        range_form.addRow(self.range_summary)
        self.range_group = QGroupBox("处理范围")
        self.range_group.setLayout(range_form)
        if self.is_camera:
            self.range_group.hide()

        self.frame_step_spin = QSpinBox()
        self.frame_step_spin.setRange(1, 100000)
        self.frame_step_spin.setSingleStep(1)
        self.frame_step_spin.setValue(min(100000, max(1, int(frame_step))))
        self.frame_step_spin.setToolTip(
            "ORB 关闭时作为固定采样间隔；ORB 配准失败时作为兜底间隔。"
        )
        self.frame_step_label = QLabel()

        self.overlap_threshold_spin = QDoubleSpinBox()
        self.overlap_threshold_spin.setRange(5.0, 99.0)
        self.overlap_threshold_spin.setDecimals(0)
        self.overlap_threshold_spin.setSingleStep(5.0)
        self.overlap_threshold_spin.setSuffix(" %")
        self.overlap_threshold_spin.setValue(min(99.0, max(5.0, float(overlap_threshold))))

        self.orb_nfeatures_spin = QSpinBox()
        self.orb_nfeatures_spin.setRange(0, 10000)
        self.orb_nfeatures_spin.setSingleStep(100)
        self.orb_nfeatures_spin.setValue(min(10000, max(0, int(orb_nfeatures))))
        self.orb_nfeatures_spin.setSpecialValueText("关闭 ORB")
        self.orb_nfeatures_spin.setToolTip(
            "设为 0 可关闭 ORB；视频和摄像头按固定间隔采样，图片组逐张标注。"
        )

        self.orb_max_side_spin = QSpinBox()
        self.orb_max_side_spin.setRange(320, 1920)
        self.orb_max_side_spin.setSingleStep(64)
        self.orb_max_side_spin.setSuffix(" px")
        self.orb_max_side_spin.setValue(min(1920, max(320, int(orb_max_side))))
        self.orb_max_side_spin.setToolTip("ORB 分析图的最长边尺寸。")

        self.smart_base_step_spin = QSpinBox()
        if self.is_image_sequence:
            self.smart_base_step_spin.setRange(1, 1)
            self.smart_base_step_spin.setValue(1)
        else:
            self.smart_base_step_spin.setRange(1, 30)
            self.smart_base_step_spin.setValue(min(30, max(1, int(smart_base_step))))
        self.smart_base_step_spin.setSingleStep(1)
        self.smart_base_step_spin.setSuffix(f" {self.unit}")
        self.smart_base_step_spin.setToolTip(
            "视频和摄像头共用自适应 ORB 检测步长；图片组固定为 1，逐张检测。"
        )

        smart_form = QFormLayout()
        smart_form.addRow("重叠率阈值", self.overlap_threshold_spin)
        smart_form.addRow("ORB 特征点数", self.orb_nfeatures_spin)
        smart_form.addRow("分析图最长边", self.orb_max_side_spin)
        self.smart_base_step_label = QLabel(
            "检测步长基准（图片组固定为 1）" if self.is_image_sequence else "检测步长基准"
        )
        smart_form.addRow(self.smart_base_step_label, self.smart_base_step_spin)
        self.smart_options_group = QGroupBox("智能采样参数")
        self.smart_options_group.setLayout(smart_form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(
            "开始自动预标注" if self.is_camera else "开始批量预标注"
        )
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.range_group)
        step_form = QFormLayout()
        step_form.addRow(self.frame_step_label, self.frame_step_spin)
        layout.addLayout(step_form)
        if self.is_image_sequence:
            self.frame_step_label.hide()
            self.frame_step_spin.hide()
        layout.addWidget(self.smart_options_group)
        layout.addWidget(buttons)
        self.orb_nfeatures_spin.valueChanged.connect(self._update_sampling_controls)
        self._update_sampling_controls()
        self._update_range_summary()

    def _update_range_summary(self, *_args) -> None:
        start = self.start_frame_spin.value()
        end = self.end_frame_spin.value()
        count = max(0, end - start + 1)
        self.range_summary.setText(f"共 {count} / {self.total_frames} {self.unit}")

    def _update_sampling_controls(self, *_args) -> None:
        orb_enabled = self.orb_nfeatures_spin.value() > 0
        self.overlap_threshold_spin.setEnabled(orb_enabled)
        self.orb_max_side_spin.setEnabled(orb_enabled)
        self.smart_base_step_spin.setEnabled(orb_enabled and not self.is_image_sequence)

        if self.is_image_sequence:
            self.frame_step_spin.setValue(1)
            self.frame_step_spin.setEnabled(False)
        else:
            self.frame_step_spin.setEnabled(True)
            self.frame_step_label.setText(
                "配准失败兜底间隔（帧）" if orb_enabled else "固定采样间隔（帧）"
            )

    def accept(self) -> None:
        start = self.start_frame_spin.value()
        end = self.end_frame_spin.value()
        if not (1 <= start <= end <= self.total_frames):
            QMessageBox.warning(
                self, "批量处理范围无效", "结束位置不能早于起始位置或超过媒体总数。"
            )
            self.end_frame_spin.setFocus()
            return
        super().accept()

    def values(self) -> dict:
        return {
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


class AnnotatorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Annotation Review Studio")
        self.resize(1450, 900)

        self.current_file: Optional[Path] = None
        self.is_video = False
        self.is_image_sequence = False
        self.current_image_sequence: list[Path] = []
        self.image_sequence_index = 0
        self.file_image_sequences: dict[Path, list[Path]] = {}
        self.image_sequence_sizes: dict[int, tuple[int, int]] = {}
        self.dirty_annotation_indices: set[int] = set()
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera = False
        self.camera_device_index = 0
        self.camera_thread: Optional[CameraCaptureThread] = None
        self._camera_thread_pending_delete: Optional[CameraCaptureThread] = None
        self.camera_last_sequence = -1
        self.camera_timer = QTimer(self)
        self.camera_timer.setInterval(30)
        self.camera_timer.timeout.connect(self._update_camera_frame)
        self.total_frames = 0
        self.frame_idx = 0
        self.frame: Optional[np.ndarray] = None
        self.image_w = 0
        self.image_h = 0

        # key 为视频帧序号或图片组内序号；独立图片仍使用 key=0。
        # 即使某帧没有框，也保留空列表，以便正确记录清空/负样本状态。
        self.annotations: dict[int, list[dict]] = {}
        self.dirty = False

        self.file_list: list[Path] = []
        self.file_index = -1
        self.media_dir: Optional[Path] = None
        # 递归导入多个数据集时，为每个媒体文件保留所属数据集上下文。
        self.browse_root: Optional[Path] = None
        self.file_dataset_keys: dict[Path, str] = {}
        self.dataset_contexts: dict[str, dict] = {}
        self.active_dataset_key: Optional[str] = None
        self.output_root: Optional[Path] = None
        self.default_output_root: Optional[Path] = None
        self.labels_relative_dir = Path("labels")
        self.frame_images_relative_dir = Path("images")
        self.current_labels_dir: Optional[Path] = None
        self.frame_images_dir: Optional[Path] = None
        self.classes_file: Optional[Path] = None

        self.classes: list[str] = []
        self.last_class_name: Optional[str] = None

        self.ai_model = None
        self.model_path: Optional[Path] = None
        self.predict_thread: Optional[PredictThread] = None
        self.predicting = False
        # 实时验证使用独立线程，不占用标注任务的 predicting 状态，也不阻塞视频播放。
        self.validation_thread: Optional[PredictThread] = None
        self.validation_mode = False
        # 每次切换媒体或验证模式时递增；旧会话的推理结果不得覆盖新会话。
        self.validation_generation = 0
        self.validation_boxes: list[dict] = []

        self.is_playing = False
        self.video_fps = 30.0
        self.playback_timer = QTimer(self)
        # 使用周期定时器按视频 FPS 推进；解码与显示开销不再额外叠加到每帧间隔。
        self.playback_timer.setSingleShot(False)
        self.playback_timer.timeout.connect(self._advance_playback)

        # 使用 QSettings 持久化推理参数及上次加载的模型路径。
        self.app_settings = QSettings("YOLOAnnotator", "YOLOAnnotator")
        self.camera_device_index = self._read_int_setting("camera/device_index", 0, 0, 9)
        saved_secondary_path = str(self.app_settings.value("validation/secondary_model_path", "")).strip()
        self.secondary_model_path: Optional[Path] = (
            Path(saved_secondary_path).expanduser() if saved_secondary_path else None
        )
        self.secondary_ai_model = None
        self.validation_model_mode = str(
            self.app_settings.value("validation/dual_model_mode", "all")
        )
        if self.validation_model_mode not in {"all", "overlap"}:
            self.validation_model_mode = "all"
        self.validation_overlap_iou = self._read_float_setting(
            "validation/dual_model_overlap_iou", 0.30, 0.01, 1.00
        )
        self.show_validation_imgsz = self.app_settings.value(
            "inference/show_validation_imgsz", False, type=bool
        )
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
        self.orb_nfeatures = self._read_int_setting("video/orb_nfeatures", 2000, 0, 10000)
        self.orb_max_side = self._read_int_setting("video/orb_max_side", 640, 320, 1920)
        self.smart_base_step = self._read_int_setting("video/smart_base_step", 5, 1, 30)
        # 批量处理范围仅属于当前媒体，不写入全局设置；每次切换媒体时重置。
        self.batch_start_frame = 1
        self.batch_end_frame = 1
        self.batch_overlap_threshold = 70.0
        self.batch_thread: Optional[BatchPredictThread] = None
        self.batch_added_count = 0
        self.batch_captured_count = 0
        self.batch_error_count = 0
        self.batch_skipped_count = 0
        self._batch_source_kind: Optional[str] = None
        self._camera_batch_new_captures: list[tuple[Path, str]] = []
        self._batch_stopped_by_user = False
        self._batch_stop_requested = False
        self._close_after_batch = False
        # 批量预标注期间主画面跟随每个成功采样帧；任务结束后恢复原选中画面。
        self._batch_restore_state: Optional[tuple] = None
        self._batch_final_status_text = ""
        self._updating_file_list = False

        self._build_ui()
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        saved_model_path = str(self.app_settings.value("model/last_path", "")).strip()
        startup_model_paths = []
        if saved_model_path:
            startup_model_paths.append(Path(saved_model_path).expanduser())
        configured_model_path = Path(AI_MODEL_PATH).expanduser()
        if configured_model_path not in startup_model_paths:
            startup_model_paths.append(configured_model_path)

        model_loaded = False
        for candidate_path in startup_model_paths:
            if candidate_path.is_file() and self._load_model(candidate_path, startup=True):
                model_loaded = True
                break
        secondary_model_warning = None
        if self.secondary_model_path is not None and HAS_YOLO:
            if self.secondary_model_path.is_file():
                try:
                    self.secondary_ai_model = YOLO(str(self.secondary_model_path.resolve()))
                    self.secondary_model_path = self.secondary_model_path.resolve()
                except Exception as exc:
                    secondary_model_warning = (
                        f"副模型加载失败：{type(exc).__name__}: {exc}"
                    )
            else:
                secondary_model_warning = (
                    f"副模型文件不存在：{self.secondary_model_path}"
                )

        if not model_loaded and not HAS_YOLO:
            self.status_lbl.setText(
                "就绪。未安装 ultralytics；仍可进行人工标注，但无法使用 AI 预标注。"
            )
        elif not model_loaded:
            if saved_model_path and not Path(saved_model_path).expanduser().is_file():
                self.status_lbl.setText(
                    f"上次使用的模型文件不存在：{saved_model_path}。请重新加载模型。"
                )
            else:
                self.status_lbl.setText(
                    "就绪。可打开图像/视频目录或本机摄像头；加载 YOLO 权重后可启用 AI 功能。"
                )
        if secondary_model_warning:
            self.status_lbl.setText(secondary_model_warning)
        self._update_model_info()
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

        self.open_camera_action = QAction("摄像头", self)
        self.open_camera_action.setToolTip("打开摄像头。")
        self.open_camera_action.triggered.connect(self._open_camera)
        toolbar.addAction(self.open_camera_action)

        self.output_dir_action = QAction("设置标注输出目录", self)
        self.output_dir_action.triggered.connect(self._choose_output_dir)
        toolbar.addAction(self.output_dir_action)

        self.save_action = QAction("保存标注", self)
        self.save_action.setShortcut("S")
        self.save_action.triggered.connect(self._save)
        toolbar.addAction(self.save_action)

        toolbar.addSeparator()
        self.model_action = QAction("加载模型", self)
        self.model_action.triggered.connect(self._choose_model)
        toolbar.addAction(self.model_action)

        self.model_settings_action = QAction("设置", self)
        self.model_settings_action.setToolTip("推理与标注设置。")
        self.model_settings_action.triggered.connect(self._show_model_settings)
        toolbar.addAction(self.model_settings_action)

        self.validation_mode_action = QAction("实时验证", self)
        self.validation_mode_action.setCheckable(True)
        self.validation_mode_action.setToolTip("显示视频或摄像头画面的模型检测结果。")
        self.validation_mode_action.toggled.connect(self._toggle_validation_mode)
        toolbar.addAction(self.validation_mode_action)

        toolbar.addSeparator()
        self.prev_frame_action = QAction("上一标注帧", self)
        self.prev_frame_action.setShortcut("A")
        self.prev_frame_action.setToolTip("跳转到上一条有标注的帧。")
        self.prev_frame_action.triggered.connect(lambda: self._step_frame(-1))
        toolbar.addAction(self.prev_frame_action)

        self.playback_action = QAction("播放", self)
        self.playback_action.setShortcut("Space")
        self.playback_action.setToolTip("播放或暂停。")
        self.playback_action.triggered.connect(self._toggle_playback)
        toolbar.addAction(self.playback_action)

        self.next_frame_action = QAction("下一标注帧", self)
        self.next_frame_action.setShortcut("D")
        self.next_frame_action.setToolTip("跳转到下一条有标注的帧。")
        self.next_frame_action.triggered.connect(lambda: self._step_frame(1))
        toolbar.addAction(self.next_frame_action)

        toolbar.addSeparator()
        self.ai_action = QAction("当前帧 AI 预标注", self)
        self.ai_action.triggered.connect(self._run_ai_predict)
        toolbar.addAction(self.ai_action)

        self.batch_ai_action = QAction("AI 批量预标注", self)
        self.batch_ai_action.triggered.connect(self._run_batch_ai_predict)
        toolbar.addAction(self.batch_ai_action)

        toolbar.addSeparator()
        self.delete_action = QAction("删除选中框", self)
        self.delete_action.setToolTip("删除选中的框。")
        self.delete_action.triggered.connect(self._delete_selected_box)
        toolbar.addAction(self.delete_action)

        toolbar.addSeparator()
        self.clear_current_action = QAction("清空当前画面", self)
        self.clear_current_action.setToolTip("清空当前画面的所有框。")
        self.clear_current_action.triggered.connect(
            self._clear_current_frame_annotations
        )
        toolbar.addAction(self.clear_current_action)

        self.delete_file_action = QAction("删除文件", self)
        self.delete_file_action.triggered.connect(self._delete_current_image_file)
        toolbar.addAction(self.delete_file_action)

        self.clear_cache_action = QAction("放弃修改", self)
        self.clear_cache_action.setToolTip("放弃当前文件未保存的修改。")
        self.clear_cache_action.triggered.connect(
            self._clear_unsaved_annotation_cache
        )
        toolbar.addAction(self.clear_cache_action)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        # 保持 tracking：拖动进度条时连续随机定位并显示对应帧。
        # 实时验证侧只保留最新待处理帧，不会为每次拖动事件创建推理队列。
        self.slider.setTracking(True)
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
        self.output_dir_lbl = QLabel("输出目录：尚未选择")
        self.output_dir_lbl.setWordWrap(True)
        self.output_dir_lbl.setToolTip("标注文件、视频帧缓存图片和类别表的统一输出位置；已有标注也只从此目录读取。")
        file_layout.addWidget(self.output_dir_lbl)
        self.file_filter_label = QLabel("搜索：")
        self.file_filter = QLineEdit()
        self.file_filter.setPlaceholderText("输入文件名或相对路径关键词")
        self.file_filter.setClearButtonEnabled(True)
        self.file_filter.setToolTip("按文件名或相对路径搜索。")
        self.file_filter.textChanged.connect(self._filter_file_list)
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.setSpacing(6)
        search_layout.addWidget(self.file_filter_label)
        search_layout.addWidget(self.file_filter, 1)
        self.file_count_lbl = QLabel("尚未加载文件夹")
        self.file_list_widget = QListWidget()
        self.file_list_widget.currentItemChanged.connect(self._on_file_item_changed)
        file_layout.addLayout(search_layout)
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
            self.multiscale_enabled,
            self.multiscale_sizes,
            self.show_validation_imgsz,
            self.inference_min_box_side,
            self.manual_replace_overlap_enabled,
            self.manual_replace_overlap_iou,
            self.camera_device_index,
            str(self.secondary_model_path or ""),
            self.validation_model_mode,
            self.validation_overlap_iou,
            self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        try:
            (
                new_inference_conf,
                new_inference_iou,
                new_inference_imgsz,
                new_inference_max_det,
                new_inference_min_box_side,
                new_manual_replace_overlap_enabled,
                new_manual_replace_overlap_iou,
                new_multiscale_enabled,
                new_multiscale_sizes,
                new_show_validation_imgsz,
                new_camera_device_index,
                secondary_model_path_text,
                new_validation_model_mode,
                new_validation_overlap_iou,
            ) = dialog.values()
        except ValueError as exc:
            QMessageBox.warning(self, "设置参数无效", str(exc))
            return

        # 先验证并加载候选副模型；加载失败时不覆盖当前可用配置。
        new_secondary_path = None
        new_secondary_model = None
        if secondary_model_path_text:
            new_secondary_path = Path(secondary_model_path_text).expanduser().resolve()
            if not new_secondary_path.is_file():
                QMessageBox.warning(
                    self, "副模型文件不存在", f"找不到副模型权重文件：\n{new_secondary_path}"
                )
                return
            if (
                self.secondary_model_path == new_secondary_path
                and self.secondary_ai_model is not None
            ):
                new_secondary_model = self.secondary_ai_model
            else:
                if not HAS_YOLO:
                    QMessageBox.warning(
                        self, "缺少依赖", "缺少 ultralytics，无法加载副模型。\n请先安装：pip install ultralytics"
                    )
                    return
                try:
                    new_secondary_model = YOLO(str(new_secondary_path))
                except Exception as exc:
                    QMessageBox.warning(
                        self, "副模型加载失败", f"{type(exc).__name__}: {exc}"
                    )
                    return

        camera_was_open = self.is_camera
        old_camera_device_index = self.camera_device_index
        (
            self.inference_conf,
            self.inference_iou,
            self.inference_imgsz,
            self.inference_max_det,
            self.inference_min_box_side,
            self.manual_replace_overlap_enabled,
            self.manual_replace_overlap_iou,
            self.multiscale_enabled,
            self.multiscale_sizes,
            self.show_validation_imgsz,
        ) = (
            new_inference_conf,
            new_inference_iou,
            new_inference_imgsz,
            new_inference_max_det,
            new_inference_min_box_side,
            new_manual_replace_overlap_enabled,
            new_manual_replace_overlap_iou,
            new_multiscale_enabled,
            new_multiscale_sizes,
            new_show_validation_imgsz,
        )
        self.secondary_model_path = new_secondary_path
        self.secondary_ai_model = new_secondary_model
        self._update_model_info()
        self.validation_model_mode = new_validation_model_mode
        self.validation_overlap_iou = float(new_validation_overlap_iou)

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
        self.app_settings.setValue(
            "inference/show_validation_imgsz", self.show_validation_imgsz
        )
        camera_device_changed = new_camera_device_index != self.camera_device_index
        self.camera_device_index = new_camera_device_index
        self.app_settings.setValue("camera/device_index", self.camera_device_index)
        self.app_settings.setValue(
            "validation/secondary_model_path",
            str(self.secondary_model_path) if self.secondary_model_path is not None else "",
        )
        self.app_settings.setValue("validation/dual_model_mode", self.validation_model_mode)
        self.app_settings.setValue("validation/dual_model_overlap_iou", self.validation_overlap_iou)
        self.app_settings.sync()
        self.open_camera_action.setToolTip("打开摄像头。")
        if camera_device_changed and camera_was_open:
            # 设置应用后立即重启采集线程，切换到新设备；保持当前实时验证开关状态。
            self.status_lbl.setText(
                f"正在切换摄像头设备：{old_camera_device_index} → {self.camera_device_index}……"
            )
            self._open_camera()
        else:
            self.status_lbl.setText(
                "设置已保存："
                f"conf={self.inference_conf:.2f}, iou={self.inference_iou:.2f}, "
                f"单尺度尺寸={self.inference_imgsz}, max_det={self.inference_max_det}, "
                f"最小目标边长={self.inference_min_box_side}px, "
                f"新框重叠清理={self.manual_replace_overlap_enabled} "
                f"(IoU={self.manual_replace_overlap_iou:.2f}), "
                f"多分辨率推理={self.multiscale_enabled} ({self.multiscale_sizes}), "
                f"摄像头设备={self.camera_device_index}"
            )
        if self.validation_mode:
            # 应用参数后使旧配置的在途结果失效；线程结束后会自动补做最新请求。
            self.validation_generation += 1
            if self.frame is not None and not (camera_device_changed and camera_was_open):
                self._run_validation()

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
                    "缺少 ultralytics，请先安装：\npip install ultralytics",
                )
            return False
        if not path.is_file():
            if not startup:
                QMessageBox.warning(self, "模型不存在", f"找不到模型文件：\n{path}")
            return False

        try:
            loaded_model = YOLO(str(path))
            if self.validation_mode:
                self.validation_generation += 1
            self.ai_model = loaded_model
            self.model_path = path.resolve()
            if not self.classes:
                self.classes = self._names_from_model(self.ai_model)
            self.app_settings.setValue("model/last_path", str(self.model_path))
            self.app_settings.sync()
            self._update_model_info()
            self.status_lbl.setText(f"模型已加载：{path}")
            self._sync_ui_state()
            if self.validation_mode and self.frame is not None:
                self._run_validation()
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
            text = "主模型：未加载"
            tooltip = text
        else:
            text = f"主模型：{self.model_path.name}"
            tooltip = str(self.model_path)
        if self.secondary_model_path is not None:
            if self.secondary_ai_model is not None:
                text += f"\n实时验证副模型：{self.secondary_model_path.name}"
                tooltip += f"\n实时验证副模型：{self.secondary_model_path}"
            else:
                text += "\n实时验证副模型：未加载"
                tooltip += f"\n副模型路径：{self.secondary_model_path}"
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
            self.output_dir_lbl.setText(f"输出目录：{self.output_root}")
            self.output_dir_lbl.setToolTip(str(self.output_root))

    def _choose_output_dir(self) -> None:
        """允许用户独立指定标注结果的输出根目录。"""
        if self.media_dir is None or self.default_output_root is None:
            QMessageBox.information(self, "尚未打开数据集", "请先打开数据集文件夹。")
            return
        current_file = self.current_file
        previous_output_root = self.output_root
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
            "选择标注输出目录",
            str(initial_dir),
            QFileDialog.ShowDirsOnly,
        )
        if not folder:
            return

        output_root = Path(folder).expanduser()
        destination_classes = output_root / "classes.txt"
        existing_classes = None
        if destination_classes.is_file():
            try:
                existing_classes = [
                    line.strip()
                    for line in destination_classes.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip()
                ]
            except OSError as exc:
                QMessageBox.warning(
                    self, "无法检查类别文件", f"无法读取目标目录的 classes.txt：\n{exc}"
                )
                return
            if self.classes and existing_classes != self.classes:
                answer = QMessageBox.warning(
                    self,
                    "类别文件不一致",
                    "所选输出目录中的 classes.txt 与当前类别表不同。\n继续后将使用当前类别表，可能覆盖目录中原有的类别映射。是否继续？",
                    QMessageBox.Yes | QMessageBox.Cancel,
                    QMessageBox.Cancel,
                )
                if answer != QMessageBox.Yes:
                    return

        had_unsaved_changes = self.dirty
        if not self._confirm_unsaved_changes():
            return

        output_changed = (
            previous_output_root is None
            or Path(previous_output_root).expanduser().resolve() != output_root.resolve()
        )
        self._configure_output_paths(output_root)
        if not self.classes and existing_classes:
            self.classes = existing_classes
        self._invalidate_label_index_cache()
        context = self.dataset_contexts.get(self.active_dataset_key or "", {})
        if output_changed:
            context.setdefault("resolved_label_paths", {}).clear()
        self._store_active_dataset_context_state()
        self.app_settings.setValue("paths/last_output_dir", str(output_root))
        self.app_settings.sync()

        if (output_changed or had_unsaved_changes) and current_file is not None:
            # 目录切换或放弃/保存未保存修改后，重新载入，避免旧的内存状态串入目标目录。
            if self._load_file(current_file):
                current_status = self.status_lbl.text()
                self.status_lbl.setText(
                    f"输出目录已切换，当前媒体已重新载入。{current_status} "
                    "旧输出目录中的标签不会自动迁移，也不会继续读取。"
                )
            else:
                self.status_lbl.setText("输出目录已切换，但当前媒体载入失败，请重新打开媒体。")
        else:
            self.status_lbl.setText("输出目录已设置；已有标签只从此目录读取，后续保存也写入此目录。")

    def _open_folder(self) -> None:
        if not self._confirm_unsaved_changes():
            return

        folder = QFileDialog.getExistingDirectory(
            self,
            "选择数据集文件夹",
            str(SCRIPT_DIR),
            QFileDialog.ShowDirsOnly,
        )
        if not folder:
            return
        selected_dir = Path(folder).resolve()
        if not selected_dir.is_dir():
            QMessageBox.warning(self, "目录错误", f"所选目录不存在：\n{selected_dir}")
            return

        dataset_roots = discover_yolo_dataset_roots(selected_dir)
        contexts: dict[str, dict] = {}
        file_context_keys: dict[Path, str] = {}
        entries: list[Path] = []

        if dataset_roots:
            # 只导入识别出的 YOLO 数据集内容，不把父目录中的任意照片/视频混入列表。
            for dataset_root in dataset_roots:
                (
                    media_dir,
                    resolved_root,
                    labels_relative_dir,
                    frame_images_relative_dir,
                    source_classes_file,
                    layout_mode,
                ) = resolve_dataset_layout(dataset_root)
                key = str(resolved_root.resolve())
                context = {
                    "dataset_root": resolved_root,
                    "media_dir": media_dir,
                    "labels_relative_dir": labels_relative_dir,
                    "frame_images_relative_dir": frame_images_relative_dir,
                    "source_classes_file": source_classes_file,
                    "layout_mode": layout_mode,
                    "output_root": resolved_root,
                    "label_index_cache": {},
                    "resolved_label_paths": {},
                    "nested_roots": [],
                    "classes": None,
                    "last_class_name": None,
                }
                contexts[key] = context
                if not media_dir.is_dir():
                    continue

                nested_roots = [
                    root for root in dataset_roots
                    if root != resolved_root and resolved_root in root.parents
                ]
                context["nested_roots"] = list(nested_roots)
                candidate_labels_dir = resolved_root / labels_relative_dir
                candidate_frame_images_dir = resolved_root / frame_images_relative_dir

                def is_media_file(candidate: Path) -> bool:
                    if not candidate.is_file():
                        return False
                    suffix = candidate.suffix.lower()
                    if layout_mode == "video_dataset":
                        # 视频数据集仍排除 images/ 抽帧缓存；仅额外导入程序抓拍的 camera_captures/ 图片。
                        is_camera_capture = (
                            suffix in IMAGE_SUFFIXES
                            and "camera_captures" in candidate.relative_to(media_dir).parts[:-1]
                        )
                        if suffix not in VIDEO_SUFFIXES and not is_camera_capture:
                            return False
                    elif layout_mode == "image_dataset":
                        if suffix not in IMAGE_SUFFIXES:
                            return False
                    elif suffix not in ALL_SUFFIXES:
                        return False

                    if any(candidate == root or root in candidate.parents for root in nested_roots):
                        return False
                    if candidate_labels_dir != media_dir:
                        try:
                            candidate.relative_to(candidate_labels_dir)
                            return False
                        except ValueError:
                            pass
                    if candidate_frame_images_dir != media_dir:
                        try:
                            candidate.relative_to(candidate_frame_images_dir)
                            return False
                        except ValueError:
                            pass
                    return True

                found = sorted(
                    (path for path in media_dir.rglob("*") if is_media_file(path)),
                    key=lambda path: str(path.relative_to(media_dir)).casefold(),
                )
                for path in found:
                    entries.append(path)
                    file_context_keys[path.resolve()] = key

        else:
            # 没有标准 images/ + labels/ 结构时，保留原有单目录导入行为。
            (
                media_dir,
                dataset_root,
                labels_relative_dir,
                frame_images_relative_dir,
                source_classes_file,
                layout_mode,
            ) = resolve_dataset_layout(selected_dir)
            if not media_dir.is_dir():
                QMessageBox.warning(self, "目录错误", f"媒体目录不存在：\n{media_dir}")
                return
            context = {
                "dataset_root": dataset_root,
                "media_dir": media_dir,
                "labels_relative_dir": labels_relative_dir,
                "frame_images_relative_dir": frame_images_relative_dir,
                "source_classes_file": source_classes_file,
                "layout_mode": layout_mode,
                "output_root": dataset_root,
                "label_index_cache": {},
                "resolved_label_paths": {},
                "nested_roots": [],
                "classes": None,
                "last_class_name": None,
            }
            key = str(dataset_root.resolve())
            contexts[key] = context
            candidate_labels_dir = dataset_root / labels_relative_dir
            candidate_frame_images_dir = dataset_root / frame_images_relative_dir

            def is_legacy_media_file(candidate: Path) -> bool:
                if not candidate.is_file() or candidate.suffix.lower() not in ALL_SUFFIXES:
                    return False
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

            entries = sorted(
                (path for path in media_dir.rglob("*") if is_legacy_media_file(path)),
                key=lambda path: str(path.relative_to(media_dir)).casefold(),
            )
            file_context_keys = {path.resolve(): key for path in entries}

        if not entries:
            QMessageBox.warning(
                self,
                "没有可处理的文件",
                "所选目录及其子目录中没有找到符合当前数据集结构的图片或视频。\n"
                "标准数据集应包含 images/；标签只会从当前输出目录中查找。",
            )
            return

        self.browse_root = selected_dir
        self.dataset_contexts = contexts
        self.file_dataset_keys = file_context_keys
        self.active_dataset_key = None
        ordered_entries = sorted(
            set(entries),
            key=lambda path: str(path.relative_to(selected_dir)).casefold()
            if selected_dir == path or selected_dir in path.parents
            else str(path).casefold(),
        )
        self.file_list, self.file_image_sequences = group_image_sequences(ordered_entries)
        self.file_index = 0
        self.classes = []
        self.last_class_name = None
        self.annotations = {}
        self._set_dirty(False)
        self._refresh_file_list()

        if not self._load_file(self.file_list[0]):
            return

        dataset_count = len(self.dataset_contexts)
        if dataset_count > 1:
            self.status_lbl.setText(
                f"已递归识别 {dataset_count} 个数据集，共导入 {len(self.file_list)} 个媒体项。"
                "切换文件时会自动切换对应的类别与输出目录。"
            )
        else:
            context = self.dataset_contexts.get(self.active_dataset_key or "", {})
            if context.get("layout_mode") == "video_dataset":
                self.status_lbl.setText(
                    "已打开视频数据集；抽帧缓存不会加入文件列表。标签从当前输出目录读取和保存。"
                )
        self._sync_ui_state()

    def _store_active_dataset_context_state(self) -> None:
        """保存当前数据集的输出目录和类别状态。"""
        if not self.active_dataset_key:
            return
        context = self.dataset_contexts.get(self.active_dataset_key)
        if context is None:
            return
        if self.output_root is not None:
            context["output_root"] = self.output_root
        context["classes"] = list(self.classes)
        context["last_class_name"] = self.last_class_name

    def _activate_file_dataset_context(self, path: Path) -> None:
        """切换文件时激活其所属数据集的路径和类别上下文。"""
        key = self.file_dataset_keys.get(path.resolve())
        if key is None or key == self.active_dataset_key:
            return

        self._store_active_dataset_context_state()
        context = self.dataset_contexts.get(key)
        if context is None:
            return

        self.active_dataset_key = key
        self.media_dir = context["media_dir"]
        self.default_output_root = context["dataset_root"]
        self.labels_relative_dir = context["labels_relative_dir"]
        self.frame_images_relative_dir = context["frame_images_relative_dir"]
        output_root = Path(context.get("output_root") or context["dataset_root"])
        self._configure_output_paths(output_root)

        context.setdefault("label_index_cache", {}).clear()

        cached_classes = context.get("classes")
        if cached_classes is not None:
            self.classes = list(cached_classes)
        else:
            source_classes_file = context["source_classes_file"]
            self.classes = []
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
            context["classes"] = list(self.classes)
        self.last_class_name = context.get("last_class_name")

    def _relative_media_path(self, path: Path) -> Path:
        if self.media_dir is None:
            raise RuntimeError("当前没有有效的媒体目录上下文。")
        return path.relative_to(self.media_dir)

    def _label_lookup_path(self, media_relative_path: Path) -> Path:
        """将图片分组目录纳入匹配键；标准 labels/ 布局不增加额外前缀。"""
        label_parts = self.labels_relative_dir.parts
        split_parts = label_parts[1:] if label_parts and label_parts[0].casefold() == "labels" else ()
        return Path(*split_parts, *media_relative_path.parts)

    def _clear_media_label_path_cache(self, path: Path) -> None:
        """媒体重新载入时清除该媒体的解析结果，允许用户修复后重新匹配。"""
        context = self.dataset_contexts.get(self.active_dataset_key or "", {})
        resolved_paths = context.setdefault("resolved_label_paths", {})
        relative = self._relative_media_path(path)
        if path.suffix.lower() in VIDEO_SUFFIXES:
            prefix_path = relative.parent / f"{relative.stem}_frame_"            prefix = self._label_path_key(self._label_lookup_path(prefix_path))
            keys = [key for key in resolved_paths if key.startswith(prefix)]
        else:
            key = self._label_path_key(
                self._label_lookup_path(relative.with_suffix(".txt"))
            )
            keys = [key]

        for key in keys:
            resolved_paths.pop(key, None)

    def _image_label_path(self, path: Path, search_if_missing: bool = True) -> Path:
        assert self.current_labels_dir is not None
        relative = self._relative_media_path(path).with_suffix(".txt")
        return self._label_path(
            self._label_lookup_path(relative), self.current_labels_dir / relative,
            search_if_missing=search_if_missing,
        )

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

    def _video_frame_label_path(
        self, frame_index: int, search_if_missing: bool = True
    ) -> Path:
        assert self.current_labels_dir is not None
        relative = self._video_frame_stem(frame_index).with_suffix(".txt")
        return self._label_path(
            self._label_lookup_path(relative), self.current_labels_dir / relative,
            search_if_missing=search_if_missing,
        )

    def _label_path(
        self, relative_label_path: Path, default_path: Path, *, search_if_missing: bool
    ) -> Path:
        """读取或首次解析标签路径；保存时关闭查找，只使用已缓存结果。"""
        context = self.dataset_contexts.get(self.active_dataset_key or "", {})
        resolved_paths = context.setdefault("resolved_label_paths", {})
        key = self._label_path_key(relative_label_path)
        if key in resolved_paths:
            return resolved_paths[key] or default_path

        if not search_if_missing:
            raise RuntimeError(f"标签路径尚未初始化：{relative_label_path}。请重新打开当前媒体。")
        existing_path = self._find_indexed_label(relative_label_path)
        resolved_paths[key] = existing_path
        return existing_path or default_path

    @staticmethod
    def _label_path_key(path: Path) -> str:
        """生成跨平台、不区分大小写的相对路径索引键。"""
        return path.as_posix().replace("\\", "/").casefold()

    def _label_index(self) -> dict:
        """只索引当前输出目录中的 TXT；每个匹配键只保留排序后的第一个文件。"""
        if self.output_root is None:
            return {"by_relative": {}, "by_name": {}, "all_paths": [], "video_frame_indices": {}}

        output_root = Path(self.output_root).expanduser().resolve()
        context = self.dataset_contexts.get(self.active_dataset_key or "", {})
        cache = context.setdefault("label_index_cache", {})
        cache_key = str(output_root)
        if cache_key in cache:
            return cache[cache_key]

        ignored_names = {
            ".git", ".hg", ".svn", "__pycache__", ".venv", "venv",
            "node_modules", ".idea", ".vscode",
        }
        excluded_roots = []
        for directory in (self.media_dir, self.frame_images_dir):
            if directory is None:
                continue
            resolved = Path(directory).resolve()
            if resolved != output_root and output_root in resolved.parents:
                excluded_roots.append(resolved)

        # 输出目录若是多个数据集的共同父目录，不把其他数据集的标签混入索引。
        for key, other_context in self.dataset_contexts.items():
            if key == self.active_dataset_key:
                continue
            other_root = Path(other_context["dataset_root"]).resolve()
            if other_root != output_root and output_root in other_root.parents:
                excluded_roots.append(other_root)
        excluded_roots.extend(Path(item).resolve() for item in context.get("nested_roots", []))

        by_relative: dict[str, Path] = {}
        by_name: dict[str, Path] = {}
        all_paths: list[Path] = []
        if output_root.is_dir():
            for current, directories, filenames in os.walk(
                output_root, topdown=True, followlinks=False
            ):
                current_path = Path(current)
                current_resolved = current_path.resolve()
                if any(
                    current_resolved == excluded or excluded in current_resolved.parents
                    for excluded in excluded_roots
                ):
                    directories[:] = []
                    continue

                directories[:] = sorted(
                    [
                        name for name in directories
                        if name.casefold() not in ignored_names
                        and (current_path / name).resolve() not in excluded_roots
                    ],
                    key=str.casefold,
                )
                for filename in sorted(filenames, key=str.casefold):
                    if Path(filename).suffix.casefold() != ".txt" or filename.casefold() == "classes.txt":
                        continue
                    candidate = current_path / filename
                    all_paths.append(candidate)
                    by_name.setdefault(candidate.name.casefold(), candidate)

                    # 为输出目录内不同名称的标签子目录建立相对路径别名。
                    relative_parts = candidate.relative_to(output_root).parts
                    for offset in range(len(relative_parts)):
                        key = self._label_path_key(Path(*relative_parts[offset:]))
                        by_relative.setdefault(key, candidate)

        index = {
            "by_relative": by_relative,
            "by_name": by_name,
            "all_paths": all_paths,
            "video_frame_indices": {},
        }
        cache[cache_key] = index
        return index

    def _find_indexed_label(self, relative_label_path: Path) -> Optional[Path]:
        """按相对路径优先、文件名回退的规则，返回第一个匹配文件。"""
        index = self._label_index()
        candidate = index["by_relative"].get(self._label_path_key(relative_label_path))
        if candidate is not None and candidate.is_file():
            return candidate

        # 文件名回退仅限输出根目录下的平铺标签，不跨媒体子目录误配。
        if self.output_root is None:
            return None
        output_root = Path(self.output_root).expanduser().resolve()
        candidate = index["by_name"].get(relative_label_path.name.casefold())
        if (
            candidate is not None
            and len(candidate.relative_to(output_root).parts) <= 2
            and candidate.is_file()
        ):
            return candidate
        return None

    def _existing_image_label_path(self, path: Path) -> Optional[Path]:
        label_path = self._image_label_path(path)
        return label_path if label_path.is_file() else None

    def _existing_video_frame_label_path(self, frame_index: int) -> Optional[Path]:
        label_path = self._video_frame_label_path(frame_index)
        return label_path if label_path.is_file() else None

    def _update_video_frame_label_status(
        self,
        frame_index: int,
        label_path: Optional[Path],
        loaded_from_disk: bool = False,
        include_video_summary: bool = False,
    ) -> None:
        """反馈当前帧标签状态；打开视频时可同时汇总整段视频的已标注帧数。"""
        if self.current_file is None:
            return

        box_count = len(self.annotations.get(frame_index, []))
        frame_text = f"第 {frame_index + 1} 帧"
        marked_frames = self._marked_video_frame_indices() if include_video_summary else None
        summary_text = (
            f"当前视频共有 {len(marked_frames)} 个带标注框的帧。"
            if marked_frames is not None else ""
        )
        if loaded_from_disk and label_path is not None:
            message = (
                f"已载入现有标注：{self.current_file.name} / {frame_text}，"
                f"{box_count} 个标注框。可直接选择并修改。"
            )
        elif frame_index in self.annotations:
            message = (
                f"当前帧标注缓存：{self.current_file.name} / {frame_text}，"
                f"{box_count} 个标注框。"
            )
        elif marked_frames:
            # 当前帧可能没有标注，但后续帧存在已有标注；不能将整段视频误报为无标签。
            message = (
                f"已打开视频：{self.current_file.name} / {frame_text}；"
                "当前帧暂无标注。"
            )
        elif marked_frames is not None:
            message = (
                f"已打开视频：{self.current_file.name} / {frame_text}；"
                "当前视频暂无包含有效标注框的帧。"
            )
        else:
            message = (
                f"已打开视频：{self.current_file.name} / {frame_text}；"
                "当前输出目录中没有当前帧对应的标签文件。"
            )

        if summary_text:
            message = f"{message} {summary_text}"
        self.status_lbl.setText(message)

    @staticmethod
    def _validate_yolo_label_file(txt_path: Path) -> Optional[tuple[int, str]]:
        """验证标签是否完全符合当前矩形框格式；返回首个问题的行号和原因。"""
        try:
            lines = txt_path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError) as exc:
            return 0, f"无法读取标签文件：{exc}"

        for line_number, line in enumerate(lines, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) not in (5, 6):
                if len(parts) >= 7 and len(parts) % 2 == 1:
                    reason = "疑似 YOLO 多边形分割标注"
                else:
                    reason = f"数值数量为 {len(parts)}，当前矩形框格式要求 5 或 6 个"
                return line_number, reason
            try:
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
            except (ValueError, OverflowError):
                return line_number, "包含非数字或非法类别编号"
            if class_id < 0:
                return line_number, "类别编号不能为负数"
            if not np.isfinite(values).all():
                return line_number, "坐标或置信度不是有限数值"
            cx, cy, width, height = values[:4]
            if width <= 0 or height <= 0:
                return line_number, "矩形框宽度和高度必须大于 0"
            if len(values) == 5 and not 0.0 <= values[4] <= 1.0:
                return line_number, "置信度必须在 0 到 1 之间"
        return None

    def _incompatible_video_label(self, path: Path) -> Optional[tuple[Path, int, str]]:
        """检查该视频已存在的帧标签，返回首个不兼容项以便提醒。"""
        index = self._label_index()
        relative_video = self._relative_media_path(path)
        prefix = f"{relative_video.stem}_frame_".casefold()
        for label_path in index["all_paths"]:
            stem = label_path.stem
            if not stem.casefold().startswith(prefix):
                continue
            frame_text = stem[len(prefix):]
            if not frame_text.isdigit():
                continue
            issue = self._validate_yolo_label_file(label_path)
            if issue is not None:
                return label_path, issue[0], issue[1]
        return None

    def _load_file(self, path: Path) -> bool:
        # 切换数据集上下文后、清空当前媒体状态前检查标签格式；发现问题时提醒用户，但不阻断打开。
        if (self.is_camera or self.camera_thread is not None) and not self._stop_camera():
            return False
        # 图片组只在文件列表中显示一个代表项；实际打开路径可以是组内任意成员。
        sequence = self.file_image_sequences.get(path.resolve())
        if sequence:
            self.current_image_sequence = list(sequence)
            self.image_sequence_index = next(
                (index for index, member in enumerate(sequence) if member.resolve() == path.resolve()),
                0,
            )
            path = self.current_image_sequence[self.image_sequence_index]
            self.is_image_sequence = len(self.current_image_sequence) > 1
        else:
            self.current_image_sequence = []
            self.image_sequence_index = 0
            self.is_image_sequence = False
        # 媒体切换只使旧推理结果失效，不强制中断正在运行的 YOLO 调用。
        self.validation_generation += 1
        self._activate_file_dataset_context(path)
        incompatible = None
        if path.suffix.lower() in VIDEO_SUFFIXES:
            video_issue = self._incompatible_video_label(path)
            if video_issue is not None:
                label_path, line_number, reason = video_issue
                incompatible = (label_path, line_number, reason)
        else:
            label_path = self._existing_image_label_path(path)
            if label_path is not None:
                issue = self._validate_yolo_label_file(label_path)
                if issue is not None:
                    incompatible = (label_path, issue[0], issue[1])
        if incompatible is not None:
            label_path, line_number, reason = incompatible
            line_text = f"第 {line_number} 行" if line_number else "文件读取阶段"
            QMessageBox.warning(
                self,
                "标签格式异常",
                f"{label_path.name}（{line_text}）。\n保存可能丢失数据。",
            )
        self._clear_media_label_path_cache(path)
        self._stop_playback()
        self.video_fps = 30.0
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.annotations = {}
        self.dirty_annotation_indices.clear()
        self.image_sequence_sizes = {}
        self.validation_boxes = []
        self.canvas.set_validation_boxes([])
        self.frame_idx = self.image_sequence_index if self.is_image_sequence else 0
        self.frame = None
        self.current_file = path
        self._set_dirty(False)
        self.canvas.reset_view()

        # 批量处理范围与当前媒体绑定：切换文件时重置，并在读取媒体数量后填入末尾位置。
        self.batch_start_frame = 1
        self.batch_end_frame = 1
        self.is_video = path.suffix.lower() in VIDEO_SUFFIXES
        if self.is_video:
            self.is_image_sequence = False
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
            reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if np.isfinite(reported_fps) and reported_fps > 0:
                self.video_fps = min(reported_fps, 240.0)
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
                self.batch_end_frame = self.total_frames
                self.frame_idx = 0
                self.frame = first_frame
                self.image_h, self.image_w = first_frame.shape[:2]
                self.slider.setRange(0, 0)
                label_path = self._existing_video_frame_label_path(0)
                loaded_from_disk = False
                if label_path is not None:
                    self._load_yolo_txt(label_path, 0)
                    loaded_from_disk = 0 in self.annotations
                self._refresh_frame(reset_view=True)
                self._update_video_frame_label_status(
                    0, label_path, loaded_from_disk, include_video_summary=True
                )
            else:
                self.total_frames = frame_count
                self.batch_end_frame = self.total_frames
                self.slider.setRange(0, self.total_frames - 1)
                self._seek(0, reset_view=True)
                if self.frame is not None:
                    # 首帧没有标签时，也要立即统计后续帧的已有标注，避免误报整段视频无标签。
                    first_label_path = self._existing_video_frame_label_path(0)
                    self._update_video_frame_label_status(
                        0,
                        first_label_path,
                        loaded_from_disk=(0 in self.annotations and first_label_path is not None),
                        include_video_summary=True,
                    )
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
            self.total_frames = len(self.current_image_sequence) if self.is_image_sequence else 1
            self.batch_end_frame = self.total_frames
            self.frame_idx = self.image_sequence_index if self.is_image_sequence else 0
            self.image_h, self.image_w = image.shape[:2]
            if self.is_image_sequence:
                self.image_sequence_sizes[self.frame_idx] = (self.image_w, self.image_h)
            self.slider.setRange(0, self.total_frames - 1)
            label_path = self._existing_image_label_path(path)
            if label_path is not None:
                self._load_yolo_txt(label_path, self.frame_idx)
            self._refresh_frame(reset_view=True)
            if self.is_image_sequence:
                self.status_lbl.setText(
                    f"已打开图片组：{path.parent.name} / {path.stem}，"
                    f"第 {self.frame_idx + 1} / {self.total_frames} 张。"
                    + (f"已有 {len(self.annotations.get(self.frame_idx, []))} 个标注框。" if label_path else "")
                )
            elif label_path is not None:
                loaded_count = len(self.annotations.get(0, []))
                self.status_lbl.setText(
                    f"已载入已有标注：{path.name}，{loaded_count} 个标注框。可直接选择并修改。"
                )
            else:
                self.status_lbl.setText(f"已打开图片：{path.name}；当前输出目录中没有对应的标签文件。")

        self._set_file_list_current()
        self._sync_ui_state()
        if self.validation_mode:
            self._run_validation()
        return True

    def _open_camera(self) -> None:
        """打开设置中选定的本机摄像头；无需预先打开数据集目录。"""
        if not self._confirm_unsaved_changes():
            return
        if self.predicting or (self.batch_thread is not None and self.batch_thread.isRunning()):
            self.status_lbl.setText("当前 AI 任务尚未完成，暂时无法打开摄像头。")
            return
        if (self.is_camera or self.camera_thread is not None) and not self._stop_camera():
            return

        self.validation_generation += 1
        self._stop_playback()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.current_file = None
        self._set_dirty(False)
        self.is_video = False
        self.is_image_sequence = False
        self.current_image_sequence = []
        self.image_sequence_index = 0
        self.image_sequence_sizes = {}
        self.dirty_annotation_indices.clear()
        self.total_frames = 0
        self.frame_idx = 0
        self.frame = None
        self.image_w = 0
        self.image_h = 0
        self.annotations = {}
        self.validation_boxes = []
        self.canvas.set_validation_boxes([])
        self.canvas.set_boxes([])
        self.canvas.clear_frame()
        self.canvas.reset_view()
        self.camera_last_sequence = -1
        self.is_camera = True
        self.is_playing = bool(self.validation_mode)
        self.file_index = -1
        self._set_file_list_current()

        thread = CameraCaptureThread(self.camera_device_index)
        thread.setParent(self)
        self.camera_thread = thread
        thread.error_signal.connect(self._on_camera_error)
        thread.finished.connect(self._on_camera_thread_finished)
        thread.start()
        self.camera_timer.start()
        self.status_lbl.setText(f"正在连接本机摄像头（设备 {self.camera_device_index}）……")
        self._sync_ui_state()

    def _on_camera_error(self, message: str) -> None:
        if self.sender() is not self.camera_thread:
            return
        self.camera_timer.stop()
        self.is_camera = False
        self.is_playing = False
        self.validation_generation += 1
        self.validation_boxes = []
        self.canvas.set_validation_boxes([])
        self.status_lbl.setText(message)
        self._sync_ui_state()

    def _on_camera_thread_finished(self) -> None:
        thread = self.sender()
        if thread is self.camera_thread:
            self.camera_thread = None
            if self.batch_thread is not None and self.batch_thread.isRunning():
                # 批量线程仍可能读取该线程的最终序号/帧；延迟释放到批量线程结束。
                self._camera_thread_pending_delete = thread
            else:
                thread.deleteLater()
            if self.is_camera:
                self.camera_timer.stop()
                self.is_camera = False
                self.is_playing = False
                self.validation_generation += 1
                self.status_lbl.setText("摄像头采集已停止。")
                self._sync_ui_state()
        elif thread is not self._camera_thread_pending_delete:
            # 例如 _stop_camera() 等待线程结束后已先清空 camera_thread 引用。
            thread.deleteLater()

    def _stop_camera(self) -> bool:
        """停止摄像头线程；若驱动未及时返回，不释放仍在运行的 QThread。"""
        self.camera_timer.stop()
        self.is_playing = False
        thread = self.camera_thread
        if thread is not None:
            thread.request_stop()
            if not thread.wait(3000):
                self.status_lbl.setText(
                    "摄像头采集线程尚未退出，暂时无法切换媒体；请检查摄像头驱动是否卡住。"
                )
                return False
            self.camera_thread = None
        self.is_camera = False
        self.camera_last_sequence = -1
        self._update_playback_action()
        return True

    def _update_camera_frame(self) -> None:
        """从采集线程的单帧缓冲区取最新画面；不在 Qt 事件队列中积压帧。"""
        if not self.is_camera or self.camera_thread is None:
            return
        if self.validation_mode and not self.is_playing:
            return
        sequence, frame = self.camera_thread.latest_frame()
        if frame is None or sequence == self.camera_last_sequence:
            return
        first_frame = self.frame is None
        self.camera_last_sequence = sequence
        self.frame = frame
        self.frame_idx = sequence
        self.image_h, self.image_w = frame.shape[:2]
        self.canvas.set_frame(frame)
        self.canvas.set_boxes([])
        self.canvas.set_validation_boxes(self.validation_boxes if self.validation_mode else [])
        self.slider_label.setText("摄像头实时画面")
        self.tree.clear()
        if self.validation_mode:
            self._run_validation()
        if first_frame:
            self._sync_ui_state()

    def _camera_capture_destination(self) -> tuple[Path, str]:
        """返回抓拍目录和对应的数据集上下文键。"""
        active_context = self.dataset_contexts.get(self.active_dataset_key or "")
        if (
            self.media_dir is not None
            and self.current_labels_dir is not None
            and active_context is not None
        ):
            # 已打开数据集时延续原有抓拍分组及标签相对路径，不改变数据集布局。
            capture_dir = (
                self.media_dir
                if active_context.get("camera_default_capture_root")
                else self.media_dir / "camera_captures"
            )
            return capture_dir, str(self.active_dataset_key)

        # 没有已打开的数据集时，使用脚本所在目录作为独立 YOLO 数据集根目录。
        dataset_root = SCRIPT_DIR
        media_dir = dataset_root / "images"
        labels_dir = dataset_root / "labels"
        media_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        context_key = str(dataset_root.resolve())
        context = self.dataset_contexts.get(context_key)
        if context is None:
            context = {
                "dataset_root": dataset_root,
                "media_dir": media_dir,
                "labels_relative_dir": Path("labels"),
                "frame_images_relative_dir": Path("images"),
                "source_classes_file": dataset_root / "classes.txt",
                "layout_mode": "image_dataset",
                "output_root": dataset_root,
                "label_index_cache": {},
                "resolved_label_paths": {},
                "nested_roots": [],
                "classes": None,
                "last_class_name": None,
                "camera_default_capture_root": True,
            }
            self.dataset_contexts[context_key] = context
        else:
            # 复用已有脚本目录上下文中的类别、输出路径和标签路径缓存。
            context["dataset_root"] = dataset_root
            context["media_dir"] = media_dir
            context["labels_relative_dir"] = Path("labels")
            context["frame_images_relative_dir"] = Path("images")
            context["source_classes_file"] = dataset_root / "classes.txt"
            context["layout_mode"] = "image_dataset"
            context["output_root"] = dataset_root
            context["camera_default_capture_root"] = True

        self._store_active_dataset_context_state()
        self.active_dataset_key = None
        self.browse_root = dataset_root
        return media_dir, context_key

    def _register_camera_capture(self, image_path: Path, context_key: str) -> None:
        """将新抓拍图片加入文件列表，并按统一规则刷新图片组。"""
        self._register_camera_captures([(image_path, context_key)])

    def _register_camera_captures(
        self, captures: list[tuple[Path, str]]
    ) -> None:
        """批量登记摄像头生成的图片，避免逐张重建整个分组索引。"""
        if not captures:
            return
        # 展开旧组成员后重新分组，确保连续抓拍的 camera_ 时间戳图片自动形成序列。
        entries: list[Path] = []
        for path in self.file_list:
            sequence = self.file_image_sequences.get(path.resolve())
            entries.extend(sequence if sequence else [path])
        existing = {path.resolve() for path in entries}
        for image_path, context_key in captures:
            resolved_path = image_path.resolve()
            self.file_dataset_keys[resolved_path] = context_key
            if resolved_path not in existing:
                entries.append(image_path)
                existing.add(resolved_path)
        entries.sort(
            key=lambda path: (
                str(path.relative_to(self.browse_root)).casefold()
                if self.browse_root is not None and self.browse_root in path.parents
                else str(path).casefold()
            )
        )
        self.file_list, self.file_image_sequences = group_image_sequences(entries)
        # 摄像头仍是当前媒体源，不把文件列表的当前索引改成新抓拍图；
        # 用户之后主动点击图片组时，才进入普通标注流程。
        self._refresh_file_list()

    def _capture_camera_frame(self) -> None:
        """保存当前摄像头画面并加入文件列表，保持摄像头预览继续运行。"""
        if not self.is_camera or self.frame is None:
            return
        try:
            capture_dir, context_key = self._camera_capture_destination()
            capture_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_path = capture_dir / f"camera_{timestamp}.jpg"
            if not save_image_unicode(image_path, self.frame.copy()):
                raise OSError(f"抓拍图片保存失败：{image_path}")
        except Exception as exc:
            QMessageBox.critical(self, "摄像头抓拍失败", f"{type(exc).__name__}: {exc}")
            return

        self._register_camera_capture(image_path, context_key)
        self.status_lbl.setText(
            f"抓拍图片已保存：{image_path.name}。摄像头预览继续运行，可从文件列表打开图片进行标注。"
        )

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
            if len(parts) not in (5, 6):
                self.status_lbl.setText(
                    f"YOLO 标签格式错误，已跳过：{txt_path.name} 第 {line_number} 行"
                )
                continue
            try:
                class_id = int(parts[0])
                cx, cy, width, height = map(float, parts[1:5])
                source_confidence = float(parts[5]) if len(parts) == 6 else None
                if source_confidence is not None and (
                    not np.isfinite(source_confidence) or not 0.0 <= source_confidence <= 1.0
                ):
                    source_confidence = None
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
                loaded_box = {
                    "class_id": class_id,
                    "class_name": self._class_name_for_id(class_id),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "source": "loaded",
                }
                if source_confidence is not None:
                    loaded_box["source_confidence"] = source_confidence
                boxes.append(loaded_box)
            except (ValueError, OverflowError):
                self.status_lbl.setText(
                    f"YOLO 标签内容无效，已跳过：{txt_path.name} 第 {line_number} 行"
                )

        # 空 txt 同样需要记录，代表用户/数据集明确标注了负样本。
        self.annotations[frame_index] = boxes

    # ---------- 画面与帧切换 ----------

    def _refresh_frame(
        self, reset_view: bool = False, refresh_tree: bool = True
    ) -> None:
        if self.frame is None:
            self.canvas.update()
            self.canvas.set_boxes([])
            self.tree.clear()
            self.slider_label.setText("0 / 0")
            return

        self.canvas.set_frame(self.frame, reset_view=reset_view)
        self.canvas.set_boxes(self.annotations.get(self.frame_idx, []))
        if self.validation_mode:
            self.canvas.set_validation_boxes(self.validation_boxes)
        else:
            self.canvas.set_validation_boxes([])

        self.slider.blockSignals(True)
        self.slider.setValue(self.frame_idx)
        self.slider.blockSignals(False)
        self.slider_label.setText(f"{self.frame_idx + 1} / {self.total_frames}")
        if refresh_tree:
            self._refresh_tree()

    def _seek(self, index: int, reset_view: bool = False) -> None:
        """随机定位到指定帧；仅用于用户跳转，不用于连续播放。"""
        if not self.is_video or self.cap is None or self.total_frames <= 0:
            return
        index = min(max(int(index), 0), self.total_frames - 1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.status_lbl.setText(f"无法读取第 {index + 1} 帧。")
            return
        self._display_video_frame(index, frame, reset_view=reset_view)

    def _display_video_frame(
        self, index: int, frame: np.ndarray, reset_view: bool = False
    ) -> None:
        """显示已顺序读取或随机读取的视频帧，并统一处理标签和验证状态。"""
        # 视频控制只负责更新当前画面；实时验证独立消费最新帧。
        # 无论连续播放还是手动随机定位，都保留当前检测框，直到下一次推理完成。
        self.frame = frame
        self.frame_idx = index
        self.image_h, self.image_w = frame.shape[:2]

        label_path = self._existing_video_frame_label_path(index)
        loaded_from_disk = False
        if index not in self.annotations and label_path is not None:
            self._load_yolo_txt(label_path, index)
            loaded_from_disk = index in self.annotations

        self._refresh_frame(reset_view=reset_view)
        self._update_video_frame_label_status(index, label_path, loaded_from_disk)
        self._sync_ui_state()
        if self.validation_mode:
            self._run_validation()

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
            if len(parts) not in (5, 6):
                continue
            try:
                class_id = int(parts[0])
                cx, cy, width, height = map(float, parts[1:5])
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
        """返回当前视频所有包含有效标注框的帧序号（从 0 开始）。"""
        if not self.is_video or self.current_file is None:
            return []

        # 内存中的空列表代表用户显式清空该帧，必须覆盖磁盘上的旧标签。
        marked = {
            frame_index
            for frame_index, boxes in self.annotations.items()
            if 0 <= frame_index < self.total_frames and boxes
        }

        index = self._label_index()
        context = self.dataset_contexts.get(self.active_dataset_key or "", {})
        relative_video = self._relative_media_path(self.current_file)
        prefix = f"{relative_video.stem}_frame_"
        prefix_key = prefix.casefold()
        candidate_indices = index["video_frame_indices"].get(prefix_key)
        if candidate_indices is None:
            candidate_indices = set()
            for path in index["all_paths"]:
                stem = path.stem
                if stem.casefold().startswith(prefix_key):
                    frame_text = stem[len(prefix):]
                    if frame_text.isdigit():
                        candidate_indices.add(int(frame_text))
            index["video_frame_indices"][prefix_key] = candidate_indices

        for frame_index in candidate_indices:
            if not 0 <= frame_index < self.total_frames or frame_index in self.annotations:
                continue
            label_path = self._existing_video_frame_label_path(frame_index)
            if label_path is not None and self._label_file_has_boxes(label_path):
                marked.add(frame_index)

        return sorted(marked)

    def _step_frame(self, delta: int) -> None:
        """视频跳转到相邻带标注帧；图片组按顺序切换图片。"""
        if self.predicting:
            return
        self._stop_playback()

        if self.is_image_sequence:
            target = self.frame_idx + (-1 if delta < 0 else 1)
            if not 0 <= target < len(self.current_image_sequence):
                self.status_lbl.setText(
                    "已经是图片组第一张。" if delta < 0 else "已经是图片组最后一张。"
                )
                return
            self._display_image_sequence_frame(target)
            return

        if not self.is_video:
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

    def _display_image_sequence_frame(self, index: int) -> None:
        """切换图片组成员；保留组内所有已载入/修改的标注缓存。"""
        if not self.is_image_sequence or not self.current_image_sequence:
            return
        index = min(max(int(index), 0), len(self.current_image_sequence) - 1)
        path = self.current_image_sequence[index]
        image = load_image_unicode(path)
        if image is None:
            self.status_lbl.setText(f"图片读取失败：{path.name}。仍保留当前画面。")
            return

        was_cached = index in self.annotations
        label_path = self._existing_image_label_path(path)
        if label_path is not None and not was_cached:
            issue = self._validate_yolo_label_file(label_path)
            if issue is not None:
                line_text = f"第 {issue[0]} 行" if issue[0] else "文件读取阶段"
                QMessageBox.warning(
                    self, "标签格式异常", f"{label_path.name}（{line_text}）。\n保存可能丢失数据。"
                )

        self.image_sequence_index = index
        self.current_file = path
        self.frame_idx = index
        self.frame = image
        self.image_h, self.image_w = image.shape[:2]
        self.image_sequence_sizes[index] = (self.image_w, self.image_h)
        self.validation_boxes = []
        if index not in self.annotations and label_path is not None:
            self._load_yolo_txt(label_path, index)
        self._refresh_frame(reset_view=True)
        if index in self.annotations:
            cache_label = "缓存标注" if was_cached else "已载入标注"
            self.status_lbl.setText(
                f"图片组：第 {index + 1} / {self.total_frames} 张；"
                f"{cache_label} {len(self.annotations[index])} 个标注框。"
            )
        else:
            self.status_lbl.setText(
                f"图片组：第 {index + 1} / {self.total_frames} 张；当前图片尚无标签。"
            )
        self._update_window_title()
        self._sync_ui_state()
        if self.validation_mode:
            self._run_validation()

    def _on_slider_changed(self, value: int) -> None:
        if value == self.frame_idx:
            return
        if self.is_video:
            self._stop_playback()
            self._seek(value)
        elif self.is_image_sequence:
            self._display_image_sequence_frame(value)

    def _refresh_file_list(self) -> None:
        """重建左侧文件列表，并保持当前文件选中状态。"""
        if not hasattr(self, "file_list_widget"):
            return
        self._updating_file_list = True
        self.file_list_widget.blockSignals(True)
        self.file_list_widget.clear()
        for index, path in enumerate(self.file_list):
            if self.browse_root is not None:
                try:
                    display_name = str(path.relative_to(self.browse_root))
                except ValueError:
                    display_name = path.name
            elif self.media_dir is not None:
                try:
                    display_name = str(path.relative_to(self.media_dir))
                except ValueError:
                    display_name = path.name
            else:
                display_name = path.name
            sequence = self.file_image_sequences.get(path.resolve())            tooltip = str(path)
            if sequence and len(sequence) > 1:
                display_name = f"{display_name}  [{len(sequence)} 张]"
                tooltip += "\n" + "\n".join(member.name for member in sequence)
            item = QListWidgetItem(display_name)
            item.setToolTip(tooltip)
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
            searchable_text = f"{item.text()} {item.toolTip()}".casefold()
            matched = not query or query in searchable_text
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
        previous_path = self.current_file
        self.file_index = target_index
        if not self._load_file(self.file_list[target_index]):
            self.file_index = previous_index
            # 目标文件无法读取时恢复原文件及其组内位置，避免界面跳回组首或停留空白。
            if previous_path is not None:
                self._load_file(previous_path)
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
        if self.validation_mode:
            return
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
        """处理主工作区快捷键，同时保留文本输入和按钮控件的原生键盘行为。"""
        if event.type() != QEvent.KeyPress or event.isAutoRepeat() or not self.isActiveWindow():
            return super().eventFilter(watched, event)

        # 空格切换视频/摄像头播放状态；焦点位于输入控件或按钮时不拦截。
        focus_widget = QApplication.focusWidget()
        protected_widgets = (
            QLineEdit,
            QAbstractSpinBox,
            QComboBox,
            QTextEdit,
            QPlainTextEdit,
            QPushButton,
            QToolButton,
        )
        if isinstance(focus_widget, protected_widgets) or QApplication.activeModalWidget() is not None:
            return super().eventFilter(watched, event)

        if event.key() == Qt.Key_Backspace:
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

    def _toggle_validation_mode(self, enabled: bool) -> None:
        """切换独立实时验证层；验证检测框不进入标注数据。"""
        enabled = bool(enabled)
        if enabled and self.ai_model is None:
            self.validation_mode_action.blockSignals(True)
            self.validation_mode_action.setChecked(False)
            self.validation_mode_action.blockSignals(False)
            QMessageBox.information(self, "需要 YOLO 模型", "请先加载 YOLO 权重，再启用实时验证。")
            return

        self.validation_generation += 1
        self.validation_mode = enabled
        if self.is_camera:
            self.is_playing = enabled
            if not self.camera_timer.isActive():
                self.camera_timer.start()
        self.validation_boxes = []
        self.canvas.set_validation_mode(enabled)
        self.canvas.set_validation_boxes([])
        self._sync_ui_state()
        if not enabled:
            if self.is_camera:
                self.is_playing = False
                self._update_camera_frame()
                self._update_playback_action()
            else:
                self._refresh_frame()
            self.status_lbl.setText("实时验证已关闭，恢复普通画面显示。")
            return
        if self.is_camera:
            if self.frame is not None:
                self._run_validation()
            self._update_playback_action()
        else:
            self._refresh_frame()
            self._run_validation()

    def _validation_source_key(self) -> Optional[str]:
        if self.is_camera:
            return f"camera:{self.camera_device_index}"
        return str(self.current_file) if self.current_file is not None else None

    def _validation_request_signature(
        self, imgsz_values: Optional[tuple[int, ...]] = None
    ) -> Optional[tuple]:
        """返回当前待验证画面的身份及推理参数，用于判断是否需要补做最新帧。"""
        source_key = self._validation_source_key()
        if (
            not self.validation_mode
            or self.ai_model is None
            or source_key is None
            or self.frame is None
        ):
            return None
        if imgsz_values is None:
            try:
                imgsz_values = self._active_imgsz_values()
            except ValueError:
                return None
        return (
            self.validation_generation,
            source_key,
            int(self.frame_idx),
            id(self.ai_model),
            id(self.secondary_ai_model) if self.secondary_ai_model is not None else None,
            str(self.secondary_model_path) if self.secondary_model_path is not None else None,
            self.validation_model_mode,
            float(self.validation_overlap_iou),
            float(self.inference_conf),
            float(self.inference_iou),
            tuple(imgsz_values),
            int(self.inference_max_det),
            int(self.inference_min_box_side),
        )

    def _run_validation(self) -> None:
        """异步推理当前画面；忙碌时不排队，线程结束后只补做最新且不同的画面。"""
        if not self.validation_mode or self.ai_model is None:
            return
        source_key = self._validation_source_key()
        if source_key is None or self.frame is None:
            return

        try:
            active_imgsz_values = self._active_imgsz_values()
        except ValueError as exc:
            self.status_lbl.setText(f"无法启动实时验证：{exc}")
            QMessageBox.warning(self, "实时验证参数无效", str(exc))
            return

        request_signature = self._validation_request_signature(active_imgsz_values)
        if request_signature is None:
            return
        if self.validation_thread is not None:
            # 不保存帧队列，也不逐帧排队；当前 self.frame 始终代表最新画面。
            # 推理结束时比较签名，只有最新画面/会话/参数确实变化才启动下一次推理。
            return

        # 保留当前显示的上一帧结果；新帧推理完成后再一次性替换，避免框闪烁。
        # 该线程独立于播放和标注任务，视频播放不等待推理完成。
        self.status_lbl.setText(
            f"摄像头实时验证中：采集帧 {self.frame_idx}……"
            if self.is_camera
            else f"实时验证中：{self.current_file.name}，第 {self.frame_idx + 1} 帧……"
        )
        thread = PredictThread(
            self.ai_model,
            self.frame,
            source_key,
            self.frame_idx,
            self.inference_conf,
            self.inference_iou,
            active_imgsz_values,
            self.inference_max_det,
            self.inference_min_box_side,
            secondary_model=self.secondary_ai_model,
            dual_model_mode=self.validation_model_mode,
            dual_model_overlap_iou=self.validation_overlap_iou,
        )
        # 记录媒体/验证会话；同一会话内允许较早帧的结果更新当前画面，
        # 但切换媒体或验证模式后，旧会话结果必须丢弃。
        thread.validation_generation = self.validation_generation
        thread.validation_signature = request_signature
        # 由窗口 QObject 持有线程对象，直到 finished 信号确认线程已退出。
        # finished 后再安排 deleteLater，避免快速拖动触发帧切换时过早销毁运行中的 QThread。
        thread.setParent(self)
        self.validation_thread = thread
        thread.finished_signal.connect(self._on_validation_finished)
        thread.error_signal.connect(self._on_validation_error)
        thread.finished.connect(self._on_validation_thread_stopped)
        thread.finished.connect(thread.deleteLater)
        thread.start()
        self._sync_ui_state()

    def _on_validation_finished(
        self, file_path: str, frame_index: int, detections: list
    ) -> None:
        # 同一媒体会话内，结果允许晚于对应帧到达：这正是异步播放的预期行为。
        # 只按媒体/验证会话过滤，不按当前 frame_idx 过滤，否则播放越流畅越容易丢光结果。
        thread = self.sender()
        if (
            not self.validation_mode
            or self._validation_source_key() != file_path
            or thread is None
            or getattr(thread, "validation_generation", -1) != self.validation_generation
        ):
            return

        self.validation_boxes = [
            {
                "class_id": int(detection["model_class_id"]),
                "class_name": str(detection["class_name"]),
                "model_index": detection.get("model_index"),
                "x1": float(detection["x1"]),
                "y1": float(detection["y1"]),
                "x2": float(detection["x2"]),
                "y2": float(detection["y2"]),
                "confidence": float(detection["confidence"]),
                "source_imgsz": detection.get("source_imgsz"),
                "show_source_imgsz": bool(self.show_validation_imgsz and self.multiscale_enabled),
            }
            for detection in detections
        ]
        self.canvas.set_validation_boxes(self.validation_boxes)
        secondary_warning = getattr(thread, "secondary_warning", None)
        status_text = (
            f"摄像头实时验证完成：检测到 {len(self.validation_boxes)} 个目标。"
            if self.is_camera
            else f"实时验证完成：第 {frame_index + 1} 帧检测到 {len(self.validation_boxes)} 个目标。"
        )
        if secondary_warning:
            status_text += f" 副模型推理失败，已仅显示主模型结果：{secondary_warning}"
        self.status_lbl.setText(status_text)

    def _on_validation_thread_stopped(self) -> None:
        """在线程确实结束后回收引用，再对最新画面补做一次推理。"""
        thread = self.sender()
        # 只清理发出 finished 的线程；旧回调不能清除其他推理任务的引用。
        if thread is None or thread is not self.validation_thread:
            return
        # QThread.finished 在 run() 返回后发出，此时线程已结束，可以安全释放引用。
        self.validation_thread = None
        self._sync_ui_state()
        # 只比较当前最新请求与刚结束的请求；中间帧不排队、不追赶。
        # 即使播放/拖动期间经过很多帧，也只会再推理此刻最新的那一帧。
        current_signature = self._validation_request_signature()
        if current_signature is not None and current_signature != getattr(
            thread, "validation_signature", None
        ):
            self._run_validation()

    def _on_validation_error(self, file_path: str, frame_index: int, error: str) -> None:
        thread = self.sender()
        if (
            self.validation_mode
            and self._validation_source_key() == file_path
            and thread is not None
            and getattr(thread, "validation_generation", -1) == self.validation_generation
        ):
            # 单帧推理失败不清除上一组有效结果，也不弹出模态窗口打断播放。
            prefix = (
                "摄像头实时验证失败"
                if self.is_camera
                else f"实时验证失败（第 {frame_index + 1} 帧）"
            )
            self.status_lbl.setText(f"{prefix}，保留上次检测结果：{error}")

    def _run_ai_predict(self) -> None:
        if self.ai_model is None:
            QMessageBox.warning(
                self,
                "模型不可用",
                "请先加载 YOLO 权重。",
            )
            return
        if self.frame is None or self.current_file is None:
            return
        if self.predicting or self.validation_thread is not None:
            self.status_lbl.setText("当前 YOLO 推理尚未完成，请稍后再启动 AI 预标注。")
            return

        self._stop_playback()
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

    def _map_ai_boxes_to_dataset(
        self, new_boxes: list[dict]
    ) -> tuple[list[dict], int]:
        """将模型检测类别映射到当前数据集类别，供普通标注和摄像头归档共用。"""
        if self.ai_model is None:
            return [], len(new_boxes)
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
                "confidence": float(box.get("confidence", 0.0)),
            })
        return mapped_boxes, skipped

    def _apply_ai_boxes(
        self,
        frame_index: int,
        new_boxes: list[dict],
        ensure_empty_frame: bool = False,
        replace_existing: bool = False,
        refresh_current: bool = True,
    ) -> tuple[int, int]:
        """映射模型类别；单帧预标注可替换旧框，批量预标注默认合并。"""
        mapped_boxes, skipped = self._map_ai_boxes_to_dataset(new_boxes)

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
            self._set_dirty(True, frame_index=frame_index)
        if refresh_current and self.frame_idx == frame_index and self.frame is not None:
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
        # 同一个工具栏按钮在批量任务运行期间承担停止操作。
        if self.batch_thread is not None:
            if self.batch_thread.isRunning():
                self._stop_batch_ai_predict()
            else:
                self.status_lbl.setText("上一批量任务正在结束，请稍后再启动新的任务。")
            return
        if self.validation_thread is not None:
            self.status_lbl.setText("实时验证推理尚未完成，请稍后再启动批量 AI 预标注。")
            return
        if self.ai_model is None:
            QMessageBox.warning(self, "模型不可用", "请先加载 YOLO 权重。")
            return

        is_camera_source = bool(
            self.is_camera
            and self.camera_thread is not None
            and self.camera_thread.isRunning()
        )
        if self.validation_mode:
            self.status_lbl.setText("请先关闭实时验证，再启动自动批量预标注。")
            return
        if not is_camera_source and not (
            (self.is_video or self.is_image_sequence) and self.current_file is not None
        ):
            QMessageBox.information(
                self, "需要媒体输入", "请先选择视频、图片组或打开本机摄像头。"
            )
            return
        if self.predicting:
            self.status_lbl.setText("当前已有 AI 推理任务正在执行。")
            return
        if not is_camera_source and self.total_frames <= 0:
            QMessageBox.warning(self, "无法批量标注", "无法读取媒体数量，暂不能批量处理。")
            return

        if is_camera_source:
            start_frame = end_frame = 1
            dialog_total_frames = 1
            source_kind = "camera"
        else:
            try:
                start_frame, end_frame = normalize_batch_frame_range(
                    self.total_frames, self.batch_start_frame, self.batch_end_frame
                )
            except ValueError as exc:
                QMessageBox.warning(self, "批量处理范围无效", str(exc))
                return
            dialog_total_frames = self.total_frames
            source_kind = "image_sequence" if self.is_image_sequence else "video"

        sampling_dialog = BatchSamplingDialog(
            start_frame=start_frame,
            end_frame=end_frame,
            total_frames=dialog_total_frames,
            frame_step=self.frame_step,
            overlap_threshold=self.batch_overlap_threshold,
            orb_nfeatures=self.orb_nfeatures,
            orb_max_side=self.orb_max_side,
            smart_base_step=self.smart_base_step,
            source_kind=source_kind,
            parent=self,
        )
        if sampling_dialog.exec_() != QDialog.Accepted:
            return

        sampling_values = sampling_dialog.values()
        overlap_threshold = sampling_values["overlap_threshold"]
        start_frame = sampling_values["start_frame"]
        end_frame = sampling_values["end_frame"]
        step = sampling_values["frame_step"]
        orb_nfeatures = sampling_values["orb_nfeatures"]
        orb_max_side = sampling_values["orb_max_side"]
        smart_base_step = sampling_values["smart_base_step"]

        self.batch_start_frame = start_frame
        self.batch_end_frame = end_frame
        self.orb_nfeatures = orb_nfeatures
        self.orb_max_side = orb_max_side
        self.batch_overlap_threshold = overlap_threshold
        if not self.is_image_sequence:
            self.frame_step = step
            self.smart_base_step = smart_base_step
        # 图片组的检测步长固定为 1；摄像头与视频共用自适应步长参数。
        self.app_settings.setValue("video/frame_step", self.frame_step)
        self.app_settings.setValue("video/orb_nfeatures", self.orb_nfeatures)
        self.app_settings.setValue("video/orb_max_side", self.orb_max_side)
        self.app_settings.setValue("video/smart_base_step", self.smart_base_step)
        self.app_settings.sync()

        orb_enabled = self.orb_nfeatures > 0
        if source_kind == "camera":
            sampling_description = (
                f"智能重叠率 {overlap_threshold:.0f}%，配准失败兜底 {step} 帧"
                if orb_enabled else f"固定间隔 {step} 帧（ORB 已关闭）"
            )
        elif self.is_image_sequence:
            sampling_description = (
                f"逐张检测，重叠率阈值 {overlap_threshold:.0f}%"
                if orb_enabled else "逐张标注（ORB 已关闭）"
            )
        else:
            sampling_description = (
                f"智能重叠率 {overlap_threshold:.0f}%，配准失败兜底 {step} 帧"
                if orb_enabled else f"固定间隔 {step} 帧（ORB 已关闭）"
            )

        self._stop_playback()
        self._batch_source_kind = source_kind
        self._batch_restore_state = None if is_camera_source else (
            self.frame,
            self.frame_idx,
            self.image_w,
            self.image_h,
            self.current_file,
            self.image_sequence_index,
        )
        self._batch_final_status_text = ""
        self.batch_added_count = 0
        self.batch_captured_count = 0
        self.batch_error_count = 0
        self.batch_skipped_count = 0
        self._camera_batch_new_captures = []
        self._camera_batch_session_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            if is_camera_source else None
        )
        self._batch_stopped_by_user = False
        self._batch_stop_requested = False
        self.predicting = True
        if is_camera_source:
            self.status_lbl.setText(
                f"摄像头自动预标注已启动：{sampling_description}；仅归档检测到目标的画面。"
            )
        else:
            unit = "张图片" if self.is_image_sequence else "帧"
            self.status_lbl.setText(
                f"批量预标注：第 {start_frame}–{end_frame} {unit}，{sampling_description}。"
            )

        self.batch_thread = BatchPredictThread(
            self.ai_model,
            str(self.current_file) if self.is_video else None,
            self.total_frames if not is_camera_source else 0,
            step,
            self.inference_conf,
            self.inference_iou,
            self._active_imgsz_values(),
            self.inference_max_det,
            self.inference_min_box_side,
            start_frame=start_frame,
            end_frame=end_frame,
            overlap_threshold=overlap_threshold,
            orb_nfeatures=self.orb_nfeatures,
            orb_max_side=self.orb_max_side,
            smart_base_step=1 if self.is_image_sequence else self.smart_base_step,
            image_paths=list(self.current_image_sequence) if self.is_image_sequence else None,
            camera_thread=self.camera_thread if is_camera_source else None,
        )
        self.batch_thread.frame_result.connect(self._on_batch_frame_result)
        self.batch_thread.progress_signal.connect(self._on_batch_progress)
        self.batch_thread.error_signal.connect(self._on_batch_error)
        self.batch_thread.completed_signal.connect(self._on_batch_completed)
        self.batch_thread.finished.connect(self._on_batch_thread_stopped)
        self.batch_thread.start()
        self._sync_ui_state()

    def _on_batch_frame_result(
        self,
        frame_index: int,
        new_boxes: Optional[list],
        image_w: int,
        image_h: int,
        preview_frame: np.ndarray,
    ) -> None:
        if self._batch_source_kind == "camera":
            self._store_camera_batch_result(
                frame_index, new_boxes or [], image_w, image_h, preview_frame
            )
            return

        # 视频和图片组共用同一结果通道；只在 YOLO 触发后刷新预览。
        if self.is_image_sequence:
            if not 0 <= frame_index < len(self.current_image_sequence):
                return
            image_path = self.current_image_sequence[frame_index]
            self.current_file = image_path
            self.image_sequence_index = frame_index
            self.image_sequence_sizes[frame_index] = (image_w, image_h)
            existing_label = (
                self._existing_image_label_path(image_path)
                if frame_index not in self.annotations else None
            )
            if existing_label is not None:
                self._load_yolo_txt(
                    existing_label, frame_index, image_w=image_w, image_h=image_h
                )
            self.frame_idx = frame_index
            self._update_window_title()
        else:
            self.frame_idx = frame_index
            # 先载入视频帧磁盘上已有的标签，避免覆盖之前的人工标注。
            if frame_index not in self.annotations:
                existing_label = self._existing_video_frame_label_path(frame_index)
                if existing_label is not None:
                    self._load_yolo_txt(
                        existing_label, frame_index, image_w=image_w, image_h=image_h
                    )

        self.frame = preview_frame
        self.image_w = image_w
        self.image_h = image_h
        if new_boxes is not None:
            added, skipped = self._apply_ai_boxes(
                frame_index, new_boxes, ensure_empty_frame=True, refresh_current=False
            )
            self.batch_added_count += added
            self.batch_skipped_count += skipped
        self._refresh_frame(refresh_tree=False)

    def _store_camera_batch_result(
        self,
        frame_index: int,
        new_boxes: list[dict],
        image_w: int,
        image_h: int,
        preview_frame: np.ndarray,
    ) -> None:
        """将有效摄像头预标注结果以 JPG/TXT 成对写入磁盘。"""
        if not new_boxes or preview_frame is None:
            return

        image_path: Optional[Path] = None
        label_path: Optional[Path] = None
        temp_label: Optional[Path] = None
        temp_classes: Optional[Path] = None
        resolved_path: Optional[Path] = None
        saved_image = False
        saved_label = False
        try:
            capture_dir, context_key = self._camera_capture_destination()
            session_id = getattr(self, "_camera_batch_session_id", None)
            if not session_id:
                session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                self._camera_batch_session_id = session_id
            image_path = capture_dir / (
                f"camera_batch_{session_id}_{int(frame_index) + 1:09d}.jpg"
            )
            resolved_path = image_path.resolve()

            self.file_dataset_keys[resolved_path] = context_key
            self._activate_file_dataset_context(image_path)
            mapped_boxes, skipped = self._map_ai_boxes_to_dataset(new_boxes)
            self.batch_skipped_count += skipped
            if not mapped_boxes or not self._has_valid_boxes_for_save(
                mapped_boxes, int(image_w), int(image_h)
            ):
                self.file_dataset_keys.pop(resolved_path, None)
                if skipped:
                    self.status_lbl.setText(
                        f"摄像头检测到目标，但 {skipped} 个检测框与数据集类别不匹配，未保存该画面。"
                    )
                return

            capture_dir.mkdir(parents=True, exist_ok=True)
            label_path = self._image_label_path(image_path, search_if_missing=True)
            if image_path.exists() or label_path.exists():
                raise FileExistsError(f"摄像头归档目标已存在：{image_path.name}")

            temp_token = f"{os.getpid()}_{threading.get_ident()}"
            temp_label = label_path.with_name(f".{label_path.name}.{temp_token}.tmp")
            classes_file = self.classes_file
            if classes_file is not None:
                temp_classes = classes_file.with_name(f".{classes_file.name}.{temp_token}.tmp")

            if not save_image_unicode(image_path, preview_frame):
                raise OSError(f"摄像头检测图片保存失败：{image_path}")
            saved_image = True
            write_yolo_label(temp_label, mapped_boxes, int(image_w), int(image_h))
            if temp_classes is not None and self.classes:
                temp_classes.parent.mkdir(parents=True, exist_ok=True)
                temp_classes.write_text("\n".join(self.classes) + "\n", encoding="utf-8")

            os.replace(temp_label, label_path)
            saved_label = True
            if temp_classes is not None and self.classes:
                os.replace(temp_classes, classes_file)

            context = self.dataset_contexts.get(context_key)
            if context is not None:
                context.setdefault("label_index_cache", {}).clear()
                self._store_active_dataset_context_state()
            self._camera_batch_new_captures.append((image_path, context_key))
            self.batch_captured_count += 1
            self.batch_added_count += len(mapped_boxes)
        except Exception as exc:
            if temp_label is not None:
                temp_label.unlink(missing_ok=True)
            if temp_classes is not None:
                temp_classes.unlink(missing_ok=True)
            if saved_label and label_path is not None:
                label_path.unlink(missing_ok=True)
            if saved_image and image_path is not None:
                image_path.unlink(missing_ok=True)
            if resolved_path is not None:
                self.file_dataset_keys.pop(resolved_path, None)
            self.batch_error_count += 1
            self.status_lbl.setText(
                f"摄像头检测结果归档失败：{type(exc).__name__}: {exc}"
            )
        self._sync_ui_state()

    def _on_batch_progress(
        self,
        current_frame: int,
        total_frames: int,
        latest_overlap_percent: Optional[float] = None,
    ) -> None:
        overlap_text = ""
        if self.orb_nfeatures > 0:
            overlap_value = (
                f"{latest_overlap_percent:.1f}%"
                if latest_overlap_percent is not None else "--"
            )
            overlap_text = f"；最近有效重叠率 {overlap_value}"
        if self._batch_source_kind == "camera":
            self.status_lbl.setText(
                f"摄像头自动预标注：采集帧 {current_frame}"
                f"{overlap_text}；已归档 {self.batch_captured_count} 张，"
                f"新增 {self.batch_added_count} 个检测框。"
            )
        else:
            unit = "张" if self.is_image_sequence else "帧"
            self.status_lbl.setText(
                f"批量 AI 预标注进行中：已扫描所选范围 {current_frame} / {total_frames} {unit}"
                f"{overlap_text}；新增 {self.batch_added_count} 个检测框。"
            )

    def _on_batch_error(self, frame_index: int, error: str) -> None:
        self.batch_error_count += 1
        if self._batch_source_kind == "camera":
            source_text = f"摄像头采集帧 {frame_index + 1}" if frame_index >= 0 else "摄像头采集"
        elif frame_index >= 0 and self.is_image_sequence and frame_index < len(self.current_image_sequence):
            source_text = self.current_image_sequence[frame_index].name
        else:
            source_text = f"第 {frame_index + 1} 帧" if frame_index >= 0 else "媒体读取"
        self.status_lbl.setText(f"批量 AI 预标注遇到错误（{source_text}）：{error}")

    def _on_batch_completed(
        self,
        sampled: int,
        failed: int,
        stopped: bool,
        fatal_error: bool,
        valid_comparisons: int = 0,
        fallback_samples: int = 0,
    ) -> None:
        self._batch_stopped_by_user = stopped
        self.batch_error_count = max(self.batch_error_count, failed)
        if fatal_error:
            state_text = "因推理错误提前终止（请检查显存；必要时降低推理输入尺寸）"
        else:
            state_text = "已停止" if stopped else "完成"
        if self._batch_source_kind == "camera":
            result_text = (
                f"摄像头自动预标注{state_text}：YOLO 推理 {sampled} 次，"
                f"归档 {self.batch_captured_count} 张有效图片，新增 {self.batch_added_count} 个检测框，"
                f"失败 {self.batch_error_count} 次，类别不匹配跳过 {self.batch_skipped_count} 个框。"
            )
        else:
            unit = "张图片" if self.is_image_sequence else "个采样帧"
            result_text = (
                f"批量 AI 预标注{state_text}：处理 {sampled} {unit}，"
                f"新增 {self.batch_added_count} 个检测框，失败 {self.batch_error_count} 次，"
                f"因类别名称不匹配跳过 {self.batch_skipped_count} 个检测框。"
            )
        if self.orb_nfeatures > 0:
            result_text += (
                f" ORB 有效配准 {valid_comparisons} 次，"
                f"兜底采样 {fallback_samples} 次。"
            )
        self._batch_final_status_text = result_text
        self.status_lbl.setText(result_text)

    def _on_batch_thread_stopped(self) -> None:
        # 扫描线程使用独立输入源；恢复任务开始前选中的媒体与画面。
        if self._camera_batch_new_captures:
            self._register_camera_captures(self._camera_batch_new_captures)
            self._camera_batch_new_captures = []
        restore_state = self._batch_restore_state
        if restore_state is not None:
            (
                self.frame,
                self.frame_idx,
                self.image_w,
                self.image_h,
                self.current_file,
                self.image_sequence_index,
            ) = restore_state
            self._batch_restore_state = None
            self._refresh_frame()
            self._update_window_title()

        self.predicting = False
        self.batch_thread = None
        self._batch_stop_requested = False
        self._batch_source_kind = None
        if self._camera_thread_pending_delete is not None:
            self._camera_thread_pending_delete.deleteLater()
            self._camera_thread_pending_delete = None
        self._sync_ui_state()
        if self._batch_final_status_text:
            self.status_lbl.setText(self._batch_final_status_text)
        if self._close_after_batch:
            self._close_after_batch = False            QTimer.singleShot(0, self.close)

    def _stop_batch_ai_predict(self) -> None:
        self._stop_playback()
        if (
            self.batch_thread is not None
            and self.batch_thread.isRunning()
            and not self._batch_stop_requested
        ):
            self._batch_stop_requested = True
            self.batch_thread.request_stop()
            self._sync_ui_state()
            self.status_lbl.setText("正在停止批量 AI 预标注，等待当前帧推理结束……")

    def _on_ai_error(self, file_path: str, frame_index: int, error: str) -> None:
        if self.current_file is not None and str(self.current_file) == file_path:
            self.status_lbl.setText(f"AI 预标注失败：{error}")
            QMessageBox.critical(self, "AI 预标注失败", error)

    def _on_ai_thread_stopped(self) -> None:
        self.predicting = False
        self.predict_thread = None
        self._sync_ui_state()
        if self.is_playing:
            self._schedule_playback_step()

    # ---------- 清理标注 ----------

    def _clear_current_frame_annotations(self) -> None:
        """清空当前图片/视频帧全部标注，并将其作为待保存的空标签记录。"""
        if self.current_file is None or self.frame is None or self.predicting:
            return

        frame_key = self.frame_idx if (self.is_video or self.is_image_sequence) else 0
        existing_boxes = self.annotations.get(frame_key, [])
        box_count = len(existing_boxes)
        frame_description = (
            f"视频第 {frame_key + 1} 帧" if self.is_video
            else f"图片组第 {frame_key + 1} 张" if self.is_image_sequence
            else self.current_file.name
        )
        answer = QMessageBox.question(
            self,
            "清空当前画面",
            f"确定清空当前画面的 {box_count} 个标注框？\n"
            "保存后会删除对应标签；视频帧还会删除抽帧缓存图。原始照片和视频不会删除。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        # 保留空列表作为待保存的删除标记；保存时删除空标签，而不是写入负样本 TXT。
        self.annotations[frame_key] = []
        self._set_dirty(True)
        self._refresh_frame()
        self._sync_ui_state()
        self.status_lbl.setText(
            f"已清空 {frame_description} 的标注；保存后将删除对应标签。"
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
        cache_unit = "张图片" if self.is_image_sequence else "帧"
        cache_scope = "当前图片组" if self.is_image_sequence else "当前文件"
        answer = QMessageBox.question(
            self,
            "放弃修改",
            f"将丢弃{cache_scope}全部未保存修改（{cached_frame_count} {cache_unit}、{cached_box_count} 个框）。\n"
            "磁盘上已保存的标签不会删除。是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        current_frame_index = self.frame_idx
        self.annotations.clear()
        self._set_dirty(False)

        # 缓存清空后，重新从磁盘读取当前画面的持久化标签；其他帧/图片在跳转时按需加载。
        if self.is_video:
            self._seek(current_frame_index)
        elif self.is_image_sequence:
            label_path = self._existing_image_label_path(self.current_file)
            if label_path is not None:
                self._load_yolo_txt(label_path, current_frame_index)
            self._refresh_frame()
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

    @staticmethod
    def _has_valid_boxes_for_save(boxes: list[dict], image_w: int, image_h: int) -> bool:
        """判断标注列表中是否至少有一个能写入 YOLO 文件的有效框。"""
        if image_w <= 0 or image_h <= 0:
            return False
        for box in boxes:
            try:
                class_id = int(box["class_id"])
                if class_id < 0:
                    continue
                x1 = max(0.0, min(float(box["x1"]), float(image_w)))
                y1 = max(0.0, min(float(box["y1"]), float(image_h)))
                x2 = max(0.0, min(float(box["x2"]), float(image_w)))
                y2 = max(0.0, min(float(box["y2"]), float(image_h)))
                if x2 > x1 and y2 > y1:
                    return True
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        return False

    def _delete_label_file(self, label_path: Path) -> int:
        """删除已解析的标签路径，不重新搜索候选文件。"""
        if not label_path.is_file():
            return 0
        try:
            label_path.unlink()
            return 1
        except OSError as exc:
            raise OSError(f"无法删除空标签文件：{label_path}（{exc}）") from exc

    def _invalidate_label_index_cache(self) -> None:
        """保存操作修改磁盘标签后，清除当前数据集的索引缓存。"""
        context = self.dataset_contexts.get(self.active_dataset_key or "")
        if context is not None:
            context.setdefault("label_index_cache", {}).clear()

    def _delete_video_frame_images(self, frame_index: int) -> int:
        """删除当前输出目录中对应的视频帧缓存图片。"""
        if self.frame_images_dir is None:
            return 0
        candidate = self._video_frame_image_path(frame_index)
        try:
            if candidate.is_file():
                candidate.unlink()
                return 1
        except OSError as exc:
            raise OSError(f"无法删除视频帧缓存图片：{candidate}（{exc}）") from exc
        return 0

    def _save(self) -> bool:
        if self.current_file is None or self.current_labels_dir is None:
            return False
        if self.frame is None:
            QMessageBox.warning(self, "无法保存", "当前没有有效画面。")
            return False
        if self.is_image_sequence:
            if not self.dirty_annotation_indices:
                self.status_lbl.setText("图片组没有待保存的修改。")
                return False
        elif not self.annotations:
            self.status_lbl.setText("没有待保存的标注。清空画面后可保存为空标签。")
            return False
        elif not self.is_video and 0 not in self.annotations:
            self.status_lbl.setText("当前图片没有标注缓存可保存。")
            return False

        try:
            if self.is_video:
                if self.cap is None:
                    raise RuntimeError("视频对象已关闭，请重新打开视频。")

                restore_index = self.frame_idx
                frame_indices = sorted(self.annotations)
                # 复用载入时解析出的标签路径，确保读取、保存和删除作用于同一文件。
                label_targets = {
                    frame_index: self._video_frame_label_path(frame_index, search_if_missing=False)
                    for frame_index in frame_indices
                }

                saved_frames = 0
                deleted_labels = 0
                deleted_frame_images = 0
                for frame_index in frame_indices:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"读取视频第 {frame_index + 1} 帧失败，保存中止。")

                    height, width = frame.shape[:2]
                    boxes = self.annotations[frame_index]
                    label_path = label_targets[frame_index]
                    if self._has_valid_boxes_for_save(boxes, width, height):
                        image_path = self._video_frame_image_path(frame_index)
                        if not save_image_unicode(image_path, frame):
                            raise OSError(f"视频帧图片保存失败：{image_path}")
                        write_yolo_label(label_path, boxes, width, height)
                        saved_frames += 1
                    else:
                        deleted_labels += self._delete_label_file(label_path)
                        deleted_frame_images += self._delete_video_frame_images(frame_index)

                self._seek(restore_index)
                saved_info = (
                    f"已保存 {saved_frames} 个带标注的视频帧；删除 {deleted_labels} 个空标签文件，"
                    f"删除 {deleted_frame_images} 张对应的帧缓存图片。"
                )
            elif self.is_image_sequence:
                saved_images = 0
                deleted_labels = 0
                modified_indices = sorted(self.dirty_annotation_indices)
                for image_index in modified_indices:
                    if not 0 <= image_index < len(self.current_image_sequence):
                        raise RuntimeError(f"图片组索引无效：{image_index}")
                    if image_index not in self.annotations:
                        raise RuntimeError(f"第 {image_index + 1} 张图片的标注缓存不存在，保存中止。")
                    image_path = self.current_image_sequence[image_index]
                    dimensions = self.image_sequence_sizes.get(image_index)
                    if dimensions is None:
                        source_image = load_image_unicode(image_path)
                        if source_image is None:
                            raise RuntimeError(f"无法读取第 {image_index + 1} 张图片：{image_path}")
                        image_h, image_w = source_image.shape[:2]
                        self.image_sequence_sizes[image_index] = (image_w, image_h)
                    else:
                        image_w, image_h = dimensions
                    label_path = self._image_label_path(
                        image_path, search_if_missing=False
                    )
                    boxes = self.annotations[image_index]
                    if self._has_valid_boxes_for_save(boxes, image_w, image_h):
                        write_yolo_label(label_path, boxes, image_w, image_h)
                        saved_images += 1
                    else:
                        deleted_labels += self._delete_label_file(label_path)
                saved_info = (
                    f"图片组保存完成：写入 {saved_images} 张，删除 {deleted_labels} 个空标签文件；"
                    "原始图片未更改。"
                )
            else:
                boxes = self.annotations[0]
                label_path = self._image_label_path(self.current_file, search_if_missing=False)
                if self._has_valid_boxes_for_save(boxes, self.image_w, self.image_h):
                    write_yolo_label(label_path, boxes, self.image_w, self.image_h)
                    saved_info = f"已保存：{label_path}"
                else:
                    deleted_labels = self._delete_label_file(label_path)
                    saved_info = f"当前图片没有有效标注框，已删除 {deleted_labels} 个标签文件；原始图片未更改。"

            if self.classes and self.classes_file is not None:
                self.classes_file.parent.mkdir(parents=True, exist_ok=True)
                self.classes_file.write_text(
                    "\n".join(self.classes) + "\n", encoding="utf-8"
                )

            self._set_dirty(False)
            self._store_active_dataset_context_state()
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
        finally:
            self._invalidate_label_index_cache()

    def _delete_current_image_file(self) -> None:
        """删除当前图片及其精确匹配的标签；图片组仅删除当前成员。"""
        if (
            self.current_file is None
            or self.frame is None
            or self.is_video
            or self.is_camera
            or self.predicting
            or (self.batch_thread is not None and self.batch_thread.isRunning())
        ):
            return

        image_path = self.current_file.resolve()
        if image_path.suffix.casefold() not in IMAGE_SUFFIXES or not image_path.is_file():
            return

        sequence = list(self.current_image_sequence) if self.is_image_sequence else [self.current_file]
        sequence_paths = [Path(item).resolve() for item in sequence]
        current_index = self.image_sequence_index if self.is_image_sequence else 0
        if not 0 <= current_index < len(sequence_paths) or sequence_paths[current_index] != image_path:
            current_index = next((i for i, item in enumerate(sequence_paths) if item == image_path), 0)

        answer = QMessageBox.question(
            self,
            "删除文件",
            "删除当前图片？\n对应的标注文件也将删除，此操作不可恢复。",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if answer != QMessageBox.Yes:
            return

        # 先缓存同组其他成员的未保存标注，删除当前成员时不影响它们。
        preserved = {}
        if self.is_image_sequence:
            for index, member in enumerate(sequence_paths):
                if index == current_index:
                    continue
                if index in self.annotations:
                    preserved[member] = (
                        [dict(box) for box in self.annotations[index]],
                        self.image_sequence_sizes.get(index),
                        index in self.dirty_annotation_indices,
                    )

        self._activate_file_dataset_context(image_path)
        label_path = self._image_label_path(Path(image_path), search_if_missing=True)
        label_is_shared = False
        if label_path.is_file():
            active_context_key = self.active_dataset_key
            for other_path, other_context_key in list(self.file_dataset_keys.items()):
                if other_path == image_path or other_context_key != active_context_key:
                    continue
                if not other_path.is_file() or other_path.suffix.casefold() not in IMAGE_SUFFIXES:
                    continue
                other_label = self._image_label_path(Path(other_path), search_if_missing=True)
                if other_label.is_file() and other_label.resolve() == label_path.resolve():
                    label_is_shared = True
                    break

        paths_to_remove = [(image_path, image_path.with_name(f".{image_path.name}.deleting"))]
        if label_path.is_file() and not label_is_shared:
            paths_to_remove.append((label_path, label_path.with_name(f".{label_path.name}.deleting")))

        moved = []
        try:
            for original, temporary in paths_to_remove:
                if temporary.exists():
                    temporary = temporary.with_name(temporary.name + f"_{os.getpid()}_{threading.get_ident()}")
                os.replace(original, temporary)
                moved.append((original, temporary))
        except OSError as exc:
            for original, temporary in reversed(moved):
                try:
                    os.replace(temporary, original)
                except OSError:
                    pass
            QMessageBox.critical(self, "删除失败", f"文件未能完整删除：\n{exc}")
            return

        cleanup_errors = []
        for _original, temporary in moved:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_errors.append(str(exc))

        self._clear_media_label_path_cache(Path(image_path))
        self.file_dataset_keys.pop(image_path, None)
        context = self.dataset_contexts.get(self.active_dataset_key or "")
        if context is not None:
            context.setdefault("label_index_cache", {}).clear()

        # 展开旧图片组、移除单张图片后重新分组，确保剩余成员可继续单独打开。
        entries = []
        for item in self.file_list:
            members = self.file_image_sequences.get(item.resolve())
            entries.extend(members if members else [item])
        entries = [item for item in entries if item.resolve() != image_path]
        self.file_list, self.file_image_sequences = group_image_sequences(entries)
        self.file_index = min(max(0, self.file_index), len(self.file_list) - 1) if self.file_list else -1

        # 优先跳转原图片组中当前成员之后的图片；当前成员已是最后一张时再循环到组首。
        ordered_group = sequence_paths[current_index + 1:] + sequence_paths[:current_index]
        next_group_member = next(
            (item for item in ordered_group if item != image_path and item.is_file()),
            None,
        )
        target_path = None
        target_sequence_index = 0
        if next_group_member is not None:
            for item in self.file_list:
                members = self.file_image_sequences.get(item.resolve())
                if members:
                    member_index = next(
                        (i for i, member in enumerate(members) if member.resolve() == next_group_member.resolve()),
                        None,
                    )
                    if member_index is not None:
                        target_path = item
                        target_sequence_index = member_index
                        break
                elif item.resolve() == next_group_member.resolve():
                    target_path = item
                    break
        if target_path is None and self.file_list:
            target_path = self.file_list[self.file_index]

        self.annotations = {}
        self.dirty_annotation_indices.clear()
        self.image_sequence_sizes = {}
        self._set_dirty(False)
        if target_path is not None:
            target_index = next(i for i, item in enumerate(self.file_list) if item.resolve() == target_path.resolve())
            self.file_index = target_index
            if self._load_file(target_path):
                if preserved:
                    current_members = [item.resolve() for item in self.current_image_sequence] if self.is_image_sequence else [self.current_file.resolve()]
                    for index, member in enumerate(current_members):
                        record = preserved.get(member)
                        if record is None:
                            continue
                        boxes, dimensions, was_dirty = record
                        self.annotations[index] = boxes
                        if dimensions is not None:
                            self.image_sequence_sizes[index] = dimensions
                        if was_dirty and self.is_image_sequence:
                            self.dirty_annotation_indices.add(index)
                    self.dirty = bool(self.dirty_annotation_indices)
                    self._update_window_title()
                    self._refresh_frame()
                if self.is_image_sequence and target_sequence_index:
                    self._display_image_sequence_frame(target_sequence_index)
        else:
            self.current_file = None
            self.is_video = False
            self.is_image_sequence = False
            self.current_image_sequence = []
            self.image_sequence_index = 0
            self.total_frames = 0
            self.frame_idx = 0
            self.frame = None
            self.annotations = {}
            self.slider.setRange(0, 0)
            self._refresh_frame()
            self._update_window_title()
        self._refresh_file_list()
        self._sync_ui_state()
        if cleanup_errors:
            self.status_lbl.setText("图片已从列表移除，但临时文件清理失败。")
        elif label_is_shared:
            self.status_lbl.setText(f"已删除图片：{image_path.name}；共用标注文件已保留。")
        else:
            self.status_lbl.setText(f"已删除图片及对应标注：{image_path.name}")

    # ---------- 窗口状态与退出 ----------

    def _update_window_title(self) -> None:
        title = f" - {self.current_file.name}" if self.current_file else ""
        marker = " *" if self.dirty else ""
        self.setWindowTitle(f"YOLO Annotation Review Studio{title}{marker}")

    def _set_dirty(self, value: bool, frame_index: Optional[int] = None) -> None:
        self.dirty = bool(value)
        if self.is_image_sequence:
            if self.dirty:
                index = self.frame_idx if frame_index is None else int(frame_index)
                if 0 <= index < len(self.current_image_sequence):
                    self.dirty_annotation_indices.add(index)
            else:
                self.dirty_annotation_indices.clear()
        self._update_window_title()
        # 缓存清理按钮只在当前媒体确实存在未保存修改时启用。
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
        box.setText("当前有未保存的修改。是否保存？")
        save_button = box.addButton("保存", QMessageBox.AcceptRole)
        discard_button = box.addButton("放弃修改", QMessageBox.DestructiveRole)
        cancel_button = box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(save_button)
        box.exec_()

        clicked = box.clickedButton()
        if clicked is save_button:
            return self._save()
        if clicked is discard_button:
            # 放弃修改时恢复该数据集最近一次已保存/载入的类别状态，避免把未保存的新类别
            # 缓存进数据集上下文，并在之后切回时继续使用。
            if self.active_dataset_key:
                context = self.dataset_contexts.get(self.active_dataset_key, {})
                cached_classes = context.get("classes")
                if cached_classes is not None:
                    self.classes = list(cached_classes)
                self.last_class_name = context.get("last_class_name")
            self._set_dirty(False)
            return True
        return False

    def _toggle_playback(self) -> None:
        """视频模式切换播放；摄像头验证模式切换播放，标注模式执行抓拍。"""
        if self.is_camera:
            if not self.validation_mode:
                self._capture_camera_frame()
                return
            self.is_playing = not self.is_playing
            if self.is_playing:
                self.camera_timer.start()
                self._update_camera_frame()
                self.status_lbl.setText("摄像头实时验证已继续。")
            else:
                self.status_lbl.setText("摄像头实时验证已暂停。")
            self._update_playback_action()
            return

        if self.is_playing:
            self._stop_playback()
            self.status_lbl.setText("视频播放已暂停。")
            return

        if not self.is_video or self.cap is None or self.total_frames <= 1:
            return
        if self.predicting:
            self.status_lbl.setText("当前 AI 预标注任务尚未完成，暂时无法播放。")
            return

        self.is_playing = True
        self._update_playback_action()
        if self.frame_idx >= self.total_frames - 1:
            self._seek(0)
        if not self.predicting:
            self.status_lbl.setText("视频正在播放。")
            self._schedule_playback_step()

    def _update_playback_action(self) -> None:
        if not hasattr(self, "playback_action"):
            return
        if self.is_camera and not self.validation_mode:
            self.playback_action.setText("抓拍")
            self.playback_action.setToolTip("抓拍当前画面。")
        elif self.is_camera:
            self.playback_action.setText("暂停" if self.is_playing else "播放")
            self.playback_action.setToolTip("暂停摄像头画面。" if self.is_playing else "继续摄像头画面。")
        else:
            self.playback_action.setText("暂停" if self.is_playing else "播放")
            self.playback_action.setToolTip("暂停视频。" if self.is_playing else "播放视频。")

    def _schedule_playback_step(self) -> None:
        if not self.is_playing or not self.is_video or self.predicting:
            return
        fps = self.video_fps if np.isfinite(self.video_fps) and self.video_fps > 0 else 30.0
        interval_ms = max(1, int(round(1000.0 / fps)))
        if not self.playback_timer.isActive():
            self.playback_timer.start(interval_ms)
        else:
            self.playback_timer.setInterval(interval_ms)

    def _advance_playback(self) -> None:
        if not self.is_playing or not self.is_video:
            return
        if self.predicting:
            # 标注类 AI 任务仍然独占视频；实时验证不设置 predicting，不会进入此分支。
            self.playback_timer.stop()
            return

        next_index = self.frame_idx + 1
        if next_index >= self.total_frames:
            self._stop_playback()
            self.status_lbl.setText("视频播放结束。")
            return

        # 连续播放必须顺序解码：当前帧读取后，VideoCapture 的内部位置已经指向下一帧。
        # 这里直接 read()，绝不对每一帧调用 CAP_PROP_POS_FRAMES 随机定位。
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self._stop_playback()
            self.status_lbl.setText("视频播放结束或无法继续读取视频帧。")
            return
        self._display_video_frame(next_index, frame)
        if self.frame_idx != next_index:
            self._stop_playback()
            return
        # 周期定时器会按视频 FPS 自动触发下一帧；不等待实时验证线程。

    def _stop_playback(self) -> None:
        self.is_playing = False
        if hasattr(self, "playback_timer"):
            self.playback_timer.stop()
        self._update_playback_action()

    def _sync_ui_state(self) -> None:
        has_file = self.current_file is not None and self.frame is not None
        has_camera = self.is_camera and self.frame is not None
        has_media = has_file or has_camera
        has_video = has_file and self.is_video
        # finished 回调处理前仍保持 busy，避免在 Qt 信号队列清理线程引用前
        # 再次启动同类推理任务。
        validation_busy = self.validation_thread is not None
        self.open_action.setEnabled(not self.predicting)
        self.open_camera_action.setEnabled(
            not self.predicting
            and not (self.batch_thread is not None and self.batch_thread.isRunning())
        )
        self.output_dir_action.setEnabled(
            self.media_dir is not None and not self.predicting
        )
        self.model_action.setEnabled(not self.predicting and not validation_busy)
        self.model_settings_action.setEnabled(not self.predicting and not validation_busy)
        self.validation_mode_action.setEnabled(
            has_media and self.ai_model is not None and not self.predicting
        )
        self.save_action.setEnabled(has_file and not self.predicting)
        self.delete_file_action.setEnabled(
            has_file and not self.is_video and not self.is_camera
            and not self.predicting and not self.validation_mode
        )
        self.canvas.setEnabled(has_media and not self.predicting)
        self.slider.setEnabled(
            has_file and (self.is_video or self.is_image_sequence) and not self.predicting
        )
        self.tree.setEnabled(has_file and not self.predicting and not self.validation_mode)
        self.tree.setVisible(has_file and not self.validation_mode)
        self.ai_action.setEnabled(
            has_file and self.ai_model is not None and not self.predicting
            and not validation_busy and not self.validation_mode
        )
        batch_running = self.batch_thread is not None and self.batch_thread.isRunning()
        self.batch_ai_action.setText(
            "停止批量预标注" if batch_running else "AI 批量预标注"
        )
        if batch_running:
            self.batch_ai_action.setEnabled(not self._batch_stop_requested)
        else:
            self.batch_ai_action.setEnabled(
                self.batch_thread is None
                and (
                    has_video
                    or (has_file and self.is_image_sequence)
                    or has_camera
                )
                and self.ai_model is not None
                and not self.predicting
                and not validation_busy
                and not self.validation_mode
            )
        has_sequence = has_file and self.is_image_sequence
        if self.is_image_sequence:
            self.prev_frame_action.setText("上一张")
            self.prev_frame_action.setToolTip("切换到图片组中的上一张图片。")
            self.next_frame_action.setText("下一张")
            self.next_frame_action.setToolTip("切换到图片组中的下一张图片。")
        else:
            self.prev_frame_action.setText("上一标注帧")
            self.prev_frame_action.setToolTip("跳转到上一条有标注的帧。")
            self.next_frame_action.setText("下一标注帧")
            self.next_frame_action.setToolTip("跳转到下一条有标注的帧。")
        # 视频只跳转带标注框的帧；图片组中每张图片都作为有效导航目标。
        self.prev_frame_action.setEnabled(
            (has_video or has_sequence) and not self.predicting
        )
        self.next_frame_action.setEnabled(
            (has_video or has_sequence) and not self.predicting
        )
        self.playback_action.setEnabled(
            ((has_file and self.is_video and self.total_frames > 1) or has_camera)
            and not self.predicting
        )
        self._update_playback_action()
        self.delete_action.setEnabled(
            self._selected_box_index() >= 0 and not self.predicting and not self.validation_mode
        )
        self.clear_current_action.setEnabled(
            has_file and not self.predicting and not self.validation_mode
        )
        self.clear_cache_action.setToolTip(
            "放弃当前图片组未保存的修改。"
            if self.is_image_sequence else "放弃当前文件未保存的修改。"
        )
        self.clear_cache_action.setEnabled(
            has_file and self.dirty and not self.predicting and not self.validation_mode
        )
        self.file_list_widget.setEnabled(not self.predicting and bool(self.file_list))

    def closeEvent(self, event) -> None:
        if self.batch_thread is not None and self.batch_thread.isRunning():
            # 不在主线程 wait() 阻塞期间丢失批量线程排队返回的检测结果信号。
            self._close_after_batch = True
            self._stop_batch_ai_predict()
            event.ignore()
            self.status_lbl.setText("正在停止批量预标注；完成后将继续退出。")
            return
        if not self._confirm_unsaved_changes():
            event.ignore()
            return
        if self.predict_thread is not None and self.predict_thread.isRunning():
            self.predict_thread.wait()
        if self.validation_thread is not None and self.validation_thread.isRunning():
            self.validation_thread.wait()
        if (self.is_camera or self.camera_thread is not None) and not self._stop_camera():
            event.ignore()
            QMessageBox.warning(
                self, "摄像头尚未停止",
                "摄像头采集线程仍在运行，程序暂不退出，以避免销毁仍在运行的线程。"
            )
            return
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