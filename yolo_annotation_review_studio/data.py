# -*- coding: utf-8 -*-
"""数据集目录、图像读写、YOLO 标签及通用检测框算法。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import ALL_SUFFIXES, VIDEO_SUFFIXES

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
