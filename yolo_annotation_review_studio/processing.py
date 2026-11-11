# -*- coding: utf-8 -*-
"""后台线程、YOLO 推理与 ORB 智能采样。"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from .data import class_aware_nms, filter_boxes_overlapping_reference

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
        finally:
            capture.release()

    def request_stop(self) -> None:
        self._stop_event.set()

    def latest_frame(self) -> tuple[int, Optional[np.ndarray]]:
        with self._lock:
            if self._latest_frame is None:
                return self._sequence, None
            return self._sequence, self._latest_frame.copy()


class BatchPredictThread(QThread):
    """顺序扫描视频；固定模式按间隔推理，智能模式按 ORB 重叠率自适应检测。"""

    frame_result = pyqtSignal(int, object, int, int, object)
    progress_signal = pyqtSignal(int, int, object)
    error_signal = pyqtSignal(int, str)
    completed_signal = pyqtSignal(int, int, bool, bool, int, int)

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
        start_frame: int = 1,
        end_frame: Optional[int] = None,
        sampling_mode: str = "fixed",
        overlap_threshold: float = 70.0,
        orb_nfeatures: int = 2000,
        orb_max_side: int = 640,
        smart_base_step: int = 5,
    ):
        super().__init__()
        self.model = model
        self.video_path = video_path
        self.total_frames = int(total_frames)
        self.start_frame = max(1, int(start_frame))
        self.end_frame = int(end_frame) if end_frame is not None else self.total_frames
        if self.end_frame < self.start_frame or self.end_frame > self.total_frames:
            raise ValueError("批量推理帧范围超出视频长度或起止顺序无效。")
        self.range_total_frames = self.end_frame - self.start_frame + 1
        self.frame_step = max(1, int(frame_step))
        self.fallback_frame_gap = self.frame_step
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz_values = tuple(int(v) for v in imgsz_values)
        self.max_det = int(max_det)
        self.min_box_side = max(0, int(min_box_side))
        self.sampling_mode = "smart" if str(sampling_mode).lower() == "smart" else "fixed"
        self.overlap_threshold = min(99.0, max(1.0, float(overlap_threshold)))
        self.orb_nfeatures = min(10000, max(100, int(orb_nfeatures)))
        self.orb_max_side = min(1920, max(320, int(orb_max_side)))
        self.smart_base_step = min(30, max(1, int(smart_base_step)))
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
        scanned_frames = 0
        valid_comparisons = 0
        fallback_samples = 0
        latest_overlap_percent: Optional[float] = None
        try:
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频：{self.video_path}")

            start_index = self.start_frame - 1
            end_index = self.end_frame - 1
            if start_index > 0:
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

            smart_mode = self.sampling_mode == "smart"
            orb = cv2.ORB_create(nfeatures=self.orb_nfeatures) if smart_mode else None
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) if smart_mode else None
            reference_features: Optional[ORBFeatureSet] = None
            next_detection_frame = start_index
            last_valid_overlap: Optional[float] = None
            last_valid_frame: Optional[int] = None
            fallback_anchor_frame = start_index
            min_detection_step = max(1, self.smart_base_step // 2)

            frame_index = start_index
            while frame_index <= end_index and not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                scanned_frames = frame_index - start_index + 1
                relative_index = frame_index - start_index
                should_sample = False
                should_detect = False
                used_fallback = False
                yolo_success = False
                current_features: Optional[ORBFeatureSet] = None
                analysis: Optional[ORBPairAnalysis] = None

                if not smart_mode:
                    should_sample = (relative_index % self.frame_step == 0)
                else:
                    should_detect = frame_index >= next_detection_frame
                    if should_detect:
                        try:
                            current_gray = prepare_orb_frame(frame, self.orb_max_side)
                            current_features = extract_orb_features(current_gray, orb)
                        except (ValueError, cv2.error):
                            current_gray = None
                            current_features = None

                        if reference_features is None:
                            should_sample = True
                        else:
                            if current_features is not None:
                                analysis = analyze_orb_pair(
                                    reference_features,
                                    current_features,
                                    current_gray.shape[:2],
                                    matcher,
                                )
                            else:
                                analysis = ORBPairAnalysis(False)

                            if analysis.valid:
                                valid_comparisons += 1
                                fallback_anchor_frame = frame_index
                                latest_overlap_percent = analysis.overlap_percent
                                should_sample = analysis.overlap_percent < self.overlap_threshold
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
                            else:
                                frames_since_valid = max(0, frame_index - fallback_anchor_frame)
                                if frames_since_valid >= self.fallback_frame_gap:
                                    should_sample = True
                                    used_fallback = True
                                next_interval = min_detection_step

                        if should_sample:
                            next_interval = self.smart_base_step
                        else:
                            next_detection_frame = frame_index + max(
                                1, int(next_interval or self.smart_base_step)
                            )

                if should_sample:
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
                        if smart_mode:
                            # 推理失败时不切换参考帧，避免丢弃上一有效覆盖基准。
                            next_detection_frame = frame_index + self.smart_base_step
                    # 每次触发 YOLO 都只发出一次画面更新；None 表示推理失败，不代表空标注。
                    self.frame_result.emit(frame_index, boxes, frame_w, frame_h, frame)
                    sampled += 1

                    if smart_mode:
                        if yolo_success and current_features is not None:
                            # 只在 YOLO 推理成功后更新参考帧；本帧 ORB 特征直接复用。
                            reference_features = current_features
                            last_valid_overlap = None
                            last_valid_frame = None
                            fallback_anchor_frame = frame_index
                        if not yolo_success:
                            fallback_anchor_frame = frame_index
                        # 采样后按基准间隔重新检测，避免逐帧重复推理。
                        next_detection_frame = frame_index + self.smart_base_step
                        if used_fallback:
                            fallback_samples += 1

                if should_sample or (smart_mode and (should_detect or scanned_frames % 10 == 0)):
                    self.progress_signal.emit(
                        scanned_frames, self.range_total_frames, latest_overlap_percent
                    )
                if fatal_error:
                    break
                frame_index += 1

            if scanned_frames:
                self.progress_signal.emit(
                    scanned_frames, self.range_total_frames, latest_overlap_percent
                )
            stopped = self._stop_event.is_set()
        except Exception as exc:
            failed += 1
            self.error_signal.emit(-1, f"{type(exc).__name__}: {exc}")
        finally:
            cap.release()
            self.completed_signal.emit(
                sampled, failed, stopped, fatal_error,
                valid_comparisons, fallback_samples,
            )
