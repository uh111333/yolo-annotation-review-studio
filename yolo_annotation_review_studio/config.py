# -*- coding: utf-8 -*-
"""应用默认配置与媒体文件类型。"""
from __future__ import annotations

import os
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALL_SUFFIXES = VIDEO_SUFFIXES | IMAGE_SUFFIXES

# 以项目根目录为基准，避免从不同工作目录启动时资源路径发生变化。
SCRIPT_DIR = Path(__file__).resolve().parent.parent
AI_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    r"E:\code\Python\PythonProject1\runs\AC\yolo11n_AC2\weights\best.pt",
)
