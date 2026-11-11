# -*- coding: utf-8 -*-
"""应用启动入口。"""
from __future__ import annotations

import sys
from PyQt5.QtWidgets import QApplication

from .window import AnnotatorWindow

def main() -> int:
    app = QApplication(sys.argv)
    window = AnnotatorWindow()
    window.show()
    return app.exec_()

if __name__ == "__main__":
    raise SystemExit(main())
