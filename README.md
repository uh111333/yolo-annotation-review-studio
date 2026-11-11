# YOLO Annotation Review Studio

YOLO 图像与视频标注复核工具。项目采用**少量大模块**组织代码：独立算法和可复用界面组件分开，紧密耦合的主窗口流程仍集中管理，不把每组按钮或每个小功能拆成单独文件。

## 项目结构

```text
yolo-annotation-review-studio-project/
├─ run.py                              # 推荐启动入口
├─ yolo-annotation-review-studio.py    # 兼容旧文件名
├─ requirements.txt                   # 基础依赖
├─ requirements-ai.txt                # 含 Ultralytics 的 AI 依赖
├─ pyproject.toml                      # 项目元数据
├─ README.md                           # 使用与维护说明
└─ yolo_annotation_review_studio/
   ├─ config.py                        # 默认路径、媒体格式、模型路径
   ├─ data.py                          # 数据集识别、图像/标签读写、检测框算法
   ├─ processing.py                    # ORB、YOLO 推理、摄像头与批量线程
   ├─ ui.py                            # 标注画布、参数/类别/模型对话框
   ├─ window.py                        # 主窗口及各功能流程
   └─ main.py                          # QApplication 启动入口
```

主窗口保留为一个整体，因为媒体状态、标注缓存、播放、摄像头和批量处理之间存在较多直接协作。只有当 `window.py` 后续继续明显膨胀，或某一部分能独立维护和测试时，再考虑拆成少数几个模块；不按方法数量机械拆分。

## 安装与运行（Windows）

在项目根目录执行：

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-ai.txt
python run.py
```

若只使用不依赖 YOLO 的基础功能，可安装 `requirements.txt`。AI 预标注需要安装 `requirements-ai.txt`。

也可以继续使用原文件名启动：

```powershell
python yolo-annotation-review-studio.py
```

## 模型路径

默认模型路径沿用原程序设置，也可用环境变量覆盖：

```powershell
$env:YOLO_MODEL_PATH = "D:\models\best.pt"
python run.py
```

模型权重、视频和数据集建议放在项目目录之外，避免误提交到版本库。

## 修改代码时看哪里

| 需求 | 文件 |
|---|---|
| 默认路径、媒体扩展名 | `config.py` |
| 数据集识别、标签路径规则、YOLO 标签与检测框算法 | `data.py` |
| ORB 重叠率、YOLO 推理、后台线程 | `processing.py` |
| 画布、参数设置和类别选择弹窗 | `ui.py` |
| 播放、摄像头、实时验证、批量预标注、保存等界面流程 | `window.py` |
| 启动流程 | `main.py`、根目录 `run.py` |

依赖关系尽量保持单向：`config.py` 不依赖其他模块；`data.py` 不依赖 GUI；`processing.py` 调用数据/算法模块；`ui.py` 使用配置和通用算法；`window.py` 负责协调。不要让底层算法反向导入 `window.py`。

## 日常项目管理（推荐 Git）

Git 用来记录每次修改，避免保留多个 `final_v3_fixed.py` 文件。安装 Git 后，在项目根目录初始化一次：

```powershell
git init
git add .
git commit -m "Initial project"
```

之后每完成一项功能并检查通过，就提交一次：

```powershell
git status
git diff
git add .
git commit -m "Improve batch preview"
```

较大的尝试先开分支：

```powershell
git switch -c feature/new-feature
```

修改满意后再合并。修改前后可用 `git diff` 看具体变化；发现问题可用 `git log` 找到上一个版本。`.gitignore` 已排除虚拟环境、缓存和模型权重。

## 基础检查

```powershell
python -m compileall -q yolo_annotation_review_studio run.py
```

此命令只检查语法，不会启动窗口、摄像头或 GPU 推理。真实运行仍需在安装了 PyQt5、OpenCV 和相应 AI 依赖的环境中验证。
