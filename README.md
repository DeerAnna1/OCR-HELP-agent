# GuideVision 伴行眼

AI 视觉辅助系统，帮助视障用户感知周围环境。

手机端自动识别场景，语音交互驱动，无需手动操作。

## 功能

- **自动避障** — 实时检测前方障碍物，语音提示方向和距离
- **物体识别** — 识别日常物品（杯子、椅子、门等），语音播报位置
- **文字识别** — 自动检测文字区域，播报内容（药品标签、食品标签等）
- **语音交互** — 直接对手机说话，AI 回答关于眼前场景的问题
- **智能模式** — AI 自主判断当前场景，自动切换工作模式

## 手机端使用（Android）

### 下载安装

1. 从 [GitHub Actions](https://github.com/DeerAnna1/OCR-HELP-agent/actions) 下载最新 APK
2. 传到手机（微信/邮件/网盘均可）
3. 打开 APK 文件，允许安装未知来源应用
4. 打开 GuideVision，授权摄像头和麦克风权限

### 使用方式

打开应用后系统自动运行：

| 场景 | AI 行为 |
|------|---------|
| 前方有障碍物 | 自动切换行走模式，语音提示"前方有椅子，请注意" |
| 对着文字 | 自动切换阅读模式，播报文字内容 |
| 对手机说话 | 切换问答模式，回答关于眼前场景的问题 |

### 语音命令

| 说法 | 效果 |
|------|------|
| "这是什么" | 识别并播报眼前的物体 |
| "前面有什么" | 描述前方场景 |
| "帮我找杯子" | 寻找指定物体 |
| "读一下" | 读取眼前的文字 |
| "前面能走吗" | 判断前方是否可以通行 |
| "有没有椅子" | 搜索特定物体 |
| "停" / "开始" | 控制系统 |

## 桌面端开发

```bash
# 安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# macOS 额外依赖
brew install tesseract tesseract-lang
brew install portaudio

# 运行
python main.py
```

桌面端按 `q` 退出，通过语音或自动模式切换工作模式。

## 项目结构

```
├── agent/                  # AI 决策层
│   ├── auto_mode.py        # 自动模式选择器（场景分析）
│   ├── state_machine.py    # 核心状态机
│   └── modes/              # 5种工作模式
│       ├── walk_mode.py    # 行走避障
│       ├── find_mode.py    # 寻找物体
│       ├── grab_mode.py    # 抓取引导
│       ├── read_mode.py    # 文字阅读
│       └── ask_mode.py     # 问答对话
├── perception/             # 感知层
│   ├── object_detection.py # 物体检测（YOLO/轻量轮廓分析）
│   ├── hand_tracking.py    # 手部追踪（MediaPipe）
│   ├── ocr.py              # 文字识别（Tesseract）
│   └── depth_estimation.py # 深度估计
├── scene/                  # 场景理解层
│   ├── spatial_analyzer.py # 空间分析
│   └── risk_assessor.py    # 风险评估
├── output/                 # 输出层
│   ├── voice.py            # 语音合成
│   └── vibration.py        # 振动反馈
├── main.py                 # 桌面入口
├── main_mobile.py          # Android 入口
├── platform_adapter.py     # 跨平台适配
├── config.py               # 全局配置
└── buildozer.spec          # Android 打包配置
```

## 技术栈

| 模块 | 桌面端 | Android 端 |
|------|--------|-----------|
| 物体检测 | YOLOv8 (ultralytics) | OpenCV 轮廓分析 |
| 手部追踪 | MediaPipe | 不可用 |
| 文字识别 | Tesseract | 不可用 |
| 深度估计 | MiDaS + 启发式 | 启发式（基于物体大小） |
| 语音合成 | pyttsx3 | plyer TTS |
| 语音识别 | SpeechRecognition | plyer STT |
| UI | OpenCV 窗口 | Kivy |

## 构建 APK

APK 通过 GitHub Actions 自动构建：

1. 推送代码到 GitHub
2. Actions 自动运行 `buildozer android debug`
3. 在 Actions 页面下载 APK artifact

## 权限说明

| 权限 | 用途 |
|------|------|
| CAMERA | 摄像头实时画面 |
| RECORD_AUDIO | 语音交互 |
| VIBRATE | 危险提示振动 |
| INTERNET | 语音识别网络请求 |

## License

MIT
