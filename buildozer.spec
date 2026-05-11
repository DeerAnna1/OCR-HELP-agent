[app]

# App 配置
title = GuideVision
package.name = guidevision
package.domain = com.guidevision

# 源文件
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json,tflite,txt,task
source.exclude_dirs = tests,venv,.git,__pycache__,build,bin,.buildozer,models

# 入口点 (buildozer 默认找 main.py，main.py 中检测 Android 后自动跳转 main_mobile.py)
android.entrypoint = org.kivy.android.PythonActivity
android.bootstrap = sdl2

# 版本
version = 1.0.0

# 图标
icon.filename = %(source.dir)s/icon.png

# 启动画面
presplash.filename = %(source.dir)s/presplash.png

# 横屏/竖屏
orientation = portrait

# 全屏
fullscreen = 0

# Android 配置
android.permissions = CAMERA,RECORD_AUDIO,VIBRATE,INTERNET
android.api = 33
android.minapi = 21
android.ndk = 25b
android.archs = arm64-v8a

# 依赖 (只保留 Android 可用的轻量依赖)
requirements = python3,
    kivy,
    pillow,
    numpy,
    plyer,
    pyjnius,
    android,
    cython

# 日志级别
log_level = 2

# P4A 配置
p4a.branch = master
p4a.bootstrap = sdl2

# 构建 APK
android.release_artifact = apk

# 包含额外的源文件和模块（不包含重型 .pt 模型文件）
source.include_patterns = agent/*,perception/*,scene/*,utils/*,output/*,config.py,platform_adapter.py
