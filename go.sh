#!/usr/bin/bash
# Day8 切片实验启动器：把 cd / 环境变量 / venv python 全打包，
# 避免在终端里粘贴长命令被折行吃空格。用法：bash <本文件绝对路径>
cd "$(dirname "$0")" || exit 1
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
.venv/Scripts/python.exe experiments/exp01_chunk.py
