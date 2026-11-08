# -*- coding: utf-8 -*-
"""
pytest 配置：把项目根目录加入 sys.path，使测试可以直接 import lhengine。
"""
import os
import sys

# 项目根目录 = 本文件所在目录的上一级
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
