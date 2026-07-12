"""
Windows DLL 预加载 — 在项目模块导入前强制加载 pyarrow/pandas/sklearn/torch
避免多线程导入时的 DLL 加载竞态崩溃 (WinError 6714)
"""

import os as _os


def _preload():
    """在主模块导入前预加载所有 DLL 密集型库 + 环境修复。"""
    _mods = []
    try:
        import pyarrow; _mods.append("pyarrow")
        import pandas; _mods.append("pandas")
        import sklearn; _mods.append("sklearn")
        import torch; _mods.append("torch")
    except Exception as e:
        print(f"[preload] Warning: {e}")
    return _mods
