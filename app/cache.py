# cache.py
from __future__ import annotations

from typing import Any, Dict
from pathlib import Path
from datetime import datetime, date
import json
import os

import numpy as np


# =====================================
# 基础配置：缓存目录
# =====================================

# 允许通过环境变量覆盖缓存目录，如：
#   export EPISODE_CACHE_DIR=/path/to/cache
_CACHE_DIR_ENV = os.getenv("EPISODE_CACHE_DIR")

if _CACHE_DIR_ENV:
    CACHE_DIR = Path(_CACHE_DIR_ENV).expanduser().resolve()
else:
    # 默认：放在当前包目录下的 "_episode_cache" 文件夹
    CACHE_DIR = Path(__file__).resolve().parent / "_episode_cache"


def _ensure_cache_dir() -> None:
    """确保缓存目录存在。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _episode_cache_path(episode_id: int | str) -> Path:
    """给定 episode_id，返回对应的缓存文件路径。"""
    _ensure_cache_dir()
    return CACHE_DIR / f"episode_{episode_id}.json"


# =====================================
# JSON 序列化辅助
# =====================================

def _json_default(obj: Any) -> Any:
    """
    处理 JSON 不认识的类型：
    - datetime/date → ISO 字符串
    - numpy 数值/数组 → Python 标准类型
    - 其他 → str(...)
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)

def _normalize_for_json(obj: Any) -> Any:
    """
    递归清洗整个对象，使之适合 json.dump：
    - dict 的 key 统一转成 str（避免 numpy.int64 等类型）
    - numpy 类型转成 Python 标准类型
    - ndarray / list / tuple 递归处理
    """
    # 1. 先处理 dict
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # 处理 key：转成 str 最保险，也可以按需转 int
            if isinstance(k, (np.integer, np.floating)):
                key = str(k.item())
            else:
                key = str(k)
            new_dict[key] = _normalize_for_json(v)
        return new_dict

    # 2. list / tuple / set 等可迭代容器
    if isinstance(obj, (list, tuple, set)):
        return [_normalize_for_json(x) for x in obj]

    # 3. numpy 数组
    if isinstance(obj, np.ndarray):
        return [_normalize_for_json(x) for x in obj.tolist()]

    # 4. numpy 标量
    if isinstance(obj, np.generic):
        return _json_default(obj)

    # 5. datetime / date 等交给 _json_default
    if isinstance(obj, (datetime, date)):
        return _json_default(obj)

    # 6. 其他类型：直接返回，交给 json / default 处理
    return obj


# =====================================
# 对外主接口：load / save
# =====================================

def load_episode_cache(episode_id: int | str) -> Dict[str, Any]:
    """
    读取某个 episode 的缓存。

    返回：
        - 若缓存存在且 parse 成功 → dict
        - 若不存在或读取失败 → {}
    """
    path = _episode_cache_path(episode_id)
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # 理论上 data 就是你 run_episode_pipeline 里保存的 cache dict
        if isinstance(data, dict):
            return data
        # 防御性：如果不是 dict，就返回空
        return {}
    except Exception as e:  # noqa: BLE001
        # 这里不抛异常，避免影响主流程，只打印日志
        print(f"[cache] Failed to load cache for episode {episode_id}: {e}")
        return {}


def save_episode_cache(episode_id: int | str, **cache_bundles: Any) -> None:
    """
    保存某个 episode 的缓存。

    使用方法与你现有代码完全兼容：
        cache = {...}
        save_episode_cache(episode_id, **cache)

    实际写入的 JSON 结构形如：
    {
        "news_bundle": {...},
        "clustering_bundle": {...},
        "representative_bundle": {...},
        "fulltext_bundle": {...},
        "summary_bundle": {...},
        "_meta": {
            "episode_id": "...",
            "saved_at": "2025-11-30T12:34:56.789012"
        }
    }
    """
    path = _episode_cache_path(episode_id)

    to_save: Dict[str, Any] = dict(cache_bundles)
    to_save.setdefault("_meta", {})
    to_save["_meta"]["episode_id"] = str(episode_id)
    to_save["_meta"]["saved_at"] = datetime.utcnow().isoformat()

    # ✅ 在写入前统一做一层 “json-safe” 清洗
    normalized = _normalize_for_json(to_save)

    tmp_path = path.with_suffix(".tmp")

    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(
                normalized,
                f,
                ensure_ascii=False,
                indent=2,
                default=_json_default,  # 这里更多是兜底
            )
        tmp_path.replace(path)
    except Exception as e:  # noqa: BLE001
        print(f"[cache] Failed to save cache for episode {episode_id}: {e}")
        # 如果失败，尝试删除 tmp 文件以免残留
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


# =====================================
# 辅助函数（可选）
# =====================================

def clear_episode_cache(episode_id: int | str) -> None:
    """
    删除单个 episode 的缓存文件。
    """
    path = _episode_cache_path(episode_id)
    if path.exists():
        try:
            path.unlink()
            print(f"[cache] Deleted cache for episode {episode_id}")
        except Exception as e:  # noqa: BLE001
            print(f"[cache] Failed to delete cache for episode {episode_id}: {e}")


def clear_all_episode_cache() -> None:
    """
    删除所有 episode 缓存文件（谨慎使用，可以挂在管理脚本/CLI 上）。
    """
    _ensure_cache_dir()
    cnt = 0
    for p in CACHE_DIR.glob("episode_*.json"):
        try:
            p.unlink()
            cnt += 1
        except Exception as e:  # noqa: BLE001
            print(f"[cache] Failed to delete {p}: {e}")
    print(f"[cache] Cleared {cnt} episode cache files from {CACHE_DIR}")
