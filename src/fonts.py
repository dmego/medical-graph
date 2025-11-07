#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from matplotlib import font_manager, rcParams


def _installed_font_names():
    try:
        return {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        return set()


def setup_matplotlib_chinese(prefer_family: str = "sans-serif") -> str:
    """
    Configure Matplotlib to use a Chinese-capable font if available.

    Returns the chosen font name (or a fallback) for visibility.
    """
    # Common Chinese fonts across macOS/Linux/Windows
    candidates = [
        # macOS
        "PingFang SC",
        "Hiragino Sans GB",
        "Songti SC",
        "Heiti SC",
        "STHeiti",
        # Cross-platform / Google
        "Noto Sans CJK SC",
        # Windows
        "Microsoft YaHei",
        # Linux community fonts
        "WenQuanYi Zen Hei",
        # Fallbacks that often include wide glyph coverage
        "SimHei",
        "Arial Unicode MS",
    ]

    installed = _installed_font_names()
    chosen = None
    for name in candidates:
        if name in installed:
            chosen = name
            break

    # Try to register system fonts (macOS) if none matched
    if chosen is None:
        possible_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
        for p in possible_paths:
            if os.path.exists(p):
                try:
                    font_manager.fontManager.addfont(p)
                except Exception:
                    pass
        installed = _installed_font_names()
        for name in candidates:
            if name in installed:
                chosen = name
                break

    # Apply configuration
    rcParams["axes.unicode_minus"] = False  # avoid missing Unicode minus
    if prefer_family:
        rcParams["font.family"] = prefer_family
    if chosen:
        current = list(rcParams.get("font.sans-serif", []))
        if chosen in current:
            # Move to front
            current = [chosen] + [f for f in current if f != chosen]
        else:
            current = [chosen] + current
        rcParams["font.sans-serif"] = current
        return chosen
    # Fallback: keep defaults
    return "DejaVu Sans"