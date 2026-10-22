"""Builds the optional C++ extension ``neural_mcts_chess._core`` (move generation + MCTS).

The pure-Python fallback (python-chess backend) is used automatically when the
extension is unavailable, so a failed build is not fatal.
"""

import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

extra = ["-O3", "-fno-math-errno"]
if os.environ.get("NEURAL_MCTS_CHESS_NATIVE", "1") == "1" and sys.platform != "win32":
    extra.append("-mcpu=native" if os.uname().machine == "arm64" else "-march=native")

ext = Pybind11Extension(
    "neural_mcts_chess._core",
    sources=["csrc/chess.cpp", "csrc/mcts.cpp", "csrc/bindings.cpp"],
    include_dirs=["csrc"],
    cxx_std=17,
    extra_compile_args=extra,
    optional=True,
)

setup(ext_modules=[ext], cmdclass={"build_ext": build_ext})
