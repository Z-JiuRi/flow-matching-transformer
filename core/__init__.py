# core/__init__.py
"""核心模块"""

from .trainer import Trainer
from .tester import Tester
from .inferencer import Inferencer

__all__ = ['Trainer', 'Tester', 'Inferencer']
