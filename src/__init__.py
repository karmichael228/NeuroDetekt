"""
🧠 NeuroDetekt - Система обнаружения вторжений

Модульная архитектура:
- models: LSTM и GRU автоэнкодер модели
- training: классы для обучения моделей
- testing: классы для тестирования и оценки
- visualization: графики и визуализация
- utils: утилиты и вспомогательные функции
"""

from .main import NeuroDetekt

__version__ = "1.0.0"
__author__ = "NeuroDetekt Team"

__all__ = ['NeuroDetekt'] 