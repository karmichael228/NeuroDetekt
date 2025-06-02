"""
📊 Модуль визуализации NeuroDetekt

Содержит:
- Plots: класс для создания графиков
- Функции для визуализации результатов обучения и тестирования
"""

from .plots import TrainingPlotter, ResultsPlotter, ComparisonPlotter

__all__ = [
    'TrainingPlotter',
    'ResultsPlotter', 
    'ComparisonPlotter'
] 