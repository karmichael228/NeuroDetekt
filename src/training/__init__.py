"""
🎯 Модуль тренировки NeuroDetekt

Содержит:
- Trainer: основной класс для обучения моделей
- Специализированные тренеры для разных архитектур
"""

from .trainer import Trainer, LSTMTrainer
from .autoencoder_trainer import AutoencoderTrainer

__all__ = [
    'Trainer',
    'LSTMTrainer', 
    'AutoencoderTrainer'
] 