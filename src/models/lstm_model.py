#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧠 LSTM модель для детекции аномалий в системных вызовах

Архитектура: Embedding → LSTM → Dropout → Linear → LogSoftmax
"""

import torch
import torch.nn as nn
from torch.optim import Adam

__author__ = "karmichael228"
__license__ = "MIT"


class LSTMModel(nn.Module):
    """LSTM модель для прогнозирования последовательностей системных вызовов.
    
    Attributes:
        embedding: Слой эмбеддинга для преобразования индексов в векторы
        lstm_layers: Стек LSTM слоев  
        dropout: Dropout для предотвращения переобучения
        output_layer: Выходной линейный слой
        softmax: LogSoftmax для классификации
    """
    
    def __init__(self, vocab_size=229, embedding_dim=128, hidden_size=128, num_layers=2, dropout=0.25):
        """Инициализирует LSTM модель.
        
        Args:
            vocab_size (int): Размер словаря (количество уникальных системных вызовов)
            embedding_dim (int): Размерность эмбеддинга  
            hidden_size (int): Размер скрытого состояния LSTM
            num_layers (int): Количество слоев LSTM
            dropout (float): Коэффициент дропаута
        """
        super(LSTMModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Слои модели
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Создаем стек LSTM слоев
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_size = embedding_dim if i == 0 else hidden_size
            self.lstm_layers.append(nn.LSTM(input_size, hidden_size, batch_first=True))
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        """Прямой проход через модель.
        
        Args:
            x (torch.Tensor): Входной тензор [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Выходной тензор [batch_size, seq_len, vocab_size]
        """
        # Эмбеддинг входной последовательности
        x = self.embedding(x)
        
        # Проход через стек LSTM слоев
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)
            x = self.dropout(x)
        
        # Выходной слой и логарифмический софтмакс
        x = self.output_layer(x)
        x = self.softmax(x)
        
        return x
    
    def get_model_info(self):
        """Возвращает информацию о модели."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'LSTM',
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }


def create_lstm_model(vocab_size=229, embedding_dim=128, hidden_size=128, num_layers=2, dropout=0.25, device="cuda", learning_rate=0.0001):
    """Создает LSTM модель с оптимизатором и функцией потерь.

    Args:
        vocab_size (int): Размер словаря системных вызовов
        embedding_dim (int): Размерность эмбеддинга
        hidden_size (int): Размер скрытого состояния LSTM
        num_layers (int): Количество слоев LSTM
        dropout (float): Коэффициент дропаута
        device (str): Устройство ('cuda' или 'cpu')
        learning_rate (float): Скорость обучения
        
    Returns:
        tuple: (модель, оптимизатор, функция потерь)
    """
    # Создаем модель
    model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    # Перемещаем на устройство
    device_obj = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    model = model.to(device_obj)
    
    # Создаем оптимизатор и функцию потерь
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    
    # Выводим информацию о модели
    info = model.get_model_info()
    print(f"✅ LSTM модель создана:")
    print(f"   📊 Параметры: {info['total_parameters']:,}")
    print(f"   🧠 Архитектура: {info['embedding_dim']}→{info['hidden_size']}×{info['num_layers']}")
    print(f"   💻 Устройство: {device_obj}")
    
    return model, optimizer, criterion 