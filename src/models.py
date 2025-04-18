#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation of LSTM model used in NeuroDetekt development."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

__author__ = "John H. Ring IV, and Colin M. Van Oort"
__license__ = "MIT"


class LSTMModel(nn.Module):
    """Модель LSTM для прогнозирования последовательностей системных вызовов.
    
    Attributes:
        embedding: Слой эмбеддинга
        lstm_layers: Слои LSTM
        dropout: Dropout для предотвращения переобучения
        output_layer: Выходной линейный слой
        softmax: LogSoftmax для классификации
    """
    
    def __init__(self, vocab_size=229, embedding_dim=200, hidden_size=200, num_layers=2, dropout=0.5):
        """Инициализирует модель LSTM.
        
        Args:
            vocab_size (int): Размер словаря (количество уникальных системных вызовов)
            embedding_dim (int): Размерность эмбеддинга
            hidden_size (int): Размер скрытого состояния LSTM
            num_layers (int): Количество слоев LSTM
            dropout (float): Коэффициент дропаута
        """
        super(LSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Создаем список слоев LSTM
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            # Первый слой принимает эмбеддинги, остальные - выход предыдущего слоя
            input_size = embedding_dim if i == 0 else hidden_size
            self.lstm_layers.append(nn.LSTM(input_size, hidden_size, batch_first=True))
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        """Прямой проход через модель.
        
        Args:
            x (torch.Tensor): Входной тензор размера [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Выходной тензор размера [batch_size, seq_len, vocab_size]
        """
        # Эмбеддинг входного тензора
        x = self.embedding(x)
        
        # Проход через слои LSTM
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)
            x = self.dropout(x)
        
        # Выходной слой и логарифмический софтмакс
        x = self.output_layer(x)
        x = self.softmax(x)
        
        return x


def create_lstm_model(vocab_size=229, cells=200, depth=2, dropout=0.5, device="cuda", learning_rate=0.01):
    """Создает LSTM модель, оптимизатор и функцию потерь.

    Args:
        vocab_size (int): Размер словаря (количество уникальных системных вызовов)
        cells (int): Размер скрытого состояния LSTM и эмбеддинга
        depth (int): Количество слоев LSTM
        dropout (float): Коэффициент дропаута
        device (str): Устройство для вычислений ('cuda' или 'cpu')
        learning_rate (float): Скорость обучения
        
    Returns:
        tuple: (модель, оптимизатор, функция потерь)
    """
    # Создаем модель
    model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=cells,
        hidden_size=cells,
        num_layers=depth,
        dropout=dropout
    )
    
    # Перемещаем модель на указанное устройство
    model = model.to(device)
    
    # Создаем оптимизатор
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Создаем функцию потерь
    criterion = nn.NLLLoss()
    
    return model, optimizer, criterion
