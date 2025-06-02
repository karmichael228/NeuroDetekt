#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRU-Autoencoder для детекции аномалий в системных вызовах.
Модель обучается реконструировать нормальные последовательности.
Высокая ошибка реконструкции указывает на аномалию.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

__author__ = "KARMICHAEL228"
__license__ = "MIT"


class GRUAutoEncoder(nn.Module):
    """GRU-автоенкодер для детекции аномалий в последовательностях.
    
    Упрощенная архитектура:
    1. Энкодер: Эмбеддинг + GRU -> латентное представление
    2. Декодер: Латентное представление -> GRU + проекция -> реконструкция
    """
    
    def __init__(self, vocab_size=229, embedding_dim=128, hidden_size=128, 
                 num_layers=2, latent_dim=64, dropout=0.2):
        """Инициализирует GRU-автоенкодер."""
        super(GRUAutoEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        
        # Эмбеддинг слой
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Энкодер - GRU слои
        self.encoder_gru = nn.GRU(
            embedding_dim, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Проекция в латентное пространство
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_size, latent_dim),
            nn.Tanh()
        )
        
        # Проекция из латентного пространства
        self.from_latent = nn.Linear(latent_dim, hidden_size * num_layers)
        
        # Проекция для входов декодера
        self.latent_to_input = nn.Linear(latent_dim, hidden_size)
        
        # Декодер - GRU слои
        self.decoder_gru = nn.GRU(
            hidden_size,  # Вход - это латентное представление, расширенное по времени
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Выходной слой
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов модели."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
            elif 'embedding' in name:
                nn.init.uniform_(param.data, -0.1, 0.1)
    
    def encode(self, x):
        """Энкодирует входную последовательность в латентное представление."""
        # Эмбеддинг
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # Энкодер GRU
        _, hidden = self.encoder_gru(embedded)  # hidden: [num_layers, batch_size, hidden_size]
        
        # Берем последний слой скрытого состояния
        last_hidden = hidden[-1]  # [batch_size, hidden_size]
        
        # Проекция в латентное пространство
        latent = self.to_latent(last_hidden)  # [batch_size, latent_dim]
        
        return latent
    
    def decode(self, latent, seq_len):
        """Декодирует латентное представление обратно в последовательность."""
        batch_size = latent.size(0)
        
        # Проекция в скрытое состояние для декодера
        hidden_states = self.from_latent(latent)  # [batch_size, hidden_size * num_layers]
        hidden = hidden_states.view(batch_size, self.num_layers, self.hidden_size)  # [batch_size, num_layers, hidden_size]
        hidden = hidden.transpose(0, 1).contiguous()  # [num_layers, batch_size, hidden_size]
        
        # Создаем входы для декодера - повторяем латентное представление по времени
        repeated_latent = latent.unsqueeze(1).repeat(1, seq_len, 1)  # [batch_size, seq_len, latent_dim]
        
        # Проецируем латентное представление в размерность hidden_size для входа в GRU
        decoder_input = torch.tanh(self.latent_to_input(repeated_latent.view(-1, self.latent_dim)))  # [batch_size * seq_len, hidden_size]
        decoder_input = decoder_input.view(batch_size, seq_len, self.hidden_size)  # [batch_size, seq_len, hidden_size]
        
        # Декодер GRU
        decoder_output, _ = self.decoder_gru(decoder_input, hidden)
        decoder_output = self.dropout(decoder_output)
        
        # Проекция в пространство словаря
        output = self.output_layer(decoder_output)  # [batch_size, seq_len, vocab_size]
        
        return output
    
    def forward(self, x):
        """Прямой проход через автоенкодер."""
        seq_len = x.size(1)
        
        # Энкодирование
        latent = self.encode(x)
        
        # Декодирование
        reconstruction = self.decode(latent, seq_len)
        
        return reconstruction, latent
    
    def get_reconstruction_error(self, x):
        """Вычисляет ошибку реконструкции для детекции аномалий (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)."""
        self.eval()
        with torch.no_grad():
            reconstruction, _ = self.forward(x)
            
            # Векторизованное вычисление ошибок реконструкции
            batch_size, seq_len = x.shape
            
            # Вычисляем log_softmax для всех позиций одновременно
            log_probs = F.log_softmax(reconstruction, dim=-1)  # [batch_size, seq_len, vocab_size]
            
            # Создаем маску для игнорирования padding (токен 0)
            padding_mask = (x != 0).float()  # [batch_size, seq_len]
            
            # Получаем логарифмические вероятности правильных токенов
            # Используем gather для извлечения нужных вероятностей
            true_token_log_probs = log_probs.gather(2, x.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]
            
            # Применяем маску и вычисляем отрицательный log-likelihood
            masked_log_probs = true_token_log_probs * padding_mask  # Обнуляем padding
            negative_log_likelihood = -masked_log_probs.sum(dim=1)  # Сумма по времени [batch_size]
            
            # Нормализуем на количество валидных токенов
            valid_token_counts = padding_mask.sum(dim=1)  # [batch_size]
            valid_token_counts = torch.clamp(valid_token_counts, min=1)  # Избегаем деления на 0
            
            normalized_errors = negative_log_likelihood / valid_token_counts
            
            return normalized_errors


def create_gru_autoencoder(vocab_size=229, embedding_dim=128, hidden_size=128, 
                          num_layers=2, latent_dim=64, dropout=0.2, 
                          device="cuda", learning_rate=0.00005):
    """Создает GRU-автоенкодер, оптимизатор и функцию потерь.
    
    Args:
        vocab_size (int): Размер словаря
        embedding_dim (int): Размерность эмбеддинга
        hidden_size (int): Размер скрытого состояния GRU
        num_layers (int): Количество слоев GRU
        latent_dim (int): Размерность латентного пространства
        dropout (float): Коэффициент дропаута
        device (str): Устройство для вычислений
        learning_rate (float): Скорость обучения
        
    Returns:
        tuple: (модель, оптимизатор, функция потерь)
    """
    # Создаем модель
    model = GRUAutoEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        latent_dim=latent_dim,
        dropout=dropout
    )
    
    # Перемещаем модель на устройство
    model = model.to(device)
    
    # Создаем оптимизатор
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Функция потерь для автоенкодера
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Игнорируем padding
    
    return model, optimizer, criterion


class GRUAutoEncoderDetector:
    """Детектор аномалий на основе GRU-автоенкодера."""
    
    def __init__(self, model, threshold=None):
        """Инициализирует детектор.
        
        Args:
            model (GRUAutoEncoder): Обученная модель автоенкодера
            threshold (float): Порог для детекции аномалий
        """
        self.model = model
        self.threshold = threshold
    
    def fit_threshold(self, normal_data, percentile=95):
        """Определяет порог на основе нормальных данных.
        
        Args:
            normal_data (torch.utils.data.DataLoader): Нормальные данные
            percentile (float): Перцентиль для определения порога
        """
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for batch_x, batch_y in normal_data:
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                
                # Берем только нормальные образцы (label = 0)
                normal_mask = (batch_y == 0)
                if normal_mask.sum() > 0:
                    normal_x = batch_x[normal_mask]
                    batch_errors = self.model.get_reconstruction_error(normal_x)
                    errors.extend(batch_errors.cpu().numpy())
        
        # Устанавливаем порог как percentile-й перцентиль ошибок на нормальных данных
 
        print(f"🎯 Установлен порог детекции: {self.threshold:.4f} (перцентиль {percentile}%)")
        
        return self.threshold
    
    def predict(self, x):
        """Предсказывает аномалии.
        
        Args:
            x (torch.Tensor): Входные данные
            
        Returns:
            tuple: (предсказания [0/1], ошибки реконструкции)
        """
        if self.threshold is None:
            raise ValueError("Сначала необходимо установить порог с помощью fit_threshold()")
        
        errors = self.model.get_reconstruction_error(x)
        predictions = (errors > self.threshold).long()
        
        return predictions, errors
    
    def predict_proba(self, x):
        """Возвращает вероятности аномалий.
        
        Args:
            x (torch.Tensor): Входные данные
            
        Returns:
            torch.Tensor: Ошибки реконструкции (больше = более вероятная аномалия)
        """
        return self.model.get_reconstruction_error(x) 