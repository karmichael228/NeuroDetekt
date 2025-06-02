#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 Вспомогательные классы и функции для обучения

Содержит:
- TimeTracker: отслеживание времени эпох
- LossTracker: отслеживание потерь и метрик
- EarlyStopping: ранняя остановка обучения
"""

import time
import numpy as np
import torch
from pathlib import Path


class TimeTracker:
    """Отслеживает время выполнения эпох."""
    
    def __init__(self):
        self.times = []
        self.current_time = 0
    
    def start_epoch(self):
        """Начинает отсчет времени эпохи."""
        self.current_time = time.time()
    
    def end_epoch(self):
        """Заканчивает отсчет времени эпохи и возвращает длительность."""
        elapsed = time.time() - self.current_time
        self.times.append(elapsed)
        return elapsed
    
    def get_average_time(self):
        """Возвращает среднее время эпохи."""
        return np.mean(self.times) if self.times else 0.0
    
    def get_total_time(self):
        """Возвращает общее время обучения."""
        return sum(self.times)
    
    def estimate_remaining(self, current_epoch, total_epochs):
        """Оценивает оставшееся время обучения."""
        if not self.times:
            return 0.0
        avg_time = self.get_average_time()
        remaining_epochs = total_epochs - current_epoch
        return avg_time * remaining_epochs


class LossTracker:
    """Отслеживает потери и метрики во время обучения."""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.epoch_times = []  # Добавляем трекинг времени эпох
        
        # Создаем CSV файл с заголовками
        self.log_file = self.log_dir / "training_log.csv"
        with open(self.log_file, 'w') as f:
            f.write("epoch,train_loss,val_loss,train_acc,val_acc,time\n")
    
    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, epoch_time):
        """Обновляет метрики для эпохи."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        self.epoch_times.append(epoch_time)  # Сохраняем время эпохи
        
        # Записываем в CSV файл
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch},{train_loss},{val_loss},{train_acc},{val_acc},{epoch_time}\n")
    
    def get_best_metrics(self):
        """Возвращает лучшие метрики."""
        if not self.val_losses:
            return None
        
        best_val_idx = np.argmin(self.val_losses)
        return {
            'epoch': best_val_idx + 1,
            'train_loss': self.train_losses[best_val_idx],
            'val_loss': self.val_losses[best_val_idx],
            'train_acc': self.train_accs[best_val_idx],
            'val_acc': self.val_accs[best_val_idx]
        }
    
    def save_history(self):
        """Сохраняет историю обучения в файл."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'epoch_times': self.epoch_times  # Добавляем время эпох в историю
        }
        
        import pickle
        with open(self.log_dir / "training_history.pkl", 'wb') as f:
            pickle.dump(history, f)
        
        return history


class EarlyStopping:
    """Реализация ранней остановки обучения."""
    
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', 
                 restore_best_weights=True):
        """
        Args:
            patience (int): Сколько эпох ждать улучшения
            verbose (bool): Выводить сообщения о прогрессе
            delta (float): Минимальное изменение для улучшения
            path (str): Путь для сохранения лучшей модели
            restore_best_weights (bool): Восстанавливать лучшие веса
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = Path(path)
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        """Проверяет необходимость ранней остановки."""
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'   ⏳ EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print(f'   🔄 Восстановлены лучшие веса (val_loss: {self.val_loss_min:.6f})')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        """Сохраняет модель при улучшении валидационной потери."""
        if self.verbose:
            print(f'   💾 Валидационная потеря улучшилась: '
                  f'{self.val_loss_min:.6f} → {val_loss:.6f}')
        
        # Сохраняем состояние модели
        torch.save(model.state_dict(), self.path)
        
        # Сохраняем копию весов в памяти для быстрого восстановления
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
        
        self.val_loss_min = val_loss
    
    def load_best_model(self, model):
        """Загружает лучшую модель."""
        if self.path.exists():
            model.load_state_dict(torch.load(self.path, weights_only=True))
            return True
        return False


def save_model_checkpoint(model, optimizer, epoch, train_loss, val_loss, filepath):
    """Сохраняет полный checkpoint модели.
    
    Args:
        model: Модель PyTorch
        optimizer: Оптимизатор
        epoch (int): Номер эпохи
        train_loss (float): Потеря на обучении
        val_loss (float): Потеря на валидации
        filepath (str): Путь для сохранения
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'timestamp': time.time()
    }
    
    torch.save(checkpoint, filepath)


def load_model_checkpoint(model, optimizer, filepath):
    """Загружает checkpoint модели.
    
    Args:
        model: Модель PyTorch
        optimizer: Оптимизатор
        filepath (str): Путь к checkpoint
        
    Returns:
        dict: Информация о загруженном checkpoint
    """
    checkpoint = torch.load(filepath, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'train_loss': checkpoint.get('train_loss', 0.0),
        'val_loss': checkpoint.get('val_loss', 0.0),
        'timestamp': checkpoint.get('timestamp', 0.0)
    }


def format_time(seconds):
    """Форматирует время в читаемый вид.
    
    Args:
        seconds (float): Время в секундах
        
    Returns:
        str: Отформатированное время
    """
    if seconds < 60:
        return f"{seconds:.1f}с"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}м"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}ч"


def print_training_summary(total_time, best_metrics, final_metrics=None):
    """Выводит красивое резюме обучения.
    
    Args:
        total_time (float): Общее время обучения в секундах
        best_metrics (dict): Лучшие метрики
        final_metrics (dict, optional): Финальные метрики
    """
    print(f"\n{'='*60}")
    print(f"🏁 РЕЗЮМЕ ОБУЧЕНИЯ")
    print('='*60)
    print(f"⏱️ Общее время: {format_time(total_time)}")
    
    if best_metrics:
        print(f"🏆 Лучшие результаты (эпоха {best_metrics['epoch']}):")
        print(f"   📉 Train Loss: {best_metrics['train_loss']:.6f}")
        print(f"   📊 Val Loss: {best_metrics['val_loss']:.6f}")
        print(f"   🎯 Train Acc: {best_metrics['train_acc']:.4f}")
        print(f"   ✅ Val Acc: {best_metrics['val_acc']:.4f}")
    
    if final_metrics:
        print(f"\n📈 Финальные результаты:")
        for key, value in final_metrics.items():
            print(f"   {key}: {value:.4f}")
    
    print('='*60) 