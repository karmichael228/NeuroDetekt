"""Вспомогательные функции для обучения на PyTorch"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


class TimeTracker:
    """Отслеживает время выполнения эпохи"""
    
    def __init__(self):
        self.times = []
        self.current_time = 0
    
    def start_epoch(self):
        """Начало эпохи"""
        self.current_time = time.time()
    
    def end_epoch(self):
        """Конец эпохи"""
        elapsed = time.time() - self.current_time
        self.times.append(elapsed)
        return elapsed


class LossTracker:
    """Отслеживает потери и метрики во время обучения"""
    
    def __init__(self, log_dir):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.log_file = self.log_dir / "training_log.csv"
        
        # Создаем файл лога
        with open(self.log_file, 'w') as f:
            f.write("epoch,train_loss,val_loss,train_acc,val_acc,time\n")
    
    def update(self, epoch, train_loss, val_loss, train_acc, val_acc, epoch_time):
        """Обновляет метрики и записывает в лог"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        
        # Записываем в CSV
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch},{train_loss},{val_loss},{train_acc},{val_acc},{epoch_time}\n")
    
    def plot_losses(self):
        """Строит график потерь"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.log_dir / "losses.png")
        plt.close()
    
    def plot_accuracy(self):
        """Строит график точности"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_accs, label='Train Accuracy')
        plt.plot(self.val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(self.log_dir / "accuracy.png")
        plt.close()


class EarlyStopping:
    """Реализация ранней остановки обучения"""
    
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): Сколько эпох ждать улучшения прежде чем остановить обучение
            verbose (bool): Если True, выводит сообщения о прогрессе
            delta (float): Минимальное изменение, которое считается улучшением
            path (str): Путь для сохранения лучшей модели
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        '''Сохраняет модель, когда валидационная потеря уменьшается'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def calculate_accuracy(outputs, targets):
    """Вычисляет точность для задачи языкового моделирования.
    
    Args:
        outputs (torch.Tensor): Выходной тензор модели
        targets (torch.Tensor): Целевой тензор
        
    Returns:
        float: Точность (доля правильных предсказаний)
    """
    # Получаем индексы предсказанных классов
    _, pred = outputs.max(dim=2)
    
    # Формируем маску для padding
    non_pad_mask = (targets != 0)
    
    # Вычисляем точность только для non-padding элементов
    correct = pred.eq(targets).masked_select(non_pad_mask).sum().item()
    total = non_pad_mask.sum().item()
    
    return correct / total if total > 0 else 0


def validate_model(model, test_loader, test_labels, device):
    """Валидирует модель на тестовом наборе и вычисляет метрики.
    
    Args:
        model (nn.Module): Обученная модель
        test_loader (DataLoader): Загрузчик тестовых данных
        test_labels (torch.Tensor): Метки для тестовых данных (0 - нормальные, 1 - атаки)
        device (torch.device): Устройство для вычислений
        
    Returns:
        dict: Словарь с метриками (accuracy, precision, recall, f1_score, roc_auc)
    """
    model.eval()
    all_losses = []
    
    # Получаем loss для каждой последовательности в тестовом наборе
    with torch.no_grad():
        for sequences in test_loader:
            sequences = sequences.to(device)
            batch_size = sequences.size(0)
            
            # Предсказание для каждой последовательности
            outputs = model(sequences[:, :-1])
            targets = sequences[:, 1:]
            
            # Вычисляем loss для каждой последовательности отдельно
            for i in range(batch_size):
                output = outputs[i].view(-1, outputs.size(-1))
                target = targets[i].view(-1)
                
                # Игнорируем padding
                mask = (target != 0)
                if mask.sum() > 0:
                    output = output[mask]
                    target = target[mask]
                    
                    loss = torch.nn.functional.nll_loss(output, target, reduction='mean')
                    all_losses.append(loss.item())
    
    # Преобразуем потери в вероятности аномалий (чем выше loss, тем выше вероятность аномалии)
    anomaly_scores = np.array(all_losses)
    
    # Нормализуем оценки [0, 1]
    if len(anomaly_scores) > 0 and max(anomaly_scores) > min(anomaly_scores):
        anomaly_scores = (anomaly_scores - min(anomaly_scores)) / (max(anomaly_scores) - min(anomaly_scores))
    
    # Выбираем порог на основе F1-меры
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.05, 0.95, 0.05):
        y_pred = (anomaly_scores > threshold).astype(int)
        f1 = f1_score(test_labels, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Финальные предсказания с лучшим порогом
    predictions = (anomaly_scores > best_threshold).astype(int)
    
    # Вычисляем метрики
    metrics = {
        'accuracy': accuracy_score(test_labels, predictions),
        'precision': precision_score(test_labels, predictions),
        'recall': recall_score(test_labels, predictions),
        'f1_score': f1_score(test_labels, predictions),
        'roc_auc': roc_auc_score(test_labels, anomaly_scores),
        'threshold': best_threshold
    }
    
    return metrics
