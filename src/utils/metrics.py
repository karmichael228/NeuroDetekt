#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 Модуль метрик для оценки качества моделей

Содержит функции для вычисления различных метрик:
- Точность (accuracy)
- Precision, Recall, F1-score
- ROC AUC и другие метрики
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_accuracy(outputs, targets):
    """Вычисляет точность для задачи языкового моделирования.
    
    Args:
        outputs (torch.Tensor): Выходной тензор модели [batch_size, seq_len, vocab_size]
        targets (torch.Tensor): Целевой тензор [batch_size, seq_len]
        
    Returns:
        float: Точность (доля правильных предсказаний)
    """
    # Получаем индексы предсказанных классов
    _, pred = outputs.max(dim=2)
    
    # Формируем маску для padding (исключаем нулевые токены)
    non_pad_mask = (targets != 0)
    
    # Вычисляем точность только для non-padding элементов
    correct = pred.eq(targets).masked_select(non_pad_mask).sum().item()
    total = non_pad_mask.sum().item()
    
    return correct / total if total > 0 else 0.0


def calculate_metrics(y_true, y_pred, y_scores=None):
    """Вычисляет полный набор метрик для бинарной классификации.
    
    Args:
        y_true (array-like): Истинные метки (0/1)
        y_pred (array-like): Предсказанные метки (0/1)
        y_scores (array-like, optional): Оценки вероятностей для ROC AUC
        
    Returns:
        dict: Словарь с метриками
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Добавляем ROC AUC если есть оценки вероятностей
    if y_scores is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        except ValueError:
            metrics['roc_auc'] = 0.0
    
    return metrics


def find_optimal_threshold(y_true, y_scores, metric='f1'):
    """Находит оптимальный порог для максимизации указанной метрики.
    
    Args:
        y_true (array-like): Истинные метки
        y_scores (array-like): Оценки аномальности
        metric (str): Метрика для оптимизации ('f1', 'precision', 'recall')
        
    Returns:
        tuple: (лучший_порог, лучшее_значение_метрики)
    """
    best_threshold = 0.5
    best_score = 0.0
    
    # Проверяем различные пороги
    for threshold in np.arange(0.05, 0.95, 0.05):
        y_pred = (y_scores > threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Неподдерживаемая метрика: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def evaluate_anomaly_detection(normal_scores, attack_scores, threshold_percentile=95):
    """Оценивает качество детекции аномалий.
    
    Args:
        normal_scores (array-like): Оценки для нормальных данных
        attack_scores (array-like): Оценки для атак
        threshold_percentile (int): Перцентиль для установки порога
        
    Returns:
        dict: Результаты оценки
    """
    # Устанавливаем порог на основе нормальных данных
    threshold = np.percentile(normal_scores, threshold_percentile)
    
    # Создаем метки и предсказания
    all_scores = np.concatenate([normal_scores, attack_scores])
    all_labels = np.concatenate([
        np.zeros(len(normal_scores)),  # Нормальные = 0
        np.ones(len(attack_scores))    # Атаки = 1
    ])
    all_predictions = (all_scores > threshold).astype(int)
    
    # Вычисляем метрики
    metrics = calculate_metrics(all_labels, all_predictions, all_scores)
    
    # Дополнительная статистика
    separation = np.mean(attack_scores) - np.mean(normal_scores)
    
    return {
        'threshold': threshold,
        'threshold_percentile': threshold_percentile,
        'metrics': metrics,
        'separation': separation,
        'normal_stats': {
            'mean': np.mean(normal_scores),
            'std': np.std(normal_scores),
            'median': np.median(normal_scores)
        },
        'attack_stats': {
            'mean': np.mean(attack_scores),
            'std': np.std(attack_scores),
            'median': np.median(attack_scores)
        }
    }


def print_metrics_report(metrics, title="Метрики модели"):
    """Выводит красиво отформатированный отчет по метрикам.
    
    Args:
        metrics (dict): Словарь с метриками
        title (str): Заголовок отчета
    """
    print(f"\n{'='*60}")
    print(f"📊 {title}")
    print('='*60)
    
    if isinstance(metrics, dict) and 'metrics' in metrics:
        # Результат от evaluate_anomaly_detection
        main_metrics = metrics['metrics']
        print(f"🎯 Порог: {metrics['threshold']:.4f} "
              f"({metrics['threshold_percentile']}% перцентиль)")
        print(f"📏 Разделение: {metrics['separation']:.4f}")
        print(f"\n📈 Основные метрики:")
        
        for metric, value in main_metrics.items():
            emoji = {'accuracy': '🎯', 'precision': '🔍', 'recall': '📊', 
                    'f1_score': '🏆', 'roc_auc': '📈'}.get(metric, '📊')
            print(f"   {emoji} {metric.title()}: {value:.4f} ({value*100:.2f}%)")
        
        print(f"\n📊 Статистика ошибок:")
        normal = metrics['normal_stats']
        attack = metrics['attack_stats']
        print(f"   🟢 Нормальные: μ={normal['mean']:.4f}, σ={normal['std']:.4f}")
        print(f"   🔴 Атаки: μ={attack['mean']:.4f}, σ={attack['std']:.4f}")
        
    else:
        # Обычный словарь метрик
        for metric, value in metrics.items():
            emoji = {'accuracy': '🎯', 'precision': '🔍', 'recall': '📊', 
                    'f1_score': '🏆', 'roc_auc': '📈'}.get(metric, '📊')
            print(f"   {emoji} {metric.title()}: {value:.4f} ({value*100:.2f}%)")
    
    print('='*60) 