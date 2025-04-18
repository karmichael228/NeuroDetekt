#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
    Инструмент для оптимизации порога обнаружения аномалий
    и балансировки метрик Precision/Recall.
"""

import argparse
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve, roc_curve, auc, f1_score
from sklearn.metrics import precision_score, recall_score, accuracy_score

from data_processing import get_data, load_nested_test
from models import LSTMModel
from save_scores import get_scores


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Оптимизация порога обнаружения для моделей IDS.",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Путь к обученной модели.",
    )
    parser.add_argument(
        "--data_set",
        default="plaid",
        choices=["plaid"],
        help="Набор данных для оценки.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Размер батча для оценки.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Устройство для запуска оценки.",
    )
    parser.add_argument(
        "--output_dir",
        default="trials/lstm_plaid_improved",
        help="Директория для сохранения результатов.",
    )
    parser.add_argument(
        "--balance_factor",
        default=1.5,
        type=float,
        help="Фактор штрафа для ложных срабатываний (выше = меньше ложных срабатываний).",
    )

    return parser


def get_anomaly_scores(model, dataloader, device):
    """Получает оценки аномальности для каждой последовательности.
    
    Returns:
        numpy.ndarray: Массив оценок аномальности.
    """
    return get_scores(model, dataloader, device, nll=True)


def find_optimal_threshold(scores, labels, balance_factor=1.0):
    """Находит оптимальный порог для баланса Precision/Recall.
    
    Args:
        scores (numpy.ndarray): Оценки аномальности
        labels (numpy.ndarray): Истинные метки (0: нормальные, 1: атаки)
        balance_factor (float): Множитель для штрафа ложных срабатываний
        
    Returns:
        float: Оптимальный порог
        dict: Метрики для оптимального порога
    """
    # Вычисляем кривую Precision-Recall
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    
    # Избегаем деления на ноль
    precision = np.maximum(precision, 1e-10)
    recall = np.maximum(recall, 1e-10)
    
    # Вычисляем F-бета меру с учетом balance_factor
    # balance_factor > 1 придает больший вес precision (меньше ложных срабатываний)
    # balance_factor < 1 придает больший вес recall (меньше пропусков атак)
    beta_squared = 1 / (balance_factor ** 2)
    f_scores = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
    
    # Находим оптимальный индекс
    optimal_idx = np.argmax(f_scores[:-1])  # исключаем последний элемент (соответствует порогу -inf)
    optimal_threshold = thresholds[optimal_idx]
    
    # Вычисляем метрики для оптимального порога
    predictions = (scores >= optimal_threshold).astype(int)
    metrics = {
        'threshold': optimal_threshold,
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions),
        'recall': recall_score(labels, predictions),
        'f1_score': f1_score(labels, predictions),
        'f_beta': f_scores[optimal_idx]
    }
    
    return optimal_threshold, metrics


def plot_curves(scores, labels, threshold, output_path):
    """Строит и сохраняет кривые ROC и Precision-Recall.
    
    Args:
        scores (numpy.ndarray): Оценки аномальности
        labels (numpy.ndarray): Истинные метки
        threshold (float): Оптимальный порог
        output_path (Path): Путь для сохранения графиков
    """
    plt.figure(figsize=(12, 5))
    
    # ROC кривая
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC кривая (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC кривая')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Precision-Recall кривая
    plt.subplot(1, 2, 2)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    
    # Находим ближайшее значение порога к optimal_threshold
    idx = np.argmin(np.abs(thresholds - threshold)) if len(thresholds) > 0 else 0
    
    plt.plot(recall, precision, label='Precision-Recall кривая')
    if idx < len(precision):
        plt.scatter(recall[idx], precision[idx], 
                   c='red', marker='o', s=100, label=f'Оптимальный порог = {threshold:.3f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall кривая')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path / 'threshold_curves.png')
    plt.close()


def plot_score_distribution(scores, labels, threshold, output_path):
    """Строит и сохраняет распределение оценок аномалий.
    
    Args:
        scores (numpy.ndarray): Оценки аномальности
        labels (numpy.ndarray): Истинные метки
        threshold (float): Оптимальный порог
        output_path (Path): Путь для сохранения графиков
    """
    plt.figure(figsize=(10, 6))
    
    # Разделяем оценки на нормальные и аномальные
    normal_scores = scores[labels == 0]
    attack_scores = scores[labels == 1]
    
    # Строим гистограммы
    plt.hist(normal_scores, bins=50, alpha=0.5, label='Нормальные', color='green')
    plt.hist(attack_scores, bins=50, alpha=0.5, label='Атаки', color='red')
    
    # Добавляем линию порога
    plt.axvline(x=threshold, color='black', linestyle='--', 
               label=f'Порог = {threshold:.3f}')
    
    plt.xlabel('Оценка аномальности')
    plt.ylabel('Количество последовательностей')
    plt.title('Распределение оценок аномальности')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path / 'score_distribution.png')
    plt.close()


def main(model_path, data_set="plaid", batch_size=32, 
         device="cuda", output_dir="trials/lstm_plaid_improved",
         balance_factor=1.5):
    """Основная функция оптимизации порога.
    
    Args:
        model_path (str): Путь к обученной модели
        data_set (str): Имя набора данных
        batch_size (int): Размер батча
        device (str): Устройство для вычислений
        output_dir (str): Директория для сохранения результатов
        balance_factor (float): Фактор баланса между precision и recall
    """
    # Проверка доступности CUDA
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA недоступен, используем CPU")
        device = 'cpu'
    
    # Загружаем тестовые данные
    _, _, (test_loader, test_labels) = get_data(
        data_set, batch_size=batch_size, ratio=1.0
    )
    
    # Загружаем модель
    try:
        # Пытаемся загрузить как полную модель
        model = torch.load(model_path, map_location=device)
        print(f"Модель загружена из {model_path}")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        print("Пробуем загрузить как state_dict...")
        
        # Пытаемся загрузить как state_dict
        model_state = torch.load(model_path, map_location=device)
        model = LSTMModel()
        model.load_state_dict(model_state)
        model = model.to(device)
        print(f"Модель загружена как state_dict")
    
    # Получаем оценки аномальности
    print(f"Получаем оценки аномальности для {len(test_loader)} последовательностей...")
    scores = get_anomaly_scores(model, test_loader, device)
    
    # Находим оптимальный порог
    print(f"Поиск оптимального порога с balance_factor = {balance_factor}...")
    threshold, metrics = find_optimal_threshold(scores, test_labels, balance_factor)
    
    # Создаем выходную директорию
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Строим и сохраняем графики
    plot_curves(scores, test_labels, threshold, output_path)
    plot_score_distribution(scores, test_labels, threshold, output_path)
    
    # Выводим и сохраняем метрики
    print("\n" + "="*50)
    print(f"Оптимальный порог: {threshold:.4f}")
    print(f"Точность (Accuracy): {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-мера: {metrics['f1_score']:.4f}")
    print(f"F-beta (с balance_factor={balance_factor}): {metrics['f_beta']:.4f}")
    print("="*50 + "\n")
    
    # Сохраняем метрики в файл
    with open(output_path / "threshold_results.txt", "w") as f:
        f.write(f"Оптимальный порог: {threshold:.4f}\n")
        f.write(f"Точность (Accuracy): {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-мера: {metrics['f1_score']:.4f}\n")
        f.write(f"F-beta (с balance_factor={balance_factor}): {metrics['f_beta']:.4f}\n")
    
    print(f"Результаты сохранены в {output_path}")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(**vars(args)) 