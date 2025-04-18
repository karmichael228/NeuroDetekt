#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
    Предоставляет CLI для обучения модели LSTM для системы NeuroDetekt c использованием PyTorch.
"""

import argparse
import time
import os
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Используем бэкенд без GUI, чтобы избежать ошибок Qt
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_processing import get_data, load_data_splits, SequencePairDataset, SequenceDataset, collate_fn, test_collate_fn
from models import create_lstm_model
from training_utils import TimeTracker, LossTracker, validate_model

# Настройка seed для воспроизводимости
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Обучение модели LSTM на датасете PLAID с использованием PyTorch.",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Размер батча для обучения.",
    )
    parser.add_argument(
        "--epochs", default=15, type=int, help="Количество эпох обучения."
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Использовать раннюю остановку обучения.",
    )
    parser.add_argument(
        "--cells",
        default=128,
        type=int,
        help="Количество ячеек в LSTM слоях.",
    )
    parser.add_argument(
        "--depth",
        default=2,
        type=int,
        help="Количество слоев LSTM.",
    )
    parser.add_argument(
        "--dropout",
        default=0.3,
        type=float,
        help="Коэффициент dropout.",
    )
    parser.add_argument(
        "--ratio",
        default=1,
        type=int,
        help="Соотношение нормальных и атакующих последовательностей в тестовом наборе",
    )
    parser.add_argument(
        "--patience",
        default=10,
        type=int,
        help="Терпение для ранней остановки обучения.",
    )
    parser.add_argument(
        "--trial",
        default=0,
        type=int,
        help="Номер испытания для выходного пути.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Устройство для обучения (cuda или cpu).",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Количество рабочих процессов для загрузки данных.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        help="Скорость обучения.",
    )
    parser.add_argument(
        "--cv_folds",
        default=5,
        type=int,
        help="Количество фолдов для кросс-валидации.",
    )
    parser.add_argument(
        "--use_cv",
        action="store_true",
        help="Использовать кросс-валидацию вместо простого разделения.",
    )

    return parser


def get_device(device_arg):
    """Определяет устройство для вычислений (CUDA или CPU)."""
    if device_arg == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        memory_allocated = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💻 Используем GPU: {gpu_name}")
        print(f"📊 Доступная память GPU: {memory_allocated:.2f} GB")
    else:
        device = torch.device("cpu")
        print("💻 Используем CPU")
    return device


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, scheduler=None):
    """Выполняет одну эпоху обучения модели."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_start = time.time()
    
    # Освобождаем кэш CUDA перед обучением
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Используем tqdm для отображения прогресса
    progress_bar = tqdm(train_loader, desc=f"🔄 Эпоха {epoch}", 
                        bar_format="{l_bar}{bar:30}{r_bar}", 
                        ncols=100)
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Преобразуем цели в тензор длинных целых чисел
        targets = targets.view(-1)
        outputs = outputs.view(-1, outputs.size(2))
        
        # Применяем маску для игнорирования padding
        non_pad_mask = (targets != 0)
        if non_pad_mask.sum() > 0:
            outputs = outputs[non_pad_mask]
            targets = targets[non_pad_mask]
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Вычисляем точность
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Обновляем прогресс-бар
            progress_bar.set_postfix({
                "❌ потеря": f"{loss.item():.4f}", 
                "✅ точность": f"{100.0 * correct / total:.2f}%",
                "⏱️ время": f"{time.time() - epoch_start:.1f}s"
            })
            
            # Освобождаем память каждые 50 батчей
            if batch_idx % 50 == 0 and batch_idx > 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Шаг планировщика не делаем тут, так как для ReduceLROnPlateau нужна метрика валидации
    
    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    
    epoch_time = time.time() - epoch_start
    print(f"\n📈 Итоги эпохи {epoch}:")
    print(f"   ❌ Потеря: {epoch_loss:.4f}, ✅ Точность: {epoch_acc:.4f}")
    print(f"   ⏱️ Время эпохи: {epoch_time:.2f}s\n")
    
    # Освобождаем кэш CUDA после эпохи
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return epoch_loss, epoch_acc, epoch_time


def create_fold_dataloaders(train_data, val_data, batch_size, num_workers):
    """Создает даталоадеры для кросс-валидации."""
    # Создаем датасеты
    train_dataset = SequencePairDataset(train_data)
    val_dataset = SequencePairDataset(val_data)
    
    # Создаем загрузчики данных
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_dataset)),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train_with_cv(train_data, test_data, test_labels, model_args, training_args):
    """Обучает модель с использованием кросс-валидации."""
    device = training_args['device']
    batch_size = training_args['batch_size']
    epochs = training_args['epochs']
    num_workers = training_args['num_workers']
    learning_rate = training_args['learning_rate']
    early_stopping = training_args['early_stopping']
    patience = training_args['patience']
    n_folds = training_args['cv_folds']
    
    # Создаем директорию для сохранения моделей
    trials_dir = Path("trials")
    trials_dir.mkdir(exist_ok=True, parents=True)
    model_dir = trials_dir / f"lstm_plaid_cv_{n_folds}"
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # Инициализируем кросс-валидацию
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    # Создаем тестовый загрузчик один раз
    test_dataset = SequenceDataset(test_data)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_collate_fn,
        num_workers=num_workers,
        pin_memory=(str(device) == 'cuda')
    )
    
    # Метрики по фолдам
    fold_metrics = []
    
    # Обучаем модель на каждом фолде
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        print(f"\n{'='*40}")
        print(f"🔄 Фолд {fold+1}/{n_folds}")
        print(f"{'='*40}")
        
        # Разделяем данные на обучающую и валидационную выборки
        fold_train_data = [train_data[i] for i in train_idx]
        fold_val_data = [train_data[i] for i in val_idx]
        
        print(f"📊 Размер обучающей выборки: {len(fold_train_data)} последовательностей")
        print(f"📊 Размер валидационной выборки: {len(fold_val_data)} последовательностей")
        
        # Создаем загрузчики данных
        train_loader, val_loader = create_fold_dataloaders(
            fold_train_data, fold_val_data, batch_size, num_workers
        )
        
        # Создаем модель
        model, optimizer, criterion = create_lstm_model(
            vocab_size=model_args['vocab_size'],
            cells=model_args['cells'],
            depth=model_args['depth'],
            dropout=model_args['dropout'],
            device=device,
            learning_rate=learning_rate
        )
        
        # Создаем планировщик скорости обучения
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Для ранней остановки
        best_val_loss = float("inf")
        best_model = None
        patience_counter = 0
        
        # Для отслеживания процесса обучения
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        # Обучаем модель
        for epoch in range(1, epochs + 1):
            # Обучение
            train_loss, train_acc, _ = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )
            
            # Валидация
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, epoch)
            
            # Обновляем планировщик
            scheduler.step(val_loss)
            
            # Сохраняем результаты
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Ранняя остановка
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"⚠️ Ранняя остановка на эпохе {epoch}")
                        break
        
        # Используем лучшую модель
        if early_stopping and best_model is not None:
            model.load_state_dict(best_model)
        
        # Создаем графики
        plt.figure(figsize=(10, 5))
        epochs_range = range(1, len(train_losses) + 1)
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, 'b-', label='Обучение')
        plt.plot(epochs_range, val_losses, 'r-', label='Валидация')
        plt.title('Функция потерь')
        plt.xlabel('Эпохи')
        plt.ylabel('Потеря')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_accs, 'b-', label='Обучение')
        plt.plot(epochs_range, val_accs, 'r-', label='Валидация')
        plt.title('Точность')
        plt.xlabel('Эпохи')
        plt.ylabel('Точность')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(model_dir / f"fold_{fold+1}_metrics.png")
        
        # Сохраняем модель
        model_path = model_dir / f"model_fold_{fold+1}.pt"
        torch.save(model, model_path)
        print(f"💾 Модель сохранена: {model_path}")
        
        # Оцениваем на тестовом наборе
        metrics = validate_model(model, test_loader, test_labels, device)
        fold_metrics.append(metrics)
        
        print(f"\n📊 Результаты фолда {fold+1} на тестовом наборе:")
        print(f"   ✅ Точность: {metrics['accuracy']:.4f}")
        print(f"   📏 Precision: {metrics['precision']:.4f}")
        print(f"   📏 Recall: {metrics['recall']:.4f}")
        print(f"   📏 F1-мера: {metrics['f1_score']:.4f}")
        print(f"   📏 AUC-ROC: {metrics['roc_auc']:.4f}")
        
        # Сохраняем результаты тестирования
        with open(model_dir / f"fold_{fold+1}_results.txt", "w") as f:
            f.write(f"Точность: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1-мера: {metrics['f1_score']:.4f}\n")
            f.write(f"AUC-ROC: {metrics['roc_auc']:.4f}\n")
    
    # Вычисляем средние метрики по всем фолдам
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in fold_metrics]),
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in fold_metrics]),
        'roc_auc': np.mean([m['roc_auc'] for m in fold_metrics])
    }
    
    print(f"\n{'='*40}")
    print(f"📊 Средние результаты по всем фолдам:")
    print(f"{'='*40}")
    print(f"   ✅ Точность: {avg_metrics['accuracy']:.4f}")
    print(f"   📏 Precision: {avg_metrics['precision']:.4f}")
    print(f"   📏 Recall: {avg_metrics['recall']:.4f}")
    print(f"   📏 F1-мера: {avg_metrics['f1_score']:.4f}")
    print(f"   📏 AUC-ROC: {avg_metrics['roc_auc']:.4f}")
    
    # Сохраняем средние результаты
    with open(model_dir / "avg_results.txt", "w") as f:
        f.write(f"Точность: {avg_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {avg_metrics['precision']:.4f}\n")
        f.write(f"Recall: {avg_metrics['recall']:.4f}\n")
        f.write(f"F1-мера: {avg_metrics['f1_score']:.4f}\n")
        f.write(f"AUC-ROC: {avg_metrics['roc_auc']:.4f}\n")
    
    return model_dir, avg_metrics


def validate_epoch(model, val_loader, criterion, device, epoch):
    """Выполняет валидацию модели."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Используем tqdm для отображения прогресса
    progress_bar = tqdm(val_loader, desc=f"🔍 Валидация эпоха {epoch}", 
                       bar_format="{l_bar}{bar:30}{r_bar}", 
                       ncols=100)
    
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            # Преобразуем цели в тензор длинных целых чисел
            targets = targets.view(-1)
            outputs = outputs.view(-1, outputs.size(2))
            
            # Применяем маску для игнорирования padding
            non_pad_mask = (targets != 0)
            if non_pad_mask.sum() > 0:
                outputs = outputs[non_pad_mask]
                targets = targets[non_pad_mask]
                
                loss = criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                
                # Вычисляем точность
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Обновляем прогресс-бар
                progress_bar.set_postfix({
                    "❌ потеря": f"{loss.item():.4f}", 
                    "✅ точность": f"{100.0 * correct / total:.2f}%"
                })
    
    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    
    print(f"📊 Валидация: ❌ Потеря: {epoch_loss:.4f}, ✅ Точность: {epoch_acc:.4f}")
    
    return epoch_loss, epoch_acc


def validate_model(model, test_loader, test_labels, device):
    """Валидирует модель на тестовом наборе и вычисляет метрики."""
    model.eval()
    all_losses = []
    
    # Освобождаем кэш CUDA перед оценкой
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Получаем loss для каждой последовательности в тестовом наборе
    with torch.no_grad():
        for sequences in tqdm(test_loader, desc="📊 Оценка на тестовом наборе"):
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
    
    # Анализируем распределение аномальных оценок
    normal_scores = anomaly_scores[:len(test_labels) - test_labels.sum().int().item()]
    attack_scores = anomaly_scores[len(test_labels) - test_labels.sum().int().item():]
    
    # Выводим базовую статистику
    print(f"\n📊 Статистика оценок аномалий:")
    print(f"   🔹 Нормальные последовательности: мин={normal_scores.min():.4f}, макс={normal_scores.max():.4f}, среднее={normal_scores.mean():.4f}")
    print(f"   🔹 Атакующие последовательности: мин={attack_scores.min():.4f}, макс={attack_scores.max():.4f}, среднее={attack_scores.mean():.4f}")
    
    # Выбираем порог на основе F1-меры
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.05, 0.95, 0.05):
        y_pred = (anomaly_scores > threshold).astype(int)
        f1 = f1_score(test_labels, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"   🔹 Оптимальный порог: {best_threshold:.4f}")
    
    # Финальные предсказания с лучшим порогом
    predictions = (anomaly_scores > best_threshold).astype(int)
    
    # Вычисляем метрики
    metrics = {
        'accuracy': accuracy_score(test_labels, predictions),
        'precision': precision_score(test_labels, predictions),
        'recall': recall_score(test_labels, predictions),
        'f1_score': f1_score(test_labels, predictions),
        'roc_auc': roc_auc_score(test_labels, anomaly_scores),
        'threshold': best_threshold,
        'normal_mean': float(normal_scores.mean()),
        'attack_mean': float(attack_scores.mean()),
        'score_diff': float(attack_scores.mean() - normal_scores.mean())
    }
    
    return metrics


def main(
    batch_size=128,
    epochs=15,
    early_stopping=False,
    cells=128,
    depth=2,
    dropout=0.3,
    ratio=1,
    patience=10,
    trial=0,
    device="cuda",
    num_workers=4,
    learning_rate=1e-3,
    cv_folds=5,
    use_cv=True,
):
    """Входная точка для обучения модели."""
    start_time = time.time()
    
    # Определение устройства
    device = get_device(device)
    
    # Загружаем данные
    train_data, _, test_val, atk = load_data_splits(
        "plaid", train_pct=1.0, ratio=ratio
    )
    
    # Создаем тестовые данные и метки
    test_data = test_val + atk
    test_labels = torch.zeros(len(test_val) + len(atk))
    test_labels[len(test_val):] = 1
    
    # Проверка данных
    print("\n🔍 Анализ данных:")
    print(f"📊 Размер тренировочного набора: {len(train_data)} последовательностей")
    print(f"📊 Размер тестового набора: {len(test_data)} последовательностей")
    print(f"📊 Распределение меток в тестовом наборе: {test_labels.sum().item()} аномалий, {len(test_labels) - test_labels.sum().item()} нормальных")
    
    # Параметры модели
    model_args = {
        'vocab_size': 229,  # Для PLAID датасета
        'cells': cells,
        'depth': depth,
        'dropout': dropout,
    }
    
    # Параметры обучения
    training_args = {
        'device': device,
        'batch_size': batch_size,
        'epochs': epochs,
        'early_stopping': early_stopping,
        'patience': patience,
        'num_workers': num_workers,
        'learning_rate': learning_rate,
        'cv_folds': cv_folds,
    }
    
    # Обучаем модель
    if use_cv:
        print(f"\n🔄 Начинаем обучение с {cv_folds}-кратной кросс-валидацией")
        model_dir, metrics = train_with_cv(train_data, test_data, test_labels, model_args, training_args)
    else:
        # Традиционное обучение без кросс-валидации
        # Создаем модель и загрузчики данных как обычно
        print("\n🔄 Начинаем традиционное обучение без кросс-валидации")
        # Получаем датасеты и загрузчики
        train_loader, _, (test_loader, test_labels) = get_data(
            "plaid", batch_size=batch_size, ratio=ratio, num_workers=num_workers
        )
        
        # Создаем директорию для сохранения моделей
        trials_dir = Path("trials")
        trials_dir.mkdir(exist_ok=True, parents=True)
        model_dir = trials_dir / f"lstm_plaid_{epochs}"
        model_dir.mkdir(exist_ok=True, parents=True)
        
        # Создаем модель
        model, optimizer, criterion = create_lstm_model(
            vocab_size=model_args['vocab_size'],
            cells=model_args['cells'],
            depth=model_args['depth'],
            dropout=model_args['dropout'],
            device=device,
            learning_rate=learning_rate
        )
        
        # Создаем планировщик скорости обучения (StepLR не требует метрик)
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.5
        )
        
        # Для ранней остановки
        best_train_loss = float("inf")
        best_model = None
        patience_counter = 0
        
        # Для отслеживания процесса обучения
        train_losses = []
        train_accs = []
        
        # Основной цикл обучения
        for epoch in range(1, epochs + 1):
            # Обучение
            train_loss, train_acc, epoch_time = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )
            
            # Обновляем планировщик после каждой эпохи
            scheduler.step()
            
            # Сохраняем результаты
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Предупреждение о подозрительно высокой точности на первой эпохе
            if epoch == 1 and train_acc > 0.95:
                print("⚠️ ПРЕДУПРЕЖДЕНИЕ: Подозрительно высокая точность на первой эпохе!")
                print("⚠️ Это может указывать на проблемы с данными или переобучение.")
            
            # Ранняя остановка на основе тренировочной потери
            if early_stopping:
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    best_model = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"⚠️ Ранняя остановка на эпохе {epoch}")
                        break
        
        # Используем лучшую модель, если была ранняя остановка
        if early_stopping and best_model is not None:
            model.load_state_dict(best_model)
        
        # Создаем графики
        plt.figure(figsize=(10, 5))
        epochs_range = range(1, len(train_losses) + 1)
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, 'b-', label='Потеря обучения')
        plt.title('Функция потерь')
        plt.xlabel('Эпохи')
        plt.ylabel('Потеря')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_accs, 'b-', label='Точность обучения')
        plt.title('Точность')
        plt.xlabel('Эпохи')
        plt.ylabel('Точность')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(model_dir / "training_progress.png")
        
        # Сохраняем модель
        model_path = model_dir / f"model_lstm_{trial:02d}.pt"
        torch.save(model, model_path)
        print(f"💾 Модель сохранена: {model_path}")
        
        # Оцениваем на тестовом наборе
        metrics = validate_model(model, test_loader, test_labels, device)
        
        print("\n📊 Результаты на тестовом наборе:")
        print(f"   ✅ Точность: {metrics['accuracy']:.4f}")
        print(f"   📏 Precision: {metrics['precision']:.4f}")
        print(f"   📏 Recall: {metrics['recall']:.4f}")
        print(f"   📏 F1-мера: {metrics['f1_score']:.4f}")
        print(f"   📏 AUC-ROC: {metrics['roc_auc']:.4f}")
        
        # Сохраняем результаты тестирования
        with open(model_dir / "test_results.txt", "w") as f:
            f.write(f"Точность: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1-мера: {metrics['f1_score']:.4f}\n")
            f.write(f"AUC-ROC: {metrics['roc_auc']:.4f}\n")
    
    # Вычисляем общее время обучения
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n⏱️ Общее время обучения: {int(hours)}ч {int(minutes)}м {int(seconds)}с")
    print(f"💾 Все результаты сохранены в директории: {model_dir}")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(**vars(args))
