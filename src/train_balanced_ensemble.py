#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
    Улучшенная версия обучения ансамбля LSTM моделей с балансировкой данных
    и оптимизацией для IDS.
"""

import argparse
import time
import os
import random
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Проверка совместимости NumPy перед импортом torch
try:
    import numpy as np
    numpy_version = np.__version__
    major_version = int(numpy_version.split('.')[0])
    if major_version >= 2:
        print(f"⚠️ Обнаружена версия NumPy {numpy_version}. PyTorch может работать некорректно с NumPy 2.x.")
        print("   Рекомендуется использовать NumPy 1.x для работы с PyTorch.")
        print("   Попробуйте выполнить: conda install numpy=1.24.3")
        sys.exit(1)
except ImportError:
    print("⚠️ Не удалось импортировать NumPy. Убедитесь, что пакет установлен.")
    sys.exit(1)

try:
    import torch
    import matplotlib
    matplotlib.use('Agg')  # Используем бэкенд без GUI
    import matplotlib.pyplot as plt
    from torch.nn import functional as F
    from torch.optim import Adam, lr_scheduler
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
except ImportError as e:
    print(f"⚠️ Ошибка импорта: {e}")
    print("   Убедитесь, что все необходимые пакеты установлены.")
    sys.exit(1)

try:
    from data_processing import get_data, load_data_splits, SequencePairDataset, SequenceDataset, collate_fn, test_collate_fn
    from models import create_lstm_model, LSTMModel
    from training_utils import TimeTracker, LossTracker, validate_model
except ImportError as e:
    print(f"⚠️ Ошибка импорта локальных модулей: {e}")
    print("   Убедитесь, что вы находитесь в корневой директории проекта.")
    sys.exit(1)

# Настройка seed для воспроизводимости
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Игнорировать определенные предупреждения
warnings.filterwarnings("ignore", category=UserWarning, message="Failed to initialize NumPy")


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Обучение ансамбля LSTM моделей с балансировкой данных.",
    )
    parser.add_argument(
        "--batch_size",
        default=96,
        type=int,
        help="Размер батча для обучения.",
    )
    parser.add_argument(
        "--epochs", 
        default=15, 
        type=int, 
        help="Количество эпох обучения."
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
        default=0.25,
        type=float,
        help="Коэффициент dropout.",
    )
    parser.add_argument(
        "--patience",
        default=5,
        type=int,
        help="Терпение для ранней остановки обучения.",
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
        "--ensemble_size",
        default=3,
        type=int,
        help="Количество моделей в ансамбле.",
    )
    parser.add_argument(
        "--balance_factor",
        default=1.5,
        type=float,
        help="Фактор балансировки для штрафа ложных срабатываний.",
    )
    parser.add_argument(
        "--output_dir",
        default="trials/lstm_plaid_improved",
        help="Директория для сохранения моделей и результатов.",
    )

    return parser


def get_device(device_arg):
    """Определяет устройство для вычислений (CUDA или CPU)."""
    try:
        if device_arg == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            memory_allocated = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"💻 Используем GPU: {gpu_name}")
            print(f"📊 Доступная память GPU: {memory_allocated:.2f} GB")
            
            # Проверка версии CUDA
            cuda_version = torch.version.cuda
            if cuda_version:
                print(f"🔧 Версия CUDA: {cuda_version}")
            
            return device
        else:
            if device_arg == "cuda":
                print("⚠️ CUDA запрошена, но недоступна. Используем CPU вместо GPU.")
            else:
                print("💻 Используем CPU")
            return torch.device("cpu")
    except Exception as e:
        print(f"⚠️ Ошибка при инициализации устройства: {e}")
        print("💻 Используем CPU как резервный вариант")
        return torch.device("cpu")


def create_balanced_dataset(train_data, ratio=1.0, duplicate_factor=1):
    """Создает балансированный датасет, дублируя более короткие последовательности.
    
    Args:
        train_data: Список последовательностей для обучения
        ratio: Желаемое соотношение (длинные / короткие)
        duplicate_factor: Множитель для дублирования коротких последовательностей
        
    Returns:
        Список балансированных последовательностей
    """
    # Находим медианную длину последовательности
    seq_lengths = [len(seq) for seq in train_data]
    median_length = np.median(seq_lengths)
    
    # Разделяем на короткие и длинные последовательности
    short_seqs = [seq for seq in train_data if len(seq) < median_length]
    long_seqs = [seq for seq in train_data if len(seq) >= median_length]
    
    print(f"Анализ данных до балансировки:")
    print(f"  Всего последовательностей: {len(train_data)}")
    print(f"  Короткие последовательности: {len(short_seqs)}")
    print(f"  Длинные последовательности: {len(long_seqs)}")
    print(f"  Соотношение (длинные/короткие): {len(long_seqs)/max(1, len(short_seqs)):.2f}")
    
    # Определяем количество последовательностей для добавления
    target_short_count = int(len(long_seqs) / ratio)
    if len(short_seqs) < target_short_count:
        additional_needed = target_short_count - len(short_seqs)
        
        # Дублируем короткие последовательности
        duplicated_short = []
        for _ in range(duplicate_factor):
            if len(duplicated_short) >= additional_needed:
                break
            duplicated_short.extend(random.sample(short_seqs, min(additional_needed, len(short_seqs))))
        
        # Объединяем оригинальные и дублированные последовательности
        balanced_data = long_seqs + short_seqs + duplicated_short[:additional_needed]
    else:
        # Если коротких последовательностей уже достаточно, просто объединяем
        balanced_data = long_seqs + short_seqs
    
    # Перемешиваем данные
    random.shuffle(balanced_data)
    
    print(f"Анализ данных после балансировки:")
    print(f"  Всего последовательностей: {len(balanced_data)}")
    seq_lengths = [len(seq) for seq in balanced_data]
    print(f"  Средняя длина: {np.mean(seq_lengths):.2f}")
    print(f"  Медиана длины: {np.median(seq_lengths):.2f}")
    
    return balanced_data


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
    
    # Шаг планировщика делается после эпохи, если он не нуждается в метрике валидации
    if scheduler is not None and not isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        scheduler.step()
    
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
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5
    
    # Вычисляем метрики для оптимального порога
    predictions = (scores >= optimal_threshold).astype(int)
    metrics = {
        'threshold': optimal_threshold,
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions),
        'recall': recall_score(labels, predictions),
        'f1_score': f1_score(labels, predictions),
        'f_beta': f_scores[optimal_idx] if len(f_scores) > 0 else 0.0
    }
    
    return optimal_threshold, metrics


def get_ensemble_scores(models, dataloader, device):
    """Получает усредненные оценки от ансамбля моделей.
    
    Args:
        models (list): Список моделей LSTM
        dataloader (DataLoader): Загрузчик данных
        device (str): Устройство для вычислений
        
    Returns:
        numpy.ndarray: Массив усредненных оценок аномальности
    """
    all_scores = []
    
    # Получаем оценки от каждой модели
    for model_idx, model in enumerate(models):
        model.eval()
        model_scores = []
        
        with torch.no_grad():
            for sequences in tqdm(dataloader, desc=f"Модель {model_idx+1}/{len(models)}"):
                sequences = sequences.to(device)
                batch_size = sequences.size(0)
                
                # Предсказание для каждой последовательности
                outputs = model(sequences[:, :-1])
                targets = sequences[:, 1:]
                
                # Вычисляем loss для каждой последовательности отдельно
                batch_scores = []
                for i in range(batch_size):
                    output = outputs[i].view(-1, outputs.size(-1))
                    target = targets[i].view(-1)
                    
                    # Игнорируем padding
                    mask = (target != 0)
                    if mask.sum() > 0:
                        output = output[mask]
                        target = target[mask]
                        
                        loss = torch.nn.functional.nll_loss(output, target, reduction='mean')
                        batch_scores.append(loss.item())
                    else:
                        # Если все элементы padding, используем высокую оценку
                        batch_scores.append(10.0)
                
                model_scores.extend(batch_scores)
        
        all_scores.append(model_scores)
    
    # Преобразуем в numpy массив и вычисляем среднее
    all_scores = np.array(all_scores)
    avg_scores = np.mean(all_scores, axis=0)
    
    # Нормализуем оценки [0, 1]
    if len(avg_scores) > 0 and max(avg_scores) > min(avg_scores):
        avg_scores = (avg_scores - min(avg_scores)) / (max(avg_scores) - min(avg_scores))
    
    return avg_scores


def create_safe_dataloader(dataset, batch_size, shuffle, collate_fn, num_workers, device, is_test=False):
    """Создает безопасный загрузчик данных с обработкой ошибок.
    
    Args:
        dataset: Набор данных
        batch_size: Размер батча
        shuffle: Флаг перемешивания данных
        collate_fn: Функция объединения данных
        num_workers: Количество рабочих процессов
        device: Устройство для вычислений
        is_test: Флаг тестового режима
        
    Returns:
        DataLoader: Загрузчик данных
    """
    # Проверка доступности мультипроцессинга
    use_multiprocessing = num_workers > 0
    
    try:
        if use_multiprocessing:
            # Попытка создать загрузчик с несколькими рабочими
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=collate_fn if not is_test else test_collate_fn,
                num_workers=num_workers,
                pin_memory=(str(device) == 'cuda')
            )
    except RuntimeError as e:
        # Если есть ошибка, связанная с мультипроцессингом, отключаем его
        print(f"⚠️ Ошибка при создании загрузчика данных с {num_workers} рабочими: {e}")
        print("   Отключаем многопроцессорную загрузку данных.")
        
    # Резервный вариант - однопроцессный загрузчик
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn if not is_test else test_collate_fn,
        num_workers=0,  # Используем 0 для предотвращения ошибок мультипроцессинга
        pin_memory=(str(device) == 'cuda')
    )


def main(
    batch_size=96,
    epochs=15,
    early_stopping=True,
    cells=128,
    depth=2,
    dropout=0.25,
    patience=5,
    device="cuda",
    num_workers=4,
    learning_rate=1e-3,
    ensemble_size=3,
    balance_factor=1.5,
    output_dir="trials/lstm_plaid_improved"
):
    """Входная точка для обучения ансамбля моделей."""
    start_time = time.time()
    
    # Определение устройства
    device = get_device(device)
    
    # Создаем директорию для сохранения результатов
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    try:
        # Загружаем данные
        print(f"📂 Загрузка данных...")
        train_data, _, test_val, atk = load_data_splits(
            "plaid", train_pct=1.0, ratio=1.0
        )
        
        # Создаем балансированные данные для обучения
        balanced_train_data = create_balanced_dataset(train_data, ratio=1.0, duplicate_factor=2)
        
        # Создаем тестовые данные и метки
        test_data = test_val + atk
        test_labels = torch.zeros(len(test_val) + len(atk))
        test_labels[len(test_val):] = 1
        
        # Создаем тестовый загрузчик
        test_dataset = SequenceDataset(test_data)
        test_loader = create_safe_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=test_collate_fn,
            num_workers=num_workers,
            device=device,
            is_test=True
        )
        
        # Параметры модели
        model_args = {
            'vocab_size': 229,  # Для PLAID датасета
            'cells': cells,
            'depth': depth,
            'dropout': dropout,
        }
        
        # Обучаем ансамбль моделей
        ensemble_models = []
        for i in range(ensemble_size):
            print(f"\n{'='*50}")
            print(f"🔄 Обучение модели {i+1}/{ensemble_size}")
            print(f"{'='*50}")
            
            # Создаем датасет и загрузчик для текущей модели
            # Меняем немного данные для каждой модели для разнообразия
            if i > 0:
                # Перемешиваем данные для каждой модели
                random.shuffle(balanced_train_data)
            
            train_dataset = SequencePairDataset(balanced_train_data)
            train_loader = create_safe_dataloader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
                device=device
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
            scheduler = lr_scheduler.StepLR(
                optimizer, step_size=3, gamma=0.5
            )
            
            # Для ранней остановки
            best_train_loss = float("inf")
            best_model_state = None
            patience_counter = 0
            
            # Для отслеживания процесса обучения
            train_losses = []
            train_accs = []
            
            # Обучаем модель
            for epoch in range(1, epochs + 1):
                # Обучение
                try:
                    train_loss, train_acc, epoch_time = train_epoch(
                        model, train_loader, optimizer, criterion, device, epoch, scheduler
                    )
                    
                    # Сохраняем результаты
                    train_losses.append(train_loss)
                    train_accs.append(train_acc)
                    
                    # Ранняя остановка на основе тренировочной потери
                    if early_stopping:
                        if train_loss < best_train_loss:
                            best_train_loss = train_loss
                            best_model_state = model.state_dict().copy()
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                print(f"⚠️ Ранняя остановка на эпохе {epoch}")
                                break
                except Exception as e:
                    print(f"⚠️ Ошибка при обучении эпохи {epoch}: {e}")
                    print("   Пропускаем эту эпоху и продолжаем.")
                    continue
            
            # Используем лучшую модель, если была ранняя остановка
            if early_stopping and best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            # Создаем графики
            try:
                plt.figure(figsize=(12, 5))
                epochs_range = range(1, len(train_losses) + 1)
                
                plt.subplot(1, 2, 1)
                plt.plot(epochs_range, train_losses, 'b-', label='Потеря обучения')
                plt.title(f'Функция потерь модели {i+1}')
                plt.xlabel('Эпохи')
                plt.ylabel('Потеря')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.subplot(1, 2, 2)
                plt.plot(epochs_range, train_accs, 'b-', label='Точность обучения')
                plt.title(f'Точность модели {i+1}')
                plt.xlabel('Эпохи')
                plt.ylabel('Точность')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plt.savefig(output_path / f"model_{i+1}_training.png")
                plt.close()
            except Exception as e:
                print(f"⚠️ Ошибка при создании графиков: {e}")
            
            # Сохраняем модель
            model_path = output_path / f"model_{i+1}.pt"
            torch.save(model, model_path)
            print(f"💾 Модель {i+1} сохранена: {model_path}")
            
            # Добавляем модель в ансамбль
            ensemble_models.append(model)
            
            # Очистка памяти CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Оцениваем ансамбль на тестовом наборе
        print("\n" + "="*50)
        print(f"📊 Оценка ансамбля на тестовом наборе")
        print("="*50)
        
        # Получаем усредненные оценки от ансамбля
        try:
            ensemble_scores = get_ensemble_scores(ensemble_models, test_loader, device)
            
            # Находим оптимальный порог
            threshold, metrics = find_optimal_threshold(ensemble_scores, test_labels, balance_factor)
            
            # Выводим метрики
            print(f"\n📊 Результаты ансамбля с оптимальным порогом {threshold:.4f}:")
            print(f"   ✅ Точность: {metrics['accuracy']:.4f}")
            print(f"   📏 Precision: {metrics['precision']:.4f}")
            print(f"   📏 Recall: {metrics['recall']:.4f}")
            print(f"   📏 F1-мера: {metrics['f1_score']:.4f}")
            print(f"   📏 F-бета (с balance_factor={balance_factor}): {metrics['f_beta']:.4f}")
            
            # Сохраняем результаты тестирования
            with open(output_path / "ensemble_results.txt", "w") as f:
                f.write(f"Оптимальный порог: {threshold:.4f}\n")
                f.write(f"Точность: {metrics['accuracy']:.4f}\n")
                f.write(f"Precision: {metrics['precision']:.4f}\n")
                f.write(f"Recall: {metrics['recall']:.4f}\n")
                f.write(f"F1-мера: {metrics['f1_score']:.4f}\n")
                f.write(f"F-бета (с balance_factor={balance_factor}): {metrics['f_beta']:.4f}\n")
            
            # Строим графики распределения оценок
            plt.figure(figsize=(10, 6))
            
            # Разделяем оценки на нормальные и аномальные
            normal_scores = ensemble_scores[test_labels == 0]
            attack_scores = ensemble_scores[test_labels == 1]
            
            # Строим гистограммы
            plt.hist(normal_scores, bins=50, alpha=0.5, label='Нормальные', color='green')
            plt.hist(attack_scores, bins=50, alpha=0.5, label='Атаки', color='red')
            
            # Добавляем линию порога
            plt.axvline(x=threshold, color='black', linestyle='--', 
                       label=f'Порог = {threshold:.3f}')
            
            plt.xlabel('Оценка аномальности')
            plt.ylabel('Количество последовательностей')
            plt.title('Распределение оценок аномальности ансамбля')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(output_path / 'ensemble_score_distribution.png')
            plt.close()
        except Exception as e:
            print(f"⚠️ Ошибка при оценке ансамбля: {e}")
        
        # Вычисляем общее время обучения
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n⏱️ Общее время обучения: {int(hours)}ч {int(minutes)}м {int(seconds)}с")
        print(f"💾 Все результаты сохранены в директории: {output_path}")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    sys.exit(main(**vars(args))) 