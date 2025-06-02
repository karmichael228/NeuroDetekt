#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Базовые классы для тренировки моделей

Содержит:
- Trainer: базовый класс для обучения
- LSTMTrainer: специализированный тренер для LSTM моделей
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.helpers import EarlyStopping, LossTracker, TimeTracker, format_time
from ..utils.metrics import calculate_accuracy


class Trainer:
    """Базовый класс для обучения моделей."""

    def __init__(self, model, optimizer, criterion, device="cuda", log_dir="logs"):
        """
        Args:
            model: Модель PyTorch
            optimizer: Оптимизатор
            criterion: Функция потерь
            device (str): Устройство для вычислений
            log_dir (str): Директория для логов
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )

        # Перемещаем модель на устройство
        self.model = self.model.to(self.device)

        # Инициализируем трекеры
        self.time_tracker = TimeTracker()
        self.loss_tracker = LossTracker(log_dir)
        self.early_stopping = None

        print(f"✅ Тренер инициализирован на устройстве: {self.device}")

    def setup_early_stopping(self, patience=7, delta=0, path="best_model.pt", verbose=True):
        """Настраивает раннюю остановку."""
        self.early_stopping = EarlyStopping(
            patience=patience, verbose=verbose, delta=delta, path=path
        )

    def train_epoch(self, train_loader):
        """Обучает модель одну эпоху. Должен быть переопределен в наследниках."""
        raise NotImplementedError("Метод train_epoch должен быть переопределен")

    def validate_epoch(self, val_loader):
        """Валидирует модель одну эпоху. Должен быть переопределен в наследниках."""
        raise NotImplementedError("Метод validate_epoch должен быть переопределен")

    def train(self, train_loader, val_loader, epochs=10, verbose=True):
        """Основной цикл обучения.

        Args:
            train_loader: DataLoader для обучающих данных
            val_loader: DataLoader для валидационных данных
            epochs (int): Количество эпох
            verbose (bool): Выводить прогресс

        Returns:
            dict: История обучения
        """
        print(f"\n🚀 Начало обучения на {epochs} эпох")
        print(f"   📊 Обучающих батчей: {len(train_loader)}")
        print(f"   🔍 Валидационных батчей: {len(val_loader)}")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            self.time_tracker.start_epoch()

            # Обучение
            train_loss, train_acc = self.train_epoch(train_loader)
            print(
                f"📚 Эпоха {epoch:3d}/{epochs} | ОБУЧЕНИЕ   | Loss: {train_loss:.4f} | Accuracy: {train_acc:.3f}"
            )

            # Валидация
            val_loss, val_acc = self.validate_epoch(val_loader)
            print(
                f"📊 Эпоха {epoch:3d}/{epochs} | ВАЛИДАЦИЯ  | Loss: {val_loss:.4f} | Accuracy: {val_acc:.3f}",
                end="",
            )

            # Время эпохи
            epoch_time = self.time_tracker.end_epoch()

            # Обновляем трекеры
            self.loss_tracker.update(epoch, train_loss, val_loss, train_acc, val_acc, epoch_time)

            # Проверяем ранню остановку и выводим информацию о сохранении модели
            model_saved = False
            if self.early_stopping:
                prev_best_score = self.early_stopping.best_score
                self.early_stopping(val_loss, self.model)
                # Проверяем улучшилась ли модель
                if self.early_stopping.best_score != prev_best_score:
                    model_saved = True
                    print(" | 💾 ЛУЧШАЯ МОДЕЛЬ сохранена!")
                else:
                    print(
                        f" | ⏳ Patience: {self.early_stopping.counter}/{self.early_stopping.patience}"
                    )

                if self.early_stopping.early_stop:
                    print(f"\n⏰ Ранняя остановка на эпохе {epoch}")
                    break
            else:
                print("")  # Просто перенос строки если нет early stopping

            # Дополнительная информация о времени эпохи
            if verbose:
                remaining_time = self.time_tracker.estimate_remaining(epoch, epochs)
                print(
                    f"⏱️  Время эпохи: {format_time(epoch_time)} | Осталось: {format_time(remaining_time)}"
                )
                print()  # Пустая строка для разделения эпох

        total_time = time.time() - start_time

        # Сохраняем историю и выводим резюме
        history = self.loss_tracker.save_history()
        best_metrics = self.loss_tracker.get_best_metrics()

        from ..utils.helpers import print_training_summary

        print_training_summary(total_time, best_metrics)

        return history


class LSTMTrainer(Trainer):
    """Специализированный тренер для LSTM моделей."""

    def __init__(
        self, model, optimizer, criterion, device="cuda", log_dir="logs", gradient_clip=1.0
    ):
        """
        Args:
            gradient_clip (float): Максимальная норма градиента для клиппинга
        """
        super().__init__(model, optimizer, criterion, device, log_dir)
        self.gradient_clip = gradient_clip

        print(f"🧠 LSTM тренер готов (gradient_clip={gradient_clip})")

    def train_epoch(self, train_loader):
        """Обучает LSTM модель одну эпоху."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        # Прогресс бар
        pbar = tqdm(train_loader, desc="Обучение", leave=False, colour="blue")

        for batch in pbar:
            # batch - это кортеж (inputs, targets) от collate_fn
            if isinstance(batch, tuple):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            else:
                # Старый формат - batch - это тензор
                batch = batch.to(self.device)
                inputs = batch[:, :-1]  # Все токены кроме последнего
                targets = batch[:, 1:]  # Все токены кроме первого

            # Обнуляем градиенты
            self.optimizer.zero_grad()

            # Прямой проход
            outputs = self.model(inputs)

            # Вычисляем потери
            loss = self.criterion(
                outputs.contiguous().view(-1, outputs.size(-1)), targets.contiguous().view(-1)
            )

            # Обратный проход
            loss.backward()

            # Клиппинг градиентов
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            # Обновляем веса
            self.optimizer.step()

            # Статистика
            total_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, targets)
            num_batches += 1

            # Обновляем прогресс бар
            pbar.set_postfix(
                {
                    "loss": f"{total_loss/num_batches:.4f}",
                    "acc": f"{total_accuracy/num_batches:.3f}",
                }
            )

        return total_loss / num_batches, total_accuracy / num_batches

    def validate_epoch(self, val_loader):
        """Валидирует LSTM модель одну эпоху."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Валидация", leave=False, colour="green")

            for batch in pbar:
                # batch - это кортеж (inputs, targets) от collate_fn
                if isinstance(batch, tuple):
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    # Старый формат - batch - это тензор
                    batch = batch.to(self.device)
                    inputs = batch[:, :-1]
                    targets = batch[:, 1:]

                # Прямой проход
                outputs = self.model(inputs)

                # Вычисляем потери
                loss = self.criterion(
                    outputs.contiguous().view(-1, outputs.size(-1)), targets.contiguous().view(-1)
                )

                # Статистика
                total_loss += loss.item()
                total_accuracy += calculate_accuracy(outputs, targets)
                num_batches += 1

                # Обновляем прогресс бар
                pbar.set_postfix(
                    {
                        "loss": f"{total_loss/num_batches:.4f}",
                        "acc": f"{total_accuracy/num_batches:.3f}",
                    }
                )

        return total_loss / num_batches, total_accuracy / num_batches
