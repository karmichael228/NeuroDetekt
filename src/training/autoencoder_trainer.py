#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 Тренер для GRU автоэнкодера

Специализированный класс для обучения автоэнкодеров
с функциями детекции аномалий.
"""

import pickle
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..utils.metrics import evaluate_anomaly_detection, print_metrics_report
from .trainer import Trainer


class AutoencoderTrainer(Trainer):
    """Специализированный тренер для автоэнкодеров."""

    def __init__(self, model, optimizer, criterion, device="cuda", log_dir="logs"):
        """
        Args:
            model: GRU автоэнкодер
            optimizer: Оптимизатор
            criterion: Функция потерь (обычно CrossEntropy)
            device (str): Устройство для вычислений
            log_dir (str): Директория для логов
        """
        # Не вызываем super().__init__() так как нам нужна другая логика
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )

        # Перемещаем модель
        self.model = self.model.to(self.device)

        # История обучения
        self.history = {
            "train_losses": [],
            "val_mean_errors": [],
            "val_std_errors": [],
            "val_median_errors": [],
            "val_samples": [],
            "epochs_with_validation": [],
        }

        print(f"🤖 Автоэнкодер тренер готов на устройстве: {self.device}")

    def train_epoch(self, train_loader):
        """Обучает автоэнкодер одну эпоху."""
        self.model.train()
        epoch_loss = 0.0
        total_samples = 0

        train_bar = tqdm(train_loader, desc="🔄 Обучение", colour="blue", leave=False)

        for batch in train_bar:
            # Обрабатываем разные форматы batch данных
            if isinstance(batch, tuple):
                batch_x, _ = batch  # Для автоэнкодера нужны только входные данные
            else:
                batch_x = batch

            batch_x = batch_x.to(self.device)

            self.optimizer.zero_grad()
            reconstruction, _ = self.model(batch_x)

            # Вычисляем потери
            batch_size, seq_len, vocab_size = reconstruction.shape
            reconstruction_flat = reconstruction.view(-1, vocab_size)
            target_flat = batch_x.view(-1)
            loss = self.criterion(reconstruction_flat, target_flat)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)

            train_bar.set_postfix(
                {
                    "loss": f"{epoch_loss/total_samples:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        return epoch_loss / total_samples

    def validate_epoch(self, val_loader, desc="Валидация"):
        """Валидирует автоэнкодер и возвращает статистику ошибок."""
        self.model.eval()
        val_errors = []

        val_bar = tqdm(val_loader, desc=f"📊 {desc}", leave=False, colour="green")

        with torch.no_grad():
            for batch in val_bar:
                # Обрабатываем разные форматы batch данных
                if isinstance(batch, tuple):
                    batch_x, _ = batch  # Для автоэнкодера нужны только входные данные
                else:
                    batch_x = batch

                batch_x = batch_x.to(self.device)
                errors = self.model.get_reconstruction_error(batch_x)
                val_errors.extend(errors.cpu().numpy())

                val_bar.set_postfix(
                    {"образцов": len(val_errors), "средн.ошибка": f"{np.mean(val_errors):.4f}"}
                )

        if len(val_errors) > 0:
            return {
                "mean_error": np.mean(val_errors),
                "std_error": np.std(val_errors),
                "median_error": np.median(val_errors),
                "samples_processed": len(val_errors),
                "errors": np.array(val_errors),
            }
        else:
            return {
                "mean_error": 0.0,
                "std_error": 0.0,
                "median_error": 0.0,
                "samples_processed": 0,
                "errors": np.array([]),
            }

    def train_autoencoder(
        self,
        train_loader,
        val_loader,
        epochs=25,
        patience=8,
        validate_every_n_epochs=1,
        threshold_percentile=95,
        model_name="autoencoder",
        output_dir="trials/autoencoder",
    ):
        """
        Полный цикл обучения автоэнкодера с валидацией.

        Args:
            train_loader: DataLoader для обучения (только нормальные данные)
            val_loader: DataLoader для валидации (только нормальные данные)
            epochs (int): Максимальное количество эпох
            patience (int): Терпение для ранней остановки
            validate_every_n_epochs (int): Частота валидации
            threshold_percentile (int): Перцентиль для установки порога
            model_name (str): Имя модели для сохранения
            output_dir (str): Директория для сохранения

        Returns:
            dict: Результаты обучения
        """
        print(f"\n🎯 Обучение автоэнкодера на {epochs} эпох")
        print(f"   📊 Валидация каждые {validate_every_n_epochs} эпох(и)")
        print(f"   ⏰ Терпение: {patience} эпох")

        # Создаем директорию
        model_dir = Path(output_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        best_val_error = float("inf")
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # ОБУЧЕНИЕ
            epoch_loss = self.train_epoch(train_loader)
            self.history["train_losses"].append(epoch_loss)

            # ВАЛИДАЦИЯ
            val_stats = None
            if epoch % validate_every_n_epochs == 0:
                val_stats = self.validate_epoch(val_loader, f"Валидация эпохи {epoch}")
                self.history["val_mean_errors"].append(val_stats["mean_error"])
                self.history["val_std_errors"].append(val_stats["std_error"])
                self.history["val_median_errors"].append(val_stats["median_error"])
                self.history["val_samples"].append(val_stats["samples_processed"])
                self.history["epochs_with_validation"].append(epoch)

            epoch_time = time.time() - epoch_start

            # Вывод статистики эпохи
            if val_stats:
                print(
                    f"📊 Эпоха {epoch:2d}/{epochs} | "
                    f"Loss: {epoch_loss:.4f} | "
                    f"Val μ: {val_stats['mean_error']:.4f} | "
                    f"Val σ: {val_stats['std_error']:.4f} | "
                    f"Val median: {val_stats['median_error']:.4f} | "
                    f"Samples: {val_stats['samples_processed']} | "
                    f"Время: {epoch_time:.1f}с"
                )

                # Early stopping
                if val_stats["mean_error"] < best_val_error:
                    best_val_error = val_stats["mean_error"]
                    patience_counter = 0

                    # Сохраняем лучшую модель
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "epoch": epoch,
                            "train_loss": epoch_loss,
                            "val_error": val_stats["mean_error"],
                            "val_std": val_stats["std_error"],
                        },
                        model_dir / f"{model_name}_best.pth",
                    )
                    print(f"   💾 Лучшая модель! Val error: {best_val_error:.4f}")

                else:
                    patience_counter += validate_every_n_epochs
                    print(f"   ⏳ Patience: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print(f"\n⏰ Ранняя остановка на эпохе {epoch}")
                        break
            else:
                print(
                    f"📊 Эпоха {epoch:2d}/{epochs} | "
                    f"Loss: {epoch_loss:.4f} | "
                    f"Время: {epoch_time:.1f}с"
                )

        training_time = time.time() - start_time
        print(f"\n✅ Обучение завершено за {training_time/60:.1f} минут")

        # Загружаем лучшую модель
        checkpoint = torch.load(model_dir / f"{model_name}_best.pth", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Устанавливаем порог на валидационных данных
        print(f"\n🎯 Установка финального порога...")
        val_stats_final = self.validate_epoch(val_loader, "Финальный порог")
        threshold = np.percentile(val_stats_final["errors"], threshold_percentile)
        print(f"✅ Финальный порог: {threshold:.4f} (перцентиль {threshold_percentile}%)")

        return {
            "model_dir": model_dir,
            "model_name": model_name,
            "threshold": threshold,
            "threshold_percentile": threshold_percentile,
            "training_time": training_time,
            "training_history": self.history,
            "best_val_error": best_val_error,
            "stopped_epoch": checkpoint["epoch"],
            "final_val_stats": val_stats_final,
        }

    def test_autoencoder(self, test_loader, test_labels, threshold, results_dict=None):
        """
        Тестирует автоэнкодер на смешанных данных (нормальные + атаки).

        Args:
            test_loader: DataLoader с тестовыми данными
            test_labels: Метки (0=нормальные, 1=атаки)
            threshold (float): Порог для классификации
            results_dict (dict, optional): Словарь для дополнения результатов

        Returns:
            dict: Результаты тестирования
        """
        print(f"\n🧪 Тестирование автоэнкодера...")

        self.model.eval()
        test_errors = []

        test_bar = tqdm(test_loader, desc="🧪 Тестирование", colour="red")
        with torch.no_grad():
            for batch in test_bar:
                # Обрабатываем разные форматы batch данных
                if isinstance(batch, tuple):
                    batch_x, _ = batch  # Для автоэнкодера нужны только входные данные
                else:
                    batch_x = batch

                batch_x = batch_x.to(self.device)
                errors = self.model.get_reconstruction_error(batch_x)
                test_errors.extend(errors.cpu().numpy())
                test_bar.set_postfix({"образцов": len(test_errors)})

        test_errors = np.array(test_errors)
        test_labels_np = test_labels.numpy() if hasattr(test_labels, "numpy") else test_labels

        # Разделяем ошибки по типам
        normal_errors = test_errors[test_labels_np == 0]
        attack_errors = test_errors[test_labels_np == 1]

        # Оценка качества
        evaluation = evaluate_anomaly_detection(
            normal_errors,
            attack_errors,
            threshold_percentile=(
                int((threshold / max(test_errors)) * 100) if max(test_errors) > 0 else 95
            ),
        )

        # Выводим результаты
        print_metrics_report(evaluation, "Результаты автоэнкодера")

        # Дополнительная статистика
        separation = np.mean(attack_errors) - np.mean(normal_errors)
        print(f"\n📊 Дополнительная статистика:")
        print(
            f"   📊 Тестовых данных: {len(test_errors)} ({len(normal_errors)} норм, {len(attack_errors)} атак)"
        )
        print(f"   📏 Разделение: {separation:.4f}")
        print(f"   📏 Порог: {threshold:.4f}")

        # Формируем результаты
        results = {
            "test_errors": test_errors,
            "test_labels": test_labels_np,
            "threshold": threshold,
            "evaluation": evaluation,
            "separation": separation,
            "error_statistics": {
                "normal_mean": np.mean(normal_errors),
                "normal_std": np.std(normal_errors),
                "attack_mean": np.mean(attack_errors),
                "attack_std": np.std(attack_errors),
            },
        }

        # Объединяем с предыдущими результатами если переданы
        if results_dict:
            results.update(results_dict)

        return results

    def save_results(self, results, output_path):
        """Сохраняет результаты в файл."""
        with open(output_path, "wb") as f:
            pickle.dump(results, f)
        print(f"💾 Результаты сохранены в {output_path}")
