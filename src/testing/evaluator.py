#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Классы для оценки обученных моделей

Содержит:
- Evaluator: базовый класс для оценки
- LSTMEvaluator: оценка LSTM моделей
- AutoencoderEvaluator: оценка автоэнкодеров
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models import LSTMModel
from ..utils.metrics import (
    calculate_metrics,
    evaluate_anomaly_detection,
    print_metrics_report,
)


class Evaluator:
    """Базовый класс для оценки моделей."""

    def __init__(self, model, device="cuda"):
        """
        Args:
            model: Обученная модель PyTorch
            device (str): Устройство для вычислений
        """
        self.model = model
        self.device = torch.device(
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"🧪 Evaluator инициализирован на устройстве: {self.device}")

    def get_scores(self, dataloader):
        """Получает оценки аномальности для данных. Должен быть переопределен."""
        raise NotImplementedError("Метод get_scores должен быть переопределен")

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, model_class, model_params=None, device="cuda"):
        """Загружает модель из checkpoint."""
        # Создаем модель
        if model_params is None:
            model_params = {}

        model = model_class(**model_params)

        # Загружаем веса
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"📦 Загружен checkpoint эпохи {checkpoint.get('epoch', 'N/A')}")
        else:
            # Предполагаем, что это state_dict
            model.load_state_dict(checkpoint)

        return cls(model, device)


class LSTMEvaluator(Evaluator):
    """Оценщик для LSTM моделей."""

    def __init__(self, model, device="cuda"):
        super().__init__(model, device)
        print("🧠 LSTM Evaluator готов")

    def get_scores(self, dataloader, nll=True):
        """
        Получает оценки аномальности для LSTM модели.

        Args:
            dataloader: DataLoader с данными
            nll (bool): Использовать negative log likelihood как оценку

        Returns:
            np.array: Оценки аномальности для каждой последовательности
        """
        self.model.eval()
        all_scores = []

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="📊 Вычисление оценок", colour="blue")

            for batch in pbar:
                batch = batch.to(self.device)

                # Для каждой последовательности в батче
                for i in range(len(batch)):
                    sequence = batch[i]

                    # Убираем последний элемент для input и первый для target
                    input_seq = sequence[:-1].unsqueeze(0)  # [1, seq_len-1]
                    target_seq = sequence[1:]  # [seq_len-1]

                    # Получаем предсказания
                    output = self.model(input_seq)  # [1, seq_len-1, vocab_size]
                    output = output.squeeze(0)  # [seq_len-1, vocab_size]

                    # Находим валидные (не padding) позиции
                    valid_mask = target_seq != 0

                    if valid_mask.sum() == 0:
                        # Если вся последовательность - padding
                        all_scores.append(0.0)
                        continue

                    # Фильтруем валидные позиции
                    valid_output = output[valid_mask]  # [valid_len, vocab_size]
                    valid_target = target_seq[valid_mask]  # [valid_len]

                    # Вычисляем negative log-likelihood
                    loss = torch.nn.functional.nll_loss(
                        valid_output, valid_target, reduction="mean"
                    )

                    all_scores.append(loss.item())

                pbar.set_postfix({"образцов": len(all_scores)})

        scores_array = np.array(all_scores)

        # Нормализуем оценки
        if len(scores_array) > 0:
            # Отсекаем выбросы
            upper_bound = np.percentile(scores_array, 99.9)
            scores_array = np.clip(scores_array, 0, upper_bound)

            # Нормализуем в диапазон [0, 1] если nll=False
            if not nll:
                if scores_array.max() > scores_array.min():
                    scores_array = (scores_array - scores_array.min()) / (
                        scores_array.max() - scores_array.min()
                    )

        return scores_array

    def evaluate_anomaly_detection(
        self, normal_loader, attack_loader, threshold_method="percentile", threshold_value=95
    ):
        """
        Оценивает качество детекции аномалий LSTM модели.

        Args:
            normal_loader: DataLoader с нормальными данными
            attack_loader: DataLoader с данными атак
            threshold_method (str): Метод установки порога ('percentile', 'optimal')
            threshold_value: Значение для установки порога

        Returns:
            dict: Результаты оценки
        """
        print(f"🧪 Оценка детекции аномалий LSTM модели...")

        # Получаем оценки для нормальных данных
        print("📊 Оценка нормальных данных...")
        normal_scores = self.get_scores(normal_loader)

        # Получаем оценки для атак
        print("🔴 Оценка атак...")
        attack_scores = self.get_scores(attack_loader)

        # Устанавливаем порог
        if threshold_method == "percentile":
            threshold = np.percentile(normal_scores, threshold_value)
        else:
            # Можно добавить другие методы
            threshold = np.percentile(normal_scores, threshold_value)

        # Оценка качества
        evaluation = evaluate_anomaly_detection(normal_scores, attack_scores, threshold_value)

        print_metrics_report(evaluation, "LSTM детекция аномалий")

        return {
            "normal_scores": normal_scores,
            "attack_scores": attack_scores,
            "threshold": threshold,
            "evaluation": evaluation,
        }


class AutoencoderEvaluator(Evaluator):
    """Оценщик для автоэнкодеров."""

    def __init__(self, model, device="cuda"):
        super().__init__(model, device)
        print("🤖 Autoencoder Evaluator готов")

    def get_reconstruction_errors(self, dataloader):
        """
        Получает ошибки реконструкции для автоэнкодера.

        Args:
            dataloader: DataLoader с данными

        Returns:
            np.array: Ошибки реконструкции для каждой последовательности
        """
        self.model.eval()
        all_errors = []

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="📊 Ошибки реконструкции", colour="purple")

            for batch_x in pbar:
                batch_x = batch_x.to(self.device)
                errors = self.model.get_reconstruction_error(batch_x)
                all_errors.extend(errors.cpu().numpy())

                pbar.set_postfix({"образцов": len(all_errors)})

        return np.array(all_errors)

    def evaluate_anomaly_detection(self, normal_loader, attack_loader, threshold_percentile=95):
        """
        Оценивает качество детекции аномалий автоэнкодера.

        Args:
            normal_loader: DataLoader с нормальными данными
            attack_loader: DataLoader с данными атак
            threshold_percentile (int): Перцентиль для установки порога

        Returns:
            dict: Результаты оценки
        """
        print(f"🧪 Оценка детекции аномалий автоэнкодера...")

        # Получаем ошибки для нормальных данных
        print("📊 Ошибки нормальных данных...")
        normal_errors = self.get_reconstruction_errors(normal_loader)

        # Получаем ошибки для атак
        print("🔴 Ошибки атак...")
        attack_errors = self.get_reconstruction_errors(attack_loader)

        # Оценка качества
        evaluation = evaluate_anomaly_detection(normal_errors, attack_errors, threshold_percentile)

        print_metrics_report(evaluation, "Автоэнкодер детекция аномалий")

        return {
            "normal_errors": normal_errors,
            "attack_errors": attack_errors,
            "evaluation": evaluation,
        }

    def test_with_mixed_data(self, test_loader, test_labels, threshold):
        """
        Тестирует автоэнкодер на смешанных данных.

        Args:
            test_loader: DataLoader со смешанными данными
            test_labels: Метки (0=нормальные, 1=атаки)
            threshold (float): Порог для классификации

        Returns:
            dict: Результаты тестирования
        """
        print("🧪 Тестирование на смешанных данных...")

        # Получаем ошибки
        errors = self.get_reconstruction_errors(test_loader)

        # Предсказания
        predictions = (errors > threshold).astype(int)

        # Метрики
        test_labels_np = test_labels.numpy() if hasattr(test_labels, "numpy") else test_labels
        metrics = calculate_metrics(test_labels_np, predictions, errors)

        print_metrics_report(metrics, "Тестирование на смешанных данных")

        return {
            "errors": errors,
            "predictions": predictions,
            "labels": test_labels_np,
            "threshold": threshold,
            "metrics": metrics,
        }


def load_and_evaluate_lstm(checkpoint_path, test_data, vocab_size=229, device="cuda"):
    """
    Загружает и оценивает LSTM модель.

    Args:
        checkpoint_path (str): Путь к checkpoint
        test_data: Тестовые данные
        vocab_size (int): Размер словаря
        device (str): Устройство

    Returns:
        dict: Результаты оценки
    """
    # Загружаем модель
    evaluator = LSTMEvaluator.load_from_checkpoint(
        checkpoint_path, LSTMModel, {"vocab_size": vocab_size}, device
    )

    # Оценка будет зависеть от формата test_data
    # Здесь нужно адаптировать под конкретные данные
    return evaluator


def save_evaluation_results(results, output_path):
    """Сохраняет результаты оценки."""
    import pickle

    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"💾 Результаты оценки сохранены в {output_path}")
