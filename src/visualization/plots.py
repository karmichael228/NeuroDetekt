#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 Модуль для создания графиков и визуализаций

Содержит классы для:
- Визуализации процесса обучения
- Анализа результатов тестирования
- Сравнения разных моделей
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

warnings.filterwarnings("ignore")

# Настройка стиля
plt.style.use("default")
sns.set_palette("husl")


class TrainingPlotter:
    """Класс для визуализации процесса обучения."""

    def __init__(self, save_dir="plots", dpi=150, figsize=(12, 8)):
        """
        Args:
            save_dir (str): Директория для сохранения графиков
            dpi (int): Разрешение графиков
            figsize (tuple): Размер фигур
        """
        self.save_dir = Path(save_dir) / "plots"  # Создаем подпапку plots
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.dpi = dpi
        self.figsize = figsize

        print(f"📊 TrainingPlotter готов (сохранение в {self.save_dir})")

    def plot_training_history(
        self, history, title="История обучения", save_name="training_history.png", model_type=None
    ):
        """
        Строит графики для истории обучения в зависимости от типа модели.

        Args:
            history (dict): История обучения
            title (str): Заголовок графика
            save_name (str): Имя файла для сохранения
            model_type (str): Тип модели ('lstm', 'autoencoder', None для автоопределения)
        """
        # Автоматическое определение типа модели если не указан
        if model_type is None:
            model_type = self._detect_model_type(history)

        print(f"🎨 Создание графика для {model_type.upper()} модели...")

        if model_type == "autoencoder":
            return self._plot_autoencoder_history(history, title, save_name)
        elif model_type == "lstm":
            return self._plot_lstm_history(history, title, save_name)
        else:
            print(f"⚠️ Неизвестный тип модели: {model_type}, использую базовый формат")
            return self._plot_generic_history(history, title, save_name)

    def _detect_model_type(self, history):
        """Автоматически определяет тип модели по содержимому истории."""
        # Проверяем наличие специфичных для автоэнкодера ключей
        autoencoder_keys = ["val_mean_errors", "val_std_errors", "val_median_errors"]
        if any(key in history for key in autoencoder_keys):
            return "autoencoder"

        # Проверяем наличие accuracy метрик (характерно для LSTM)
        lstm_keys = ["train_accs", "val_accs", "train_acc", "val_acc"]
        if any(key in history for key in lstm_keys):
            return "lstm"

        # По умолчанию
        return "generic"

    def _plot_autoencoder_history(self, history, title, save_name):
        """Специализированный график для автоэнкодера."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        epochs = range(1, len(history["train_losses"]) + 1)

        # 1. График потерь обучения (reconstruction loss)
        axes[0, 0].plot(epochs, history["train_losses"], "b-", label="Обучение", linewidth=2)
        axes[0, 0].set_title("Потери")
        axes[0, 0].set_xlabel("Эпоха")
        axes[0, 0].set_ylabel("LOSS")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Добавляем минимальную потерю
        min_loss = min(history["train_losses"])
        min_epoch = history["train_losses"].index(min_loss) + 1
        axes[0, 0].scatter([min_epoch], [min_loss], color="red", s=50, zorder=5)
        axes[0, 0].annotate(
            f"Min: {min_loss:.4f}",
            xy=(min_epoch, min_loss),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

        # 2. Ошибки валидации (основная метрика для автоэнкодера)
        if "val_mean_errors" in history and history["val_mean_errors"]:
            val_epochs = range(1, len(history["val_mean_errors"]) + 1)
            mean_errors = history["val_mean_errors"]

            axes[0, 1].plot(val_epochs, mean_errors, "g-", label="Средняя ошибка", linewidth=2.5)

            # Добавляем доверительный интервал если есть стандартные отклонения
            if "val_std_errors" in history and history["val_std_errors"]:
                std_errors = history["val_std_errors"]
                axes[0, 1].fill_between(
                    val_epochs,
                    np.array(mean_errors) - np.array(std_errors),
                    np.array(mean_errors) + np.array(std_errors),
                    alpha=0.3,
                    color="green",
                    label="±σ",
                )

            axes[0, 1].set_title("Ошибки валидации")
            axes[0, 1].set_xlabel("Эпоха")
            axes[0, 1].set_ylabel("Ошибка реконструкции")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Показываем лучшую эпоху
            min_val_error = min(mean_errors)
            min_val_epoch = mean_errors.index(min_val_error) + 1
            axes[0, 1].scatter([min_val_epoch], [min_val_error], color="red", s=50, zorder=5)
            axes[0, 1].annotate(
                f"Best: {min_val_error:.4f}",
                xy=(min_val_epoch, min_val_error),
                xytext=(10, -20),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
            )
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "Нет данных валидации",
                transform=axes[0, 1].transAxes,
                ha="center",
                va="center",
            )
            axes[0, 1].set_title("Ошибки валидации")

        # 3. Время на эпоху
        if "epoch_times" in history and history["epoch_times"]:
            epoch_times = history["epoch_times"]
            axes[1, 0].plot(epochs, epoch_times, "purple", linewidth=2, marker="o", markersize=4)
            axes[1, 0].set_title("Время обучения на эпоху")
            axes[1, 0].set_xlabel("Эпоха")
            axes[1, 0].set_ylabel("Время (сек)")
            axes[1, 0].grid(True, alpha=0.3)

            # Среднее время
            avg_time = np.mean(epoch_times)
            axes[1, 0].axhline(
                avg_time, color="red", linestyle="--", alpha=0.7, label=f"Среднее: {avg_time:.1f}с"
            )
            axes[1, 0].legend()

            # Показываем общее время
            total_time = sum(epoch_times)
            axes[1, 0].text(
                0.02,
                0.98,
                f"Общее время: {total_time:.1f}с",
                transform=axes[1, 0].transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                verticalalignment="top",
            )
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "Нет данных времени",
                transform=axes[1, 0].transAxes,
                ha="center",
                va="center",
            )
            axes[1, 0].set_title("Время обучения на эпоху")

        # 4. Детальная статистика автоэнкодера
        axes[1, 1].axis("off")
        stats_text = self._generate_autoencoder_stats(history)
        axes[1, 1].text(
            0.05,
            0.95,
            stats_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        print(f"💾 График автоэнкодера сохранен: {save_path}")
        return str(save_path)

    def _plot_lstm_history(self, history, title, save_name):
        """Специализированный график для LSTM."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        epochs = range(1, len(history["train_losses"]) + 1)

        # 1. График потерь
        axes[0, 0].plot(epochs, history["train_losses"], "b-", label="Обучение", linewidth=2)

        # Определяем формат валидационных потерь
        val_losses_key = "val_losses" if "val_losses" in history else "val_loss"
        if val_losses_key in history and history[val_losses_key]:
            val_epochs = range(1, len(history[val_losses_key]) + 1)
            axes[0, 0].plot(
                val_epochs, history[val_losses_key], "r-", label="Валидация", linewidth=2
            )

            # Показываем лучшую эпоху
            min_val_loss = min(history[val_losses_key])
            min_val_epoch = history[val_losses_key].index(min_val_loss) + 1
            axes[0, 0].scatter([min_val_epoch], [min_val_loss], color="gold", s=60, zorder=5)
            axes[0, 0].annotate(
                f"Best: {min_val_loss:.4f}",
                xy=(min_val_epoch, min_val_loss),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )

        axes[0, 0].set_title("Потери")
        axes[0, 0].set_xlabel("Эпоха")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. График точности
        train_accs_key = "train_accs" if "train_accs" in history else "train_acc"
        val_accs_key = "val_accs" if "val_accs" in history else "val_acc"

        if train_accs_key in history and history[train_accs_key]:
            axes[0, 1].plot(epochs, history[train_accs_key], "b-", label="Обучение", linewidth=2)

            if val_accs_key in history and history[val_accs_key]:
                val_epochs = range(1, len(history[val_accs_key]) + 1)
                axes[0, 1].plot(
                    val_epochs, history[val_accs_key], "r-", label="Валидация", linewidth=2
                )

                # Показываем лучшую точность
                max_val_acc = max(history[val_accs_key])
                max_val_epoch = history[val_accs_key].index(max_val_acc) + 1
                axes[0, 1].scatter([max_val_epoch], [max_val_acc], color="gold", s=60, zorder=5)
                axes[0, 1].annotate(
                    f"Best: {max_val_acc:.3f}",
                    xy=(max_val_epoch, max_val_acc),
                    xytext=(10, -20),
                    textcoords="offset points",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                )

            axes[0, 1].set_title("Точность")
            axes[0, 1].set_xlabel("Эпоха")
            axes[0, 1].set_ylabel("Accuracy")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 1.05)  # Фиксируем диапазон для accuracy
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "Нет данных точности",
                transform=axes[0, 1].transAxes,
                ha="center",
                va="center",
            )
            axes[0, 1].set_title("Точность")

        # 3. Время на эпоху
        if "epoch_times" in history and history["epoch_times"]:
            epoch_times = history["epoch_times"]
            axes[1, 0].plot(epochs, epoch_times, "green", linewidth=2, marker="o", markersize=4)
            axes[1, 0].set_title("Время обучения на эпоху")
            axes[1, 0].set_xlabel("Эпоха")
            axes[1, 0].set_ylabel("Время (сек)")
            axes[1, 0].grid(True, alpha=0.3)

            # Среднее время
            avg_time = np.mean(epoch_times)
            axes[1, 0].axhline(
                avg_time, color="red", linestyle="--", alpha=0.7, label=f"Среднее: {avg_time:.1f}с"
            )
            axes[1, 0].legend()
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "Нет данных времени",
                transform=axes[1, 0].transAxes,
                ha="center",
                va="center",
            )
            axes[1, 0].set_title("Время обучения на эпоху")

        # 4. Learning Rate или статистика
        if "learning_rates" in history and history["learning_rates"]:
            axes[1, 1].plot(epochs, history["learning_rates"], "m-", linewidth=2)
            axes[1, 1].set_title("Learning Rate")
            axes[1, 1].set_xlabel("Эпоха")
            axes[1, 1].set_ylabel("LR")
            axes[1, 1].set_yscale("log")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis("off")
            stats_text = self._generate_lstm_stats(history)
            axes[1, 1].text(
                0.05,
                0.95,
                stats_text,
                transform=axes[1, 1].transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8),
            )

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        print(f"💾 График LSTM сохранен: {save_path}")
        return str(save_path)

    def _plot_generic_history(self, history, title, save_name):
        """Универсальный график для неопределенного типа модели."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        epochs = range(1, len(history["train_losses"]) + 1)

        # График потерь
        axes[0, 0].plot(epochs, history["train_losses"], "b-", label="Обучение", linewidth=2)
        if "val_losses" in history and history["val_losses"]:
            val_epochs = range(1, len(history["val_losses"]) + 1)
            axes[0, 0].plot(val_epochs, history["val_losses"], "r-", label="Валидация", linewidth=2)
        axes[0, 0].set_title("Потери")
        axes[0, 0].set_xlabel("Эпоха")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Остальные графики - базовая реализация
        for i in range(3):
            row, col = divmod(i + 1, 2)
            axes[row, col].text(
                0.5,
                0.5,
                "Данные недоступны",
                transform=axes[row, col].transAxes,
                ha="center",
                va="center",
            )

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        print(f"💾 Базовый график сохранен: {save_path}")
        return str(save_path)

    def _generate_autoencoder_stats(self, history):
        """Генерирует статистику для автоэнкодера."""
        stats = "📊 СТАТИСТИКА АВТОЭНКОДЕРА\n"
        stats += "─" * 30 + "\n\n"

        # Основные метрики
        if "train_losses" in history:
            final_loss = history["train_losses"][-1]
            min_loss = min(history["train_losses"])
            stats += f"🔸 Финальный Loss: {final_loss:.4f}\n"
            stats += f"🔸 Минимальный Loss: {min_loss:.4f}\n"

        # Валидационные ошибки
        if "val_mean_errors" in history and history["val_mean_errors"]:
            final_val_error = history["val_mean_errors"][-1]
            min_val_error = min(history["val_mean_errors"])
            best_epoch = history["val_mean_errors"].index(min_val_error) + 1
            stats += f"🔸 Лучшая эпоха: {best_epoch}\n"
            stats += f"🔸 Лучшая вал. ошибка: {min_val_error:.4f}\n"
            stats += f"🔸 Финальная вал. ошибка: {final_val_error:.4f}\n"

        # Время обучения
        if "epoch_times" in history and history["epoch_times"]:
            total_time = sum(history["epoch_times"])
            avg_time = np.mean(history["epoch_times"])
            stats += f"\n⏱️ ВРЕМЯ:\n"
            stats += f"🔸 Общее: {total_time:.1f}с\n"
            stats += f"🔸 Среднее/эпоха: {avg_time:.1f}с\n"

        # Общая информация
        epochs_count = len(history.get("train_losses", []))
        stats += f"\n📈 Эпох обучено: {epochs_count}\n"

        # Дополнительные метрики если есть
        if "val_std_errors" in history and history["val_std_errors"]:
            final_std = history["val_std_errors"][-1]
            stats += f"🔸 Станд. откл. (фин.): {final_std:.4f}\n"

        return stats

    def _generate_lstm_stats(self, history):
        """Генерирует статистику для LSTM."""
        stats = "📊 СТАТИСТИКА LSTM\n"
        stats += "─" * 25 + "\n\n"

        # Loss метрики
        if "train_losses" in history:
            final_train_loss = history["train_losses"][-1]
            stats += f"🔸 Final Train Loss: {final_train_loss:.4f}\n"

        val_losses_key = "val_losses" if "val_losses" in history else "val_loss"
        if val_losses_key in history and history[val_losses_key]:
            final_val_loss = history[val_losses_key][-1]
            min_val_loss = min(history[val_losses_key])
            best_epoch = history[val_losses_key].index(min_val_loss) + 1
            stats += f"🔸 Final Val Loss: {final_val_loss:.4f}\n"
            stats += f"🔸 Best Val Loss: {min_val_loss:.4f}\n"
            stats += f"🔸 Лучшая эпоха: {best_epoch}\n"

        # Accuracy метрики
        train_accs_key = "train_accs" if "train_accs" in history else "train_acc"
        val_accs_key = "val_accs" if "val_accs" in history else "val_acc"

        if train_accs_key in history and history[train_accs_key]:
            final_train_acc = history[train_accs_key][-1]
            stats += f"🔸 Final Train Acc: {final_train_acc:.3f}\n"

        if val_accs_key in history and history[val_accs_key]:
            final_val_acc = history[val_accs_key][-1]
            max_val_acc = max(history[val_accs_key])
            stats += f"🔸 Final Val Acc: {final_val_acc:.3f}\n"
            stats += f"🔸 Best Val Acc: {max_val_acc:.3f}\n"

        # Время обучения
        if "epoch_times" in history and history["epoch_times"]:
            total_time = sum(history["epoch_times"])
            avg_time = np.mean(history["epoch_times"])
            stats += f"\n⏱️ ВРЕМЯ:\n"
            stats += f"🔸 Общее: {total_time:.1f}с\n"
            stats += f"🔸 Среднее/эпоха: {avg_time:.1f}с\n"

        # Общая информация
        epochs_count = len(history.get("train_losses", []))
        stats += f"\n📈 Эпох обучено: {epochs_count}"

        return stats


class ResultsPlotter:
    """Класс для визуализации результатов тестирования."""

    def __init__(self, save_dir="plots", dpi=150, figsize=(12, 8)):
        self.save_dir = Path(save_dir) / "plots"  # Создаем подпапку plots
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.dpi = dpi
        self.figsize = figsize

        print(f"📊 ResultsPlotter готов")

    def plot_score_distributions(
        self,
        normal_scores,
        attack_scores,
        threshold=None,
        title="Распределение оценок",
        save_name="score_distributions.png",
    ):
        """
        Строит распределения оценок для нормальных данных и атак.

        Args:
            normal_scores (array): Оценки для нормальных данных
            attack_scores (array): Оценки для атак
            threshold (float): Порог классификации
            title (str): Заголовок
            save_name (str): Имя файла
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Гистограммы
        axes[0, 0].hist(
            normal_scores,
            bins=50,
            alpha=0.7,
            color="green",
            label=f"Нормальные (n={len(normal_scores)})",
        )
        axes[0, 0].hist(
            attack_scores, bins=50, alpha=0.7, color="red", label=f"Атаки (n={len(attack_scores)})"
        )
        if threshold is not None:
            axes[0, 0].axvline(
                threshold,
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"Порог: {threshold:.4f}",
            )
        axes[0, 0].set_title("Распределение оценок")
        axes[0, 0].set_xlabel("Оценка аномальности")
        axes[0, 0].set_ylabel("Частота")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Box plot
        data_for_box = [normal_scores, attack_scores]
        axes[0, 1].boxplot(data_for_box, labels=["Нормальные", "Атаки"])
        if threshold is not None:
            axes[0, 1].axhline(threshold, color="black", linestyle="--", linewidth=2)
        axes[0, 1].set_title("Box Plot оценок")
        axes[0, 1].set_ylabel("Оценка аномальности")
        axes[0, 1].grid(True, alpha=0.3)

        # Violin plot
        df = pd.DataFrame(
            {
                "Оценка": np.concatenate([normal_scores, attack_scores]),
                "Тип": ["Нормальные"] * len(normal_scores) + ["Атаки"] * len(attack_scores),
            }
        )
        sns.violinplot(data=df, x="Тип", y="Оценка", ax=axes[1, 0])
        if threshold is not None:
            axes[1, 0].axhline(threshold, color="black", linestyle="--", linewidth=2)
        axes[1, 0].set_title("Violin Plot")
        axes[1, 0].grid(True, alpha=0.3)

        # Статистика
        axes[1, 1].axis("off")
        stats_text = f"""
Статистика оценок:

Нормальные данные:
• Среднее: {np.mean(normal_scores):.4f}
• Медиана: {np.median(normal_scores):.4f}
• Ст. откл.: {np.std(normal_scores):.4f}
• Мин-Макс: {np.min(normal_scores):.4f} - {np.max(normal_scores):.4f}

Атаки:
• Среднее: {np.mean(attack_scores):.4f}
• Медиана: {np.median(attack_scores):.4f}
• Ст. откл.: {np.std(attack_scores):.4f}
• Мин-Макс: {np.min(attack_scores):.4f} - {np.max(attack_scores):.4f}

Разделение: {np.mean(attack_scores) - np.mean(normal_scores):.4f}
"""
        if threshold is not None:
            stats_text += f"\nПорог: {threshold:.4f}"

        axes[1, 1].text(
            0.05,
            0.95,
            stats_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        print(f"💾 График сохранен: {save_path}")
        return str(save_path)

    def plot_roc_curve(self, y_true, y_scores, title="ROC кривая", save_name="roc_curve.png"):
        """Строит ROC кривую."""
        from sklearn.metrics import auc, roc_curve

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # ROC кривая
        axes[0].plot(
            fpr, tpr, color="darkorange", linewidth=3, label=f"ROC кривая (AUC = {roc_auc:.4f})"
        )
        axes[0].plot(
            [0, 1],
            [0, 1],
            color="navy",
            linewidth=2,
            linestyle="--",
            label="Случайный классификатор",
        )
        axes[0].fill_between(fpr, tpr, alpha=0.2, color="darkorange")
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel("False Positive Rate (FPR)")
        axes[0].set_ylabel("True Positive Rate (TPR)")
        axes[0].set_title("ROC Кривая")
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)

        # График порогов
        # Находим оптимальный порог (максимизируем TPR - FPR)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]

        axes[0].plot(
            optimal_fpr,
            optimal_tpr,
            "ro",
            markersize=8,
            label=f"Оптимальный порог = {optimal_threshold:.4f}",
        )
        axes[0].legend(loc="lower right")

        # График зависимости метрик от порога
        j_scores = tpr - fpr  # Youden's J statistic
        axes[1].plot(thresholds, tpr, "b-", label="True Positive Rate", linewidth=2)
        axes[1].plot(thresholds, fpr, "r-", label="False Positive Rate", linewidth=2)
        axes[1].plot(thresholds, j_scores, "g-", label="Youden's J (TPR-FPR)", linewidth=2)
        axes[1].axvline(
            optimal_threshold,
            color="black",
            linestyle="--",
            label=f"Оптимальный порог: {optimal_threshold:.4f}",
        )
        axes[1].set_xlabel("Порог классификации")
        axes[1].set_ylabel("Значение метрики")
        axes[1].set_title("Метрики vs Порог")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        print(f"💾 ROC кривая сохранена: {save_path}")
        print(f"📊 ROC-AUC: {roc_auc:.4f}")
        print(
            f"🎯 Оптимальный порог: {optimal_threshold:.4f} (TPR={optimal_tpr:.3f}, FPR={optimal_fpr:.3f})"
        )

        return {
            "auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "optimal_tpr": optimal_tpr,
            "optimal_fpr": optimal_fpr,
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
        }

    def plot_precision_recall_curve(
        self,
        y_true,
        y_scores,
        title="Precision-Recall кривая",
        save_name="precision_recall_curve.png",
    ):
        """Строит Precision-Recall кривую."""
        from sklearn.metrics import average_precision_score, precision_recall_curve

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        ap_score = average_precision_score(y_true, y_scores)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # PR кривая
        axes[0].plot(
            recall, precision, color="blue", linewidth=3, label=f"PR кривая (AP = {ap_score:.4f})"
        )
        axes[0].fill_between(recall, precision, alpha=0.2, color="blue")

        # Baseline (случайный классификатор)
        baseline = np.sum(y_true) / len(y_true)
        axes[0].axhline(
            baseline,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Baseline (random): {baseline:.4f}",
        )

        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel("Recall (True Positive Rate)")
        axes[0].set_ylabel("Precision")
        axes[0].set_title("Precision-Recall Кривая")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # F1-score vs threshold
        if len(thresholds) > 0:
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
            f1_scores = np.nan_to_num(f1_scores)

            best_f1_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_f1_idx]
            best_f1 = f1_scores[best_f1_idx]

            axes[1].plot(thresholds, f1_scores, "g-", linewidth=2, label="F1-Score")
            axes[1].plot(thresholds, precision[:-1], "b-", linewidth=2, label="Precision")
            axes[1].plot(thresholds, recall[:-1], "r-", linewidth=2, label="Recall")
            axes[1].axvline(
                best_threshold,
                color="black",
                linestyle="--",
                label=f"Лучший F1 порог: {best_threshold:.4f}",
            )
            axes[1].set_xlabel("Порог классификации")
            axes[1].set_ylabel("Значение метрики")
            axes[1].set_title(f"Метрики vs Порог (макс F1={best_f1:.3f})")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xlim([0, 1])
            axes[1].set_ylim([0, 1])

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        print(f"💾 PR кривая сохранена: {save_path}")
        print(f"📊 Average Precision: {ap_score:.4f}")

        return {
            "average_precision": ap_score,
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
        }

    def create_comprehensive_analysis(
        self,
        y_true,
        y_scores,
        y_pred=None,
        threshold=None,
        title="Комплексный анализ",
        save_name="comprehensive_analysis.png",
    ):
        """Создает комплексный анализ результатов."""
        from sklearn.metrics import classification_report, confusion_matrix

        if y_pred is None and threshold is not None:
            y_pred = (y_scores >= threshold).astype(int)

        fig = plt.figure(figsize=(20, 12))

        # 1. ROC кривая
        plt.subplot(2, 4, 1)
        roc_data = self.plot_roc_curve(y_true, y_scores, save_name="temp_roc.png")

        # 2. Precision-Recall кривая
        plt.subplot(2, 4, 2)
        pr_data = self.plot_precision_recall_curve(y_true, y_scores, save_name="temp_pr.png")

        # 3. Распределение оценок
        plt.subplot(2, 4, 3)
        normal_scores = y_scores[y_true == 0]
        attack_scores = y_scores[y_true == 1]

        plt.hist(
            normal_scores,
            bins=50,
            alpha=0.7,
            color="green",
            label=f"Нормальные (n={len(normal_scores)})",
            density=True,
        )
        plt.hist(
            attack_scores,
            bins=50,
            alpha=0.7,
            color="red",
            label=f"Атаки (n={len(attack_scores)})",
            density=True,
        )
        if threshold is not None:
            plt.axvline(
                threshold,
                color="black",
                linestyle="--",
                linewidth=2,
                label=f"Порог: {threshold:.4f}",
            )
        plt.xlabel("Оценка аномальности")
        plt.ylabel("Плотность")
        plt.title("Распределение оценок")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. Confusion Matrix
        if y_pred is not None:
            plt.subplot(2, 4, 4)
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Нормальные", "Атаки"],
                yticklabels=["Нормальные", "Атаки"],
            )
            plt.title("Матрица ошибок")
            plt.xlabel("Предсказанные")
            plt.ylabel("Истинные")

        # 5-8. Статистика и метрики
        if y_pred is not None:
            plt.subplot(2, 4, (5, 8))
            plt.axis("off")

            report = classification_report(
                y_true, y_pred, target_names=["Нормальные", "Атаки"], output_dict=True
            )

            stats_text = f"""
📊 ДЕТАЛЬНАЯ СТАТИСТИКА МОДЕЛИ

🎯 ROC-AUC Метрики:
• AUC Score: {roc_data['auc']:.4f}
• Оптимальный порог: {roc_data['optimal_threshold']:.4f}
• TPR на опт. пороге: {roc_data['optimal_tpr']:.3f}
• FPR на опт. пороге: {roc_data['optimal_fpr']:.3f}

📈 Precision-Recall:
• Average Precision: {pr_data['average_precision']:.4f}

🔍 Классификация (порог = {threshold:.4f}):
• Accuracy: {report['accuracy']:.4f}
• Macro F1: {report['macro avg']['f1-score']:.4f}
• Weighted F1: {report['weighted avg']['f1-score']:.4f}

📋 По классам:
Нормальные:
  • Precision: {report['Нормальные']['precision']:.4f}
  • Recall: {report['Нормальные']['recall']:.4f}
  • F1-Score: {report['Нормальные']['f1-score']:.4f}

Атаки:
  • Precision: {report['Атаки']['precision']:.4f}
  • Recall: {report['Атаки']['recall']:.4f}
  • F1-Score: {report['Атаки']['f1-score']:.4f}

🎲 Распределение данных:
• Нормальные: {len(normal_scores)} ({len(normal_scores)/len(y_true)*100:.1f}%)
• Атаки: {len(attack_scores)} ({len(attack_scores)/len(y_true)*100:.1f}%)

💡 Разделение классов:
• Среднее (норм): {np.mean(normal_scores):.4f}
• Среднее (атаки): {np.mean(attack_scores):.4f}
• Разность: {np.mean(attack_scores) - np.mean(normal_scores):.4f}
"""

            plt.text(
                0.05,
                0.95,
                stats_text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
            )

        plt.suptitle(title, fontsize=18, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)

        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        print(f"💾 Комплексный анализ сохранен: {save_path}")

        return {
            "roc_data": roc_data,
            "pr_data": pr_data,
            "classification_report": report if y_pred is not None else None,
        }


class ComparisonPlotter:
    """Класс для сравнения разных моделей."""

    def __init__(self, save_dir="plots", dpi=150, figsize=(12, 8)):
        self.save_dir = Path(save_dir) / "plots"  # Создаем подпапку plots
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.dpi = dpi
        self.figsize = figsize

        print(f"📊 ComparisonPlotter готов")

    def plot_metrics_comparison(
        self, results_dict, title="Сравнение моделей", save_name="models_comparison.png"
    ):
        """
        Сравнивает метрики разных моделей.

        Args:
            results_dict (dict): Словарь {model_name: {'accuracy': val, 'precision': val, ...}}
            title (str): Заголовок
            save_name (str): Имя файла
        """
        metrics = ["accuracy", "precision", "recall", "f1_score"]
        model_names = list(results_dict.keys())

        # Подготавливаем данные
        data = []
        for model_name in model_names:
            for metric in metrics:
                if metric in results_dict[model_name]:
                    data.append(
                        {
                            "Модель": model_name,
                            "Метрика": metric.title(),
                            "Значение": results_dict[model_name][metric],
                        }
                    )

        df = pd.DataFrame(data)

        # График
        plt.figure(figsize=self.figsize)
        sns.barplot(data=df, x="Метрика", y="Значение", hue="Модель")
        plt.title(title, fontsize=16, fontweight="bold")
        plt.ylabel("Значение метрики")
        plt.ylim(0, 1)
        plt.legend(title="Модель", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)

        # Добавляем значения на столбцы
        for container in plt.gca().containers:
            plt.gca().bar_label(container, fmt="%.3f", rotation=90, padding=3)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        print(f"💾 Сравнение сохранено: {save_path}")
        return str(save_path)

    def plot_training_comparison(
        self, histories_dict, title="Сравнение обучения", save_name="training_comparison.png"
    ):
        """
        Сравнивает процесс обучения разных моделей.

        Args:
            histories_dict (dict): Словарь {model_name: history}
            title (str): Заголовок
            save_name (str): Имя файла
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        colors = plt.cm.tab10(np.linspace(0, 1, len(histories_dict)))

        for i, (model_name, history) in enumerate(histories_dict.items()):
            color = colors[i]

            # Train losses
            if "train_losses" in history:
                epochs = range(1, len(history["train_losses"]) + 1)
                axes[0, 0].plot(
                    epochs, history["train_losses"], color=color, label=f"{model_name}", linewidth=2
                )

            # Val losses
            if "val_losses" in history and history["val_losses"]:
                val_epochs = range(1, len(history["val_losses"]) + 1)
                axes[0, 1].plot(
                    val_epochs,
                    history["val_losses"],
                    color=color,
                    label=f"{model_name}",
                    linewidth=2,
                )

            # Train accuracy
            if "train_accs" in history and history["train_accs"]:
                epochs = range(1, len(history["train_accs"]) + 1)
                axes[1, 0].plot(
                    epochs, history["train_accs"], color=color, label=f"{model_name}", linewidth=2
                )

            # Val accuracy
            if "val_accs" in history and history["val_accs"]:
                val_epochs = range(1, len(history["val_accs"]) + 1)
                axes[1, 1].plot(
                    val_epochs, history["val_accs"], color=color, label=f"{model_name}", linewidth=2
                )

        # Настройка осей
        axes[0, 0].set_title("Train Loss")
        axes[0, 0].set_xlabel("Эпоха")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_title("Validation Loss")
        axes[0, 1].set_xlabel("Эпоха")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_title("Train Accuracy")
        axes[1, 0].set_xlabel("Эпоха")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].set_title("Validation Accuracy")
        axes[1, 1].set_xlabel("Эпоха")
        axes[1, 1].set_ylabel("Accuracy")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        print(f"💾 Сравнение обучения сохранено: {save_path}")
        return str(save_path)


def create_summary_report(results, output_path="results_summary.html"):
    """Создает HTML отчет с результатами."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NeuroDetekt - Отчет о результатах</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .metrics {{ background-color: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .warning {{ background-color: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧠 NeuroDetekt - Отчет о результатах</h1>
            <p>Система обнаружения вторжений на основе нейронных сетей</p>
        </div>
        
        <h2>📊 Сводка результатов</h2>
        <div class="metrics">
            <p>Здесь будут результаты...</p>
        </div>
    </body>
    </html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"📄 HTML отчет создан: {output_path}")
