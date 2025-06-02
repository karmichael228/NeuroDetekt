#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NeuroDetekt - Главный модуль системы обнаружения вторжений

Объединяет все компоненты системы:
- Модели (LSTM, GRU-Autoencoder)
- Тренировка и валидация
- Тестирование и оценка
- Визуализация результатов

Использование:
    python main.py --mode train --model lstm --config config.yaml
    python main.py --mode test --model autoencoder --checkpoint model.pth
    python main.py --mode compare --models lstm,autoencoder
"""

import argparse
import yaml
from pathlib import Path
import torch
import sys
import time
import os
import numpy as np
from datetime import datetime

# Импорты модулей системы
from .models import LSTMModel, GRUAutoEncoder, create_lstm_model
from .training import LSTMTrainer, AutoencoderTrainer
from .testing import LSTMEvaluator, AutoencoderEvaluator
from .visualization import TrainingPlotter, ResultsPlotter, ComparisonPlotter
from .utils import get_data, load_data_splits, calculate_metrics, print_metrics_report


class NeuroDetekt:
    """Главный класс системы NeuroDetekt."""
    
    def __init__(self, config_path=None):
        """
        Args:
            config_path (str): Путь к файлу конфигурации
        """
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🧠 NeuroDetekt инициализирован")
        print(f"   💻 Устройство: {self.device}")
        print(f"   ⚙️ Конфигурация: {config_path or 'по умолчанию'}")
    
    def load_config(self, config_path):
        """Загружает конфигурацию из файла или использует значения по умолчанию."""
        default_config = {
            'data': {
                'dataset': 'plaid',
                'batch_size': 64,
                'sequence_length': 100
            },
            'lstm': {
                'vocab_size': 229,
                'embedding_dim': 128,
                'hidden_size': 128,
                'num_layers': 2,
                'dropout': 0.25,
                'learning_rate': 1e-4
            },
            'autoencoder': {
                'vocab_size': 229,
                'embedding_dim': 128,
                'hidden_size': 128,
                'num_layers': 2,
                'latent_dim': 64,  # Размерность латентного пространства
                'dropout': 0.2,
                'learning_rate': 5e-5
            },
            'training': {
                'epochs': 15,
                'patience': 8,
                'gradient_clip': 1.0
            },
            'paths': {
                'data_dir': 'data',
                'base_dir': 'trials',  # Базовая директория для экспериментов
                'logs_dir': 'logs',
                'plots_dir': 'plots'
            },
            'testing': {
                'auto_test': True,  # Автоматическое тестирование после обучения
                'save_predictions': True,
                'plot_results': True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            # Объединяем конфигурации
            self._merge_configs(default_config, user_config)
        
        return default_config
    
    def _merge_configs(self, default, user):
        """Рекурсивно объединяет конфигурации."""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value
    
    def _create_experiment_dir(self, experiment_name):
        """Создает директорию для эксперимента."""
        experiment_dir = Path(self.config['paths']['base_dir']) / experiment_name
        experiment_dir.mkdir(exist_ok=True, parents=True)
        
        # Обновляем пути в конфигурации для данного эксперимента
        self.current_experiment_dir = experiment_dir
        self.current_model_path = experiment_dir / f"{experiment_name}_best.pth"
        self.current_final_path = experiment_dir / f"{experiment_name}_final.pth"
        
        return experiment_dir
    
    def train_lstm(self, experiment_name="lstm_model", auto_test=None):
        """Обучает LSTM модель."""
        print(f"\n🧠 Обучение LSTM модели...")
        
        # Создаем директорию эксперимента
        experiment_dir = self._create_experiment_dir(experiment_name)
        print(f"📁 Директория эксперимента: {experiment_dir}")
        
        # Загружаем данные
        train_loader, val_loader, (test_loader, test_labels) = get_data(
            self.config['data']['dataset'],
            batch_size=self.config['data']['batch_size']
        )
        
        # Создаем модель
        model, optimizer, criterion = create_lstm_model(
            vocab_size=self.config['lstm']['vocab_size'],
            embedding_dim=self.config['lstm']['embedding_dim'],
            hidden_size=self.config['lstm']['hidden_size'],
            num_layers=self.config['lstm']['num_layers'],
            dropout=self.config['lstm']['dropout'],
            device=str(self.device),
            learning_rate=self.config['lstm']['learning_rate']
        )
        
        # Создаем тренер
        trainer = LSTMTrainer(
            model, optimizer, criterion,
            device=str(self.device),
            log_dir=str(experiment_dir),
            gradient_clip=self.config['training']['gradient_clip']
        )
        
        # Настраиваем раннюю остановку
        trainer.setup_early_stopping(
            patience=self.config['training']['patience'],
            path=str(self.current_model_path)
        )
        
        # Обучаем
        print(f"🚀 Начало обучения эксперимента '{experiment_name}'...")
        start_time = time.time()
        
        history = trainer.train(
            train_loader, val_loader,
            epochs=self.config['training']['epochs']
        )
        
        training_time = time.time() - start_time
        
        # Сохраняем финальную модель
        torch.save(model.state_dict(), self.current_final_path)
        print(f"💾 Модели сохранены:")
        print(f"   Best: {self.current_model_path}")
        print(f"   Final: {self.current_final_path}")
        
        # Визуализируем тренировку
        print(f"🔧 Создание плоттера...")
        try:
            plotter = TrainingPlotter(str(experiment_dir))
            print(f"🔧 Создание графика обучения...")
            training_plot = plotter.plot_training_history(history, f"LSTM обучение - {experiment_name}", model_type='lstm')
            print(f"📊 График обучения: {training_plot}")
        except Exception as e:
            print(f"⚠️ Ошибка при создании графика: {e}")
            print(f"   (Продолжаем без визуализации...)")
        
        # Сохраняем историю обучения
        import pickle
        history_path = experiment_dir / f"{experiment_name}_training_history.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        print(f"📈 История обучения: {history_path}")
        
        # Создаем summary файл
        print(f"🔧 Вызов _save_training_summary...")
        try:
            self._save_training_summary(experiment_dir, experiment_name, history, training_time, 'LSTM')
        except Exception as e:
            print(f"❌ Ошибка при создании summary: {e}")
            import traceback
            traceback.print_exc()
        
        results = {
            'model': model,
            'history': history,
            'trainer': trainer,
            'experiment_dir': experiment_dir,
            'model_path': self.current_model_path,
            'training_time': training_time
        }
        
        # Автоматическое тестирование
        if auto_test is None:
            auto_test = self.config['testing'].get('auto_test', True)
        
        print(f"🔧 Проверка auto_test: {auto_test}")
        if auto_test:
            print(f"\n🧪 Запуск автоматического тестирования...")
            try:
                test_results = self._test_trained_model(
                    experiment_name, 'lstm', test_loader, test_labels
                )
                results['test_results'] = test_results
                print(f"✅ Автотестирование завершено")
                
                # Пересоздаем summary с результатами тестирования
                try:
                    self._save_training_summary(experiment_dir, experiment_name, history, training_time, 'LSTM', test_results)
                except Exception as e:
                    print(f"❌ Ошибка при создании финального summary: {e}")
                    
            except Exception as e:
                print(f"❌ Ошибка автотестирования: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️ Автотестирование отключено")
            # Создаем summary без результатов тестирования
            try:
                self._save_training_summary(experiment_dir, experiment_name, history, training_time, 'LSTM')
            except Exception as e:
                print(f"❌ Ошибка при создании summary: {e}")
        
        # Финальное сообщение пользователю
        print(f"\n{'='*60}")
        print(f"🎉 ОБУЧЕНИЕ LSTM МОДЕЛИ ЗАВЕРШЕНО")
        print(f"{'='*60}")
        print(f"📁 Директория эксперимента: {experiment_dir}")
        print(f"🏆 Лучшая модель: {self.current_model_path}")
        print(f"📄 Summary с метриками: {experiment_dir / f'{experiment_name}_summary.txt'}")
        if 'test_results' in results:
            print(f"📊 Результаты тестирования включены в summary")
        print(f"⏱️ Время обучения: {training_time:.1f} секунд")
        print(f"{'='*60}")
        
        return results
    
    def _test_trained_model(self, experiment_name, model_type, test_loader, test_labels):
        """Автоматически тестирует обученную модель."""
        try:
            # Загружаем лучшую модель
            if model_type.lower() == 'lstm':
                model, _, _ = create_lstm_model(
                    vocab_size=self.config['lstm']['vocab_size'],
                    embedding_dim=self.config['lstm']['embedding_dim'],
                    hidden_size=self.config['lstm']['hidden_size'],
                    num_layers=self.config['lstm']['num_layers'],
                    dropout=self.config['lstm']['dropout'],
                    device=str(self.device),
                    learning_rate=self.config['lstm']['learning_rate']
                )
                model.load_state_dict(torch.load(self.current_model_path, map_location=self.device))
                model.eval()
                
                # Простая оценка на тестовых данных
                from .utils.metrics import calculate_accuracy
                test_results = self._evaluate_lstm_on_test(
                    model, test_loader, test_labels, experiment_name
                )
            
            elif model_type.lower() in ['autoencoder', 'gru-autoencoder']:
                # Автотестирование для автоэнкодера
                test_results = self._evaluate_autoencoder_on_test(
                    test_loader, test_labels, experiment_name
                )
            
            else:
                print(f"⚠️ Автотестирование для {model_type} пока не реализовано")
                return None
                
            return test_results
            
        except Exception as e:
            print(f"❌ Ошибка при автотестировании: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _evaluate_lstm_on_test(self, model, test_loader, test_labels, experiment_name):
        """Оценивает LSTM модель на тестовых данных."""
        model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        all_outputs = []
        all_targets = []
        
        criterion = torch.nn.CrossEntropyLoss()
        
        print(f"🔍 Оценка на тестовых данных...")
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = batch.to(self.device)
                
                # Подготавливаем данные
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                
                # Прогоняем через модель
                outputs = model(inputs)
                
                # Вычисляем потери и точность
                loss = criterion(
                    outputs.contiguous().view(-1, outputs.size(-1)),
                    targets.contiguous().view(-1)
                )
                
                from .utils.metrics import calculate_accuracy
                accuracy = calculate_accuracy(outputs, targets)
                
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
                
                # Сохраняем для анализа
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        # Дополнительный анализ для обнаружения аномалий
        # Преобразуем LSTM выходы в оценки аномальности
        print(f"🔬 Создание оценок аномальности...")
        anomaly_scores = []
        true_labels = test_labels.cpu().numpy()
        
        for batch_outputs, batch_targets in zip(all_outputs, all_targets):
            # Используем перплексию как меру аномальности
            batch_scores = []
            for seq_outputs, seq_targets in zip(batch_outputs, batch_targets):
                # Вычисляем средний cross-entropy для последовательности
                seq_loss = torch.nn.functional.cross_entropy(
                    seq_outputs.view(-1, seq_outputs.size(-1)),
                    seq_targets.view(-1),
                    reduction='mean'
                )
                batch_scores.append(seq_loss.item())
            anomaly_scores.extend(batch_scores)
        
        anomaly_scores = np.array(anomaly_scores)
        
        # Создаем расширенную визуализацию и анализ
        print(f"📊 Создание визуализации и анализа...")
        try:
            from .visualization import ResultsPlotter
            plotter = ResultsPlotter(str(self.current_experiment_dir))
            
            # ROC-AUC анализ
            roc_results = plotter.plot_roc_curve(
                true_labels, anomaly_scores,
                title=f"ROC-AUC анализ - {experiment_name}",
                save_name=f"{experiment_name}_roc_curve.png"
            )
            
            # Precision-Recall анализ
            pr_results = plotter.plot_precision_recall_curve(
                true_labels, anomaly_scores,
                title=f"Precision-Recall анализ - {experiment_name}",
                save_name=f"{experiment_name}_precision_recall.png"
            )
            
            # Комплексный анализ
            optimal_threshold = roc_results['optimal_threshold']
            comprehensive_results = plotter.create_comprehensive_analysis(
                true_labels, anomaly_scores, threshold=optimal_threshold,
                title=f"Комплексный анализ - {experiment_name}",
                save_name=f"{experiment_name}_comprehensive_analysis.png"
            )
            
            print(f"📈 Расширенная визуализация создана")
            
        except Exception as e:
            print(f"⚠️ Ошибка при создании расширенной визуализации: {e}")
            roc_results = {'auc': 0, 'optimal_threshold': 0.5}
            pr_results = {'average_precision': 0}
            comprehensive_results = {}
        
        # Сохраняем результаты тестирования
        test_results = {
            'test_loss': avg_loss,
            'test_accuracy': avg_accuracy,
            'num_batches': num_batches,
            'total_samples': len(test_loader.dataset),
            'roc_auc': roc_results.get('auc', 0),
            'average_precision': pr_results.get('average_precision', 0),
            'optimal_threshold': roc_results.get('optimal_threshold', 0.5),
            'anomaly_scores': anomaly_scores,
            'true_labels': true_labels
        }
        
        # Сохраняем детальные результаты в файл
        results_path = self.current_experiment_dir / f"{experiment_name}_test_results.txt"
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write(f"🧪 Детальные результаты тестирования модели {experiment_name}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"📋 ОСНОВНЫЕ МЕТРИКИ:\n")
            f.write(f"Тип модели: LSTM\n")
            f.write(f"Test Loss: {avg_loss:.4f}\n")
            f.write(f"Test Accuracy: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)\n")
            f.write(f"Батчей: {num_batches}\n")
            f.write(f"Образцов: {len(test_loader.dataset)}\n\n")
            
            f.write(f"🎯 ОБНАРУЖЕНИЕ АНОМАЛИЙ:\n")
            f.write(f"ROC-AUC: {roc_results.get('auc', 0):.4f}\n")
            f.write(f"Average Precision: {pr_results.get('average_precision', 0):.4f}\n")
            f.write(f"Оптимальный порог: {roc_results.get('optimal_threshold', 0.5):.4f}\n\n")
            
            f.write(f"📊 СТАТИСТИКА ОЦЕНОК:\n")
            normal_scores = anomaly_scores[true_labels == 0] 
            attack_scores = anomaly_scores[true_labels == 1]
            f.write(f"Нормальные данные:\n")
            f.write(f"  • Среднее: {np.mean(normal_scores):.4f}\n")
            f.write(f"  • Медиана: {np.median(normal_scores):.4f}\n")
            f.write(f"  • Ст. откл.: {np.std(normal_scores):.4f}\n")
            f.write(f"  • Количество: {len(normal_scores)}\n\n")
            
            f.write(f"Атаки:\n")
            f.write(f"  • Среднее: {np.mean(attack_scores):.4f}\n")
            f.write(f"  • Медиана: {np.median(attack_scores):.4f}\n")
            f.write(f"  • Ст. откл.: {np.std(attack_scores):.4f}\n")
            f.write(f"  • Количество: {len(attack_scores)}\n\n")
            
            f.write(f"💡 РАЗДЕЛЕНИЕ КЛАССОВ:\n")
            f.write(f"Разность средних: {np.mean(attack_scores) - np.mean(normal_scores):.4f}\n")
            
            # Практические результаты с оптимальным порогом
            optimal_threshold = roc_results.get('optimal_threshold', 0.5)
            predictions = (anomaly_scores >= optimal_threshold).astype(int)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            f.write(f"\n🎯 КЛАССИФИКАЦИЯ (порог = {optimal_threshold:.4f}):\n")
            f.write(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}\n")
            f.write(f"Precision: {precision_score(true_labels, predictions):.4f}\n")
            f.write(f"Recall: {recall_score(true_labels, predictions):.4f}\n")
            f.write(f"F1-Score: {f1_score(true_labels, predictions):.4f}\n\n")
            
            # Практический пример
            total_normal = len(normal_scores)
            total_attacks = len(attack_scores)
            detected_attacks = np.sum(predictions[true_labels == 1])
            false_alarms = np.sum(predictions[true_labels == 0])
            
            f.write(f"📈 ПРАКТИЧЕСКИЕ РЕЗУЛЬТАТЫ:\n")
            f.write(f"Из {total_attacks} атак обнаружено: {detected_attacks} ({detected_attacks/total_attacks*100:.1f}%)\n")
            f.write(f"Ложных тревог из {total_normal}: {false_alarms} ({false_alarms/total_normal*100:.1f}%)\n")
        
        print(f"📊 Результаты тестирования:")
        print(f"   Test Loss: {avg_loss:.4f}")
        print(f"   Test Accuracy: {avg_accuracy:.3f} ({avg_accuracy*100:.1f}%)")
        print(f"   ROC-AUC: {roc_results.get('auc', 0):.4f}")
        print(f"   Average Precision: {pr_results.get('average_precision', 0):.4f}")
        print(f"   Результаты сохранены: {results_path}")
        
        return test_results
    
    def _save_training_summary(self, experiment_dir, experiment_name, history, training_time, model_type, test_results=None):
        """Сохраняет сводку обучения в формате как в примере."""
        summary_path = experiment_dir / f"{experiment_name}_summary.txt"
        
        # Получаем текущую дату и время
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Проверяем, какие ключи используются в истории
        if 'val_loss' in history:
            # Старый формат LSTM
            best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
            best_train_loss = history['train_loss'][best_epoch - 1]
            best_val_loss = history['val_loss'][best_epoch - 1]
            best_train_acc = history['train_acc'][best_epoch - 1]
            best_val_acc = history['val_acc'][best_epoch - 1]
            total_epochs = len(history['train_loss'])
        elif 'val_losses' in history:
            # Новый формат LSTM
            best_epoch = history['val_losses'].index(min(history['val_losses'])) + 1
            best_train_loss = history['train_losses'][best_epoch - 1]
            best_val_loss = history['val_losses'][best_epoch - 1]
            best_train_acc = history['train_accs'][best_epoch - 1]
            best_val_acc = history['val_accs'][best_epoch - 1]
            total_epochs = len(history['train_losses'])
        elif 'val_mean_errors' in history:
            # Формат автоэнкодера
            if history['val_mean_errors']:
                best_epoch = history['val_mean_errors'].index(min(history['val_mean_errors'])) + 1
                best_train_loss = history['train_losses'][best_epoch - 1]
                best_val_error = history['val_mean_errors'][best_epoch - 1]
            else:
                best_epoch = len(history['train_losses'])
                best_train_loss = history['train_losses'][-1]
                best_val_error = 0
            total_epochs = len(history['train_losses'])
            # Для автоэнкодера нет accuracy метрик
            best_train_acc = None
            best_val_acc = None
            best_val_loss = best_val_error  # Используем ошибку как loss
        else:
            print(f"⚠️ Неизвестный формат истории обучения: {list(history.keys())}")
            return
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            # Заголовок
            f.write(f"Обучение {model_type} модели '{experiment_name}' - {current_datetime}\n")
            f.write("=" * 60 + "\n\n")
            
            # Параметры модели
            f.write("Параметры:\n")
            if model_type.upper() == 'LSTM':
                f.write(f"  - Размер батча: {self.config['data']['batch_size']}\n")
                f.write(f"  - Эпохи: {total_epochs}\n")
                f.write(f"  - Learning rate: {self.config['lstm']['learning_rate']}\n")
                f.write(f"  - Архитектура: {self.config['lstm']['hidden_size']} hidden, {self.config['lstm']['num_layers']} layers, dropout {self.config['lstm']['dropout']}\n")
                f.write(f"  - Устройство: {self.device}\n")
                f.write(f"  - Early stopping: включено\n")
            elif model_type.upper() == 'GRU-AUTOENCODER':
                f.write(f"  - Размер батча: {self.config['data']['batch_size']}\n")
                f.write(f"  - Эпохи: {total_epochs}\n")
                f.write(f"  - Learning rate: {self.config['autoencoder']['learning_rate']}\n")
                f.write(f"  - Архитектура: {self.config['autoencoder']['hidden_size']} hidden, {self.config['autoencoder']['num_layers']} layers, dropout {self.config['autoencoder']['dropout']}\n")
                f.write(f"  - Устройство: {self.device}\n")
                f.write(f"  - Early stopping: включено\n")
            
            f.write(f"\n")
            
            # Результаты тестирования (если есть)
            if test_results:
                f.write("Результаты:\n")
                
                # Вычисляем метрики классификации с оптимальным порогом
                if 'optimal_threshold' in test_results and 'anomaly_scores' in test_results and 'true_labels' in test_results:
                    threshold = test_results['optimal_threshold']
                    scores = test_results['anomaly_scores']
                    labels = test_results['true_labels']
                    
                    predictions = (scores >= threshold).astype(int)
                    
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    accuracy = accuracy_score(labels, predictions)
                    precision = precision_score(labels, predictions)
                    recall = recall_score(labels, predictions)
                    f1 = f1_score(labels, predictions)
                    
                    f.write(f"  - Accuracy: {accuracy:.4f}\n")
                    f.write(f"  - Precision: {precision:.4f}\n")
                    f.write(f"  - Recall: {recall:.4f}\n")
                    f.write(f"  - F1-мера: {f1:.4f}\n")
                    f.write(f"  - Оптимальный порог: {threshold:.4f}\n")
                    
                    if 'roc_auc' in test_results:
                        f.write(f"  - ROC-AUC: {test_results['roc_auc']:.4f}\n")
                    if 'average_precision' in test_results:
                        f.write(f"  - Average Precision: {test_results['average_precision']:.4f}\n")
                        
                else:
                    # Базовые метрики если детальных нет
                    f.write(f"  - Test Loss: {test_results.get('test_loss', 'N/A')}\n")
                    f.write(f"  - Test Accuracy: {test_results.get('test_accuracy', 'N/A')}\n")
                    if 'roc_auc' in test_results:
                        f.write(f"  - ROC-AUC: {test_results['roc_auc']:.4f}\n")
            else:
                # Если тестирование не проводилось, показываем метрики обучения
                f.write("Результаты обучения:\n")
                f.write(f"  - Лучшая эпоха: {best_epoch}\n")
                f.write(f"  - Train Loss: {best_train_loss:.4f}\n")
                f.write(f"  - Val Loss: {best_val_loss:.4f}\n")
                
                # Accuracy метрики только для LSTM
                if best_train_acc is not None and best_val_acc is not None:
                    f.write(f"  - Train Accuracy: {best_train_acc:.4f}\n")
                    f.write(f"  - Val Accuracy: {best_val_acc:.4f}\n")
                elif model_type.upper() == 'GRU-AUTOENCODER':
                    # Для автоэнкодера показываем ошибки реконструкции
                    f.write(f"  - Val Mean Error: {best_val_loss:.4f}\n")
            
            f.write(f"\nВремя обучения: {training_time:.2f} секунд\n")
        
        print(f"📄 Сводка эксперимента: {summary_path}")
    
    def train_autoencoder(self, experiment_name="autoencoder_model", auto_test=None):
        """Обучает GRU автоэнкодер."""
        print(f"\n🤖 Обучение GRU автоэнкодера...")
        
        # Создаем директорию эксперимента
        experiment_dir = self._create_experiment_dir(experiment_name)
        print(f"📁 Директория эксперимента: {experiment_dir}")
        
        # Загружаем данные (только нормальные для автоэнкодера)
        train_loader, val_loader, (test_loader, test_labels) = get_data(
            self.config['data']['dataset'],
            batch_size=self.config['data']['batch_size'],
            normal_only=True  # Только нормальные данные
        )
        
        # Создаем модель
        from .models.gru_autoencoder import create_gru_autoencoder
        model, optimizer, criterion = create_gru_autoencoder(
            vocab_size=self.config['autoencoder']['vocab_size'],
            embedding_dim=self.config['autoencoder']['embedding_dim'],
            hidden_size=self.config['autoencoder']['hidden_size'],
            num_layers=self.config['autoencoder']['num_layers'],
            latent_dim=self.config['autoencoder']['latent_dim'],
            dropout=self.config['autoencoder']['dropout'],
            device=str(self.device),
            learning_rate=self.config['autoencoder']['learning_rate']
        )
        
        # Создаем тренер
        trainer = AutoencoderTrainer(
            model, optimizer, criterion,
            device=str(self.device),
            log_dir=str(experiment_dir)
        )
        
        # Обучаем
        print(f"🚀 Начало обучения автоэнкодера '{experiment_name}'...")
        start_time = time.time()
        
        results = trainer.train_autoencoder(
            train_loader, val_loader,
            epochs=self.config['training']['epochs'],
            patience=self.config['training']['patience'],
            model_name=experiment_name,
            output_dir=str(experiment_dir)
        )
        
        training_time = time.time() - start_time
        
        # Визуализируем
        plotter = TrainingPlotter(str(experiment_dir))
        training_plot = plotter.plot_training_history(
            results['training_history'], 
            f"Автоэнкодер обучение - {experiment_name}",
            model_type='autoencoder'
        )
        print(f"📊 График обучения: {training_plot}")
        
        # Сохраняем историю обучения
        import pickle
        history_path = experiment_dir / f"{experiment_name}_training_history.pkl"
        with open(history_path, 'wb') as f:
            pickle.dump(results['training_history'], f)
        print(f"📈 История обучения: {history_path}")
        
        experiment_results = {
            'model': model,
            'results': results,
            'trainer': trainer,
            'experiment_dir': experiment_dir,
            'training_time': training_time
        }
        
        # Автоматическое тестирование
        if auto_test is None:
            auto_test = self.config['testing'].get('auto_test', True)
            
        if auto_test:
            print(f"\n🧪 Запуск автоматического тестирования автоэнкодера...")
            test_results = self._test_trained_model(
                experiment_name, 'autoencoder', test_loader, test_labels
            )
            experiment_results['test_results'] = test_results
            
            # Пересоздаем summary с результатами тестирования
            try:
                self._save_training_summary(experiment_dir, experiment_name, results['training_history'], training_time, 'GRU-Autoencoder', test_results)
            except Exception as e:
                print(f"❌ Ошибка при создании финального summary: {e}")
        else:
            print(f"⚠️ Автотестирование отключено")
            # Создаем summary без результатов тестирования
            try:
                self._save_training_summary(experiment_dir, experiment_name, results['training_history'], training_time, 'GRU-Autoencoder')
            except Exception as e:
                print(f"❌ Ошибка при создании summary: {e}")
        
        # Финальное сообщение пользователю
        print(f"\n{'='*60}")
        print(f"🎉 ОБУЧЕНИЕ GRU-AUTOENCODER ЗАВЕРШЕНО")
        print(f"{'='*60}")
        print(f"📁 Директория эксперимента: {experiment_dir}")
        print(f"🏆 Лучшая модель: {experiment_dir / f'{experiment_name}_best.pth'}")
        print(f"📄 Summary с метриками: {experiment_dir / f'{experiment_name}_summary.txt'}")
        if 'test_results' in experiment_results:
            print(f"📊 Результаты тестирования включены в summary")
        print(f"⏱️ Время обучения: {training_time:.1f} секунд")
        print(f"{'='*60}")
        
        return experiment_results
    
    def test_model(self, model_path, model_type="lstm", experiment_name=None):
        """Тестирует обученную модель."""
        print(f"\n🧪 Тестирование {model_type.upper()} модели...")
        
        # Если эксперимент не указан, создаем имя из пути
        if experiment_name is None:
            experiment_name = f"test_{Path(model_path).stem}"
        
        # Создаем директорию для результатов тестирования
        test_dir = self._create_experiment_dir(f"test_{experiment_name}")
        
        # Загружаем тестовые данные
        _, _, (test_loader, test_labels) = get_data(
            self.config['data']['dataset'],
            batch_size=self.config['data']['batch_size']
        )
        
        if model_type.lower() == "lstm":
            # Создаем и загружаем LSTM модель
            evaluator = LSTMEvaluator.load_from_checkpoint(
                model_path, LSTMModel,
                model_params={'vocab_size': self.config['lstm']['vocab_size']},
                device=str(self.device)
            )
            
            # Тестируем (нужно разделить данные на нормальные и атаки)
            # Это упрощенная версия - в реальности нужно правильно разделить данные
            results = evaluator.evaluate_anomaly_detection(
                test_loader, test_loader  # Заглушка
            )
            
        elif model_type.lower() == "autoencoder":
            # Создаем и загружаем автоэнкодер
            evaluator = AutoencoderEvaluator.load_from_checkpoint(
                model_path, GRUAutoEncoder,
                model_params={'vocab_size': self.config['autoencoder']['vocab_size']},
                device=str(self.device)
            )
            
            # Тестируем
            results = evaluator.evaluate_anomaly_detection(
                test_loader, test_loader  # Заглушка
            )
        
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
        
        # Визуализируем результаты
        plotter = ResultsPlotter(str(test_dir))
        if 'normal_scores' in results and 'attack_scores' in results:
            plotter.plot_score_distributions(
                results['normal_scores'], 
                results['attack_scores'],
                results.get('threshold'),
                f"Результаты {model_type.upper()}"
            )
        
        return results
    
    def compare_models(self, model_specs, comparison_name="model_comparison"):
        """Сравнивает несколько моделей.
        
        Args:
            model_specs: список кортежей (path, model_type) или строк "type:path"
            comparison_name: имя для сравнения
        """
        print(f"\n📊 Сравнение моделей...")
        
        # Создаем директорию для сравнения
        comparison_dir = self._create_experiment_dir(comparison_name)
        
        results = {}
        histories = {}
        
        # Парсим спецификации моделей
        if isinstance(model_specs, str):
            model_specs = model_specs.split(',')
        
        parsed_specs = []
        for spec in model_specs:
            if isinstance(spec, tuple):
                path, model_type = spec
            else:
                if ':' in spec:
                    model_type, path = spec.split(':', 1)
                else:
                    raise ValueError(f"Неверный формат спецификации модели: {spec}")
            parsed_specs.append((path, model_type))
        
        for path, model_type in parsed_specs:
            print(f"   Тестирование {model_type}: {path}")
            model_results = self.test_model(path, model_type, f"{comparison_name}_{model_type}")
            results[f"{model_type}"] = model_results.get('evaluation', {}).get('metrics', {})
            
            # Загружаем историю обучения если есть
            history_path = Path(path).parent / "training_history.pkl"
            if history_path.exists():
                import pickle
                with open(history_path, 'rb') as f:
                    histories[model_type] = pickle.load(f)
        
        # Создаем сравнительные графики
        plotter = ComparisonPlotter(str(comparison_dir))
        
        if results:
            plotter.plot_metrics_comparison(results, "Сравнение метрик моделей")
        
        if histories:
            plotter.plot_training_comparison(histories, "Сравнение процесса обучения")
        
        return results
    
    def create_config_template(self, output_path="config_template.yaml"):
        """Создает шаблон конфигурационного файла."""
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        print(f"📄 Шаблон конфигурации создан: {output_path}")

    def _evaluate_autoencoder_on_test(self, test_loader, test_labels, experiment_name):
        """Оценивает автоэнкодер на тестовых данных."""
        print(f"🔍 Оценка автоэнкодера на тестовых данных...")
        
        # Загружаем обученную модель
        from .models.gru_autoencoder import create_gru_autoencoder
        model, _, _ = create_gru_autoencoder(
            vocab_size=self.config['autoencoder']['vocab_size'],
            embedding_dim=self.config['autoencoder']['embedding_dim'],
            hidden_size=self.config['autoencoder']['hidden_size'],
            num_layers=self.config['autoencoder']['num_layers'],
            latent_dim=self.config['autoencoder']['latent_dim'],
            dropout=self.config['autoencoder']['dropout'],
            device=str(self.device),
            learning_rate=self.config['autoencoder']['learning_rate']
        )
        
        # Загружаем веса лучшей модели
        checkpoint = torch.load(self.current_model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        
        # Тестируем автоэнкодер
        from .training import AutoencoderTrainer
        trainer = AutoencoderTrainer(model, None, None, device=str(self.device))
        
        # Получаем порог из checkpoint
        threshold = checkpoint.get('val_error', 0) * 1.5  # Примерный порог
        
        # Если есть сохраненный порог, используем его
        if hasattr(self, 'current_experiment_dir'):
            # Ищем pickle файл с результатами обучения
            import pickle
            try:
                history_path = self.current_experiment_dir / f"{experiment_name}_training_history.pkl"
                if history_path.exists():
                    with open(history_path, 'rb') as f:
                        history = pickle.load(f)
                    # Устанавливаем порог как 95-й перцентиль валидационных ошибок
                    if 'val_mean_errors' in history and history['val_mean_errors']:
                        threshold = np.percentile(history['val_mean_errors'], 95)
            except:
                pass
        
        # Получаем ошибки реконструкции для всех тестовых данных
        print(f"🔬 Вычисление ошибок реконструкции...")
        all_errors = []
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                # Обрабатываем разные форматы batch данных
                if isinstance(batch, tuple):
                    batch_x, _ = batch
                else:
                    batch_x = batch
                
                batch_x = batch_x.to(self.device)
                errors = model.get_reconstruction_error(batch_x)
                all_errors.extend(errors.cpu().numpy())
        
        anomaly_scores = np.array(all_errors)
        true_labels = test_labels.cpu().numpy() if hasattr(test_labels, 'cpu') else test_labels
        
        # Разделяем ошибки по типам данных
        normal_errors = anomaly_scores[true_labels == 0]
        attack_errors = anomaly_scores[true_labels == 1]
        
        # Установка оптимального порога
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(true_labels, anomaly_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"🎯 Оптимальный порог: {optimal_threshold:.4f}")
        
        # Создаем расширенную визуализацию и анализ
        print(f"📊 Создание визуализации и анализа...")
        try:
            from .visualization import ResultsPlotter
            plotter = ResultsPlotter(str(self.current_experiment_dir))
            
            # ROC-AUC анализ
            roc_results = plotter.plot_roc_curve(
                true_labels, anomaly_scores,
                title=f"ROC-AUC анализ автоэнкодера - {experiment_name}",
                save_name=f"{experiment_name}_roc_curve.png"
            )
            
            # Precision-Recall анализ
            pr_results = plotter.plot_precision_recall_curve(
                true_labels, anomaly_scores,
                title=f"Precision-Recall анализ автоэнкодера - {experiment_name}",
                save_name=f"{experiment_name}_precision_recall.png"
            )
            
            # Комплексный анализ
            comprehensive_results = plotter.create_comprehensive_analysis(
                true_labels, anomaly_scores, threshold=optimal_threshold,
                title=f"Комплексный анализ автоэнкодера - {experiment_name}",
                save_name=f"{experiment_name}_comprehensive_analysis.png"
            )
            
            print(f"📈 Расширенная визуализация создана")
            
        except Exception as e:
            print(f"⚠️ Ошибка при создании расширенной визуализации: {e}")
            roc_results = {'auc': 0, 'optimal_threshold': optimal_threshold}
            pr_results = {'average_precision': 0}
            comprehensive_results = {}
        
        # Сохраняем результаты тестирования
        test_results = {
            'reconstruction_threshold': threshold,
            'optimal_threshold': optimal_threshold,
            'total_samples': len(anomaly_scores),
            'normal_samples': len(normal_errors),
            'attack_samples': len(attack_errors),
            'roc_auc': roc_results.get('auc', 0),
            'average_precision': pr_results.get('average_precision', 0),
            'anomaly_scores': anomaly_scores,
            'true_labels': true_labels,
            'normal_mean_error': np.mean(normal_errors),
            'attack_mean_error': np.mean(attack_errors),
            'separation': np.mean(attack_errors) - np.mean(normal_errors)
        }
        
        # Сохраняем детальные результаты в файл
        results_path = self.current_experiment_dir / f"{experiment_name}_test_results.txt"
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write(f"🧪 Детальные результаты тестирования автоэнкодера {experiment_name}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"📋 ОСНОВНЫЕ МЕТРИКИ:\n")
            f.write(f"Тип модели: GRU-Autoencoder\n")
            f.write(f"Тестовых образцов: {len(anomaly_scores)}\n")
            f.write(f"Нормальных: {len(normal_errors)}\n")
            f.write(f"Атак: {len(attack_errors)}\n\n")
            
            f.write(f"🎯 ОБНАРУЖЕНИЕ АНОМАЛИЙ:\n")
            f.write(f"ROC-AUC: {roc_results.get('auc', 0):.4f}\n")
            f.write(f"Average Precision: {pr_results.get('average_precision', 0):.4f}\n")
            f.write(f"Оптимальный порог: {optimal_threshold:.4f}\n\n")
            
            f.write(f"📊 СТАТИСТИКА ОШИБОК РЕКОНСТРУКЦИИ:\n")
            f.write(f"Нормальные данные:\n")
            f.write(f"  • Среднее: {np.mean(normal_errors):.4f}\n")
            f.write(f"  • Медиана: {np.median(normal_errors):.4f}\n")
            f.write(f"  • Ст. откл.: {np.std(normal_errors):.4f}\n")
            f.write(f"  • Количество: {len(normal_errors)}\n\n")
            
            f.write(f"Атаки:\n")
            f.write(f"  • Среднее: {np.mean(attack_errors):.4f}\n")
            f.write(f"  • Медиана: {np.median(attack_errors):.4f}\n")
            f.write(f"  • Ст. откл.: {np.std(attack_errors):.4f}\n")
            f.write(f"  • Количество: {len(attack_errors)}\n\n")
            
            f.write(f"💡 РАЗДЕЛЕНИЕ КЛАССОВ:\n")
            f.write(f"Разность средних: {np.mean(attack_errors) - np.mean(normal_errors):.4f}\n")
            
            # Практические результаты с оптимальным порогом
            predictions = (anomaly_scores >= optimal_threshold).astype(int)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            f.write(f"\n🎯 КЛАССИФИКАЦИЯ (порог = {optimal_threshold:.4f}):\n")
            f.write(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}\n")
            f.write(f"Precision: {precision_score(true_labels, predictions):.4f}\n")
            f.write(f"Recall: {recall_score(true_labels, predictions):.4f}\n")
            f.write(f"F1-Score: {f1_score(true_labels, predictions):.4f}\n\n")
            
            # Практический пример
            detected_attacks = np.sum(predictions[true_labels == 1])
            false_alarms = np.sum(predictions[true_labels == 0])
            
            f.write(f"📈 ПРАКТИЧЕСКИЕ РЕЗУЛЬТАТЫ:\n")
            f.write(f"Из {len(attack_errors)} атак обнаружено: {detected_attacks} ({detected_attacks/len(attack_errors)*100:.1f}%)\n")
            f.write(f"Ложных тревог из {len(normal_errors)}: {false_alarms} ({false_alarms/len(normal_errors)*100:.1f}%)\n")
        
        print(f"📊 Результаты тестирования автоэнкодера:")
        print(f"   ROC-AUC: {roc_results.get('auc', 0):.4f}")
        print(f"   Average Precision: {pr_results.get('average_precision', 0):.4f}")
        print(f"   Разделение классов: {np.mean(attack_errors) - np.mean(normal_errors):.4f}")
        print(f"   Результаты сохранены: {results_path}")
        
        return test_results


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="NeuroDetekt - Система обнаружения вторжений",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode", 
        choices=["train", "test", "compare", "config"],
        required=True,
        help="Режим работы"
    )
    
    parser.add_argument(
        "--model",
        choices=["lstm", "autoencoder"],
        help="Тип модели для обучения/тестирования"
    )
    
    parser.add_argument(
        "--config",
        help="Путь к файлу конфигурации"
    )
    
    parser.add_argument(
        "--checkpoint",
        help="Путь к checkpoint модели для тестирования"
    )
    
    parser.add_argument(
        "--models",
        help="Список моделей для сравнения (через запятую)"
    )
    
    parser.add_argument(
        "--name",
        default="model",
        help="Имя для сохранения модели"
    )
    
    # Параметры архитектуры модели
    parser.add_argument(
        "--hidden-size", 
        type=int, 
        help="Размер скрытого слоя (по умолчанию из конфигурации)"
    )
    
    parser.add_argument(
        "--num-layers", 
        type=int, 
        help="Количество слоев (по умолчанию из конфигурации)"
    )
    
    parser.add_argument(
        "--dropout", 
        type=float, 
        help="Dropout rate (по умолчанию из конфигурации)"
    )
    
    parser.add_argument(
        "--latent-dim", 
        type=int, 
        help="Latent dimension для автоэнкодера (по умолчанию из конфигурации)"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        help="Learning rate (по умолчанию из конфигурации)"
    )
    
    # Параметры обучения
    parser.add_argument(
        "--epochs", 
        type=int, 
        help="Количество эпох (по умолчанию из конфигурации)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        help="Размер батча (по умолчанию из конфигурации)"
    )
    
    parser.add_argument(
        "--patience", 
        type=int, 
        help="Patience для early stopping (по умолчанию из конфигурации)"
    )
    
    args = parser.parse_args()
    
    # Создаем систему
    system = NeuroDetekt(args.config)
    
    # Переопределяем параметры из командной строки
    if args.hidden_size is not None:
        system.config['lstm']['hidden_size'] = args.hidden_size
        system.config['autoencoder']['hidden_size'] = args.hidden_size
        
    if args.num_layers is not None:
        system.config['lstm']['num_layers'] = args.num_layers
        system.config['autoencoder']['num_layers'] = args.num_layers
        
    if args.dropout is not None:
        system.config['lstm']['dropout'] = args.dropout
        system.config['autoencoder']['dropout'] = args.dropout
        
    if args.latent_dim is not None:
        system.config['autoencoder']['latent_dim'] = args.latent_dim
        
    if args.learning_rate is not None:
        system.config['lstm']['learning_rate'] = args.learning_rate
        system.config['autoencoder']['learning_rate'] = args.learning_rate
        
    if args.epochs is not None:
        system.config['training']['epochs'] = args.epochs
        
    if args.batch_size is not None:
        system.config['data']['batch_size'] = args.batch_size
        
    if args.patience is not None:
        system.config['training']['patience'] = args.patience
    
    # Создаем необходимые директории
    for path in system.config['paths'].values():
        Path(path).mkdir(exist_ok=True, parents=True)
    
    # Выводим используемую конфигурацию
    if args.mode == "train":
        model_type = args.model or "unknown"
        print(f"\n⚙️ Конфигурация для {model_type.upper()}:")
        if model_type == "lstm":
            print(f"   Hidden Size: {system.config['lstm']['hidden_size']}")
            print(f"   Layers: {system.config['lstm']['num_layers']}")
            print(f"   Dropout: {system.config['lstm']['dropout']}")
            print(f"   Learning Rate: {system.config['lstm']['learning_rate']}")
        elif model_type == "autoencoder":
            print(f"   Hidden Size: {system.config['autoencoder']['hidden_size']}")
            print(f"   Layers: {system.config['autoencoder']['num_layers']}")
            print(f"   Latent Dim: {system.config['autoencoder']['latent_dim']}")
            print(f"   Dropout: {system.config['autoencoder']['dropout']}")
            print(f"   Learning Rate: {system.config['autoencoder']['learning_rate']}")
        print(f"   Epochs: {system.config['training']['epochs']}")
        print(f"   Batch Size: {system.config['data']['batch_size']}")
        print(f"   Patience: {system.config['training']['patience']}")
    
    try:
        if args.mode == "train":
            if not args.model:
                print("❌ Для обучения необходимо указать тип модели (--model)")
                sys.exit(1)
            
            if args.model == "lstm":
                results = system.train_lstm(args.name)
                print(f"✅ LSTM модель обучена и сохранена как {args.name}")
                
            elif args.model == "autoencoder":
                results = system.train_autoencoder(args.name)
                print(f"✅ Автоэнкодер обучен и сохранен как {args.name}")
        
        elif args.mode == "test":
            if not args.checkpoint:
                print("❌ Для тестирования необходимо указать путь к модели (--checkpoint)")
                sys.exit(1)
            
            if not args.model:
                print("❌ Для тестирования необходимо указать тип модели (--model)")
                sys.exit(1)
            
            results = system.test_model(args.checkpoint, args.model)
            print(f"✅ Модель протестирована")
        
        elif args.mode == "compare":
            if not args.models:
                print("❌ Для сравнения необходимо указать модели (--models)")
                sys.exit(1)
            
            results = system.compare_models(args.models)
            print(f"✅ Модели сравнены")
        
        elif args.mode == "config":
            system.create_config_template()
            print("✅ Шаблон конфигурации создан")
    
    except KeyboardInterrupt:
        print("\n⏹️ Прервано пользователем")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 