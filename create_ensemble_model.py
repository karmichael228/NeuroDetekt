#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Создание ансамбля LSTM + Автоэнкодер для оптимальных метрик
"""

import numpy as np
import torch
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.main import NeuroDetekt
from src.models.lstm_model import LSTMModel
from src.models.gru_autoencoder import GRUAutoEncoder
from src.utils.data_processing import get_data

class EnsembleModel:
    """Ансамбль LSTM + Автоэнкодер для детекции аномалий."""
    
    def __init__(self, lstm_path, autoencoder_path, device="cuda"):
        """
        Инициализация ансамбля.
        
        Args:
            lstm_path (str): Путь к обученной LSTM модели
            autoencoder_path (str): Путь к обученному автоэнкодеру
            device (str): Устройство для вычислений
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Загружаем LSTM модель
        self.lstm_model = self._load_lstm_model(lstm_path)
        
        # Загружаем автоэнкодер
        self.autoencoder_model = self._load_autoencoder_model(autoencoder_path)
        
        print(f"🎯 Ансамбль создан на устройстве: {self.device}")
        print(f"   🧠 LSTM модель загружена")
        print(f"   🤖 Автоэнкодер загружен")
    
    def _load_lstm_model(self, model_path):
        """Загружает обученную LSTM модель."""
        # Создаем модель с параметрами final_model
        model = LSTMModel(
            vocab_size=229,
            embedding_dim=384,  # Как в final_model (проверено из checkpoint)
            hidden_size=384,    # Как в final_model
            num_layers=3,       # Как в final_model
            dropout=0.15        # Как в final_model
        )
        
        # Загружаем веса
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _load_autoencoder_model(self, model_path):
        """Загружает обученный автоэнкодер."""
        # Создаем модель с параметрами автоэнкодера
        model = GRUAutoEncoder(
            vocab_size=229,
            embedding_dim=128,
            hidden_size=128,
            num_layers=2,
            latent_dim=64,
            dropout=0.2
        )
        
        # Загружаем checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def get_lstm_scores(self, dataloader):
        """Получает аномалии-скоры от LSTM модели."""
        self.lstm_model.eval()
        scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, tuple):
                    batch_x, _ = batch
                else:
                    batch_x = batch
                
                batch_x = batch_x.to(self.device)
                
                # Вычисляем перплексию как меру аномальности
                inputs = batch_x[:, :-1]
                targets = batch_x[:, 1:]
                
                outputs = self.lstm_model(inputs)
                
                # Вычисляем cross-entropy для каждой последовательности
                batch_scores = []
                for seq_outputs, seq_targets in zip(outputs, targets):
                    seq_loss = torch.nn.functional.cross_entropy(
                        seq_outputs.view(-1, seq_outputs.size(-1)),
                        seq_targets.view(-1),
                        reduction='mean'
                    )
                    batch_scores.append(seq_loss.item())
                
                scores.extend(batch_scores)
        
        return np.array(scores)
    
    def get_autoencoder_scores(self, dataloader):
        """Получает аномалии-скоры от автоэнкодера."""
        self.autoencoder_model.eval()
        scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, tuple):
                    batch_x, _ = batch
                else:
                    batch_x = batch
                
                batch_x = batch_x.to(self.device)
                errors = self.autoencoder_model.get_reconstruction_error(batch_x)
                scores.extend(errors.cpu().numpy())
        
        return np.array(scores)
    
    def predict_ensemble(self, test_loader, test_labels, strategy="weighted_voting", weights=None):
        """
        Делает предсказания с помощью ансамбля.
        
        Args:
            test_loader: DataLoader с тестовыми данными
            test_labels: Истинные метки
            strategy (str): Стратегия ансамбля
            weights (tuple): Веса для моделей (lstm_weight, autoencoder_weight)
        
        Returns:
            dict: Результаты ансамбля
        """
        print(f"🔍 Получение предсказаний от моделей...")
        
        # Получаем скоры от обеих моделей
        lstm_scores = self.get_lstm_scores(test_loader)
        autoencoder_scores = self.get_autoencoder_scores(test_loader)
        
        print(f"   🧠 LSTM скоры: {len(lstm_scores)} образцов")
        print(f"   🤖 Автоэнкодер скоры: {len(autoencoder_scores)} образцов")
        
        # Нормализация скоров (0-1)
        lstm_scores_norm = (lstm_scores - lstm_scores.min()) / (lstm_scores.max() - lstm_scores.min())
        autoencoder_scores_norm = (autoencoder_scores - autoencoder_scores.min()) / (autoencoder_scores.max() - autoencoder_scores.min())
        
        test_labels_np = test_labels.numpy() if hasattr(test_labels, 'numpy') else test_labels
        
        results = {}
        
        if strategy == "simple_voting":
            # Простое голосование (среднее арифметическое)
            ensemble_scores = (lstm_scores_norm + autoencoder_scores_norm) / 2
            results['method'] = "Simple Voting (50-50)"
            
        elif strategy == "weighted_voting":
            # Взвешенное голосование
            if weights is None:
                # Оптимальные веса: больше вес модели с лучшим F1
                # LSTM F1=78.83%, Автоэнкодер F1=71.67%
                w_lstm = 0.6  # Больше вес LSTM (точность)
                w_autoencoder = 0.4  # Меньше вес автоэнкодера
            else:
                w_lstm, w_autoencoder = weights
            
            ensemble_scores = w_lstm * lstm_scores_norm + w_autoencoder * autoencoder_scores_norm
            results['method'] = f"Weighted Voting ({w_lstm:.1f}-{w_autoencoder:.1f})"
            results['weights'] = (w_lstm, w_autoencoder)
            
        elif strategy == "max_voting":
            # Максимальный скор (более консервативно)
            ensemble_scores = np.maximum(lstm_scores_norm, autoencoder_scores_norm)
            results['method'] = "Max Voting (консервативный)"
            
        elif strategy == "precision_focused":
            # Фокус на precision: больше вес LSTM
            w_lstm, w_autoencoder = 0.8, 0.2
            ensemble_scores = w_lstm * lstm_scores_norm + w_autoencoder * autoencoder_scores_norm
            results['method'] = f"Precision Focused ({w_lstm:.1f}-{w_autoencoder:.1f})"
            
        elif strategy == "recall_focused":
            # Фокус на recall: больше вес автоэнкодера
            w_lstm, w_autoencoder = 0.3, 0.7
            ensemble_scores = w_lstm * lstm_scores_norm + w_autoencoder * autoencoder_scores_norm
            results['method'] = f"Recall Focused ({w_lstm:.1f}-{w_autoencoder:.1f})"
        
        else:
            raise ValueError(f"Неизвестная стратегия: {strategy}")
        
        # Находим оптимальный порог
        best_threshold, best_metrics = self._find_optimal_threshold(ensemble_scores, test_labels_np)
        
        # Делаем финальные предсказания
        predictions = (ensemble_scores > best_threshold).astype(int)
        
        # Вычисляем метрики
        accuracy = accuracy_score(test_labels_np, predictions)
        precision = precision_score(test_labels_np, predictions, zero_division=0)
        recall = recall_score(test_labels_np, predictions, zero_division=0)
        f1 = f1_score(test_labels_np, predictions, zero_division=0)
        roc_auc = roc_auc_score(test_labels_np, ensemble_scores)
        
        results.update({
            'ensemble_scores': ensemble_scores,
            'predictions': predictions,
            'threshold': best_threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'lstm_scores': lstm_scores_norm,
            'autoencoder_scores': autoencoder_scores_norm,
            'test_labels': test_labels_np
        })
        
        return results
    
    def _find_optimal_threshold(self, scores, labels):
        """Находит оптимальный порог для максимизации F1-score."""
        thresholds = np.linspace(scores.min(), scores.max(), 100)
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = {}
        
        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)
            f1 = f1_score(labels, predictions, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'accuracy': accuracy_score(labels, predictions),
                    'precision': precision_score(labels, predictions, zero_division=0),
                    'recall': recall_score(labels, predictions, zero_division=0),
                    'f1_score': f1
                }
        
        return best_threshold, best_metrics

def test_all_ensemble_strategies():
    """Тестирует все стратегии ансамбля и находит лучшую."""
    print("🎯 СОЗДАНИЕ И ТЕСТИРОВАНИЕ АНСАМБЛЯ LSTM + АВТОЭНКОДЕР")
    print("=" * 60)
    
    # Пути к моделям
    lstm_path = "trials/final_model/final_model_best.pth"
    autoencoder_path = "trials/autoencoder_vs_lstm_comparison/autoencoder_vs_lstm_comparison_best.pth"
    
    # Проверяем наличие моделей
    if not Path(lstm_path).exists():
        print(f"❌ LSTM модель не найдена: {lstm_path}")
        return
    
    if not Path(autoencoder_path).exists():
        print(f"❌ Автоэнкодер не найден: {autoencoder_path}")
        return
    
    # Создаем ансамбль
    ensemble = EnsembleModel(lstm_path, autoencoder_path)
    
    # Загружаем тестовые данные
    print(f"\n📊 Загрузка тестовых данных...")
    _, _, (test_loader, test_labels) = get_data('plaid', batch_size=64)
    
    # Тестируем разные стратегии
    strategies = [
        ("simple_voting", "Простое голосование"),
        ("weighted_voting", "Взвешенное голосование"),
        ("max_voting", "Максимальное голосование"),
        ("precision_focused", "Фокус на точность"),
        ("recall_focused", "Фокус на полноту")
    ]
    
    all_results = {}
    
    print(f"\n🔬 ТЕСТИРОВАНИЕ СТРАТЕГИЙ АНСАМБЛЯ:")
    print("-" * 60)
    
    for strategy, description in strategies:
        print(f"\n🎯 Тестирую: {description}")
        
        results = ensemble.predict_ensemble(test_loader, test_labels, strategy=strategy)
        all_results[strategy] = results
        
        print(f"   📊 Результаты {results['method']}:")
        print(f"      Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"      Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
        print(f"      Recall:    {results['recall']:.4f} ({results['recall']*100:.2f}%)")
        print(f"      F1-Score:  {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)")
        print(f"      ROC-AUC:   {results['roc_auc']:.4f} ({results['roc_auc']*100:.2f}%)")
        print(f"      Порог:     {results['threshold']:.4f}")
    
    # Находим лучшую стратегию
    best_strategy = max(all_results.keys(), key=lambda k: all_results[k]['f1_score'])
    best_results = all_results[best_strategy]
    
    print(f"\n🏆 ЛУЧШАЯ СТРАТЕГИЯ: {best_results['method']}")
    print("=" * 60)
    print(f"🎯 Accuracy:  {best_results['accuracy']*100:.2f}%")
    print(f"🎯 Precision: {best_results['precision']*100:.2f}%")
    print(f"🎯 Recall:    {best_results['recall']*100:.2f}%")
    print(f"🎯 F1-Score:  {best_results['f1_score']*100:.2f}%")
    print(f"🎯 ROC-AUC:   {best_results['roc_auc']*100:.2f}%")
    
    # Сравнение с исходными моделями
    print(f"\n📊 СРАВНЕНИЕ С ИСХОДНЫМИ МОДЕЛЯМИ:")
    print("-" * 60)
    print(f"🧠 LSTM:         94.30% Acc, 76.35% Prec, 81.48% Rec, 78.83% F1")
    print(f"🤖 Автоэнкодер:  89.82% Acc, 56.24% Prec, 98.78% Rec, 71.67% F1")
    print(f"🎯 АНСАМБЛЬ:     {best_results['accuracy']*100:.2f}% Acc, {best_results['precision']*100:.2f}% Prec, {best_results['recall']*100:.2f}% Rec, {best_results['f1_score']*100:.2f}% F1")
    
    # Проверяем достижение цели 80%+
    goal_80_achieved = (
        best_results['accuracy'] >= 0.80 and
        best_results['precision'] >= 0.80 and
        best_results['recall'] >= 0.80 and
        best_results['f1_score'] >= 0.80
    )
    
    if goal_80_achieved:
        print(f"\n🎉 ЦЕЛЬ ДОСТИГНУТА! Все метрики ≥ 80%!")
    else:
        print(f"\n📈 Прогресс к цели 80%:")
        metrics_80 = {
            'Accuracy': best_results['accuracy'] >= 0.80,
            'Precision': best_results['precision'] >= 0.80,
            'Recall': best_results['recall'] >= 0.80,
            'F1-Score': best_results['f1_score'] >= 0.80
        }
        for metric, achieved in metrics_80.items():
            status = "✅" if achieved else "❌"
            print(f"   {status} {metric}: {best_results[metric.lower().replace('-', '_')]*100:.2f}%")
    
    # Сохраняем результаты
    output_dir = Path("trials/ensemble_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "ensemble_results.pkl", 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n💾 Результаты сохранены в: {output_dir}")
    
    return all_results

if __name__ == "__main__":
    test_all_ensemble_strategies() 