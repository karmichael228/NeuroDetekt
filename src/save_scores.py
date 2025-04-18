#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""
    Инструменты для оценки ранее обученных моделей и сохранения результатов на диск.
    Часть системы NeuroDetekt для обнаружения вторжений.
"""

import argparse
import time
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_processing import get_data, load_nested_test, SequenceDataset, test_collate_fn
from models import LSTMModel


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"Evaluate trained models and save scores.",
    )
    parser.add_argument(
        "--data_set",
        default="plaid",
        choices=["plaid"],
        help="Data set to evaluate on.",
    )
    parser.add_argument(
        "--path",
        help="Location of model checkpoint.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run evaluation on.",
    )

    return parser


def get_scores(model, dataloader, device, nll=False):
    """Calculates the probability of each sequence occurring.

    Parameters
    ----------
    model : PyTorch model
        Model to get probabilities from.
    dataloader : DataLoader
        Data loader containing sequences to score.
    device : str
        Device to run model on ('cuda' or 'cpu').
    nll : bool
        If True, return negative log likelihood.

    Returns
    -------
    Array of scores.
    """
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # Получаем предсказания модели
            outputs = model(batch)
            
            # Создаем маску для игнорирования padding
            non_zero_mask = (batch != 0)
            
            # Для каждой последовательности в батче
            for i in range(len(batch)):
                sequence = batch[i]
                output = outputs[i]
                
                # Находим индексы ненулевых элементов (не padding)
                non_zero_indices = torch.nonzero(sequence).squeeze().cpu().numpy()
                
                if len(non_zero_indices) == 0:
                    continue
                
                # Получаем максимальные вероятности для каждого элемента последовательности
                probs = torch.exp(output[non_zero_indices[:-1], sequence[non_zero_indices[1:]]])
                
                # Вычисляем общую вероятность последовательности как произведение отдельных вероятностей
                sequence_prob = probs.prod().item()
                all_probs.append(sequence_prob)
    
    probs_array = np.array(all_probs)
    
    if nll:
        return np.clip(-np.log2(probs_array), a_min=0, a_max=1e100)
    else:
        return probs_array


def save_scores(path, data_set, batch_size=32, device='cuda'):
    """Save scores for a given model to disk

    Parameters
    ----------
    path : str
        Path to model checkpoint
    data_set : str
        Dataset to evaluate on.
    batch_size : int
        Batch size for evaluation.
    device : str
        Device to run evaluation on ('cuda' or 'cpu').
    """
    # Проверяем доступность CUDA
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA недоступен, используем CPU")
        device = 'cpu'
    
    path = Path(path)
    val, attack = load_nested_test(data_set)
    train_loader, _, _ = get_data(data_set, batch_size=batch_size)
    
    attack = list(chain(*attack))
    val = list(chain(*val))
    
    # Создаем датасет и DataLoader для тестовых данных
    test_dataset = SequenceDataset(val + attack)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=test_collate_fn,
        num_workers=4,
        pin_memory=(device == 'cuda')
    )
    
    tokens = path.stem.split("_")
    new_path = path.parent / f"eval_{tokens[1]}_{tokens[2]}"
    
    if not Path(str(new_path) + ".npz").exists():
        # Загружаем модель
        try:
            # Сначала пробуем загрузить как state_dict
            model_state = torch.load(path, map_location=device)
            # Определяем параметры модели из имени файла или из структуры состояния
            vocab_size = 229  # Для PLAID датасета
            
            # Создаем модель и загружаем параметры
            model = LSTMModel(vocab_size=vocab_size).to(device)
            
            # Проверяем, что загружено - полная модель или state_dict
            if isinstance(model_state, LSTMModel):
                model = model_state
            else:
                model.load_state_dict(model_state)
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            print("Пробуем загрузить как полную модель...")
            model = torch.load(path, map_location=device)
        
        # Получаем оценки
        t0 = time.time()
        scores = get_scores(model, test_loader, device, nll=True)
        t1 = time.time()
        
        # Вычисляем базовую оценку на тренировочных данных
        # Используем первый батч для оценки базовой линии
        print("Вычисляем базовую оценку на тренировочных данных...")
        first_batch = next(iter(train_loader))
        baseline_scores = []
        with torch.no_grad():
            inputs, targets = first_batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Для каждой последовательности в батче
            for i in range(len(inputs)):
                output = outputs[i].view(-1, outputs.size(-1))
                target = targets[i].view(-1).to(device)
                
                # Находим индексы ненулевых элементов (не padding)
                non_zero_mask = (target != 0)
                if non_zero_mask.sum() > 0:
                    output = output[non_zero_mask]
                    target = target[non_zero_mask]
                    
                    loss = torch.nn.functional.nll_loss(output, target, reduction='mean')
                    baseline_scores.append(loss.item())
        
        baseline = np.median(baseline_scores) if baseline_scores else 0.5
        
        # Сохраняем результаты
        results = {
            'scores': scores,
            'baseline': baseline,
            'time': t1 - t0
        }
        np.savez_compressed(new_path, **results)
        print(f"Оценки сохранены в {new_path}, время выполнения: {t1 - t0:.2f}s")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    save_scores(args.path, args.data_set, args.batch_size, args.device)
