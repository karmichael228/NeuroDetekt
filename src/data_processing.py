#!/usr/bin/env python3
# -*- coding: utf-8 -*

"""Computes and loads data splits for PLAID

    Provides computing and loading of data splits used in training and evaluation see below for main external use cases.
    Operates on pre-processed data sets from plaid_preprocessing.py.
        - get_data : Provides data splits for training and evaluation at the trace level
        - load_nested_test : Provides test set for evaluation at the application level

"""

from itertools import chain
from pathlib import Path
from typing import Hashable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence


def load_files(data_set, nested=False):
    """Loads requested system call data set from disk

    Parameters
    ----------
    data_set : {"plaid"}
        The data set to be returned.
    nested : bool
        Return attack sequences nested by application. Default False returns a flat list.

    Returns
    -------
    attack_sequences : List[List[str]] or List[List[List[str]]]
        List of attack system call sequences. When nested=False each element is an attack sequence represented as a list
        of strings. If nested=True each element is a list of all attack sequences belonging to a single application.
    base_sequences : List[List[str]]
        List of baseline system call sequences.

    """
    if data_set != "plaid":
        raise ValueError("data_set must be plaid")

    def get_seq(files):
        ret = []
        for f in files:
            with open(f) as file:
                seq = file.read().strip().split(" ")
                if 4495 >= len(seq) >= 8:
                    ret.append(seq)
        return ret

    attack_files = sorted(list(Path("data/PLAID/attack").rglob("*.txt")))
    print(f"Найдено файлов атак: {len(attack_files)}")
    
    baseline_files = list(Path("data/PLAID/baseline").rglob("*.txt"))
    print(f"Найдено базовых файлов: {len(baseline_files)}")

    if nested:
        attack_sequences = []
        folders = set([x.parent for x in attack_files])
        for folder in folders:
            tmp = [x for x in attack_files if x.parent == folder]
            attack_sequences.append(get_seq(tmp))
    else:
        attack_sequences = get_seq(attack_files)
    base_sequences = get_seq(baseline_files)
    
    print(f"Распознано последовательностей атак: {len(attack_sequences)}")
    print(f"Распознано базовых последовательностей: {len(base_sequences)}")
    
    return attack_sequences, base_sequences


class Encoder:
    """Converts data to a dense integer encoding

    Attributes:
        file_path: location to save/load syscall map
        syscall_map: mapping from item to encoded value
    """

    file_path = Path()
    syscall_map: dict = dict()

    def __init__(self, file_path: str) -> None:
        self.file_path = Path(file_path)
        if self.file_path.exists():
            self.syscall_map = np.load(self.file_path, allow_pickle=True).item()
            print(f"Загружен энкодер с {len(self.syscall_map)} системных вызовов")

    def encode(self, syscall: Hashable) -> int:
        """Encodes an individual item

        Unique items are sequentially encoded (ie first item -> 0 next unique item -> 1). The mapping dict is updated
        with new encodings as necessary and immediately written to disk.

        Args:
            syscall: item to encode

        Returns:
            integer encoding of syscall
        """
        if syscall in self.syscall_map:
            return self.syscall_map[syscall]
        syscall_enc = len(self.syscall_map) + 1
        self.syscall_map[syscall] = syscall_enc
        np.save(self.file_path, self.syscall_map)

        return syscall_enc


class SequenceDataset(Dataset):
    """PyTorch Dataset для последовательностей системных вызовов"""
    
    def __init__(self, sequences):
        if not sequences:
            print("Внимание: пустой список последовательностей!")
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)


class SequencePairDataset(Dataset):
    """PyTorch Dataset для пар последовательностей (вход-выход)"""
    
    def __init__(self, sequences):
        if not sequences:
            print("Внимание: пустой список последовательностей!")
        self.sequences = sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) < 2:
            # Если последовательность слишком короткая, дублируем элемент
            print(f"Внимание: короткая последовательность {len(seq)} в индексе {idx}")
            if len(seq) == 0:
                # Создаем последовательность из двух нулей, если пустая
                seq = np.array([0, 0], dtype=np.float32)
            else:
                # Дублируем единственный элемент
                seq = np.array([seq[0], seq[0]], dtype=np.float32)
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)


def collate_fn(batch):
    """Функция для батчирования последовательностей разной длины"""
    # Разделяем входные и выходные последовательности
    inputs, targets = zip(*[(x, y) for x, y in batch])
    
    # Паддинг последовательностей
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return inputs_padded, targets_padded


def test_collate_fn(batch):
    """Функция для батчирования тестовых последовательностей"""
    # Паддинг последовательностей
    seqs_padded = pad_sequence(batch, batch_first=True, padding_value=0)
    return seqs_padded


def load_data_splits(data_set, train_pct=1.0, ratio=1.0):
    """[Internal] Lazy-loads data splits for training and evaluation

    Internal function for processing and loading training and evaluation splits for specified data set.
    For external use get_data.
    Calls with the same data set and ratio will use the same split.
    Training percentage affects only existing training split (repeat calls will not cause data leakage).

    Parameters
    ----------
    data_set : {"plaid"}
        Data set to be loaded and split.
    train_pct : float (0, 1]
        Percentage of training data to be returned
    ratio : float
        Ratio of baseline to attack sequences in the testing split

    Returns
    -------
        train : List
            Data set for training seq-seq system call language model. Consists of only baseline sequences.
        val : List
            Empty list to maintain compatibility with existing code.
        test_val : List
            Baseline system call sequences for use in final model evaluation.
        atk : List
            Attack system call sequences for use in final model evaluation.

    """
    train_split = 0.8  # 80% нормальных данных для обучения, 20% - для тестирования
    if data_set != "plaid":
        raise ValueError("data_set must be plaid")

    if ratio != 1:
        out_path = Path(f"out/{data_set}_split80_20_{ratio}.npz")
    else:
        out_path = Path(f"out/{data_set}_split80_20.npz")

    print(f"Проверка наличия файла {out_path}")
    if out_path.exists():
        print(f"Загрузка предварительно сохраненных данных из {out_path}")
        data = np.load(out_path, allow_pickle=True)
        train, val, test_val, atk = data["arr_0"]
        print(f"Загружено: train={len(train)}, test_val={len(test_val)}, atk={len(atk)}")
    else:
        print(f"Создание новых данных, так как {out_path} не существует")
        out_path.parent.mkdir(exist_ok=True, parents=True)
        encoder = Encoder(f"data/{data_set}_encoder.npy")

        atk_files, normal_files = load_files(data_set)

        # Определяем размер тренировочного набора
        train_size = int(len(normal_files) * train_split)
        
        # Перемешиваем нормальные данные
        normal_idxs = np.arange(len(normal_files))
        np.random.shuffle(normal_idxs)
        
        # Разделяем на обучающую и тестовую выборки
        train_files = []
        for idx in normal_idxs[:train_size]:
            train_files.append(normal_files[idx])
            
        test_val_files = []
        for idx in normal_idxs[train_size:]:
            test_val_files.append(normal_files[idx])

        print(f"Разделение: train={len(train_files)}, test_val={len(test_val_files)}, atk={len(atk_files)}")

        vec_encode = np.vectorize(encoder.encode)
        train = [vec_encode(row).astype(np.float32) for row in train_files]
        val = []  # Пустой валидационный набор для совместимости
        atk = [vec_encode(row).astype(np.float32) for row in atk_files]
        test_val = [vec_encode(row).astype(np.float32) for row in test_val_files]

        # Сохраняем данные в правильном формате
        print(f"Сохранение данных в {out_path}")
        data_to_save = np.array([train, val, test_val, atk], dtype=object)
        np.savez_compressed(out_path, arr_0=data_to_save)

    if train_pct < 1:
        train_idxs = np.arange(int(len(train) * train_pct))
        np.random.shuffle(train_idxs)
        tmp = []
        for idx in train_idxs:
            tmp.append(train[idx])
        train = tmp

    # Проверка на пустые данные
    if len(train) == 0:
        # Создаем фиктивные данные для обучения, если не найдены реальные
        print("Внимание: создаются фиктивные данные для обучения, так как обучающий набор пуст!")
        train = [np.array([1, 2, 3, 4, 5], dtype=np.float32) for _ in range(10)]
    
    if len(test_val) == 0:
        # Создаем фиктивные данные для тестирования
        print("Внимание: создаются фиктивные данные для тестирования!")
        test_val = [np.array([1, 2, 3, 4, 5], dtype=np.float32) for _ in range(5)]
        
    if len(atk) == 0:
        # Создаем фиктивные данные для атак
        print("Внимание: создаются фиктивные данные для атак!")
        atk = [np.array([5, 4, 3, 2, 1], dtype=np.float32) for _ in range(5)]

    return train, val, test_val, atk


def get_data(data_set, batch_size=64, train_pct=1.0, ratio=1.0, num_workers=4):
    """Lazy-loads data splits for training and evaluation

    Converts load_data_splits outputs into ready to go data structures.

    Parameters
    ----------
    data_set : {"plaid"}
        Data set to be loaded and split.
    batch_size : int
        Batch size for data splits
    train_pct : float (0, 1]
        Percentage of training data to be returned
    ratio : float
        Ratio of baseline to attack sequences in the testing split
    num_workers : int
        Number of worker processes for data loading

    Returns
    -------
        train_loader : DataLoader
            Data loader for training dataset.
        val_loader : DataLoader
            Empty data loader (для совместимости с существующим кодом).
        (test_loader, test_labels): (DataLoader, torch.Tensor)
            Data loader for test dataset and corresponding labels.

    """
    if data_set != "plaid":
        raise ValueError("data_set must be plaid")

    train, val, test_val, atk = load_data_splits(
        data_set, train_pct=train_pct, ratio=ratio
    )

    # Создаем датасеты для PyTorch
    train_dataset = SequencePairDataset(train)
    
    print(f"Создано датасетов: train={len(train_dataset)}")
    
    # Используем меньшее количество рабочих процессов для небольших данных
    actual_workers = min(num_workers, 1) 
    
    # Создаем DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=min(batch_size, len(train_dataset)), 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=actual_workers,
        pin_memory=True
    )
    
    # Создаем пустой валидационный загрузчик для совместимости
    empty_dataset = SequencePairDataset([])
    val_loader = DataLoader(
        empty_dataset, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=1
    )
    
    # Создаем тестовый датасет
    test_sequences = test_val + atk
    test_dataset = SequenceDataset(test_sequences)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=min(batch_size, len(test_dataset)), 
        shuffle=False, 
        collate_fn=test_collate_fn,
        num_workers=actual_workers,
        pin_memory=True
    )
    
    # Создаем метки для тестового набора
    test_labels = torch.zeros(len(test_val) + len(atk))
    test_labels[len(test_val):] = 1
    
    return train_loader, val_loader, (test_loader, test_labels)


def load_nested_test(data_set):
    """Loads nested version of the testing set

    Loads testing set nested by application. Baseline sequences are randomly assigned to an application such that there
    is an equal number of test and attack applications with a given number of traces.

    Parameters
    ----------
    data_set : {"plaid"}
        Data set to load test split of

    Returns
    -------
        test_val_new : List[List[List[int]]]
            Baseline sequences nested by application
        atk_new : List[List[List[int]]]
            Attack sequences nested by application

    """
    test_val = load_data_splits(data_set)[2]

    # attack ordering is not guaranteed so we need to redo it
    if data_set == "plaid":
        attack_raw = load_files(data_set, nested=True)[0]
    else:
        raise ValueError("data_set must be plaid")

    encoder = Encoder(f"data/{data_set}_encoder.npy")
    vec_encode = np.vectorize(encoder.encode)

    attack_lens = [len(x) for x in attack_raw]
    attack_raw = list(chain(*attack_raw))

    idx = 0
    test_val_new = []
    atk_new = []
    for folder_len in attack_lens:
        subdir_val = []
        subdir_atk = []
        for _ in range(folder_len):
            subdir_val.append(test_val[idx])
            subdir_atk.append(vec_encode(attack_raw[idx]))
            idx += 1
        test_val_new.append(subdir_val)
        atk_new.append(subdir_atk)

    return test_val_new, atk_new


if __name__ == "__main__":
    plaid_data = load_data_splits("plaid")
    print(len(plaid_data[0]))
    for elm in plaid_data:
        print(len(elm))
    enc = np.load("data/plaid_encoder.npy", allow_pickle=True).item()
    print("PLAID vocab size ", len(enc) + 1)
