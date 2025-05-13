#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Проверка окружения для обучения модели NeuroDetekt.
    Этот скрипт проверяет наличие необходимых зависимостей и их совместимость.
"""

import sys
import os
from pathlib import Path

def print_header(title):
    """Печатает заголовок раздела проверки."""
    print("\n" + "="*50)
    print(f"  {title}")
    print("="*50)

def check_version(name, module):
    """Проверяет и выводит версию модуля."""
    try:
        version = getattr(module, "__version__", "неизвестно")
        print(f"  ✅ {name}: {version}")
        return True
    except Exception as e:
        print(f"  ❌ {name}: Ошибка: {e}")
        return False

def check_files(files):
    """Проверяет наличие необходимых файлов."""
    all_ok = True
    for file_path in files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ Файл найден: {file_path}")
        else:
            print(f"  ❌ Файл не найден: {file_path}")
            all_ok = False
    return all_ok

def check_cuda():
    """Проверяет доступность CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            devices = [torch.cuda.get_device_name(i) for i in range(device_count)]
            cuda_version = torch.version.cuda
            print(f"  ✅ CUDA доступна, версия: {cuda_version}")
            print(f"  ✅ Доступно устройств: {device_count}")
            for i, device in enumerate(devices):
                memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"    - GPU {i}: {device}, память: {memory:.2f} GB")
            return True
        else:
            print("  ⚠️ CUDA недоступна. Обучение будет выполняться на CPU (медленно).")
            return False
    except Exception as e:
        print(f"  ❌ Ошибка при проверке CUDA: {e}")
        return False

def check_numpy_pytorch_compatibility():
    """Проверяет совместимость NumPy и PyTorch."""
    try:
        import numpy as np
        import torch
        
        numpy_version = np.__version__
        pytorch_version = torch.__version__
        
        numpy_major = int(numpy_version.split('.')[0])
        
        if numpy_major >= 2:
            print(f"  ⚠️ Обнаружена версия NumPy {numpy_version}")
            print(f"     PyTorch {pytorch_version} может работать некорректно с NumPy 2.x.")
            print("     Рекомендуется использовать NumPy 1.x для работы с PyTorch.")
            print("     Выполните: conda install numpy=1.24.3")
            return False
        else:
            print(f"  ✅ NumPy {numpy_version} совместим с PyTorch {pytorch_version}")
            return True
    except ImportError as e:
        print(f"  ❌ Ошибка импорта: {e}")
        return False

def check_data_availability():
    """Проверяет наличие данных для обучения."""
    try:
        from data_processing import load_data_splits
        print("  🔍 Проверяем доступность данных...")
        train_data, test_data, test_val, atk = load_data_splits(
            "plaid", train_pct=1.0, ratio=1.0
        )
        if train_data and len(train_data) > 0:
            print(f"  ✅ Данные загружены успешно: {len(train_data)} последовательностей для обучения")
            print(f"  ✅ Тестовых последовательностей: {len(test_val)} нормальных, {len(atk)} атак")
            return True
        else:
            print("  ❌ Не удалось загрузить данные для обучения")
            return False
    except Exception as e:
        print(f"  ❌ Ошибка при проверке данных: {e}")
        return False

def main():
    """Основная функция проверки окружения."""
    print_header("Проверка окружения для NeuroDetekt")
    
    # Проверка Python
    python_version = sys.version.split()[0]
    print(f"Python: {python_version}")
    
    # Проверка необходимых модулей
    print_header("Необходимые модули")
    all_modules_ok = True
    
    try:
        import numpy as np
        all_modules_ok &= check_version("NumPy", np)
    except ImportError:
        print("  ❌ NumPy: Не установлен")
        all_modules_ok = False
    
    try:
        import torch
        all_modules_ok &= check_version("PyTorch", torch)
    except ImportError:
        print("  ❌ PyTorch: Не установлен")
        all_modules_ok = False
    
    try:
        import matplotlib
        all_modules_ok &= check_version("Matplotlib", matplotlib)
    except ImportError:
        print("  ❌ Matplotlib: Не установлен")
        all_modules_ok = False
    
    try:
        import sklearn
        all_modules_ok &= check_version("scikit-learn", sklearn)
    except ImportError:
        print("  ❌ scikit-learn: Не установлен")
        all_modules_ok = False
    
    try:
        import tqdm
        all_modules_ok &= check_version("tqdm", tqdm)
    except ImportError:
        print("  ❌ tqdm: Не установлен")
        all_modules_ok = False
    
    # Проверка CUDA
    print_header("Проверка CUDA")
    check_cuda()
    
    # Проверка совместимости NumPy и PyTorch
    print_header("Проверка совместимости")
    numpy_pytorch_ok = check_numpy_pytorch_compatibility()
    
    # Проверка наличия необходимых файлов
    print_header("Проверка необходимых файлов")
    required_files = [
        "src/data_processing.py",
        "src/models.py",
        "src/training_utils.py",
        "src/train_balanced_ensemble.py"
    ]
    files_ok = check_files(required_files)
    
    # Проверка данных
    print_header("Проверка данных")
    data_ok = check_data_availability()
    
    # Итоговое заключение
    print_header("Заключение")
    if all_modules_ok and numpy_pytorch_ok and files_ok and data_ok:
        print("  ✅ Все проверки пройдены успешно. Окружение готово для обучения.")
        print("  🚀 Вы можете запустить обучение командой:")
        print("     python src/train_balanced_ensemble.py --batch_size 96 --epochs 15 --ensemble_size 3 --balance_factor 1.5")
        return 0
    else:
        print("  ⚠️ Обнаружены проблемы, которые могут помешать обучению.")
        if not numpy_pytorch_ok:
            print("  🔧 Исправьте версию NumPy командой: conda install numpy=1.24.3")
        if not all_modules_ok:
            print("  🔧 Установите недостающие пакеты: conda env update -f environment.yml")
        if not files_ok:
            print("  🔧 Убедитесь, что все необходимые файлы присутствуют в правильных местах")
        if not data_ok:
            print("  🔧 Проверьте наличие и корректность данных")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 