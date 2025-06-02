# 🧠 NeuroDetekt - Модульная архитектура

Система обнаружения вторжений на основе нейронных сетей с четкой модульной структурой.

## 📁 Структура проекта

```
src/
├── models/                 # 🧠 Модели нейронных сетей
│   ├── __init__.py
│   ├── lstm_model.py      # LSTM модель для детекции аномалий
│   └── gru_autoencoder.py # GRU автоэнкодер
│
├── training/              # 🎯 Модуль тренировки
│   ├── __init__.py
│   ├── trainer.py         # Базовые классы тренеров
│   ├── autoencoder_trainer.py  # Тренер автоэнкодера
│   └── validator.py       # Валидация моделей
│
├── testing/               # 🧪 Модуль тестирования
│   ├── __init__.py
│   └── evaluator.py       # Оценка обученных моделей
│
├── visualization/         # 📊 Модуль визуализации
│   ├── __init__.py
│   └── plots.py          # Графики и визуализация
│
├── utils/                 # 🔧 Утилиты
│   ├── __init__.py
│   ├── data_processing.py # Обработка данных
│   ├── metrics.py         # Метрики и вычисления
│   └── helpers.py         # Вспомогательные функции
│
├── main.py               # 🚀 Главный модуль системы
└── __init__.py
```

## 🚀 Использование

### Обучение LSTM модели
```bash
python src/main.py --mode train --model lstm --name my_lstm_model
```

### Обучение автоэнкодера
```bash
python src/main.py --mode train --model autoencoder --name my_autoencoder
```

### Тестирование модели
```bash
python src/main.py --mode test --model lstm --checkpoint models/my_lstm_model_best.pth
```

### Сравнение моделей
```bash
python src/main.py --mode compare --models lstm:models/lstm.pth,autoencoder:models/ae.pth
```

### Создание конфигурации
```bash
python src/main.py --mode config
```

## 📋 Конфигурация

Система поддерживает YAML конфигурацию для настройки параметров:

```yaml
data:
  dataset: plaid
  batch_size: 64
  sequence_length: 100

lstm:
  vocab_size: 229
  embedding_dim: 128
  hidden_size: 128
  num_layers: 2
  dropout: 0.25
  learning_rate: 0.0001

autoencoder:
  vocab_size: 229
  embedding_dim: 128
  hidden_size: 128
  num_layers: 2
  dropout: 0.2
  learning_rate: 0.00005

training:
  epochs: 15
  patience: 8
  gradient_clip: 1.0

paths:
  data_dir: data
  models_dir: models
  logs_dir: logs
  plots_dir: plots
```

## 🧩 Модули

### 🧠 Models
- **LSTMModel**: LSTM архитектура для детекции аномалий
- **GRUAutoencoder**: Автоэнкодер на основе GRU

### 🎯 Training
- **Trainer**: Базовый класс для обучения
- **LSTMTrainer**: Специализированный тренер для LSTM
- **AutoencoderTrainer**: Тренер для автоэнкодеров

### 🧪 Testing
- **Evaluator**: Базовый класс для оценки
- **LSTMEvaluator**: Оценка LSTM моделей
- **AutoencoderEvaluator**: Оценка автоэнкодеров

### 📊 Visualization
- **TrainingPlotter**: Графики процесса обучения
- **ResultsPlotter**: Визуализация результатов
- **ComparisonPlotter**: Сравнение моделей

### 🔧 Utils
- **data_processing**: Загрузка и обработка данных
- **metrics**: Вычисление метрик качества
- **helpers**: Вспомогательные классы и функции

## 🎯 Преимущества новой архитектуры

1. **Модульность**: Каждый компонент в отдельном модуле
2. **Переиспользование**: Легко использовать компоненты в других проектах
3. **Расширяемость**: Простое добавление новых моделей и функций
4. **Читаемость**: Четкая структура и документация
5. **Тестируемость**: Каждый модуль можно тестировать отдельно
6. **Единый интерфейс**: Все функции доступны через main.py

## 📈 Результаты

Система автоматически создает:
- 📊 Графики обучения
- 📈 Метрики качества
- 🎯 Сравнительные анализы
- 💾 Сохраненные модели
- 📄 Логи обучения
 