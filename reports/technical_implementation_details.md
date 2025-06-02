# ДЕТАЛЬНАЯ ТЕХНИЧЕСКАЯ РЕАЛИЗАЦИЯ СИСТЕМЫ НЕЙРОДЕТЕКТ

## 🔧 ТЕХНОЛОГИЧЕСКИЙ СТЕК

### Базовые технологии
- **Python 3.8+** - основной язык программирования
- **PyTorch 1.12.1** - фреймворк глубокого обучения с GPU поддержкой
- **CUDA 11.6** - ускорение вычислений на GPU
- **NumPy 1.24.3** - численные вычисления и массивы
- **Pandas 1.5.3** - обработка структурированных данных
- **Scikit-learn 1.2.2** - машинное обучение и метрики
- **Matplotlib 3.7.1** - базовая визуализация
- **Seaborn 0.12.2** - статистическая визуализация
- **PyYAML 6.0** - конфигурационные файлы
- **Pickle** - сериализация объектов Python

---

## 📊 ЭТАП 1: ПОДГОТОВКА И ПРЕДОБРАБОТКА ДАННЫХ

### 1.1 Архитектура модуля обработки данных

```python
# Иерархия классов для обработки данных
src/utils/data_processing.py
├── load_files()           # Загрузка файлов PLAID
├── Encoder класс          # Кодирование системных вызовов  
├── SequenceDataset()      # PyTorch Dataset для последовательностей
├── SequencePairDataset()  # Dataset для пар вход-выход
├── load_data_splits()     # Разделение данных 60/20/20
└── get_data()             # Создание DataLoader'ов
```

### 1.2 Технологии загрузки данных

**Алгоритм загрузки файлов:**
```python
def load_files(data_set, nested=False):
    # 1. Рекурсивный поиск файлов в директориях
    attack_files = sorted(list(Path("data/PLAID/attack").rglob("*.txt")))
    baseline_files = list(Path("data/PLAID/baseline").rglob("*.txt"))
    
    # 2. Фильтрация по длине последовательностей (8-4495)
    if 4495 >= len(seq) >= 8:
        ret.append(seq)
    
    # 3. Возврат сырых последовательностей системных вызовов
```

**Статистика загруженных данных:**
- Найдено файлов атак: 1,145
- Найдено базовых файлов: 38,178  
- Общий объем: 39,323 последовательности
- Фильтрация: только последовательности длиной 8-4,495 вызовов

### 1.3 Система кодирования последовательностей

**Класс Encoder - технология преобразования:**
```python
class Encoder:
    # Создание словаря: строка -> число
    # Каждый уникальный системный вызов получает ID
    # Специальные токены: PAD=0, UNK для неизвестных
    
    def encode(self, syscall_name):
        return self.vocab.get(syscall_name, self.unk_id)
```

**Преимущества подхода:**
- Детерминированное кодирование
- Поддержка неизвестных вызовов  
- Эффективное хранение (int32 vs string)
- Совместимость с PyTorch tensors

### 1.4 Стратегия разделения данных

**Технология разделения 60/20/20:**
```python
train_split = 0.6   # 60% нормальных для обучения
val_split = 0.2     # 20% нормальных для валидации  
test_split = 0.2    # 20% нормальных + ВСЕ атаки для тестирования

# Критически важно: рандомное перемешивание
normal_idxs = np.arange(len(normal_files))
np.random.shuffle(normal_idxs)
```

**Обоснование архитектуры:**
- Автоэнкодер обучается ТОЛЬКО на нормальных данных
- Валидация предотвращает переобучение
- Тестовая выборка включает все типы атак
- Стратифицированное разделение сохраняет пропорции

### 1.5 PyTorch DataLoader архитектура

**Технология батчирования последовательностей:**
```python
def collate_fn(batch):
    # Проблема: последовательности разной длины
    # Решение: Dynamic padding до максимальной длины в батче
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
```

**Оптимизации производительности:**
- `num_workers=0` - отключение multiprocessing (стабильность)
- `pin_memory=False` - экономия GPU памяти
- `batch_first=True` - совместимость с PyTorch LSTM
- Dynamic batching - оптимальное использование памяти

### 1.6 Кэширование и персистентность

**Двухуровневая система сохранения:**
```python
# Уровень 1: Pickle (быстро, Python-specific)
pickle_path = Path(f"out/{data_set}_split60_20_20.pkl")
save_pickle([train, val, test_val, atk], pickle_path)

# Уровень 2: NumPy (совместимость, компактно)  
np.savez(out_path, data_to_save)
```

**Преимущества архитектуры:**
- Мгновенная загрузка предобработанных данных
- Воспроизводимость экспериментов
- Экономия времени на повторных запусках
- Резервное копирование в двух форматах

---

## 🧠 ЭТАП 2: АРХИТЕКТУРЫ НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ

### 2.1 GRU-Автоэнкодер: техническая реализация

**Архитектурная схема:**
```
Input → Embedding(151→64) → GRU(64→128, 2layers) → Hidden State
Hidden State → GRU(128→64, 2layers) → Linear(128→151) → Output
```

**Технические параметры:**
```python
class GRUAutoEncoder(nn.Module):
    vocab_size = 151        # Уникальные системные вызовы
    embedding_dim = 64      # Размерность эмбеддингов
    hidden_dim = 128        # Скрытые состояния GRU
    num_layers = 2          # Глубина сети
    dropout = 0.2           # Регуляризация
```

**Технология детекции аномалий:**
```python
def get_reconstruction_error(self, x):
    # 1. Прямой проход через энкодер
    encoded = self.encode(x)
    
    # 2. Декодирование обратно в исходную размерность
    decoded = self.decode(encoded, x.size(1))
    
    # 3. Вычисление MSE между входом и реконструкцией
    error = F.mse_loss(decoded, x.float(), reduction='none')
    
    # 4. Усреднение по последовательности
    return error.mean(dim=1)
```

### 2.2 LSTM модель: адаптация для детекции аномалий

**Классификационная архитектура:**
```python
class LSTMModel(nn.Module):
    embedding = nn.Embedding(151, 384)      # Большие эмбеддинги
    lstm = nn.LSTM(384, 384, 3, dropout=0.15)  # 3 слоя, 384 нейрона
    classifier = nn.Linear(384, 151)        # Softmax классификация
```

**Технология адаптации к детекции аномалий:**
```python
# Перплексия как мера аномальности
def get_perplexity_score(outputs, targets):
    loss = F.cross_entropy(outputs.view(-1, vocab_size), 
                          targets.view(-1), reduction='mean')
    perplexity = torch.exp(loss)
    return perplexity.item()

# Высокая перплексия = аномалия
```

**Обоснование подхода:**
- Модель обучается предсказывать следующий системный вызов
- Нормальные последовательности имеют низкую перплексию  
- Аномальные последовательности трудно предсказать → высокая перплексия

---

## ⚙️ ЭТАП 3: ТЕХНОЛОГИИ ОБУЧЕНИЯ И ВАЛИДАЦИИ

### 3.1 Архитектура тренировочного pipeline

**Модульная система тренеров:**
```python
src/training/
├── trainer.py              # Базовый абстрактный класс
├── lstm_trainer.py         # LSTM-специфичные методы
└── autoencoder_trainer.py  # Автоэнкодер-специфичные методы
```

### 3.2 Технологии раннего остановления

**Алгоритм EarlyStopping:**
```python
class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-4, mode='min'):
        self.patience = patience        # Терпение: 8 эпох без улучшения
        self.min_delta = min_delta     # Минимальное значимое улучшение
        self.best_score = None
        self.counter = 0
        
    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0  # Сброс счетчика
        else:
            self.counter += 1
            
        return self.counter >= self.patience
```

### 3.3 Технология мониторинга обучения

**Система отслеживания метрик:**
```python
class LossTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def update(self, train_loss, val_loss, train_acc, val_acc):
        # Логирование каждой эпохи
        # Автоматическая визуализация кривых обучения
        # Сохранение в pickle для анализа
```

### 3.4 Оптимизация и регуляризация

**Adam оптимизатор с разными настройками:**
```python
# GRU-Автоэнкодер (быстрое обучение)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# LSTM (медленное стабильное обучение)  
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
```

**Gradient Clipping:**
```python
# Предотвращение взрывающихся градиентов
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Dropout стратегии:**
- GRU-AE: 0.2 (агрессивная регуляризация)
- LSTM: 0.15 (более консервативная)

### 3.5 Технология сохранения моделей

**Стратегия checkpoint'ов:**
```python
def save_checkpoint(self, epoch, model, optimizer, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': self.config,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, path)
```

---

## 📈 ЭТАП 4: ТЕХНОЛОГИИ ТЕСТИРОВАНИЯ И ВАЛИДАЦИИ

### 4.1 Архитектура системы оценки

**Модульная система evaluator'ов:**
```python
src/testing/evaluator.py
├── Evaluator              # Базовый класс  
├── LSTMEvaluator         # LSTM-специфичная оценка
└── AutoencoderEvaluator  # Автоэнкодер-специфичная оценка
```

### 4.2 Технология вычисления метрик

**Комплексная система метрик:**
```python
def calculate_metrics(y_true, y_pred, y_scores=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'specificity': specificity_score(y_true, y_pred),
        'npv': npv_score(y_true, y_pred)
    }
    
    if y_scores is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
        metrics['average_precision'] = average_precision_score(y_true, y_scores)
        
    return metrics
```

### 4.3 Технология оптимального порога

**Алгоритм поиска оптимального порога:**
```python
from sklearn.metrics import roc_curve

def find_optimal_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Метод Юдена: максимизация (TPR - FPR)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold
```

### 4.4 Статистическая валидация

**Bootstrap анализ стабильности:**
```python
def bootstrap_metrics(y_true, y_pred, n_bootstrap=1000):
    bootstrap_scores = []
    n_samples = len(y_true)
    
    for i in range(n_bootstrap):
        # Случайная выборка с возвращением
        indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_true = y_true[indices]
        bootstrap_pred = y_pred[indices]
        
        score = f1_score(bootstrap_true, bootstrap_pred)
        bootstrap_scores.append(score)
    
    # 95% доверительный интервал
    ci_lower = np.percentile(bootstrap_scores, 2.5)
    ci_upper = np.percentile(bootstrap_scores, 97.5)
    
    return ci_lower, ci_upper
```

---

## 📊 ЭТАП 5: ТЕХНОЛОГИИ ВИЗУАЛИЗАЦИИ И АНАЛИЗА

### 5.1 Архитектура модуля визуализации

**Структура plotting системы:**
```python
src/visualization/
├── training_plotter.py     # Кривые обучения
├── results_plotter.py      # ROC, PR кривые  
├── comparison_plotter.py   # Сравнение моделей
└── analysis_plotter.py     # Статистический анализ
```

### 5.2 Технология создания комплексных графиков

**ROC-AUC анализ (autoencoder_vs_lstm_comparison_roc_curve.png):**
```python
def plot_roc_curve(y_true, y_scores, title="ROC Curve"):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, linewidth=3, 
             label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.6)
    
    # Отмечаем оптимальный порог
    optimal_idx = np.argmax(tpr - fpr)
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], 
                color='red', s=100, zorder=5)
```

**Precision-Recall анализ (autoencoder_vs_lstm_comparison_precision_recall.png):**
```python
def plot_precision_recall_curve(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, linewidth=3,
             label=f'PR curve (AP = {avg_precision:.4f})')
    
    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.6)
```

### 5.3 Комплексный анализ (comprehensive_analysis.png)

**4-панельная визуализация:**
```python
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Панель 1: Распределение ошибок реконструкции
ax1.hist(normal_errors, bins=50, alpha=0.7, label='Normal', density=True)
ax1.hist(attack_errors, bins=50, alpha=0.7, label='Attacks', density=True)
ax1.axvline(threshold, color='red', linestyle='--', label='Threshold')

# Панель 2: ROC кривая
plot_roc_on_axis(ax2, y_true, y_scores)

# Панель 3: Precision-Recall кривая  
plot_pr_on_axis(ax3, y_true, y_scores)

# Панель 4: Confusion Matrix
plot_confusion_matrix_on_axis(ax4, y_true, y_pred)
```

### 5.4 История обучения (training_history.png)

**Двухпанельная визуализация:**
```python
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curves
    ax1.plot(history['train_loss'], label='Training Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves (если доступны)
    if 'train_acc' in history:
        ax2.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
        ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
```

---

## 🔬 ЭТАП 6: ТЕХНОЛОГИИ СРАВНИТЕЛЬНОГО АНАЛИЗА

### 6.1 Статистические тесты значимости

**Критерий Макнемара для сравнения классификаторов:**
```python
from statsmodels.stats.contingency_tables import mcnemar

def mcnemar_test(y_true, y_pred1, y_pred2):
    # Создаем таблицу сопряженности
    correct_1 = (y_true == y_pred1)
    correct_2 = (y_true == y_pred2)
    
    # Случаи расхождения
    diff_12 = np.sum(correct_1 & ~correct_2)  # Модель 1 права, 2 нет
    diff_21 = np.sum(~correct_1 & correct_2)  # Модель 2 права, 1 нет
    
    # Статистика Макнемара
    statistic = (abs(diff_12 - diff_21) - 1)**2 / (diff_12 + diff_21)
    p_value = 1 - chi2.cdf(statistic, 1)
    
    return statistic, p_value
```

### 6.2 Технология ансамблевых методов

**Взвешенное голосование моделей:**
```python
class EnsembleModel:
    def predict_ensemble(self, test_loader, strategy="weighted_voting", weights=(0.6, 0.4)):
        # Получаем скоры от обеих моделей
        lstm_scores = self.get_lstm_scores(test_loader)
        ae_scores = self.get_autoencoder_scores(test_loader)
        
        # Нормализация в [0,1]
        lstm_norm = (lstm_scores - lstm_scores.min()) / (lstm_scores.max() - lstm_scores.min())
        ae_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min())
        
        # Взвешенное комбинирование
        w_lstm, w_ae = weights
        ensemble_scores = w_lstm * lstm_norm + w_ae * ae_norm
        
        return ensemble_scores
```

---

## 💾 ТЕХНОЛОГИИ ПЕРСИСТЕНТНОСТИ И ВОСПРОИЗВОДИМОСТИ

### Система управления экспериментами
```python
class NeuroDetekt:
    def _create_experiment_dir(self, experiment_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = Path(self.config['paths']['base_dir']) / f"{experiment_name}_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем конфигурацию эксперимента
        config_path = experiment_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        return experiment_dir
```

### Детерминированность экспериментов
```python
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 📊 АНАЛИЗ ГРАФИКОВ ИЗ TRIALS

### График 1: ROC-кривая (autoencoder_vs_lstm_comparison_roc_curve.png)
**Технический анализ:**
- **AUC = 0.9296** - превосходная способность различения классов
- **Оптимальная точка Юдена** - максимизирует (TPR - FPR)
- **Крутой подъем в начале** - высокая чувствительность при низкой FPR
- **Площадь под кривой** указывает на почти идеальную сепарабельность классов

### График 2: Precision-Recall кривая (autoencoder_vs_lstm_comparison_precision_recall.png)  
**Технический анализ:**
- **Average Precision = 0.5291** - умеренная производительность на несбалансированных данных
- **Резкое падение precision** при высоком recall - характерно для автоэнкодеров
- **Baseline = 0.13** (доля позитивных примеров)
- **Область под кривой** показывает компромисс precision/recall

### График 3: Комплексный анализ (comprehensive_analysis.png)
**4-панельный технический анализ:**

**Панель 1 - Распределения ошибок:**
- Четкое разделение между нормальными (μ=2.28) и атаками (μ=3.97)
- Перекрытие распределений объясняет ложные срабатывания
- Порог 2.99 оптимально разделяет классы

**Панель 2 - ROC анализ:**
- Подтверждает AUC = 0.9296
- Оптимальная рабочая точка отмечена красным

**Панель 3 - PR анализ:**
- Визуализирует трейдофф precision vs recall
- Показывает влияние дисбаланса классов

**Панель 4 - Confusion Matrix:**
- TP=1131, FP=880, TN=6757, FN=14
- Наглядно демонстрирует высокий recall, но умеренный precision

### График 4: История обучения (training_history.png)
**Анализ конвергенции:**
- **Стабильная конвергенция** за 15 эпох
- **Отсутствие переобучения** - валидационная кривая следует за обучающей
- **Раннее остановление** сработало правильно
- **Гладкие кривые** указывают на стабильное обучение

### График 5: Финальные результаты (final_model_results.png)
**LSTM модель анализ:**
- Более сбалансированная confusion matrix
- Меньше ложных срабатываний
- Стабильная производительность на разных типах данных

### График 6: Обучение финальной модели (final_model_training.png)
**Анализ процесса обучения LSTM:**
- **22 эпохи** до сходимости (vs 15 у автоэнкодера)
- **Более медленная конвергенция** из-за сложности архитектуры
- **Стабильное улучшение** метрик на протяжении обучения

---

## 🎯 ТЕХНОЛОГИЧЕСКИЕ ВЫВОДЫ

### Преимущества реализованной архитектуры:
1. **Модульность** - четкое разделение ответственности
2. **Масштабируемость** - легкое добавление новых моделей
3. **Воспроизводимость** - детерминированные эксперименты
4. **Эффективность** - оптимизация под GPU
5. **Мониторинг** - комплексная система отслеживания
6. **Персистентность** - надежное сохранение результатов

### Технологические инновации:
1. **Адаптация LSTM** для unsupervised детекции через перплексию
2. **Двухуровневое кэширование** данных (pickle + numpy)
3. **Dynamic padding** для последовательностей переменной длины
4. **Комплексная визуализация** с 4-панельными графиками
5. **Статистически обоснованный** выбор порогов детекции
6. **Ансамблевые методы** для комбинирования моделей

Данная техническая реализация обеспечивает производственную готовность системы и служит надежной основой для следующего этапа - интеграции с reinforcement learning. 