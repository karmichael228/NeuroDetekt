def get_data(data_set, batch_size=64, train_test_split=None, cross_validation=None, num_workers=4):
    """Загружает данные и создает DataLoader'ы для обучения и тестирования.

    Parameters
    ----------
    data_set : {"plaid"}
        Набор данных для загрузки и разделения.
    batch_size : int
        Размер батча для загрузчиков данных.
    train_test_split : tuple(int, int), optional
        Tuple (train%, test%) для разделения данных, например (80, 20).
    cross_validation : int, optional
        Количество фолдов для кросс-валидации.
    num_workers : int
        Количество рабочих процессов для загрузки данных.

    Returns
    -------
    train_data : List
        Список последовательностей системных вызовов для обучения.
    test_data : List
        Список последовательностей системных вызовов для тестирования.
    test_labels : torch.Tensor
        Метки для тестовых данных (0 - нормальные, 1 - атаки).
    encoder : Encoder
        Энкодер для преобразования системных вызовов в числовые идентификаторы.
    attack_data : List
        Список последовательностей атакующих системных вызовов.
    baseline_data : List
        Список последовательностей нормальных системных вызовов.
    """
    if data_set != "plaid":
        raise ValueError("data_set must be plaid")

    # Загружаем данные
    train, val, test_val, atk, encoder, atk_files, normal_files = load_data_splits(
        data_set, train_test_split=train_test_split, cross_validation=cross_validation
    )
    
    # Формируем тестовые данные (нормальные + атакующие)
    test_data = test_val + atk
    
    # Формируем метки для тестовых данных (0 - нормальные, 1 - атаки)
    test_labels = torch.zeros(len(test_val) + len(atk))
    test_labels[len(test_val):] = 1
    
    return train, test_data, test_labels, encoder, atk, test_val 