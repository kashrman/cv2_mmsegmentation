# Базовый конфиг для кода EDA и доразметки

import os

PROJECT_ROOT = '../../../'
DATA_SRC_ROOT = os.path.join(PROJECT_ROOT, 'data', 'src') # исходные наборы
DATA_ERR_ROOT = os.path.join(PROJECT_ROOT, 'data', 'err') # сюда откладываем ошибки разметки
DATA_ADD_ROOT = os.path.join(PROJECT_ROOT, 'data', 'add') # для доразметки
DATA_CLEARED_ROOT = os.path.join(PROJECT_ROOT, 'data', 'cleared') # очищенные и проверенные наборы
os.makedirs(DATA_ERR_ROOT, exist_ok=True)
os.makedirs(DATA_CLEARED_ROOT, exist_ok=True)

EDA_RESULTS = os.path.join(PROJECT_ROOT, 'practicum_work', 'artifacts', 'eda_results')
os.makedirs(EDA_RESULTS, exist_ok=True)


CLASS_NAMES = {0: 'background', 1: 'cat', 2: 'dog'}
CLASS_COLORS = [[1, 1, 1], [254, 1, 1], [1, 254, 1]]

