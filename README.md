# DA-2-11

---
Инструмент для обнаружения аномалий на примере датасета `diabetes` из `sklearn`.

`IsolationForest.py` — полный пайплайн: загрузка данных, обучение модели IsolationForest, предсказание аномалий, подсчёт статистики и визуализация результатов.

## Обзор возможностей

- Загрузка встроенного датасета diabetes из sklearn.
- Обучение модели IsolationForest с заданными параметрами.
- Предсказание меток (норма = 1, аномалия = -1).
- Подсчёт общего числа аномалий.
- Визуализация данных через PCA (сохранение в anomalies_scatter.png + отображение графика).

---

## Установка (рекомендовано: виртуальное окружение)

### Linux / macOS
```bash
# создать venv и активировать
python3 -m venv .venv
source .venv/bin/activate

# обновить pip и установить зависимости
python -m pip install --upgrade pip
pip install -r requirements.txt
```
### Windows

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

---

## Примеры использования 

```bash
python IsolationForest.py
```

