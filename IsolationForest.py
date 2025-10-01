import numpy as np
import sys
from sklearn.datasets import load_diabetes
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_dataset():
    """
    Загружает датасет diabetes из sklearn.
    Возвращает (X, y).
    Бросает исключение при ошибке загрузки.
    """
    try:
        data = load_diabetes()
    except Exception as e:
        raise RuntimeError(f"Ошибка при загрузке датасета load_diabetes(): {e}")

    if not hasattr(data, "data") or not hasattr(data, "target"):
        raise RuntimeError("Загруженный объект не содержит 'data' или 'target' атрибутов.")

    X = np.asarray(data.data)
    y = np.asarray(data.target)

    if X.size == 0:
        raise RuntimeError("Данные X пусты после загрузки.")
    if y.size == 0:
        raise RuntimeError("Метки y пусты после загрузки.")

    print(f"[INFO] Датасет загружен успешно: X.shape={X.shape}, y.shape={y.shape}")
    return X, y


def train_isolation_forest(X, n_estimators=100, contamination=0.1, random_state=42):
    """
    Обучает IsolationForest на X.
    Возвращает обученную модель.
    """
    if X is None:
        raise ValueError("Входные данные X равны None.")
    if X.ndim != 2:
        raise ValueError(f"Ожидается 2D массив X, получили ndim={X.ndim}.")

    try:
        model = IsolationForest(
            n_estimators=int(n_estimators),
            contamination=float(contamination),
            random_state=int(random_state)
        )
        model.fit(X)
    except Exception as e:
        raise RuntimeError(f"Ошибка при обучении IsolationForest: {e}")

    print(f"[INFO] IsolationForest обучена: n_estimators={n_estimators}, contamination={contamination}")
    return model


def predict_anomalies(model, X):
    """
    Получает предсказания от модели:
    возвращает массив меток 1 (норма) и -1 (аномалия).
    """
    if model is None:
        raise ValueError("Модель равна None.")
    if X is None:
        raise ValueError("Входные данные X равны None.")

    try:
        preds = model.predict(X)
    except Exception as e:
        raise RuntimeError(f"Ошибка при получении предсказаний: {e}")

    unique = np.unique(preds)
    if not np.all(np.isin(unique, [-1, 1])):
        raise RuntimeError(f"Ожидались метки в наборе {{-1, 1}}, но получили {unique}")

    print(f"[INFO] Предсказания получены: уникальные метки = {unique}")
    return preds


def count_anomalies(preds):
    """
    Подсчитывает количество аномалий (метка -1).
    """
    if preds is None:
        raise ValueError("preds равен None.")
    if preds.size == 0:
        return 0

    n_anom = int(np.sum(preds == -1))
    print(f"[RESULT] Количество обнаруженных аномалий: {n_anom} из {preds.size} ({n_anom / preds.size:.2%})")
    return n_anom


def plot_results(X, preds, out_path="anomalies_scatter.png"):
    """
    Визуализирует данные
    Для отображения используется PCA (2 компоненты).
    Аномалии помечаются другим цветом/маркером.
    Сохраняет картинку в файл out_path и показывает plt.show().
    """
    if X is None or preds is None:
        raise ValueError("X или preds равны None.")

    if X.ndim != 2:
        raise ValueError("Ожидается 2D X для визуализации (будет применён PCA если >2).")

    try:
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(X)
    except Exception as e:
        raise RuntimeError(f"Ошибка при применении PCA для 2D проекции: {e}")

    mask_anom = preds == -1
    mask_norm = preds == 1

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X2[mask_norm, 0], X2[mask_norm, 1], s=30, label="Норма (1)", alpha=0.7, edgecolors='k')
    ax.scatter(X2[mask_anom, 0], X2[mask_anom, 1], s=60, label="Аномалия (-1)", marker='x', linewidths=2)

    ax.set_title("IsolationForest: выявленные аномалии (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    ax.grid(True)

    try:
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"[INFO] График сохранён в: {out_path}")
        plt.show()
    except Exception as e:
        raise RuntimeError(f"Ошибка при сохранении/отображении графика: {e}")


def main():
    """
    Пайплайн:
    1) Загружаем датасет
    2) Обучаем модель
    3) Находим аномалии
    4) Выводим распределение
    """
    try:
        X, y = load_dataset()
    except Exception as e:
        print(f"[ERROR] Не удалось загрузить датасет: {e}", file=sys.stderr)
        return

    try:
        model = train_isolation_forest(X, n_estimators=100, contamination=0.1, random_state=42)
    except Exception as e:
        print(f"[ERROR] Не удалось обучить IsolationForest: {e}", file=sys.stderr)
        return

    try:
        preds = predict_anomalies(model, X)
    except Exception as e:
        print(f"[ERROR] Не удалось получить предсказания: {e}", file=sys.stderr)
        return

    try:
        count_anomalies(preds)
    except Exception as e:
        print(f"[ERROR] Ошибка при подсчёте аномалий: {e}", file=sys.stderr)
        return

    try:
        plot_results(X, preds, out_path="anomalies_scatter.png")
    except Exception as e:
        print(f"[ERROR] Ошибка при визуализации результатов: {e}", file=sys.stderr)
        return


if __name__ == "__main__":
    print(sys.executable)
    main()
