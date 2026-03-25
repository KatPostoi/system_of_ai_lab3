"""
Главная точка входа для лабораторной работы 3
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Подавить INFO и WARNING
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключить oneDNN

import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from task_3 import (
    evaluate_top_models,
    load_and_preprocess_data,
    plot_predictions,
    run_hyperparameter_search,
    save_model,
    show_sample_image,
    train_final_model,
)


def main():
    print("=== ЛАБОРАТОРНАЯ РАБОТА 3 ===")
    print("Автоматический подбор гиперпараметров для CIFAR-10\n")

    # 1. Загрузка и предобработка данных
    X_train, y_train, X_test, y_test, classes = load_and_preprocess_data()
    show_sample_image(X_train, y_train, classes)

    # 2. Поиск гиперпараметров
    tuner = run_hyperparameter_search(X_train, y_train)

    # 3. Оценка топ-5 моделей
    results = evaluate_top_models(tuner, X_test, y_test)

    # 4. Выбор и обучение лучшей модели
    best_acc, best_model = results[0]
    print("\n ЛУЧШАЯ МОДЕЛЬ:")
    print(f"Точность на тестовой выборке: {best_acc:.4f}")
    best_model.summary()

    # Финальное обучение
    history = train_final_model(best_model, X_train, y_train)

    # Финальная оценка
    final_loss, final_acc = best_model.evaluate(X_test, y_test, verbose=0)
    print("\nФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
    print(f"Точность на тестовой выборке: {final_acc:.4f}")

    # 5. Визуализация предсказаний
    plot_predictions(best_model, X_test, y_test, classes)

    # 6. Сохранение модели
    save_model(best_model)

    print("\n Лабораторная работа завершена успешно!")

if __name__ == "__main__":
    main()

