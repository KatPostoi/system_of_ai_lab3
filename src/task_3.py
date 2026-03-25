"""
Лабораторная работа 3: Автоматический подбор гиперпараметров с keras-tuner
для CIFAR-10
"""


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
from kerastuner.tuners import Hyperband


def load_and_preprocess_data():
    """Загрузка и предобработка данных CIFAR-10"""
    print("Загрузка данных CIFAR-10...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Названия классов
    classes = ['самолет', 'автомобиль', 'птица', 'кот', 'олень', 'собака', 'лягушка',
               'лошадь', 'корабль', 'грузовик']

    # Нормализация
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

    return X_train, y_train, X_test, y_test, classes

def show_sample_image(X_train, y_train, classes):
    """Отображение примера изображения"""
    n = 35
    plt.figure(figsize=(6, 6))
    plt.imshow(X_train[n])
    plt.title(f"Тип объекта: {classes[np.argmax(y_train[n])]}")
    plt.axis('off')
    plt.show()

def build_model(hp):
    """Функция построения модели для тюнинга"""
    model = Sequential()

    # 1-й сверточный блок
    model.add(Conv2D(
        filters=hp.Int('conv1_filters', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('conv1_kernel', values=[3, 5]),
        activation='relu',
        input_shape=(32, 32, 3),
        padding='same'
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hp.Float('dropout1', min_value=0.1, max_value=0.3, step=0.1)))

    # 2-й сверточный блок
    model.add(Conv2D(
        filters=hp.Int('conv2_filters', min_value=64, max_value=256, step=32),
        kernel_size=hp.Choice('conv2_kernel', values=[3, 5]),
        activation='relu',
        padding='same'
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hp.Float('dropout2', min_value=0.2, max_value=0.4, step=0.1)))

    # 3-й сверточный блок
    model.add(Conv2D(
        filters=hp.Int('conv3_filters', min_value=128, max_value=512, step=64),
        kernel_size=hp.Choice('conv3_kernel', values=[3, 5]),
        activation='relu',
        padding='same'
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hp.Float('dropout3', min_value=0.3, max_value=0.5, step=0.1)))

    # Полносвязные слои
    model.add(Flatten())
    model.add(Dense(
        units=hp.Int('dense_units', min_value=128, max_value=512, step=64),
        activation='relu'
    ))
    model.add(Dropout(hp.Float('dense_dropout', min_value=0.3, max_value=0.5, step=0.1)))
    model.add(Dense(10, activation='softmax'))

    # Компиляция
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def run_hyperparameter_search(X_train, y_train):
    """Запуск поиска гиперпараметров БЕЗ tensorboard"""
    tuner = Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=20,
        factor=3,
        directory='cifar10_tuning',
        project_name='cifar10_cnn',
        seed=42,
        overwrite=True  # Очищает предыдущие результаты
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("Начинаем автоматический подбор гиперпараметров...")
    tuner.search(
        X_train, y_train,
        epochs=30,  # Уменьшил для быстрого теста
        validation_split=0.2,
        batch_size=64,  # Уменьшил batch_size
        callbacks=[early_stopping],
        verbose=1
    )

    return tuner


def evaluate_top_models(tuner, X_test, y_test):
    """Оценка топ-5 моделей"""
    top5_models = tuner.get_best_models(num_models=5)
    print("\n=== ТОП-5 ЛУЧШИХ АРХИТЕКТУР ===")

    results = []
    for i, model in enumerate(top5_models):
        print(f"\nМодель #{i+1}:")
        model.summary()
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy = {test_acc:.4f}")
        results.append((test_acc, model))
        print("-" * 50)

    return sorted(results, key=lambda x: x[0], reverse=True)

def train_final_model(best_model, X_train, y_train):
    """Финальное обучение лучшей модели"""
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("\nОбучаем лучшую модель на всех данных...")
    history = best_model.fit(
        X_train, y_train,
        epochs=50,
        validation_split=0.2,
        batch_size=128,
        callbacks=[early_stopping],
        verbose=1
    )

    return history

def plot_predictions(model, X_test, y_test, classes, num_images=12):
    """Визуализация предсказаний"""
    predictions = model.predict(X_test[:num_images])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test[:num_images], axis=1)

    plt.figure(figsize=(15, 10))
    for i in range(num_images):
        plt.subplot(3, 4, i+1)
        plt.imshow(X_test[i])
        color = 'green' if predicted_classes[i] == true_classes[i] else 'red'
        plt.title(f'Пред: {classes[predicted_classes[i]]}\nИст: {classes[true_classes[i]]}',
                 color=color, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_model(model, filename='best_cifar10_cnn.h5'):
    """Сохранение модели"""
    model.save(filename)
    print(f"Модель сохранена как '{filename}'")
