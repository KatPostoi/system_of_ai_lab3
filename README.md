# system_of_ai_lab1
Лабораторная работа по системам искусственного интеллекта: распознавание рукописных цифр MNIST.

## Запуск проекта
- Запустить код: `MPLBACKEND=QtAgg ./.venv/bin/python src/main.py` 
- Создай виртуальное окружение: `python -m venv .venv`
- Активируй окружение: Linux/macOS `source .venv/bin/activate` или Windows `.venv\\Scripts\\activate`
- Установи зависимости: `pip install -r requirements.txt`
- Запуск (совместимый со старой схемой): `./.venv/bin/python src/main.py`
- Альтернативный запуск как пакет: `./.venv/bin/python -m src.main`

## Проверки качества
- Линт: `./.venv/bin/ruff check .`
- Форматирование: `./.venv/bin/ruff format .`
- Проверка типов (Pyright): `pyright`

## Структура
- `src/config.py` — гиперпараметры и константы
- `src/ml_types.py` — доменные типы и контракты
- `src/data.py` — загрузка и предобработка MNIST
- `src/model.py` — создание модели
- `src/train.py` — обучение и оценка
- `src/visualize.py` — интерактивная визуализация (`plt.show`)
- `src/task_1.py` — фасад сценария ЛР
- `src/main.py` — entrypoint
