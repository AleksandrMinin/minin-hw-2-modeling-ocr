# Barcode detection

Решение задачи ocr штрих-кодов на изображениях.


### Датасет

Включает 302 изображений штрих-кодов, для которых распознаны цифры под ними.
Скачать данные можно [отсюда](https://disk.yandex.ru/d/kUkdcBR78Fzoxg).
После скачивания файлов, необходимо поменять пути в src/constants.py:
1) DATA_PATH - папка со всеми данными
2) DF_PATH - путь до num_under_barcodes.tsv
3) TRAIN_IMAGES_PATH - папка с вырезанными штрих-кодами
(создать папку и положить содержимое toloka_crop_barcodes, train_barcodes, val_barcodes)


### Подготовка пайплайна

1. Создание и активация окружения
    ```
    python3 -m venv /path/to/new/virtual/environment
    ```
    ```
    source /path/to/new/virtual/environment/bin/activate
    ```

2. Установка пакетов

    В активированном окружении:

    a. Обновить pip
    ```
    pip install --upgrade pip 
    ```
    b. Выполнить команду
    ```
    pip install -r requirements.txt
    ```

3. Настройка ClearML

    a. [В своем профиле ClearML](https://app.community.clear.ml/profile) нажимаем:
      "Settings" -> "Workspace" -> "Create new credentials"
      
    b. Появится инструкция для "LOCAL PYTHON", копируем её.
    
    с. Пишем в консоли `clearml-init` и вставляем конфиг из инструкции.

### Обучение на сгенерированных штрих-кодах
Перед основным обучением модели, необходимо провести предобучение на сгенерированных штрих-кодах.
Запуск тренировки c `nohup`:

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python pretrain_crnn.py > pretrain_log.out
```

Запуск тренировки без `nohup`:

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python pretrain_crnn.py
```
Лучшую модель необходимо сохранить в ./gen_weights/model.best.pth

### Обучение
Запуск тренировки c `nohup`:

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 nohup python train.py > log.out
```

Запуск тренировки без `nohup`:

```
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python train.py
```
Лучшую модель необходимо сохранить в ./weights/model.best.pth

### ClearML
Метрики и конфигурации экспериментов:
1. [pretrain_1](https://app.clear.ml/projects/f86aa4664160426aa4f0e91fd4d061f8/experiments/36642f3b700146f08b85bb1eca66a4b0/output/execution)
2. [experiment_1](https://app.clear.ml/projects/f86aa4664160426aa4f0e91fd4d061f8/experiments/e09e9f8b846f4584836201587e15fdb9/output/execution)
3. [experiment_2](https://app.clear.ml/projects/f86aa4664160426aa4f0e91fd4d061f8/experiments/4813cb246310461aa472ee0ae26218e0/output/execution)
4. [pretrain_3](https://app.clear.ml/projects/f86aa4664160426aa4f0e91fd4d061f8/experiments/893548adf1a44a1cb1d71b628beb6057/output/execution)
5. [experiment_3](https://app.clear.ml/projects/f86aa4664160426aa4f0e91fd4d061f8/experiments/5fbc6dd692a44930a1e9be150856c1a3/output/execution)
6.[pretrain_4](https://app.clear.ml/projects/f86aa4664160426aa4f0e91fd4d061f8/experiments/ab092967fdbc47f39dc84331513fddd8/output/execution)
7.[experiment_4](https://app.clear.ml/projects/f86aa4664160426aa4f0e91fd4d061f8/experiments/2c28186938344180bb7c9d7986d69a8e/output/execution)

### DVC
#### Добавление модели в DVC
1. Добавление модели в DVC
    
    Копируем в `weights` обученную модель
    ```
    cd weights
    dvc add model.pt
    dvc push
   ```
   Если появится ошибка с правами, можно дополнительно указать путь до приватного ключа:
   ```
   dvc remote modify myremote keyfile /path/to/your/private_key
   ```
   Про генерацию ssh-ключа [здесь](https://selectel.ru/blog/tutorials/how-to-generate-ssh/).

2. Делаем коммит с новой моделью:
    ```
    git add .
    git commit -m "add new model"
   ```

#### Загрузка лучшей модели из DVC к себе
   ```
    git pull origin main
    dvc pull
   ```

### Запуск литера
Из папки с проектом выполнить:
   ```
   python -m pip install wemake-python-styleguide==0.16.1
   flake8 src/
   ```

### Запуск тестов на pytest
Из папки с проектом выполнить:
   ```
   PYTHONPATH=. pytest tests -p no:warnings
   ```
### Model to ONNX
Перенос модели в формат ONNX:
   ```
   python -W ignore::UserWarning model_to_onnx.py
   ```
