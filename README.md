# Учебный проект, посвященный сжатию изображений при помощи нейронных сетей

## Пример работы

Результаты 100 эпох при квантовании 2
![Результаты 100 эпох при квантовании 2](./doc/result_b2_epoch100.png)

Результаты 100 эпох при квантовании 3
![Результаты 100 эпох при квантовании 3](./doc/result_b3_epoch100.png)


## Действия для запуска
    pyton -m venv venv
    venv/Scripts/activate
    pip install -r requirements.txt
    python EntropySetup.py build_ext --inplace
    python CNNImageCodec.py