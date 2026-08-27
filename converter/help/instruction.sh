# Установить нужные компоненты
    bash init/pip.sh
# Наполнить dataset
    python3 track/trackAdd.py
# Запустить формирование модели (Pt)
    python3 model/train.py
# Pt - > ONNNX - > RKNN
    python3 convert/convert.py