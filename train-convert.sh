#!bin/bash

IP="$1"


source /home/ohrimenkoed/Документы/eva/2026/MlServer/venv/bin/activate

cd train
python3 train.py

cd  ../converter/convert/
python3 convert.py

cp ../model/target/best.rknn ../../result

cd ../../result
clear
deactivate
echo "Model for orange:"
ls -lh best.rknn


if [ -n "$IP" ]; then
    echo "Копируем на Orange Pi..."
    scp best.rknn orangepi@"$IP":/home/orangepi/orange-1/model/target/
fi