# train-converter-rknn

Проект обучает YOLO-модель, конвертирует .pt &rarr; .onnx &rarr; .rknn, опционально копирует по `scp` на OrangePi

1. Разметить датасет `data/YORDATASET`
2. Указать путь до датасета в  `train/data.yaml`
3. Запуск:

```bash
bash train-convert.sh 
```

С указанием `IP` для копирования по `scp`:

```bash
bash train-convert.sh 192.168.0.100
```
