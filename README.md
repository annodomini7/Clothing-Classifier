# Clothing Classifier

**Коган Анна Алексеевна**

Классификатор одежды на основе ResNet50 с поддержкой TensorRT и Triton Inference
Server.

## Постановка задачи

Определение типа одежды по изображению. Полезно для маркетплейсов и
онлайн-магазинов для автоматической разметки товаров.

## Формат данных

- **Вход**: изображение JPG/PNG, размер 224×224×3
- **Выход**: вектор вероятностей размерности 107 (количество классов, для
  которых в датасете хотя бы 10 экземпляра есть, это число можно настроить в
  параметре min_samples_per_class в файле dataset.py в модуле
  ClothingDataModule)

## Метрики

- Accuracy
- Top-5 Accuracy
- Macro F1-Score

## Датасет

[Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
— почти 45тыс изображений, 143 класса, из них 107 с хотя бы 10 экземплярами.

---

## Установка

```bash
uv sync
```

## Обучение модели

```bash
cd clothing_classifier
uv run python train.py
```

Результаты сохраняются в:

- `checkpoints/` — веса модели
- `plots/` — графики (TensorBoard, CSV)
- `triton/export/onnx/` — ONNX модель
- MLflow UI: http://localhost:8080

### Параметры обучения

Конфигурация в `configs/`:

- `model/resnet50.yaml` — архитектура модели
- `training/default.yaml` — параметры обучения
- `optimizer/adamw.yaml` — оптимизатор
