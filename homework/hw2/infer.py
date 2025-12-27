# -*- coding: utf-8 -*-

from ultralytics import YOLO


def main():
    # 1. Загружаем лучшие веса после обучения
    model = YOLO("runs/pets_yolov8n2/weights/best.pt")

    # 2. Путь к изображению(ям) для теста
    # Можно указать одну картинку или паттерн типа "dataset/valid/images/*.jpg"
    source = "homework/hw2/dataset/valid/images"  # прогнать все валид-картинки

    # 3. Запускаем инференс с сохранением результатов
    results = model.predict(
        source=source,
        conf=0.25,          # порог confidence
        save=True,          # сохранять картинки с разрисованными bbox
        save_txt=True,      # сохранить txt с предсказаниями
        project="runs",
        name="pets_infer",  # папка с результатами
    )

    print("Готово. Результаты в runs/pets_infer/.")


if __name__ == "__main__":
    main()
