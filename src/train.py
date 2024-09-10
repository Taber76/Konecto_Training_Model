from ultralytics import YOLO
import os
os.environ['ULTRALYTICS_SETTINGS'] = 'D:/Escritorio/Proyectos/Konecto_Training_Model/src/settings.yaml'


# Cambia esto si deseas usar un modelo diferente, como yolov8s.pt, yolov8m.pt, o yolov8l.pt
model_path = 'yolov8n.pt'
data_path = 'data.yaml'
epochs = 50                # Número de épocas para entrenar
img_size = 320             # Tamaño de imagen para el entrenamiento


# Inicializar y entrenar el modelo
def train_yolov8():
    model = YOLO(model_path)
    model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size
    )


if __name__ == "__main__":
    train_yolov8()
