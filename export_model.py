from ultralytics import YOLO

model = YOLO(r"D:\yolov11.2\runs\detect\train6\weights\last.pt")

model.export(format="tflite")
