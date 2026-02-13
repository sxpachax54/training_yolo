from ultralytics import YOLO

# โหลดโมเดลจาก Path ของคุณ
model = YOLO(r"D:\yolov11.2\runs\detect\train6\weights\last.pt")

# สั่งแปลงเป็น TFLite
model.export(format="tflite")