from ultralytics import YOLO

# เปลี่ยนไปใช้โมเดลที่ฝึกฝนแล้ว
model = YOLO('runs/detect/train2/weights/best.pt') 

results = model("test/010.jpg")
results[0].show()