from ultralytics import YOLO

# 1. โหลดโมเดลเริ่มต้น
model = YOLO('yolo11n.pt') 

# 2. เริ่มการฝึกฝนโดยใช้ชุดข้อมูลของคุณ
# 'data.yaml' จะบอกโมเดลว่าคลาสคือ ['Green', 'Red', 'Yellow']
results = model.train(data='datasets/data.yaml', epochs=100) 

# เมื่อฝึกเสร็จ โมเดลใหม่จะถูกบันทึกใน runs/detect/trainX/weights/best.pt