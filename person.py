import cv2
import numpy as np
from ultralytics import YOLO
import math
from datetime import datetime, timedelta
import json
import os
import time

# YOLO modelini yuklash
model = YOLO("yolov8n.pt")

# Berilgan to'rtburchak koordinatalari
x, y, w, h = 793, 735, 487, 579

def is_person_in_area(person_box, area_box):
    px1, py1, px2, py2 = person_box
    ax1, ay1, ax2, ay2 = area_box
    return (ax1 < px1 < ax2 or ax1 < px2 < ax2) and (ay1 < py1 < ay2 or ay1 < py2 < ay2)

def save_time_data(total_time, start_time):
    data = {
        "total_time": total_time.total_seconds(),
        "start_time": start_time.isoformat() if start_time else None
    }
    with open("time_data.json", "w") as f:
        json.dump(data, f)

def load_time_data():
    if os.path.exists("time_data.json"):
        with open("time_data.json", "r") as f:
            data = json.load(f)
        total_time = timedelta(seconds=data["total_time"])
        start_time = datetime.fromisoformat(data["start_time"]) if data["start_time"] else None
        return total_time, start_time
    return timedelta(), None

def main():
    # Videoni ochish (0 - kompyuterning asosiy kamerasi)
    cap = cv2.VideoCapture(0)

    # Vaqtni kuzatish uchun o'zgaruvchilar
    total_time, start_time = load_time_data()
    person_in_area = start_time is not None

    while True:
        success, frame = cap.read()
        if not success:
            print("Kamera bilan bog'lanishda xatolik. Qayta urinish...")
            cap.release()
            time.sleep(5)  # 5 soniya kutish
            cap = cv2.VideoCapture(0)
            continue

        # YOLO orqali obyektlarni aniqlash
        results = model(frame, stream=True)

        person_detected = False
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if int(box.cls[0]) == 0:  # 0 - odam sinfi
                    x_center, y_center, width, height = box.xywh[0]
                    x1 = int(x_center - width/2)
                    y1 = int(y_center - height/2)
                    x2 = int(x_center + width/2)
                    y2 = int(y_center + height/2)

                    if is_person_in_area((x1, y1, x2, y2), (x, y, x+w, y+h)):
                        person_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        conf = math.ceil(box.conf[0] * 100) / 100
                        cv2.putText(frame, f"Human: {conf}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Berilgan to'rtburchakni chizish
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

        # Vaqtni hisoblash
        current_time = datetime.now()
        if person_detected and not person_in_area:
            start_time = current_time
            person_in_area = True
        elif not person_detected and person_in_area:
            total_time += current_time - start_time
            start_time = None
            person_in_area = False

        if person_in_area:
            current_duration = current_time - start_time + total_time
        else:
            current_duration = total_time

        # Vaqtni ko'rsatish
        hours, remainder = divmod(int(current_duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"Vaqt: {hours:02d}:{minutes:02d}:{seconds:02d}"
        cv2.putText(frame, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Oynani kichikroq ko'rsatish
        cv2.namedWindow("Human Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Human Detection", 640, 480)

        # Natijani ko'rsatish
        cv2.imshow("Human Detection", frame)

        # Vaqt ma'lumotlarini saqlash
        save_time_data(total_time, start_time)

        # 'q' tugmasi bosilganda dasturni to'xtatish
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Resurslarni bo'shatish
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print(f"Xatolik yuz berdi: {e}")
            print("Dastur qayta ishga tushirilmoqda...")
            time.sleep(5)  # 5 soniya kutish