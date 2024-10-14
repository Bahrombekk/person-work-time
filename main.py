import cv2
import numpy as np
from ultralytics import YOLO
import math
from datetime import datetime, timedelta
import json
import os
import time

# YOLO modelini yuklash
model = YOLO("yolov8m.pt")

# Berilgan to'rtburchak koordinatalari va ularning ismlari
rectangles = [
    {"name": "Bahrombek", "coords": (861, 784, 332, 427)},
    {"name": "Akmal", "coords": (1284, 777, 366, 419)},
    {"name": "Ismoil", "coords": (887, 452, 291, 208)},
    {"name": "Shaxrillo", "coords": (1201, 493, 230, 196)}
]

def is_person_in_area(person_box, area_box):
    px1, py1, px2, py2 = person_box
    ax, ay, aw, ah = area_box
    ax1, ay1, ax2, ay2 = ax, ay, ax + aw, ay + ah
    return (ax1 < px1 < ax2 or ax1 < px2 < ax2) and (ay1 < py1 < ay2 or ay1 < py2 < ay2)

def save_time_data(total_times, start_times):
    data = {
        f"area_{i}": {
            "total_time": total_time.total_seconds(),
            "start_time": start_time.isoformat() if start_time else None
        } for i, (total_time, start_time) in enumerate(zip(total_times, start_times))
    }
    with open("time_data.json", "w") as f:
        json.dump(data, f)

def load_time_data():
    if os.path.exists("time_data.json"):
        try:
            with open("time_data.json", "r") as f:
                data = json.load(f)
            total_times = []
            start_times = []
            for i in range(4):
                area_data = data.get(f"area_{i}", {"total_time": 0, "start_time": None})
                total_times.append(timedelta(seconds=area_data["total_time"]))
                start_times.append(datetime.fromisoformat(area_data["start_time"]) if area_data["start_time"] else None)
            return total_times, start_times
        except (json.JSONDecodeError, KeyError, ValueError):
            print("Saqlangan ma'lumotlarni o'qishda xatolik. Yangi ma'lumotlar yaratilmoqda.")
    return [timedelta() for _ in range(4)], [None for _ in range(4)]

def main():
    # Videoni ochish (0 - kompyuterning asosiy kamerasi)
    cap = cv2.VideoCapture("rtsp://admin:DAS2024@@192.168.136.234:554/Streamin/Channels/401",
                           0)
    
    # Kamera framening o'lchamlarini olish
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # FPSni olish
    fps = cap.get(cv2.CAP_PROP_FPS)

    # VideoWriter obyektini yaratish
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output3.mp4', fourcc, fps, (width, height))
    
    # Vaqtni kuzatish uchun o'zgaruvchilar
    total_times, start_times = load_time_data()
    persons_in_areas = [start_time is not None for start_time in start_times]

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Kamera bilan bog'lanishda xatolik. Qayta urinish...")
                cap.release()
                time.sleep(5)  # 5 soniya kutish
                cap = cv2.VideoCapture("rtsp://admin:DAS2024@@192.168.136.234:554/Streamin/Channels/401")
                continue

            # YOLO orqali obyektlarni aniqlash
            results = model(frame, stream=True)

            persons_detected = [False for _ in range(4)]
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    if int(box.cls[0]) == 0:  # 0 - odam sinfi
                        x_center, y_center, width, height = box.xywh[0]
                        x1 = int(x_center - width/2)
                        y1 = int(y_center - height/2)
                        x2 = int(x_center + width/2)
                        y2 = int(y_center + height/2)

                        for i, rect in enumerate(rectangles):
                            if is_person_in_area((x1, y1, x2, y2), rect["coords"]):
                                persons_detected[i] = True
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                conf = math.ceil(box.conf[0] * 100) / 100
                                cv2.putText(frame, f"Human: {conf}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Berilgan to'rtburchaklarni chizish va ismlarini ko'rsatish
            for i, rect in enumerate(rectangles):
                x, y, w, h = rect["coords"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, rect["name"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Vaqtni hisoblash
            current_time = datetime.now()
            for i in range(4):
                if persons_detected[i] and not persons_in_areas[i]:
                    start_times[i] = current_time
                    persons_in_areas[i] = True
                elif not persons_detected[i] and persons_in_areas[i]:
                    total_times[i] += current_time - start_times[i]
                    start_times[i] = None
                    persons_in_areas[i] = False

                if persons_in_areas[i]:
                    current_duration = current_time - start_times[i] + total_times[i]
                else:
                    current_duration = total_times[i]

                # Vaqtni ko'rsatish
                hours, remainder = divmod(int(current_duration.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = f"{rectangles[i]['name']}: {hours:02d}:{minutes:02d}:{seconds:02d}"
                cv2.putText(frame, time_str, (20, 40 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)

            # Frameни videoga yozish
            out.write(frame)

            # Oynani kichikroq ko'rsatish
            cv2.namedWindow("Human Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Human Detection", 640, 480)

            # Natijani ko'rsatish
            cv2.imshow("Human Detection", frame)

            # Vaqt ma'lumotlarini saqlash
            save_time_data(total_times, start_times)

            # 'q' tugmasi bosilganda dasturni to'xtatish
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Resurslarni bo'shatish
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as e:
            print(f"Xatolik yuz berdi: {e}")
            print("Dastur qayta ishga tushirilmoqda...")
            time.sleep(5)  # 5 soniya kutish
