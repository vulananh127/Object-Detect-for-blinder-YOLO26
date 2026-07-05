from ultralytics import YOLO
import cv2
import time
import threading
from gtts import gTTS
import pygame
import os
import numpy as np

FOCAL_LENGTH   = 942.76 
ALERT_DISTANCE = 200    

REAL_WIDTHS = {
    "bed":   172,   # cm
    "chair":  40,
    "table":  150,
    "door":   100,
    "stair":  82,
    "fridge": 52,
    "washing_machine": 57,
}

pygame.mixer.init()

last_spoken_time  = {}   
cooldown_seconds  = 3


def phat_loa_ngam(text):
    try:
       
        safe_name = "".join(c for c in text if c.isalnum() or c == " ")[:40]
        filename  = f"cache_{safe_name}.mp3".replace(" ", "_")

        if not os.path.exists(filename):
            tts = gTTS(text=text, lang="vi")
            tts.save(filename)

        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()

    except Exception as e:
        print(f"Lỗi phát âm thanh: {e}")

def doc_canh_bao(class_name, text):
    current_time = time.time()
    last_time    = last_spoken_time.get(class_name, 0)

    if (current_time - last_time > cooldown_seconds) and \
       not pygame.mixer.music.get_busy():
        last_spoken_time[class_name] = current_time
        t = threading.Thread(target=phat_loa_ngam, args=(text,))
        t.daemon = True
        t.start()


def estimate_distance(class_name_raw, pixel_width):
    real_w = REAL_WIDTHS.get(class_name_raw)
    if not real_w or pixel_width == 0:
        return None
    return (real_w * FOCAL_LENGTH) / pixel_width

def format_distance(dist_cm):
    if dist_cm < 100:
        return f"{int(dist_cm)} xăng ti mét"
    else:
        meters = dist_cm / 100
        return f"{meters:.1f} mét".replace(".", " phẩy ")


def viet_hoa(ten):
    labels = {
        "Person": "Người",
        "person": "Người",
        "bed":   "Cái giường",
        "Bed":   "Cái giường",
        "chair": "Cái ghế",
        "Chair": "Cái ghế",
        "table": "Cái bàn",
        "Table": "Cái bàn",
        "door":  "Cái cửa",
        "Door":  "Cái cửa",
        "stair": "Cầu thang",
        "Stairs": "Cầu thang",
        "fridge": "Cái tủ lạnh",
        "Fridge": "Cái tủ lạnh",
        "kitchen-object": "Đồ dùng bếp",
        "Kitchen Object":"Đồ dùng bếp",
        "computer-device": "Thiết bị máy tính",
        "Computer Device": "Thiết bị máy tính",
        "phone": "Điện thoại",
        "Phone": "Điện thoại",
        "washing-machine": "Cái máy giặt",
        "Washing Machine": "Cái máy giặt",
        "fan": "Cái quạt",
        "Fan": "Cái quạt",
        "toilet": "Nhà vệ sinh",
        "Toilet": "Nhà vệ sinh",
        "sink": "Bồn rửa",
        "Sink": "Bồn rửa",
        "glass": "Kính",
        "Glass": "Kính",
        "sharp-object": "Vật sắc nhọn",
        "Sharp Object": "Vật sắc nhọn",
    }
    return labels.get(ten, ten)


print("Đang tải mô hình...")
model = YOLO("models/best3.pt")

# camera_url = "http://192.168.137.112:8080/video"
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

W_FRAME = 640


def resize_fit_in(image, target_w, target_h):
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return result




while True:
    success, frame = cap.read()
    if not success:
        break

    # frame = cv2.resize(frame, (640, 480))
    frame= resize_fit_in(frame, 640, 480)
    results = model(frame, conf=0.6)

    detections = []
    for box in results[0].boxes:
        class_id  = int(box.cls[0])
        class_raw = model.names[class_id]        
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        pixel_w   = x2 - x1
        distance  = estimate_distance(class_raw, pixel_w)

        detections.append({
            "class_raw": class_raw,
            "label":     viet_hoa(class_raw),
            "distance":  distance,              
            "x1": x1, "x2": x2,
            "y1": y1, "y2": y2,
        })

    detections.sort(key=lambda d: d["distance"] if d["distance"] else 9999)

    for det in detections:
        x1, y1   = det["x1"], det["y1"]
        x2, y2   = det["x2"], det["y2"]
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        distance = det["distance"]
        direction = ""
        if x_center < W_FRAME / 3:
            if y_center < 640 / 3:
                direction = "bên trái"
        elif x_center > (2 * W_FRAME) / 3:
            direction = "bên phải"
        else:
            direction = "phía trước"
        cau_noi   = f"Chú ý, {det['label']} ở {direction}"
        
        if distance is not None:
            if distance < ALERT_DISTANCE:
                dist_text = format_distance(distance)
                
                # cau_noi   = f"Chú ý, {det['label']} ở {direction}, cách {dist_text}"
                doc_canh_bao(det['label'], cau_noi)

                cv2.rectangle(frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame,
                            f"{det['class_raw']} | {distance:.0f}cm | {direction}",
                            (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame,
                            f"{det['class_raw']} | {distance:.0f}cm",
                            (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # cau_noi = f"Chú ý, {det['label']} ở {direction}"
            if distance is not None:
                distance_text = f"{distance:.0f}cm"
            else:
                distance_text = "N/A"
            doc_canh_bao(det['label'], cau_noi)
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 165, 255), 2)
            cv2.putText(frame,
                            f"{det['class_raw']} | {distance_text}cm",
                            (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"[{det['class_raw']}] {direction} | "
              f"dist={distance:.0f}cm" if distance else f"[{det['label']}] {direction} | dist=N/A")

    cv2.imshow("He Thong AI Phat Hien Vat The", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()