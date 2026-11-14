import cv2
import json
import time
import os
import sys
from collections import Counter
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics belum terinstall. Jalankan: pip install -r requirements.txt")
    sys.exit(1)

CONFIG_FILE = "sources.json"
MODEL_FILE = "yolov8n.pt"

os.system("")

GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

ALLOWED = {
    "person", "bicycle", "car", "motorcycle",
    "bus", "truck", "traffic light", "stop sign"
}

def load_cameras():
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "cameras" in data:
        data = data["cameras"]

    if not isinstance(data, list):
        raise ValueError("cctv.json harus berupa list, atau object dengan key 'cameras'")

    return data

def open_stream(source):
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap

def draw_panel(frame, location, counts, fps, source_status):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (350, 245), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)

    cv2.putText(frame, "VissionTrack", (28, 43),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255,255,255), 2)
    cv2.putText(frame, location[:38], (28, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,255,220), 1)
    cv2.putText(frame, f"STATUS : {source_status}", (28, 94),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,220,220), 1)
    cv2.putText(frame, f"FPS    : {fps:.1f}", (28, 116),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,220,220), 1)

    y = 145
    order = ["person", "car", "motorcycle", "bus", "truck", "bicycle"]
    for name in order:
        cv2.putText(frame, f"{name:<11}: {counts.get(name, 0)}",
                    (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255,255,255), 1)
        y += 21

    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (w - 235, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255,255,255), 1)
    return frame

def choose_camera(cameras):
    print("\n=== CCTV PUBLIC ===")
    for i, cam in enumerate(cameras, 1):
        print(f"{i}. {cam['name']} — {cam['location']}")
    print("C. Masukkan URL stream sendiri")
    print("Q. Keluar")

    choice = input("\nPilih CCTV: ").strip().lower()
    if choice == "q":
        return None

    if choice == "c":
        url = input("Masukkan URL RTSP/HLS/MJPEG: ").strip()
        if not url:
            return None
        return {"name": "Custom CCTV", "location": "Custom", "url": url}

    try:
        idx = int(choice) - 1
        return cameras[idx]
    except (ValueError, IndexError):
        print("Pilihan tidak valid.")
        return choose_camera(cameras)

def log_frame(frame_no, fps, counts, total):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{BOLD}{CYAN}[{ts}]{RESET} "
          f"frame={frame_no} "
          f"fps={GREEN}{fps:5.1f}{RESET} "
          f"detected={YELLOW}{total}{RESET}")
    if total == 0:
        print(f"   {DIM}— no object —{RESET}")
    else:
        for name, c in counts.most_common():
            bar = "█" * min(c, 40)
            print(f"   {name:<11} {YELLOW}{bar}{RESET} {c}")

def main():
    print("======================================")
    print("OBJECT DETECTOR")
    print("======================================")
    print("Loading model:", MODEL_FILE)

    model = YOLO(MODEL_FILE)
    cameras = load_cameras()

    cam = choose_camera(cameras)
    if cam is None:
        return

    print(f"\nMembuka: {cam['name']}")
    print("Tekan Q untuk kembali/keluar.")
    print("Tekan S untuk menyimpan screenshot.")
    print("Tekan R untuk reconnect stream.\n")

    cap = open_stream(cam["url"])
    last_time = time.time()
    fps = 0.0
    frame_no = 0
    reconnects = 0

    window_name = f"VissionTrack - {cam['name']}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while True:
        ok, frame = cap.read()

        if not ok or frame is None:
            reconnects += 1
            cv2.putText(frame if frame is not None else
                        (cv2.UMat(720, 1280, cv2.CV_8UC3).get()),
                        "STREAM TERPUTUS - reconnecting...",
                        (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 2)
            cv2.imshow(window_name, frame if frame is not None else
                       cv2.UMat(720, 1280, cv2.CV_8UC3).get())

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r") or reconnects >= 30:
                cap.release()
                time.sleep(1)
                cap = open_stream(cam["url"])
                reconnects = 0
            continue

        reconnects = 0
        frame_no += 1

        frame = cv2.resize(frame, (1280, 720))

        results = model.predict(
            source=frame,
            conf=0.35,
            imgsz=640,
            verbose=False,
            classes=[0, 1, 2, 3, 5, 7, 9, 11]
        )

        counts = Counter()

        result = results[0]
        if result.boxes is not None:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id]

                if name not in ALLOWED:
                    continue

                counts[name] += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 2)
                label = f"{name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                )
                cv2.rectangle(frame, (x1, max(0, y1-th-8)),
                              (x1+tw+6, y1), (255,180,0), -1)
                cv2.putText(frame, label, (x1+3, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0,0,0), 2)

        now = time.time()
        dt = now - last_time
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_time = now

        # ===== LOG KE TERMINAL =====
        if frame_no % 10 == 0:
            log_frame(frame_no, fps, counts, sum(counts.values()))

        frame = draw_panel(
            frame,
            cam["location"],
            counts,
            fps,
            "LIVE"
        )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            cap.release()
            time.sleep(0.5)
            cap = open_stream(cam["url"])
        elif key == ord("s"):
            os.makedirs("screenshots", exist_ok=True)
            filename = datetime.now().strftime("screenshots/cctv_%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(filename, frame)
            print("Screenshot:", filename)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()