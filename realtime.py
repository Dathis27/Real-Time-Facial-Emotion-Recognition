"""Real-time emotion recognition using the model trained by train.py.
Run:  python realtime.py      (q = quit, s = screenshot)
"""
import json, time
from collections import deque, Counter
import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("emotion_model.keras")
labels = json.load(open("labels.json"))
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

COLORS = {"happy": (0, 200, 0), "sad": (200, 100, 0), "angry": (0, 0, 220),
          "fear": (160, 0, 160), "surprise": (0, 200, 220),
          "disgust": (0, 120, 80), "neutral": (200, 200, 200)}

cap = cv2.VideoCapture(0)
hist = deque(maxlen=8)
prev, fps = time.time(), 0.0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        roi = cv2.resize(gray[y:y + h, x:x + w], (48, 48)).astype("float32")
        probs = model.predict(roi[None, ..., None], verbose=0)[0]
        hist.append(int(np.argmax(probs)))
        idx = Counter(hist).most_common(1)[0][0]
        name, col = labels[idx], COLORS.get(labels[idx], (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
        cv2.putText(frame, f"{name} {probs[idx]*100:.0f}%", (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        for i, (lab, p) in enumerate(zip(labels, probs)):   # probability bars
            cv2.rectangle(frame, (10, 30 + i * 22), (10 + int(p * 150), 44 + i * 22),
                          COLORS.get(lab, (255, 255, 255)), -1)
            cv2.putText(frame, f"{lab} {p*100:.0f}%", (170, 42 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        break  # first face only for the bar panel

    now = time.time(); fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6); prev = now
    cv2.putText(frame, f"FPS {fps:.0f}", (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("Emotion Recognition", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break
    if k == ord("s"):
        cv2.imwrite(f"shot_{int(now)}.png", frame)

cap.release(); cv2.destroyAllWindows()
