#!/usr/bin/env python3
"""On-device camera perception for the guide robot (Raspberry Pi 5, all offline).

Run:      python3 perception.py            (announce objects + faces by voice)
          python3 perception.py --show     (also open a debug window)
Enroll:   python3 perception.py enroll Sarah   (look at the camera, ~3 s)
"""
import os
import queue
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

M = os.path.expanduser('~/models')
PIPER_MODEL = f'{M}/en_US-lessac-medium.onnx'        # medium voices = 22050 Hz
PIPER_RATE = '22050'
YUNET = f'{M}/face_detection_yunet_2023mar.onnx'
SFACE = f'{M}/face_recognition_sface_2021dec.onnx'
YOLO_MODEL = os.path.expanduser('~/models/yolov8n_ncnn_model')   # or 'yolov8n.pt'
FACES_DB = os.path.expanduser('~/robot/faces.npz')

W, H = 640, 480
MATCH_THRESH = 0.363          # SFace cosine threshold
OBJECT_COOLDOWN = 6.0         # s between repeats of the same label+zone
FACE_COOLDOWN = 10.0
MIN_AREA = 0.03               # ignore objects smaller than 3% of the frame
CLOSE_H = 0.60                # box taller than 60% of frame = "close"
OBJECTS = {'chair', 'couch', 'bed', 'dining table', 'toilet', 'tv', 'refrigerator',
           'sink', 'potted plant', 'dog', 'cat', 'bottle', 'cup', 'laptop',
           'suitcase', 'backpack', 'person'}


class Speaker:
    """Non-blocking offline TTS: Piper -> aplay. Drops phrases if backed up."""

    def __init__(self):
        self.q = queue.Queue(maxsize=3)
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text):
        print('SAY:', text, flush=True)
        try:
            self.q.put_nowait(text)
        except queue.Full:
            pass

    def _run(self):
        while True:
            text = self.q.get()
            p1 = subprocess.Popen(['piper', '--model', PIPER_MODEL, '--output-raw'],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL)
            p2 = subprocess.Popen(['aplay', '-q', '-r', PIPER_RATE, '-f', 'S16_LE',
                                   '-t', 'raw', '-'], stdin=p1.stdout)
            p1.stdin.write((text + '\n').encode())
            p1.stdin.close()
            p2.wait()


def zone(cx):
    return 'on your left' if cx < W / 3 else 'on your right' if cx > 2 * W / 3 else 'ahead'


def load_db():
    if not os.path.exists(FACES_DB):
        return {}
    d = np.load(FACES_DB)
    return {k: d[k] for k in d.files}


def make_face_models():
    det = cv2.FaceDetectorYN.create(YUNET, '', (W, H), 0.8, 0.3, 5000)
    rec = cv2.FaceRecognizerSF.create(SFACE, '')
    return det, rec


def new_camera():
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(main={'size': (W, H), 'format': 'RGB888'}))
    cam.start()
    time.sleep(1.0)
    return cam      # RGB888 arrays are already in OpenCV's BGR order


def enroll(name):
    cam = new_camera()
    det, rec = make_face_models()
    feats = []
    print(f'Enrolling {name}: look at the camera...')
    t0 = time.time()
    while len(feats) < 15 and time.time() - t0 < 30:
        frame = cam.capture_array()
        _, faces = det.detect(frame)
        if faces is not None and len(faces) == 1:
            feats.append(rec.feature(rec.alignCrop(frame, faces[0])).flatten())
            time.sleep(0.2)
    cam.stop()
    if len(feats) < 5:
        print('Not enough good face captures, try again with better light.')
        return
    mean = np.mean(feats, axis=0)
    mean /= np.linalg.norm(mean)
    db = load_db()
    db[name] = mean
    os.makedirs(os.path.dirname(FACES_DB), exist_ok=True)
    np.savez(FACES_DB, **db)
    print(f'Saved {name} ({len(feats)} samples) to {FACES_DB}')


def identify(rec, feat, db):
    best, score = None, -1.0
    for name, ref in db.items():
        s = rec.match(feat, ref.reshape(1, -1).astype(np.float32),
                      cv2.FaceRecognizerSF_FR_COSINE)
        if s > score:
            best, score = name, s
    return (best, score) if score >= MATCH_THRESH else (None, score)


def main(show):
    speaker = Speaker()
    cam = new_camera()
    det, rec = make_face_models()
    yolo = YOLO(YOLO_MODEL, task='detect')
    db = load_db()
    last_said = {}          # key -> timestamp
    last_face_time = 0.0
    n = 0
    speaker.say('Vision system ready.')

    def due(key, cooldown):
        now = time.time()
        if now - last_said.get(key, 0) >= cooldown:
            last_said[key] = now
            return True
        return False

    while True:
        frame = cam.capture_array()
        n += 1

        # ---- faces (every 3rd frame) ----
        if n % 3 == 0:
            _, faces = det.detect(frame)
            for f in (faces if faces is not None else []):
                x, y, w, h = f[:4]
                where = zone(x + w / 2)
                feat = rec.feature(rec.alignCrop(frame, f))
                name, score = identify(rec, feat, db) if db else (None, 0)
                if name:
                    if due(('face', name), FACE_COOLDOWN):
                        speaker.say(f'{name} is {where}.')
                else:
                    if due(('face', 'unknown'), FACE_COOLDOWN):
                        speaker.say(f'Unknown person {where}.')
                last_face_time = time.time()
                if show:
                    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        # ---- objects ----
        r = yolo(frame, imgsz=320, conf=0.5, verbose=False)[0]
        for b in r.boxes:
            label = yolo.names[int(b.cls)]
            if label not in OBJECTS:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            if (x2 - x1) * (y2 - y1) / (W * H) < MIN_AREA:
                continue
            if label == 'person' and time.time() - last_face_time < FACE_COOLDOWN:
                continue                    # a face announcement already covered them
            z = zone((x1 + x2) / 2)
            near = ', close' if (y2 - y1) / H > CLOSE_H else ''
            if due((label, z), OBJECT_COOLDOWN):
                speaker.say(f'{label} {z}{near}.')
            if show:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        if show:
            cv2.imshow('robot', frame)
            if cv2.waitKey(1) == 27:
                break


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'enroll':
        enroll(sys.argv[2])
    else:
        main('--show' in sys.argv)
