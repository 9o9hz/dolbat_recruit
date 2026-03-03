import cv2
import mediapipe as mp
import math
import os
from ultralytics import YOLO

# =========================
# 설정 (YOLO)
# =========================
MODEL_PATH = "/home/j/yolo_face/yolov8n-face.pt"  # ✅ 절대경로로 고정
MIN_AREA_RATIO = 0.04
MIN_WIDTH_RATIO = 0.18
CONF_THRES = 0.25
IOU_THRES = 0.45

# =========================
# 카메라 설정
# =========================
CAM_DEV_1 = "/dev/video0"
CAM_DEV_2 = "/dev/video2"
FRAME_W, FRAME_H = 640, 480
FPS = 30

print("PWD:", os.getcwd())
print("MODEL_PATH:", MODEL_PATH)
print("EXISTS:", os.path.exists(MODEL_PATH))

# ✅ YOLO 모델 로드 (딱 1번만!)
yolo_model = YOLO(MODEL_PATH)

# =========================
# 카메라 초기화
# =========================
cap1 = cv2.VideoCapture(CAM_DEV_1, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(CAM_DEV_2, cv2.CAP_V4L2)

for cap in (cap1, cap2):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS)

# =========================
# MediaPipe 초기화
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# [함수] 손 제스처 판별
# =========================
def get_gesture(hand_landmarks):
    open_fingers = []
    wrist = hand_landmarks.landmark[0]  # 손목

    tip_ids = [8, 12, 16, 20]  # 검지, 중지, 약지, 소지 끝
    pip_ids = [6, 10, 14, 18]  # 검지, 중지, 약지, 소지 중간 마디

    for i in range(4):
        tip = hand_landmarks.landmark[tip_ids[i]]
        pip = hand_landmarks.landmark[pip_ids[i]]

        dist_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        dist_pip = math.hypot(pip.x - wrist.x, pip.y - wrist.y)

        open_fingers.append(dist_tip > dist_pip)

    num_open = open_fingers.count(True)

    if num_open == 0:
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]

        if thumb_tip.y < thumb_ip.y < thumb_mcp.y:
            return "Thumbs Up"
        elif thumb_tip.y > thumb_ip.y > thumb_mcp.y:
            return "DOLBAT"
        else:
            return "Rock!"

    elif open_fingers == [True, True, False, False]:
        return "^.^ v"
    elif num_open >= 3:
        return "Hi-"
    else:
        return "???"

# =========================
# [함수] 통합 처리 (얼굴 + 손)
# =========================
def process_combined(frame):
    if frame is None:
        return frame

    # ✅ 밝은 연두색(라임 느낌) - OpenCV는 BGR
    FACE_COLOR = (0, 255, 128)  # 박스/라벨/테스트박스/글씨 모두 이 색으로
    HAND_TEXT_COLOR = (255, 255, 255)  # 손 제스처 텍스트는 흰색(원하면 FACE_COLOR로 바꿔도 됨)


    H, W = frame.shape[:2]

    # 1) YOLO 얼굴 인식
    yolo_results = yolo_model.predict(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
    for box in yolo_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bw, bh = x2 - x1, y2 - y1

        # 가까운 얼굴만(너가 잡아둔 조건 유지)
        if (bw * bh) / (W * H) < MIN_AREA_RATIO and (bw / W) < MIN_WIDTH_RATIO:
            continue

        # ✅ 박스 두께 확 올리기
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), FACE_COLOR, 6)

        # ✅ 라벨 배경 채우기 (가독성 상승)
        label = "~.~"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(
            frame,
            (int(x1), int(y1) - th - 14),
            (int(x1) + tw + 10, int(y1)),
            FACE_COLOR,
            -1
        )

        # ✅ 글씨도 같은 색(FACE_COLOR)으로 (검정 X)
        # 배경도 FACE_COLOR라서 글씨가 겹칠 수 있음 → 글씨를 더 두껍게 + 외곽선(흰색)로 해결
        # (원치 않으면 OUTLINE_COLOR 줄 지우고 putText 1번만 쓰면 됨)
        OUTLINE_COLOR = (255, 255, 255)
        cv2.putText(frame, label, (int(x1) + 5, int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, OUTLINE_COLOR, 4)
        cv2.putText(frame, label, (int(x1) + 5, int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, FACE_COLOR, 2)

    # 2) MediaPipe 손 인식
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = get_gesture(hand_landmarks)
            cx, cy = int(hand_landmarks.landmark[0].x * W), int(hand_landmarks.landmark[0].y * H)
            cv2.putText(frame, gesture, (cx, cy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, HAND_TEXT_COLOR, 3)

    return frame

# =========================
# 메인 루프
# =========================
cv2.namedWindow("Camera 1", cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera 2", cv2.WINDOW_NORMAL)

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()

    if ret1:
        f1 = cv2.flip(f1, 1)  # 좌우반전
        f1 = process_combined(f1)
        cv2.imshow("Camera 1", cv2.resize(f1, (800, 600)))

    if ret2:
        f2 = cv2.flip(f2, 1)  # 좌우반전
        f2 = process_combined(f2)
        cv2.imshow("Camera 2", cv2.resize(f2, (800, 600)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cap2.release()
cv2.destroyAllWindows()