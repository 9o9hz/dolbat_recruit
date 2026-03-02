import cv2
import mediapipe as mp
import math
from ultralytics import YOLO

# ====== 설정 (YOLO) ======
MODEL_PATH = "./yolov8n-face.pt"
MIN_AREA_RATIO = 0.04
MIN_WIDTH_RATIO = 0.18
CONF_THRES = 0.25
IOU_THRES = 0.45

# ====== 카메라 설정 ======
CAM_DEV_1 = "/dev/video0"
CAM_DEV_2 = "/dev/video2"

# 모델 로드 (YOLO)
yolo_model = YOLO(MODEL_PATH)

# 카메라 초기화
cap1 = cv2.VideoCapture(CAM_DEV_1, cv2.CAP_V4L2)
cap2 = cv2.VideoCapture(CAM_DEV_2, cv2.CAP_V4L2)

for cap in [cap1, cap2]:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

# ====== 미디어파이프 초기화 ======
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# [함수] 손 제스처 판별
def get_gesture(hand_landmarks):
    open_fingers = []
    wrist = hand_landmarks.landmark[0] # 기준점: 손목 좌표

    tip_ids = [8, 12, 16, 20] # 검지, 중지, 약지, 소지 끝
    pip_ids = [6, 10, 14, 18] # 검지, 중지, 약지, 소지 중간 마디

    # 1. 거리 기반 손가락 펴짐/접힘 판단
    for i in range(4):
        tip = hand_landmarks.landmark[tip_ids[i]]
        pip = hand_landmarks.landmark[pip_ids[i]]

        dist_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
        dist_pip = math.hypot(pip.x - wrist.x, pip.y - wrist.y)

        # 끝마디가 중간마디보다 손목에서 멀면 펴진 것으로 간주
        if dist_tip > dist_pip:
            open_fingers.append(True)
        else:
            open_fingers.append(False)

    # 2. 제스처 판단 로직
    # 펼쳐진 손가락 개수 확인
    num_open = open_fingers.count(True)

    # A. 네 손가락이 모두 접혀있는 상태 (주먹 기반 제스처)
    if num_open == 0:
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[3]
        thumb_mcp = hand_landmarks.landmark[2]

        if thumb_tip.y < thumb_ip.y < thumb_mcp.y:
            return "Thumbs Up"
        elif thumb_tip.y > thumb_ip.y > thumb_mcp.y:
            return "DOLBAT"
        else:
            return "Rock"

    # B. (검지와 중지만 펴진 상태)
    elif open_fingers == [True, True, False, False]:
        return "^.^ v"

    # C. (손가락이 3개 이상 펴진 상태)
    
    elif num_open >= 3:
        return "Hi-"
    
    else:
        return "???"


# [함수] 통합 처리 (얼굴 + 손)
def process_combined(frame):
    if frame is None: return frame
    H, W = frame.shape[:2]
    
    # 💡 색상 설정 (RGB -> BGR 변환)
    MY_COLOR = (45, 94, 36) # RGB(36, 94, 45)를 BGR로 표현
    
    # 1. YOLO 얼굴 인식
    yolo_results = yolo_model.predict(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
    for box in yolo_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bw, bh = x2 - x1, y2 - y1
        if (bw*bh)/(W*H) < MIN_AREA_RATIO and (bw/W) < MIN_WIDTH_RATIO: continue
        
        # 💡 박스 색상 변경 (MY_COLOR 사용)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), MY_COLOR, 2)
        
        # 💡 라벨 명칭 변경 (~.~) 및 색상 변경
        cv2.putText(frame, "~.~", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, MY_COLOR, 2)

    # 2. MediaPipe 손 인식 (이하 동일)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = get_gesture(hand_landmarks)
            cx, cy = int(hand_landmarks.landmark[0].x * W), int(hand_landmarks.landmark[0].y * H)
            cv2.putText(frame, gesture, (cx, cy - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return frame

# ====== 메인 루프 ======
cv2.namedWindow("Camera 1", cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera 2", cv2.WINDOW_NORMAL)

while True:
    ret1, f1 = cap1.read()
    ret2, f2 = cap2.read()

    if ret1:
        # 💡 [추가] 좌우 반전 (1: 좌우 반전, 0: 상하 반전, -1: 좌우/상하 모두 반전)
        f1 = cv2.flip(f1, 1) 
        
        f1 = process_combined(f1)
        cv2.imshow("Camera 1", cv2.resize(f1, (800, 600)))

    if ret2:
        # 💡 [추가] 카메라 2도 동일하게 반전
        f2 = cv2.flip(f2, 1)
        
        f2 = process_combined(f2)
        cv2.imshow("Camera 2", cv2.resize(f2, (800, 600)))

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()