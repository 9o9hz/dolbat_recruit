import cv2
import mediapipe as mp
import math

# ====== 카메라 설정 ====== 
# CAM_DEV_1 정의를 삭제하고 CAM_DEV_2만 남깁니다.
CAM_DEV_2 = "/dev/video0"

# cap1 관련 모든 코드를 삭제했습니다.
cap2 = cv2.VideoCapture(CAM_DEV_2, cv2.CAP_V4L2)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap2.set(cv2.CAP_PROP_FPS, 30)

# ====== 미디어파이프 초기화 ======
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

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

    # B. 가위 (검지와 중지만 펴진 상태)
    elif open_fingers == [True, True, False, False]:
        return "^.^ v"

    # C. 보 (손가락이 3개 이상 펴진 상태)
    # 엄지를 제외한 네 손가락 중 3개 이상만 펴져도 '보'로 인식하도록 넉넉하게 잡았습니다.
    elif num_open >= 3:
        return "Hi-"
    
    else:
        return "???"

def process_frame(frame):
    if frame is None:
        return frame
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = get_gesture(hand_landmarks)
            
            # 아까 발생했던 h -> H 오류를 방지하기 위해 정석대로 작성
            H, W, C = frame.shape
            cx, cy = int(hand_landmarks.landmark[0].x * W), int(hand_landmarks.landmark[0].y * H)
            
            cv2.putText(frame, gesture, (cx, cy - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        
    return frame

# ====== 실행 부분 ======
cv2.namedWindow("Camera 2", cv2.WINDOW_NORMAL)

print("press q to quit")

while True:
    ret2, frame2 = cap2.read()

    if ret2:
        # 좌우 반전 추가 (원치 않으시면 아래 줄을 삭제하세요)
        frame2 = cv2.flip(frame2, 1)
        
        frame2 = process_frame(frame2)
        frame2_resized = cv2.resize(frame2, (800, 600))
        cv2.imshow("Camera 2", frame2_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제도 cap2만 진행
cap2.release()
cv2.destroyAllWindows()