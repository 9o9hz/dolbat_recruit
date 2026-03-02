import cv2
from ultralytics import YOLO
import sys

# ====== 하드웨어 강제 고정 ======
CAM_DEV = "/dev/video2"  # 컴퓨터 내장 웹캠
MODEL_PATH = "./yolov8n-face.pt"

# ====== YOLO 및 필터 설정 ======
CONF_THRES = 0.25
IOU_THRES = 0.45
MIN_AREA_RATIO = 0.04
MIN_WIDTH_RATIO = 0.18

# 모델 로드
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Model Load Error: {e}")
    sys.exit()

# 카메라 초기화 (기존 연결을 확실히 끊기 위해 다시 정의)
cap = cv2.VideoCapture(CAM_DEV, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print(f"FATAL: Camera {CAM_DEV}를 찾을 수 없습니다.")
    sys.exit()


def process_frame(frame, model):
    if frame is None: return frame
    H, W = frame.shape[:2]
    
    MY_COLOR = (45, 94, 36) # RGB -> BGR로 변경

    results = model.predict(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
    
    if results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bw, bh = x2 - x1, y2 - y1
            
            # 크기 필터링
            if (bw * bh) / (W * H) < MIN_AREA_RATIO and (bw / W) < MIN_WIDTH_RATIO:
                continue

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), MY_COLOR, 2)
            cv2.putText(frame, "~.~", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, MY_COLOR, 2)
    return frame

# ====== 실행 부분 ======
cv2.destroyAllWindows()
cv2.namedWindow("ONLY_CAMERA_0", cv2.WINDOW_NORMAL)

print("시작되었습니다. 'q'를 누르면 종료됩니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 1. 좌우 반전
    frame = cv2.flip(frame, 1)
    
    # 2. 얼굴 인식 처리
    frame = process_frame(frame, model)
    
    # 3. 화면 표시
    cv2.imshow("ONLY_CAMERA_0", cv2.resize(frame, (800, 600)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()