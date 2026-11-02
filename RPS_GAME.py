import tflite_runtime.interpreter as tflite
import numpy as np
import time
import cv2 

# ==========================================
# 1. LiteRT (TFLite) 모델 로딩 및 설정
# ==========================================
#modelPath = "best_full_integer_quant.tflite"
modelPath = "best_float32.tflite"
#modelPath = "best_float16.tflite"
#modelPath = "best_int8.tflite"
#modelPath = "best_integer_quant.tflite"
print('model path:', modelPath)

interpreter = tflite.Interpreter(model_path=modelPath)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_index = input_details[0]['index']
output_index = output_details[0]['index']
input_dtype = input_details[0]['dtype']
output_dtype = output_details[0]['dtype']
input_scale, input_zero = input_details[0]['quantization']
output_scale, output_zero = output_details[0]['quantization']

# 파라미터 및 클래스 정의
IMG_SIZE = 320
CONF_TH = 0.15
IOU_TH  = 0.45

ansToText = {0: 'SCISSORS', 1: 'ROCK', 2: 'PAPER'}

# UI 테마 컬러 (BGR 포맷)
COLOR_P1 = (255, 200, 0)     # Neon Cyan / Blue
COLOR_P2 = (203, 192, 255)   # Neon Pink / Magenta
COLOR_ACCENT = (0, 255, 255) # Yellow Glow

# ==========================================
# 2. UI 시각화 헬퍼 함수
# ==========================================
def draw_translucent_rect(img, x, y, w, h, color=(0, 0, 0), alpha=0.65):
    """반투명 오버레이 패널 그리기"""
    sub_img = img[y:y+h, x:x+w]
    rect = np.full(sub_img.shape, color, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 1 - alpha, rect, alpha, 0)
    img[y:y+h, x:x+w] = res

def draw_hud_box(img, box, color, thickness=2, corner_len=15):
    """Sleek한 L자형 모서리가 강조된 HUD 바운딩 박스"""
    x1, y1, x2, y2 = box

    # 기본 외각선 (얇게)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # 4개 모서리 L자 강조 (두껍게)
    t = thickness + 1
    # 좌상단
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, t, cv2.LINE_AA)
    # 우상단
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, t, cv2.LINE_AA)
    # 좌하단
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, t, cv2.LINE_AA)
    # 우하단
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, t, cv2.LINE_AA)

def draw_badge(img, text, x, y, bg_color, text_color=(0, 0, 0), scale=0.55, thickness=2):
    """플레이어 라벨 배지"""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad_x, pad_y = 8, 5

    bx1, by1 = x, y - th - (pad_y * 2)
    bx2, by2 = x + tw + (pad_x * 2), y

    if bx1 >= 0 and by1 >= 0 and bx2 <= img.shape[1] and by2 <= img.shape[0]:
        cv2.rectangle(img, (bx1, by1), (bx2, by2), bg_color, -1, cv2.LINE_AA)
        cv2.putText(img, text, (x + pad_x, y - pad_y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness - 1, cv2.LINE_AA)

# ==========================================
# 3. 가위바위보 승패 판정 함수
# ==========================================
def judge_rps(p1_cid, p2_cid):
    if p1_cid == p2_cid:
        return "DRAW!", (200, 200, 200)
    if (p1_cid - p2_cid) % 3 == 1:
        return "PLAYER 1 WIN!", COLOR_P1
    else:
        return "PLAYER 2 WIN!", COLOR_P2

# ==========================================
# 4. 전처리 & NMS 헬퍼 함수
# ==========================================
def letterbox(img, new_shape=(320, 320), color=(114, 114, 114)):
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(nw / w, nh / h)

    new_w, new_h = int(w * r), int(h * r)
    resized = cv2.resize(img, (new_w, new_h))

    pad_w = nw - new_w
    pad_h = nh - new_h
    pad_x = pad_w // 2
    pad_y = pad_h // 2

    padded = cv2.copyMakeBorder(
        resized, pad_y, pad_y, pad_x, pad_x,
        cv2.BORDER_CONSTANT, value=color
    )
    return padded, r, pad_x, pad_y

def nms(boxes, scores, iou_th):
    if not boxes:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        iou = (w * h) / (areas[i] + areas[order[1:]] - w * h + 1e-6)

        order = order[np.where(iou <= iou_th)[0] + 1]

    return keep

# ==========================================
# 5. YOLO 탐지 및 결과 반환
# ==========================================
def detect_hands(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_lb, r, pad_x, pad_y = letterbox(img_rgb, (IMG_SIZE, IMG_SIZE))
    img = img_lb.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    if input_dtype == np.int8:
        img_int8 = (img / input_scale) + input_zero
        img = img_int8.astype(np.int8)

    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    raw = interpreter.get_tensor(output_index)[0].transpose()

    if output_dtype == np.int8:
        raw = (raw.astype(np.float32) - output_zero) * output_scale

    boxes, scores, class_ids = [], [], []

    for det in raw:
        cx, cy, w, h = det[:4]
        cls_scores = det[4:7]

        cls_id = int(np.argmax(cls_scores))
        score = float(cls_scores[cls_id])

        if score < CONF_TH:
            continue

        cx *= IMG_SIZE; cy *= IMG_SIZE
        w  *= IMG_SIZE; h  *= IMG_SIZE

        x1 = (cx - w/2 - pad_x) / r
        y1 = (cy - h/2 - pad_y) / r
        x2 = (cx + w/2 - pad_x) / r
        y2 = (cy + h/2 - pad_y) / r

        boxes.append([
            int(np.clip(x1, 0, frame.shape[1])),
            int(np.clip(y1, 0, frame.shape[0])),
            int(np.clip(x2, 0, frame.shape[1])),
            int(np.clip(y2, 0, frame.shape[0]))
        ])
        scores.append(score)
        class_ids.append(cls_id)

    keep = nms(boxes, scores, IOU_TH)

    detections = []
    for i in keep:
        box = boxes[i]
        x_center = (box[0] + box[2]) / 2
        detections.append({
            'box': box,
            'cid': class_ids[i],
            'score': scores[i],
            'x_center': x_center
        })

    return detections

# ==========================================
# 6. 메인 테스트 루프
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow('RPS Arena', cv2.WINDOW_NORMAL)
cv2.resizeWindow('RPS Arena', 640+40, 480+60)

startTime = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # 거울 모드
    h, w = frame.shape[:2]

    detections = detect_hands(frame)

    # 1. 상단 스코어보드 반투명 패널
    draw_translucent_rect(frame, 0, 0, w, 75, color=(15, 15, 15), alpha=0.65)
    cv2.line(frame, (0, 75), (w, 75), (50, 50, 50), 1)

    # 2. 감지 상태별 UI 구성
    if len(detections) >= 2:
        detections.sort(key=lambda d: d['x_center'])
        p1 = detections[0]
        p2 = detections[1]

        # P1 (좌측 플레이어 - Cyan Theme)
        draw_hud_box(frame, p1['box'], COLOR_P1, thickness=2)
        p1_label = f"P1: {ansToText[p1['cid']]} ({int(p1['score']*100)}%)"
        draw_badge(frame, p1_label, p1['box'][0], max(p1['box'][1] - 8, 25), COLOR_P1)

        # P2 (우측 플레이어 - Pink Theme)
        draw_hud_box(frame, p2['box'], COLOR_P2, thickness=2)
        p2_label = f"P2: {ansToText[p2['cid']]} ({int(p2['score']*100)}%)"
        draw_badge(frame, p2_label, p2['box'][0], max(p2['box'][1] - 8, 25), COLOR_P2)

        # 승패 판정
        result_text, result_color = judge_rps(p1['cid'], p2['cid'])

        # 상단 HUD 대결 내역
        match_str = f"{ansToText[p1['cid']]}  VS  {ansToText[p2['cid']]}"
        cv2.putText(frame, match_str, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)

        # 승리 결과 출력
        cv2.putText(frame, result_text, (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, result_color, 2, cv2.LINE_AA)

    elif len(detections) == 1:
        p = detections[0]
        draw_hud_box(frame, p['box'], (0, 255, 150), thickness=2)
        p_label = f"HAND: {ansToText[p['cid']]} ({int(p['score']*100)}%)"
        draw_badge(frame, p_label, p['box'][0], max(p['box'][1] - 8, 25), (0, 255, 150))

        cv2.putText(frame, "WAITING FOR SECOND PLAYER...", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, "Detected: 1 Hand", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

    else:
        cv2.putText(frame, "WAITING FOR PLAYERS...", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, "Show 2 Hands to Start Game", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 200, 255), 2, cv2.LINE_AA)

    # 3. FPS 배지 오버레이
    curTime = time.time()
    fps = 1 / (curTime - startTime + 1e-6)
    startTime = curTime

    fps_str = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_str, (w - 110, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # 4. 하단 안내 텍스트 패널
    draw_translucent_rect(frame, 0, h - 25, w, 25, color=(10, 10, 10), alpha=0.5)
    cv2.putText(frame, "Press 'Q' to Exit", (15, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    cv2.imshow('RPS Arena', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
