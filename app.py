import cv2
from ultralytics import YOLO
import easyocr
import pandas as pd
from datetime import datetime

model_path = "best.pt"
model = YOLO(model_path)

reader = easyocr.Reader(['en'])

video_path = "tc.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

mini_screen_scale = 0.4

results_data = []

excel_file = "detected_text_results.xlsx"
df = pd.DataFrame(columns=['Frame', 'BoundingBox', 'ExtractedText', 'Timestamp'])
df.to_excel(excel_file, index=False)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
        cropped_region = frame[y1:y2, x1:x2]
        text = reader.readtext(cropped_region, detail=0)
        extracted_text = ' '.join(text) if text else "No Text Found"

        results_data.append({
            'Frame': frame_count,
            'BoundingBox': f'({x1}, {y1}), ({x2}, {y2})',
            'ExtractedText': extracted_text,
            'Timestamp': current_time
        })

        font_scale = 1.0
        font_thickness = 2
        text_size, _ = cv2.getTextSize(extracted_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_width, text_height = text_size
        background_top_left = (x1, y1 - text_height - 10)
        background_bottom_right = (x1 + text_width, y1)

        cv2.rectangle(frame, background_top_left, background_bottom_right, (0, 0, 0), -1)
        cv2.putText(frame, extracted_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    date_time_text = f"Date & Time: {current_time}"
    dt_font_scale = 1.2
    dt_text_size, _ = cv2.getTextSize(date_time_text, cv2.FONT_HERSHEY_SIMPLEX, dt_font_scale, 2)
    dt_background_top_left = (10, 30)
    dt_background_bottom_right = (10 + dt_text_size[0], 30 - dt_text_size[1] - 10)

    cv2.rectangle(frame, dt_background_top_left, dt_background_bottom_right, (0, 0, 0), -1)
    cv2.putText(frame, date_time_text, (10, 30 - 5), cv2.FONT_HERSHEY_SIMPLEX, dt_font_scale, (255, 255, 255), 2)

    mini_frame = cv2.resize(frame, None, fx=mini_screen_scale, fy=mini_screen_scale)
    cv2.imshow("ANPR Inference with OCR (Mini Screen)", mini_frame)

    if results_data:
        temp_df = pd.DataFrame(results_data)
        temp_df.to_excel(excel_file, index=False, header=False, startrow=len(temp_df)-len(results_data))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Results saved to detected_text_results.xlsx")
