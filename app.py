# # demo.py
# import cv2
# from ultralytics import YOLO

# # Load YOLOv8 pre-trained model
# model = YOLO("yolov8n.pt")  # Small model, detects 80 classes including 'person'

# # Open video (0 = webcam, or give path like "crowd.mp4")
# cap = cv2.VideoCapture(0)  # change to "sample.mp4" if you have a crowd video

# # Set safety threshold
# THRESHOLD = 10  

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run YOLO detection
#     results = model(frame, verbose=False)

#     # Count persons (class 0 = person in COCO dataset)
#     count = 0
#     for box in results[0].boxes:
#         if int(box.cls[0]) == 0:
#             count += 1
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     # Show count on frame
#     cv2.putText(frame, f"People Count: {count}", (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     # Show alert
#     if count > THRESHOLD:
#         print(f"⚠️ ALERT: Crowd exceeded! Count = {count}")
#         cv2.putText(frame, "ALERT: CROWD LIMIT EXCEEDED!", (20, 80),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#     else:
#         print(f"✅ Safe: Current crowd = {count}")

#     # Show video
#     cv2.imshow("Crowd Detection Demo", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()


# app.py - HOG People Detector + Streamlit upload demo
import streamlit as st
import cv2
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

st.set_page_config(layout="wide", page_title="Crowd Density Demo")
st.title("👥 Crowd Density & Stampede Prevention (HOG demo)")

uploaded_file = st.file_uploader("Upload a crowd video (mp4/mov/avi/mkv)", type=["mp4","mov","avi","mkv"])
threshold = st.slider("Safety threshold (people)", 1, 500, 30)
show_heatmap = st.checkbox("Show heatmap overlay", value=True)

# Initialize HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_people_hog(frame):
    # Resize for speed
    h0, w0 = frame.shape[:2]
    scale = 640 / max(w0, h0)
    if scale < 1:
        frame_small = cv2.resize(frame, (int(w0*scale), int(h0*scale)))
    else:
        frame_small = frame.copy()
    rects, _ = hog.detectMultiScale(frame_small, winStride=(8,8), padding=(8,8), scale=1.05)
    # scale rects back
    if scale < 1:
        rects = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for (x,y,w,h) in rects]
    return rects

def make_heatmap(count_map, frame):
    # count_map: same size as frame but lower res; we will upsample
    heat = cv2.resize(count_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat_norm = np.uint8(255 * (heat / (heat.max()+1e-6)))
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heat_color, 0.4, 0)
    return overlay

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    counts = []
    timestamps = []
    # grid for heatmap (small)
    grid_h, grid_w = 20, 30
    count_grid = np.zeros((grid_h, grid_w), dtype=np.float32)

    stframe = st.empty()
    chart = st.empty()

    frame_no = 0
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS)>0 else 20

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        rects = detect_people_hog(frame)
        count = len(rects)
        counts.append(count)
        timestamps.append(frame_no / fps)

        # update grid counts
        for (x,y,w,h) in rects:
            cx = x + w//2
            cy = y + h//2
            gi = int((cy / frame.shape[0]) * grid_h)
            gj = int((cx / frame.shape[1]) * grid_w)
            gi = min(max(gi,0),grid_h-1); gj = min(max(gj,0),grid_w-1)
            count_grid[gi, gj] += 1

        # draw boxes
        for (x,y,w,h) in rects:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"People: {count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # safety alert
        if count > threshold:
            st.warning(f"⚠️ ALERT: Crowd exceeded threshold! ({count})")

        display_frame = frame.copy()
        if show_heatmap:
            display_frame = make_heatmap(count_grid, display_frame)

        stframe.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # update chart every few frames
        if frame_no % int(fps*1) == 0:
            fig, ax = plt.subplots()
            ax.plot(timestamps, counts, label="People Count")
            ax.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Count"); ax.legend()
            chart.pyplot(fig)

    cap.release()
    st.success("Processing finished. Use Download link to save logs (optional).")
