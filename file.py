# # app.py
# import streamlit as st
# import cv2
# import tempfile
# from ultralytics import YOLO

# # Load YOLO model
# model = YOLO("yolov8n.pt")

# st.title("👥 Crowd Density & Stampede Prevention Demo")

# # Upload video
# uploaded_file = st.file_uploader("📂 Upload a video", type=["mp4", "avi", "mov","mkv"])

# THRESHOLD = st.slider("⚠️ Safety Threshold", 1, 100, 10)

# if uploaded_file is not None:
#     # Save temp file
#     tfile = tempfile.NamedTemporaryFile(delete=False)
#     tfile.write(uploaded_file.read())

#     cap = cv2.VideoCapture(tfile.name)
#     stframe = st.empty()

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model(frame, verbose=False)

#         # Count people
#         count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)

#         # Draw boxes
#         for box in results[0].boxes:
#             if int(box.cls[0]) == 0:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         # Overlay text
#         cv2.putText(frame, f"Count: {count}", (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         if count > THRESHOLD:
#             st.error(f"⚠️ ALERT: Crowd exceeded! ({count} people)")
#         else:
#             st.success(f"✅ Safe: {count} people")

#         # Show frame in Streamlit
#         stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

#     cap.release()
