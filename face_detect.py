import cv2
import mediapipe as mp

# Initialize FaceMesh
mp_face_mesh = mp.tasks.vision
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create options
options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path='face_landmarker.task'
    ),
    running_mode=VisionRunningMode.IMAGE
)

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.putText(frame,
                "Webcam Working",
                (20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()