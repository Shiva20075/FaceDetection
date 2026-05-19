import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load model
base_options = python.BaseOptions(
    model_asset_path='face_landmarker.task'
)

# Configure face landmarker
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

# Create detector
detector = vision.FaceLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect landmarks
    detection_result = detector.detect(mp_image)

    # If face detected
    if detection_result.face_landmarks:

        for face_landmarks in detection_result.face_landmarks:

            h, w, _ = frame.shape

            for landmark in face_landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(frame, (x, y), 1, (0,255,0), -1)

        cv2.putText(frame,
                    "Face Detected",
                    (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

    else:

        cv2.putText(frame,
                    "No Face Detected",
                    (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2)

    cv2.imshow("Face Mesh", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()