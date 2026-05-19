import cv2
import mediapipe as mp
from scipy.spatial import distance
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================================================
# EAR FUNCTION
# =========================================================
def calculate_EAR(eye_points):

    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])

    ear = (A + B) / (2.0 * C)

    return ear

# =========================================================
# LOAD MODEL
# =========================================================
base_options = python.BaseOptions(
    model_asset_path='face_landmarker.task'
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

# =========================================================
# WEBCAM
# =========================================================
cap = cv2.VideoCapture(0)

# =========================================================
# VARIABLES
# =========================================================
blink_count = 0
blink_frames = 0
blink_detected = False

left_movement = False
right_movement = False

smile_detected = False

center_nose_x = None

face_missing_frames = 0
MAX_MISSING_FRAMES = 15

challenge_completed = False
challenge_stage = 0

# =========================================================
# LANDMARKS
# =========================================================

# Left Eye
LEFT_EYE = [33, 160, 158, 133, 153, 144]

# Right Eye
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Nose
NOSE_TIP = 1

# Mouth
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
TOP_MOUTH = 13
BOTTOM_MOUTH = 14

# =========================================================
# CHALLENGE TEXT
# =========================================================
challenge_texts = [
    "BLINK 2 TIMES",
    "TURN HEAD LEFT",
    "TURN HEAD RIGHT",
    "SMILE",
    "VERIFICATION COMPLETE"
]

# =========================================================
# RESET FUNCTION
# =========================================================
def reset_system():

    global blink_count
    global blink_frames
    global blink_detected

    global left_movement
    global right_movement

    global smile_detected

    global center_nose_x

    global challenge_completed
    global challenge_stage

    blink_count = 0
    blink_frames = 0
    blink_detected = False

    left_movement = False
    right_movement = False

    smile_detected = False

    center_nose_x = None

    challenge_completed = False
    challenge_stage = 0

# =========================================================
# MAIN LOOP
# =========================================================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    detection_result = detector.detect(mp_image)

    # =====================================================
    # FACE DETECTED
    # =====================================================
    if detection_result.face_landmarks:

        face_missing_frames = 0

        for face_landmarks in detection_result.face_landmarks:

            # =================================================
            # LEFT EYE
            # =================================================
            left_eye_points = []

            for idx in LEFT_EYE:

                landmark = face_landmarks[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                left_eye_points.append((x, y))

                cv2.circle(frame, (x, y), 2, (0,255,0), -1)

            # =================================================
            # RIGHT EYE
            # =================================================
            right_eye_points = []

            for idx in RIGHT_EYE:

                landmark = face_landmarks[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                right_eye_points.append((x, y))

                cv2.circle(frame, (x, y), 2, (0,255,0), -1)

            # =================================================
            # EAR CALCULATION
            # =================================================
            left_ear = calculate_EAR(left_eye_points)
            right_ear = calculate_EAR(right_eye_points)

            ear = (left_ear + right_ear) / 2.0

            # =================================================
            # BETTER BLINK DETECTION
            # =================================================
            if ear < 0.23:

                blink_frames += 1

            else:

                if blink_frames >= 2 and not blink_detected:

                    blink_count += 1
                    blink_detected = True

                blink_frames = 0

            # Reset blink state
            if ear > 0.26:
                blink_detected = False

            # =================================================
            # HEAD MOVEMENT
            # =================================================
            nose = face_landmarks[NOSE_TIP]

            nose_x = int(nose.x * w)

            if center_nose_x is None:
                center_nose_x = nose_x

            # LEFT MOVEMENT
            if nose_x < center_nose_x - 30:
                left_movement = True

            # RIGHT MOVEMENT
            if nose_x > center_nose_x + 30:
                right_movement = True

            # =================================================
            # SMILE DETECTION
            # =================================================
            left_mouth = face_landmarks[LEFT_MOUTH]
            right_mouth = face_landmarks[RIGHT_MOUTH]
            top_mouth = face_landmarks[TOP_MOUTH]
            bottom_mouth = face_landmarks[BOTTOM_MOUTH]

            mouth_width = distance.euclidean(
                (left_mouth.x*w, left_mouth.y*h),
                (right_mouth.x*w, right_mouth.y*h)
            )

            mouth_height = distance.euclidean(
                (top_mouth.x*w, top_mouth.y*h),
                (bottom_mouth.x*w, bottom_mouth.y*h)
            )

            smile_ratio = mouth_width / (mouth_height + 1)

            if smile_ratio > 3.5:
                smile_detected = True

            # =================================================
            # CHALLENGE SYSTEM
            # =================================================

            # Blink challenge
            if challenge_stage == 0:

                if blink_count >= 2:
                    challenge_stage = 1

            # Left movement
            elif challenge_stage == 1:

                if left_movement:
                    challenge_stage = 2

            # Right movement
            elif challenge_stage == 2:

                if right_movement:
                    challenge_stage = 3

            # Smile
            elif challenge_stage == 3:

                if smile_detected:
                    challenge_stage = 4
                    challenge_completed = True

            # =================================================
            # DISPLAY EAR
            # =================================================
            cv2.putText(frame,
                        f"EAR: {ear:.2f}",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255,0,0),
                        2)

            # =================================================
            # DISPLAY BLINKS
            # =================================================
            cv2.putText(frame,
                        f"Blinks: {blink_count}",
                        (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,255,0),
                        2)

            # =================================================
            # DISPLAY MOVEMENTS
            # =================================================
            if left_movement:

                cv2.putText(frame,
                            "LEFT MOVEMENT DETECTED",
                            (20,130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255,255,0),
                            2)

            if right_movement:

                cv2.putText(frame,
                            "RIGHT MOVEMENT DETECTED",
                            (20,170),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255,255,0),
                            2)

            # =================================================
            # DISPLAY SMILE
            # =================================================
            if smile_detected:

                cv2.putText(frame,
                            "SMILE DETECTED",
                            (20,210),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255,0,255),
                            2)

            # =================================================
            # DISPLAY CURRENT CHALLENGE
            # =================================================
            cv2.putText(frame,
                        challenge_texts[challenge_stage],
                        (20,280),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,165,255),
                        3)

            # =================================================
            # FINAL STATUS
            # =================================================
            if challenge_completed:

                cv2.putText(frame,
                            "REAL HUMAN VERIFIED",
                            (20,350),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.1,
                            (0,255,0),
                            3)

            else:

                cv2.putText(frame,
                            "LIVENESS CHECK RUNNING",
                            (20,350),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,0,255),
                            3)

    # =====================================================
    # NO FACE DETECTED
    # =====================================================
    else:

        face_missing_frames += 1

        if face_missing_frames > MAX_MISSING_FRAMES:

            reset_system()

        cv2.putText(frame,
                    "NO FACE DETECTED",
                    (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

    # =====================================================
    # SHOW WINDOW
    # =====================================================
    cv2.imshow("Advanced AI Liveness Detection", frame)

    # =====================================================
    # QUIT
    # =====================================================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================================================
# CLEANUP
# =========================================================
cap.release()
cv2.destroyAllWindows()