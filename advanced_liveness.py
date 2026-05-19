import cv2
import mediapipe as mp
from scipy.spatial import distance
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def calculate_EAR(eye_points):

    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])

    ear = (A + B) / (2.0 * C)

    return ear

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


cap = cv2.VideoCapture(0)

blink_count = 0
blink_frames = 0
blink_detected = False

left_movement = False
right_movement = False

smile_detected = False
smile_frames = 0

challenge_completed = False
challenge_stage = 0

center_nose_x = None

face_missing_frames = 0
MAX_MISSING_FRAMES = 15

# -------- ANTI SPOOF --------
previous_face_width = None
face_width_changes = 0
real_head_rotation = False
real_face_depth = False



LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

NOSE_TIP = 1

LEFT_MOUTH = 61
RIGHT_MOUTH = 291
TOP_MOUTH = 13
BOTTOM_MOUTH = 14

LEFT_FACE = 234
RIGHT_FACE = 454

challenge_texts = [
    "BLINK 2 TIMES",
    "TURN HEAD LEFT",
    "TURN HEAD RIGHT",
    "SMILE",
    "VERIFICATION COMPLETE"
]

def reset_system():

    global blink_count
    global blink_frames
    global blink_detected

    global left_movement
    global right_movement

    global smile_detected
    global smile_frames

    global challenge_completed
    global challenge_stage

    global center_nose_x

    global previous_face_width
    global face_width_changes
    global real_head_rotation
    global real_face_depth

    blink_count = 0
    blink_frames = 0
    blink_detected = False

    left_movement = False
    right_movement = False

    smile_detected = False
    smile_frames = 0

    challenge_completed = False
    challenge_stage = 0

    center_nose_x = None

    previous_face_width = None
    face_width_changes = 0
    real_head_rotation = False
    real_face_depth = False


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

    if detection_result.face_landmarks:

        face_missing_frames = 0

        for face_landmarks in detection_result.face_landmarks:


            left_eye_points = []

            for idx in LEFT_EYE:

                landmark = face_landmarks[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                left_eye_points.append((x, y))

                cv2.circle(frame, (x, y), 2, (0,255,0), -1)

            right_eye_points = []

            for idx in RIGHT_EYE:

                landmark = face_landmarks[idx]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                right_eye_points.append((x, y))

                cv2.circle(frame, (x, y), 2, (0,255,0), -1)

            left_ear = calculate_EAR(left_eye_points)
            right_ear = calculate_EAR(right_eye_points)

            ear = (left_ear + right_ear) / 2.0

            if ear < 0.22:

                blink_frames += 1

            else:

                if blink_frames >= 2 and not blink_detected:

                    blink_count += 1
                    blink_detected = True

                blink_frames = 0

            # Reset blink state
            if ear > 0.26:
                blink_detected = False

            left_eye_width = distance.euclidean(
                left_eye_points[0],
                left_eye_points[3]
            )

            right_eye_width = distance.euclidean(
                right_eye_points[0],
                right_eye_points[3]
            )

            eye_ratio = left_eye_width / (right_eye_width + 1)

            cv2.putText(frame,
                        f"Eye Ratio: {eye_ratio:.2f}",
                        (20,320),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255,255,255),
                        2)

            # Real head turn changes perspective
            if eye_ratio > 1.12 or eye_ratio < 0.88:
                real_face_depth = True


            nose = face_landmarks[NOSE_TIP]

            nose_x = int(nose.x * w)

            if center_nose_x is None:
                center_nose_x = nose_x

            # LEFT MOVEMENT
            if nose_x < center_nose_x - 18:
                left_movement = True

            # RIGHT MOVEMENT
            if nose_x > center_nose_x + 18:
                right_movement = True


            left_face = face_landmarks[LEFT_FACE]
            right_face = face_landmarks[RIGHT_FACE]

            left_x = int(left_face.x * w)
            right_x = int(right_face.x * w)

            face_width = abs(right_x - left_x)

            cv2.circle(frame,
                       (left_x, int(left_face.y*h)),
                       3,
                       (255,255,0),
                       -1)

            cv2.circle(frame,
                       (right_x, int(right_face.y*h)),
                       3,
                       (255,255,0),
                       -1)

            # Reject tiny/far faces
            if face_width < 120:

                cv2.putText(frame,
                            "FACE TOO FAR / POSSIBLE SPOOF",
                            (20,520),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,0,255),
                            3)

                continue

            if previous_face_width is not None:

                width_change = abs(
                    face_width - previous_face_width
                )

                # Real head rotation changes geometry
                if width_change > 8:
                    face_width_changes += 1

            previous_face_width = face_width

            # CONFIRM REAL HEAD ROTATION
            if face_width_changes >= 2:
                real_head_rotation = True

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

            cv2.putText(frame,
                        f"Smile Ratio: {smile_ratio:.2f}",
                        (20,240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255,255,255),
                        2)

            if smile_ratio > 4.5:

                smile_frames += 1

            else:

                smile_frames = 0

            if smile_frames >= 5:
                smile_detected = True


            # BLINK
            if challenge_stage == 0:

                if blink_count >= 2:
                    challenge_stage = 1

            # LEFT TURN
            elif challenge_stage == 1:

                if (
                    left_movement
                    and real_head_rotation
                    and real_face_depth
                ):
                    challenge_stage = 2

            # RIGHT TURN
            elif challenge_stage == 2:

                if (
                    right_movement
                    and real_head_rotation
                    and real_face_depth
                ):
                    challenge_stage = 3

            # SMILE
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
                            "LEFT HEAD TURN DETECTED",
                            (20,130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255,255,0),
                            2)

            if right_movement:

                cv2.putText(frame,
                            "RIGHT HEAD TURN DETECTED",
                            (20,170),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255,255,0),
                            2)

            # =================================================
            # DISPLAY ROTATION VALIDATION
            # =================================================
            if real_head_rotation and real_face_depth:

                cv2.putText(frame,
                            "REAL 3D HEAD ROTATION",
                            (20,210),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,255,255),
                            2)

            else:

                cv2.putText(frame,
                            "FLAT IMAGE / PHONE DETECTED",
                            (20,210),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,0,255),
                            2)

            # =================================================
            # DISPLAY SMILE
            # =================================================
            if smile_detected:

                cv2.putText(frame,
                            "SMILE DETECTED",
                            (20,280),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255,0,255),
                            2)

            else:

                cv2.putText(frame,
                            "SMILE NOT DETECTED",
                            (20,280),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,0,255),
                            2)

            # =================================================
            # CURRENT CHALLENGE
            # =================================================
            cv2.putText(frame,
                        challenge_texts[challenge_stage],
                        (20,380),
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
                            (20,460),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2,
                            (0,255,0),
                            4)

            else:

                cv2.putText(frame,
                            "LIVENESS CHECK RUNNING",
                            (20,460),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,0,255),
                            3)

    # =====================================================
    # NO FACE
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


    cv2.imshow("FINAL AI LIVENESS DETECTION SYSTEM", frame)

    # =====================================================
    # QUIT
    # =====================================================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================================================
# CLEANUP
# =====================================================
cap.release()
cv2.destroyAllWindows()