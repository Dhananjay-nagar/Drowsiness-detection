import cv2
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance
from datetime import datetime


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 20
lStart, lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
rStart, rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
cap = cv2.VideoCapture(0)

sleep_start_time = None
awake_start_time = datetime.now()
sleep_detected = False

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detector(gray, 0)
    
    for subject in subjects:
        shape = predictor(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEar = eye_aspect_ratio(leftEye)
        rightEar = eye_aspect_ratio(rightEye)
        ear = (leftEar + rightEar) / 2.0
        
        if ear < thresh:
            sleep_start_time = sleep_start_time or datetime.now()
            awake_start_time = None
        else:
            if sleep_start_time:
                sleep_detected = True
                sleep_start_time = None
                awake_start_time = awake_start_time or datetime.now()
                
        if awake_start_time:
            awake_duration = datetime.now() - awake_start_time
            if awake_duration.total_seconds() >= 1:  # Reduced duration to switch to AWAKE state
                awake_start_time = None
                sleep_detected = False  # Reset sleep detection when eyes are open
    
    if sleep_detected:
        cv2.putText(frame, "SLEEP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "AWAKE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()