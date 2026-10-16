import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

face = mp.solutions.face_mesh
hand = mp.solutions.hands
Draw = mp.solutions.drawing_utils

# Face Mesh Model
aaa = face.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Hand Detection Model
bbb = hand.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

while True:
    ret, frame = cap.read()
    # Apne screen ka resolution daalein, jaise 1280x720 ya 1920x1080
    frame = cv2.resize(frame, (1380, 720))

    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process Face and Hands
    res = aaa.process(img)
    result = bbb.process(img)

    # Face Mesh - Cyan
    if res.multi_face_landmarks:
        for x in res.multi_face_landmarks:
            Draw.draw_landmarks(
                frame,
                x,
                face.FACEMESH_TESSELATION,
                Draw.DrawingSpec(
                    color=(255, 255, 0),
                    thickness=1,
                    circle_radius=0
                ),
                Draw.DrawingSpec(
                    color=(255, 255, 0),
                    thickness=1,
                    circle_radius=0
                )
            )

    # Hands - Electric Blue Lines + Cyan Points
    if result.multi_hand_landmarks:
        for i in result.multi_hand_landmarks:
            Draw.draw_landmarks(
                frame,
                i,
                hand.HAND_CONNECTIONS,
                Draw.DrawingSpec(
                    color=(255, 80, 0),
                    thickness=5,
                    circle_radius=2
                ),
                Draw.DrawingSpec(
                    color=(255, 255, 0),
                    thickness=5,
                    circle_radius=5
                )
            )

    # Display Webcam
    cv2.imshow("Face Mesh + Hand Tracking", frame)

    # Press Q to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
aaa.close()
bbb.close()
cv2.destroyAllWindows()

