import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)

draw = mp.solutions.drawing_utils
face = mp.solutions.face_mesh

mp_face = face.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

while True:
    ret, frame = video.read()

    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_face.process(img)

    if result.multi_face_landmarks:
        for aa in result.multi_face_landmarks:

            draw.draw_landmarks(
                frame,
                aa,
                face.FACEMESH_CONTOURS,
                draw.DrawingSpec(
                    color=(255, 255, 255),
                    thickness=1,
                    circle_radius=1
                ),
                draw.DrawingSpec(
                    color=(255, 255, 255),
                    thickness=1
                )
            )

    cv2.imshow("White Face Mesh", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
mp_face.close()
cv2.destroyAllWindows()

