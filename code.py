
import cv2
import mediapipe as mp
import streamlit as st

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="VisionAI | Face & Hand Tracking",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #080d18;
    color: #f1f5f9;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: #101827;
    border-right: 1px solid #263449;
}

.hero {
    background: linear-gradient(120deg, #17243b, #102c38);
    border: 1px solid #2b4657;
    border-radius: 20px;
    padding: 25px 28px;
    margin-bottom: 22px;
}

.eyebrow {
    color: #5eead4;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero h1 {
    color: #f8fafc;
    font-size: 34px;
    margin: 8px 0;
    font-weight: 800;
}

.hero p {
    color: #a9b9ca;
    margin-bottom: 0;
    font-size: 14px;
}

.metric {
    background: #111b2b;
    border: 1px solid #263449;
    border-radius: 15px;
    padding: 17px;
    min-height: 100px;
}

.metric-label {
    color: #93a4b8;
    font-size: 12px;
    font-weight: 600;
}

.metric-value {
    color: #5eead4;
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

.section-title {
    font-size: 18px;
    font-weight: 750;
    color: #f1f5f9;
    margin: 12px 0;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #263449;
    border-radius: 15px;
}

.stButton button {
    border-radius: 10px;
    font-weight: 700;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------------- MEDIAPIPE ----------------
mp_face = mp.solutions.face_mesh
mp_hand = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class VisionProcessor(VideoProcessorBase):

    def __init__(self, show_face=True, show_hands=True):
        self.show_face = show_face
        self.show_hands = show_hands

        self.face_mesh = mp_face.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.hands = mp_hand.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.show_face:
            face_result = self.face_mesh.process(rgb)

            if face_result.multi_face_landmarks:
                for landmarks in face_result.multi_face_landmarks:
                    mp_draw.draw_landmarks(
                        image,
                        landmarks,
                        mp_face.FACEMESH_TESSELATION,
                        mp_draw.DrawingSpec(
                            color=(255, 255, 0),
                            thickness=1,
                            circle_radius=0
                        ),
                        mp_draw.DrawingSpec(
                            color=(255, 255, 0),
                            thickness=1,
                            circle_radius=0
                        )
                    )

        if self.show_hands:
            hand_result = self.hands.process(rgb)

            if hand_result.multi_hand_landmarks:
                for landmarks in hand_result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        image,
                        landmarks,
                        mp_hand.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(
                            color=(255, 80, 0),
                            thickness=3,
                            circle_radius=2
                        ),
                        mp_draw.DrawingSpec(
                            color=(255, 255, 0),
                            thickness=3,
                            circle_radius=3
                        )
                    )

        return frame.from_ndarray(image, format="bgr24")

    def __del__(self):
        try:
            self.face_mesh.close()
            self.hands.close()
        except Exception:
            pass


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 👁️ VisionAI")
    st.caption("REAL-TIME COMPUTER VISION")
    st.divider()

    st.markdown("### Tracking Controls")

    show_face = st.toggle("Face Mesh", value=True)
    show_hands = st.toggle("Hand Tracking", value=True)

    st.divider()

    st.markdown("### Landmark Colors")
    st.markdown("🟡 **Face:** Cyan / Yellow")
    st.markdown("🔵 **Hands:** Blue / Cyan")

    st.divider()

    st.caption("Powered by OpenCV + MediaPipe")
    st.caption("Camera permission is required.")


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">COMPUTER VISION STUDIO</div>
    <h1>Face Mesh & Hand Tracking</h1>
    <p>Real-time facial landmarks and hand skeleton detection in your browser.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- STATUS CARDS ----------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric">
        <div class="metric-label">FACE MESH</div>
        <div class="metric-value">468+</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric">
        <div class="metric-label">HAND LANDMARKS</div>
        <div class="metric-value">21 / hand</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric">
        <div class="metric-label">ACTIVE MODULES</div>
        <div class="metric-value">{int(show_face) + int(show_hands)} / 2</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- CAMERA SECTION ----------------
st.markdown('<div class="section-title">📷 Live Camera Studio</div>',
            unsafe_allow_html=True)

if not show_face and not show_hands:
    st.info("Enable Face Mesh or Hand Tracking from the sidebar.")

st.caption(
    "Click START to enable your webcam. Allow camera access when your browser asks."
)

webrtc_streamer(
    key="visionai-camera",
    video_processor_factory=lambda: VisionProcessor(
        show_face=show_face,
        show_hands=show_hands
    ),
    media_stream_constraints={
        "video": {
            "width": {"ideal": 960},
            "height": {"ideal": 540},
            "frameRate": {"ideal": 20, "max": 30}
        },
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    async_processing=True
)

st.divider()

st.markdown("""
<div style="text-align:center;color:#8393a8;font-size:12px;padding:8px">
    VisionAI · Real-Time Face & Hand Tracking · Built with Python
</div>
""", unsafe_allow_html=True)
