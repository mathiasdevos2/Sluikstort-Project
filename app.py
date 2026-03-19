import os
import threading
from datetime import datetime

import av
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "detections.csv"
CAPTURED_DIR = "captured"

MODEL1_DIR = "models/afval_detector"
MODEL1_PATH = os.path.join(MODEL1_DIR, "keras_model.h5")
MODEL1_LABELS_PATH = os.path.join(MODEL1_DIR, "labels.txt")

MODEL2_DIR = "models/afval_type_detector"
MODEL2_PATH = os.path.join(MODEL2_DIR, "keras_model.h5")
MODEL2_LABELS_PATH = os.path.join(MODEL2_DIR, "labels.txt")

CONFIDENCE_THRESHOLD = 0.80
DEFAULT_INTERVAL_SECONDS = 2
AUTO_SAVE_COOLDOWN_SECONDS = 10

RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

EXPECTED_COLUMNS = [
    "timestamp",
    "label",
    "confidence",
    "image_path",
    "quantity",
    "waste_type",
    "source",
]

st.set_page_config(
    page_title="Sluikstort Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

os.makedirs(CAPTURED_DIR, exist_ok=True)

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
        color: #f9fafb;
    }

    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: #f9fafb;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        color: #ffffff;
    }

    .sub-title {
        color: #cbd5e1;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }

    .card {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 18px 20px;
        border-radius: 18px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        margin-bottom: 14px;
    }

    .kpi-label {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin-bottom: 8px;
    }

    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
    }

    .kpi-small {
        margin-top: 8px;
        font-size: 0.9rem;
        color: #93c5fd;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        color: #ffffff;
    }

    .info-box {
        background: rgba(37, 99, 235, 0.15);
        border: 1px solid rgba(96, 165, 250, 0.35);
        padding: 14px 16px;
        border-radius: 14px;
        color: #dbeafe;
        margin-bottom: 14px;
    }

    .result-box {
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid rgba(52, 211, 153, 0.35);
        padding: 14px 16px;
        border-radius: 14px;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .warning-box {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(251, 191, 36, 0.35);
        padding: 14px 16px;
        border-radius: 14px;
        margin-top: 12px;
        margin-bottom: 12px;
        color: #fde68a;
    }

    .danger-box {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(248, 113, 113, 0.35);
        padding: 14px 16px;
        border-radius: 14px;
        margin-top: 12px;
        margin-bottom: 12px;
        color: #fecaca;
    }

    div[data-testid="stDataFrame"] {
        background: rgba(30, 41, 59, 0.85);
        border-radius: 14px;
        padding: 6px;
    }

    .stButton > button {
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #1e40af);
        color: white;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stDateInput > div,
    .stMultiSelect > div,
    .stNumberInput > div {
        background-color: rgba(30, 41, 59, 0.9) !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# STREAM CLASS
# =========================================================
class LiveVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_frame = None
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self.lock:
            self.latest_frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =========================================================
# STATE INIT
# =========================================================
DEFAULT_STATE = {
    "combined_result": None,
    "live_running": False,
    "live_result": None,
    "live_frame": None,
    "last_auto_save_time": None,
    "last_auto_save_message": "",
    "snapshot_result": None,
    "snapshot_frame": None,
    "stream_result": None,
    "stream_frame": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================================================
# HELPERS
# =========================================================
def section_start(title: str) -> None:
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )


def kpi_card(title: str, value: str, extra: str = "") -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-small">{extra}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def format_confidence(series: pd.Series) -> pd.Series:
    return (series * 100).round(1).astype(str) + "%"


def ensure_data_file() -> None:
    if not os.path.exists(DATA_PATH):
        pd.DataFrame(columns=EXPECTED_COLUMNS).to_csv(DATA_PATH, index=False)


def load_data() -> pd.DataFrame:
    ensure_data_file()
    df = pd.read_csv(DATA_PATH)

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            if col == "waste_type":
                df[col] = "onbekend"
            elif col == "source":
                df[col] = "simulatie"
            else:
                df[col] = None

    return df[EXPECTED_COLUMNS]


def save_data(df: pd.DataFrame) -> None:
    df = df[EXPECTED_COLUMNS]
    df.to_csv(DATA_PATH, index=False)


@st.cache_resource
def load_teachable_model(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


@st.cache_data
def load_labels(label_path: str) -> list[str]:
    """
    Leest labels correct in uit labels.txt.
    Voorbeeld:
    0 glazen fles
    1 plastic zak
    =>
    ['glazen fles', 'plastic zak']
    """
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    labels = []
    for line in lines:
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            labels.append(parts[1].strip())
        else:
            labels.append(line.strip())

    return labels


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray | None:
    if frame_bgr is None:
        return None

    image = cv2.resize(frame_bgr, (224, 224))
    image = image.astype(np.float32)
    image = (image / 127.5) - 1
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_image_path(image_path: str) -> np.ndarray | None:
    frame = cv2.imread(image_path)
    if frame is None:
        return None
    return preprocess_frame(frame)


def predict_with_model(model, labels: list[str], image_path: str):
    data = preprocess_image_path(image_path)
    if data is None:
        return None, None, None

    prediction = model.predict(data, verbose=0)[0]
    index = int(np.argmax(prediction))
    label = labels[index]
    confidence = float(prediction[index])

    result_df = pd.DataFrame({
        "label": labels,
        "score": prediction
    }).sort_values("score", ascending=False)

    return label, confidence, result_df


def predict_with_model_from_frame(model, labels: list[str], frame_bgr: np.ndarray):
    data = preprocess_frame(frame_bgr)
    if data is None:
        return None, None, None

    prediction = model.predict(data, verbose=0)[0]
    index = int(np.argmax(prediction))
    label = labels[index]
    confidence = float(prediction[index])

    result_df = pd.DataFrame({
        "label": labels,
        "score": prediction
    }).sort_values("score", ascending=False)

    return label, confidence, result_df


def get_images_from_captured() -> list[str]:
    if not os.path.exists(CAPTURED_DIR):
        return []
    return [
        f for f in sorted(os.listdir(CAPTURED_DIR))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def is_positive_sluikstort(label: str) -> bool:
    return str(label).strip().lower() == "sluikstort"


def save_ai_detection(
    image_path: str,
    label: str,
    confidence: float,
    waste_type: str,
    quantity: int = 1,
    source: str = "ai_combined",
) -> None:
    df = load_data()

    new_row = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": label,
        "confidence": confidence,
        "image_path": image_path,
        "quantity": quantity,
        "waste_type": waste_type,
        "source": source,
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    save_data(df)


def open_local_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return cap


def capture_webcam_frame():
    cap = open_local_camera()

    if not cap.isOpened():
        return None, "Kon de webcam niet openen."

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None, "Kon geen frame van de webcam lezen."

    return frame, None


def save_frame_to_captured(frame_bgr: np.ndarray, prefix: str = "webcam") -> str:
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join(CAPTURED_DIR, filename)
    cv2.imwrite(path, frame_bgr)
    return path


def analyze_combined_from_frame(frame_bgr: np.ndarray, quantity: int) -> dict:
    empty_result = {
        "label1": None,
        "confidence1": None,
        "result_df1": None,
        "label2": None,
        "confidence2": None,
        "result_df2": None,
        "can_save": False,
        "quantity": quantity,
    }

    if frame_bgr is None:
        return empty_result

    model1 = load_teachable_model(MODEL1_PATH)
    labels1 = load_labels(MODEL1_LABELS_PATH)

    label1, confidence1, result_df1 = predict_with_model_from_frame(model1, labels1, frame_bgr)

    result = {
        "label1": label1,
        "confidence1": confidence1,
        "result_df1": result_df1,
        "label2": None,
        "confidence2": None,
        "result_df2": None,
        "can_save": False,
        "quantity": quantity,
    }

    if label1 is not None and is_positive_sluikstort(label1) and confidence1 >= CONFIDENCE_THRESHOLD:
        model2 = load_teachable_model(MODEL2_PATH)
        labels2 = load_labels(MODEL2_LABELS_PATH)

        label2, confidence2, result_df2 = predict_with_model_from_frame(model2, labels2, frame_bgr)
        result["label2"] = label2
        result["confidence2"] = confidence2
        result["result_df2"] = result_df2
        result["can_save"] = True

    return result


def auto_save_if_needed(result: dict, frame_to_save: np.ndarray, quantity: int, source_prefix: str):
    if result is None or frame_to_save is None or not result["can_save"]:
        return False, ""

    now = datetime.now()
    last_save = st.session_state.last_auto_save_time

    if last_save is not None:
        elapsed = (now - last_save).total_seconds()
        if elapsed < AUTO_SAVE_COOLDOWN_SECONDS:
            seconds_left = AUTO_SAVE_COOLDOWN_SECONDS - int(elapsed)
            return False, f"Cooldown actief. Wacht nog ongeveer {seconds_left} sec."

    saved_path = save_frame_to_captured(frame_to_save, prefix=source_prefix)
    save_ai_detection(
        image_path=saved_path,
        label="sluikstort",
        confidence=result["confidence1"],
        waste_type=result["label2"] if result["label2"] else "onbekend",
        quantity=quantity,
        source=source_prefix,
    )
    st.session_state.last_auto_save_time = now
    return True, f"Automatisch opgeslagen om {now.strftime('%H:%M:%S')} | type afval: {result['label2']}"


def render_result_block(result: dict | None) -> None:
    if not result:
        st.info("Nog geen analyse uitgevoerd.")
        return

    if result["label1"] is None:
        st.markdown(
            """
            <div class="danger-box">
                Het frame kon niet correct geanalyseerd worden.
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    st.markdown(
        f"""
        <div class="result-box">
            <b>Model 1 voorspelling:</b> {result['label1']}<br>
            <b>Confidence model 1:</b> {result['confidence1']:.2%}
        </div>
        """,
        unsafe_allow_html=True
    )

    if result["can_save"]:
        st.markdown(
            f"""
            <div class="result-box">
                <b>Model 2 afvaltype:</b> {result['label2']}<br>
                <b>Confidence model 2:</b> {result['confidence2']:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        if not is_positive_sluikstort(result["label1"]):
            st.markdown(
                """
                <div class="warning-box">
                    Model 1 voorspelt geen sluikstort. Model 2 wordt niet uitgevoerd.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="warning-box">
                    Model 1 voorspelt wel sluikstort, maar de confidence ligt onder {int(CONFIDENCE_THRESHOLD * 100)}%.
                    Daarom wordt model 2 niet uitgevoerd.
                </div>
                """,
                unsafe_allow_html=True
            )

    if result["result_df1"] is not None:
        st.markdown("### Scores model 1")
        df1 = result["result_df1"].copy()
        df1["score"] = (df1["score"] * 100).round(2)
        st.dataframe(df1, use_container_width=True)

    if result["result_df2"] is not None:
        st.markdown("### Scores model 2")
        df2 = result["result_df2"].copy()
        df2["score"] = (df2["score"] * 100).round(2)
        st.dataframe(df2, use_container_width=True)


def prepare_filtered_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    return df


# =========================================================
# DATA
# =========================================================
df = prepare_filtered_data(load_data())

# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="main-title">♻️ Sluikstort Monitoring Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">Overzicht van detecties, trends, beelden en simulaties.</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Navigatie")
page = st.sidebar.radio(
    "Kies een pagina",
    [
        "Dashboard",
        "Historiek",
        "Simulatie",
        "AI Test sluikstort",
        "AI Test afvaltype",
        "Gecombineerde AI test",
        "Live Webcam",
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

if df.empty:
    selected_labels = []
    min_confidence = 0.0
    selected_dates = []
else:
    all_labels = sorted(df["label"].dropna().astype(str).unique().tolist())
    selected_labels = st.sidebar.multiselect("Labels", all_labels, default=all_labels)
    min_confidence = st.sidebar.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.01)
    all_dates = sorted(df["date"].dropna().unique().tolist())
    selected_dates = st.sidebar.multiselect("Datums", all_dates, default=all_dates)

filtered_df = df.copy()

if not filtered_df.empty:
    if selected_labels:
        filtered_df = filtered_df[filtered_df["label"].isin(selected_labels)]
    filtered_df = filtered_df[filtered_df["confidence"] >= min_confidence]
    if selected_dates:
        filtered_df = filtered_df[filtered_df["date"].isin(selected_dates)]

# =========================================================
# PAGES
# =========================================================
if page == "Dashboard":
    if filtered_df.empty:
        st.warning("Geen data beschikbaar voor de gekozen filters.")
    else:
        total_detections = len(filtered_df)
        avg_confidence = filtered_df["confidence"].mean()
        last_detection = filtered_df["timestamp"].max()
        total_quantity = filtered_df["quantity"].fillna(0).sum()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Totaal detecties", f"{total_detections}", "Aantal opgeslagen momenten")
        with c2:
            kpi_card("Gem. confidence", f"{avg_confidence:.2%}", "Gemiddelde betrouwbaarheid")
        with c3:
            kpi_card("Laatste detectie", last_detection.strftime("%d/%m/%Y %H:%M"), "Meest recente registratie")
        with c4:
            kpi_card("Totale hoeveelheid", f"{int(total_quantity)}", "Som van quantity")

        left, right = st.columns(2)

        with left:
            section_start("Detecties per dag")
            per_day = filtered_df.groupby("date").size().reset_index(name="aantal")
            per_day["date"] = per_day["date"].astype(str)
            fig_day = px.line(per_day, x="date", y="aantal", markers=True, template="plotly_dark")
            fig_day.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Datum",
                yaxis_title="Aantal detecties",
                height=350,
            )
            st.plotly_chart(fig_day, use_container_width=True)

        with right:
            section_start("Detecties per uur")
            per_hour = filtered_df.groupby("hour").size().reset_index(name="aantal")
            fig_hour = px.bar(per_hour, x="hour", y="aantal", template="plotly_dark")
            fig_hour.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis_title="Uur van de dag",
                yaxis_title="Aantal detecties",
                height=350,
            )
            st.plotly_chart(fig_hour, use_container_width=True)

        bottom_left, bottom_right = st.columns([1.35, 1])

        with bottom_left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            section_start("Recente detecties")
            cols_to_show = ["timestamp", "label", "confidence", "quantity", "waste_type", "source", "image_path"]
            show_df = filtered_df.sort_values("timestamp", ascending=False)[cols_to_show].copy()
            show_df["timestamp"] = show_df["timestamp"].dt.strftime("%d/%m/%Y %H:%M")
            show_df["confidence"] = format_confidence(show_df["confidence"])
            st.dataframe(show_df, use_container_width=True, height=320)
            st.markdown('</div>', unsafe_allow_html=True)

        with bottom_right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            section_start("Laatste 3 foto's")
            latest = filtered_df.sort_values("timestamp", ascending=False).head(3)

            if latest.empty:
                st.info("Geen foto's beschikbaar.")
            else:
                for _, row in latest.iterrows():
                    image_path = row["image_path"]
                    if os.path.exists(image_path):
                        waste_type = row["waste_type"] if pd.notna(row["waste_type"]) else "onbekend"
                        st.image(
                            image_path,
                            caption=f"{row['timestamp'].strftime('%d/%m/%Y %H:%M')} | {row['label']} | {waste_type}",
                            use_container_width=True
                        )
                    else:
                        st.warning(f"Afbeelding niet gevonden: {image_path}")
            st.markdown('</div>', unsafe_allow_html=True)

elif page == "Historiek":
    if filtered_df.empty:
        st.warning("Geen historiek beschikbaar voor de gekozen filters.")
    else:
        top_left, top_right = st.columns([1.4, 1])

        with top_left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            section_start("Volledige historiek")
            show_df = filtered_df.sort_values("timestamp", ascending=False).copy()
            show_df["timestamp"] = show_df["timestamp"].dt.strftime("%d/%m/%Y %H:%M")
            show_df["confidence"] = format_confidence(show_df["confidence"])
            st.dataframe(show_df, use_container_width=True, height=500)
            st.markdown('</div>', unsafe_allow_html=True)

        with top_right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            section_start("Laatste 3 foto's")
            latest = filtered_df.sort_values("timestamp", ascending=False).head(3)

            if latest.empty:
                st.info("Geen foto's beschikbaar.")
            else:
                for _, row in latest.iterrows():
                    if os.path.exists(row["image_path"]):
                        waste_type = row["waste_type"] if pd.notna(row["waste_type"]) else "onbekend"
                        st.image(
                            row["image_path"],
                            caption=f"{row['timestamp'].strftime('%d/%m/%Y %H:%M')} | {row['label']} | {waste_type}",
                            use_container_width=True
                        )
                    else:
                        st.warning(f"Afbeelding niet gevonden: {row['image_path']}")
            st.markdown('</div>', unsafe_allow_html=True)

elif page == "Simulatie":
    st.markdown(
        """
        <div class="info-box">
            Gebruik deze pagina om handmatig een detectie toe te voegen.
            Dit is handig zolang de webcam en AI-modellen nog niet gekoppeld zijn.
            Alles wat je hier toevoegt, verschijnt meteen in het dashboard en in de historiek.
        </div>
        """,
        unsafe_allow_html=True
    )

    col_form, col_preview = st.columns([1, 1])

    with col_form:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        section_start("Nieuwe simulatie")

        label = st.selectbox("Label", ["sluikstort", "geen_sluikstort"])
        confidence = st.slider("Confidence", 0.0, 1.0, 0.90, 0.01)
        quantity = st.number_input("Hoeveelheid", min_value=0, value=1, step=1)
        waste_type = st.text_input("Type afval", value="onbekend")

        available_images = get_images_from_captured()
        chosen_image = st.selectbox("Kies een foto", available_images) if available_images else None

        if not available_images:
            st.warning("Geen foto's gevonden in de map captured/.")

        if st.button("Voeg simulatie toe"):
            if chosen_image is None:
                st.error("Kies eerst een afbeelding.")
            else:
                new_row = pd.DataFrame([{
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "label": label,
                    "confidence": confidence,
                    "image_path": os.path.join(CAPTURED_DIR, chosen_image),
                    "quantity": quantity,
                    "waste_type": waste_type if waste_type else "onbekend",
                    "source": "simulatie",
                }])
                new_df = pd.concat([load_data(), new_row], ignore_index=True)
                save_data(new_df)
                st.success("Simulatie toegevoegd aan detections.csv.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_preview:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        section_start("Preview van gekozen foto")

        if chosen_image:
            preview_path = os.path.join(CAPTURED_DIR, chosen_image)
            if os.path.exists(preview_path):
                st.image(preview_path, caption=f"Gekozen afbeelding: {chosen_image}", use_container_width=True)
        else:
            st.info("Kies links een afbeelding om hier een preview te zien.")
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "AI Test sluikstort":
    st.markdown(
        """
        <div class="info-box">
            Hier test je model 1. Dit model voorspelt of een beeld eerder
            <b>sluikstort</b> of <b>geen_sluikstort</b> bevat.
        </div>
        """,
        unsafe_allow_html=True
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        section_start("Model en afbeelding")

        st.write(f"**Modelpad:** `{MODEL1_PATH}`")
        st.write(f"**Labelpad:** `{MODEL1_LABELS_PATH}`")

        if not os.path.exists(MODEL1_PATH):
            st.error("Modelbestand niet gevonden.")
        elif not os.path.exists(MODEL1_LABELS_PATH):
            st.error("labels.txt niet gevonden.")
        else:
            images = get_images_from_captured()

            if not images:
                st.warning("Geen foto's gevonden in captured/.")
            else:
                selected_image = st.selectbox("Kies afbeelding", images, key="model1_image")
                image_path = os.path.join(CAPTURED_DIR, selected_image)
                st.image(image_path, caption="Testafbeelding", use_container_width=True)

                if st.button("Voorspel sluikstort"):
                    model = load_teachable_model(MODEL1_PATH)
                    labels = load_labels(MODEL1_LABELS_PATH)
                    label, confidence, result_df = predict_with_model(model, labels, image_path)

                    if label is None:
                        st.error("De afbeelding kon niet gelezen worden.")
                    else:
                        st.markdown(
                            f"""
                            <div class="result-box">
                                <b>Voorspelling:</b> {label}<br>
                                <b>Confidence:</b> {confidence:.2%}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        result_df["score"] = (result_df["score"] * 100).round(2)
                        st.dataframe(result_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        section_start("Topresultaten visueel")

        if os.path.exists(MODEL1_PATH) and os.path.exists(MODEL1_LABELS_PATH):
            images = get_images_from_captured()
            if images:
                selected_image_right = st.selectbox("Kies afbeelding voor grafiek", images, key="model1_image_right")
                image_path_right = os.path.join(CAPTURED_DIR, selected_image_right)

                if st.button("Toon scoregrafiek sluikstort"):
                    model = load_teachable_model(MODEL1_PATH)
                    labels = load_labels(MODEL1_LABELS_PATH)
                    _, _, result_df = predict_with_model(model, labels, image_path_right)

                    if result_df is not None:
                        chart_df = result_df.copy()
                        chart_df["score"] = chart_df["score"] * 100
                        fig = px.bar(
                            chart_df,
                            x="score",
                            y="label",
                            orientation="h",
                            template="plotly_dark",
                            title="Voorspellingsscores model 1"
                        )
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=20, r=20, t=50, b=20),
                            xaxis_title="Score (%)",
                            yaxis_title="Label",
                            height=420,
                        )
                        fig.update_yaxes(categoryorder="total ascending")
                        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "AI Test afvaltype":
    st.markdown(
        """
        <div class="info-box">
            Hier test je model 2 op bestaande foto's uit de map captured.
            Dit model probeert het type afval te herkennen.
        </div>
        """,
        unsafe_allow_html=True
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        section_start("Model en afbeelding")

        st.write(f"**Modelpad:** `{MODEL2_PATH}`")
        st.write(f"**Labelpad:** `{MODEL2_LABELS_PATH}`")

        if not os.path.exists(MODEL2_PATH):
            st.error("Modelbestand niet gevonden.")
        elif not os.path.exists(MODEL2_LABELS_PATH):
            st.error("labels.txt niet gevonden.")
        else:
            images = get_images_from_captured()

            if not images:
                st.warning("Geen foto's gevonden in captured/.")
            else:
                selected_image = st.selectbox("Kies afbeelding", images, key="model2_image")
                image_path = os.path.join(CAPTURED_DIR, selected_image)
                st.image(image_path, caption="Testafbeelding", use_container_width=True)

                if st.button("Voorspel afvaltype"):
                    model = load_teachable_model(MODEL2_PATH)
                    labels = load_labels(MODEL2_LABELS_PATH)
                    label, confidence, result_df = predict_with_model(model, labels, image_path)

                    if label is None:
                        st.error("De afbeelding kon niet gelezen worden.")
                    else:
                        st.markdown(
                            f"""
                            <div class="result-box">
                                <b>Voorspelling:</b> {label}<br>
                                <b>Confidence:</b> {confidence:.2%}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        result_df["score"] = (result_df["score"] * 100).round(2)
                        st.dataframe(result_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        section_start("Topresultaten visueel")

        if os.path.exists(MODEL2_PATH) and os.path.exists(MODEL2_LABELS_PATH):
            images = get_images_from_captured()
            if images:
                selected_image_right = st.selectbox("Kies afbeelding voor grafiek", images, key="model2_image_right")
                image_path_right = os.path.join(CAPTURED_DIR, selected_image_right)

                if st.button("Toon scoregrafiek afvaltype"):
                    model = load_teachable_model(MODEL2_PATH)
                    labels = load_labels(MODEL2_LABELS_PATH)
                    _, _, result_df = predict_with_model(model, labels, image_path_right)

                    if result_df is not None:
                        chart_df = result_df.copy().head(8)
                        chart_df["score"] = chart_df["score"] * 100
                        fig = px.bar(
                            chart_df,
                            x="score",
                            y="label",
                            orientation="h",
                            template="plotly_dark",
                            title="Top 8 voorspellingen model 2"
                        )
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=20, r=20, t=50, b=20),
                            xaxis_title="Score (%)",
                            yaxis_title="Label",
                            height=420,
                        )
                        fig.update_yaxes(categoryorder="total ascending")
                        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Gecombineerde AI test":
    st.markdown(
        f"""
        <div class="info-box">
            Deze pagina combineert beide modellen.
            Eerst beslist model 1 of het om <b>sluikstort</b> gaat.
            Alleen als model 1 <b>sluikstort</b> voorspelt met minstens <b>{int(CONFIDENCE_THRESHOLD * 100)}%</b> confidence,
            wordt model 2 uitgevoerd om het <b>type afval</b> te bepalen.
            Detecties worden hier nog niet automatisch opgeslagen.
        </div>
        """,
        unsafe_allow_html=True
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        section_start("Afbeelding en gecombineerde analyse")

        images = get_images_from_captured()

        if not os.path.exists(MODEL1_PATH):
            st.error("Model 1 niet gevonden.")
        elif not os.path.exists(MODEL2_PATH):
            st.error("Model 2 niet gevonden.")
        elif not os.path.exists(MODEL1_LABELS_PATH):
            st.error("labels.txt van model 1 niet gevonden.")
        elif not os.path.exists(MODEL2_LABELS_PATH):
            st.error("labels.txt van model 2 niet gevonden.")
        elif not images:
            st.warning("Geen afbeeldingen gevonden in captured/.")
        else:
            selected_image = st.selectbox("Kies afbeelding", images, key="combined_image")
            image_path = os.path.join(CAPTURED_DIR, selected_image)
            st.image(image_path, caption="Testafbeelding", use_container_width=True)

            quantity = st.number_input("Hoeveelheid", min_value=1, value=1, step=1)

            if st.button("Voer gecombineerde analyse uit"):
                frame = cv2.imread(image_path)
                if frame is None:
                    st.error("Kon afbeelding niet openen.")
                else:
                    st.session_state["combined_result"] = analyze_combined_from_frame(frame, quantity)
                    st.session_state["combined_result"]["image_path"] = image_path

        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        section_start("Resultaat")
        render_result_block(st.session_state.get("combined_result"))
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Live Webcam":
    st.markdown(
        f"""
        <div class="info-box">
            Deze pagina heeft <b>3 modi</b>:
            <br><br>
            <b>Handmatige snapshot:</b> neem één webcamfoto en analyseer die.
            <br>
            <b>Interval analyse:</b> neem automatisch om de X seconden een webcambeeld.
            <br>
            <b>Echte live stream:</b> toon een echte camerastream en analyseer automatisch het laatste frame uit de stream.
            <br><br>
            Zodra model 1 <b>sluikstort</b> voorspelt met minstens <b>{int(CONFIDENCE_THRESHOLD * 100)}%</b> confidence,
            wordt automatisch een screenshot opgeslagen.
            <br>
            Er zit een autosave cooldown van <b>{AUTO_SAVE_COOLDOWN_SECONDS} seconden</b> op.
        </div>
        """,
        unsafe_allow_html=True
    )

    analysis_mode = st.selectbox(
        "Analysemodus",
        ["Handmatige snapshot", "Interval analyse", "Echte live stream"]
    )

    quantity = st.number_input("Hoeveelheid", min_value=1, value=1, step=1, key="webcam_quantity")

    if analysis_mode == "Handmatige snapshot":
        col_left, col_right = st.columns([1.1, 1])

        with col_left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            section_start("Webcam snapshot")

            if st.button("Neem snapshot en analyseer"):
                frame, error = capture_webcam_frame()
                if error:
                    st.error(error)
                else:
                    st.session_state.snapshot_frame = frame
                    st.session_state.snapshot_result = analyze_combined_from_frame(frame, quantity)
                    _, message = auto_save_if_needed(
                        st.session_state.snapshot_result,
                        frame,
                        quantity,
                        "snapshot_auto"
                    )
                    st.session_state.last_auto_save_message = message

            if st.session_state.snapshot_frame is not None:
                frame_rgb = cv2.cvtColor(st.session_state.snapshot_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="Laatste snapshot", use_container_width=True)
            else:
                st.info("Neem eerst een snapshot.")

            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            section_start("Resultaat")
            render_result_block(st.session_state.snapshot_result)
            if st.session_state.last_auto_save_message:
                st.success(st.session_state.last_auto_save_message)
            st.markdown('</div>', unsafe_allow_html=True)

    elif analysis_mode == "Interval analyse":
        interval_seconds = st.slider(
            "Interval in seconden",
            min_value=1,
            max_value=10,
            value=DEFAULT_INTERVAL_SECONDS
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Start interval analyse"):
                st.session_state.live_running = True
        with col_btn2:
            if st.button("Stop interval analyse"):
                st.session_state.live_running = False

        if st.session_state.live_running:
            st_autorefresh(interval=interval_seconds * 1000, key="interval_refresh")

        left, right = st.columns([1.1, 1])

        if st.session_state.live_running:
            frame, error = capture_webcam_frame()
            if error:
                st.error(error)
            else:
                st.session_state.live_frame = frame
                st.session_state.live_result = analyze_combined_from_frame(frame, quantity)
                _, message = auto_save_if_needed(
                    st.session_state.live_result,
                    frame,
                    quantity,
                    "interval_auto"
                )
                st.session_state.last_auto_save_message = message

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            section_start("Laatste interval-frame")
            if st.session_state.live_frame is not None:
                frame_rgb = cv2.cvtColor(st.session_state.live_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="Laatste frame", use_container_width=True)
            else:
                st.info("Start eerst de interval analyse.")
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            section_start("Resultaat")
            render_result_block(st.session_state.live_result)
            if st.session_state.last_auto_save_message:
                st.success(st.session_state.last_auto_save_message)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown(
            """
            <div class="warning-box">
                In deze modus zie je een echte live stream.
                Het systeem analyseert automatisch het laatste frame uit de stream
                en slaat meteen een screenshot op zodra sluikstort wordt gedetecteerd
                met voldoende confidence.
            </div>
            """,
            unsafe_allow_html=True
        )

        stream_interval_seconds = st.slider(
            "Analyse-interval voor live stream (seconden)",
            min_value=1,
            max_value=10,
            value=2,
            key="stream_interval_seconds"
        )

        st_autorefresh(interval=stream_interval_seconds * 1000, key="stream_auto_refresh")

        ctx = webrtc_streamer(
            key="live-stream",
            video_processor_factory=LiveVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

        col_left, col_right = st.columns([1, 1])

        if ctx.video_processor:
            with ctx.video_processor.lock:
                if ctx.video_processor.latest_frame is not None:
                    frame = ctx.video_processor.latest_frame.copy()
                    st.session_state.stream_frame = frame
                    st.session_state.stream_result = analyze_combined_from_frame(frame, quantity)
                    _, message = auto_save_if_needed(
                        st.session_state.stream_result,
                        frame,
                        quantity,
                        "stream_auto"
                    )
                    st.session_state.last_auto_save_message = message

        with col_left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            section_start("Live stream analyse")
            if st.session_state.stream_frame is not None:
                frame_rgb = cv2.cvtColor(st.session_state.stream_frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="Laatste geanalyseerde live frame", use_container_width=True)
            else:
                st.info("Nog geen frame beschikbaar uit de live stream.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            section_start("Resultaat")
            render_result_block(st.session_state.stream_result)
            if st.session_state.last_auto_save_message:
                st.success(st.session_state.last_auto_save_message)
            st.markdown('</div>', unsafe_allow_html=True)