import io
import uuid
import time
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_javascript import st_javascript
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)
import paho.mqtt.client as mqtt

# =========================================================
# CONFIGURACION MQTT
# =========================================================
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883

TOPIC_COMMAND = "smartdoor/command"
TOPIC_STATUS = "smartdoor/status"
TOPIC_DETECTION = "smartdoor/detection"


# =========================================================
# MODELO DE IA
# =========================================================
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")


model = load_model()


# =========================================================
# MQTT
# =========================================================
def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8").strip().lower()

    if msg.topic == TOPIC_STATUS:
        if payload == "abierta":
            st.session_state.door_state = "abierta"
        elif payload == "cerrada":
            st.session_state.door_state = "cerrada"

    elif msg.topic == TOPIC_DETECTION:
        if payload in ("dog", "cat"):
            st.session_state.last_animal = payload


@st.cache_resource
def get_mqtt_client():
    client = mqtt.Client(
        client_id="streamlit-smartdoor-" + uuid.uuid4().hex[:8]
    )
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(TOPIC_STATUS)
    client.subscribe(TOPIC_DETECTION)
    client.loop_start()
    return client


mqtt_client = get_mqtt_client()


def send_command(command):
    mqtt_client.publish(TOPIC_COMMAND, command, retain=False)
    time.sleep(0.5)


# =========================================================
# DETECCION DE ANIMALES
# =========================================================
def detect_animal(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x, verbose=0)
        results = decode_predictions(preds, top=5)[0]

        dog_keywords = [
            "dog", "retriever", "shepherd", "poodle", "terrier",
            "beagle", "husky", "bulldog", "chihuahua", "pug",
            "doberman", "rottweiler", "labrador", "malamute",
            "spaniel", "wolfhound"
        ]

        for _, label, _ in results:
            label = label.lower()
            if "cat" in label:
                return "cat"
            if any(word in label for word in dog_keywords):
                return "dog"

        return "none"

    except Exception:
        return "none"


# =========================================================
# STREAMLIT CONFIG & CSS
# =========================================================
st.set_page_config(
    page_title="Puerta Inteligente",
    page_icon="🚪",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background-color: #F7F5F2;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.5rem 2rem 4rem 2rem;
    max-width: 680px;
}
h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.1rem !important;
    color: #1A1A1A !important;
    letter-spacing: -0.5px;
    margin-bottom: 0 !important;
}
h3 {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #9A9490 !important;
    margin-bottom: 0.75rem !important;
}
.stCaption p {
    font-size: 0.85rem;
    color: #A09B96;
    margin-top: 2px;
}
hr {
    border: none;
    border-top: 1px solid #E8E4DF;
    margin: 1.6rem 0;
}
div[data-testid="stSuccess"],
div[data-testid="stError"],
div[data-testid="stInfo"] {
    border-radius: 14px !important;
    border: none !important;
    padding: 0.9rem 1.1rem !important;
    font-size: 0.9rem !important;
}
div[data-testid="stSuccess"] {
    background-color: #EAF5EC !important;
    color: #2D6A35 !important;
}
div[data-testid="stError"] {
    background-color: #FDECEA !important;
    color: #842A25 !important;
}
div[data-testid="stInfo"] {
    background-color: #EEF3FB !important;
    color: #2A4A85 !important;
}
.stButton > button {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
    border: 1.5px solid #DDD9D4 !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.1rem !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
.stButton > button:hover {
    background-color: #F2EDE8 !important;
    border-color: #C5BFB9 !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.08) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
    box-shadow: none !important;
}
.btn-open > button {
    background-color: #2D6A35 !important;
    color: #FFFFFF !important;
    border-color: #2D6A35 !important;
}
.btn-open > button:hover {
    background-color: #245B2C !important;
    border-color: #245B2C !important;
}
.btn-close > button {
    background-color: #C0392B !important;
    color: #FFFFFF !important;
    border-color: #C0392B !important;
}
.btn-close > button:hover {
    background-color: #A93226 !important;
    border-color: #A93226 !important;
}
.stRadio label span {
    font-size: 0.875rem;
    color: #3A3633;
}
.stFileUploader {
    border: 1.5px dashed #D5CFC9 !important;
    border-radius: 14px !important;
    padding: 0.5rem !important;
    background: #FDFCFB !important;
}
.stCameraInput video,
.stCameraInput canvas {
    border-radius: 14px !important;
}
.stImage img {
    border-radius: 14px !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.08) !important;
}
.stExpander {
    border: 1.5px solid #E8E4DF !important;
    border-radius: 14px !important;
    background: #FDFCFB !important;
}
.stExpander summary {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #6B6560 !important;
}
.stSpinner > div {
    border-top-color: #2D6A35 !important;
}
.animal-card {
    background: #FFFFFF;
    border: 1.5px solid #E8E4DF;
    border-radius: 14px;
    padding: 0.85rem 1.1rem;
    font-size: 0.9rem;
    color: #3A3633;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================
if "door_state" not in st.session_state:
    st.session_state.door_state = "desconocido"

if "last_animal" not in st.session_state:
    st.session_state.last_animal = "none"

if "log" not in st.session_state:
    st.session_state.log = []

if "listening" not in st.session_state:
    st.session_state.listening = False


def add_log(msg):
    st.session_state.log.insert(0, msg)
    st.session_state.log = st.session_state.log[:10]


# =========================================================
# HEADER
# =========================================================
st.title("Puerta Inteligente")
st.caption("Control por voz · Botones · IA · MQTT")
st.markdown("<div style='margin-top:0.3rem'></div>", unsafe_allow_html=True)

# =========================================================
# ESTADO ACTUAL
# FIX: el ESP32 publica "abierta"/"cerrada", el codigo anterior
#      solo comparaba con "open"/"closed" y nunca hacia match.
# =========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Estado")
    state = st.session_state.door_state
    if state in ("abierta", "open"):
        st.markdown(
            '<div class="animal-card">🔓 &nbsp;<strong>Abierta</strong></div>',
            unsafe_allow_html=True,
        )
    elif state in ("cerrada", "closed"):
        st.markdown(
            '<div class="animal-card">🔒 &nbsp;<strong>Cerrada</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="animal-card" style="color:#9A9490;">&#8212; &nbsp;Desconocido</div>',
            unsafe_allow_html=True,
        )

with col2:
    st.subheader("Ultima deteccion")
    if st.session_state.last_animal == "dog":
        st.markdown(
            '<div class="animal-card">🐕 &nbsp;<strong>Perro</strong></div>',
            unsafe_allow_html=True,
        )
    elif st.session_state.last_animal == "cat":
        st.markdown(
            '<div class="animal-card">🐈 &nbsp;<strong>Gato</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="animal-card" style="color:#9A9490;">&#8212; &nbsp;Sin animal</div>',
            unsafe_allow_html=True,
        )

st.divider()

# =========================================================
# CONTROL POR VOZ
# FIX: flujo de dos pasos para que el microfono funcione en Streamlit.
#
#   Paso 1 (click boton): lanza el JS que activa el microfono.
#     El JS guarda el texto reconocido en sessionStorage y resuelve
#     la promesa. Se activa el flag listening=True y se hace rerun.
#
#   Paso 2 (siguiente rerun): lee sessionStorage, borra el valor,
#     procesa el comando y envia por MQTT.
#
# NOTA: requiere HTTPS en produccion. En localhost usa Chrome con
#   la flag "Insecure origins treated as secure" habilitada.
# =========================================================
st.subheader("Control por voz")

# -- Paso 2: leer transcript de sessionStorage --
read_transcript_js = """
(function() {
  var t = sessionStorage.getItem('voice_transcript');
  sessionStorage.removeItem('voice_transcript');
  return t || '';
})()
"""

if st.session_state.listening:
    transcript = st_javascript(read_transcript_js)
    st.session_state.listening = False

    if transcript and isinstance(transcript, str) and transcript.strip():
        t = transcript.strip().lower()
        st.info("Escuche: " + t)

        open_words = ["open", "abre", "abrir", "abre la puerta"]
        close_words = ["close", "closed", "cierra", "cerrar", "cierra la puerta"]

        if any(w in t for w in open_words):
            send_command("open")
            st.session_state.door_state = "abierta"
            add_log("Voz -> abrir")
            st.rerun()

        elif any(w in t for w in close_words):
            send_command("close")
            st.session_state.door_state = "cerrada"
            add_log("Voz -> cerrar")
            st.rerun()

        else:
            st.warning("No reconoci un comando valido: " + t)
    else:
        st.warning("No se capto audio. Intenta de nuevo.")

# -- Paso 1: boton que activa el microfono --
listen_js = """
new Promise((resolve) => {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    sessionStorage.setItem('voice_transcript', '');
    resolve('NO_SUPPORT');
    return;
  }
  const r = new SR();
  r.lang = 'es-ES';
  r.interimResults = false;
  r.maxAlternatives = 1;
  r.onresult = (e) => {
    const text = e.results[0][0].transcript.toLowerCase();
    sessionStorage.setItem('voice_transcript', text);
    resolve(text);
  };
  r.onerror = (e) => {
    sessionStorage.setItem('voice_transcript', '');
    resolve('ERROR:' + e.error);
  };
  r.onend = () => { resolve(''); };
  r.start();
})
"""

btn_label = "🔴  Escuchando..." if st.session_state.listening else "🎤  Hablar"
if st.button(btn_label, use_container_width=True, disabled=st.session_state.listening):
    st_javascript(listen_js)
    st.session_state.listening = True
    st.rerun()

st.divider()

# =========================================================
# BOTONES MANUALES
# FIX: actualizan door_state localmente ademas de enviar MQTT,
#      para que el estado se refleje de inmediato en pantalla.
# =========================================================
st.subheader("Control manual")

col_open, col_close = st.columns(2)

with col_open:
    st.markdown('<div class="btn-open">', unsafe_allow_html=True)
    if st.button("🔓  Abrir puerta", use_container_width=True):
        send_command("open")
        st.session_state.door_state = "abierta"
        add_log("Boton -> abrir")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with col_close:
    st.markdown('<div class="btn-close">', unsafe_allow_html=True)
    if st.button("🔒  Cerrar puerta", use_container_width=True):
        send_command("close")
        st.session_state.door_state = "cerrada"
        add_log("Boton -> cerrar")
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# =========================================================
# DETECCION DE MASCOTAS
# =========================================================
st.subheader("Deteccion de mascotas")

source = st.radio(
    "Fuente",
    ["Camara", "Subir archivo"],
    horizontal=True,
    label_visibility="collapsed",
)

image_bytes = None

if source == "Camara":
    photo = st.camera_input("Toma una foto")
    if photo:
        image_bytes = photo.getvalue()
else:
    uploaded = st.file_uploader(
        "Sube una imagen",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded:
        image_bytes = uploaded.read()

if image_bytes:
    st.image(image_bytes, caption="Imagen capturada", width=300)

    if st.button("🔍  Analizar con IA", use_container_width=True):
        with st.spinner("Analizando imagen..."):
            animal = detect_animal(image_bytes)

        if animal == "dog":
            st.session_state.last_animal = "dog"
            st.session_state.door_state = "abierta"
            send_command("dog")
            st.success("🐕 Perro detectado - puerta abierta")
            add_log("IA -> perro")
            st.rerun()

        elif animal == "cat":
            st.session_state.last_animal = "cat"
            st.session_state.door_state = "abierta"
            send_command("cat")
            st.success("🐈 Gato detectado - puerta abierta")
            add_log("IA -> gato")
            st.rerun()

        else:
            st.session_state.last_animal = "none"
            st.session_state.door_state = "cerrada"
            send_command("close")
            st.info("No se detecto una mascota. La puerta permanece cerrada.")
            add_log("IA -> none (cerrar)")
            st.rerun()

st.divider()

# =========================================================
# REGISTRO
# =========================================================
st.subheader("Registro de actividad")

if st.session_state.log:
    for i, entry in enumerate(st.session_state.log):
        opacity = max(0.35, 1.0 - i * 0.08)
        st.markdown(
            "<p style='font-size:0.83rem;color:rgba(107,101,96,"
            + str(opacity)
            + ");padding:0.35rem 0;border-bottom:1px solid #F0EBE6;margin:0;'>"
            + entry
            + "</p>",
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        "<p style='font-size:0.83rem;color:#C0BAB4;'>Sin actividad aun</p>",
        unsafe_allow_html=True,
    )

# =========================================================
# CONFIGURACION MQTT
# =========================================================
st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

with st.expander("⚙️  Configuracion MQTT"):
    st.markdown(
        "<div style='display:grid;gap:0.4rem;font-size:0.83rem;color:#6B6560;'>"
        "<div><span style='color:#9A9490;font-size:0.72rem;text-transform:uppercase;"
        "letter-spacing:0.08em;'>Broker</span><br>" + MQTT_BROKER + "</div>"
        "<div><span style='color:#9A9490;font-size:0.72rem;text-transform:uppercase;"
        "letter-spacing:0.08em;'>Puerto</span><br>" + str(MQTT_PORT) + "</div>"
        "<div><span style='color:#9A9490;font-size:0.72rem;text-transform:uppercase;"
        "letter-spacing:0.08em;'>Topic comandos</span><br>" + TOPIC_COMMAND + "</div>"
        "<div><span style='color:#9A9490;font-size:0.72rem;text-transform:uppercase;"
        "letter-spacing:0.08em;'>Topic estado</span><br>" + TOPIC_STATUS + "</div>"
        "<div><span style='color:#9A9490;font-size:0.72rem;text-transform:uppercase;"
        "letter-spacing:0.08em;'>Topic deteccion</span><br>" + TOPIC_DETECTION + "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='margin-top:0.8rem'></div>", unsafe_allow_html=True)
    if st.button("🔄  Actualizar estado"):
        st.rerun()
