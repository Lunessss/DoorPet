"""
Smart Pet Door - Streamlit (Solo MQTT)
======================================
No depende de Wokwi URLs ni de endpoints HTTP.
Todo el control se realiza mediante MQTT.

Broker:
    broker.hivemq.com
Puerto:
    1883

Topics:
    smartdoor/command    -> enviar comandos
    smartdoor/status     -> recibir estado ("abierta", "cerrada")
    smartdoor/detection  -> recibir detecciones ("dog", "cat")
"""

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
# CONFIGURACIÓN MQTT
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
            st.session_state.door_state = "open"
        elif payload == "cerrada":
            st.session_state.door_state = "closed"

    elif msg.topic == TOPIC_DETECTION:
        if payload in ("dog", "cat"):
            st.session_state.last_animal = payload


@st.cache_resource
def get_mqtt_client():
    client = mqtt.Client(
        client_id=f"streamlit-smartdoor-{uuid.uuid4().hex[:8]}"
    )

    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    client.subscribe(TOPIC_STATUS)
    client.subscribe(TOPIC_DETECTION)

    client.loop_start()

    return client


mqtt_client = get_mqtt_client()


def send_command(command: str):
    mqtt_client.publish(TOPIC_COMMAND, command, retain=False)
    time.sleep(0.5)  # dar tiempo a que el ESP32 responda


# =========================================================
# DETECCIÓN DE ANIMALES
# =========================================================
def detect_animal(image_bytes: bytes) -> str:
    """
    Retorna: 'dog', 'cat' o 'none'
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))

        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x, verbose=0)
        results = decode_predictions(preds, top=5)[0]

        dog_keywords = [
            "dog",
            "retriever",
            "shepherd",
            "poodle",
            "terrier",
            "beagle",
            "husky",
            "bulldog",
            "chihuahua",
            "pug",
            "doberman",
            "rottweiler",
            "labrador",
            "malamute",
            "spaniel",
            "wolfhound",
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
# STREAMLIT UI
# =========================================================
st.set_page_config(
    page_title="Puerta Inteligente",
    page_icon="🚪",
    layout="centered",
)

st.title("🚪 Puerta Inteligente")
st.caption("Control por voz · Botones · IA · MQTT")

# =========================================================
# SESSION STATE
# =========================================================
if "door_state" not in st.session_state:
    st.session_state.door_state = "desconocido"

if "last_animal" not in st.session_state:
    st.session_state.last_animal = "none"

if "log" not in st.session_state:
    st.session_state.log = []


def add_log(msg: str):
    st.session_state.log.insert(0, msg)
    st.session_state.log = st.session_state.log[:10]


# =========================================================
# ESTADO ACTUAL
# =========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Estado")

    if st.session_state.door_state == "open":
        st.success("🔓 ABIERTA")
    else:
        st.error("🔒 CERRADA")

with col2:
    st.subheader("Última detección")

    if st.session_state.last_animal == "dog":
        st.write("🐕 Perro")
    elif st.session_state.last_animal == "cat":
        st.write("🐈 Gato")
    else:
        st.write("— Sin animal")

st.divider()

# =========================================================
# CONTROL POR VOZ
# =========================================================
st.subheader("🎙️ Control por voz")

voice_js = """
new Promise((resolve) => {
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    resolve('');
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = 'es-ES';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.onresult = (e) => {
    resolve(e.results[0][0].transcript.toLowerCase());
  };

  recognition.onerror = () => resolve('');
  recognition.start();
})
"""

if st.button("🎤 Hablar", use_container_width=True):
    transcript = st_javascript(voice_js)

    if transcript:
        st.info(f"Escuché: {transcript}")

        if "abre" in transcript:
            send_command("open")
            st.success("Puerta abierta")
            add_log("🎙️ Voz → abrir")
            st.rerun()

        elif "cierra" in transcript:
            send_command("close")
            st.success("Puerta cerrada")
            add_log("🎙️ Voz → cerrar")
            st.rerun()

st.divider()

# =========================================================
# BOTONES MANUALES
# =========================================================
st.subheader("🔘 Control manual")

col_open, col_close = st.columns(2)

with col_open:
    if st.button("🔓 Abrir puerta", use_container_width=True):
        send_command("open")
        add_log("🔘 Botón → abrir")
        st.rerun()

with col_close:
    if st.button("🔒 Cerrar puerta", use_container_width=True):
        send_command("close")
        add_log("🔘 Botón → cerrar")
        st.rerun()

st.divider()

# =========================================================
# DETECCIÓN DE MASCOTAS
# =========================================================
st.subheader("📷 Detección de mascotas")

source = st.radio(
    "Fuente",
    ["Cámara", "Subir archivo"],
    horizontal=True,
    label_visibility="collapsed",
)

image_bytes = None

if source == "Cámara":
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

    if st.button("🔍 Analizar con IA", use_container_width=True):
        with st.spinner("Analizando..."):
            animal = detect_animal(image_bytes)

        if animal == "dog":
            st.session_state.last_animal = "dog"
            send_command("dog")
            st.success("🐕 Perro detectado")
            add_log("📷 IA → perro")
            st.rerun()

        elif animal == "cat":
            st.session_state.last_animal = "cat"
            send_command("cat")
            st.success("🐈 Gato detectado")
            add_log("📷 IA → gato")
            st.rerun()

        else:
            st.session_state.last_animal = "none"
            send_command("close")
            st.info("No se detectó un gato o perro. La puerta se cerró.")
            add_log("📷 IA → none (cerrar puerta)")
            st.rerun()

st.divider()

# =========================================================
# REGISTRO
# =========================================================
st.subheader("📋 Registro")

if st.session_state.log:
    for entry in st.session_state.log:
        st.text(entry)
else:
    st.caption("Sin actividad aún")

# =========================================================
# CONFIGURACIÓN
# =========================================================
with st.expander("⚙️ Configuración MQTT"):
    st.text(f"Broker: {MQTT_BROKER}")
    st.text(f"Puerto: {MQTT_PORT}")
    st.text(f"Topic comandos: {TOPIC_COMMAND}")
    st.text(f"Topic estado: {TOPIC_STATUS}")
    st.text(f"Topic detección: {TOPIC_DETECTION}")

    if st.button("🔄 Actualizar"):
        st.rerun()
