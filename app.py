"""
Smart Door — Streamlit App
==========================
Funciones:
  • Comandos de voz
  • Botones manuales
  • Detección de gatos y perros con TensorFlow
  • Control del ESP32/Wokwi
"""

import os
import io
import requests
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_javascript import st_javascript
from dotenv import load_dotenv
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions,
)

load_dotenv()

# ── CONFIGURACIÓN ────────────────────────────────────────────────────────
# IMPORTANTE:
# En Streamlit Secrets o en .env debes poner:
# ESP32_URL=https://tu-subdominio.wokwi.app
#
# NO uses:
# https://wokwi.com/projects/xxxxxxxx
#
DEFAULT_ESP32_URL = ""
ESP32_URL = os.getenv("ESP32_URL", DEFAULT_ESP32_URL).rstrip("/")


def is_valid_wokwi_url(url: str) -> bool:
    return url.startswith("https://") and ".wokwi.app" in url


# ── MODELO DE IA ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")


model = load_model()


# ── HELPERS ──────────────────────────────────────────────────────────────
def safe_json_response(response: requests.Response) -> dict:
    try:
        return response.json()
    except Exception:
        return {
            "error": f"Respuesta inválida (HTTP {response.status_code})",
            "response": response.text,
        }


def check_configuration():
    if not ESP32_URL:
        st.error(
            "❌ Falta configurar ESP32_URL en Secrets o .env.\n\n"
            "Ejemplo:\n"
            "ESP32_URL=https://abc123.wokwi.app"
        )
        st.stop()

    if not is_valid_wokwi_url(ESP32_URL):
        st.error(
            "❌ ESP32_URL no es válida.\n\n"
            "Debe verse así:\n"
            "https://abc123.wokwi.app\n\n"
            "No uses:\n"
            "https://wokwi.com/projects/..."
        )
        st.stop()


def request_json(method: str, path: str, **kwargs) -> dict:
    try:
        url = f"{ESP32_URL}{path}"
        response = requests.request(method, url, timeout=10, **kwargs)
        return safe_json_response(response)
    except Exception as e:
        return {"error": str(e)}


# ── COMUNICACIÓN CON ESP32 ───────────────────────────────────────────────
def send_door_command(action: str) -> dict:
    return request_json(
        "POST",
        "/door",
        json={"action": action},
    )


def send_detect_command(animal: str) -> dict:
    # El ESP32 usa /detect, NO /led
    return request_json(
        "POST",
        "/detect",
        json={"animal": animal},
    )


def get_door_status() -> dict:
    data = request_json("GET", "/status")

    if "door" not in data:
        data["door"] = "desconocido"

    return data


# ── DETECCIÓN DE ANIMALES ────────────────────────────────────────────────
def detect_animal(image_bytes: bytes) -> str:
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


# ── UI ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Puerta Inteligente",
    page_icon="🚪",
    layout="centered",
)

st.title("🚪 Puerta Inteligente")
st.caption("Control por voz · Detección de mascotas · ESP32 + Wokwi")

check_configuration()

# ── SESSION STATE ────────────────────────────────────────────────────────
if "door_state" not in st.session_state:
    st.session_state.door_state = "desconocido"

if "last_animal" not in st.session_state:
    st.session_state.last_animal = "none"

if "log" not in st.session_state:
    st.session_state.log = []


def add_log(msg: str):
    st.session_state.log.insert(0, msg)
    st.session_state.log = st.session_state.log[:10]


# ── ESTADO ACTUAL ────────────────────────────────────────────────────────
status = get_door_status()
st.session_state.door_state = status.get("door", "desconocido")

if "error" in status:
    st.warning(f"⚠️ {status['error']}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Estado")
    if st.session_state.door_state == "open":
        st.success("🔓 ABIERTA")
    elif st.session_state.door_state == "closed":
        st.error("🔒 CERRADA")
    else:
        st.info("❓ DESCONOCIDO")

with col2:
    st.subheader("Última detección")
    if st.session_state.last_animal == "dog":
        st.write("🐕 Perro")
    elif st.session_state.last_animal == "cat":
        st.write("🐈 Gato")
    else:
        st.write("— Sin animal")

st.divider()

# ── CONTROL POR VOZ ──────────────────────────────────────────────────────
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
            resp = send_door_command("open")
            if "error" not in resp:
                st.success("Puerta abierta")
                add_log("🎙️ Voz → abrir")
            else:
                st.error(resp["error"])

        elif "cierra" in transcript:
            resp = send_door_command("close")
            if "error" not in resp:
                st.success("Puerta cerrada")
                add_log("🎙️ Voz → cerrar")
            else:
                st.error(resp["error"])

st.divider()

# ── CONTROL MANUAL ───────────────────────────────────────────────────────
st.subheader("🔘 Control manual")
col_open, col_close = st.columns(2)

with col_open:
    if st.button("🔓 Abrir puerta", use_container_width=True):
        resp = send_door_command("open")
        if "error" not in resp:
            st.success("Puerta abierta")
            add_log("🔘 Botón → abrir")
        else:
            st.error(resp["error"])

with col_close:
    if st.button("🔒 Cerrar puerta", use_container_width=True):
        resp = send_door_command("close")
        if "error" not in resp:
            st.success("Puerta cerrada")
            add_log("🔘 Botón → cerrar")
        else:
            st.error(resp["error"])

st.divider()

# ── DETECCIÓN DE MASCOTAS ────────────────────────────────────────────────
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
        type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        image_bytes = uploaded.read()

if image_bytes:
    st.image(image_bytes, caption="Imagen capturada", width=300)

    if st.button("🔍 Analizar con IA", use_container_width=True):
        with st.spinner("Analizando..."):
            animal = detect_animal(image_bytes)

        st.session_state.last_animal = animal

        if animal == "dog":
            st.success("🐕 Perro detectado")
            resp = send_detect_command("dog")
            if "error" in resp:
                st.error(resp["error"])
            else:
                add_log("📷 IA → perro")

        elif animal == "cat":
            st.success("🐈 Gato detectado")
            resp = send_detect_command("cat")
            if "error" in resp:
                st.error(resp["error"])
            else:
                add_log("📷 IA → gato")

        else:
            st.info("No se detectó un gato o perro")
            add_log("📷 IA → none")

st.divider()

# ── REGISTRO ─────────────────────────────────────────────────────────────
st.subheader("📋 Registro")

if st.session_state.log:
    for entry in st.session_state.log:
        st.text(entry)
else:
    st.caption("Sin actividad aún")

# ── CONFIGURACIÓN ────────────────────────────────────────────────────────
with st.expander("⚙️ Configuración"):
    st.text_input("URL del ESP32", value=ESP32_URL, disabled=True)
    st.caption(
        "Ejemplo correcto: https://abc123.wokwi.app\n"
        "No uses la URL de https://wokwi.com/projects/..."
    )

    if st.button("🔄 Actualizar estado"):
        st.rerun()
