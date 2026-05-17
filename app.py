# Smart Door — Streamlit App (Versión gratuita con detección de gatos y perros usando TensorFlow)

```python
"""
Smart Door — Streamlit App
==========================
Funciones:
  • Comandos de voz
  • Botones manuales
  • Detección de gatos y perros con TensorFlow (sin API key)
  • Control del ESP32/Wokwi
"""

import os
import io
import base64
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

# ── Configuración ──────────────────────────────────────────────────────────
ESP32_URL = os.getenv("ESP32_URL", "http://192.168.1.100")


# ── Modelo de IA (gratis, local) ──────────────────────────────────────────
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")


model = load_model()


# ── Helper para parsear JSON de forma segura ───────────────────────────────
def safe_json_response(response: requests.Response) -> dict:
    if response.status_code == 200 and response.text.strip():
        try:
            return response.json()
        except ValueError:
            return {
                "error": "La respuesta no es un JSON válido",
                "response": response.text,
            }
    else:
        return {
            "error": f"Respuesta vacía o código {response.status_code}",
            "response": response.text,
        }


# ── Comunicación con ESP32 ────────────────────────────────────────────────
def send_door_command(action: str) -> dict:
    try:
        r = requests.post(
            f"{ESP32_URL}/door",
            json={"action": action},
            timeout=5,
        )
        return safe_json_response(r)
    except Exception as e:
        return {"error": str(e)}



def send_led_command(animal: str) -> dict:
    try:
        r = requests.post(
            f"{ESP32_URL}/led",
            json={"type": animal},
            timeout=5,
        )
        return safe_json_response(r)
    except Exception as e:
        return {"error": str(e)}



def get_door_status() -> dict:
    try:
        r = requests.get(f"{ESP32_URL}/status", timeout=3)
        data = safe_json_response(r)
        if "error" in data and "door" not in data:
            data["door"] = "desconocido"
        return data
    except Exception as e:
        return {
            "door": "desconocido",
            "error": str(e),
        }


# ── Detección de gatos y perros (sin API) ─────────────────────────────────
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


# ── Streamlit UI ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Puerta Inteligente",
    page_icon="🚪",
    layout="centered",
)

st.title("🚪 Puerta Inteligente")
st.caption("Control por voz · Detección de mascotas · Pixel art LED")


# ── Estado de sesión ──────────────────────────────────────────────────────
if "door_state" not in st.session_state:
    st.session_state.door_state = "desconocido"
if "last_animal" not in st.session_state:
    st.session_state.last_animal = "none"
if "log" not in st.session_state:
    st.session_state.log = []



def add_log(msg: str):
    st.session_state.log.insert(0, msg)
    st.session_state.log = st.session_state.log[:10]


# ── Estado actual ─────────────────────────────────────────────────────────
status = get_door_status()
st.session_state.door_state = status.get("door", "desconocido")

if "error" in status:
    st.warning(f"⚠️ ESP32: {status['error']}")

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
    animal = st.session_state.last_animal
    if animal == "dog":
        st.write("🐕 Perro")
    elif animal == "cat":
        st.write("🐈 Gato")
    else:
        st.write("— Sin animal")

st.divider()


# ── Control por voz ───────────────────────────────────────────────────────
st.subheader("🎙️ Control por voz")

voice_js = """
new Promise((resolve) => {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
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


# ── Botones manuales ──────────────────────────────────────────────────────
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


# ── Detección de mascotas ─────────────────────────────────────────────────
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
    uploaded = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
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
            resp = send_led_command("dog")
            if "error" in resp:
                st.error(resp["error"])
            add_log("📷 IA → perro")

        elif animal == "cat":
            st.success("🐈 Gato detectado")
            resp = send_led_command("cat")
            if "error" in resp:
                st.error(resp["error"])
            add_log("📷 IA → gato")

        else:
            st.info("No se detectó un gato o perro")
            add_log("📷 IA → none")

st.divider()


# ── Registro ──────────────────────────────────────────────────────────────
st.subheader("📋 Registro")

if st.session_state.log:
    for entry in st.session_state.log:
        st.text(entry)
else:
    st.caption("Sin actividad aún")


# ── Configuración ─────────────────────────────────────────────────────────
with st.expander("⚙️ Configuración"):
    st.text_input("URL del ESP32", value=ESP32_URL, disabled=True)
    st.caption("Define ESP32_URL en el archivo .env o en Secrets de Streamlit.")

    if st.button("🔄 Actualizar estado"):
        st.rerun()
