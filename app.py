"""
Smart Door — Streamlit App
==========================
Funciones:
  • Comandos de voz  → reconocimiento en navegador (Web Speech API via JS)
  • Botones manuales → abrir / cerrar puerta
  • Cámara           → captura foto → Claude Vision → detecta gato/perro
  • Feedback visual  → muestra el animal detectado y el estado de la puerta
"""

import os
import base64
import requests
import streamlit as st
import anthropic
from streamlit_javascript import st_javascript
from dotenv import load_dotenv

load_dotenv()

# ── Configuración ──────────────────────────────────────────────────────────
ESP32_URL = os.getenv("ESP32_URL", "http://192.168.1.100")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")

client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)


# ── Helper para parsear JSON de forma segura ───────────────────────────────
def safe_json_response(response: requests.Response) -> dict:
    """Convierte una respuesta HTTP a JSON de forma segura."""
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


# ── Helpers ESP32 ──────────────────────────────────────────────────────────
def send_door_command(action: str) -> dict:
    """Envía POST /door?action=open|close al ESP32."""
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
    """Envía POST /led con el tipo de animal para mostrar pixel art."""
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
    """Obtiene el estado actual de la puerta desde /status."""
    try:
        r = requests.get(f"{ESP32_URL}/status", timeout=3)
        data = safe_json_response(r)

        # Si hubo error, aseguramos que siempre exista la clave 'door'
        if "error" in data and "door" not in data:
            data["door"] = "desconocido"

        return data

    except Exception as e:
        return {
            "door": "desconocido",
            "error": str(e),
        }


# ── IA: detección de mascota ───────────────────────────────────────────────
def detect_animal(image_bytes: bytes) -> str:
    """
    Envía la imagen a Claude Vision.
    Retorna: "dog" | "cat" | "none"
    """
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Mira esta imagen con cuidado. "
                            "¿Hay un perro o un gato visible? "
                            "Responde SOLO con una de estas tres palabras exactas: "
                            "dog, cat, none. "
                            "No expliques nada más."
                        ),
                    },
                ],
            }
        ],
    )

    result = message.content[0].text.strip().lower()
    if "dog" in result:
        return "dog"
    if "cat" in result:
        return "cat"
    return "none"


# ── Streamlit UI ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Puerta Inteligente",
    page_icon="🚪",
    layout="centered",
)

# ─── CSS personalizado ─────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

      html, body, [class*="css"] { font-family: 'Space Mono', monospace; }

      .status-open   { color: #2ecc71; font-weight: 700; font-size: 1.3rem; }
      .status-closed { color: #e74c3c; font-weight: 700; font-size: 1.3rem; }
      .animal-badge  {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 8px;
      }
      .dog-badge { background: #ffeaa7; color: #6c3d00; }
      .cat-badge { background: #dfe6e9; color: #2d3436; }
      .none-badge{ background: #f5f6fa; color: #636e72; }

      .pixel-preview {
        image-rendering: pixelated;
        width: 120px;
        height: 120px;
        border: 2px solid #2d3436;
        border-radius: 8px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Encabezado ────────────────────────────────────────────────────────────
st.title("🚪 Puerta Inteligente")
st.caption("Control por voz · Detección de mascotas · Pixel art LED")

# ─── Estado de sesión ──────────────────────────────────────────────────────
if "door_state" not in st.session_state:
    st.session_state.door_state = "desconocido"
if "last_animal" not in st.session_state:
    st.session_state.last_animal = "none"
if "voice_result" not in st.session_state:
    st.session_state.voice_result = ""
if "log" not in st.session_state:
    st.session_state.log = []



def add_log(msg: str):
    st.session_state.log.insert(0, msg)
    if len(st.session_state.log) > 10:
        st.session_state.log = st.session_state.log[:10]


# ─── Estado en tiempo real ─────────────────────────────────────────────────
status = get_door_status()
st.session_state.door_state = status.get("door", "desconocido")

# Mostrar error de conexión si existe
if "error" in status:
    st.warning(f"⚠️ ESP32: {status['error']}")

col_state, col_animal = st.columns(2)

with col_state:
    st.markdown("**Estado de la puerta**")

    if st.session_state.door_state == "open":
        css_class = "status-open"
        icon = "🔓"
        label = "OPEN"
    elif st.session_state.door_state == "closed":
        css_class = "status-closed"
        icon = "🔒"
        label = "CLOSED"
    else:
        css_class = "status-closed"
        icon = "❓"
        label = "DESCONOCIDO"

    st.markdown(
        f'<p class="{css_class}">{icon} {label}</p>',
        unsafe_allow_html=True,
    )

with col_animal:
    st.markdown("**Última detección**")
    a = st.session_state.last_animal

    if a == "dog":
        st.markdown(
            '<span class="animal-badge dog-badge">🐕 Perro</span>',
            unsafe_allow_html=True,
        )
    elif a == "cat":
        st.markdown(
            '<span class="animal-badge cat-badge">🐈 Gato</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="animal-badge none-badge">— Sin animal</span>',
            unsafe_allow_html=True,
        )

st.divider()

# ─── Sección 1: Comandos de voz ────────────────────────────────────────────
st.subheader("🎙️ Control por voz")
st.caption('Di "abre la puerta" o "cierra la puerta"')

voice_js = """
new Promise((resolve) => {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    resolve("ERROR: navegador sin soporte");
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.lang = 'es-ES';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.onresult = (e) => {
    resolve(e.results[0][0].transcript.toLowerCase());
  };

  recognition.onerror = (e) => {
    resolve("ERROR: " + e.error);
  };

  recognition.start();
})
"""

if st.button("🎤 Hablar", use_container_width=True):
    with st.spinner("Escuchando..."):
        transcript = st_javascript(voice_js)

    if transcript:
        st.session_state.voice_result = transcript
        st.info(f"Escuché: {transcript}")

        if "abre" in transcript or "abrir" in transcript or "open" in transcript:
            resp = send_door_command("open")
            if "error" not in resp:
                st.session_state.door_state = "open"
                add_log("🎙️ Voz → puerta abierta")
                st.success("✅ Puerta abierta por comando de voz")
            else:
                st.error(f"Error: {resp['error']}")

        elif "cierra" in transcript or "cerrar" in transcript or "close" in transcript:
            resp = send_door_command("close")
            if "error" not in resp:
                st.session_state.door_state = "closed"
                add_log("🎙️ Voz → puerta cerrada")
                st.success("✅ Puerta cerrada por comando de voz")
            else:
                st.error(f"Error: {resp['error']}")
        else:
            st.warning("No reconocí un comando válido.")

st.divider()

# ─── Sección 2: Botones manuales ───────────────────────────────────────────
st.subheader("🔘 Control manual")
btn_open, btn_close = st.columns(2)

with btn_open:
    if st.button("🔓 Abrir puerta", use_container_width=True, type="primary"):
        resp = send_door_command("open")
        if "error" not in resp:
            st.session_state.door_state = "open"
            add_log("🔘 Botón → puerta abierta")
            st.success("Puerta abierta")
        else:
            st.error(resp["error"])

with btn_close:
    if st.button("🔒 Cerrar puerta", use_container_width=True):
        resp = send_door_command("close")
        if "error" not in resp:
            st.session_state.door_state = "closed"
            add_log("🔘 Botón → puerta cerrada")
            st.success("Puerta cerrada")
        else:
            st.error(resp["error"])

st.divider()

# ─── Sección 3: Detección de mascotas con cámara ──────────────────────────
st.subheader("📷 Detección de mascotas con IA")
st.caption(
    "Toma una foto y la IA detecta si hay un gato o un perro y abre la puerta automáticamente"
)

img_source = st.radio(
    "Fuente de imagen",
    ["Cámara", "Subir archivo"],
    horizontal=True,
    label_visibility="collapsed",
)

image_bytes = None

if img_source == "Cámara":
    photo = st.camera_input("Apunta hacia tu mascota")
    if photo:
        image_bytes = photo.getvalue()
else:
    uploaded = st.file_uploader(
        "Sube una foto JPG/PNG",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded:
        image_bytes = uploaded.read()

if image_bytes:
    st.image(image_bytes, caption="Imagen capturada", width=300)

    if st.button("🔍 Analizar con IA", use_container_width=True, type="primary"):
        with st.spinner("Claude está analizando la imagen..."):
            animal = detect_animal(image_bytes)

        st.session_state.last_animal = animal

        if animal == "dog":
            st.success("🐕 ¡Perro detectado!")
            resp = send_led_command("dog")
            if "error" in resp:
                st.error(resp["error"])
            add_log("📷 IA → perro → LED + puerta abierta")

        elif animal == "cat":
            st.success("🐈 ¡Gato detectado!")
            resp = send_led_command("cat")
            if "error" in resp:
                st.error(resp["error"])
            add_log("📷 IA → gato → LED + puerta abierta")

        else:
            st.info("No se detectó ninguna mascota.")
            add_log("📷 IA → sin mascota")

st.divider()

# ─── Sección 4: Log de actividad ───────────────────────────────────────────
st.subheader("📋 Registro de actividad")
if st.session_state.log:
    for entry in st.session_state.log:
        st.text(entry)
else:
    st.caption("Sin actividad aún.")

# ─── Configuración ─────────────────────────────────────────────────────────
with st.expander("⚙️ Configuración del sistema"):
    new_url = st.text_input("URL del ESP32", value=ESP32_URL)

    if st.button("Guardar URL"):
        ESP32_URL = new_url.strip()
        st.success("URL actualizada (solo esta sesión)")

    st.caption(
        f"API key Anthropic: {'✅ configurada' if ANTHROPIC_KEY else '❌ falta en .env'}"
    )

    if st.button("🔄 Actualizar estado"):
        st.rerun()
