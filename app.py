import io
import uuid
import time
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
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
TOPIC_STATUS  = "smartdoor/status"
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
    client = mqtt.Client(client_id="smartdoor-" + uuid.uuid4().hex[:8])
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(TOPIC_STATUS)
    client.subscribe(TOPIC_DETECTION)
    client.loop_start()
    return client

mqtt_client = get_mqtt_client()

def send_command(command):
    mqtt_client.publish(TOPIC_COMMAND, command, retain=False)
    time.sleep(0.4)


# =========================================================
# DETECCION DE ANIMALES
# =========================================================
def detect_animal(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        x = preprocess_input(np.expand_dims(np.array(img), axis=0))
        results = decode_predictions(model.predict(x, verbose=0), top=5)[0]
        dog_kw = ["dog","retriever","shepherd","poodle","terrier","beagle",
                  "husky","bulldog","chihuahua","pug","doberman","rottweiler",
                  "labrador","malamute","spaniel","wolfhound"]
        for _, label, _ in results:
            label = label.lower()
            if "cat" in label:
                return "cat"
            if any(w in label for w in dog_kw):
                return "dog"
        return "none"
    except Exception:
        return "none"


# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="Puerta Inteligente", page_icon="🚪", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #1A1A1A;
}
.stApp { background-color: #F7F5F2; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 2rem 4rem 2rem; max-width: 680px; }

/* Titulos */
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
    color: #555555 !important;
    margin-bottom: 0.75rem !important;
}

/* Caption */
.stCaption p { font-size: 0.85rem; color: #555555; margin-top: 2px; }

/* Divider */
hr { border: none; border-top: 1px solid #DEDAD5; margin: 1.6rem 0; }

/* Alerts */
div[data-testid="stSuccess"],
div[data-testid="stError"],
div[data-testid="stInfo"],
div[data-testid="stWarning"] {
    border-radius: 12px !important;
    border: none !important;
    padding: 0.9rem 1.1rem !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}
div[data-testid="stSuccess"] { background-color: #D4EDDA !important; color: #1A4D26 !important; }
div[data-testid="stError"]   { background-color: #FAD7D5 !important; color: #6B1512 !important; }
div[data-testid="stInfo"]    { background-color: #D0E4F7 !important; color: #103A60 !important; }
div[data-testid="stWarning"] { background-color: #FEF3CD !important; color: #5C3D00 !important; }

/* Botones generales */
.stButton > button {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
    border: 1.5px solid #BFBAB4 !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.1rem !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
.stButton > button:hover {
    background-color: #EDEAE5 !important;
    border-color: #9A9490 !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.10) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Boton abrir */
.btn-open .stButton > button {
    background-color: #1E5C2A !important;
    color: #FFFFFF !important;
    border-color: #1E5C2A !important;
}
.btn-open .stButton > button:hover { background-color: #174D23 !important; }

/* Boton cerrar */
.btn-close .stButton > button {
    background-color: #A62B23 !important;
    color: #FFFFFF !important;
    border-color: #A62B23 !important;
}
.btn-close .stButton > button:hover { background-color: #8C2420 !important; }

/* Radio */
.stRadio label p, .stRadio label span { color: #1A1A1A !important; font-size: 0.875rem !important; }

/* File uploader */
.stFileUploader { border-radius: 12px !important; }
[data-testid="stFileUploader"] label { color: #1A1A1A !important; font-weight: 500 !important; }

/* Camera */
.stCameraInput video, .stCameraInput canvas { border-radius: 12px !important; }
[data-testid="stCameraInput"] label { color: #1A1A1A !important; font-weight: 500 !important; }

/* Imagen */
.stImage img { border-radius: 12px !important; box-shadow: 0 4px 16px rgba(0,0,0,0.10) !important; }

/* Expander */
.stExpander { border: 1.5px solid #DEDAD5 !important; border-radius: 14px !important; background: #FDFCFB !important; }
.stExpander summary p { color: #1A1A1A !important; font-weight: 600 !important; }

/* Tarjeta estado */
.state-card {
    background: #FFFFFF;
    border: 1.5px solid #DEDAD5;
    border-radius: 14px;
    padding: 0.85rem 1.1rem;
    font-size: 0.95rem;
    font-weight: 600;
    color: #1A1A1A;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.state-card.open  { border-color: #5A9E6A; background: #EBF7EE; color: #1A4D26; }
.state-card.closed { border-color: #C47570; background: #FAEAEA; color: #6B1512; }
.state-card.unknown { color: #555555; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================
for key, val in [("door_state","desconocido"), ("last_animal","none"), ("log",[])]:
    if key not in st.session_state:
        st.session_state[key] = val


def add_log(msg):
    st.session_state.log.insert(0, msg)
    st.session_state.log = st.session_state.log[:10]


# =========================================================
# HEADER
# =========================================================
st.title("Puerta Inteligente")
st.caption("Control por voz · Botones · IA · MQTT")
st.markdown("<div style='margin-top:0.4rem'></div>", unsafe_allow_html=True)

# =========================================================
# ESTADO ACTUAL
# =========================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Estado de la puerta")
    s = st.session_state.door_state
    if s in ("abierta", "open"):
        st.markdown('<div class="state-card open">🔓 &nbsp;Abierta</div>', unsafe_allow_html=True)
    elif s in ("cerrada", "closed"):
        st.markdown('<div class="state-card closed">🔒 &nbsp;Cerrada</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="state-card unknown">&#8212; &nbsp;Desconocido</div>', unsafe_allow_html=True)

with col2:
    st.subheader("Ultima deteccion")
    a = st.session_state.last_animal
    if a == "dog":
        st.markdown('<div class="state-card">🐕 &nbsp;Perro</div>', unsafe_allow_html=True)
    elif a == "cat":
        st.markdown('<div class="state-card">🐈 &nbsp;Gato</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="state-card unknown">&#8212; &nbsp;Sin animal</div>', unsafe_allow_html=True)

st.divider()

# =========================================================
# CONTROL POR VOZ
# Solucion definitiva: componente HTML embebido con su propio
# boton. Al detectar la voz llama a window.parent.postMessage
# para pasar el texto a Streamlit, que lo recibe via
# st.session_state y procesa el comando.
# =========================================================
st.subheader("Control por voz")

# ── Leer query param enviado por el iframe ──────────────────
# El iframe no puede modificar session_state directamente.
# La solucion: cuando el usuario habla, el iframe hace
#   window.parent.location.href = "?voice=<texto>"
# Streamlit recarga con ese query param, lo leemos aqui,
# procesamos el comando y limpiamos el param.
qp = st.query_params
raw_voice = qp.get("voice", "")

if raw_voice:
    # Limpiar el param de la URL de inmediato
    st.query_params.clear()

    t = raw_voice.strip().lower()
    open_words  = ["open", "abre", "abrir", "abre la puerta"]
    close_words = ["close", "closed", "cierra", "cerrar", "cierra la puerta"]

    if any(w in t for w in open_words):
        send_command("open")
        st.session_state.door_state = "abierta"
        add_log("Voz -> abrir (" + t + ")")
    elif any(w in t for w in close_words):
        send_command("close")
        st.session_state.door_state = "cerrada"
        add_log("Voz -> cerrar (" + t + ")")
    else:
        st.warning("No reconoci un comando en: " + t)

# ── Componente HTML del microfono ───────────────────────────
# Al reconocer la voz redirige la ventana padre a ?voice=<texto>
# lo que dispara un rerun de Streamlit con el query param listo.
voice_html = """
<!DOCTYPE html>
<html>
<head>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0;
      font-family: 'DM Sans', 'Segoe UI', sans-serif; }
  body { background: transparent; padding: 4px 0; }

  #mic-btn {
    width: 100%;
    padding: 10px 18px;
    font-size: 14px;
    font-weight: 600;
    color: #1A1A1A;
    background: #FFFFFF;
    border: 1.5px solid #BFBAB4;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.15s;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }
  #mic-btn:hover { background: #EDEAE5; border-color: #9A9490; }
  #mic-btn.listening {
    background: #FAEAEA;
    border-color: #C47570;
    color: #6B1512;
    animation: pulse 1s infinite;
  }
  @keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.65; }
  }
  #status {
    margin-top: 8px;
    font-size: 13px;
    color: #555555;
    min-height: 20px;
    text-align: center;
  }
  #status.heard { color: #1A4D26; font-weight: 600; }
  #status.error { color: #6B1512; }
</style>
</head>
<body>
<button id="mic-btn" onclick="startListening()">🎤 &nbsp;Hablar</button>
<div id="status">Haz clic para hablar</div>

<script>
const btn    = document.getElementById('mic-btn');
const status = document.getElementById('status');
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

function startListening() {
  if (!SR) {
    status.textContent = 'Tu navegador no soporta reconocimiento de voz.';
    status.className = 'error';
    return;
  }
  const r = new SR();
  r.lang = 'es-ES';
  r.interimResults = false;
  r.maxAlternatives = 1;

  btn.innerHTML = '🔴 &nbsp;Escuchando...';
  btn.classList.add('listening');
  btn.disabled = true;
  status.textContent = 'Escuchando...';
  status.className = '';

  r.onresult = (e) => {
    const text = e.results[0][0].transcript;
    status.textContent = 'Escuche: ' + text;
    status.className = 'heard';
    // Redirigir la ventana padre con el texto como query param
    // Esto dispara un rerun de Streamlit con ?voice=<texto>
    const encoded = encodeURIComponent(text);
    window.parent.location.href = window.parent.location.pathname + '?voice=' + encoded;
  };

  r.onerror = (e) => {
    status.textContent = 'Error: ' + e.error + '. Intenta de nuevo.';
    status.className = 'error';
    reset();
  };

  r.onend = () => { reset(); };
  r.start();
}

function reset() {
  btn.innerHTML = '🎤 &nbsp;Hablar';
  btn.classList.remove('listening');
  btn.disabled = false;
}
</script>
</body>
</html>
"""

components.html(voice_html, height=90, scrolling=False)

st.divider()

# =========================================================
# BOTONES MANUALES
# =========================================================
st.subheader("Control manual")

col_open, col_close = st.columns(2)

with col_open:
    with st.container():
        st.markdown('<div class="btn-open">', unsafe_allow_html=True)
        if st.button("🔓  Abrir puerta", use_container_width=True):
            send_command("open")
            st.session_state.door_state = "abierta"
            add_log("Boton -> abrir")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

with col_close:
    with st.container():
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

source = st.radio("Fuente", ["Camara", "Subir archivo"], horizontal=True, label_visibility="collapsed")
image_bytes = None

if source == "Camara":
    photo = st.camera_input("Toma una foto")
    if photo:
        image_bytes = photo.getvalue()
else:
    uploaded = st.file_uploader("Sube una imagen", type=["jpg","jpeg","png"])
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
            st.success("🐕 Perro detectado — puerta abierta")
            add_log("IA -> perro")
            st.rerun()
        elif animal == "cat":
            st.session_state.last_animal = "cat"
            st.session_state.door_state = "abierta"
            send_command("cat")
            st.success("🐈 Gato detectado — puerta abierta")
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
        opacity = max(0.4, 1.0 - i * 0.07)
        st.markdown(
            "<p style='font-size:0.83rem;"
            "color:rgba(26,26,26," + str(opacity) + ");"
            "padding:0.35rem 0;"
            "border-bottom:1px solid #DEDAD5;"
            "margin:0;'>" + entry + "</p>",
            unsafe_allow_html=True,
        )
else:
    st.markdown("<p style='font-size:0.83rem;color:#888;'>Sin actividad aun</p>", unsafe_allow_html=True)

# =========================================================
# CONFIGURACION MQTT
# =========================================================
st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

with st.expander("⚙️  Configuracion MQTT"):
    rows = [
        ("Broker", MQTT_BROKER),
        ("Puerto", str(MQTT_PORT)),
        ("Topic comandos", TOPIC_COMMAND),
        ("Topic estado", TOPIC_STATUS),
        ("Topic deteccion", TOPIC_DETECTION),
    ]
    html_rows = "".join(
        "<div style='margin-bottom:0.5rem;'>"
        "<span style='font-size:0.72rem;text-transform:uppercase;"
        "letter-spacing:0.08em;color:#555555;font-weight:600;'>" + label + "</span>"
        "<br><span style='color:#1A1A1A;font-size:0.88rem;'>" + value + "</span></div>"
        for label, value in rows
    )
    st.markdown("<div style='padding:0.2rem 0;'>" + html_rows + "</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
    if st.button("🔄  Actualizar estado"):
        st.rerun()
