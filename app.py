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
TOPIC_COMMAND  = "smartdoor/command"
TOPIC_STATUS   = "smartdoor/status"
TOPIC_DETECTION = "smartdoor/detection"


# =========================================================
# MODELO
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
    c = mqtt.Client(client_id="smartdoor-" + uuid.uuid4().hex[:8])
    c.on_message = on_message
    c.connect(MQTT_BROKER, MQTT_PORT, 60)
    c.subscribe(TOPIC_STATUS)
    c.subscribe(TOPIC_DETECTION)
    c.loop_start()
    return c

mqtt_client = get_mqtt_client()

def send_command(command):
    mqtt_client.publish(TOPIC_COMMAND, command, retain=False)
    time.sleep(0.4)


# =========================================================
# DETECCION
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
            if "cat" in label:  return "cat"
            if any(w in label for w in dog_kw): return "dog"
        return "none"
    except Exception:
        return "none"


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Puerta Inteligente", page_icon="🚪", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: #1A1A1A; }
.stApp { background-color: #F7F5F2; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 2rem 4rem 2rem; max-width: 680px; }

h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.1rem !important; color: #1A1A1A !important;
    letter-spacing: -0.5px; margin-bottom: 0 !important;
}
h3 {
    font-size: 0.7rem !important; font-weight: 600 !important;
    letter-spacing: 0.12em !important; text-transform: uppercase !important;
    color: #444444 !important; margin-bottom: 0.75rem !important;
}
.stCaption p { font-size: 0.85rem; color: #555555; margin-top: 2px; }
hr { border: none; border-top: 1px solid #DEDAD5; margin: 1.6rem 0; }

div[data-testid="stSuccess"] { background-color: #D4EDDA !important; color: #1A4D26 !important; border-radius: 12px !important; border: none !important; font-weight: 500 !important; }
div[data-testid="stError"]   { background-color: #FAD7D5 !important; color: #6B1512 !important; border-radius: 12px !important; border: none !important; font-weight: 500 !important; }
div[data-testid="stInfo"]     { background-color: #D0E4F7 !important; color: #103A60 !important; border-radius: 12px !important; border: none !important; font-weight: 500 !important; }
div[data-testid="stWarning"] { background-color: #FEF3CD !important; color: #5C3D00 !important; border-radius: 12px !important; border: none !important; font-weight: 500 !important; }

.stButton > button {
    background-color: #FFFFFF !important; color: #1A1A1A !important;
    border: 1.5px solid #BFBAB4 !important; border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.875rem !important;
    font-weight: 600 !important; padding: 0.6rem 1.1rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background-color: #EDEAE5 !important; border-color: #9A9490 !important;
    transform: translateY(-1px) !important;
}
.btn-open .stButton > button  { background-color: #1E5C2A !important; color: #FFFFFF !important; border-color: #1E5C2A !important; }
.btn-open .stButton > button:hover  { background-color: #174D23 !important; }
.btn-close .stButton > button { background-color: #A62B23 !important; color: #FFFFFF !important; border-color: #A62B23 !important; }
.btn-close .stButton > button:hover { background-color: #8C2420 !important; }

.stRadio label p, .stRadio label span { color: #1A1A1A !important; font-size: 0.875rem !important; }
[data-testid="stFileUploader"] label  { color: #1A1A1A !important; font-weight: 500 !important; }
[data-testid="stCameraInput"] label   { color: #1A1A1A !important; font-weight: 500 !important; }
.stImage img { border-radius: 12px !important; box-shadow: 0 4px 16px rgba(0,0,0,0.10) !important; }
.stExpander { border: 1.5px solid #DEDAD5 !important; border-radius: 14px !important; background: #FDFCFB !important; }
.stExpander summary p { color: #1A1A1A !important; font-weight: 600 !important; }

.state-card {
    background: #FFFFFF; border: 1.5px solid #DEDAD5; border-radius: 14px;
    padding: 0.85rem 1.1rem; font-size: 0.95rem; font-weight: 600;
    color: #1A1A1A; box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.state-card.open   { border-color: #5A9E6A; background: #EBF7EE; color: #1A4D26; }
.state-card.closed { border-color: #C47570; background: #FAEAEA; color: #6B1512; }
.state-card.unknown { color: #555555; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================
defaults = {"door_state": "desconocido", "last_animal": "none", "log": []}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def add_log(msg):
    st.session_state.log.insert(0, msg)
    st.session_state.log = st.session_state.log[:10]


# =========================================================
# HEADER
# =========================================================
st.title("Puerta Inteligente")
st.caption("Control por voz · Botones · IA · MQTT")

# =========================================================
# ESTADO ACTUAL
# =========================================================
st.markdown("<div style='margin-top:0.4rem'></div>", unsafe_allow_html=True)
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
# CONTROL POR VOZ (MÓDULO CORREGIDO)
# =========================================================
st.subheader("Control por voz")

# Usamos una estructura HTML nativa dentro de un iframe controlado por Streamlit.
# Esto asegura que el navegador otorgue de forma confiable los permisos del micrófono.
voice_html = """
<div style="display:flex; flex-direction:column; gap:8px; font-family:'DM Sans', sans-serif;">
  <button id="voice-btn" onclick="startVoice()" style="
    width:100%; padding:10px 18px; font-size:14px; font-weight:600;
    color:#1A1A1A; background:#FFFFFF; border:1.5px solid #BFBAB4;
    border-radius:12px; cursor:pointer; transition:all 0.15s;">
    🎤&nbsp; Hablar
  </button>
  <div id="voice-status" style="font-size:13px; color:#555555; text-align:center; min-height:18px;">
    Haz clic para hablar
  </div>
</div>

<script>
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  
  function startVoice() {
    if (!SR) {
      alert('Tu navegador no soporta reconocimiento de voz. Usa Google Chrome.');
      return;
    }

    const btn = document.getElementById('voice-btn');
    const statusEl = document.getElementById('voice-status');

    const r = new SR();
    r.lang = 'es-ES'; 
    r.interimResults = false;
    r.maxAlternatives = 1;

    btn.innerHTML = '🔴&nbsp; Escuchando...';
    btn.style.background = '#FAEAEA';
    btn.style.borderColor = '#C47570';
    btn.style.color = '#6B1512';
    btn.disabled = true;
    statusEl.textContent = 'Escuchando... habla ahora';

    r.onresult = function(e) {
      const text = e.results[0][0].transcript;
      statusEl.textContent = 'Procesando: "' + text + '"';
      
      // Enviamos de vuelta de manera segura la cadena de texto a Streamlit 
      // usando la API de comunicación nativa de componentes
      window.parent.postMessage({
        type: 'streamlit:setComponentValue',
        value: text
      }, '*');
    };

    r.onerror = function(e) {
      statusEl.textContent = 'Error: ' + e.error + '. Intenta de nuevo.';
      resetBtn();
    };

    r.onend = function() { 
      resetBtn(); 
    };
    
    r.start();
  }

  function resetBtn() {
    const btn = document.getElementById('voice-btn');
    if (!btn) return;
    btn.innerHTML = '🎤&nbsp; Hablar';
    btn.style.background = '#FFFFFF';
    btn.style.borderColor = '#BFBAB4';
    btn.style.color = '#1A1A1A';
    btn.disabled = false;
  }
</script>
"""

# Renderizamos el componente y atrapamos su retorno en la variable voice_command
with st.container():
    voice_command = components.html(voice_html, height=75)

# Procesar los comandos de voz interceptados por el componente de manera directa
if voice_command:
    t = voice_command.strip().lower()
    
    open_words  = ["open", "abre", "abrir", "abre la puerta", "abrir la puerta"]
    close_words = ["close", "closed", "cierra", "cerrar", "cierra la puerta", "cerrar la puerta"]

    if any(w in t for w in open_words):
        send_command("open")
        st.session_state.door_state = "abierta"
        add_log("Voz -> abrir: " + t)
        st.rerun()
    elif any(w in t for w in close_words):
        send_command("close")
        st.session_state.door_state = "cerrada"
        add_log("Voz -> cerrar: " + t)
        st.rerun()
    else:
        add_log("Voz no reconocida: " + t)
        st.rerun()

st.divider()


# =========================================================
# BOTONES MANUALES
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

source = st.radio("Fuente", ["Camara", "Subir archivo"],
                  horizontal=True, label_visibility="collapsed")
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
            "<p style='font-size:0.83rem; color:rgba(26,26,26,"
            + str(opacity) + "); padding:0.35rem 0; "
            + "border-bottom:1px solid #DEDAD5; margin:0;'>"
            + entry + "</p>",
            unsafe_allow_html=True,
        )
else:
    st.markdown("<p style='font-size:0.83rem;color:#888;'>Sin actividad aun</p>",
                unsafe_allow_html=True)


# =========================================================
# CONFIGURACION MQTT
# =========================================================
st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

with st.expander("⚙️  Configuracion MQTT"):
    rows = [("Broker", MQTT_BROKER), ("Puerto", str(MQTT_PORT)),
            ("Topic comandos", TOPIC_COMMAND), ("Topic estado", TOPIC_STATUS),
            ("Topic deteccion", TOPIC_DETECTION)]
    html = "<div style='display:grid;gap:0.5rem;'>"
    for label, value in rows:
        html += (
            "<div><span style='font-size:0.72rem;text-transform:uppercase;"
            "letter-spacing:0.08em;color:#555;font-weight:600;'>" + label +
            "</span><br><span style='color:#1A1A1A;font-size:0.88rem;'>" + value +
            "</span></div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    st.markdown("<div style='margin-top:0.6rem'></div>", unsafe_allow_html=True)
    if st.button("🔄  Actualizar estado"):
        st.rerun()
