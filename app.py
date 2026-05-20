import io
import uuid
import time
import numpy as np
from PIL import Image
import streamlit as st

# ===== NUEVOS IMPORTS PARA VOZ =====
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events

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

TOPIC_COMMAND = "DoorPet/command"
TOPIC_STATUS = "DoorPet/status"
TOPIC_DETECTION = "DoorPet/detection"


# =========================================================
# MODELO IA
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

    client = mqtt.Client(client_id="Sulusa" + uuid.uuid4().hex[:8])

    client.on_message = on_message

    client.connect(MQTT_BROKER, MQTT_PORT, 60)

    client.subscribe(TOPIC_STATUS)
    client.subscribe(TOPIC_DETECTION)

    client.loop_start()

    return client


mqtt_client = get_mqtt_client()


def send_command(command: str):

    result = mqtt_client.publish(
        TOPIC_COMMAND,
        command,
        retain=False
    )

    try:
        result.wait_for_publish()
    except Exception:
        pass

    time.sleep(0.3)


# =========================================================
# DETECCION DE ANIMALES
# =========================================================
def detect_animal(image_bytes: bytes) -> str:

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB").resize((224, 224))

        x = preprocess_input(
            np.expand_dims(np.array(img), axis=0)
        )

        results = decode_predictions(
            model.predict(x, verbose=0),
            top=5
        )[0]

        dog_kw = [
            "dog", "retriever", "shepherd", "poodle",
            "terrier", "beagle", "husky", "bulldog",
            "chihuahua", "pug", "doberman",
            "rottweiler", "labrador",
            "malamute", "spaniel", "wolfhound"
        ]

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
st.set_page_config(
    page_title="DoorPet",
    page_icon="🐾",
    layout="centered",
)

# =========================================================
# SESSION STATE
# =========================================================
for key, val in [
    ("door_state", "desconocido"),
    ("last_animal", "none"),
    ("log", []),
]:
    if key not in st.session_state:
        st.session_state[key] = val


def add_log(msg: str):

    st.session_state.log.insert(0, msg)
    st.session_state.log = st.session_state.log[:10]


# =========================================================
# HEADER
# =========================================================
st.title("DoorPet 🐾")

st.markdown(
    "Control por voz · MQTT · IA"
)

st.divider()


# =========================================================
# ESTADO ACTUAL
# =========================================================
col1, col2 = st.columns(2)

with col1:

    st.subheader("Estado puerta")

    if st.session_state.door_state == "abierta":
        st.success("🔓 Abierta")

    elif st.session_state.door_state == "cerrada":
        st.error("🔒 Cerrada")

    else:
        st.info("Estado desconocido")


with col2:

    st.subheader("Ultima deteccion")

    if st.session_state.last_animal == "dog":
        st.success("🐕 Perro")

    elif st.session_state.last_animal == "cat":
        st.success("🐈 Gato")

    else:
        st.info("Sin detecciones")


st.divider()


# =========================================================
# CONTROL POR VOZ
# =========================================================
st.subheader("Control por voz")

st.write(
    "Presiona el botón y di: "
    "'abrir puerta' o 'cerrar puerta'"
)

# Botón de voz
stt_button = Button(
    label="🎤 Hablar",
    width=250
)

# JS reconocimiento
stt_button.js_on_event(
    "button_click",
    CustomJS(code="""
        var recognition = new webkitSpeechRecognition();

        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = "es-ES";

        recognition.onresult = function (e) {

            var value = "";

            for (var i = e.resultIndex; i < e.results.length; ++i) {

                if (e.results[i].isFinal) {
                    value += e.results[i][0].transcript;
                }
            }

            if (value != "") {

                document.dispatchEvent(
                    new CustomEvent(
                        "GET_TEXT",
                        {detail: value}
                    )
                );
            }
        };

        recognition.start();
    """)
)

# Streamlit <-> JS
result = streamlit_bokeh_events(
    stt_button,
    events="GET_TEXT",
    key="listen",
    refresh_on_update=False,
    override_height=75,
    debounce_time=0
)

# Procesar voz
if result:

    if "GET_TEXT" in result:

        voice_text = (
            result.get("GET_TEXT")
            .strip()
            .lower()
        )

        st.info(f"Comando reconocido: {voice_text}")

        # =========================================
        # ABRIR
        # =========================================
        if (
            "abrir puerta" in voice_text
            or "abre la puerta" in voice_text
            or "abrir" in voice_text
            or "abre" in voice_text
        ):

            send_command("open")

            st.session_state.door_state = "abierta"

            add_log(f"🎤 Voz → abrir ({voice_text})")

            st.success(
                "Puerta abierta por voz"
            )

            st.rerun()

        # =========================================
        # CERRAR
        # =========================================
        elif (
            "cerrar puerta" in voice_text
            or "cierra la puerta" in voice_text
            or "cerrar" in voice_text
            or "cierra" in voice_text
        ):

            send_command("close")

            st.session_state.door_state = "cerrada"

            add_log(f"🎤 Voz → cerrar ({voice_text})")

            st.success(
                "Puerta cerrada por voz"
            )

            st.rerun()

        else:

            st.warning(
                "Comando no reconocido"
            )


st.divider()


# =========================================================
# BOTONES MANUALES
# =========================================================
st.subheader("Control manual")

col_open, col_close = st.columns(2)

with col_open:

    if st.button(
        "🔓 Abrir puerta",
        use_container_width=True
    ):

        send_command("open")

        st.session_state.door_state = "abierta"

        add_log("Boton → abrir")

        st.rerun()


with col_close:

    if st.button(
        "🔒 Cerrar puerta",
        use_container_width=True
    ):

        send_command("close")

        st.session_state.door_state = "cerrada"

        add_log("Boton → cerrar")

        st.rerun()


st.divider()


# =========================================================
# DETECCION DE MASCOTAS
# =========================================================
st.subheader("Deteccion de mascotas")

source = st.radio(
    "Fuente",
    ["Camara", "Subir archivo"],
    horizontal=True,
)

image_bytes = None

if source == "Camara":

    photo = st.camera_input(
        "Tomar foto"
    )

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

    st.image(
        image_bytes,
        caption="Imagen",
        width=300
    )

    if st.button(
        "🔍 Analizar con IA",
        use_container_width=True
    ):

        with st.spinner(
            "Analizando..."
        ):

            animal = detect_animal(
                image_bytes
            )

        if animal == "dog":

            st.session_state.last_animal = "dog"

            send_command("dog")

            st.success(
                "🐕 Perro detectado"
            )

            add_log("IA → perro")

        elif animal == "cat":

            st.session_state.last_animal = "cat"

            send_command("cat")

            st.success(
                "🐈 Gato detectado"
            )

            add_log("IA → gato")

        else:

            st.session_state.last_animal = "none"

            send_command("close")

            st.info(
                "No se detecto mascota"
            )

            add_log("IA → none")


st.divider()


# =========================================================
# LOG
# =========================================================
st.subheader("Registro")

if st.session_state.log:

    for entry in st.session_state.log:
        st.write("•", entry)

else:
    st.info("Sin actividad")


st.divider()


# =========================================================
# MQTT INFO
# =========================================================
with st.expander("⚙️ Configuracion MQTT"):

    st.write("Broker:", MQTT_BROKER)
    st.write("Puerto:", MQTT_PORT)
    st.write("Topic comandos:", TOPIC_COMMAND)
    st.write("Topic estado:", TOPIC_STATUS)
    st.write("Topic deteccion:", TOPIC_DETECTION)
