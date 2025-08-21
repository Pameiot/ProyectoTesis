import paho.mqtt.client as mqtt

MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "MKNN"
MQTTDATOS = "PUBLIC_IOT"

cliente_mqtt = mqtt.Client()

def ConexionBroker():
    try:
        cliente_mqtt.connect(MQTT_BROKER, MQTT_PORT, 60)
        cliente_mqtt.loop_start()
    except Exception as e:
        print(f"Error conectando al broker MQTT: {e}")

def publidatosMQTT(mensaje_str, mensaje_puro):
    try:
        cliente_mqtt.publish(MQTT_TOPIC, mensaje_str)
        cliente_mqtt.publish(MQTTDATOS, mensaje_puro)
        print(f"Publicado en MQTT: {mensaje_str}")
        print(f"Publicado en MQTT ({MQTTDATOS}): {mensaje_puro}")
    except Exception as e:
        print(f"Error publicando en MQTT: {e}")
