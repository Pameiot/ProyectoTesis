from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sns

from EntrenarKNN import entrenar, mejorKnn
from EnvioMQTT import ConexionBroker, publidatosMQTT
from VerPCA import GraficoPCA
from Descarga import exportar_csv, exportar_excel

ConexionBroker()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "DATA.csv")
K_FILE = os.path.join(BASE_DIR, "k_usado.txt")

try:
    with open(K_FILE, "r", encoding="utf-8") as f:
        k_actual = int(f.read().strip())
except:
    k_actual = 4

app = Flask(__name__, template_folder="HTML")

modelo_knn = joblib.load(os.path.join(BASE_DIR, "modelo_knn.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
pca = joblib.load(os.path.join(BASE_DIR, "pca.pkl"))

datos_recibidos = []
resultados_procesados = []

encabezados_defecto = ["FECHA", "HORA", "TEMPERATURA", "HUMEDAD", "GAS", "PH", "T.AMBIENTE", "H.AMBIENTE"]


def _to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", ".")
    if s.lower() in ("", "null", "none", "nan"):
        return None
    return float(s)


def _ensure_csv():
    if os.path.exists(CSV_PATH):
        return
    df = pd.DataFrame(columns=["FechaHora", "Temp", "Humedad", "Gas", "Ph", "TempAmbiente", "HumedadAmbiente"])
    df.to_csv(CSV_PATH, sep=";", index=False, encoding="utf-8")


def _append_csv_row(fecha_hora_ddmmyyyy_hhmm, temp, hum, gas, ph, temp_amb, hum_amb):
    _ensure_csv()
    row = pd.DataFrame([{
        "FechaHora": fecha_hora_ddmmyyyy_hhmm,
        "Temp": temp,
        "Humedad": hum,
        "Gas": gas,
        "Ph": ph,
        "TempAmbiente": temp_amb,
        "HumedadAmbiente": hum_amb
    }])
    row.to_csv(CSV_PATH, sep=";", index=False, header=False, mode="a", encoding="utf-8")


@app.route("/css/<path:filename>")
def css_static(filename):
    return send_from_directory("CSS", filename)


@app.route("/recursos/<path:filename>")
def recursos_static(filename):
    return send_from_directory("Recursos", filename)


@app.route("/")
def home():
    pagina = int(request.args.get("pagina", 1))
    por_pagina = 20
    inicio = (pagina - 1) * por_pagina
    fin = inicio + por_pagina
    datos_pagina = datos_recibidos[::-1][inicio:fin]
    total_paginas = (len(datos_recibidos) + por_pagina - 1) // por_pagina
    return render_template(
        "index.html",
        encabezados=encabezados_defecto,
        datos=datos_pagina,
        pagina_actual=pagina,
        total_paginas=total_paginas
    )


@app.route("/ultimos", methods=["GET"])
def obtener_ultimos():
    cantidad = int(request.args.get("n", 10))
    ultimos = datos_recibidos[-cantidad:]
    ultimos = list(reversed(ultimos))
    return jsonify(ultimos)


@app.route("/grafica")
def grafica():
    return render_template(
        "grafica.html",
        fechas=[],
        temperaturas=[],
        humedades=[],
        phs=[],
        gases=[],
        t_ambiente=[],
        h_ambiente=[]
    )


@app.route("/historial")
def historial():
    try:
        if not os.path.exists(CSV_PATH):
            return jsonify({
                "labels": [],
                "temperaturas": [],
                "humedades": [],
                "gases": [],
                "phs": [],
                "t_amb": [],
                "h_amb": []
            })

        df = pd.read_csv(CSV_PATH, delimiter=";")
        if df.empty or "FechaHora" not in df.columns:
            return jsonify({
                "labels": [],
                "temperaturas": [],
                "humedades": [],
                "gases": [],
                "phs": [],
                "t_amb": [],
                "h_amb": []
            })

        df["FechaHora_dt"] = pd.to_datetime(df["FechaHora"], format="%d/%m/%Y %H:%M", errors="coerce")

        desde = request.args.get("desde")
        hasta = request.args.get("hasta")

        if desde and hasta:
            desde_dt = pd.to_datetime(desde + " 00:00:00", format="%Y-%m-%d %H:%M:%S", errors="coerce")
            hasta_dt = pd.to_datetime(hasta + " 23:59:59", format="%Y-%m-%d %H:%M:%S", errors="coerce")
            if pd.isna(desde_dt) or pd.isna(hasta_dt):
                return jsonify({
                    "labels": [],
                    "temperaturas": [],
                    "humedades": [],
                    "gases": [],
                    "phs": [],
                    "t_amb": [],
                    "h_amb": []
                })
            df = df[(df["FechaHora_dt"] >= desde_dt) & (df["FechaHora_dt"] <= hasta_dt)]

        df = df.dropna(subset=["FechaHora_dt"]).sort_values("FechaHora_dt")

        if df.empty:
            return jsonify({
                "labels": [],
                "temperaturas": [],
                "humedades": [],
                "gases": [],
                "phs": [],
                "t_amb": [],
                "h_amb": []
            })

        return jsonify({
            "labels": df["FechaHora"].astype(str).tolist(),
            "temperaturas": df["Temp"].astype(float).tolist(),
            "humedades": df["Humedad"].astype(float).tolist(),
            "gases": df["Gas"].astype(float).tolist(),
            "phs": df["Ph"].astype(float).tolist(),
            "t_amb": df["TempAmbiente"].astype(float).tolist(),
            "h_amb": df["HumedadAmbiente"].astype(float).tolist()
        })
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error al cargar historial: {e}"}), 500


@app.route("/knn")
def vista_knn():
    if not datos_recibidos:
        return render_template("knn.html", resultado=None, interpretacion=None, imagen_grafico=None, detalles=[])

    ultimo = datos_recibidos[-1]
    columnas_modelo = ["Temp", "Humedad", "Gas", "Ph", "TempAmbiente", "HumedadAmbiente"]

    try:
        entrada = pd.DataFrame([{
            "Temp": float(ultimo["TEMPERATURA"]),
            "Humedad": float(ultimo["HUMEDAD"]),
            "Gas": float(ultimo["GAS"]),
            "Ph": float(ultimo["PH"]),
            "TempAmbiente": float(ultimo["T.AMBIENTE"]),
            "HumedadAmbiente": float(ultimo["H.AMBIENTE"])
        }])[columnas_modelo]

        entrada_escalada = pd.DataFrame(scaler.transform(entrada), columns=columnas_modelo)
        resultado = modelo_knn.predict(entrada_escalada)[0]

        gas = entrada["Gas"].iloc[0]
        ph = entrada["Ph"].iloc[0]
        temp = entrada["Temp"].iloc[0]
        hum = entrada["Humedad"].iloc[0]

        explicacion = []

        if gas < 300:
            explicacion.append(f"El nivel de gas es {gas:.1f} ppm siendo de bajo nivel")
        elif gas < 1000:
            explicacion.append(f"El nivel de gas es {gas:.1f} ppm ,aceptable")
        else:
            explicacion.append(f"El nivel de gas es {gas:.1f} ppm, alto nivel")

        if ph < 6.5:
            explicacion.append(f"El pH es {ph:.2f}, demasiado ácido para el abono")
        elif ph > 8.5:
            explicacion.append(f"El pH es {ph:.2f}, demasiado alcalino para el abono")
        else:
            explicacion.append(f"El pH es {ph:.2f}, óptimo para abonos orgánicos")

        if 10 <= temp <= 40:
            explicacion.append(f"Temperatura: {temp:.1f} °C, adecuada para la descomposición")
        else:
            explicacion.append(f"Temperatura: {temp:.1f} °C, no adecuada para la descomposición")

        if 50 <= hum <= 85:
            explicacion.append(f"Humedad: {hum:.1f} %, adecuada para la descomposición")
        else:
            explicacion.append(f"Humedad: {hum:.1f} %, no adecuada para la descomposición")


        interpretacion = f"El entorno es clasificado como: {resultado}"

        resultados_procesados.append({
            "FechaHora": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "Temp": float(temp),
            "Humedad": float(hum),
            "Gas": float(gas),
            "Ph": float(ph),
            "TempAmb": float(ultimo["T.AMBIENTE"]),
            "HumAmb": float(ultimo["H.AMBIENTE"]),
            "Viabilidad": resultado,
            "Eval_Gas": explicacion[0],
            "Eval_Ph": explicacion[1],
            "Eval_Temp": explicacion[2],
            "Eval_Humedad": explicacion[3]
        })

        mensaje_str = f"{resultado};{interpretacion};{explicacion[0]};{explicacion[1]};{explicacion[2]};{explicacion[3]}"
        mensaje_puro = f"{temp:.2f};{hum:.2f};{gas:.2f};{ph:.2f};{float(ultimo['T.AMBIENTE']):.2f};{float(ultimo['H.AMBIENTE']):.2f}"

        try:
            publidatosMQTT(mensaje_str, mensaje_puro)
        except Exception as e:
            print("MQTT error al publicar:", e)

        df = pd.read_csv(CSV_PATH, delimiter=";")
        imagen_grafico = GraficoPCA(df, entrada_escalada)

    except Exception as e:
        resultado = f"Error al predecir: {e}"
        interpretacion = None
        imagen_grafico = None
        explicacion = []

    return render_template(
        "knn.html",
        resultado=resultado,
        interpretacion=interpretacion,
        imagen_grafico=imagen_grafico,
        detalles=explicacion
    )


@app.route("/entrenamiento", methods=["GET", "POST"])
def entrenamiento():
    global k_actual, modelo_knn, scaler, pca
    try:
        if request.method == "POST":
            if "automatico" in request.form:
                precision, error, y_real, y_pred, nuevo_k, filas, columnas = mejorKnn()
                k_actual = int(nuevo_k)
            else:
                nuevo_k = request.form.get("nuevo_k", type=int)
                if nuevo_k is None:
                    nuevo_k = k_actual
                if nuevo_k < 1:
                    nuevo_k = 1
                elif nuevo_k > 13:
                    nuevo_k = 13
                if nuevo_k % 2 == 0:
                    nuevo_k += 1
                if nuevo_k > 13:
                    nuevo_k = 13
                k_actual = int(nuevo_k)
                precision, error, y_real, y_pred, _, filas, columnas = entrenar(k_actual)

            try:
                with open(K_FILE, "w", encoding="utf-8") as f:
                    f.write(str(k_actual))
            except:
                pass
        else:
            precision, error, y_real, y_pred, _, filas, columnas = entrenar(k_actual)

        modelo_knn = joblib.load(os.path.join(BASE_DIR, "modelo_knn.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
        pca = joblib.load(os.path.join(BASE_DIR, "pca.pkl"))

        labels_v = ["Muy Viable", "Viable", "Poco Viable", "No Viable"]
        cm = confusion_matrix(y_real, y_pred, labels=labels_v)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_v, yticklabels=labels_v)
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.title("Matriz de Confusión")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        grafico_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        return render_template(
            "entrenamiento.html",
            precision=precision,
            error=error,
            grafico=grafico_base64,
            k=k_actual,
            filas=filas,
            columnas=columnas
        )
    except Exception as e:
        return f"Error durante el entrenamiento: {e}", 500


@app.route("/descargar_csv")
def descargar_csv():
    return exportar_csv(resultados_procesados)


@app.route("/descargar_excel")
def descargar_excel():
    return exportar_excel(resultados_procesados)


@app.route("/datos", methods=["POST"])
def recibir_datos():
    try:
        data = request.get_json(force=True) or {}

        campos = ["TEMPERATURA", "HUMEDAD", "GAS", "PH", "T.AMBIENTE", "H.AMBIENTE"]
        for campo in campos:
            if campo not in data:
                return {"status": "error", "mensaje": f"Campo faltante: {campo}"}, 400

        temp = _to_float(data.get("TEMPERATURA"))
        hum = _to_float(data.get("HUMEDAD"))
        gas = _to_float(data.get("GAS"))
        ph = _to_float(data.get("PH"))
        temp_amb = _to_float(data.get("T.AMBIENTE"))
        hum_amb = _to_float(data.get("H.AMBIENTE"))

        if None in (temp, hum, gas, ph, temp_amb, hum_amb):
            return {"status": "error", "mensaje": "Hay valores inválidos (null/vacíos/no numéricos)."}, 400

        now = datetime.now()
        fecha = data.get("FECHA") or now.strftime("%Y-%m-%d")
        hora = data.get("HORA") or now.strftime("%H:%M")

        data_normalizada = {
            "FECHA": str(fecha),
            "HORA": str(hora),
            "TEMPERATURA": float(temp),
            "HUMEDAD": float(hum),
            "GAS": float(gas),
            "PH": float(ph),
            "T.AMBIENTE": float(temp_amb),
            "H.AMBIENTE": float(hum_amb)
        }

        datos_recibidos.append(data_normalizada)

        try:
            fecha_hora_csv = now.strftime("%d/%m/%Y %H:%M")
            _append_csv_row(fecha_hora_csv, float(temp), float(hum), float(gas), float(ph), float(temp_amb), float(hum_amb))
        except Exception as e:
            print("Error guardando en CSV:", e)

        return {"status": "ok", "mensaje": "Datos recibidos y guardados"}, 200
    except Exception as e:
        return {"status": "error", "mensaje": f"Error en datos: {e}"}, 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

