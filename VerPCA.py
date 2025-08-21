import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import os
import joblib

# Cargar el scaler entrenado
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

def GraficoPCA(df, entrada_escalada, pca):
    try:
        # Etiquetar datos para la gráfica
        def etiqueta_viabilidad(fila):
            criterios_viables = 0
            if 6.5 <= fila['Ph'] <= 8.5:
                criterios_viables += 1
            if 50 <= fila['Humedad'] <= 85:
                criterios_viables += 1
            if 10 <= fila['Temp'] <= 40:
                criterios_viables += 1
            if fila['Gas'] < 1000:
                criterios_viables += 1

            if criterios_viables == 4:
                return "Muy Viable"
            elif criterios_viables == 3:
                return "Viable"
            elif criterios_viables == 2:
                return "Poco Viable"
            else:
                return "No Viable"

        df["Viabilidad"] = df.apply(etiqueta_viabilidad, axis=1)

        # Variables a usar
        variables = ["Temp", "Humedad", "Gas", "Ph", "TempAmbiente", "HumedadAmbiente"]
        X = df[variables].values

        # Escalar los datos ANTES de aplicar PCA
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        df["PCA1"] = X_pca[:, 0]
        df["PCA2"] = X_pca[:, 1]

        # Dato actual ya viene escalado — aplicar clipping para evitar deformaciones
        entrada_clip = entrada_escalada.clip(lower=-3, upper=3)  # ← esto es lo nuevo
        entrada_pca = pca.transform(entrada_clip)

        # Gráfico
        fig, ax = plt.subplots(figsize=(6, 4))
        for tipo in ["Muy Viable", "Viable", "Poco Viable", "No Viable"]:
            sub = df[df["Viabilidad"] == tipo]
            ax.scatter(sub["PCA1"], sub["PCA2"], label=tipo, alpha=0.6)

        ax.scatter(entrada_pca[:, 0], entrada_pca[:, 1], color='red', marker='X', s=150, label='Dato Actual')
        ax.set_xlabel("Componente 1")
        ax.set_ylabel("Componente 2")
        ax.legend()

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        imagen_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        return imagen_base64

    except Exception as e:
        return None
