# VerPCA.py
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.decomposition import PCA

FEATURES = ["Temp", "Humedad", "Gas", "Ph", "TempAmbiente", "HumedadAmbiente"]

def GraficoPCA(df: pd.DataFrame, entrada_escalada):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        scaler_path = os.path.join(base_dir, "scaler.pkl")
        model_path  = os.path.join(base_dir, "modelo_knn.pkl")
        pca3_path   = os.path.join(base_dir, "pca3.pkl")

        if not os.path.exists(scaler_path):
            return "<div class='alert alert-danger'>No se encuentra scaler.pkl. Entrena el modelo primero.</div>"
        if not os.path.exists(model_path):
            return "<div class='alert alert-danger'>No se encuentra modelo_knn.pkl. Entrena el modelo primero.</div>"
        scaler = joblib.load(scaler_path)
        knn    = joblib.load(model_path)

        faltantes = [c for c in FEATURES if c not in df.columns]
        if faltantes:
            return f"<div class='alert alert-danger'>Faltan columnas en DATA.csv para graficar: {faltantes}</div>"
        X = df[FEATURES].values
        Xs = scaler.transform(X)
        if os.path.exists(pca3_path):
            pca3 = joblib.load(pca3_path)
        else:
            pca3 = PCA(n_components=3, random_state=42).fit(Xs)
            try:
                joblib.dump(pca3, pca3_path)
            except Exception:

                pass
        X3 = pca3.transform(Xs)

        p0 = np.asarray(entrada_escalada)
        if p0.ndim == 1:
            p0 = p0.reshape(1, -1)

        x0 = pca3.transform(p0)
        y_plot = knn.predict(Xs)

        color_map = {
            "Muy Viable": "#1f77b4",  # azul
            "Viable":     "#ff7f0e",  # naranja
            "Poco Viable":"#2ca02c",  # verde
            "No Viable":  "#d62728",  # rojo
        }

        df_plot = pd.DataFrame({
            "PC1": X3[:, 0],
            "PC2": X3[:, 1],
            "PC3": X3[:, 2],
            "Casos": y_plot
        })

        fig = px.scatter_3d(
            df_plot,
            x="PC1", y="PC2", z="PC3",
            color="Casos",
            color_discrete_map=color_map,
            opacity=0.75,
            height=400,
        )
        k = getattr(knn, "n_neighbors", 5)
        dist, idx = knn.kneighbors(p0, n_neighbors=k)
        vecinos3 = X3[idx[0], :]
        fig.add_scatter3d(
            x=vecinos3[:, 0],
            y=vecinos3[:, 1],
            z=vecinos3[:, 2],
            mode="markers",
            marker=dict(
                size=8,
                color="rgba(0,0,0,0)",
                line=dict(color="black", width=2)
            ),
            name=f"{k} vecinos"
        )

        fig.add_scatter3d(
            x=[x0[0, 0]],
            y=[x0[0, 1]],
            z=[x0[0, 2]],
            mode="markers",
            marker=dict(size=10, color="crimson", symbol="x"),
            name="Dato actual"
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(itemsizing="constant"),
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3"
            )
        )
        html_div = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
        return html_div

    except Exception as e:
        return f"<div class='alert alert-danger'>Error generando PCA 3D: {e}</div>"
