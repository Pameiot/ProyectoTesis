import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(BASE_DIR, "DATA.csv")

FEATURES = ["Temp", "Humedad", "Gas", "Ph", "TempAmbiente", "HumedadAmbiente"]
def etiqueta_viabilidad(fila):
    ph = fila['Ph']
    humedad = fila['Humedad']
    temp = fila['Temp']
    gas = fila['Gas']

    c = 0
    if 6.5 <= ph <= 8.5:      # pH
        c += 1
    if 50 <= humedad <= 85:   # Humedad
        c += 1
    if 10 <= temp <= 40:      # Temperatura del material
        c += 1
    if gas < 1000:            # Metano (ppm)
        c += 1

    if c == 4:
        return "Muy Viable"
    elif c == 3:
        return "Viable"
    elif c == 2:
        return "Poco Viable"
    else:
        return "No Viable"

def PreparaDatos():
    try:
        df = pd.read_csv(DATOS, delimiter=';')
    except Exception as e:
        raise FileNotFoundError(f"No se pudo cargar {DATOS}: {e}")

    faltantes = [c for c in FEATURES if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en DATA.csv: {faltantes}")

    if df.isnull().sum().sum() > 0:
        df.fillna(df.mean(numeric_only=True), inplace=True)

    if 'Viabilidad' not in df.columns:
        df['Viabilidad'] = df.apply(etiqueta_viabilidad, axis=1)

    X = df[FEATURES].copy()
    y = df['Viabilidad'].copy()
    return X, y, df

def entrenar(k=5):
    X, y, df = PreparaDatos()
    if k % 2 == 0:
        k += 1

    #80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler_eval = StandardScaler()
    X_train_s = scaler_eval.fit_transform(X_train)
    X_test_s  = scaler_eval.transform(X_test)

    modelo_eval = KNeighborsClassifier(n_neighbors=k, weights='distance')
    modelo_eval.fit(X_train_s, y_train)

    y_pred = modelo_eval.predict(X_test_s)
    precision = accuracy_score(y_test, y_pred) * 100.0
    error = 100.0 - precision
    pca2_eval = PCA(n_components=2, random_state=42).fit(X_train_s)
    scaler_all = StandardScaler().fit(X)          
    X_all_s = scaler_all.transform(X)

    modelo_all = KNeighborsClassifier(n_neighbors=k, weights='distance')
    modelo_all.fit(X_all_s, y)                     

    pca2_all = PCA(n_components=2, random_state=42).fit(X_all_s)
    pca3_all = PCA(n_components=3, random_state=42).fit(X_all_s)

    joblib.dump(modelo_all, os.path.join(BASE_DIR, "modelo_knn.pkl"))
    joblib.dump(scaler_all, os.path.join(BASE_DIR, "scaler.pkl"))
    joblib.dump(pca2_all,   os.path.join(BASE_DIR, "pca.pkl"))  
    joblib.dump(pca3_all,   os.path.join(BASE_DIR, "pca3.pkl")) 

    with open(os.path.join(BASE_DIR, "k_usado.txt"), "w") as f:
        f.write(str(k))

    return precision, error, y_test, y_pred, k, df.shape[0], df.shape[1]
def mejorKnn(max_k=21, num_folds=5):
    """
    Busca el mejor k IMPAR en [1, max_k] con validación cruzada estratificada.
    Escalado dentro de cada fold (sin fuga).
    Luego realiza `entrenar(mejor_k)` para producir artefactos finales.
    """
    X, y, _ = PreparaDatos()
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    mejor_k = 1
    mejor_score = 0.0

    for k in range(1, max(2, max_k) + 1, 2):
        scores = []

        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            modelo = KNeighborsClassifier(n_neighbors=k, weights='distance')
            modelo.fit(X_tr_s, y_tr)
            y_pred = modelo.predict(X_te_s)

            scores.append(accuracy_score(y_te, y_pred))

        promedio = sum(scores) / len(scores)
        if promedio > mejor_score:
            mejor_score = promedio
            mejor_k = k
    return entrenar(mejor_k)
def ValidacionCruzada(k=5, num_folds=5):
    if k % 2 == 0:
        k += 1

    X, y, _ = PreparaDatos()
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    precisiones = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        modelo = KNeighborsClassifier(n_neighbors=k, weights='distance')
        modelo.fit(X_tr_s, y_tr)
        y_pred = modelo.predict(X_te_s)

        precisiones.append(accuracy_score(y_te, y_pred) * 100.0)

    promedio_precision = sum(precisiones) / num_folds
    desviacion = pd.Series(precisiones).std()
    return promedio_precision, desviacion

if __name__ == "__main__":
    _, _, _, _, mejor_k, _, _ = mejorKnn()
    precision_promedio, desviacion = ValidacionCruzada(k=mejor_k, num_folds=5)
    print(f"Mejor k: {mejor_k} | CV: {precision_promedio:.2f}% ± {desviacion:.2f}")
