import pandas as pd
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import joblib

# Ruta donde está el archivo de datos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATOS = os.path.join(BASE_DIR, "DATA.csv")

def etiqueta_viabilidad(fila):
    ph = fila['Ph']
    humedad = fila['Humedad']
    temp = fila['Temp']
    gas = fila['Gas']

    criterios_viables = 0

    if 6.5 <= ph <= 8.5:
        criterios_viables += 1
    if 50 <= humedad <= 85:
        criterios_viables += 1
    if 10 <= temp <= 40:
        criterios_viables += 1
    if gas < 1000:
        criterios_viables += 1

    if criterios_viables == 4:
        return "Muy Viable"
    elif criterios_viables == 3:
        return "Viable"
    elif criterios_viables == 2:
        return "Poco Viable"
    else:
        return "No Viable"

def PreparaDatos():
    try:
        df = pd.read_csv(DATOS, delimiter=';')
    except Exception as e:
        raise FileNotFoundError(f"No se pudo cargar el archivo: {e}")

    columnas_requeridas = ['Temp', 'Humedad', 'Gas', 'Ph', 'TempAmbiente', 'HumedadAmbiente']
    if not all(col in df.columns for col in columnas_requeridas):
        raise ValueError(f"Faltan columnas requeridas: {columnas_requeridas}")
    
    if df.isnull().sum().sum() > 0:
        df.fillna(df.mean(numeric_only=True), inplace=True)
    
    df['Viabilidad'] = df.apply(etiqueta_viabilidad, axis=1)
    
    features = df[columnas_requeridas]
    labels = df['Viabilidad']
    return features, labels, df

def entrenar(k=4):
    X, y, df = PreparaDatos()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train_scaled, y_train)

    y_pred = modelo.predict(X_test_scaled)
    precision = modelo.score(X_test_scaled, y_test) * 100
    error = 100 - precision

    pca = PCA(n_components=2)
    pca.fit(X_train_scaled)

    joblib.dump(modelo, os.path.join(BASE_DIR, "modelo_knn.pkl"))
    joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))
    joblib.dump(pca, os.path.join(BASE_DIR, "pca.pkl"))

    # Guardar k usado
    with open(os.path.join(BASE_DIR, "k_usado.txt"), "w") as f:
        f.write(str(k))

    return precision, error, y_test, y_pred, k, df.shape[0], df.shape[1]

def mejorKnn(max_k=20, num_folds=5):
    X, y, df = PreparaDatos()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    mejor_k = 1
    mejor_score = 0.0

    for k in range(1, min(max_k + 1, len(X))):
        scores = []

        for train_idx, test_idx in skf.split(X_scaled, y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            modelo = KNeighborsClassifier(n_neighbors=k)
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)

        promedio = sum(scores) / len(scores)

        if promedio > mejor_score:
            mejor_score = promedio
            mejor_k = k

    return entrenar(mejor_k)

def ValidacionCruzada(k=4, num_folds=5):
    X, y, df = PreparaDatos()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    precisiones = []

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precisiones.append(acc * 100)

    promedio_precision = sum(precisiones) / num_folds
    desviacion = pd.Series(precisiones).std()

    return promedio_precision, desviacion

if __name__ == "__main__":
    _, _, _, _, mejor_k, _, _ = mejorKnn()
    precision_promedio, desviacion = ValidacionCruzada(k=mejor_k, num_folds=5)
