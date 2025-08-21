import pandas as pd
import tempfile
from flask import send_file

def exportar_csv(resultados_procesados):
    try:
        if not resultados_procesados:
            return "No hay datos procesados para exportar", 400
        df = pd.DataFrame(resultados_procesados)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8", newline="") as tmp:
            ruta_csv = tmp.name
            df.to_csv(ruta_csv, index=False, sep=';')
        return send_file(ruta_csv, as_attachment=True, download_name="reporte_knn.csv", mimetype="text/csv")
    except Exception as e:
        return f"Error al generar CSV: {e}", 500

def exportar_excel(resultados_procesados):
    try:
        if not resultados_procesados:
            return "No hay datos procesados para exportar", 400
        df = pd.DataFrame(resultados_procesados)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            ruta_excel = tmp.name
            df.to_excel(ruta_excel, index=False)
        return send_file(ruta_excel, as_attachment=True, download_name="reporte_knn.xlsx", mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        return f"Error al generar Excel: {e}", 500
