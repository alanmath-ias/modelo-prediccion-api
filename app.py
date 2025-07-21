from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Cargar modelo entrenado
modelo = joblib.load('modelo_random_forest.pkl')

# Crear app Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "API de Predicción de Rendimiento Académico"

@app.route('/predecir', methods=['POST'])
def predecir():
    try:
        # Recibir JSON con los datos del estudiante
        datos = request.get_json()

        # Convertir a DataFrame
        df_nuevo = pd.DataFrame([datos])

        # Asegurar que el DataFrame tenga las mismas columnas del entrenamiento
        df_nuevo_encoded = pd.get_dummies(df_nuevo)
        df_nuevo_encoded = df_nuevo_encoded.reindex(columns=modelo.feature_names_in_, fill_value=0)

        # Realizar predicción
        prediccion = modelo.predict(df_nuevo_encoded)[0]

        return jsonify({'prediccion': prediccion})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Usa el puerto de Railway si existe
    app.run(host='0.0.0.0', port=port)