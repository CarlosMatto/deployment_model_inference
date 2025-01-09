import sys
import os
import pandas as pd  # Asegúrate de importar pandas
# Agregar la raíz del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model_loader import load_model

def main():
    # Load the model
    model_path = "models/trained_model_2025-01-08.joblib"
    model = load_model(model_path)
    print("paso la carga del modelo")

    # Sample prediction
    input_data = {
        "age": [63],  # Convertir a listas para compatibilidad con DataFrame
        "sex": [1],
        "cp": [3],
        "trestbps": [145],
        "chol": [233],
        "fbs": [1],
        "restecg": [1],
        "thalach": [150],
        "exang": [0],
        "oldpeak": [2],
        "slope": [2],
        "ca": [0],
        "thal": [2]
    }

    # Convertir el diccionario a un DataFrame
    input_df = pd.DataFrame(input_data)

    try:
        # Pasar el DataFrame al modelo
        prediction = model.predict(input_df)
        print(f"Predictions: {prediction}")
    except Exception as e:
        print(f"Error running inference: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()