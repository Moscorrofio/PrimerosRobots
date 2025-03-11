from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Inicializar la API
app = FastAPI()

# Definir la estructura de los datos que recibirá la API
class ClienteDatos(BaseModel):
    edad: int
    ingresos_mensuales: float
    nivel_endeudamiento: float
    valor_vivienda: float
    relacion_cuota_ingreso: float

# Datos ficticios para entrenar el modelo (simulando lo que ya tienes en el dataset)
data = pd.DataFrame({
    "edad": np.random.randint(20, 65, 500),
    "ingresos_mensuales": np.random.randint(1000000, 20000000, 500),
    "nivel_endeudamiento": np.random.uniform(0.1, 0.8, 500),
    "valor_vivienda": np.random.randint(50000000, 500000000, 500),
    "relacion_cuota_ingreso": np.random.uniform(0.1, 0.5, 500),
    "desistio": np.random.choice([0, 1], 500)  # 0 = No desistió, 1 = Desistió
})

# Normalizar los datos
scaler = StandardScaler()
X = scaler.fit_transform(data.drop(columns=["desistio"]))
y = data["desistio"]

# Entrenar el modelo logit
modelo_logit = LogisticRegression()
modelo_logit.fit(X, y)

# Ruta para predecir la probabilidad de desistimiento
@app.post("/predecir")
def predecir(datos: ClienteDatos):
    entrada = np.array([[datos.edad, datos.ingresos_mensuales, datos.nivel_endeudamiento, datos.valor_vivienda, datos.relacion_cuota_ingreso]])
    entrada_escalada = scaler.transform(entrada)
    probabilidad = modelo_logit.predict_proba(entrada_escalada)[0][1]
    
    return {"probabilidad_desistimiento": round(probabilidad, 2)}

# Ruta de prueba para ver si la API está corriendo
@app.get("/")
def home():
    return {"mensaje": "API para predecir desistimiento funcionando"}

