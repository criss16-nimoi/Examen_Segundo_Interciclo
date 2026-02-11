# Examen Segundo Interciclo — Clasificación de Animales (vaca / cerdo / gallina) + MLflow + App (Streamlit)

## 1. Descripción general
Este repositorio contiene una solución de **clasificación de imágenes** para tres clases: **vaca**, **cerdo** y **gallina**.  
Incluye:

- Preparación del dataset y verificación de rutas.
- Entrenamiento base del modelo.
- Evaluación con métricas (incluyendo **Macro-F1**).
- Registro de entrenamientos y reentrenamientos en **MLflow**.
- **Aplicación en Streamlit** para realizar predicciones seleccionando imágenes desde la PC.

**Clases (en minúsculas):** `vaca`, `cerdo`, `gallina`.

---

## 2. Tecnologías utilizadas
- Python (recomendado: Anaconda)
- Jupyter Notebook
- TensorFlow / Keras
- NumPy
- MLflow (tracking)
- Streamlit (App)

---

## 3. Estructura del repositorio (recomendada)
```text
Examen_Segundo_Interciclo/
├─ README.md
├─ Notebooks/
│  ├─ Primer_Cuaderno_Examen.ipynb
│  └─ Segundo_Cuaderno_Examen.ipynb
├─ Aplicación/
│  ├─ requirements.txt
│  └─ app/
│     ├─ app.py
│     └─ utils.py (opcional)
└─ assets/
   ├─ 01_estructura_dataset.png
   ├─ 02_mlflow_runs.png
   ├─ 03_interfaz_app.png

```
## 4. Dataset y rutas
```text
dataset_animales/
└─ raw/
   ├─ vaca/
   ├─ cerdo/
   └─ gallina/
```
## 5. Metodología
```text
flowchart TD
    A[Dataset raw<br/>vaca/cerdo/gallina] --> B[Preprocesamiento<br/>split train/val]
    B --> C[Entrenamiento base]
    C --> D[Evaluación<br/>Accuracy / Macro-F1]
    D --> E[Registro en MLflow<br/>params, metrics, artifacts]
    E --> F[Predicción en Streamlit<br/>selección de imagen local]
    F --> G[Feedback / Correcciones]
    G --> H[Reentrenamiento / Fine-tuning]
    H --> E
```
## 6. MLflow

mlflow ui --backend-store-uri "./mlruns" --port 5000
http://127.0.0.1:5000

## 7. Resultados (resumen)

Modelo base:

- Accuracy (val): [0.9998672008514404]

- Macro-F1 (val): [0.9918699264526367]

Mejor reentrenamiento:

- Accuracy (val): [1]

- Macro-F1 (val): [1]

## 8.  Conclusiones

- Se implementó un pipeline completo para clasificar imágenes en tres clases (vaca, cerdo, gallina).

- Se registraron entrenamientos y reentrenamientos en MLflow, garantizando trazabilidad.

- El reentrenamiento permite mejorar el modelo con base en errores observados.

- La app en Streamlit permite inferencia práctica con imágenes locales de manera sencilla.

## 9. Cómo ejecutar la aplicación

### 9.1 Instalar dependencias

   pip install -r requirements.txt
   
### 9.2 Ejecutar la app

   streamlit run app.py

## 10. Autores
- Zahid Armijos 
- Cristopher Jara
