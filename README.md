# Medical Insurance Charges Prediction 

> Un proyecto completo de Machine Learning para predecir costos de seguros médicos utilizando regresión y análisis exploratorio de datos.

---

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Características](#características)
- [Requisitos Previos](#requisitos-previos)
- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Uso](#uso)
- [Flujo de Datos](#flujo-de-datos)
- [Modelos Implementados](#modelos-implementados)
- [Resultados](#resultados)
- [Análisis Exploratorio](#análisis-exploratorio)
- [Contribución](#contribución)
- [Licencia](#licencia)

---

## Descripción del Proyecto

Este proyecto implementa un pipeline completo de **Machine Learning** para predecir los costos de seguros médicos basándose en características demográficas y de salud del paciente. 

El proyecto sigue las mejores prácticas de ciencia de datos:
- **Data Ingestion**: Carga de datos desde fuentes externas
- **Data Wrangling**: Limpieza y transformación de datos
- **EDA**: Análisis exploratorio exhaustivo
- **Model Development**: Entrenamiento de múltiples modelos
- **Model Refinement**: Optimización y validación

---

## Características

**Análisis completo de datos faltantes** - Identificación y tratamiento de valores nulos  
**Ingeniería de características** - Mapeo y transformación de variables categóricas  
**Múltiples modelos** - Regresión lineal, logística y Ridge  
**Pipeline automatizado** - Escalado y transformaciones polinómicas  
**Validación cruzada** - Evaluación robusta del desempeño  
**Visualizaciones informativas** - Gráficos de regresión, boxplots y heatmaps  

---

## Requisitos Previos

- **Python 3.8+**
- **pip** o **conda** para gestión de paquetes

---

## Instalación

### 1. Clonar o descargar el repositorio

```bash
git clone https://github.com/tu-usuario/medical-insurance-prediction.git
cd medical-insurance-prediction
```

### 2. Crear un entorno virtual (recomendado)

```bash
# Con venv
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# O con conda
conda create -n insurance-env python=3.9
conda activate insurance-env
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

**O instala manualmente los paquetes:**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---
```

---

## Uso

### Ejecución básica

```bash
python main.py
```

El script ejecutará automáticamente:

1. **Carga de datos** del dataset de seguros médicos
2. **Limpieza de datos** (valores faltantes, tipos de datos)
3. **Análisis exploratorio** con visualizaciones
4. **Entrenamiento de modelos** con diferentes algoritmos
5. **Evaluación de rendimiento** con métricas de R² y RMSE

### Uso con Jupyter Notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

##  Flujo de Datos

### Fase 1: Data Ingestion
```
CSV remoto → DataFrame pandas
↓
Definir headers → Valores faltantes como NaN
```

**Datos originales:**
- **7 características**: age, gender, bmi, no_of_children, smoker, region, charges
- **Valores faltantes**: Identificados como '?'

### Fase 2: Data Wrangling
```
Mapeo de categorías → Conversión de tipos
↓
Manejo de valores nulos → Redondeo de decimales
```

**Transformaciones realizadas:**
- `gender` (1,2) → "Male", "Female"
- `region` (1,2,3,4) → "NW", "NE", "SW", "SE"
- `age`: Conversión a numérico + imputación con media
- `smoker`: Imputación con moda + conversión a entero
- `charges`: Redondeo a 2 decimales

### Fase 3: EDA (Análisis Exploratorio)
```
Visualizaciones → Correlaciones → Patrones
```

### Fase 4: Model Development & Refinement
```
Train-Test Split → Modelos Base → Optimización
↓
Evaluación final → Predicciones
```

---

## Modelos Implementados

### 1. **Logistic Regression**
```python
LogisticRegression()
```
- **Propósito**: Clasificación binaria (fumador/no fumador)
- **Features**: ['charges']
- **Métrica**: Accuracy

### 2. **Linear Regression**
```python
LinearRegression()
```
- **Propósito**: Predicción de costos
- **Features**: age, gender, bmi, no_of_children, smoker, region
- **Métrica**: R² Score

### 3. **Pipeline con Transformaciones**
```python
Pipeline([
    ('scale', StandardScaler()),
    ('polynomial', PolynomialFeatures(include_bias=False)),
    ('model', LinearRegression())
])
```
- **Propósito**: Capturar relaciones no-lineales
- **Transformaciones**: Escalado + características polinómicas
- **Métrica**: R² Score

### 4. **Ridge Regression**
```python
Ridge(alpha=0.1)
```
- **Propósito**: Regresión regularizada (prevenir overfitting)
- **Versiones**:
  - Ridge básico
  - Ridge con características polinómicas (grado 2)
- **Métrica**: R² Score

---

## Resultados

Los modelos se evalúan usando las siguientes métricas:

| Modelo | Métrica | Valor |
|--------|---------|-------|
| Logistic Regression | Accuracy | TBD |
| Linear Regression | R² Score | TBD |
| Pipeline (Linear + Poly) | R² Score | TBD |
| Ridge (simple) | R² Score | TBD |
| Ridge + Polynomial | R² Score | TBD |

**Nota**: Ejecuta el script para ver los valores reales de R² Score para cada modelo.

---

## Análisis Exploratorio

### Visualizaciones Generadas

#### 1. Regresión: BMI vs Charges
```python
sns.regplot(x="bmi", y="charges", data=df, line_kws={"color":"blue"})
```
**Insight**: Relación positiva entre IMC y costos de seguros

#### 2. Boxplot: Fumador vs Charges
```python
sns.boxplot(x="smoker", y="charges", data=df)
```
**Insight**: Los fumadores tienen costos significativamente más altos

#### 3. Matriz de Correlación
```python
corr = df.select_dtypes(include=['int64','float']).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
```
**Insight**: Identificar variables más correlacionadas con los costos

---

## Conceptos Clave

### StandardScaler
Normaliza los datos a media 0 y desviación estándar 1. Esencial para algoritmos sensibles a la escala.

### PolynomialFeatures
Genera características polinómicas (interacciones y potencias) para capturar relaciones no-lineales.

### Ridge Regression (L2 Regularization)
Añade penalización a los coeficientes grandes, reduciendo el riesgo de overfitting.

### Train-Test Split
Divide los datos en 80% entrenamiento y 20% prueba para evaluación honesta del modelo.

---

## Dependencias

| Paquete | Versión | Propósito |
|---------|---------|----------|
| pandas | ≥1.0.0 | Manipulación de datos |
| numpy | ≥1.19.0 | Computación numérica |
| scikit-learn | ≥0.24.0 | Machine Learning |
| matplotlib | ≥3.1.0 | Visualización |
| seaborn | ≥0.11.0 | Visualización estadística |

---

## Customización

### Cambiar el alpha de Ridge
```python
RidgeModel = Ridge(alpha=1.0)  # Aumentar para más regularización
```

### Cambiar el grado polinómico
```python
pr = PolynomialFeatures(degree=3)  # Para grado 3
```

### Usar validación cruzada
```python
scores = cross_val_score(model, X, y, cv=5)
print(f"CV Scores: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## Mejoras Futuras

- [ ] Implementar GridSearchCV para tuning automático de hiperparámetros
- [ ] Agregar más modelos (Random Forest, Gradient Boosting, SVM)
- [ ] Crear API REST con Flask/FastAPI para predicciones en tiempo real
- [ ] Implementar feature selection automático
- [ ] Agregar tests unitarios
- [ ] Crear dashboard interactivo con Streamlit
- [ ] Documentación en HTML generada con Sphinx

---

## Contribución

Las contribuciones son bienvenidas. Para cambios importantes:

1. Fork el repositorio
2. Crea una rama con tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## Licencia

Este proyecto está bajo la licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## Autor

Desarrollado como parte de proyectos de **IBM Developer Skills Network - Data Analysis**

---

## Contacto & Soporte

- **Email**: mariosoluciona15@gmail.com
- **LinkedIn**: [Tu perfil](https://linkedin.com/in/tu-perfil)

---

## Referencias

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Seaborn Tutorial](https://seaborn.pydata.org/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

---

** Si este proyecto te fue útil, considera darle una estrella en GitHub!**

---

