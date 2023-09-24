from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

# Crear un conjunto de datos de finanzas
data_finanzas = {
    'Texto': [
        'informe de ganancias para el tercer trimestre',
        'ganancias ingresos utilidades ventas beneficios',
        'financieros resultados trimestre informe cifras',
        'análisis de mercado de acciones',
        'acciones mercado inversión rendimiento empresa',
        'empresa valor precios accionistas dividendos'
        'balance general de la empresa',
        'estrategia de inversión a largo plazo',
        'resumen de gastos mensuales',
        'saldo mensual de gastos',
        'este es tu estado de cuenta de julio',
        'resumen de transacciones en montos',
        'saldo periodo cuenta ingresos', 
        'transacción pago tarjeta intereses cuota',
        'retiro depósito movimiento periodo',

    ],
    'Tipo_de_Documento': [
        'Informe de Ganancias',
        'Informe de Ganancias',
        'Informe de Ganancias',
        'Análisis de Acciones',
        'Analisis de Acciones',
        'Analisis de Accciones'
        'Balance General',
        'Estrategia de Inversión',
        'Resumen de Gastos',
        'Saldo de un periodo',
        'Estado de cuenta', 
        'Estado de cuenta',
        'Estado de cuenta',
        'Estado de cuenta',
        'Estado de cuenta',

    ]
}

'''data_escolares = {
    'Texto': [
        'Informe de calificaciones del estudiante',
        'Plan de estudios para el próximo semestre',
        'Resumen del libro de historia',
        'Guía de ejercicios de matemáticas',
        'Tesis de graduación',
        'actividad del ciclo de vida de una app',
        'reporte de practicas de ciencias',
        'practicas de desarrollo web',
        'Este es tu estado de cuenta de julio',


    ],
    'Tipo_de_Documento': [
        'Informe de Calificaciones',
        'Plan de Estudios',
        'Resumen de Historia',
        'Guía de Matemáticas',
        'Tesis de Graduación',
        'Actividad de practica',
        'Reporte de actividad', 
        'practiva desarrollo',
    ]
}'''

#df_escolares = pd.DataFrame(data_escolares)
df_finanzas = pd.DataFrame(data_finanzas)

#unier los conjuntos de datos en uno
df = pd.concat([df_finanzas], ignore_index=True)

#dividir este conunto en dos
X = df['Texto']
y = df['Tipo_de_Documento']

vectorizer = CountVectorizer(max_features=5)  # Limitamos a las 5 palabras más comunes
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

#entrenar el modelo
clf = MultinomialNB()
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

# Guardar el modelo entrenado para su uso futuro
joblib.dump(clf, 'modelo_clasificacion.pkl')
