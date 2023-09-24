import PyPDF2
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords #palabras comunes
import string
import joblib 
from sklearn.feature_extraction.text import CountVectorizer


import nltk
nltk.download('stopwords')

nltk.download('punkt')#descargamos  los recursos de NLTK

def count_words(text):
    words = word_tokenize(text)
    num = 0
    for i in words:
        num = 1 + num
        
    return num

def palabra_mas_comun(texto):
     # Tokenizar el texto
    palabras = word_tokenize(texto.lower())  # Convertir a minúsculas
    
    # Eliminar signos de puntuación
    palabras = [palabra for palabra in palabras if palabra not in string.punctuation]
    
    # Filtrar palabras comunes (stop words)
    stop_words = set(stopwords.words("spanish"))  # Puedes cambiar "spanish" al idioma que necesites
    palabras = [palabra for palabra in palabras if palabra not in stop_words]
    
    contador_palabras = Counter(palabras)
    
    # Obtener las n palabras más comunes
    palabras_comunes = contador_palabras.most_common(5)
    
    return palabras_comunes







def count_words_per_line(text):
    lines = sent_tokenize(text)
    word_counts = [len(word_tokenize(line)) for line in lines]
    total = 0
    for i in word_counts:
        total = total + i

    return int((total / len(word_counts)))


def leer_pdf(file_path):
    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text


clf = joblib.load('modelo_clasificacion.pkl')


pdf =  "agosto 2023.pdf"

pdf_texto = leer_pdf(pdf)
print(pdf_texto)

print("promedio de palabras por renglon ", count_words_per_line(pdf_texto))
print("El numero de palabras es",count_words(pdf_texto))
print("La palabra que mas se repite es: ", palabra_mas_comun(pdf_texto))

vectorizer = CountVectorizer(max_features=5)  # Limitamos a las 5 palabras más comunes
X_new = vectorizer.fit_transform([pdf_texto])

# Realizar la predicción
predicted_class = clf.predict(X_new)

# Imprimir el resultado
print("Tipo de documento predicho:", predicted_class[0])

