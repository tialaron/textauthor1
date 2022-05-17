import allfunctions1
from sklearn.metrics import accuracy_score, roc_auc_score, homogeneity_score, mean_squared_error  # импортируем метрики, вторая считается более сбалансированной если есть дисбаланс классов
from sklearn.manifold import TSNE   # для визуализации многомерных данных
from sklearn.cluster import KMeans, SpectralClustering # Импортируем библиотеку KMeans для кластеризации
from random import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer # Методы для работы с текстами и преобразования их в последовательности

import os
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

className = ["О. Генри", "Стругацкие", "Булгаков", "Клиффорд_Саймак", "Макс Фрай", "Брэдберри"] # Объявляем интересующие нас классы
main_dir_t1 = 'H:\Pythonprojects\\textauthor1\\venv\content'
model01 = load_model('app/model_author_all.h5')
# loading
with open('app/tokenizer.pickle', 'rb') as handle:
    tokenizer2 = pickle.load(handle)

image_path = 'app/realtrack1.jpg'
#main_dir_t2 = '/content/class2/'

newTest = []
for i in range(len(className)): #Проходим по каждому классу
    newTest.append(allfunctions1.readText('H:\Pythonprojects\\textauthor1\\venv\(Айзек_Азимов) Тестовая Я робот.txt'))

xLen = 1000 #Длина отрезка текста, по которой анализируем, в словах
step = 100 #Шаг разбиения исходного текста на обучающие векторы
testWordIndexes1 = tokenizer2.texts_to_sequences(newTest)  # Проверочные тесты в индексы
wordIndexes = testWordIndexes1

xTest6Classes01 = []               #Здесь будет список из всех классов, каждый размером "кол-во окон в тексте * 20000
xTest6Classes = []                 #Здесь будет список массивов, каждый размером "кол-во окон в тексте * длину окна"(6 по 420*1000)
for wI in wordIndexes:                       #Для каждого тестового текста из последовательности индексов
    sample = (allfunctions1.getSetFromIndexes(wI, xLen, step))  # Тестовая выборка размером "кол-во окон*длину окна"(например, 420*1000)
    xTest6Classes.append(sample)  # Добавляем в список
    xTest6Classes01.append(tokenizer2.sequences_to_matrix(sample))  # Трансформируется в Bag of Words в виде "кол-во окон в тексте * 20000"
xTest6Classes01 = np.array(xTest6Classes01)                     #И добавляется к нашему списку,
xTest6Classes = np.array(xTest6Classes)                     #И добавляется к нашему списку,

xTest = xTest6Classes01

totalSumRec = 0 # Сумма всех правильных ответов
# Проходим по всем классам. А их у нас 6
nClasses = 6
for i in range(nClasses):
    # Получаем результаты распознавания класса по блокам слов длины xLen
    currPred = model01.predict(xTest[i])
    # Определяем номер распознанного класса для каждохо блока слов длины xLen
    currOut = np.argmax(currPred, axis=1)

    evVal = []
    for j in range(nClasses):
        evVal.append(len(currOut[currOut == j]) / len(xTest[i]))

    totalSumRec += len(currOut[currOut == i])
    recognizedClass = np.argmax(evVal)  # Определяем, какой класс в итоге за какой был распознан

    # Выводим результаты распознавания по текущему классу
    # isRecognized = "Это НЕПРАВИЛЬНЫЙ ответ!"
    # if (recognizedClass == i):
    #  isRecognized = "Это ПРАВИЛЬНЫЙ ответ!"
    str1 = 'Класс: ' + className[i] + " " * (11 - len(className[i])) + str(
        int(100 * evVal[i])) + "% сеть отнесла к классу " + className[recognizedClass]
    # print(str1, " " * (55-len(str1)), isRecognized, sep='')
    print(str1, " " * (55 - len(str1)))

# Выводим средний процент распознавания по всем классам вместе
print()
sumCount = 0
for i in range(nClasses):
    sumCount += len(xTest[i])
print("Средний процент распознавания ", int(100 * totalSumRec / sumCount), "%", sep='')
#testWordIndexes = tokenizer2.texts_to_sequences(testText)  # Проверочные тесты в индексы

with st.expander("Вот так выглядит процесс создания нейронной сети"):
    st.write("В процессе создания проводится сбор базы данных из текстов. Подготовка данных заключается только в распределении"
             "текстов одного автора по файлам")
    st.image(image_path)

