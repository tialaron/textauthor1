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

main_dir_t1 = 'app/content'
ae = load_model('app/model_author_ae.h5')
# loading
with open('app/tokenizer.pickle', 'rb') as handle:
    tokenizer2 = pickle.load(handle)

image_path = 'app/realtrack1.jpg'
#main_dir_t2 = '/content/class2/'

print(os.listdir(main_dir_t1))

#testWordIndexes = tokenizer2.texts_to_sequences(testText)  # Проверочные тесты в индексы

with st.expander("Вот так выглядит процесс создания нейронной сети"):
    st.write("В процессе создания проводится сбор базы данных из текстов. Подготовка данных заключается только в распределении"
             "текстов одного автора по файлам")
    st.image(image_path)
# формируем текстовые выборки
#NewTexts = []
#text_classes = []
#file_names = []
#for filename in os.listdir(main_dir_t1):
#    file_path = main_dir_t1 + filename
#    NewTexts.append(readText(file_path)) #добавляем в обучающую выборку
#    text_classes.append(0)
#    file_names.append(filename)
#for filename in os.listdir(main_dir_t2):
#    file_path = main_dir_t2 + filename
#    NewTexts.append(readText(file_path)) #добавляем в обучающую выборку
#    text_classes.append(1)
#    file_names.append(filename)

# Преобразовываем текст в последовательность индексов согласно частотному словарю
#valWordIndexes = tokenizer.texts_to_sequences(NewTexts) # Обучающие тесты в индексы

#auth_val_words = {}
#for i in range(2): # Проходим по всем классам
#    print(file_names[i], " "*(10-len(file_names[i])), len(valWordIndexes[i]), "слов")
#    auth_val_words[file_names[i]] = len(valWordIndexes[i])

#min_val_len = np.min(np.array(list(auth_val_words.values())))
#print("наименьшая длина выборки текстов:", min_val_len)

#балансируем выборку, укорачивая наборы слов по каждому классу:
#print("проводим балансировку:")
#for i in range(2): # Проходим по всем классам
#    valWordIndexes[i] = valWordIndexes[i][:min_val_len]
#    print(file_names[i], " "*(10-len(file_names[i])), len(valWordIndexes[i]), "слов")

#Задаём базовые параметры
xLen = 1000 #Длина отрезка текста, по которой анализируем, в словах
step = 100 #Шаг разбиения исходного текста на обучающие векторы

#xVal, yVal = createSetsMultiClasses(valWordIndexes, xLen, step) #извлекаем обучающую выборку
#xVal01 = tokenizer.sequences_to_matrix(xVal.tolist()) #Подаем xVal в виде списка, чтобы метод успешно сработал
# загрузить с диска удачную модель
#ae = load_model('H:\Pythonprojects\\textauthor1\\venv\model_author_ae.h5')

#z_train = enc.predict(xTrain01) #X
#z_train.shape, z_train.std()

#xVal0 = xVal01[yVal_num==0]
#xVal1 = xVal01[yVal_num==1]

#out_Val0 = ae.predict(xVal0) #X
#out_Val1 = ae.predict(xVal1) #X


#tres = 0.00066
#if mse_val > tres:
#    print('Авторы текстов разные.')
#else:
#    print('У текстов один автор.')
