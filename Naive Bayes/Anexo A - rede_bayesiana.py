# -*- coding: utf-8 -*-
"""
Created on Wed May 24 21:13:07 2023

@author: deivity.andrade
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
#%% Importando os dados
csv_path = 'C:\\Users\\deivity.andrade\\OneDrive - Companhia de Gás de Santa Catarina - SCGÁS\\Área de Trabalho\\bayesiana\\treino_bayseana.txt'
df = pd.read_table(csv_path)
csv_path = 'C:\\Users\\deivity.andrade\\OneDrive - Companhia de Gás de Santa Catarina - SCGÁS\\Área de Trabalho\\bayesiana\\teste_bayseana.txt'
dft = pd.read_table(csv_path)

#%% Preparando dados para utilização, efetuando a classificação segundo os parâmetros refeência
for i in range(len(df)):
    if df['NIF'].iloc[i] < -26:
         df['NIF'].iloc[i] = "<_-26"
    else:
        df['NIF'].iloc[i] = ">=_-26"
    
    if df['VT'].iloc[i] >= 315:
        df['VT'].iloc[i] = ">=_315"
    else:
        df['VT'].iloc[i] = "<_315"
    
    if df['RR'].iloc[i] < 30:
        df['RR'].iloc[i] = "<_30"
    else:
        df['RR'].iloc[i] = ">=_30"
    
    if df['RSBI'].iloc[i] <= 80:
        df['RSBI'].iloc[i] = "<=_80"
    else:
        df['RSBI'].iloc[i] = ">_80"

for i in range(len(dft)):
    if dft['NIF'].iloc[i] < -26:
         dft['NIF'].iloc[i] = "<_-26"
    else:
        dft['NIF'].iloc[i] = ">=_-26"
    
    if dft['VT'].iloc[i] >= 315:
        dft['VT'].iloc[i] = ">=_315"
    else:
        dft['VT'].iloc[i] = "<_315"
    
    if dft['RR'].iloc[i] < 30:
        dft['RR'].iloc[i] = "<_30"
    else:
        dft['RR'].iloc[i] = ">=_30"
    
    if dft['RSBI'].iloc[i] <= 80:
        dft['RSBI'].iloc[i] = "<=_80"
    else:
        dft['RSBI'].iloc[i] = ">_80"
#%% Divisão de previsores e classe

x_dados = df.iloc[:, 0:4].values
y_dados = df.iloc[:, 4].values

x_dadost = dft.iloc[:, 0:4].values
y_dadost = dft.iloc[:, 4].values

#%% Transformando atributos categoricos em numericos

label_encoder_NIF = LabelEncoder()
label_encoder_VT = LabelEncoder()
label_encoder_RR = LabelEncoder()
label_encoder_RSBI = LabelEncoder()

x_dados[:,0] = label_encoder_NIF.fit_transform(x_dados[:,0])
x_dados[:,1] = label_encoder_VT.fit_transform(x_dados[:,1])
x_dados[:,2] = label_encoder_RR.fit_transform(x_dados[:,2])
x_dados[:,3] = label_encoder_RSBI.fit_transform(x_dados[:,3])

x_dadost[:,0] = label_encoder_NIF.fit_transform(x_dadost[:,0])
x_dadost[:,1] = label_encoder_VT.fit_transform(x_dadost[:,1])
x_dadost[:,2] = label_encoder_RR.fit_transform(x_dadost[:,2])
x_dadost[:,3] = label_encoder_RSBI.fit_transform(x_dadost[:,3])
print (x_dados)

#%% utilizando e treinando a nayve bayes

naive_desmame = GaussianNB()
naive_desmame.fit(x_dados, y_dados)

#%% previsao
previsao = naive_desmame.predict(x_dadost)

#%% analise dos resultados
#acuracidade
accuracy_score(y_dadost, previsao)
#matrix confusao
confusion_matrix(y_dadost, previsao)
#plotando a matrix confusao
cm = ConfusionMatrix(naive_desmame)
cm.fit(x_dados, y_dados)
cm.score(x_dadost, y_dadost)

#outras metricas
print(classification_report(y_dadost, previsao))
