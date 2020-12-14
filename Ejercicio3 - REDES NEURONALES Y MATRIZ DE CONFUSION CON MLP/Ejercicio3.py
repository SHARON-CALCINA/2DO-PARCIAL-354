# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 12:37:54 2020

@author: SHARON
"""

import pandas as pd , numpy as np
credito= pd.read_csv('credit.csv')
credito=credito.replace({'?':'0'})
#credito.to_csv('credClean.csv', ',')
credito.columns=['Género', 'Edad', 'Deuda', 'EstadoCivil', 'Cliente_bancario', 
                 'Nivel_educación', 'Etnia', 'Años_Empleado', 'Predeterminado',
                 'Empleado', 'Puntaje_crédito', 'Licencia_conducir', 'Ciudadano',
                 'Código postal', 'Ingresos' ,'Estado_aprobación']

from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()
credito['Género']= encoder.fit_transform(credito.Género.values)
credito['Estado_aprobación']= encoder.fit_transform(credito.Estado_aprobación.values)

print(credito.Género.unique())
print(credito.Estado_aprobación.unique())
X=credito[['Género', 'Edad', 'Ingresos']]
y=credito['Estado_aprobación']
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test= train_test_split(X,y, test_size = 0.2)
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=[5], max_iter=500, alpha=0.0001, 
                             solver='adam', random_state=21, tol=0.0001)
mlp.fit(X_train, y_train)
predictions=mlp.predict(X_test)
from sklearn.metrics import classification_report
print('REPORTE DEL CLASIFICADOR ')
print(classification_report(y_test, predictions))
from sklearn.metrics import confusion_matrix
print('MATRIZ DE CONFUSION ')
print(confusion_matrix(y_test, predictions))
