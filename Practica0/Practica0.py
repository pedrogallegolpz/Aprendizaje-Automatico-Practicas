#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 11:21:50 2021

@author: pcgl - Pedro Gallego López

ASIGNATURA: Aprendizaje Automático UGR

Doble Grado en Ingeniería Informática y Matemáticas

"""

#########################
#                       #
#       Práctica 0      #
#                       #
#########################



import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split



def wait():
	input("Press intro to continue")


"""

EJERCICIO 1
    1. Leer la base de datos de iris que hay en scikit-learn
    2. Obtener las características (datos de entrada X) y la clase (y)
    3. Quedarse con las características 1 y 3 (primera y tercera columna de X)
    4. Visualizar con un Scatter Plot los datos, coloreando cada clase con un
       color diferente (con naranja, negro y verde), e indicando con una
       leyenda la clase a la que corresponde cada color

"""
def ejercicio1():
	"""
		La base de datos iris contiene 3 tipos de flores iris: Setosa, Versicolor y Virginica, donde se
		tienen sus cuatro caracterı́sticas: ancho y alto del sépalo además del ancho y largo del pétalo.
		Esto guardado en un numpy.ndarray de 150 × 4.
	"""
	# Leemos la base de datos
	iris = datasets.load_iris()
	
	X = iris.data[:,:3:2] 		# Cogemos las características 1 y 3 (columna 0 y 2) para ello usamos las columnas pares menores de la 3
	y = iris.target             # Cogemos las clases
	
	# Establecemos los rangos que van a tener nuestros ejes de la gráfica
	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	
	# Hacemos plot de los puntos
	colores = np.array(['#FF9300', '#000000', '#31BF00']) # [naranja, negro, verde]
	
	# Creamos el scatter plot dándole los colores a través del parámetro cmap
	#	-El parámetro edgecolor='k' es para que tengan bordes negros los puntos
	scatter = plt.scatter(X[:,0],X[:,1], c=y, cmap = colors.ListedColormap(colores), edgecolor='k')
	
	# Definimos la leyenda que tendrá la gráfica
	plt.legend(handles = scatter.legend_elements()[0],	# Cogemos los puntos de colores de la salida anterior
			   title = "Classes",
			   labels = iris.target_names.tolist(),		# Cogemos las etiquetas dadas por la BD
	           scatterpoints=1,
	           loc='lower right',          			 	# Abajo a la derecha
	           fontsize=8)
	
	plt.xlabel('Sepal length (cm)')
	plt.ylabel('Petal length (cm)')
	
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	
	# Mostramos
	plt.show()


"""

EJERCICIO 2
	Separar en training (75% delso datos) y test (25%) aleatoriamente conservando
	la proporción de elementos en cada clase tanto en training como en test. Con
	esto se pretende evitar que haya clases infra-representadas en entrenamiento
	o test

"""
def ejercicio2():
	# Importamos la base de datos
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target
	
	# Usamos la función train_test_split:
	#	shuffle = True para dar aleatoriedad
	#	stratify = y para mantener la proporción
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, shuffle=True, stratify=y)
	
	
	# COMPROBACIONES
	
	# Mostramos el número de muestras que se nos queda en cada uno
	print("Cantidad del train: "+ str(X_train.shape[0])+ " que supone un "+ str(X_train.shape[0]/X.shape[0])+"% del total")
	print("Cantidad del test: "+ str(X_test.shape[0])+ " que supone un "+ str(X_test.shape[0]/X.shape[0])+"% del total")
			
	# Comprobamos que se ha guardado la proporción
	print("En el TOTAL están las proporciones")
	print("   Clase 1: "+str(len(y[y==0])/len(y))+"%")
	print("   Clase 2: "+str(len(y[y==1])/len(y))+"%")
	print("   Clase 3: "+str(len(y[y==2])/len(y))+"%")
	
	
	print("En el TRAIN están ahora las proporciones")
	print("   Clase 1: "+str(len(y_train[y_train==0])/len(y_train))+"%")
	print("   Clase 2: "+str(len(y_train[y_train==1])/len(y_train))+"%")
	print("   Clase 3: "+str(len(y_train[y_train==2])/len(y_train))+"%")
	
	
	print("En el TEST están ahora las proporciones")
	print("   Clase 1: "+str(len(y_test[y_test==0])/len(y_test))+"%")
	print("   Clase 2: "+str(len(y_test[y_test==1])/len(y_test))+"%")
	print("   Clase 3: "+str(len(y_test[y_test==2])/len(y_test))+"%")
	
	
	
"""

EJERCICIO 3
	1. Obtener 100 valores equiespaciados entre 0 y 4pi
	2. Obtener el valor de sin(x), cos(x) y tanh(sin(x)+cos(x)) para los 100 
	   valores anteriormente calculados
	3. Visualizar las tres curvas simultáneamente en el mismo plot (con líneas
       discontinuas en verde, negro y rojo).	      
	            
"""
def ejercicio3():
	# Obtenemos los 100 valores equiespaciados entre 0 y 4pi
	x = np.linspace(0, 4*np.pi, (100))
	
	# Obtenemos los valores de las 3 funciones
	y1=np.sin(x)
	y2=np.cos(x)
	y3=np.tanh(y1+y2) 		# Aprovechamos los valores de y1 e y2
	
	# Creamos los plots
	# 	- Los valores 'g--','k--','r--' hacen referencia al color y estilo del trazo de línea
	# 	 	la letra hace referencia al color y el '--' a que sea discontinua. Así serían:
	# 	 	-Verde: 'g--'
	# 	 	-Negro: 'k--'
	# 	 	-Rojo: 'r--'
	fig, ax = plt.subplots()
	ax.plot(x, y1, 'g--', label='y=sen(x)')
	ax.plot(x, y2, 'k--', label='y=cos(x)')
	ax.plot(x, y3, 'r--', label='y=tanh(sen(x)+cos(x))')
	
	# Definimos la leyenda en la parte superior derecha
	ax.legend(loc='upper right', shadow=False)
	
	# Mostramos
	plt.show()	 
	
	
	
	
	
	
# EJECUCIÓN	
ejercicio1()
wait()
ejercicio2()
wait()
ejercicio3()
	


