#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Pedro Gallego López
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as cm
from sklearn.utils import shuffle






def wait():
	"""
	Función para pausar la ejecución
	"""
	input("\n--- Pulsar tecla para continuar ---\n")
	
	
	
####################################################################
##
##	  	 	 	FUNCIONES PARA MOSTRAR POR PANTALLA
##
####################################################################

def display_graph(X, y, title='', etiquetas=['',''], labels_axes = ['', ''], reg=[], w=[], a=None, b=None, etiquetafun=[], show_niveles=True):
	"""
	Muestra una gráfica de puntos junto con una función lineal
		 x 	 	 	 	Datos
		 y 	 	 	 	Etiquetas
		 title 	 	 	Título de la gráfica
		 etiquetas 	 	Etiquetas de los datos para la leyenda
		 label_axes  	Etiqueta de los ejes
		 reg 	 	 	Lista de funciones a representar
		 w 	 	 	 	Pesos finales de las funciones de reg (tiene que haber la misma cantidad de elementos)
		 a, b 	 	 	Parámetros de reg[0]
		 etiquetafun 	Etiquetas de las funciones reg para la leyenda (tiene que haber la misma cantidad de elementos)
		 show_niveles 	Booleano que puesto a True muestra coloreada las areas de nivel por encima y debajo del nivel 0
	
	"""
	
	# Establecemos los rangos que van a tener nuestros ejes de la gráfica
	x1_min = X[:, 0].min() - (X[:, 0].max()-X[:, 0].min())/10.0
	x1_max = X[:, 0].max() + (X[:, 0].max()-X[:, 0].min())/10.0
	
	x2_min = X[:, 1].min() - (X[:, 1].max()-X[:, 1].min())/10.0
	x2_max = X[:, 1].max() + (X[:, 1].max()-X[:, 1].min())/10.0
	
	
	# Hacemos plot de los puntos
	colores = np.array(['#FF9300', '#31BF00']) # [naranja, negro, verde]
	colores_lineas = ['#3F24FF','#E800FF','#910000'] # [azul,rosa,rojo]
	
	# Creamos el scatter plot dándole los colores a través del parámetro cmap
	#	-El parámetro edgecolor='k' es para que tengan bordes negros los puntos
	scatter = plt.scatter(X[:,0],X[:,1], c=y, cmap = cm.colors.ListedColormap(colores), edgecolor='k')
	
	# Definimos la leyenda que tendrá la gráfica
	legend_classes = plt.legend(handles = scatter.legend_elements()[0],	# Cogemos los puntos de colores de la salida anterior
								   title = "Classes",
								   labels = etiquetas,		# Cogemos las etiquetas dadas por la BD
						           scatterpoints=1,
						           loc='upper right',          			 	# Abajo a la derecha
						           fontsize=8)
	
	
	# Si le hemos pasado una función de regresión
	if(len(reg)>0):
		x1 = np.linspace(x1_min, x1_max, 100)
		x2 = np.linspace(x2_min, x2_max,100)
		X1, X2 = np.meshgrid(x1,x2)
				
		for i in range(len(reg)):
			funtion = reg[i]

	
			if(a!=None and b!=None and i==0):	# Solo válida para la primera función
				if(show_niveles):
					plt.contourf(X1, X2,+funtion(X1,X2,a,b), levels=0, cmap=ListedColormap(colores), alpha=.2)
				contour = plt.contour(X1, X2, funtion(X1,X2,a,b),[0], colors = colores_lineas[i%len(colores_lineas)])
			elif(len(w) > 0):
				if(i==0 and show_niveles):
					plt.contourf(X1, X2,+funtion(w[i],[X1,X2]), levels=0, cmap=ListedColormap(colores), alpha=.2)
				contour = plt.contour(X1, X2, funtion(w[i],[X1,X2]),[0], colors = colores_lineas[i%len(colores_lineas)])
			else:
				if(i==0 and show_niveles):
					plt.contourf(X1, X2,+funtion(X1,X2), levels=0, cmap=ListedColormap(colores), alpha=.2)
				contour = plt.contour(X1, X2, funtion(X1,X2),[0], colors = colores_lineas[i%len(colores_lineas)])
			
			if(len(etiquetafun) == len(reg)):
				contour.collections[0].set_label(etiquetafun[i])
		plt.legend(loc='lower right')
	
			
	# Añadimos la leyenda de las clases de puntos
	plt.gca().add_artist(legend_classes)
	
	# Añadimos las etiquetas de los ejes
	plt.xlabel(labels_axes[0])
	plt.ylabel(labels_axes[1])
		
	# Establecemos los ejes
	plt.xlim(x1_min, x1_max)
	plt.ylim(x2_min, x2_max)
	
	# Asignamos el título
	plt.title(title)
	
	# Mostramos
	plt.show()





	
####################################################################
##
##	  	 	 	FUNCIONES AUXILIARES DE LOS EJERCICIOS
##
####################################################################	
def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b


def signo(x):
	if x>=0:
		return 1
	return -1


def fsigno(x,y,a,b):
	return signo(y - a*x - b)

def f(x,y,a,b):
	return y - a*x - b

# FUNCIONES EJERCICIO 1.2.c)
def f1(x,y):
	return (x-10)**2+(y-20)**2-400

def f2(x,y):
	return 0.5*(x+10)**2+(y-20)**2-400

def f3(x,y):
	return 0.5*(x-10)**2-(y+20)**2-400

def f4(x,y):
	return y-20*x**2-5*x+3


def getResults(X,y,f,a=None,b=None,w=[]):
	"""
	Recuenta las buenas y malas etiquetas: TP, FP, TN, FN

	Parameters
	----------
	X : 	Datos
	y : 	Etiquetas reales
	f : 	función etiquetadora
	a,b : 	Parámetros de una recta.
	w : 	Si la función es un producto escalar 

	Returns TP, FP, TN, F

	"""
	y_result=[]
	if(a!=None and b!=None):
		y_result = [ signo(f(X[i][0],X[i][1],a,b)) for i in range(len(X))]
	elif(len(w)>0):
		y_result = [ signo(f(w,X[i])) for i in range(len(X))]
	else:
		y_result = [signo(f(X[i][0],X[i][1])) for i in range(len(X))]
		
	TP, FP, TN, FN = 0, 0, 0, 0
	for i in range(len(y)):
		if(y[i]>0):
			if(y_result[i]>0):
				TP+=1
			else:
				FN+=1
		else:
			if(y_result[i]>0):
				FP+=1
			else:
				TN+=1
				
				
	return TP,FP,TN,FN 

def accuracy(X,y,f,a=None,b=None,w=None):
	"""
	Accuracy del modelo

	Parameters
	----------
	X : 	Datos
	y : 	Etiquetas reales
	f : 	función etiquetadora
	a,b : 	Parámetros de una recta.
	w : 	Si la función es un producto escalar 

	Returns (TP+TN)/(P+N)
	"""
	TP,FP,TN,FN = getResults(X,y,f,a=a,b=b,w=w)
	return (TP+TN)/(TP+FP+TN+FN)

def recall(X,y,f,a=None,b=None,w=[]):
	"""
	Recall del modelo

	Parameters
	----------
	X : 	Datos
	y : 	Etiquetas reales
	f : 	función etiquetadora
	a,b : 	Parámetros de una recta.
	w : 	Si la función es un producto escalar 

	Returns (TP)/(TP+FP)
	"""
	TP,FP,TN,FN = getResults(X,y,f,a=a,b=b,w=w)
	
	return (TP)/(TP+FN)


def specificity(X,y,f,a=None,b=None,w=[]):
	"""
	Specificity del modelo

	Parameters
	----------
	X : 	Datos
	y : 	Etiquetas reales
	f : 	función etiquetadora
	a,b : 	Parámetros de una recta.
	w : 	Si la función es un producto escalar 

	Returns (TP)/(TP+FP)
	"""
	TP,FP,TN,FN = getResults(X,y,f,a=a,b=b,w=w)
	
	return TN/(FP+TN)


def evaluatingOutputBinaryCase(X,y,f,funname='',a=None,b=None,w=[], mostrar_grafica=True):
	"""
	Specificity del modelo

	Parameters
	----------
	X : 	  	 	Datos
	y : 	 	 	Etiquetas reales
	f : 	 	 	función etiquetadora
	funname: 	 	Nombre de la función para escribir en el título
	a,b : 	 	 	Parámetros de una recta.
	w : 	 	 	Si la función es un producto escalar 
	mostrar_grafica:Muestra gráfica  TPR vs FPR

	Returns accuracy and show a plot showing true positive rate vs false positive rate
	"""
	acc = accuracy(X,y,f,a=a,b=b,w=w)
	TPrate = recall(X,y,f,a=a,b=b,w=w)
	FPrate = 1-specificity(X,y,f,a=a,b=b,w=w)
	
	if(mostrar_grafica):
		fig, ax = plt.subplots()
	
		ax.plot([0,1], [0,1], 'bo-', ms=0.2, label='random classifier')
		ax.plot([FPrate],[TPrate],'ko-', ms=7, label='Punto de nuestro clasificador')
		
		# Definimos la leyenda en la parte superior derecha
		ax.legend(loc='upper right', shadow=False)
		
		ax.set(title='Evaluating Output: '+funname)
		ax.set_ylabel('True Positive Rate')
		ax.set_xlabel('False Positive Rate')
		
		# Mostramos
		plt.show()	 
		
	return acc
	
	
####################################################################
##
##	  	 	 	FUNCIONES QUE DEFINEN LOS EJERCICIOS
##
####################################################################	
def ejercicio1(x_unif, y, y_ruido, param_a, param_b):
	"""
	Ejercicio 1.
	EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente
	
	EJERCICIO 1.2: Generar etiquetas a los puntos anteriores usando la función f.
						a) Dibujar los puntos etiquetadas junto con la función f.
						b) Modificar aleatoriamente el 10% de las etiquetas y volver a dibujar.
						c) Representar como separador de clases las funciones f1,f2,f3,f4
	"""
	print('###### ---- EJERCICIO 1 ---- ######')

	N = 50
	dim = 2
	
		
	# Puntos uniformes
	x1 = simula_unif(N, dim, [-50,50])		
	# Puntos gaussianos
	x2 = simula_gaus(N, dim, np.array([5,7]))
	
	
	print('Ejercicio 1.1. Dibujar las nubes de puntos.')
	# Puntos uniformes
	display_graph(x1, np.zeros((N)),'N='+str(N)+', dim='+str(dim)+', rango=[-50,50] con simula_unif' )
			
	# Puntos gaussianos
	display_graph(x2, np.zeros((N)),'N='+str(N)+', dim='+str(dim)+', sigma en [5,7] con simula_gaus' )
	
	wait()

	print('Ejercicio 1.2. Generar etiquetas a los puntos anteriores usando la función f.')
	print('\t \t a) Dibujar los puntos etiquetadas junto con la función f.')
	print('\t \t b) Modificar aleatoriamente el 10% de las etiquetas y volver a dibujar.')
	print('\t \t c) Representar como separador de clases las funciones f1,f2,f3,f4')
	
	display_graph(x_unif, y,'Puntos sin ruido',etiquetas=['-1',' 1'], reg=[f], a=param_a, b=param_b, etiquetafun=['f(x,y)=y-ax-b=0'])
	
	
	
	#Recta
	display_graph(x_unif, y_ruido,'Puntos con ruido', etiquetas=['-1',' 1'], reg=[f], a=param_a, b=param_b, etiquetafun=['f(x,y)=y-ax-b=0'] )
	acc=evaluatingOutputBinaryCase(x_unif, y_ruido, f,funname='recta', a=param_a, b=param_b)
	print('\n\t \t Accuracy recta: '+str(acc))
	
	#Elipse1
	display_graph(x_unif, y_ruido,'Ejercicio1.2. Elipse1', etiquetas=['-1',' 1'], reg=[f1], etiquetafun=['f(x,y)=(x-10)^2+(y-20)^2-400=0'])
	acc=evaluatingOutputBinaryCase(x_unif, y_ruido, f1, funname='elipse 1')
	print('\t \t Accuracy elipse 1: '+str(acc))
	
	#Elipse2
	display_graph(x_unif, y_ruido,'Ejercicio1.2. Elipse2', etiquetas=['-1',' 1'], reg=[f2], etiquetafun=['f(x.y)=0.5(x+10)^2+(y-20)^2-400=0'])
	acc=evaluatingOutputBinaryCase(x_unif, y_ruido, f2, funname='elipse 2')
	print('\t \t Accuracy elipse 2: '+str(acc))
	
	#Hipérbola
	display_graph(x_unif, y_ruido,'Ejercicio1.2. Hipérbola', etiquetas=['-1',' 1'], reg=[f3], etiquetafun=['f(x,y)=0.5(x-10)^2-(y+20)^2-400=0'])
	acc=evaluatingOutputBinaryCase(x_unif, y_ruido, f3, funname='hipérbola')
	print('\t \t Accuracy hipérbola: '+str(acc))
	
	#Parábola
	display_graph(x_unif, y_ruido,'Ejercicio1.2. Parábola', etiquetas=['-1',' 1'], reg=[f4], etiquetafun=['f(x,y)=y-10x^2-5x+3=0'])
	acc=evaluatingOutputBinaryCase(x_unif, y_ruido, f4, funname='parábola')
	print('\t \t Accuracy parábola: '+str(acc))
	


###############################################################################
###############################################################################
###############################################################################


# Ejercicio 2

def funcionlineal(w,x):
	"""
	Función para ajustar
	"""
	x_new = [1,x[0],x[1]]
	
	return np.dot(x_new,w)


def ajusta_PLA(datos, label, max_iter, vini, mostrarProceso=False):
	"""
	PLA algorithm.
		- datos: 	 	  Conjunto de datos
		- label: 	 	  Etiquetas de datos
		- max_iter: 	  Máximo de iteraciones permitidas
		- vini: 	 	  Valor inicial de nuestros pesos
		- mostrarProceso: Muestra gráficamente el proceso del algoritmo
	"""
	w = vini
	hay_cambio = False
	for iteration in range(max_iter):
		hay_cambio = False
		for i in range(len(datos)):	# Para cada item de datos
			if(funcionlineal(w,datos[i])*label[i]<=0): # Si son de distinto signo
				# Actualizamos
				x = np.array([1, datos[i][0], datos[i][1]])
				w = w + label[i]*x
				hay_cambio = True
			
		if(hay_cambio == False):	# Si no ha habido ningún cambio en toda la iteración
			return w, iteration
		
		# Mostramos por pantalla el estado actual
		if(mostrarProceso):
			display_graph(datos, label,'Algoritmo PLA Ejercicio 2a: ITER:'+str(iteration), etiquetas=['-1',' 1'], reg=[funcionlineal], w=[w], etiquetafun=['f(x)=w@x=0'] )

		
	return w, max_iter 


def ejercicio2(x_unif, y, y_ruido, param_a, param_b):
	print('\n\n###### ---- EJERCICIO 2 ---- ######')
	# Ejecutamos PLA con inicilización a 0
	w1, it1 = ajusta_PLA(x_unif, y, 1000, np.array([0,0,0]), mostrarProceso=False)
	print('Valor de iteraciones necesario para converger sin ruido con inicio en 0: {}'.format(np.mean(np.asarray(it1))))
	
	# Ejecutamos PLA con inicialización aleatoria
	w2, _ = ajusta_PLA(x_unif, y, 1000, np.random.rand(3), mostrarProceso=False)
	
	# Mostramos por pantalla
	display_graph(x_unif, y,'Algoritmo PLA Ejercicio 2a sin ruido', etiquetas=['-1',' 1'], reg=[funcionlineal,funcionlineal], w=[w1,w2], etiquetafun=['Inicio en 0','Inicio aleatorio'])

	# Random initializations
	iterations = []
	for i in range(0,10):
		vini = np.random.rand(3)
		w, it = ajusta_PLA(x_unif, y, 1000, np.array(vini), mostrarProceso=False)	
		iterations.append(it)
	    
	print('Valor medio de iteraciones necesario para converger sin ruido con inicio aleatorio: {}'.format(np.mean(np.asarray(iterations))))
	
	wait()
	
	# Ahora con los datos del ejercicio 1.2.b
	# Ejecutamos PLA con inicialización a 0
	w1, it1 = ajusta_PLA(x_unif, y_ruido, 1000, np.array([0,0,0]), mostrarProceso=False)
	print('Valor de iteraciones necesario para converger con ruido con inicio en 0: {}'.format(np.mean(np.asarray(it1))))

	# Ejecutamos PLA con inicialización aleatoria
	w2, _ = ajusta_PLA(x_unif, y_ruido, 1000, np.random.rand(3), mostrarProceso=False)
	
	# Mostramos por pantalla con incialización aleatoria
	# Mostramos por pantalla con inicialización a 0
	display_graph(x_unif, y_ruido,'Algoritmo PLA Ejercicio 2a con ruido', etiquetas=['-1',' 1'], reg=[funcionlineal,funcionlineal], w=[w1,w2], etiquetafun=['Inicio en 0','Inicio aleatorio'] )
	
	# Random initializations
	iterations = []
	for i in range(0,10):
		vini = np.random.rand(3)
		w, it = ajusta_PLA(x_unif, y_ruido, 1000, np.array(vini), mostrarProceso=False)	
		iterations.append(it)
	    
	print('Valor medio de iteraciones necesario para converger con ruido con inicio aleatorio: {}'.format(np.mean(np.asarray(iterations))))
	
	wait()
	

###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT
def error_reglog(w,X,y):
	"""Expresión del error cometido por un modelo de regresión logística.
         - w: vector de pesos.
         - X: vector de características con el 1 extra para el coef indep.
         - y: vector de etiquetas.
	"""		
	return np.mean(np.log(1+np.exp((-np.array(y))*np.dot(X,w))))

def graderror_reglog(w,X,y):
	"""Gradiente del error cometido por un modelo
       de regresión logística.
         - w: vector de pesos.
         - X: vector de características con el 1 extra para el coef indep.
         - y: vector de etiquetas.
	"""
	return -y*X / (1 + np.exp(y * np.dot(X,w)))
		

def sgdRL(X, y, w, lr, epsilon):
	"""
	Algoritmo de regresión logística con SGD. Batches de tamaño 1.
	Devuelve los pesos obtenidos y el nº de iteraciones.
		- X: 	 	 	datos
		- y: 	 	 	etiquetas
		- w: 	 	 	punto inicial
		- lr: 	 	 	learning rate
		- epsilon: 	 	condición de parada (distancia entre pesos de épocas)
	"""
	indices = np.arange(len(X))
	iterations = 0
	cond_parada = False
	
	while not cond_parada:
		# Guardamos los pesos anteriores
		w_old = w.copy()
		
		# Barajamos los índices
		np.random.shuffle(indices)
		
		# Actualizamos pesos
		for i in indices:
			w = w - lr*graderror_reglog(w,X[i],y[i])
			
		# Comprobamos si se cumple la condición de parada
		cond_parada = np.linalg.norm(w_old - w) < epsilon
		
		iterations += 1
	
	
	return w, iterations


def generate_dataset(N,dim,rango, a, b):
	"""
	Genera dataset uniforme de dim dimensiones con valores dentro de rango

	Parameters
	----------
	N : Entero
		Cantidad de datos.
	dim : Entero
		Dimensión de los datos.
	rango : vector2D
		Intervalo donde se encuentran los valores de nuestros datos.
		
	a, b: reales
		Coeficientes de la recta usada para etiquetar

	Returns
	-------
		X: 	conjunto de datos con un 1 en su primera componente
		y:  etiquetas

	"""
	# Generamos los datos
	X = simula_unif(N,dim,rango)
	y = [ fsigno(X[i][0], X[i][1], a, b) for i in range(len(X))]
	
	return X, y

def ejercicio3():
	print('\n\n###### ---- EJERCICIO 3 ---- ######')
	puntos = simula_unif(2,2,[0,2])
	
	# Definimos la recta que marcará las etiquetas: y=ax+b
	a = (puntos[1][1]-puntos[0][1])/(puntos[1][0]-puntos[0][0])
	b = -a*puntos[0][0]+puntos[0][1]
	
	
	Ein = 0
	Eout = 0
	accin = 0
	accout = 0
	iterations=0
	num_exp=100
	print('Se ejecutarán 100 experimentos. El proceso puede llevar unos minutos puesto que el cálculo del accuracy, Ein y Eout en cada iteración llevan un coste computacional alto.')
	for times in range(num_exp):
		if times%25==0:
			print('Experimento nº'+str(times)+'...')
		# Generamos los datos
		X, y = generate_dataset(100,2,[0,2],a,b)
		X_mod = [np.array([1, X[i][0], X[i][1]]) for i in range(len(X))]
		
		X_test, y_test = generate_dataset(1000,2,[0,2],a,b)
		X_test_mod = [np.array([1, X_test[i][0], X_test[i][1]]) for i in range(len(X))]
		
		# Ejecutamos la regresión logística
		w, it = sgdRL(X_mod,y,np.zeros(X.shape[1]+1),0.05,0.01)
	
		if times==0:
			# Mostramos etiquetado
			display_graph(X, y,'Etiquetado en la iteración '+str(times), etiquetas=['-1',' 1'], reg=[f], a=a, b=b, etiquetafun=['Función real etiquetadora'] )
			
			# Mostramos datos test y train tras el ajuste
			display_graph(X, y,'Train en la iteración '+str(times), etiquetas=['-1',' 1'], reg=[funcionlineal], w=[w], etiquetafun=['f(x)=w@x=0'] )
			display_graph(X_test, y_test,'Test en la iteración '+str(times), etiquetas=['-1',' 1'], reg=[funcionlineal], w=[w], etiquetafun=['f(x)=w@x=0'] )
			
	
		# Calculamos el error y contamos iteraciones
		iterations += it
		Ein += error_reglog(w, X_mod, y)
		Eout += error_reglog(w, X_test_mod, y)
		accin += accuracy(X, y, funcionlineal, w=w)
		accout += accuracy(X_test, y_test, funcionlineal, w=w)
		
		
	iterations_mean = iterations/num_exp
	Ein_mean = Ein/num_exp
	Eout_mean = Eout/num_exp
	accuracy_mean_in = accin/num_exp
	accuracy_mean_out = accout/num_exp
	print('Iteraciones medias: '+str(iterations_mean))
	print('Dentro de la muestra:')
	print('\t Accuracy medio: '+str(accuracy_mean_in))
	print('\t Ein medio = '+str(Ein_mean))
	print('Fuera de la muestra:')
	print('\t Accuracy medio: '+str(accuracy_mean_out))
	print('\t Eout medio = '+str(Eout_mean))
	

	wait()
	    



###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos

# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y


def cota_hoeffding(err, n, m, delta):
    """Cota de Hoeffding para el error de generalización.
         - err: error a partir del cual generalizamos.
         - n: tamaño del conjunto usado para calcular 'err'.
         - m: tamaño de la clase de hipótesis usada para calcular 'err'.
         - delta: tolerancia.
	"""
    return err + np.sqrt((1 / (2 * n)) * np.log((2 * m) / delta))


def cota_vc(err, n, vc, delta):
    """Cota para el error de generalización basada en la dimensión VC.
         - err: error a partir del cual generalizamos.
         - n: tamaño del conjunto usado para calcular 'err'.
         - vc: dimensión VC del clasificador usado para calcular 'err'.
         - delta: tolerancia.
	"""
    return err + np.sqrt((8 / n) * np.log(4 * ((2 * n) ** vc + 1) / delta))


def Err_clasif(X,y,w):
	"""
	Funcion para calcular el error de clasificación
		- X: 	características sin el 1 extra
		- y: 	etiquetas
		- w: 	pesos
	"""	
	"""
	incorrect = 0.0
	for i in range(len(y)):
		if(np.dot(np.array([1, X[i][0], X[i][1]]),w)*y[i]<=0): # Si tienen distinto signo
			incorrect +=1.0
	
	return incorrect/len(y)
	"""
	incorrect = [signo(np.dot([1, x[0], x[1]],w)) for x in X] != y
	return np.mean(incorrect)

def Err_regresion(x,y,w):
	"""
	Funcion para calcular el error de regresión
	"""
	return np.square(np.dot(x,w)-y).mean() 	 	 	
		
def gradErr_regresion(X,Y,w):
	"""
	 Gradiente del error de regresión
	"""
	grad = 2*np.dot(np.transpose(X), (np.dot(X,w)-Y))/len(X)
	return np.array(grad)


def sgd(x, y, w, batch_size, epochs, lr):
	"""
	Algoritmo de gradiente descendente estocástico. Parámetros:
		x 	 	 	Vector de características
		y 	 	 	Etiqueta
		w 			Punto actual 
		batch_size 	Tamaño del minibatch
		epochs 	 	Número de épocas a entrenar
		lr 		 	Learning rate (coma flotante)
	
	"""
	if(batch_size<=0 and batch_size>len(x)):
		print('ERROR: El tamaño del batch tiene que ser mayor que 0 y menor que la muestra')
	
	trayectoria=[w]	# Nos servirá para ver los puntos que vamos sacando con el gradiente descendente
	
	n=0			# Cuenta el batch por el que vamos
	for i in range(epochs*(len(x)//batch_size)):	# Recomendable una 500-1000 iteraciones
		inicio = (n*batch_size)
		fin = ((n+1)*batch_size)
		
		x_shuffled, y_shuffled= shuffle(x,y)
		
		# Condición que controla que los índices tengan sentido (Cogemos el último minibatch más grande)
		if( len(x)-fin < batch_size):
			fin = len(x)- 1	
			n=0
			
		x_batch = x_shuffled[ inicio : fin ]
		y_batch = y_shuffled[ inicio : fin ]
		
		w_new = w - lr*gradErr_regresion(x_batch, y_batch, w)
		
		iguales=True
		for i in range(len(w)):
			if w_new[i]!=w[i]:
				iguales = False
				
		if(iguales): 	# Nos quedamos en un punto de estabilidad
			return w, np.array(trayectoria)
		
		w = w_new

		trayectoria += [w.copy()]

			
	return w, np.array(trayectoria)

def PLApocket(datos, label, max_iter, vini, estocasticidad=False, mostrarProceso=False):
	"""
	PLA algorithm.
		- datos: 	 	  Conjunto de datos
		- label: 	 	  Etiquetas de datos
		- max_iter: 	  Máximo de iteraciones permitidas
		- vini: 	 	  Valor inicial de nuestros pesos
		- estocasticidad: Recorrer los datos de forma aleatoria
		- mostrarProceso: Muestra gráficamente el proceso del algoritmo
	"""
	w = vini
	hay_cambio = False
	w_best = w.copy()
	err_actual = Err_clasif(datos,label,w)
	
	indices = np.arange(len(datos))
	for iteration in range(max_iter):
		hay_cambio = False
		
		if(estocasticidad):
			# Barajamos los índices
			np.random.shuffle(indices)
		
		for i in indices:	# Para cada item de datos
			if(funcionlineal(w,datos[i])*label[i]<=0): # Si son de distinto signo
				# Actualizamos
				x = np.array([1,datos[i][0],datos[i][1]])
				w = w + label[i]*x
				hay_cambio = True
				
				
		err_new = Err_clasif(datos,label,w)
		if(iteration%100==0):
			print('Iteracion: '+str(iteration)+'...')
			
		if(err_actual>err_new): # Si la solución mejora
			w_best = w.copy()
			err_actual = err_new
			
		if(hay_cambio == False):	# Si no ha habido ningún cambio en toda la iteración
			return w_best, iteration
		
		# Mostramos por pantalla el estado actual
		if(mostrarProceso):
			display_graph(datos, label,'Algoritmo PLApocket: ITER:'+str(iteration), etiquetas=['-1',' 1'], reg=[funcionlineal], w=w, etiquetafun=['f(x)=w@x=0'] )

		
	return w_best, max_iter 


def bonus():
	print('\n\n###### ---- BONUS ---- ######')

	# Lectura de los datos de entrenamiento
	x_train, y_train = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
	# Lectura de los datos para el test
	x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])
	
	
	#mostramos los datos
	fig, ax = plt.subplots()
	ax.plot(np.squeeze(x_train[np.where(y_train == -1),1]), np.squeeze(x_train[np.where(y_train == -1),2]), 'o', color='red', label='4')
	ax.plot(np.squeeze(x_train[np.where(y_train == 1),1]), np.squeeze(x_train[np.where(y_train == 1),2]), 'o', color='blue', label='8')
	ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
	ax.set_xlim((0, 1))
	plt.legend()
	plt.show()
	
	fig, ax = plt.subplots()
	ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
	ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
	ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
	ax.set_xlim((0, 1))
	plt.legend()
	plt.show()
	
	wait()
	
	#LINEAR REGRESSION FOR CLASSIFICATION 
	print('Ejecutamos SGD...')
	w_reg, _ = sgd(x_train, y_train, np.array([0.0, 0.0, 0.0], np.float64), 32, 50, 0.01)
	display_graph(x_train[:,1:3], y_train,'BONUS: Regresión', etiquetas=['4',' 8'], reg=[funcionlineal], w=[w_reg], etiquetafun=['f(x)=w@x=0'] )	
	acc_reg = evaluatingOutputBinaryCase(x_train[:,1:3], y_train, funcionlineal, funname='regresion',w=w_reg)
	
	
	#POCKET ALGORITHM
	# Pesos aleatorios
	print('Ejecutamos PLA-Pocket con pesos aleatorios...')
	w_pocketrand, iteraciones = PLApocket(x_train[:,1:3],y_train,500,np.random.rand(3), estocasticidad=False, mostrarProceso=False)
	display_graph(x_train[:,1:3], y_train,'BONUS: PLA Pocket inicio random', etiquetas=['4',' 8'], reg=[funcionlineal], w=[w_pocketrand], etiquetafun=['f(x)=w@x=0'] )
	acc_pocketrand = evaluatingOutputBinaryCase(x_train[:,1:3], y_train, funcionlineal, funname='PLA-Pocket random',w=w_pocketrand)

	
	# Pesos obtenidos con regresión
	print('Ejecutamos PLA-Pocket con los pesos del SGD como pesos iniciales...')
	w_pocketreg, iteraciones = PLApocket(x_train[:,1:3],y_train,500,w_reg, estocasticidad=False, mostrarProceso=False)
	display_graph(x_train[:,1:3], y_train,'BONUS: PLA Pocket inicio regresión', etiquetas=['4',' 8'], reg=[funcionlineal], w=[w_pocketreg], etiquetafun=['f(x)=w@x=0'])
	acc_pocketreg = evaluatingOutputBinaryCase(x_train[:,1:3], y_train, funcionlineal, funname='PLA-Pocket regresión',w=w_pocketreg)

	wait()
	
	
	#COTA SOBRE EL ERROR
	weights = [w_reg, w_pocketrand, w_pocketreg]
	accuracies = [acc_reg, acc_pocketrand, acc_pocketreg]
	
	metodo = ['Regresión', 'PLA pocket inicio random', 'PLA pocket inicio con regresión']
	for i in range(3):
		delta = 0.05
		Ein = Err_clasif(x_train[:,1:3],y_train,weights[i])
		Etest = Err_clasif(x_test[:,1:3], y_test, weights[i])
		
		print('\n-----'+metodo[i]+'-----')
		print('ERRORES:')
		print('\t Accuracy='+str(accuracies[i]))
		print('\t Ein='+str(Ein))
		print('\t Etest='+str(Etest))
		print('COTAS:')
		print('\t Cota con Vapnik-Chervonenkis: '+str(cota_vc(Ein, len(x_train), 3, delta)))
		print('\t Cota con Hoeffding:\t '+str(cota_hoeffding(Ein, len(x_train),  2 ** (64 * 3), delta)))
		print('\t Cota con Hoeffding usando test: '+str(cota_hoeffding(Etest, len(x_test), 1, delta)))
		
		
	
	display_graph(x_train[:,1:3], y_train,'BONUS TRAIN', etiquetas=['4',' 8'], reg=[funcionlineal,funcionlineal,funcionlineal], w=[w_reg, w_pocketrand, w_pocketreg], etiquetafun=['Regresión','PLApocket inicio random','PLApocket inicio regresión'], show_niveles=False)
	
	display_graph(x_test[:,1:3], y_test,'BONUS TEST', etiquetas=['4',' 8'], reg=[funcionlineal,funcionlineal,funcionlineal], w=[w_reg, w_pocketrand, w_pocketreg], etiquetafun=['Regresión','PLApocket inicio random','PLApocket inicio regresión'], show_niveles=False)

###########################################
##                            	 	 	 ##
##	DATOS PARA LOS DISTINTOS EJERCICIOS  ##
##                             	 	 	 ##
###########################################

# Fijamos la semilla
np.random.seed(2)

N = 100
dim = 2

# Puntos uniformes
x1 = simula_unif(N, dim, [-50,50])		
# Generamos parametros
param_a, param_b = simula_recta([-50,50])
# Generamos etiquetas
y = np.array([fsigno(x1[i,0],x1[i,1], param_a, param_b) for i in range(x1.shape[0])])


# Modificamos el 10% de las etiquetas -1 y el 10% de las etiquetas +1
y_ruido = y.copy()
v_change = np.arange(len(y_ruido))
np.random.shuffle(v_change)
num_poschange=len(y_ruido[y_ruido>0])/10
num_negchange=len(y_ruido[y_ruido<0])/10
neg=0
pos=0
for i in range(len(v_change)):
	if y_ruido[v_change[i]]<0 and neg<num_negchange:
		y_ruido[v_change[i]]=1
		neg+=1
	elif y_ruido[v_change[i]]>0 and pos<num_poschange:
		y_ruido[v_change[i]]=-1
		pos+=1
		
		
# Quitamos warnings
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)    


################################
##                            ##
##	EJECUCIÓN DE LA PRÁCTICA  ##
##                            ##
################################
ejercicio1(x1, y, y_ruido, param_a, param_b)
ejercicio2(x1, y, y_ruido, param_a, param_b)
ejercicio3()
bonus()