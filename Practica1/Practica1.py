# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Pedro Gallego López
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as cm
from mpl_toolkits.mplot3d import Axes3D # DISPLAY FIGURE
from sklearn.utils import shuffle



SEED = 1
np.random.seed(SEED)


"""
#######################################################

DUDAS
- Para el apartado 2.a) cómo mostramos al expresión del gradiente

########################################################
"""


"""
Función para pausar la ejecución
"""
def wait():
	input("\n--- Pulsar tecla para continuar ---\n")
	
	
"""
Función signo
"""
def sign(x):
	if x >= 0:
		return 1
	return -1
	

"""
#######################################################

FUNCIONES PARA MOSTRAR POR PANTALLA

########################################################
"""
	
	
"""
Muestra una gráfica de la superficie con la trayectoria que ha seguido nuestro
algoritmo del gradiente descendente
	fun 	 	 	función superficie
	w 	 	 	 	punto final
	trayectoria 	trayectoria de puntos desde el punto inicial al final
	labels 	 	 	etiquetas de los ejes
	title 	 	 	título de la gráfica
	(x_min,x_max) 	rango del ejeX
	(y_min,y_max) 	rango del ejeY
"""
def display_surface(fun,w,trayectoria, trayectoria2=[], labels=['u','v','E(u,v)'], title='', x_min=-10, x_max=10, y_min=-10, y_max=10, vista=1):
	# Nos quedamos solo con 20 pasos de la trayectoria para que se ve más claro
	if(len(trayectoria)>20):
		trayectoria_20 = trayectoria[::len(trayectoria)//20]
	else:
		trayectoria_20 = trayectoria
		
	if(len(trayectoria2)>20):
		trayectoria2_20 = trayectoria2[::len(trayectoria2)//20]
	elif(len(trayectoria2)>0):
		trayectoria2_20 = trayectoria2
	
		
		
	
	
	x = np.linspace(x_min, x_max, 50)
	y = np.linspace(y_min, y_max, 50)
	X, Y = np.meshgrid(x, y)
	Z = fun(X, Y)
	fig = plt.figure()
	ax = Axes3D(fig)
		
	# Metemos la trayectoria del algoritmo
	# Cuanto más avanzado se esté en la trayctoria más pequeño es el punto
	for i in range(len(trayectoria_20)):		
		ax.plot(trayectoria_20[i][0], trayectoria_20[i][1], fun(trayectoria_20[i][0], trayectoria_20[i][1]),
																			   'k+', markeredgewidth=2, 
																			   markersize=10*(20-(float(i)/1.5))/20.0)
		
	if(len(trayectoria2)>0):
		for i in range(len(trayectoria_20)):		
			ax.plot(trayectoria2_20[i][0], trayectoria2_20[i][1], fun(trayectoria2_20[i][0], trayectoria2_20[i][1]),
																				   'k*', markeredgewidth=2, 
																				   markersize=10*(20-(float(i)/1.5))/20.0)
			
	surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1, cstride=1, cmap='jet', alpha=.7)
	
	# Metemos nuestro mínimo obtenido por el gradiente descendente
	min_point = np.array([w[0],w[1]], np.float64)
	min_point_ = min_point[:, np.newaxis]
	ax.plot(min_point_[0], min_point_[1], fun(min_point_[0], min_point_[1]), 'r*', markersize=10)


	ax.set(title=title)
	ax.set_xlabel(labels[0])
	ax.set_ylabel(labels[1])
	ax.set_zlabel(labels[2])
	
	if(vista==1):
		ax.view_init(90)
	elif(vista==2):
		ax.view_init(90,90)	
	else:
		ax.view_init(30,120)		
		
		
	plt.show()
	
	
"""
Muestra una gráfica de puntos junto con una función lineal
	 x 	 	 	Datos
	 y 	 	 	Etiquetas
	 title 	 	Título de la gráfica
	 label_axes Etiqueta de los ejes
	 w 	 	 	Pesos finales de la regresión
	 reg 	 	Regression function

"""
def display_regresion(X, y, title='', etiquetas=['',''], labels_axes = ['', ''], w=[], reg=None):
	
	# Establecemos los rangos que van a tener nuestros ejes de la gráfica
	x1_min = X[:, 0].min() - (X[:, 0].max()-X[:, 0].min())/10.0
	x1_max = X[:, 0].max() + (X[:, 0].max()-X[:, 0].min())/10.0
	
	x2_min = X[:, 1].min() - (X[:, 1].max()-X[:, 1].min())/10.0
	x2_max = X[:, 1].max() + (X[:, 1].max()-X[:, 1].min())/10.0
	
	# Hacemos plot de los puntos
	colores = np.array(['#FF9300', '#31BF00']) # [naranja, negro, verde]
	
	# Creamos el scatter plot dándole los colores a través del parámetro cmap
	#	-El parámetro edgecolor='k' es para que tengan bordes negros los puntos
	scatter = plt.scatter(X[:,0],X[:,1], c=y, cmap = cm.colors.ListedColormap(colores), edgecolor='k')
	
	# Definimos la leyenda que tendrá la gráfica
	plt.legend(handles = scatter.legend_elements()[0],	# Cogemos los puntos de colores de la salida anterior
			   title = "Classes",
			   labels = etiquetas,		# Cogemos las etiquetas dadas por la BD
	           scatterpoints=1,
	           loc='upper right',          			 	# Abajo a la derecha
	           fontsize=8)
	
	plt.xlabel(labels_axes[0])
	plt.ylabel(labels_axes[1])
	
	# Si le hemos pasado una función de regresión
	if(reg != None):
		x1 = np.linspace(x1_min, x1_max, 100)
		x2 = np.linspace(x2_min, x2_max,100)
		X1, X2 = np.meshgrid(x1,x2)
		
		plt.contourf(X1, X2,+reg(w, [X1,X2]),[0,100], colors=[colores[1]], alpha=.2)	# coloreamos el nivel interior
		plt.contourf(X1, X2,-reg(w, [X1,X2]),[0,100], colors=[colores[0], colores[1]], alpha=.2) #coloreamos el exterior

		plt.contour(X1, X2, reg(w, [X1,X2]),[0])
		
		
	plt.xlim(x1_min, x1_max)
	plt.ylim(x2_min, x2_max)
	
	plt.title(title)
	
	# Mostramos
	plt.show()



"""
#######################################################

MÉTODOS DE LA PRÁCTICA

########################################################
"""
"""""""""""""""""""""""""""""""""""
función del 1,2
"""""""""""""""""""""""""""""""""""
# Función a minimizar Ejercicio 1.2
def E(u,v):
    return ( (u**3)*np.exp(v-2) - 2*(v**2)*np.exp(-u) )**2   

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return 2*( (u**3)*np.exp(v-2) - 2*(v**2)*np.exp(-u) ) * ( 3*(u**2)*np.exp(v-2) + 2*(v**2)*np.exp(-u) ) 
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    return 2*( (u**3)*np.exp(v-2) - 2*(v**2)*np.exp(-u) ) * ( (u**3)*np.exp(v-2) - 4*v*np.exp(-u) )

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])


"""""""""""""""""""""""""""""""""""
función del 1,3. 
"""""""""""""""""""""""""""""""""""
# Función a minimizar Ejercicio 1.3
def f(x,y):
	return ((x+2)**2 + 2*(y-2)**2 + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y))

#Derivada parcial de f con respecto a x
def dfx(x,y):
    return np.float64( 2*(x+2) + 4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y) )
    
#Derivada parcial de f con respecto a y
def dfy(x,y):
    return np.float64( 4*(y-2) + 4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y) )

#Gradiente de f
def gradf(x,y):
    return np.array([dfx(x,y), dfy(x,y)])

# (df)^2/(dx)^2
def dfxx(x,y):
	return np.float64( 2 - 8*(np.pi**2)*np.sin(2*np.pi*x)*np.sin(2*np.pi*y) )

# (df)^2/(dy)^2
def dfyy(x,y):
	return np.float64( 4 - 8*(np.pi**2)*np.sin(2*np.pi*x)*np.sin(2*np.pi*y) )

# (df)^2/(dxdy)
def dfxy(x,y):
	return np.float64( 8*(np.pi**2)*np.cos(2*np.pi*x)*np.cos(2*np.pi*y) )

#Hessiano de f
def hessf(x,y):
    return np.array([[dfxx(x,y), dfxy(x,y)],[dfxy(x,y), dfyy(x,y)]])



"""""""""""""""""""""""""""""""""""
función auxiliar para comprobar métodos
"""""""""""""""""""""""""""""""""""
# Función a minimizar Ejercicio 1.3
def g(x,y):
	return ((x+2)**2 + 2*(y-2)**2)

#Derivada parcial de f con respecto a x
def dgx(x,y):
    return np.float64( 2*(x+2))
    
#Derivada parcial de f con respecto a y
def dgy(x,y):
    return np.float64( 4*(y-2))

#Gradiente de f
def gradg(x,y):
    return np.array([dgx(x,y), dgy(x,y)])

# (df)^2/(dx)^2
def dgxx(x,y):
	return np.float64( 2 )

# (df)^2/(dy)^2
def dgyy(x,y):
	return np.float64( 4)

# (df)^2/(dxdy)
def dgxy(x,y):
	return np.float64( 0)

#Hessiano de f
def hessg(x,y):
    return np.array([[dgxx(x,y), dgxy(x,y)],[dgxy(x,y), dgyy(x,y)]])


"""
Algoritmo de gradiente descendente. Parámetros:
	w 			Punto actual (np.array 2D flotantes)
	lr 		 	Learning rate (coma flotante)
	f 	 	  	Función a minimizar (función derivable)
	gradf 	 	Gradiente de f (np.array 2D flotantes)
	error 	 	Umbral inferior a superar
	iterations 	Número de iteraciones máximas (entero positivo)

"""
def gradient_descent(w, lr, fun, gradf, iterations=50, error=-np.inf ): 
	trayectoria=[w]	# Nos servirá para ver los puntos que vamos sacando con el gradiente descendente
	i=0
	umbral_superado=False
	while i<iterations and umbral_superado==False:
		
		f = fun(w[0],w[1])		# valor de la función
		if(f <= error):
			umbral_superado=True
		else:	
			w_new = w - lr*gradf(w[0],w[1])
			
			#print(str(w_new)+"= "+str(w)+"-0.1*"+str( gradf(w[0],w[1]) ) )
			
			if(w_new[0]==w[0] and w_new[1]==w[1]):
				umbral_superado=True
			
			w=w_new
			
			#print("iteración "+str(i)+",   w="+str(w)+",    f(w)="+str(f) )
			trayectoria += [w.copy()]
			i = i+1

			
	return w, np.array(trayectoria), i, umbral_superado
  
  
"""
Ejecución del ejercicio 1.2. Este ejercicio está descrito sobre la función ejercicio1()
"""
def ejercicio12():
	
	eta = 0.1 
	max_iteraciones = 10000000000
	error2get = 1e-14
	initial_point = np.array([1.0,1.0], np.float64)
	
	# Apartado 1.2.a) mostramos la expresión del gradiente
	print("\n1,2.a) Expresión del gradiente:")
	print("[2*( (u**3)*np.exp(v-2) - 2*(v**2)*np.exp(-u) ) * ( 3*(u**2)*np.exp(v-2) + 2*(v**2)*np.exp(-u) ), \
		   2*( (u**3)*np.exp(v-2) - 2*(v**2)*np.exp(-u) ) * ( (u**3)*np.exp(v-2) - 4*v*np.exp(-u) )]")
	wait()
	
	# Apartado 1.2.b) y 1.2.c)
	print("\n1.2.b) y 1,2.c) Cuándo se superó el umbral de 10^(-14)")
	titulo_grafica = 'Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente'

	w, trayectoria, iteraciones, umbral_superado = gradient_descent(initial_point, eta, E, gradE, 
																 iterations=max_iteraciones, error=error2get)
	
	if(umbral_superado):
		print("\tSe han tardado "+str(iteraciones)+" iteraciones en conseguir un valor de E inferior a 10^(-14)")
		print("\tEste valor se ha conseguido en el punto ("+str(w[0])+","+str(w[1])+")")
	else:
		print("\tSe han hecho "+str(iteraciones)+" iteraciones y no se ha conseguido superar el umbral propuesto.")
	
	display_surface(E,w,trayectoria, title=titulo_grafica, x_min=0.9, x_max=1.2, y_min=0.85, y_max=1.05, vista=2)
	display_surface(E,w,trayectoria, title=titulo_grafica, x_min=0.9, x_max=1.2, y_min=0.85, y_max=1.05, vista=0)
	display_surface(E,w,trayectoria, title=titulo_grafica, x_min=0.9, x_max=1.2, y_min=0.85, y_max=1.05, vista=1)
	
	wait()
	
	
	
"""
Ejecución del ejercicio 1.3. Este ejercicio está descrito sobre la función ejercicio1()
"""
def ejercicio13a(eta=0.1, initial_point=np.array([-1.0,1.0], np.float64), iteraciones = 50):
	print("Apartado 1.3.a) con "+str(eta)+ " de learning rate")

	w, trayectoria, _, _ = gradient_descent(initial_point, eta, f, gradf, iterations=iteraciones)
	titulo_grafica = 'Ejercicio 1.3.a. Función sobre la que se calcula el descenso de gradiente: lr='+str(eta)
	
	print('Valor mínimo alcanzado con lr='+ str(eta)+': f('+str(w[0])+','+str(w[1])+')='+str(f(w[0],w[1])))
	
	for i in range(2):
		if(eta==0.1):
			display_surface(f,w,trayectoria,labels=['x','y','f(x,y)'],title=titulo_grafica,x_min=-5, x_max=0, y_min=0, y_max=4,vista=i)
		else:
			display_surface(f,w,trayectoria,labels=['x','y','f(x,y)'],title=titulo_grafica,x_min=-1.4, x_max=-0.9, y_min=0.9, y_max=1.4,vista=i)

	return w



def ejercicio13b():
	print("Apartado 1.3.b)")
	puntos_inicio = np.array([np.array([-0.5,-0.5], np.float64), 
						   np.array([1.0,1.0], np.float64), 
						   np.array([2.1,-2.1], np.float64), 
						   np.array([-3.0,3.0], np.float64), 
						   np.array([-2.0,2.0], np.float64)])
	
	for w0 in puntos_inicio:
		# Ejecutamos el gradiente descendente
		w, trayectoria, _, _ = gradient_descent(w0, 0.01, f, gradf, iterations=100)
		
		# Mostramos la gráfica
		titulo_grafica = 'Ejercicio 1.3.b. pto inicial (' + str(w0[0])+','+str(w0[1])+')'
		display_surface(f,w,trayectoria,labels=['x','y','f(x,y)'], title=titulo_grafica, x_min=-3.5, x_max=2.5,y_min=-2.5, y_max=4)
		
		print('\tInicio en f('+str(w0[0])+','+str(w0[1])+')='+str(f(w0[0],w0[1])))
		
		
		print("\tSe ha alcanzado el valor f("+str(w[0])+','+str(w[1])+')='+str(f(w[0],w[1]))+'\n')
		
	

def ejercicio13():	
	display_surface(f,np.array([-1.0,1.0]),np.array([]),labels=['x','y','f(x,y)'],title='Gráfica de f',x_min=-5, x_max=0, y_min=0, y_max=4, vista =0)
	# Primer apartado
	ejercicio13a(eta=0.01, initial_point=np.array([-1.0,1.0], np.float64), iteraciones=50)
	ejercicio13a(eta=0.1, initial_point=np.array([np.float64(-1.0),np.float64(1.0)]), iteraciones=40)
	wait()

	# Segundo apartado
	ejercicio13b()
	wait()
	

"""""""""""""""""""""""""""
Ejecución del EJERCICIO 1. Se nos pide:
	1. Implementar el algoritmo de gradiente descendente
	2. Considerar la función E(u, v) = (u**3 * e**(v−2) −2v**2 * e**(−u) )**2.
	   Usar gradiente descendente para encontrar un mínimo de esta función, 
	   comenzando desde el punto (u, v) = (1, 1) y  usando una tasa de 
	   aprendizaje η = 0,1.
		   a) Calcular analíticamente y mostrar la expresión del gradiente de la función E(u, v)
		   b) ¿Cuántas iteraciones tarda el algoritmo en obtener por primera vez un valor de E(u, v)
			  inferior a 10 −14 . (Usar flotantes de 64 bits)
		   c) ¿En qué coordenadas (u, v) se alcanzó por primera vez un valor igual o menor a 10 −14
			  en el apartado anterior.
	3. (2 puntos) Considerar ahora la función f (x, y) = (x + 2)**2 + 2(y − 2)**2 + 2 sin(2πx) sin(2πy)
		   a) Usar gradiente descendente para minimizar esta función. Usar como punto inicial
			  (x 0 = −1, y 0 = 1), (tasa de aprendizaje η = 0,01 y un máximo de 50 iteraciones.
			  Generar un gráfico de cómo desciende el valor de la función con las iteraciones. Repetir
			  el experimento pero usando η = 0,1, comentar las diferencias y su dependencia de η.
		   b) Obtener el valor mínimo y los valores de las variables (x, y) en donde se alcanzan
		      cuando el punto de inicio se fija en: (−0,5, −0,5),(1, 1), (2,1, −2,1),(−3, 3),(−2, 2),
			  Generar una tabla con los valores obtenidos. Comentar la depenpendecia del punto
			  inicial.
	4. (1.5 punto) ¿Cuál sería su conclusión sobre la verdadera dificultad de encontrar el mínimo
	   global de una función arbitraria?
	   
"""""""""""""""""""""""""""
def ejercicio1():	
	print('Ejercicio 1. EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
	
	# Primer apartado implementado arriba en la función gradient_descent
	
	# Segundo apartado
	ejercicio12()
	
	# Tercer apartado
	ejercicio13()
	
	# Cuarto apartado en la memoria
	



###############################################################################
###############################################################################
###############################################################################
###############################################################################

"""
Funciones de lectura de datos
"""
def readData(file_x, file_y):
	label5 = 1
	label1 = -1
	
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y


def lectura_datos():
	# Lectura de los datos de entrenamiento
	x_train, y_train = readData('datos/X_train.npy', 'datos/y_train.npy')
	# Lectura de los datos para el test
	x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')
	
	return x_train, y_train, x_test, y_test



"""
Simula datos en un cuadrado [-size,size]x[-size,size]. Nos devuelve N coordenadas 
2D de puntos uniformemente muestreados dentro del cuadrado definido
"""
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))


# Funcion para calcular el error
def Err(x,y,w):
	# np.dot=producto matricial || np.square = calcula el cuadrado de sus elementos || np.mean = calcula la media
	return np.square(np.dot(x,w)-y).mean() 	 	 	
		
# Gradiente del error
def gradErr(X,Y,w):
	grad = 2*np.dot(np.transpose(X), (np.dot(X,w)-Y))/len(X)

	return np.array(grad)



"""
Función de regresión f(x1,x2) = w0 + w1*x1 + w2*x2. 
	x 	 	 	Coordenadas 2D
"""
def regf(w, x):
	x_new = [1, x[0], x[1]]
	return np.dot(x_new,w)

"""
Función con datos cuadráticos
	x 	 	 	Coordenadas 2D
"""
def regfcuadratica(w,x):
	x_new = [1, x[0],x[1], x[0]*x[1], x[0]**2, x[1]**2]
	return np.dot(x_new, w)

"""
Función del apartado 2.2.b) f(x1,x2)=sign((x1-0.2)^2+x2^2-0.6))
"""
def f22b(x1, x2):
	return sign((x1-0.2)**2+x2**2-0.6) 


"""
Algoritmo de gradiente descendente estocástico. Parámetros:
	x 	 	 	Vector de características
	y 	 	 	Etiqueta
	w 			Punto actual 
	batch_size 	Tamaño del minibatch
	epochs 	 	Número de épocas a entrenar
	lr 		 	Learning rate (coma flotante)

"""
def sgd(x, y, w, batch_size, epochs, lr): # en el video dice que parámetros podemos incluir aquí
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
		
		w_new = w - lr*gradErr(x_batch, y_batch, w)
		
		iguales=True
		for i in range(len(w)):
			if w_new[i]!=w[i]:
				iguales = False
				
		if(iguales): 	# Nos quedamos en un punto de estabilidad
			return w, np.array(trayectoria)
		
		w = w_new

		trayectoria += [w.copy()]

			
	return w, np.array(trayectoria)




"""
Algoritmo de la pseudo-ersa.
	X 	 	Matriz de datos
	y 	 	Vector de etiquetas
"""
def pseudoinverse(X, y):
    return np.dot(np.dot(np.linalg.inv((np.dot(X.T,X))),X.T), y)	# w = (X.transpose * X)^{-1}*X.transpose * y




"""
Ejecución del ejercicio 2.1. Este ejercicio está descrito sobre la función ejercicio2()
"""
def ejercicio21():
	print('Apartado 2.1)')
	# Leemos los datos
	x_train, y_train, x_test, y_test = lectura_datos()
	
	# SGD
	print('Ejecutando el algoritmo de gradiente descendente estocástico')
	weights, _ = sgd(x_train, y_train, np.array([0.0, 0.0, 0.0], np.float64), 32, 50, 0.01)
	
	# Calculamos errores
	Ein = Err(x_train,y_train,  w=weights)
	Eout = Err(x_test,y_test,  w=weights)

	# Mostramos
	print ('\tBondad del resultado para grad. descendente estocastico:')
	display_regresion(x_train[:,1:3], y_train, title='Conjunto train', etiquetas=['número 1', 'número 5'],
					   labels_axes = ['Intensidad promedio', 'Simetría'], w=weights, reg=regf)
	print('\t\tEin: '+ str(Ein))	
	display_regresion(x_test[:,1:3], y_test, title='Conjunto test', etiquetas=['número 1', 'número 5'],
					   labels_axes = ['Intensidad promedio', 'Simetría'], w=weights, reg=regf)
	print('\t\tEout: '+ str(Eout)+'\n')
	
	
	# Pseudo-Inversa
	print('Ejecutando el algoritmo de la pseudo-inversa')
	weights = pseudoinverse(x_train, y_train)
	
	# Calculamos errores
	Ein = Err(x_train,y_train,  w=weights)
	Eout = Err(x_test,y_test,  w=weights)

	# Mostramos
	print ('\tBondad del resultado para grad. descendente estocastico:')
	display_regresion(x_train[:,1:3], y_train, title='Conjunto train', etiquetas=['número 1', 'número 5'],
					   labels_axes = ['Intensidad promedio', 'Simetría'], w=weights, reg=regf)
	print('\t\tEin: '+ str(Ein))	
	display_regresion(x_test[:,1:3], y_test, title='Conjunto test', etiquetas=['número 1', 'número 5'],
					   labels_axes = ['Intensidad promedio', 'Simetría'], w=weights, reg=regf)
	print('\t\tEout: '+ str(Eout)+'\n')


"""
Genera un dataset compuesto de:
	Entrenamiento:
		-x_train 	 	1000 ptos uniformemente distribuidos en [-1,1]x[-1,1]
						etiquetados por y_train según la función f22b con una
						aleatoriedad añadida
		-y_train 	 	etiquetas de x_train
	Test:
		-x_test 	 	similar a x_train
		-y_test 	 	etiquetas de x_test
"""	
def generate_dataset_lineal(N,d,size, seed=1):
	# Generamos una muestra de entrenamiento de N=1000 puntos en [-1,1]x[-1,1]
	x = simula_unif(N, d, size)
	x_train = np.array([ [1, x[i][0],x[i][1]] for i in range(N) ])
	
	# Generamos una muestra de test de N=1000 puntos en [-1,1]x[-1,1]
	#np.random.seed(np.random.randint(1,1000)) 	 	# semilla random
	np.random.seed(seed)
	x = simula_unif(N, 2, 1)
	x_test = np.array([ [1, x[i][0],x[i][1]] for i in range(N) ])
	
	
	# Definimos las etiquetas
	y_train = np.array([f22b(x_train[i][1],x_train[i][2]) for i in range(N)])
	y_test = np.array([f22b(x_test[i][1],x_test[i][2]) for i in range(N)])
	
	v_change = np.random.choice([-1,1], N, p=[0.9,0.1])		# Incorporamos aleatoriedad a las etiquetas
	y_train = y_train * v_change
	y_test = y_test * v_change

	return x_train, y_train, x_test, y_test

"""
Genera un dataset a partir de los datos de generate_dataset_lineal donde cada dato
tendrá las características: (1, x1, x2, x1*x2, x1^2, x2^2)
"""	
def generate_dataset_nl(N, d, size, seed):
	# Cogemos los datos de partida
	xo_train, y_train, xo_test, y_test = generate_dataset_lineal(N,d,size,seed)
		
	# Transformamos los datos
	x_train = np.array([[1, xo_train[i][1], xo_train[i][2], xo_train[i][1]*xo_train[i][2], xo_train[i][1]**2, xo_train[i][2]**2] for i in range(len(xo_train))])
	x_test = np.array([[1, xo_test[i][1], xo_test[i][2], xo_test[i][1]*xo_test[i][2], xo_test[i][1]**2, xo_test[i][2]**2] for i in range(len(xo_test))])
	
	return x_train, y_train, x_test, y_test
	

"""
Ejecución del ejercicio 2.2. Este ejercicio está descrito sobre la función ejercicio2()
"""
def ejercicio22(datos):	
	ITERACIONES = 1000
	N=1000 	 	# tamaño de la muestra
	print('Se van a ejecutar 1000 iteraciones. Tiempo estimado: de 2min a 3min')
	
	# Establecemos como queremos los datos
	if(datos=='lineal'):
		dataset = generate_dataset_lineal
		funcion = regf
	else:
		dataset = generate_dataset_nl
		funcion = regfcuadratica
	
	Ein_medio=0 	 	# error interno medio
	Eout_medio=0 	 	# error externo medio
	for iteracion in range(ITERACIONES):
		np.random.seed(iteracion)
		if iteracion%100 ==0:
			print('Iteración: '+ str(iteracion))
		
		# Obtenemos los datos
		x_train, y_train, x_test, y_test = dataset(N,2,1, iteracion+1000)		
		
		
		#print('Ejecutando el SGD con los datos uniformes en [-1,1]x[-1,1]')
		weights, trayectoria = sgd(x_train, y_train, np.array([0.5 for i in range(x_train.shape[1])]), 32, 15, 0.1)
		
		# Calculamos errores
		Ein = Err(x_train,y_train,  w=weights)
		Ein_medio += Ein
		
		Eout = Err(x_test, y_test, weights)
		Eout_medio += Eout
	
		# Mostramos
		if(iteracion==0):
			print ('\tBondad del resultado para grad. descendente estocastico:')
			display_regresion(x_train[:,1:x_train.shape[1]], y_train, title='Conjunto train', etiquetas=['naranja', 'verde'],
							   labels_axes = ['', ''], w=weights, reg=funcion)
			print('\t\tEin: '+ str(Ein))	
			
	
	Ein_medio = Ein_medio / ITERACIONES
	Eout_medio = Eout_medio / ITERACIONES
	print ('\tBondad del resultado lineal para los datos x = (1,x1,x2)')
	print('\t\tMedia Ein: '+ str(Ein_medio))
	print('\t\tMedia Eout: '+ str(Eout_medio))
	
	
	

	
	
	

"""""""""""""""""""""""""""
Ejecución del EJERCICIO 2. Se nos pide:
	Este ejercicio ajusta modelos de regresión a vectores de características extraidos de imágenes
	de digitos manuscritos. En particular se extraen dos características concretas que miden: el valor
	medio del nivel de gris y la simetría del número respecto de su eje vertical. Solo se seleccionarán
	para este ejercicio las imágenes de los números 1 y 5.
	
		1. (2.5 puntos) Estimar un modelo de regresión lineal a partir de los datos proporcionados por
			los vectores de características (Intensidad promedio, Simetria) usando tanto el algoritmo
			de la pseudo-inversa como el Gradiente descendente estocástico (SGD). Las etiquetas serán
			{−1, 1}, una por cada vector de cada uno de los números. Pintar las soluciones obtenidas
			junto con los datos usados en el ajuste. Valorar la bondad del resultado usando E in y
			E out (para E out calcular las predicciones usando los datos del fichero de test). ( usar
			Regress_Lin(datos, label) como llamada para la función (opcional)).
			
		2. (3 puntos) En este apartado exploramos como se transforman los errores E in y E out cuando
			aumentamos la complejidad del modelo lineal usado. Ahora hacemos uso de la función
			simula_unif (N, 2, size) que nos devuelve N coordenadas 2D de puntos uniformemente
			muestreados dentro del cuadrado definido por [−size, size] × [−size, size]
			
				a) Generar una muestra de entrenamiento de N = 1000 puntos en el cuadrado
				   X = [−1, 1] × [−1, 1]. Pintar el mapa de puntos 2D. (ver función de ayuda)
				b) Consideremos la función f (x 1 , x 2 ) = sign((x 1 − 0,2) 2 + x 22 − 0,6) que usaremos
				   para asignar una etiqueta a cada punto de la muestra anterior. Introducimos
				   ruido sobre las etiquetas cambiando aleatoriamente el signo de un 10 % de las
				   mismas. Pintar el mapa de etiquetas obtenido.
				c) Usando como vector de características (1, x 1 , x 2 ) ajustar un modelo de regresion
				   lineal al conjunto de datos generado y estimar los pesos w. Estimar el error de
				   ajuste E in usando Gradiente Descendente Estocástico (SGD).
				d) Ejecutar todo el experimento definido por (a)-(c) 1000 veces (generamos 1000
				   muestras diferentes) y
						• Calcular el valor medio de los errores E in de las 1000 muestras.
						• Generar 1000 puntos nuevos por cada iteración y calcular con ellos el valor
						  de E out en dicha iteración. Calcular el valor medio de E out en todas las
						  iteraciones.
				e) Valore que tan bueno considera que es el ajuste con este modelo lineal a la vista
				   de los valores medios obtenidos de E in y E out
			
"""""""""""""""""""""""""""
def ejercicio2():
	print('Ejercicio 2. EJERCICIO SOBRE REGRESION LINEAL\n')
	
	# Primer apartado
	#ejercicio21()
	#wait()
	
	# Segundo apartado
	print('Apartado 2.2)')
	#ejercicio22('lineal')
	#wait()
	ejercicio22('no lineal')
	#wait()


	



###############################################################################
###############################################################################
###############################################################################
###############################################################################



"""
Método de Newton
"""
def newton_method(w, iteraciones, lr, hessfun, gradfun):
	trayectoria = [w]
	for i in range(iteraciones):
		w_new = w - lr*np.dot(np.linalg.inv(hessfun(w[0],w[1])), gradfun(w[0],w[1]))
	
		# Criterio de parada. Comentado para generar la gráfica completa
		"""	
		iguales=True
		for i in range(len(w)):
			if w_new[i]!=w[i]:
				iguales = False
				
				
		if(iguales): 	# Nos quedamos en un punto de estabilidad
			print('son iguales')
			return w, np.array(trayectoria)
		"""
		w = w_new
		trayectoria += [w.copy()]
		
	return w, trayectoria
	
"""""""""""""""""""""""""""
BONUS

Método de Newton Implementar el algoritmo de minimización de Newton
y aplicarlo a la función f (x, y) dada en el ejercicio.3. Desarrolle los mismos experimentos
usando los mismos puntos de inicio.

		• Generar un gráfico de como desciende el valor de la función con las iteraciones.
		• Extraer conclusiones sobre las conductas de los algoritmos comparando la curva de
		  decrecimiento de la función calculada en el apartado anterior y la correspondiente
		  obtenida con gradiente descendente.
		  
"""""""""""""""""""""""""""
def ejercicio3():
	ITERACIONES = 75
	
	puntos_inicio = np.array([np.array([-0.5,-0.5], np.float64), 
						   np.array([1.0,1.0], np.float64), 
						   np.array([2.1,-2.1], np.float64), 
						   np.array([-3.0,3.0], np.float64), 
						   np.array([-1.0,1.0], np.float64)])
	
	for i in range(len(puntos_inicio)):
		weights1, trayectoria0 = newton_method(puntos_inicio[i], ITERACIONES,0.1, hessg, gradg)
		weights1, trayectoria1 = newton_method(puntos_inicio[i], ITERACIONES,0.1, hessf, gradf)
		
		w2, trayectoria2, _, _ = gradient_descent(puntos_inicio[i], 0.1, f, gradf, iterations=ITERACIONES)
		
		y0 = []
		y1 = []
		y2 = []
		for i in range(len(trayectoria1)):
			y0.append(f(trayectoria0[i][0],trayectoria0[i][1]))
			y1.append(f(trayectoria1[i][0],trayectoria1[i][1]))
			if i<len(trayectoria2):
				y2.append(f(trayectoria2[i][0],trayectoria2[i][1]))
			else:
				y2.append(f(trayectoria2[len(trayectoria2)-1][0],trayectoria2[len(trayectoria2)-1][1]))
		fig, ax = plt.subplots()
		
		ax.plot(range(len(trayectoria0)), y0, 'bo--', ms=3, label='Newton Raphson g: PARABOLOIDE')
		ax.plot(range(len(trayectoria1)), y1, 'ro--', ms=3, label='Newton Raphson f')
		ax.plot(range(len(trayectoria1)), y2, 'go--', ms=3, label='Gradiente Descendente')
		
		# Definimos la leyenda en la parte superior derecha
		ax.legend(loc='upper right', shadow=False)
		
		ax.set(title='Newton Raphson vs Gradient Descent')
		ax.set_xlabel('Iteracion')
		ax.set_ylabel('f(w)')
		
		# Mostramos
		plt.show()	 
	      
		# Mostrar gráfica en 3D. Comentado puesto que no da mucha información.
		"""           
		# Rellenamos las trayectorias:
		t2=[]
		for i in range(len(trayectoria1)):
			if i<len(trayectoria2):
				t2.append(trayectoria2[i])
			else:
				t2.append(trayectoria2[len(trayectoria2)-1])
			
		for i in range(2):
			display_surface(g,weights1,trayectoria1, labels=['x','y','f(x,y)'],title='Método de Newton vs Gradiente descendente', x_min=-3.5, x_max=2.5,y_min=-2.5, y_max=4,vista=i)
		"""
			
		
		
		

"""
Ejecución de toda la practica
"""
#ejercicio1()

ejercicio2()

#ejercicio3()

