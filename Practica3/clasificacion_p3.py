#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTOR: Gallego López, Pedro

ÁMBITO: APRENDIZAJE AUTOMÁTICO

El objetivo es modelar un problema de clasificación multiclase de AA concreto.
Se nos da una base de datos relacionada con la calidad de un motor. Según los
defectos pertenecerá a una de las clases.

Enlace:
https://archive.ics.uci.edu/ml/datasets/Dataset+for+Sensorless+Drive+Diagnosis

"""

#######################################################
#######################################################
## 	 	 	 	 	 	 	 	 	 	 	 	 	 ##
##	 	 	  PROBLEMA DE CLASIFICACIÓN 	 	 	 ##
## 	 	 	 	 	 	 	 	 	 	 	 	 	 ##
#######################################################
#######################################################

# Vectores
import numpy as np

# Visionado de gráficas
import matplotlib.pyplot as plt
import matplotlib as cm

# Scikit Learn: herramientas de Machine Learning
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import plot_confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Para matriz de correlación
import seaborn as sns

# Para trabajar con ficheros
import os

# Para medición de tiempos
from timeit import default_timer


# Para leer los datos del .csv
import pandas as pd



# VARIABLES GLOBALES
SEED = 1
MOSTRAR_GRAFICOS_TSNE = False 	# LLega a tardar media hora. En la memoria vienen las gráficas
MOSTRAR_CURVA_DE_APRENDIZAJE = False # La ejecución tarda 3 horas (Medido en GoogleCollab)


def wait():
	"""
	Función para pausar la ejecución
	"""
	input("\n--- Pulsar tecla para continuar ---\n")



#######################################################
#	 	 	 	LECTURA DE DATOS 	 	 	 	 	  #
#######################################################
def read_data(PATH, filename, test_size=0.2):
	"""
	Description
	-----------
	Lee los datos de PATH+filename como un csv en donde la última columna
	es la característica a predecir
	
	Parameters
	----------
	PATH : string
		Path del archivo a leer.
	filename : string
		Nombre del archivo con formato .csv (incluido en el string).
	test_size : float in [0,1]
		Porcentaje de datos reservados para test. The default is 0.2.

	Returns
	-------
	X_train : numpy.ndarray
		Conjunto de datos de entrenamiento.
	y_train : numpy.ndarray
		Conjunto de etiquetas de entrenamiento.
	X_test : numpy.ndarray
		Conjunto de datos de test.
	y_test : numpy.ndarray
		Conjunto de etiquetas de entrenamiento.

	"""
	
	# Leemos: en este caso el separador es un espacio y no tienen encabezado las columnas
	data = pd.read_csv(PATH+filename, sep=" ", header=None)
			
	# Dividimos los datos en características y etiquetas
	X = data.iloc[:,:-1].values 	# .values nos lo convierte en un ndarray
	y = data.iloc[:,-1].values
	
	
	# Creamos los conjuntos de train y test
	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state = SEED)
	
	
	return X_train, y_train, X_test, y_test


		
		
#######################################################
#	 	 	   VISUALIZACIÓN DE DATOS 	 	 	 	  #
#######################################################
def plt_correlation_matrix(nparray, title=''):
	"""
	Description
	-----------
	Muestra por pantalla la matriz de correlación

	Parameters
	----------
	nparray : numpy array
		Array de datos.
	title : string, optional
		Título del plot. The default is ''.

	Returns
	-------
	None.

	"""
	sns.heatmap(pd.DataFrame(nparray).corr().abs())
	
	plt.title(title)
	
	plt.show()
	
def plt_histogram(classes=np.array([]), errores=np.array([])):
	"""
	Muestra un histograma donde cada barra representa el número de instancias
	en una determinada clase (rango de valores)

	Parameters
	----------
	classes : ndarray 
		Vector de características (etiquetas).
		
	errores : ndarray
		Vector que lleva el nº de errores en la pos i al predecir la clase i+1

	Returns
	-------
	None.

	"""
	if len(classes) > 0:
		# Definimos el color
		cmap = cm.cm.get_cmap('tab20')

		x_axis = classes
		
		# Definimos los intervalos de las barras
		intervalos = np.arange(min(classes)-0.5,max(classes)+1.5)
	elif len(errores) > 0:
		# Definimos el color
		cmap = cm.cm.get_cmap('autumn')		
		x_axis = []
	
		for i in range(len(errores)):
			for j in range(errores[i]):
				x_axis.append(i+1)
				
		# Definimos los intervalos de las barras
		intervalos = np.arange(0.5,len(errores)+1.5)		
	else:
		x_axis = -1
		intervalos = []
		print("\nError en plt_histogram. Parámetros de llamada incorrectos.")
	

	# Creamos el hisotgrama
	n, bins, patches = plt.hist(x=x_axis, bins=intervalos, color='#F2AB6D', rwidth=0.85)
	
	
	
	
	# Escribimos la cantidad encima de cada barra
	for i in range(len(patches)):
		if len(classes) > 0:
			anotacion = len(classes[classes==i+1])
		elif len(errores) > 0:
			anotacion = errores[i]
			
			
		height = patches[i].get_height()
		plt.annotate(str(anotacion), xy=(patches[i].get_x()+patches[i].get_width()/2, height), xytext=(0,5), textcoords='offset points', ha='center', va='bottom', fontsize=8.5)
	
	# Escalamos
	bin_centers = 0.5 * (bins[:-1] + bins[1:])

	# scale values to interval [0,1]
	col = bin_centers - min(bin_centers)
	col /= max(col)
	
	# Coloreamos
	for c, p in zip(col, patches):
		plt.setp(p, 'facecolor', cmap(c))
	
	plt.xlabel('Clase')
	plt.xticks(range(1,12))
	if len(classes) > 0:
		plt.title('Histograma de número de instancias por clases')
		plt.ylabel('número de instancias')
		plt.ylim([0,7000])
	elif len(errores) > 0:
		plt.title('Histograma de número de errores por clases')
		plt.ylabel('número de errores')
		plt.ylim([0,max(errores)*1.5])
	else:
		plt.ylabel('######-error')
		print("\nError en plt_histogram. Parámetros de llamada incorrectos.")	

	plt.yscale('linear')
	
			
	plt.show()


def plt_var_explicada_pca(X):
	"""
	Muestra un plot de la varianza explicada por número de componentes principales

	Parameters
	----------
	X : data set numpy array

	Returns
	-------
	plot evolucion.

	"""
	# Simulamos las transformaciones 
	var = VarianceThreshold()
	X_spectro = var.fit_transform(X)
	std = StandardScaler()
	X_spectro = std.fit_transform(X_spectro)
	pca = PCA()
	X_spectro = pca.fit_transform(X_spectro)
	
	# Creamos el plot
	fig, ax = plt.subplots()
	
	# Calculamos la acumulación
	varianza_explicada = [np.sum(pca.explained_variance_ratio_[:i+1])*100 for i in range(len(pca.explained_variance_ratio_))]
	
	ax.plot(np.arange(1,pca.n_components_ + 1), varianza_explicada, 'ko-', linewidth=0.5, ms=2, label='Componentes')
			
	ax.set(title='Varianza explicada acumulada según los componentes de PCA')
	ax.set_xlabel('Número de Componentes')
	ax.set_ylabel('% Varianza explicada')
	
	# Mostramos		
	plt.show()
	
	
	
def plt_tsne(X, etiquetas):
	"""
	Hace un plot en 2D y 3D tras la disminución de variables propuesta por la
	técnica de t-SNE

	Parameters
	----------
	X : matriz numpy
		datos.
	etiquetas : int
		etiquetas de X.

	Returns
	-------
	plot t-SNE 2D y 3D.

	"""
	
	
	tsne_filename = 'clasif_tsne2D_without_validation.txt'
	if os.path.exists(tsne_filename):
		X_embedded2D = np.loadtxt(tsne_filename)
	else:
		X_embedded2D = TSNE (n_components=2).fit_transform(X)	
		np.savetxt(tsne_filename,X_embedded2D)
		
	
	
	# Gráfica de 3 dimensiones
	tsne_filename = 'clasif_tsne3D_without_validation.txt'
	if os.path.exists(tsne_filename):
		X_embedded3D = np.loadtxt(tsne_filename)
	else:
		X_embedded3D = TSNE (n_components=3).fit_transform(X)	
		np.savetxt(tsne_filename,X_embedded3D)

	# Definimos las etiquetas de las clases para hacer los plots
	etiquetas_clases=[str(i+1) for i in range(max(etiquetas)+1)]



	# Creamos el plot y mostramos
	fig, ax = plt.subplots()
	scatter=ax.scatter(X_embedded2D[:,0], X_embedded2D[:,1], c=etiquetas, cmap='tab20', s=2)
	ax.set(title='T-Dsitributed Stochastic Neighbor Embedding 2D')
	ax.legend(handles = scatter.legend_elements()[0], title = "Classes", labels = etiquetas_clases, loc='lower left')
	plt.show()
	

	# Creamos el plot y mostramos
	ax = plt.figure(figsize=(15,9)).gca(projection='3d')
	scatter=ax.scatter(xs=X_embedded3D[:,0], ys=X_embedded3D[:,1], zs=X_embedded3D[:,2], c=etiquetas, cmap='tab20', s=5)	
	ax.set(title='T-Dsitributed Stochastic Neighbor Embedding 3D')
	ax.legend(handles = scatter.legend_elements()[0], title = "Classes", labels = etiquetas_clases, loc='lower left')
	plt.show()





def plt_main_components_PCA(X,etiquetas):
	"""
	Muestra un plot de las dos principales componentes

	Parameters
	----------
	X : data set numpy array
	etiquetas : etiquetas

	Returns
	-------
	plot.

	"""
	# Discretizamos
	etiquetas2=etiquetas

	
	# Simulamos las transformaciones 
	var = VarianceThreshold()
	Componentes = var.fit_transform(X)
	std = StandardScaler()
	Componentes = std.fit_transform(Componentes)
	pca = PCA(3)
	Componentes = pca.fit_transform(Componentes)
	
	# Definimos las etiquetas de las clases para hacer los plots
	etiquetas_clases=[str(i) for i in range(max(np.unique(etiquetas.astype(int)))+1)]

	# Creamos el plot
	fig, ax = plt.subplots()
		
	scatter=ax.scatter(Componentes[:,0], Componentes[:,1], c=etiquetas2.astype(int), cmap='tab20', s=2)
			
	ax.set(title='Principales componentes de PCA')
	ax.set_xlabel('Componente nº1')
	ax.set_ylabel('Componente nº2')
	plt.xlim((-10,10))
	plt.ylim((-10,10))
	ax.legend(handles = scatter.legend_elements()[0], title = "Classes", labels = etiquetas_clases, loc='lower left')



	# Mostramos		
	plt.show()
	
	# Creamos el plot
	fig, ax = plt.subplots()
		
	scatter=ax.scatter(Componentes[:,0], Componentes[:,2], c=etiquetas2.astype(int), cmap='tab20', s=2)
			
	ax.set(title='Principales componentes de PCA')
	ax.set_xlabel('Componente nº1')
	ax.set_ylabel('Componente nº3')
	plt.xlim((-10,10))
	plt.ylim((-10,10))
	ax.legend(handles = scatter.legend_elements()[0], title = "Classes", labels = etiquetas_clases, loc='lower left')
	
	
	
	# Mostramos		
	plt.show()
	
	# Creamos el plot
	fig, ax = plt.subplots()
		
	scatter=ax.scatter(Componentes[:,1], Componentes[:,2], c=etiquetas2.astype(int), cmap='tab20', s=2)
			
	ax.set(title='Principales componentes de PCA')
	ax.set_xlabel('Componente nº2')
	ax.set_ylabel('Componente nº3')
	plt.xlim((-10,10))
	plt.ylim((-10,10))
	ax.legend(handles = scatter.legend_elements()[0], title = "Classes", labels = etiquetas_clases, loc='lower left')

	
	
	
	# Mostramos		
	plt.show()
	





def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
	Fuente:
		https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
	
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt



def show_results(X_train, y_train, X_test, y_test, model):
	"""
	Genera dos plots:
		1.- Gráfica de residuos
		2.- Gráfica valores predecidos vs valores observados

	Parameters
	----------
	X_train : Dataset de entrenamiento
	y_train : Etiquetas de entrenamiento
	X_test : Dataset de test
	y_test : Etiquetas de test
	model : Modelo predictor

	Returns
	-------
	Gráficas (1) y (2).

	"""	
	print("\n ----------- RESULTADOS DE EJECUCIÓN: TRAIN, TEST ----------- ")	

	for name, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
		y_pred = model.predict(X)
		print(name)
		print("\tAccuracy en {}: {:.3f}".format(name, model.score(X,y)))
		print("\tF1-Score en {}: {:.3f}".format(name, f1_score(y,y_pred,average='macro'))) # average='macro' para multilabel
		
		# Creamos el plot de resiudos
		fig, ax = plt.subplots()
			
		errores_por_clase = []
		for i in range(11):
			errores=0
			
			for j in np.where(y==i+1)[0]:
				if y_pred[j]!=i+1:
					errores=errores+1
					
			errores_por_clase.append(errores)
		
		
		plt_histogram(errores=errores_por_clase)
		
		k=y!=y_pred
		print("\tNúmero de fallos: "+str(len(k[k==1])))
		print("\tTamaño de "+name+": "+str(len(X)))
		
		
		print('\n\tMostrando la gráfica de residuos y la matriz de confusión de '+name+'...')
		
		ax.plot(np.unique(y), errores_por_clase, 'bo-', linewidth=0, ms=2, label='y-y_pred')
		ax.plot([min(y), max(y)],[0,0], 'ro-', linewidth=2, ms=0.5, label='error 0')
				
		ax.set(title='Gráfica de residuos en '+str(name))
		ax.set_xlabel('Valores observados')
		ax.set_ylabel('Residuo')

		ax.legend(loc='lower right', shadow=False)
		# Mostramos		
		plt.show()
		
		
		# Matriz de confusión
		fig, ax = plt.subplots(1, 1, figsize = (8, 6))
		disp = plot_confusion_matrix(model, X, y, cmap = cm.cm.Blues, values_format = 'd', ax = ax)
		disp.ax_.set_title("Matriz de confusión")
		disp.ax_.set_xlabel("Etiqueta predicha")
		disp.ax_.set_ylabel("Etiqueta real")
		plt.show()
		
		
		
	if MOSTRAR_CURVA_DE_APRENDIZAJE:
		fig, axes = plt.subplots(3, 1, figsize=(10, 15))
		plot_learning_curve(model, 'Learning Curves', X_train, y_train, axes=axes, ylim=(0.83, 1.01), cv = 5)
		plt.show()


	
	





#######################################################
#	 	 	  	   FUNCIONES PRINCIPALES 	  	   	  #
#######################################################
def preprocesamiento_list_for_pipeline():
	"""
	Crea dos listas de pipeline

	Returns
	-------
	[steps1, steps2]

	"""
	# Juntamos los preprocesados en un pipeline: VarianceThreshold, StandardScaler y PCA
	steps1 = [("var",VarianceThreshold()),("Standardize", StandardScaler()),("PCA", PCA(0.95))]
	
	# Juntamos los preprocesados en un pipeline: Transformaciones de segundo orden, variance threshold y StandardScaler
	steps2 = [("Second Order Pol", PolynomialFeatures(2)),("var2",VarianceThreshold(0.1)),("Standardize2", StandardScaler())]
	
	return steps1, steps2


def get_train_test_preprocessed(X_train, y_train, X_test, y_test):
	"""
	Descripción
	-----------
	Preprocesa los conjuntos de train y test con los pipelines de 
	preprocesamiento_pipeline(). 

	Parameters
	----------
	X_train : dataset entrenamiento
	y_train : etiquetas entrenamiento
	X_test : dataset test
	y_test : etiquetas test
	
	Returns
	-------
	X_train_mod : train preprocesado
	X_test_mod : test preprocesado

	"""
	# Obtenemos los pipelines
	steps1, steps2 = preprocesamiento_list_for_pipeline()
	pipeline1 = Pipeline(steps1)
	pipeline2 = Pipeline(steps2)

	# Transformamos los datos
	X_train_mod = pipeline1.fit_transform(X_train, y_train)
	X_test_mod = pipeline1.transform(X_test)
	
	#dibujamos el histograma
	plt_histogram(classes=y_train)	
	plt_histogram(classes=y_test)	
	
		
	
	# Mostramos la evolución de componentes según el nivel de porcentaje de 
	# varianza que queremos explicar
	plt_main_components_PCA(X_train, y_train)
	plt_var_explicada_pca(X_train)


	# Si queremos ver los gráficos TSNE poner la variable global
	# MOSTRAR_GRAFICOS_TSNE = True
	if MOSTRAR_GRAFICOS_TSNE:
		plt_tsne(X_train_mod, y_train)		
		
	
	# Mostramos la matriz de correlación antes y después del PCA
	plt_correlation_matrix(X_train, title='Matriz de correlación: datos originales')
	plt_correlation_matrix(X_train_mod, title='Matriz de correlación: después de aplicar PCA')

	
	# Transformamos los datos
	X_train_mod = pipeline2.fit_transform(X_train_mod, y_train)
	X_test_mod = pipeline2.transform(X_test_mod)

	# Mostramos la matriz de correlación después de la transformación de segundo orden
	plt_correlation_matrix(X_train_mod, title='Matriz de correlación: después de aplicar PCA\n y transformación de 2orden')

		
	return X_train_mod, X_test_mod


def fit_clf(X_train, y_train):
	"""
	Nos escoge y entrena un modelo a partir de los datos de entrenamiento 
	pasados por parámetro

	Parameters
	----------
	X_train : Dataset (numpy array) de valores reales
	y_train : Etiquetas (valores reales) de X_train

	Returns
	-------
	best_model : modelo entrenado elegido a través de corss_validation

	"""
	# Fijamos el número de iteraciones
	maxiter = 1000
	
	# Definimos el pipeline de nuestros datos. Contiene el preprocesado y el
	# modelo de clasificación
	steps1, steps2 = preprocesamiento_list_for_pipeline()
	pipe = Pipeline(steps1+steps2+[("clf", LogisticRegression())])
	
	
	# Definimos el espacio de modelos
	param_grid = [
		{"clf": [LogisticRegression(multi_class = 'ovr',
                                    penalty = 'l2',
                                    max_iter = maxiter)],
         "clf__C": np.logspace(-1,4,3, base=2)},
	    {"clf": [RidgeClassifier(random_state = SEED,
                                 max_iter = maxiter)],
         "clf__alpha": np.logspace(-1,4,3, base=2)},
        {"clf": [Perceptron(penalty = 'l2',
                            random_state = SEED,
                            max_iter = maxiter)],
         "clf__alpha": np.logspace(-1,4,3, base=2)}]
	
	
	# Ejecutamos Cross-Validation para encontrar el mejor modelo
	print("Ejecutando Cross-Validation para la selección del mejor modelo lineal (Tiempo estimado: 15 minutos)...")
	start = default_timer()
	
	
	best_model = GridSearchCV(pipe, 
							  param_grid, 
							  scoring='accuracy',	
							  cv=5,
							  n_jobs=-1)

	best_model.fit(X_train, y_train)
	
	end = default_timer()
	elapsed = end - start
	print("Hecho. ({:.3f}s)\n".format(elapsed))
	
	# Mostramos los resultados
	print("\n ------------- MEJOR MODELO DE CLASIFICACIÓN ------------- ")	
	print("Parámetros:{}".format(best_model.best_params_))
	print("Número de variables usadas:  {}".format(best_model.best_estimator_['clf'].coef_.shape[1]))
	print("Accuracy en CV: {:.3f}".format(best_model.best_score_))
	
	return best_model
		



def	ejecucion_clasificacion():
	# Leemos los datos y separamos, sin procesarlos, los conjuntos train y test
	X_train, y_train, X_test, y_test = read_data('./data/clasificacion/','Sensorless_drive_diagnosis.txt')
	
	# Visualizamos los datos para entender mejor el problema
	print("Visualizamos los datos para entender mejor el problema...")
	X_train_mod, X_test_mod = get_train_test_preprocessed(X_train, y_train, X_test, y_test)
	wait()
	
	# Ejecutamos la elección del modelo
	best_model = fit_clf(X_train, y_train)
	
	# Mostramos los resultados de train y test
	show_results(X_train, y_train, X_test, y_test, best_model)
	
	
	
##############################################################################
##############################################################################
##
##	 	 	 	 	 EJECUCIÓN
##
##############################################################################
##############################################################################



ejecucion_clasificacion()






















































































































































































































