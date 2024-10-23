# Importamos las bibliotecas necesarias
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np  # álgebra lineal
import pandas as pd  # procesamiento de datos, lectura de archivos CSV
import math
import datetime
import platform
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# Cargamos los conjuntos de datos de entrenamiento y prueba
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Separamos las características (X) y las etiquetas (y) del conjunto de entrenamiento
X = train.iloc[:, 1:785]
y = train.iloc[:, 0]
X_test = test.iloc[:, 0:784]

# Normalizamos los datos dividiendo por 255
X_tsn = X / 255

# Realizamos un t-SNE para visualizar los datos (comentado)
# tsne = TSNE()
# tsne_res = tsne.fit_transform(X_tsn)
# plt.figure(figsize=(14, 12))
# plt.scatter(tsne_res[:, 0], tsne_res[:, 1], c=y, s=2)
# plt.xticks([])
# plt.yticks([])
# plt.colorbar()
# plt.show()

# Dividimos el conjunto de entrenamiento en entrenamiento y validación
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1212)

# Reformateamos los datos para que tengan la forma adecuada para una red neuronal convolucional
x_train_re = X_train.to_numpy().reshape(33600, 28, 28)
y_train_re = y_train.values
x_validation_re = X_validation.to_numpy().reshape(8400, 28, 28)
y_validation_re = y_validation.values
x_test_re = test.to_numpy().reshape(28000, 28, 28)

# Definimos las dimensiones de las imágenes
(_, IMAGE_WIDTH, IMAGE_HEIGHT) = x_train_re.shape
IMAGE_CHANNELS = 1
print('IMAGE_WIDTH:', IMAGE_WIDTH)
print('IMAGE_HEIGHT:', IMAGE_HEIGHT)
print('IMAGE_CHANNELS:', IMAGE_CHANNELS)

# Visualizamos una de las imágenes de entrenamiento
plt.imshow(x_train_re[0], cmap=plt.cm.binary)
plt.show()

# Imprimimos algunos ejemplos de entrenamiento para ver cómo están escritas las cifras
numbers_to_display = 25
num_cells = math.ceil(math.sqrt(numbers_to_display))
plt.figure(figsize=(10, 10))
for i in range(numbers_to_display):
    plt.subplot(num_cells, num_cells, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train_re[i], cmap=plt.cm.binary)
    plt.xlabel(y_train_re[i])
plt.show()
