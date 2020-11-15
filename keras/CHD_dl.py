import numpy
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import preprocessing

seed = 7
numpy.random.seed(seed)

dataset = loadtxt('CHD_C.csv', delimiter=',')
X = dataset[:, 0:7]
X_z = scale(X)
y = dataset[:, 8]

model = Sequential()
model.add(Dense(12, input_dim=7, activation='relu'))
model.add(Dense(10, input_dim=7, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(6, activation='relu'))
# model.add(Dense(1, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
result = []
epochs_test = [150, 300, 600, 800, 1000]
# epochs_test = [150, 300]
for i in epochs_test:
    History = model.fit(X, y, epochs=i, batch_size=15)
    _, accuracy = model.evaluate(X, y)
    result.append(accuracy)
    print('Accuracy: %.2f' % (accuracy * 100))
    plt.clf()
    plt.plot(History.history['accuracy'])
    plt.plot(History.history['loss'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['accu', 'loss'], loc='upper left')
    plt.savefig('X' + str(i), dpi=300)

result_z = []
for i in epochs_test:
    History = model.fit(X_z, y, epochs=i, batch_size=15)
    _, accuracy = model.evaluate(X_z, y)
    result_z.append(accuracy)
    print('Accuracy: %.2f' % (accuracy * 100))
    plt.clf()
    plt.plot(History.history['accuracy'])
    plt.plot(History.history['loss'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['accu', 'loss'], loc='upper left')
    plt.savefig('X_z' + str(i), dpi=300)




# time = [1, 2, 3, 4, 5]
# result2 = []
# for i in time:
#     if i <= 5:
#         model.fit(X, y, epochs=150, batch_size=10)
#         _, accuracy = model.evaluate(X, y)
#         result2.append(accuracy)
#         print('Accuracy: %.2f' % (accuracy * 100))
#     else:
#         break

