import numpy
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

seed = 7
numpy.random.seed(seed)

dataset = loadtxt('CHD_C.csv', delimiter=',')
X = dataset[:,0:7]
y = dataset[:, 7]

model = Sequential()
model.add(Dense(12, input_dim=7, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=150, batch_size=10)
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))