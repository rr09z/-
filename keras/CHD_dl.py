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
result = []
epochs_test = [150, 300, 600, 800, 1000]
for i in epochs_test:
    model.fit(X, y, epochs=i, batch_size=10)
    _, accuracy = model.evaluate(X, y)
    result.append(accuracy)
    print('Accuracy: %.2f' % (accuracy*100))#可以说屌用没有，输入一万次怕是都这个吊样

time = [1, 2, 3, 4, 5]
result2 = []
for i in time:
    if i <= 5:
        model.fit(X, y, epochs=150, batch_size=10)
        _, accuracy = model.evaluate(X, y)
        result2.append(accuracy)
        print('Accuracy: %.2f' % (accuracy * 100))
    else:
        break
