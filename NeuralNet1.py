# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# https://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/



print("Evaluate ====================================================================")


# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
i = int(len(Y)/2)
print(i)
model.fit(X[0:i,:], Y[0:i], epochs=150, batch_size=1, verbose=2)
# evaluate the model
scores = model.evaluate(X[0:i,:], Y[0:i])
print("\nTotal%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


print("Predict =====================================================================")


# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
correct = 0
for x in range(i,len(Y)):
	if Y[x] == rounded[x]: correct += 1
print("\naccuracy: %.2f%%" % (100*correct/i))
