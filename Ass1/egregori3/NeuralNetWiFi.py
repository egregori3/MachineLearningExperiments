print("Evaluate ====================================================================")


# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy


# load dataset
dataset = numpy.loadtxt("wifi_localization.txt", delimiter="\t")
# split into input (X) and output (Y) variables
X = dataset[:,0:7]
Y = dataset[:,7]

# create model
model = Sequential()
model.add(Dense(7, input_dim=7, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model

model.fit(X, Y/4.0, epochs=150, batch_size=10, verbose=2)
# evaluate the model
scores = model.evaluate(X, Y/4.0)
print("\nTotal%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


print("Predict =====================================================================")


# calculate predictions
predictions = model.predict(X)
print(predictions)

# round predictions
#rounded = [round(x[0]) for x in predictions]
#correct = 0
#for x in range(0,len(Y)):
#    if Y[x] == rounded[x]: correct += 1
#print("\naccuracy: %.2f%%" % (100*correct/i))
