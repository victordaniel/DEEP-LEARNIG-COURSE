from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

# Load dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print("hi-hello")


# Preprocess data
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
classes = 10
Y_train = to_categorical(Y_train, classes)
Y_test = to_categorical(Y_test, classes)

# Set network parameters
input_size = 784
batch_size = 100
hidden_neurons = 100
epochs = 20

# Build model
# model = Sequential([
#     Dense(hidden_neurons, input_dim=input_size),
#     Activation('sigmoid'),
#     Dense(classes),
#     Activation('softmax')
# ])

model= Sequential()
model.add(Dense(10,input_dim=784,activation='sigmoid'))


# Compile model
model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.0001), metrics=['accuracy'])


check= ModelCheckpoint("best_model.keras",monitor='val_accuracy',save_best_only=True,mode='max')


# Train model
result=model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test,Y_test),verbose=1,callbacks=[check])

# Evaluate model
score = model.evaluate(X_test, Y_test, verbose=1)
print(score[0])
print('Test accuracy:', score[1])

# Visualize weights
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy


plt.plot(result.history['val_loss'],label='validation_loss')
plt.plot(result.history['loss'],label='train_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()   


# // 
# weights = model.layers[0].get_weights()
# fig = plt.figure()
# w = weights[0].T

# for neuron in range(hidden_neurons):
#     ax = fig.add_subplot(10, 10, neuron + 1)
#     ax.axis("off")
#     ax.imshow(numpy.reshape(w[neuron], (28, 28)), cmap=cm.Greys_r)

# plt.savefig("neuron_images.png", dpi=300)
# //
plt.show()
