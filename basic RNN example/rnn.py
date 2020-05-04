from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

X = [[[0],[1],[1],[0],[0],[0],[0]],
     [[0],[0],[0],[2],[2],[0],[0]],
     [[0],[0],[0],[0],[3],[3],[3]],
     [[0],[2],[2],[0],[0],[0],[2]],
     [[0],[0],[3],[3],[0],[0],[3]],
     [[0],[0],[0],[0],[1],[1],[0]]]

X = np.array(X, dtype=np.float32)
max_features = 4 

labels = np.array([1,2,3,2,3,1],dtype=np.int32)
y = to_categorical(labels, num_classes=max_features)

#building model
model = Sequential()
model.add(LSTM(units=128, input_shape=(None,1)))
model.add(Dense(4))
model.add(Activation("sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

hist = model.fit(X, y, epochs=200)
model.save("basic_rnn.h5")

#predicting sequence value:
pred = model.predict(X)
predicted_class = np.argmax(pred, axis=1)

print("True: ",labels)
print("Predicted: ",predicted_class)
