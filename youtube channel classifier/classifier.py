import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Activation
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import nltk

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.config.experimental.set_virtual_device_configuration(gpus[0], 
					[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

df = pd.read_csv("youtube_data.csv")
df = df.dropna()
classes = ['travel', 'science and technology', 'food', 'manufacturing', 'history', 'art and music']

strToInt = {'travel': 0,
			'science and technology': 1,
			'food': 2,
			'manufacturing': 3,
			'history': 4,
			'art and music': 5}

X = df.drop(labels=["Category", "Video Id"], axis=1)
y = df["Category"]
X = X.reset_index()
y = y.reset_index()
y = y["Category"]

for i in range(len(y)):
	if y[i] in strToInt:
		y[i] = strToInt[y[i]]

y = to_categorical(y, num_classes=len(classes))

training_text = []

for i in range(len(X)):
	training_text.append(X["Title"][i]+" "+X["Description"][i])

stop_words = stopwords.words("english")

lemmatizer = WordNetLemmatizer()

xtrain = []
for sent in training_text:
	sent = re.sub("[^a-zA-Z]", " ", sent)
	sent = sent.lower().split()
	sent = [lemmatizer.lemmatize(word) for word in sent if word not in set(stop_words)]
	sent = " ".join(sent)
	xtrain.append(sent)

paddingLen = 70
maxFeatureLen = 20
vocabSize = 10000

oneHotEncoded = [one_hot(word, vocabSize) for word in xtrain]

oneHotEncoded = pad_sequences(oneHotEncoded, 
						truncating="post", padding="post",
						maxlen=paddingLen)

oneHotEncoded = np.array(oneHotEncoded)
X, y = shuffle(oneHotEncoded, y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#creating model

model = Sequential()
model.add(Embedding(vocabSize, maxFeatureLen, input_length=paddingLen))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(units=128, return_sequences=True, recurrent_dropout=0.2))
model.add(LSTM(units=128))
model.add(Dense(units=len(classes)))
model.add(Activation("softmax"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print(model.summary())

hist = model.fit(X_train, y_train, batch_size=24, epochs=10,
	 			validation_data=(X_test, y_test), verbose=1)

train_loss = hist.history["loss"]
train_accuracy = hist.history["accuracy"]
val_loss = hist.history["val_loss"]
val_accuracy = hist.history["val_accuracy"]

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,10),train_loss,label="Train_Loss")
plt.plot(np.arange(0,10),train_accuracy,label="Train_Accuracy")
plt.plot(np.arange(0,10),val_accuracy,label="Validation_Accuracy")
plt.plot(np.arange(0,10),val_loss,label="Validation_Loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch#")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("output.png")
plt.show()
plt.close()
