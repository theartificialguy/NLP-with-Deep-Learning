import nltk
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.layers import LSTM, Activation, SpatialDropout1D
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import re

train_dir = "data/train.csv"
df = pd.read_csv(train_dir)

df = df.dropna()
df = df.reset_index()
X = df.drop(labels=['label', 'id'], axis=1)
y = df['label']

xdata = X.copy()
xdata.reset_index(inplace=True)

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

xtitle = []
for i in range(len(xdata)):
	sent = re.sub('[^a-zA-Z]', ' ', xdata['title'][i])
	sent = sent.lower().split()
	sent = [lemmatizer.lemmatize(word) for word in sent if word not in set(stop_words)]
	sent = " ".join(sent)
	xtitle.append(sent)

vocab_size = 5000
embedding_feature_len = 30
max_sent_len = 20
batch_size = 32
epochs = 10

one_hot_representation = [one_hot(words, vocab_size) for words in xtitle]
padded_sequences = pad_sequences(one_hot_representation, truncating="post",
				padding="post", maxlen=max_sent_len)

X = np.array(padded_sequences)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(vocab_size, embedding_feature_len, input_length=max_sent_len))
model.add(SpatialDropout1D(rate=0.2))
model.add(LSTM(units=128))
model.add(Dense(units=1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())

hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

y_pred = model.predict_classes(x_test)
