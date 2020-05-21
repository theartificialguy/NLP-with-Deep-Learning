import tensorflow as tf 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np 
import pandas as pd 

test_title = ["spark an inner revolution"]

labels = ["Reliable", "Unreliable"]

vocab_size = 5000
paddingLen = 20
oneHotRep = [one_hot(words, vocab_size) for words in test_title]
padded = pad_sequences(oneHotRep, truncating="post", padding="post", maxlen=paddingLen)

x = np.array(padded)

model = load_model("fake_news.h5")

pred = model.predict_classes(x)[0]
print(labels[int(pred)])