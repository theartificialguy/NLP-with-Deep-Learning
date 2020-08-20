from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.layers import Conv1D, Dropout, SpatialDropout1D
from tensorflow.keras.layers import MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import nltk
import re

df = pd.read_csv('/content/gdrive/My Drive/twitt30k.csv')

X = df['twitts'].tolist()
y = df['sentiment']

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

x_train = []
for sent in X:
    sent = re.sub("[^a-zA-Z]", " ", sent)
    sent = sent.lower().split()
    sent = [lemmatizer.lemmatize(word) for word in sent if word not in set(stop_words)]
    sent = " ".join(sent)
    x_train.append(sent)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train)

vocab_size = len(tokenizer.word_index) + 1

encoded_texts = tokenizer.texts_to_sequences(x_train)

testing_list = []
for l in x_train:
    testing_list.append(len(l.split(' ')))
max_sentence_length = max(testing_list)

padded_X = pad_sequences(encoded_texts, maxlen=max_sentence_length, padding='post', truncating='post')

X_train, X_test, y_train, y_test = train_test_split(padded_X, y, test_size=0.15, random_state=42)

embedding_features = 300

model = Sequential()
model.add(Embedding(vocab_size, embedding_features, input_length=max_sentence_length))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(64, recurrent_activation='relu', recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(32, recurrent_activation='relu', recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test))

model.save('/content/gdrive/My Drive/twitter.h5')

#Testing
from tensorflow.keras.models import load_model

lstm_model = load_model('/content/gdrive/My Drive/twitter.h5')

sentiments = ['negative', 'positive']

def text_preprocess(text):
    encoded = tokenizer.texts_to_sequences(text)
    padded = pad_sequences(encoded, maxlen=max_sentence_length, padding='post', truncating='post')
    return padded

tweet1 = ['i want to kill myself'] #neg sentiment 0
tweet2 = ['Thank you very much'] #pos sentiment 1
txt1 = text_preprocess(tweet1)
txt2 = text_preprocess(tweet2)
output1 = sentiments[lstm_model.predict_classes(txt1)[0][0]]
output2 = sentiments[lstm_model.predict_classes(txt2)[0][0]]

print(output1, output2)
