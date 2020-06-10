import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.layers import Masking, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

data_path = 'data/spa.txt'
data = pd.read_table(data_path)

data = data.rename(columns={'Go.':'eng', 'Ve.':'spa'})

X = data['eng'].apply(lambda x:x.lower())
y = data['spa'].apply(lambda x:x.lower())
X = X.apply(lambda x:re.sub("[^a-zA-Z]"," ",x))
y = y.apply(lambda x:re.sub("[^a-zA-Z]"," ",x))
y = y.apply(lambda x:'START_ '+x+' _END')

# creating vocabulary of eng and spa
eng_vocab, spa_vocab = set(), set()
for sent in X:
  for word in sent.split():
    if word not in eng_vocab:
      eng_vocab.add(word)
for sent in y:
 for word in sent.split():
  if word not in spa_vocab:
    spa_vocab.add(word)
engVocab = sorted(list(eng_vocab))
spaVocab = sorted(list(spa_vocab))

#Find maximum sentence length in the source and target data
source_length_list=[] #47
for l in X:
    source_length_list.append(len(l.split(' ')))
max_eng_sent_length= max(source_length_list)

target_length_list=[] #50
for l in y:
    target_length_list.append(len(l.split(' ')))
max_spa_sent_length= max(target_length_list)

# creating a word to index(word2idx) for eng(source) and spa(target)
eng_word2idx = dict([(word, i+1) for i, word in enumerate(engVocab)])
spa_word2idx = dict([(word, i+1) for i, word in enumerate(spaVocab)])

# creating an index to word(idx2word) for eng(source) and spa(target)
eng_idx2word= dict([(i, word) for word, i in  eng_word2idx.items()])
spa_idx2word =dict([(i, word) for word, i in spa_word2idx.items()])

# preparing X, y for training
X, y = shuffle(X, y, random_state=2)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

num_encoder_tokens = len(engVocab)
num_decoder_tokens = len(spaVocab)+1

# creating generator batch function
def generate_batch(X=x_train, y=y_train, batch_size=128):
  while True:
    for i in range(0, len(X), batch_size):
      encoder_input_data = np.zeros(shape=(batch_size, max_eng_sent_length), dtype="float32")
      decoder_input_data = np.zeros(shape=(batch_size, max_spa_sent_length), dtype="float32")
      decoder_output_data = np.zeros(shape=(batch_size, max_spa_sent_length, num_decoder_tokens), dtype="float32")
      for j, (input_text, target_text) in enumerate(zip(X[i:i+batch_size], y[i:i+batch_size])):
        for k, word in enumerate(input_text.split()):
          encoder_input_data[j, k] = eng_word2idx[word]
        for k, word in enumerate(target_text.split()):
          if k < len(target_text.split())-1:
            decoder_input_data[j, k] = spa_word2idx[word]
          if k > 0:
            decoder_output_data[j, k-1, spa_word2idx[word]] = 1
      yield([encoder_input_data, decoder_input_data], decoder_output_data)

train_samples = len(x_train)
val_samples = len(x_test)
batch_size = 128
epochs = 50
latent_dim=256

# defining model --> encoder and decoder
# encoder -->
encoder_inputs = Input(shape=(None,))
enc_emb_layer = Embedding(num_encoder_tokens, latent_dim, mask_zero=True)(encoder_inputs)
enc_lstm_layer = LSTM(units=latent_dim, return_state=True)
encoder_outputs, h_state, c_state = enc_lstm_layer(enc_emb_layer)
encoder_states = [h_state, c_state]

# decoder -->
decoder_inputs = Input(shape=(None, ))
dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero=True)
dec_emb = dec_emb_layer(decoder_inputs)
dec_lstm_layer = LSTM(units=latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = dec_lstm_layer(dec_emb, initial_state=encoder_states)
dec_dense = Dense(units=num_decoder_tokens, activation="softmax")
decoder_outputs = dec_dense(decoder_outputs)

# creating model -->
model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
#print(model.summary())

checkpoint = ModelCheckpoint('my_models/nmt_eng2spa.h5',
							 monitor='val_loss',
							 mode='min',
							 save_best_only=True,
							 verbose=1)

reduceLR = ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=3,verbose=1,min_delta=0.0001)

callbacks = [checkpoint, reduceLR]

hist = model.fit(generate_batch(X=x_train, y=y_train),
                           steps_per_epoch=train_samples//batch_size,
                           epochs=epochs,
                           callbacks=callbacks,
                           verbose=1,
                           validation_data=generate_batch(X=x_test, y=y_test),
                           validation_steps=val_samples//batch_size)

model.save_weights('/content/gdrive/My Drive/nmt_weights_last.h5')
