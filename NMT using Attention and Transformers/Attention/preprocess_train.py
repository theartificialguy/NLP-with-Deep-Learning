import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Bidirectional
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from utils.encoder import Encoder
from utils.attention import Attention
from utils.decoder import Decoder
import numpy as np
import pandas as pd
import string
import time
import re
import os

punc = set(string.punctuation)

pd.pandas.set_option('display.max_columns', None)

df = pd.read_csv('/content/gdrive/My Drive/Hindi_English_Truncated_Corpus.csv')

nan_values = df[df.isnull().any(axis=1)].index #getting rid of any nan values

df = df.drop(nan_values)

#lower casing
df['english_sentence'] = df['english_sentence'].apply(lambda x:x.lower())
df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x:x.lower())
#removing digits
df['english_sentence'] = df['english_sentence'].apply(lambda x: re.sub("[^a-zA-Z]", " ", x))
df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))
df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: re.sub("[0-9]", "", x))
#removing space
df['english_sentence'] = df['english_sentence'].apply(lambda x: x.strip())
df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: x.strip())
df['english_sentence'] = df['english_sentence'].apply(lambda x: re.sub(" +", " ", x))
df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: re.sub(" +", " ", x))
#removing punctuation
df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: ''.join(char for char in x if char not in punc))
#adding start and end sequence
df['english_sentence'] = df['english_sentence'].apply(lambda x: 'start_ ' + x + ' _end')
df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: 'start_ ' + x + ' _end')

english_data = df['english_sentence'].to_list()
hindi_data = df['hindi_sentence'].to_list()

#max length of sentences in each data
eng, hin = [], []
for l in english_data:
    eng.append(len(l.split(' ')))

for l in hindi_data:
    hin.append(len(l.split(' ')))

max_eng_sentence_length = 50 #max(eng)
max_hindi_sentence_length = 60 #max(hin)

english_tokenizer = Tokenizer(filters='')
hindi_tokenizer = Tokenizer(filters='')

english_tokenizer.fit_on_texts(english_data)
hindi_tokenizer.fit_on_texts(hindi_data)

#getting vocab sizes
english_vocab = len(english_tokenizer.word_index)+1
hindi_vocab = len(hindi_tokenizer.word_index)+1

#converting to sequences
english_encoded_text = english_tokenizer.texts_to_sequences(english_data)
hindi_encoded_text = hindi_tokenizer.texts_to_sequences(hindi_data)

#padding
english_padded_text = pad_sequences(english_encoded_text, maxlen=max_eng_sentence_length, padding='post')
hindi_padded_text = pad_sequences(hindi_encoded_text, maxlen=max_hindi_sentence_length, padding='post')

x_train, x_test, y_train, y_test = train_test_split(english_padded_text, hindi_padded_text, test_size=0.2)

#creating data in batch to utilize memory efficiently
BATCH_SIZE = 64

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(BATCH_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

source_batch, target_batch = next(iter(dataset))

BUFFER_SIZE = len(x_train)
steps_per_epoch = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 300

#creating encoder, attention, decoder layers objects
encoder = Encoder(english_vocab, embedding_dim, units, BATCH_SIZE)
attention_layer = Attention(10)
decoder = Decoder(hindi_vocab, embedding_dim, units, BATCH_SIZE)

optimizer = Adam()
loss = SparseCategoricalCrossentropy(from_logits=True, reduction='nonTrue')

def loss_function(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss_ = loss(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ = loss_*mask
    return tf.reduce_mean(loss_)

#TRAIN
def train(source, target, encoder_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        #creating encoder
        encoder_output, forward_h,_,_,_ = encoder(source, encoder_hidden)
        decoder_hidden = encoder_hidden
        #as first input to decoder is 'START_'
        decoder_input = tf.expand_dims([hindi_tokenizer.word_index['start_']]*BATCH_SIZE, 1)
        #teacher forcing = pass the actual word to the Decoder at each time step.
        for t in range(1, target.shape[1]):
            #decoder
            decoder_output, _ = decoder(decoder_input, decoder_hidden, encoder_output)
            loss += loss_function(target[:, t], decoder_output)
            decoder_input = tf.expand_dims(target[:, t], 1)

        batch_loss = (loss/int(target.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        
        return batch_loss

#checkpoint
checkpoint_dir = '/content/gdrive/My Drive/checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

#training
epochs = 10

for epoch in range(epochs):
    start = time.time()
    encoder_hidden = encoder.initialize_hidden_state()
    
    total_loss = 0

    for (batch, (source, target)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train(source, target, encoder_hidden)
        total_loss += batch_loss
        if batch%100 == 0:
            print('Epoch {}, Batch {}, loss {}'.format(epoch + 1, batch, round(batch_loss.numpy(),2)))
    if (epoch+1)%2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
    print('Epoch {}, Loss {:.4f}'.format(epoch + 1, (total_loss / steps_per_epoch)))
    print('Time taken for 1 epoch {} min\n'.format(round((time.time() - start)/60, 1)))

