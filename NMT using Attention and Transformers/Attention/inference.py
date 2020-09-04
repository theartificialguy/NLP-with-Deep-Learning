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

def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub("[^a-zA-Z]", " ", sentence)
    sentence = sentence.rstrip().strip()
    sentence = re.sub(" +", " ", sentence)
    sentence = 'start_ ' + sentence + ' _end'
    return sentence

def evaluate(sentence):
    attention_plot = np.zeros((max_hindi_sentence_length, max_eng_sentence_length))
    sentence = preprocess_sentence(sentence)
    inputs = [english_tokenizer.word_index[i] for i in sentence.split()]
    padded_input = pad_sequences([inputs], maxlen=max_eng_sentence_length, padding='post')
    input_tensor = tf.convert_to_tensor(padded_input)

    result = ''

    #creating encoder
    hidden = [tf.zeros((1, units)) for i in range(4)]
    encoder_output, encoder_hidden,_,_,_ = encoder(input_tensor, hidden)

    #creating decoder
    decoder_hidden = encoder_hidden
    decoder_input = tf.expand_dims([hindi_tokenizer.word_index['start_']], 0)

    for t in range(max_hindi_sentence_length):
        predictions, attention_weights = decoder(decoder_input, hidden, encoder_output)

        #getting word prediction
        predicted_index = tf.argmax(predictions[0]).numpy()
        result += hindi_tokenizer.index_word[predicted_index]+' '

        #checking end of string
        if hindi_tokenizer.index_word[predicted_index] == '_end':
            # print('end encountered! returning!')
            return result, sentence
        
        #feeding back the predicted_index into the model 
        decoder_input = tf.expand_dims([predicted_index], 0)
    
    return result, sentence

def translate(sentence):
    predicted_sentence, sent = evaluate(sentence)
    print('INPUT SENTENCE: {}'.format(sentence))
    print('TRANSLATED SENTENCE: {}'.format(predicted_sentence[:len(predicted_sentence)-5]))

checkpoint_dir = '/content/gdrive/My Drive/checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

#restoring last saved checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

translate("what is wrong with you")