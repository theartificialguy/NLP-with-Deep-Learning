import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Bidirectional
from tensorflow.keras.layers import LSTM, Dense, Embedding

class Decoder(Model):
	
    def __init__(self, vocab_size, embedding_dim, decoder_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.decoder_units = decoder_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(decoder_units, 
                         return_sequences=True, 
                         return_state=True, 
                         recurrent_initializer='glorot_uniform',
                         dropout=0.2)
        self.fully_connected = Dense(vocab_size)
        self.attention = Attention(self.decoder_units)

    def call(self, x, hidden, encoder_output):
        context_vector, attention_weights = self.attention(hidden, encoder_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output,_,_ = self.lstm(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fully_connected(output)
        return x, attention_weights