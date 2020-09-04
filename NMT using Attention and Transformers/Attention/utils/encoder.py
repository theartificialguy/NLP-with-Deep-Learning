import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Bidirectional
from tensorflow.keras.layers import LSTM, Dense, Embedding

class Encoder(Model):
	
    def __init__(self, vocab_size, embedding_dim, encoder_units, batch_size):
        super(Encoder, self).__init__()
        self.encoder_units = encoder_units
        self.batch_size = batch_size
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = Bidirectional(LSTM(encoder_units, 
                                       return_sequences=True, 
                                       return_state=True, 
                                       recurrent_initializer='glorot_uniform',
                                       dropout=0.2))

    def call(self, x, hidden):
        x = self.embedding(x)
        output, forward_h, forward_c, backward_h, backward_c = self.lstm(x, initial_state=hidden)
        return output, forward_h, forward_c, backward_h, backward_c
    
    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_size, self.encoder_units)) for i in range(4)]