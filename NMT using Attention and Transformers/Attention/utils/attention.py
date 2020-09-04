import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Bidirectional
from tensorflow.keras.layers import LSTM, Dense, Embedding

class Attention(Layer):
	
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, hidden, encoder_output):
        for hidden_ in hidden:
            hidden_with_time = tf.expand_dims(hidden_, 1)
            score = self.V(K.tanh(self.W1(encoder_output)+self.W2(hidden_with_time)))
            attention_weights = K.softmax(score, axis=1)
            context_vector = attention_weights*encoder_output
            context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights