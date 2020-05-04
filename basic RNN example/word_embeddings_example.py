import tensorflow
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.layers import LSTM, Activation

docs = ['My name is yash',
	'I like mango shake',
	'yash is computer vision guru',
	'artificial intelligence is awesome']

vocab_size = 10000
max_doc_len = 7
embedding_len = 5

one_hot_representation = [one_hot(word, vocab_size) for word in docs]
one_hot_representation = pad_sequences(one_hot_representation, truncating="post", padding="post", maxlen=max_doc_len)
                      
model = Sequential()
model.add(Embedding(vocab_size, embedding_len, input_length=max_doc_len)) 
model.add(Dense(1))
model.add(Activation("softmax"))
model.compile(optimizer="rmsprop",loss="mse")
output = model.predict(one_hot_representation)
print(output)
print(output.shape)
