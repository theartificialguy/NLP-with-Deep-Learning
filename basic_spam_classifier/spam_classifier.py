import nltk
import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words("english")

def preprocess(data, vectorizer):
	corpus = []
	for i in range(len(data)):
		processed = re.sub("[^a-zA-Z]", " ", data["message"][i])
		processed = processed.lower().split()
		processed = [lemmatizer.lemmatize(word) for word in processed if word not in set(stop_words)]
		processed = " ".join(processed)
		corpus.append(processed)

	X = vectorizer.fit_transform(corpus).toarray()
	return X

def preprocessTest(data, vectorizer):
	corpus = []
	for i in range(len(data)):
		processed = re.sub("[^a-zA-Z]", " ", data[i])
		processed = processed.lower().split()
		processed = [lemmatizer.lemmatize(word) for word in processed if word not in set(stop_words)]
		processed = " ".join(processed)
		corpus.append(processed)

	X = vectorizer.fit_transform(corpus).toarray()
	return X

tf = TfidfVectorizer(max_features=5000)
X = preprocess(data, tf)
print(X.shape)
y = pd.get_dummies(data["label"])
y = y.iloc[:,1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

confusion_m = confusion_matrix(y_test, y_pred)
print(confusion_m)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

test_string = ["Hey there, this is an email informing u that this is useless pls dont invest here."]

x = preprocessTest(test_string, tf)

def PadSequences(x):
	result = np.zeros(shape=(1, 5000))
	result[:(x.shape[0]), :(x.shape[1])] = x
	return result

x_to_test = PadSequences(x)

output = model.predict(x_to_test)
print(output)
