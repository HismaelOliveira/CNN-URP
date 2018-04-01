import numpy as np 
import re
import itertools
from collections import Counter


def load_data_and_labels():
	positiveE = list(open("./data/pru.csv", "r", encoding='utf-8').readlines())
	positiveE = [s.strip() for s in positiveE]
	negativeE = list(open("./data/npru.csv", "r", encoding='utf-8').readlines())
	negativeE = [s.strip() for s in negativeE]

	xT = positiveE + negativeE
	xT = [s.split(" ") for s in xT]

	positiveL = [[0,1] for _ in positiveE]
	negativeL = [[1,0] for _ in negativeE]

	y = np.concatenate([positiveL,negativeL],0)
	return [xT, y]

def pad_sentences(sentences, padding_word="<PAD/>"):
	sequenceL = max(len(x) for x in sentences)
	paddedS = []

	for i in range(len(sentences)):
		sentence = sentences[i]
		num_padding = sequenceL - len(sentence)
		newSentence = sentence + [padding_word] * num_padding
		paddedS.append(newSentence)
	return paddedS

def build_vocab(sentences):
	wordC = Counter(itertools.chain(*sentences))
	vocabularyInv = [x[0] for x in wordC.most_common()]
	vocabularyInv = list(sorted(vocabularyInv))

	vocabulary = {x: i for i,x in enumerate(vocabularyInv)}
	return [vocabulary, vocabularyInv]

def build_input_data(sentences, labels, vocabulary):
	x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
	y = np.array(labels)
	return [x,y]

def load_data():
	sentences, labels = load_data_and_labels()
	sentences_padded = pad_sentences(sentences)
	vocabulary, vocabularyInv = build_vocab(sentences_padded)
	x, y = build_input_data(sentences_padded, labels, vocabulary)
	return [x, y, vocabulary, vocabularyInv]