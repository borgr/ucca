import math
import distance
from munkres import Munkres, print_matrix
import numpy as np

def preprocess_word(word):
	"""standardize word form for the alignment task"""
	return word.strip().lower()

def align(sen1, sen2):
	"""finds the best mapping of words from one sentence to the other"""
	#find lengths
	sen1 = list(map(preprocess_word, sen1.split()))
	sen2 = list(map(preprocess_word, sen2.split()))
	lengthDif = len(sen1) - len(sen2)
	if lengthDif > 0:
		shorter = sen2
		longer = sen1
	else:
		shorter = sen1
		longer = sen2
		lengthDif = abs(lengthDif)
	shorter += ["emptyWord"] * lengthDif

	#create matrix	
	matrix = np.zeros((len(longer), len(longer)))
	for i in range(len(longer)):
		for j in range(len(longer) - lengthDif):
			matrix[i,j] = distance.levenshtein(longer[i], shorter[j])
	print(matrix)
	
	#compare with munkres
	m = Munkres()
	indexes = m.compute(matrix)
	print("mapping is:",[(shorter[i], longer[j]) for (i,j) in indexes])

