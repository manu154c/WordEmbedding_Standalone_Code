"""
@Auther : MANU C
Created : 15-03-18
Last Modified : 18-03-18

Title : Skip-Gram Word2Vec Implementation

Status : Added Neural network code

Libraries Used : NLTK, NUMPY

Python 3.6

"""


from django.shortcuts import render
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import pdb
import string


# Create your views here.


#Input array
X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

#Output
y=np.array([[1],[1],[0]])

#Variable initialization
epoch=5000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))


def index(request):

	if request.method == "POST":

		input_text = request.POST['input_text']

		input_tokens = text_cleaning(input_text) # process the input text

		input_dictionary = create_dictionary(input_tokens)

		input_vector = create_vectors(input_dictionary)

		neural_output = perform_neural_networking(input_vector)

		output = neural_output
	
		return render(request, 'process/ajax_output_result.html', {'output' : output})

	else:

		return render(request, 'process/get_request.html')


# serve as preprocessing of input.
# removes stop words, punctuation from the input text.
# also performs stemming
def text_cleaning(input_doc):
	
	# nltk tokenize 
    tokens = word_tokenize(input_doc)

    # convert to lower case
    tokens_lower = [w.lower() for w in tokens]

    # remove punctuation from each word using "python string"
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens_lower]
    stripped = filter(None, stripped)
    stripped = list(stripped)

    # stopword removal using nltk english stopwords.
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in stripped if not w in stop_words]

    # stemming using nltk.stem.lancaster
    stemmer = LancasterStemmer()
    words = [stemmer.stem(w) for w in filtered_sentence]
    words = list(set(words))
    
    return words


# Creates a dictionary from the input document.
def create_dictionary(input_token):
    #unique_values_list = list(set(input_token))
    output = {}
    i = 0
    # pdb.set_trace()
    for item in input_token:
        if item not in output:
            output[item] = i
            i = i + 1
            
    # pdb.set_trace()

    return output


def create_vectors(input_dictionary):

	size_of_vector = len(input_dictionary)

	main_list = []

	for key, value in input_dictionary.items():
		inside_list = [0]*size_of_vector
		#pdb.set_trace()
		value = int(value)
		inside_list[value] = 1
		main_list.append(inside_list) 

	#pdb.set_trace()

	return main_list

#Sigmoid Function
def sigmoid (x):
	return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
	return x * (1 - x)


def perform_neural_networking(input_vector):

	#Input array
	X=np.array(input_vector)

	#Output
	y=np.array([[1],[1],[0]])

	#Variable initialization
	epoch=5000 #Setting training iterations
	lr=0.1 #Setting learning rate
	inputlayer_neurons = X.shape[1] #number of features in data set
	hiddenlayer_neurons = 3 #number of hidden layers neurons
	output_neurons = 1 #number of neurons at output layer

	#weight and bias initialization
	wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
	bh=np.random.uniform(size=(1,hiddenlayer_neurons))
	wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
	bout=np.random.uniform(size=(1,output_neurons))

	for i in range(epoch):
		output = forward_propogation()

	return output

def forward_propogation():

	#Forward Propogation
	hidden_layer_input1=np.dot(X,wh)
	hidden_layer_input=hidden_layer_input1 + bh
	hiddenlayer_activations = sigmoid(hidden_layer_input)
	output_layer_input1=np.dot(hiddenlayer_activations,wout)
	output_layer_input= output_layer_input1+ bout
	output = sigmoid(output_layer_input)

	return output


def backward_propogation():

	#Backpropagation
	E = y-output
	#pdb.set_trace()
	slope_output_layer = derivatives_sigmoid(output)
	slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
	d_output = E * slope_output_layer
	Error_at_hidden_layer = d_output.dot(wout.T)
	d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
	wout += hiddenlayer_activations.T.dot(d_output) *lr
	bout += np.sum(d_output, axis=0,keepdims=True) *lr
	wh += X.T.dot(d_hiddenlayer) *lr
	bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

	return 1