"""
@Auther : MANU C
Created : 15-03-18
Last Modified : 08-08-18

Title : Skip-Gram Word2Vec Implementation

Status : Bigram Model 

Next Work : Convert to skip-gram

Libraries Used : NLTK, NUMPY

Python 3.6

"""


from django.shortcuts import render
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, brown
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import pdb
import string


# Create your views here.

"""
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
"""

def index(request):

	if request.method == "POST":

		input_text = request.POST['input_text']

		input_tokens = text_cleaning(input_text) # process the input text

		input_dictionary = create_dictionary(input_tokens)

		input_vector = create_vectors(input_dictionary, input_tokens)

		bigram_list = find_bigrams(input_vector)

		neural_output = perform_neural_networking(bigram_list)

		output_format = create_output(neural_output, input_dictionary)

		output = output_format
	
		return render(request, 'process/ajax_output_result.html', {'output' : output})

	else:

		return render(request, 'process/get_request.html')



def train_using_nltk_brown_corpus(request):

	brown_list_sentence = brown.sents()
	remove_str = [".", ",", "--", ";", "``", "&", "?", ")", "("]
	input_tokens = []
	for item in brown_list_sentence:
		for i in item:
			if i not in remove_str:
				input_tokens.append(i)

	input_tokens = input_tokens[:500]


	input_dictionary = create_dictionary(input_tokens)

	input_vector = create_vectors(input_dictionary, input_tokens)

	bigram_list = find_bigrams(input_vector)

	neural_output = perform_neural_networking_train(bigram_list)

	output_format = create_output_train(neural_output, input_dictionary)

	output = output_format
	
	return render(request, 'process/ajax_train_result.html', {'output' : output})



def find_bigrams(input_list):
  bigram_list = []
  for i in range(len(input_list)-1):
      bigram_list.append((input_list[i], input_list[i+1]))
  return bigram_list


def create_output(neural_output, input_dictionary):

	i = 0
	table = ""

	for node in input_dictionary:
		table = table + " <tr class='' > "
		table = table + " <td> " + str(node) + " </td> "
		table = table + " <td> " + str(neural_output[i][0]) + " </td> "
		table = table + " <td> " + str(neural_output[i][1]) + " </td> "
		table = table + " <td> " + str(neural_output[i][2]) + " </td> "
		table = table + " </tr>"
		i = i + 1

	#pdb.set_trace()

	return table

def create_output_train(neural_output, input_dictionary):

	i = 0
	j = 0
	table = ""
	loops = 100

	for node in input_dictionary:
		table = table + " <tr class='' > "
		table = table + " <td> " + str(node) + " </td> "
		for j in range(loops):
			table = table + " <td> " + str(neural_output[i][j]) + " </td> "		
			j = j + 1
		table = table + " </tr>"
		i = i + 1

	#pdb.set_trace()

	return table

# serve as preprocessing of input.
# removes stop words, punctuation from the input text.
# also performs stemming
def text_cleaning(input_doc):
	
	# nltk tokenize 
    tokens = word_tokenize(input_doc)
    table = str.maketrans('', '', string.punctuation)
    punctuation_removed_list = [w.translate(table) for w in tokens]
    token_list = list(filter(None, punctuation_removed_list))

    """ NOT NEEDED FOR WORD EMBEDDING

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

    """
    return token_list


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


def create_vectors(input_dictionary, input_token):

	size_of_vector = len(input_dictionary)
	main_list = []

	for value in input_token:
		inside_list = [0]*size_of_vector
		#pdb.set_trace()
		index = int(input_dictionary[value])
		inside_list[index] = 1
		main_list.append(inside_list) 

	#pdb.set_trace()

	return main_list

#Sigmoid Function
def sigmoid (x):
	return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
	return x * (1 - x)

def softmax(x):
    """Calculate softmax based probability for given input vector
    # Arguments
        x: numpy array/list
    # Returns
        softmax of input array
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def perform_neural_networking(input_vector):

	for unique_tuple in input_vector:


		#Input array
		
		X=np.array(input_vector[0][0])
		#pdb.set_trace()
		#no_of_input_neuron = X.shape[0]
		inside_list = [[0]]
		
		#Output
		
		y=np.array(input_vector[0][1])

		#Variable initialization
		epoch=5 #Setting training iterations
		lr=0.1 #Setting learning rate
		inputlayer_neurons = X.shape[0] #number of words in dictionary
		hiddenlayer_neurons = 3 #number of hidden layers neurons
		output_neurons = X.shape[0] #number of neurons at output layer

		#weight and bias initialization
		wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
		bh=np.random.uniform(size=(1,hiddenlayer_neurons))
		wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
		bout=np.random.uniform(size=(1,output_neurons))

		#pdb.set_trace()

		#forward_propogation(X, wh, bh, wout, bout) # test_call

		for i in range(epoch):
			dict_output = forward_propogation(X, wh, bh, wout, bout)

			output = dict_output['output']
			hiddenlayer_activations = dict_output['hiddenlayer_activations']

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
			wh += np.transpose([X]).dot(d_hiddenlayer) *lr
			bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

			#pdb.set_trace()

	print("Weight\n")	
	#pdb.set_trace()	
	return wh



def perform_neural_networking_train(input_vector):

	for unique_tuple in input_vector:


		#Input array
		
		X=np.array(input_vector[0][0])
		#pdb.set_trace()
		#no_of_input_neuron = X.shape[0]
		inside_list = [[0]]
		
		#Output
		
		y=np.array(input_vector[0][1])

		#Variable initialization
		epoch=10 #Setting training iterations
		lr=0.1 #Setting learning rate
		inputlayer_neurons = X.shape[0] #number of words in dictionary
		hiddenlayer_neurons = 100 #number of hidden layers neurons
		output_neurons = X.shape[0] #number of neurons at output layer

		#weight and bias initialization
		wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
		bh=np.random.uniform(size=(1,hiddenlayer_neurons))
		wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
		bout=np.random.uniform(size=(1,output_neurons))

		#pdb.set_trace()

		#forward_propogation(X, wh, bh, wout, bout) # test_call

		for i in range(epoch):
			dict_output = forward_propogation(X, wh, bh, wout, bout)

			output = dict_output['output']
			hiddenlayer_activations = dict_output['hiddenlayer_activations']

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
			wh += np.transpose([X]).dot(d_hiddenlayer) *lr
			bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

			#pdb.set_trace()

	print("Weight\n")	
	#pdb.set_trace()	
	return wh



def forward_propogation(X, wh, bh, wout, bout):

	#Forward Propogation
	#pdb.set_trace()
	hidden_layer_input1=np.dot(X,wh)
	hidden_layer_input=hidden_layer_input1 + bh # Linear Transformation
	# no non-linear transformation in hidden layer
	hiddenlayer_activations = hidden_layer_input # sigmoid(hidden_layer_input)
	output_layer_input1=np.dot(hiddenlayer_activations,wout)
	output_layer_input= output_layer_input1+ bout
	output = softmax(output_layer_input) # non linear transformation in output layer
	#pdb.set_trace()

	return {"output" : output, "hiddenlayer_activations" : hiddenlayer_activations}


"""def backward_propogation():

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
"""