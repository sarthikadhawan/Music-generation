from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.layers.recurrent import GRU
from keras.models import model_from_json
from sklearn.externals import joblib
from playabc import *
import os.path
import argparse
#from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import glob

from keras.layers.convolutional import Convolution2D
from keras.layers import Convolution3D, MaxPooling3D

from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten
#from keras.layers.wrappers import TimeDistributed

import xmltodict as x2d
import numpy as np
from hmmlearn import hmm

from collections import Counter, defaultdict
from itertools import count
import math
import h5py
import midi
import glob
from sklearn.externals import joblib


import pysynth as ps


from keras.models import load_model
import sys
import webbrowser

import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
import soundfile as sf
import pickle
import scipy.io.wavfile


genre=""
model=""
seed=1


def is_valid(generated,m,l,k):
    data=""
    c=1
    for line in generated.splitlines():

	s=''

	if c==1 and line.split(":")[0]!='X':
		s=""
		s=s+"X:1\n"
		c=c+1
	if c==2 and line.split(":")[0]!='T':
		s=s+"T:a\n"
		c=c+1
	if c==3 and line.split(" ")[0]!='%':
		s=s+"%:a\n"
		c=c+1
	if c==4 and line.split(":")[0]!='S':
		s=s+"S:a\n"
		c=c+1
	


	if c==5 and line.split(":")[0]!='M' :
		s=s+"M:"+str(m[0])+"\n"
		c=c+1
	if (c==6 and line.split(":")[0]!='L') or (c==6 and line.split(":")[0]!='P'):
		s=s+"L:"+str(l[0])+"\n"
		c=c+1
	if c==7 and line.split(":")[0]!='K':
		s=s+"K:"+str(k[0])+"\n"
		c=c+1
	else:
		s=s+line+"\n"



	c=c+1
        

	
        '''elif len(line.split(":"))>=2: 
		if not line.split(":")[1].strip() :
            		s=s+line.split(":")[0]+":a"
		else:
			s=s+line.split(":")[0]+":"+line.split(":")[1]
        else:
            s=s+line.split(":")[0]'''
        if not s:
		return 0
        data=data+s

            
    return data

def write_to_file(generated,filee, genre):
	file = open(os.getcwd()+'/usurvey/'+filee+".abc","w")
	file.write(generated)
	file.write("\n")
	print(os.getcwd())
	if (genre=='rockpop'):
		os.system('python '+os.getcwd()+'/usurvey/'+'playabc.py --syn_s')
	else:
		os.system('python '+os.getcwd()+'/usurvey/'+'playabc.py')

def sample(preds, temperature=1.0):

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

start_index=0



def gruandlstm(filee, genre, seed, model):
	#path = get_file('nietzsche.txt')
	paths = glob.glob(os.getcwd()+'/usurvey/'+genre+'/*.abc')
	text=''
	for path in paths:
	    text += (open(path).read())

	#print('corpus length:', len(text))

	chars = sorted(list(set(text))) #Set of characters.
	#print('total chars:', len(chars))

	#Declare two mappings, char to idx and idx to char.
	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))

	starts=[]
	m=[]
	l=[]
	k=[]
	v=0
	for i in range (len(text)):
	    if (text[i]=='X' and text[i+1]==':'):
		    starts.append(i)
		    v=v+1
	    if (text[i]=='M' and text[i+1]==':'):
		    m.append(text[i+2]+text[i+3]+text[i+4])
		    i=i+4
	    if (text[i]=='L' and text[i+1]==':'):
		    l.append(text[i+2]+text[i+3]+text[i+4])
		    i=i+4
	    if (text[i]=='K' and text[i+1]==':'):
		    k.append(text[i+2])
		    i=i+2
	    

	# cut the text in semi-redundant sequences of maxlen characters
	if (genre=='rockpop' and model=='GRU'):
		maxlen=50
	else:
		maxlen=40

	step = 3
	sentences = []
	next_chars = []
	for i in range(0, len(text) - maxlen, step):
	    sentences.append(text[i: i + maxlen])
	    next_chars.append(text[i + maxlen])

	x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
	y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
	for i, sentence in enumerate(sentences):
	    for t, char in enumerate(sentence):
		x[i, t, char_indices[char]] = 1
	    y[i, char_indices[next_chars[i]]] = 1



	kernel_size = [5,5,5]
	input_shape = (1, len(sentences), maxlen, len(chars))
	model = load_model(os.getcwd()+'/usurvey/'+filee)

	optimizer = RMSprop(lr=0.01)

	model.compile(loss='mean_squared_error', optimizer='adam')

	for iteration in range(0, v):
	    


	    generated=''
	    for var in [0.2, 0.5, 0.75, 1.0,1.2]:

		    #generated = ''
		    #start_index = random.randint(0, len(text) - maxlen - 1)
		    start_index=seed%(len(text) - maxlen - 1)
		    
		   
		    sentence = text[start_index: start_index + maxlen]
		    generated += sentence
	
		    for i in range(200):
				x_pred = np.zeros((1, maxlen, len(chars)))
				for t, char in enumerate(sentence):
				    x_pred[0, t, char_indices[char]] = 1.
				preds = model.predict(x_pred, verbose=0)[0]
				next_index = sample(preds,var)
				next_char = indices_char[next_index]
				if next_char=='\n':
					continue
				generated += next_char
				sentence = sentence[1:]
				sentence=sentence + next_char

	    start_index+=starts[iteration]
	    

	

	    if is_valid(generated,m,l,k)!=0:
		    print (is_valid(generated,m,l,k))
		    if iteration==5:
		    	write_to_file(is_valid(generated,m,l,k),"1", genre)

def convert(notes,q):
    count=0
    lister=[[0 for j in range(12)] for i in range(11)]
    for i in range(11):
        for j  in range(12):
            lister[i][j]=count
            count+=1
            if(count==128):
                break
        if(count==128):
            break
    
    for  i in range(11):
        for j in range(12):
            if(q==lister[i][j]):
                a=notes[j]
                b=i
    return a+str(b)+"*",5


notes=['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']






def open_h5_data(filename):
    final_data=[]
    final_keys=[]
    
    #Open the HDF5 file.
    hf = h5py.File(filename, 'r')
    song_list=hf.keys()
    song=song_list[0]
    #There are 330 keys (groups), each pertaining to a different song.
    #for song in song_list:
        #Each song is an h5 group (tuple) of 2 keys: data and features.
    Xdata=hf[song]['data'][:] #data dataset is a C array containing the audio signal
    Xlabels=hf[song]['labels'][:] #labels dataset

    return Xdata, Xlabels
 

def open_midi(genre):
	f=[]
	print("hereeeeeeeeeeee")
	print(os.getcwd())
	paths = glob.glob(os.getcwd()+'/usurvey/'+genre+'/*.mid')
	print(paths)
	print(genre+'/*.mid')
	print("endpath")

	for path in paths:
		pattern = midi.read_midifile(path)
		f=[]
		for track in pattern:

		    for event in track:
			if isinstance(event, midi.NoteEvent): 
			    f.append(event.get_pitch())
	return f

def tmatrix(lst):
    # defaultdict that'll produce a unique index for each unique character
    # encountered in lst
    indices = defaultdict(count().next)
    unique_count = len(set(lst))
    b = [[0 for _ in xrange(unique_count)] for _ in xrange(unique_count)]
    for (x, y), c in Counter(zip(lst, lst[1:])).iteritems():
        b[indices[x]][indices[y]] = c
    return b






def hmm(filee, genre, seed):



	mapping={}
	for i in range(1,101):
	    mapping[i]=(2**((i-69)/12.0))*440
	f=open_midi(genre)
	print("IN HMMMMa")
	print(f)
	no_of_notes=len(set(f))
	transmat=tmatrix(f)

	note_map={}
	note_map_reverse={}

	counter=0

	for element in set(f):
	    note_map[counter]=element
	    note_map_reverse[element]=counter
	    counter+=1

	    

	for x in range (len(transmat)):
	    s=sum(transmat[x])*1.0
	    transmat[x][:] = [1.0/len(transmat[x]) if s==0  else xx / s for xx in transmat[x] ]

	#model.startprob_ = np.array([0.2, 0.3, 0.5])
	arr=[]
	for i in range (no_of_notes):
	    arr.append(0)

	arr[0]=1
	f2=[]
	for ff in f:
	    f2.append(note_map_reverse[ff])
	
	final=[]

	model = joblib.load(os.getcwd()+"/usurvey/"+filee)
	ss=seed%10
	final_seq=[ss,(ss+1),(ss+2),(ss+3)]
	X = np.atleast_2d(final_seq).T
	final_seq=model.decode(X)[1]
	final+=final_seq.tolist()

	for q in range(5):
	    X = np.atleast_2d(final).T
	    final_seq=model.decode(X)[1]
	    final+=final_seq.tolist()



	print(note_map)
	print(final_seq)
	
	final_note_seq=[]

	for element in final_seq:
		if element in note_map.keys():
		    final_note_seq.append(note_map[element])




	final_note_seq_freq=[]

	

	for element in final_note_seq:
	    final_note_seq_freq.append(mapping[element])

	list2=final_note_seq
	l=[]
	for i in list2:
	    l.append(tuple(convert(notes,i)))

	l=tuple(l)


	ps.make_wav(l, fn = os.getcwd()+"/templates/"+"1"+".wav")

def gans(genre,seed):

	file = open(os.getcwd()+'/usurvey/'+'GenWeights.pkl','r')

	gen = pickle.load(file)

	np.random.seed(seed)
	noise = np.random.uniform(size = [1,10])

	ans = (gen.predict(noise))
	combined = np.vstack((ans, ans)).T
	scipy.io.wavfile.write(os.getcwd()+'/templates/'+'1.wav',1000,combined)

def callfile(model, genre, seed):
	model=model
	genre=genre
	seed=seed
	print("in callfile")
	print (model)
	print (genre)
	print (seed)

	if model=='GRU' or model=='LSTM':
		if genre=='classical':
			gruandlstm(model.lower()+'_'+'classical.h5', genre, seed, model)
		elif genre=='rockpop':
			gruandlstm(model.lower()+'_'+'rockpop.h5', genre, seed, model)

	elif model=='HMM':
		if genre=='classical':
			hmm(model.lower()+'_'+'classical.h5', genre, seed)
		elif genre=='rockpop':
			hmm(model.lower()+'_'+'rockpop.h5', genre, seed)

	elif model=='GANS':
		gans(genre,seed)









