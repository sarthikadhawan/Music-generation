# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render

from predict_hmm_lstm_gru import *
from django.http import HttpResponseRedirect

from django.contrib import admin
from django.conf.urls.static import static
from django.conf import settings
from django.http import HttpResponse
from django.shortcuts import render, redirect
import numpy as np
import pandas as pd
import csv
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from django.views.decorators.csrf import csrf_exempt

# Create your views here.


@csrf_exempt
def home(request):
	if request.method == 'POST':
		monlist=[]
  		monlist.append(request.POST['model'].encode("latin-1"))
  		monlist.append(request.POST['genre'].encode("latin-1"))
  		monlist.append(int(request.POST['seed']))
  	  	print(monlist)
  	  	callfile(monlist[0],monlist[1],monlist[2])
   		return HttpResponseRedirect("/play")

		#s=str(i0)+str(i1)+str(i2)+str(selected_option)
		#print s
		#print "Ash"

	return render(request, "home.html")


@csrf_exempt
def home1(request):
	if request.method == 'POST':
		monlist=[]
  		monlist.append(request.POST['model'].encode("latin-1"))
  		monlist.append(request.POST['genre'].encode("latin-1"))
  		monlist.append(int(request.POST['seed']))
  	  	print(monlist)
  	  	callfile(monlist[0],monlist[1],monlist[2])
   		return HttpResponseRedirect("/play")

		#s=str(i0)+str(i1)+str(i2)+str(selected_option)
		#print s
		#print "Ash"

	return render(request, "home1.html")


def play(request):
	return render(request, "xx.html")


def trial(request):
	return render(request, "home1.html")


def graphs(request):
	return render(request, "graphs.html")

def about(request):
	return render(request, "about.html")

def LSTM(request):
	return render(request, "LSTM.html")

def GRU(request):
	return render(request, "GRUV.html")

def RBM(request):
	return render(request, "RBM.html")
'''def home(request):
	global mylist
	finallist=[]
	finallist.append(mylist)
	if (ml_script(finallist)==0):
		s="Result: Negative"
	else:
		s="Result: Positive"
	return HttpResponse(s)
	#return HttpResponse(ml_script([[23, 2, 17, 3, False, 0, 0, False, 2, False, 0, False, 0, False, False, False, False, False, False, False, False, False, False, False, False, 0, 6, 5, 0, 0, 0, 0]]))
'''