import json
import csv
import numpy as np
from time import time,sleep
from data_input import data_input
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn import neural_network
from itertools import izip

import os

def LoadData(filePath):
	reader = csv.reader(open(filePath))
	jobList = []
	for i,row in enumerate(reader):
		if i > 0:
			jobList.append(data_input(row))
	return jobList

def fit(merged,salary):
	tModel = time()	
	model = linear_model.LinearRegression()
	model.fit(merged,salary)
	print 'Train model done in',time()-tModel,'seconds'
	return model


def TokenizingData(jobs):
	tToken = time()
	salary = []
	des = []
	titles = []
	for i in xrange(trainNum):
		job = jobs[i].data
		salary.append(np.log(float(job['SalaryMid'])))
		des.append(job['organization'] + ' ' + job['joblocation'] + ' ' + job['education'] + ' ' + 
		job['experience'] + ' ' + job['employmentType'] + ' ' + job['industry'] + ' ' + job['jobfunction'])
		titles.append(job['title'])
	print 'Tokenizing'
	vect = TfidfVectorizer(min_df=1,ngram_range=(1,3),max_features=24000000)
	vect2 = TfidfVectorizer(min_df=1,ngram_range=(1,3),max_features=24000000)
	jobtitles = vect.fit_transform(titles)
	descrip = vect2.fit_transform(des)	
	merged = hstack((jobtitles,descrip))
	print 'Tokenized job data in',time()-tToken,'seconds'
	return merged,salary,vect,vect2


paths = json.load(open('SETTINGS.json','rb'))
t0 = time()
print 'Loading data from',paths['TRAIN_DATA_PATH']
jobs = LoadData(paths['TRAIN_DATA_PATH'])
trainNum = len(jobs)
merged,salary,vect,vect2= TokenizingData(jobs)
print 'Traning model'
model = fit(merged,salary)
print 'Save model as',paths['MODEL_PATH']
joblib.dump(model,paths['MODEL_PATH'],compress=3)
joblib.dump(vect,paths['VECTORIZER_1_PATH'],compress=3)
joblib.dump(vect2,paths['VECTORIZER_2_PATH'],compress=3)