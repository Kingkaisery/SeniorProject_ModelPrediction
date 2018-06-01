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
	#model = svm.SVR(kernel='poly', degree=3)
	#model = neighbors.KNeighborsRegressor(n_neighbors=10)
	#model = tree.DecisionTreeRegressor()
	#model = neural_network.MLPRegressor(hidden_layer_sizes=(50, ))
	#model = linear_model.BayesianRidge()
	model.fit(merged,salary)
	print 'Train model done in',time()-tModel,'seconds'
	return model


def TokenizingData(jobs):
	tToken = time()
	salary = []
	des = []
	titles = []
	level = []
	org = []
	lo = []
	edu = []
	exp = []
	emp = []
	ind = []
	jobfunc = []
	for i in xrange(trainNum):
		job = jobs[i].data
		titles.append(job['title'])
		level.append(job['level'])
		org.append(job['organization'])
		lo.append(job['joblocation'])
		edu.append(job['education'])
		exp.append(job['experience'])
		ind.append(job['industry'])
		jobfunc.append(job['jobfunction'])
		salary.append(np.log(float(job['SalaryMid'])))
		#des.append(job['level'] + ' ' + job['organization'] + ' ' + job['joblocation'] + ' ' + job['education'] + ' ' + 
		#job['experience'] + ' ' + job['employmentType'] + ' ' + job['industry'] + ' ' + job['jobfunction'])		
	print 'Tokenizing'
	vect1 = TfidfVectorizer(min_df=1,ngram_range=(1,3),max_features=24000000)
	vect2 = TfidfVectorizer(min_df=1,ngram_range=(1,3),max_features=24000000)
	vect3 = TfidfVectorizer(min_df=1,ngram_range=(1,3),max_features=24000000)
	vect4 = TfidfVectorizer(min_df=1,ngram_range=(1,3),max_features=24000000)
	vect5 = TfidfVectorizer(min_df=1,ngram_range=(1,3),max_features=24000000)
	vect6 = TfidfVectorizer(min_df=1,ngram_range=(1,3),max_features=24000000)
	vect7 = TfidfVectorizer(min_df=1,ngram_range=(1,3),max_features=24000000)
	vect8 = TfidfVectorizer(min_df=1,ngram_range=(1,3),max_features=24000000)
	jobtitles = vect1.fit_transform(titles)
	#descrip = vect2.fit_transform(des)
	joblevel = vect2.fit_transform(level)
	joborg = vect3.fit_transform(org)
	joblo = vect4.fit_transform(lo)
	jobedu = vect5.fit_transform(edu)
	jobexp = vect6.fit_transform(exp)
	jobind = vect7.fit_transform(ind)
	jobfunction = vect8.fit_transform(jobfunc)	
	merged = hstack((jobtitles,joblevel,joborg,joblo,jobedu,jobexp,jobind,jobfunction))
	print 'Tokenized job data in',time()-tToken,'seconds'
	return merged,salary,vect1,vect2,vect3,vect4,vect5,vect6,vect7,vect8


paths = json.load(open('SETTINGS.json','rb'))
t0 = time()
print 'Loading data from',paths['TRAIN_DATA_PATH']
jobs = LoadData(paths['TRAIN_DATA_PATH'])
trainNum = len(jobs)
merged,salary,vect1,vect2,vect3,vect4,vect5,vect6,vect7,vect8= TokenizingData(jobs)
print 'Traning model'
model = fit(merged,salary)
print 'Save model as',paths['MODEL_PATH']
joblib.dump(model,paths['MODEL_PATH'],compress=3)
joblib.dump(vect1,paths['VECTORIZER_1_PATH'],compress=3)
joblib.dump(vect2,paths['VECTORIZER_2_PATH'],compress=3)
joblib.dump(vect3,paths['VECTORIZER_3_PATH'],compress=3)
joblib.dump(vect4,paths['VECTORIZER_4_PATH'],compress=3)
joblib.dump(vect5,paths['VECTORIZER_5_PATH'],compress=3)
joblib.dump(vect6,paths['VECTORIZER_6_PATH'],compress=3)
joblib.dump(vect7,paths['VECTORIZER_7_PATH'],compress=3)
joblib.dump(vect8,paths['VECTORIZER_8_PATH'],compress=3)