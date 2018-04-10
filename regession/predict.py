import json
from sklearn.externals import joblib
import csv
from data_input import data_input
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
from itertools import izip

def LoadData(filePath):
	reader = csv.reader(open(filePath))
	jobList = []
	for i,row in enumerate(reader):
		if i > 0:
			jobList.append(data_input(row))
	return jobList

def TokenizingTestData(data,start,end,vect,vect2):
	targets = []
	desTests = []
	titleTests = []
	for i in xrange(start,end):
		job = data[i].data
		desTests.append(job['organization'] + ' ' + job['joblocation'] + ' ' + job['education'] + ' ' + 
		job['experience'] + ' ' + job['employmentType'] + ' ' + job['industry'] + ' ' + job['jobfunction'])
		titleTests.append(job['title'])
	des = vect.transform(desTests)
	title = vect2.transform(titleTests)
	tests = hstack((des,title))
	return tests

def getTestData(data,start,end,vect,vect2):
	targets = []
	desTests = []
	titleTests = []
	for i in xrange(start,end):
		job = data[i].data
		desTests.append(job['organization'] + ' ' + job['joblocation'] + ' ' + job['education'] + ' ' + 
		job['experience'] + ' ' + job['employmentType'] + ' ' + job['industry'] + ' ' + job['jobfunction'])
		titleTests.append(job['title'])
	title = vect.transform(titleTests)
	des = vect2.transform(desTests)
	tests = hstack((title,des))
	return tests

paths = json.load(open('SETTINGS.json','rb'))
print 'Loading models'
model = joblib.load(paths['MODEL_PATH'])
vect = joblib.load(paths['VECTORIZER_1_PATH'])
vect2 = joblib.load(paths['VECTORIZER_2_PATH'])

print 'loading testing data',paths['TEST_DATA_PATH']
validData = LoadData(paths['TEST_DATA_PATH'])
titles = [validData[i].data['title'] for i in xrange(len(validData))]
validSalary = [validData[i].data['SalaryMid'] for i in xrange(len(validData))]
validTests = getTestData(validData,0,len(validData),vect,vect2)

print 'predicting'
predictions = np.exp(model.predict(validTests))

print 'writing to csv'
with open(paths['SUBMISSION_PATH'],'wb') as fOut:
    out = csv.writer(fOut)
    for row in izip(titles,validSalary,predictions):
        out.writerow(row)
