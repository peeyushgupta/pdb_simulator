from __future__ import division
import sys,re
import time,pickle
import numpy as np
import heapq
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
import random
from operator import truediv,mul,sub
import csv
import math
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import tree
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import copy
import operator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")

t1 = [0,1,2,3,4,5,6]
q1 = [0.1,0.5,0.51,0.52,0.53,0.54,0.55]
t1 = t1[1:]
#q1 = q1[1:]

budget = 6
weight_t1 = [max(1-float(element - 1)/budget,0) for element in t1]
improv_q1 = [x - q1[i - 1] for i, x in enumerate(q1) if i > 0]

print weight_t1
print improv_q1
print len(weight_t1)
print len(improv_q1)
a = np.dot(weight_t1,improv_q1)
print a



dl2,nl2 = pickle.load(open('MuctTestGender2_XY.p','rb'))
print nl2
print len(nl2)

#dl2,nl2 = pickle.load(open('MuctTestGender6_XY.p','rb'))
#dl= dl2[200:400]
#nl = nl2[200:400]
'''
z = zip(dl2, nl2)

random.shuffle(z)
dl2, nl2 = zip(*z)
'''



#dl2,nl2 = [],[]
#dl,nl = pickle.load(open('MuctTestGender6_XY.p','rb'))
#dl,nl = pickle.load(open('MuctTestGender2_XY.p','rb'))

sys.setrecursionlimit(1500)

t_load_start = time.time()

gender_gnb = pickle.load(open('gender_muct_gnb_calibrated.p', 'rb'))
gender_rf = pickle.load(open('gender_muct_rf_calibrated.p', 'rb'))
gender_dt = joblib.load(open('gender_muct_dt_calibrated.p', 'rb'))
gender_knn = joblib.load(open('gender_muct_knn_calibrated.p', 'rb'))

t_load_end = time.time()

t_load = (t_load_end - t_load_start)

print 'loading finished'



rf_thresholds, gnb_thresholds, et_thresholds,  svm_thresholds = [], [], [] , []
rf_tprs, gnb_tprs, et_tprs,  svm_tprs = [], [] ,[], []
rf_fprs, gnb_fprs, et_fprs, svm_fprs  = [], [] ,[], []
rf_probabilities, gnb_probabilities, et_probabilities, svm_probabilities = [], [], [], []

f1 = open('QueryExecutionResult.txt','w+')
#dl,nl = [],[]

#imageIndex = [i for i in sorted(random.sample(xrange(len(dl)), 100))]

#dl = np.array(dl)
#nl = np.array(nl)
listInitial,list0,list1,list2,list3,list01,list02,list03,list12,list13,list23,list012,list013,list023,list123=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]


def createSample():
	f1 = open('SampleInformation.txt','w+')
	for i in range(5):
		dl_train, dl_test, nl_train, nl_test = train_test_split(dl, nl, test_size=0.1428, random_state=i)
		pickle.dump([dl_test,nl_test],open('MuctTrainGender'+str(i)+'_XY.p','wb'))
		totalMale = (nl_test==1).sum()
		totalObject = len(nl_test)
		selectivity = totalMale/float(totalObject)
		print>>f1,'selectivity of: %i is :%f, total positive:%d'%(i,selectivity,totalMale)


def createRandomSample():
	j=1
	f1 = open('RandomSampleInformation.txt','w+')
	imageIndex = [i for i in sorted(random.sample(xrange(len(dl)), 200))]
	dl_test = [dl[i] for i in  imageIndex]
	nl_test = [nl[i] for i in imageIndex]
	
	
	
	rand_smpl = [dl_test,nl_test]
	nl_test = np.array(nl_test)
	totalMale = (nl_test==1).sum()
	#totalMale = sum(nl_test==1)*1.0/len(nl_test) #Male
	
	totalObject = len(nl_test)
	selectivity = totalMale/float(totalObject)
	print>>f1,'selectivity of: %i is :%f, total positive object:%d, total object:%d'%(i,selectivity,totalMale,totalObject)
	
	pickle.dump(rand_smpl,open('MuctTestGenderRandomSample'+str(j)+'_XY.p','wb'))



	

def genderPredicate1(rl):
	gProb = gender_gnb.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate2(rl):
	gProb = gender_extraTree.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
	
def genderPredicate3(rl):
	gProb = gender_rf.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile

def genderPredicate4(rl):
	gProb = gender_svm.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate5(rl):
	gProb = gender_lr.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate6(rl):
	gProb = gender_dt.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate7(rl):
	gProb = gender_knn.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate8(rl):
	gProb = gender_lda.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate9(rl):
	gProb = gender_sgd.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate10(rl):
	gProb = gender_nusvc.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile
	
def genderPredicate17(rl):
	gProb = gender_dt_new.predict_proba(rl)
	gProbSmile = gProb[:,1]
	return gProbSmile

	
def setup():
	included_cols = [0]
	
	skipRow= 0
	
	
	with open('UncertaintyExperiments/Feature1/listInitialDetails.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 1
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				#print content
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				listInitial.append(temp1)
			rowNum = rowNum+1
	
	
	with open('UncertaintyExperiments/Feature1/list0Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				#print content
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list0.append(temp1)
			rowNum = rowNum+1
	
	with open('UncertaintyExperiments/Feature1/list1Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list1.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature1/list2Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list2.append(temp1)
			rowNum = rowNum+1
	
	with open('UncertaintyExperiments/Feature1/list3Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list3.append(temp1)
			rowNum = rowNum+1
		
	with open('UncertaintyExperiments/Feature1/list01Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list01.append(temp1)
			rowNum = rowNum+1
		
	with open('UncertaintyExperiments/Feature1/list02Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list02.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature1/list03Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list03.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature1/list12Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list12.append(temp1)
			rowNum = rowNum+1
		
	with open('UncertaintyExperiments/Feature1/list13Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list13.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature1/list23Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list23.append(temp1)
			rowNum = rowNum+1
	
	with open('UncertaintyExperiments/Feature1/list012Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list012.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature1/list013Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list013.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature1/list023Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list023.append(temp1)
			rowNum = rowNum+1
			
	with open('UncertaintyExperiments/Feature1/list123Details.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')
		rowNum=0
		endRow = 10
		for row in reader:
			content = list(row[i] for i in included_cols)
			if rowNum >= skipRow and rowNum <= endRow:
				temp = content[0].split(',')
				temp1 = [temp[0],temp[1],temp[2]]
				list123.append(temp1)
			rowNum = rowNum+1
	
	
	
def chooseNextBest(prevClassifier,uncertainty):
	#print currentProbability
	noOfClassifiers = len(prevClassifier)
	uncertaintyList = []
	
	#print prevClassifier
	nextClassifier = -1 
	
	# for objects gone through zero classifiers. This is the initialization stage.
	
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==0) :
		uncertaintyList = listInitial
	
	
	# for objects only gone through one classifiers
	
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==0) :
		uncertaintyList = list0
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==0) :
		uncertaintyList = list1
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==0) :
		uncertaintyList = list2
	if  (prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==1) :
		uncertaintyList = list3
	
	# for objects gone through two classifiers
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==0) :
		uncertaintyList = list01
	if  (prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==0) :
		uncertaintyList = list02
	if  prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==0 and prevClassifier[3] ==1 :
		uncertaintyList = list03
	if  prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==0 :
		uncertaintyList = list12
	if  prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==1 :
		uncertaintyList = list13
	if  prevClassifier[0] ==0 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==1 :
		uncertaintyList = list23
	
	# for objects gone through three classifiers
	
	if  prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==0 :
		uncertaintyList = list012
	if  prevClassifier[0] ==0 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==1 :
		uncertaintyList = list123
	if  prevClassifier[0] ==1 and prevClassifier[1] ==0  and prevClassifier[2] ==1 and prevClassifier[3] ==1 :
		uncertaintyList = list023
	if  prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==0 and prevClassifier[3] ==1 :
		uncertaintyList = list013
	
	if  prevClassifier[0] ==1 and prevClassifier[1] ==1  and prevClassifier[2] ==1 and prevClassifier[3] ==1 :
		return ['NA',0]
	#print 'uncertaintyList'
	#print uncertaintyList
	[nextClassifier,deltaU] = chooseBestBasedOnUncertainty(uncertaintyList, uncertainty)
		
			
	return [nextClassifier,deltaU]
	
def convertEntropyToProb(entropy):
	#print 'entropy: %f'%(entropy)
	for i in range(50):
		f= -0.01*i * np.log2(0.01*i) -(1-0.01*i)*np.log2(1-0.01*i)
		#print f
		if abs(f-entropy) < 0.02:
			#print 0.01*i
			break
	#print 'entropy found: %f'%(0.01*i)
	return 0.01*i
	
	
	

def chooseBestBasedOnUncertainty(uncertaintyList, uncertainty):
	bestClassifier = -1
	index = 0
	#print 'current uncertainty:%f'%(uncertainty)
	#print 'uncertaintyList'
	#print uncertaintyList
	for i in range(len(uncertaintyList)):
		element = uncertaintyList[i]
		if float(element[0]) >= float(uncertainty) :
			index = i
			break
	uncertaintyListElement =  uncertaintyList[index]
	bestClassifier = uncertaintyListElement[1]
	#print bestClassifier
	
	deltaUncertainty = uncertaintyListElement[2]
	#print deltaUncertainty
	
	return [bestClassifier,deltaUncertainty]
	
	
def chooseNextBestBasedOnBlocking(prevClassifier,currentUncertainty,currentProbability):
	miniBlock= []
	print 'inside chooseNextBestBasedOnBlocking'
	# first collecting objects which  are in the same state
	state = 'init'
	stateCollection =[]
	featureVector = []
	#print 'prevClassifier'
	#print prevClassifier
	#print("CurrentProbability: {} ".format(currentProbability))
	
	for i in range(len(prevClassifier)):
		if(prevClassifier.get(i)[0][0] ==0 and prevClassifier.get(i)[0][1] ==0 and prevClassifier.get(i)[0][2]==0 and prevClassifier.get(i)[0][3]==0 ):
			state = 'init'
		if(prevClassifier.get(i)[0][0] ==1 and prevClassifier.get(i)[0][1] ==0 and prevClassifier.get(i)[0][2]==0 and prevClassifier.get(i)[0][3]==0 ):
			state = '0'
		if(prevClassifier.get(i)[0][0] ==0 and prevClassifier.get(i)[0][1] ==1 and prevClassifier.get(i)[0][2]==0 and prevClassifier.get(i)[0][3]==0 ):
			state = '1'
		if(prevClassifier.get(i)[0][0] ==0 and prevClassifier.get(i)[0][1] ==0 and prevClassifier.get(i)[0][2]==1 and prevClassifier.get(i)[0][3]==0 ):
			state = '2'
		if(prevClassifier.get(i)[0][0] ==0 and prevClassifier.get(i)[0][1] ==0 and prevClassifier.get(i)[0][2]==0 and prevClassifier.get(i)[0][3]==1 ):
			state = '3'
		if(prevClassifier.get(i)[0][0] ==1 and prevClassifier.get(i)[0][1] ==1 and prevClassifier.get(i)[0][2]==0 and prevClassifier.get(i)[0][3]==0 ):
			state = '01'
		if(prevClassifier.get(i)[0][0] ==1 and prevClassifier.get(i)[0][1] ==0 and prevClassifier.get(i)[0][2]==1 and prevClassifier.get(i)[0][3]==0 ):
			state = '02'
		if(prevClassifier.get(i)[0][0] ==1 and prevClassifier.get(i)[0][1] ==0 and prevClassifier.get(i)[0][2]==0 and prevClassifier.get(i)[0][3]==1 ):
			state = '03'
		if(prevClassifier.get(i)[0][0] ==0 and prevClassifier.get(i)[0][1] ==1 and prevClassifier.get(i)[0][2]==1 and prevClassifier.get(i)[0][3]==0 ):
			state = '12'
		if(prevClassifier.get(i)[0][0] ==0 and prevClassifier.get(i)[0][1] ==1 and prevClassifier.get(i)[0][2]==0 and prevClassifier.get(i)[0][3]==1 ):
			state = '13'
		if(prevClassifier.get(i)[0][0] ==0 and prevClassifier.get(i)[0][1] ==0 and prevClassifier.get(i)[0][2]==1 and prevClassifier.get(i)[0][3]==1 ):
			state = '23'
		if(prevClassifier.get(i)[0][0] ==1 and prevClassifier.get(i)[0][1] ==1 and prevClassifier.get(i)[0][2]==1 and prevClassifier.get(i)[0][3]==0 ):
			state = '012'
		if(prevClassifier.get(i)[0][0] ==1 and prevClassifier.get(i)[0][1] ==1 and prevClassifier.get(i)[0][2]==0 and prevClassifier.get(i)[0][3]==1 ):
			state = '013'
		if(prevClassifier.get(i)[0][0] ==1 and prevClassifier.get(i)[0][1] ==0 and prevClassifier.get(i)[0][2]==1 and prevClassifier.get(i)[0][3]==1 ):
			state = '023'
		if(prevClassifier.get(i)[0][0] ==0 and prevClassifier.get(i)[0][1] ==1 and prevClassifier.get(i)[0][2]==1 and prevClassifier.get(i)[0][3]==1 ):
			state = '123'
		if(prevClassifier.get(i)[0][0] ==1 and prevClassifier.get(i)[0][1] ==1 and prevClassifier.get(i)[0][2]==1 and prevClassifier.get(i)[0][3]==1 ):
			state = 'NA'
		
		stateCollection.append(state)
	
	
	print("stateCollection: {} ".format(stateCollection))
	

	block0,block1,block2,block3,maxBlock =[],[],[],[],[] # These three variables store information about best block and next best classifier for that block.
	maxNextBestClassifier =''
	deltaUncertainty0, deltaUncertainty1, deltaUncertainty2, deltaUncertainty3, maxDeltaUncertainty = 100,100,100,100,100
	size0,size1,size2,size3 =0,0,0,0
	valMax = sys.float_info.max
	flag = 0
	
	strSet = ['init','0','1','2','3','01','02','03','12','13','23','012','013','023','123']
	for k in range(len(strSet)):
		str = strSet[k]
		subCollection = [i for i, j in enumerate(stateCollection) if j == str]  # it will contain the index of images which have gone through classifier 0 and 1.
		#print subCollection
		#print("state: {} ".format(str))
		#print("subcollection: {} ".format(subCollection))
		if len(subCollection)>0:
			for i in range(len(subCollection)):
				#featureValue = [currentProbability.get(subCollection[i])[0],currentProbability.get(subCollection[i])[1]]
				featureValue = [combineProbability (currentProbability.get(subCollection[i]))]
				#probList = currentProbability.get(subCollection[i])
				#featureValue = [p for p in probList[0] if p !=-1]
				#featureValue = [currentUncertainty[i]]
				#print("featureValue: {} ".format(featureValue))
				featureVector.append(featureValue)
			
			unique_data = [list(x) for x in set(tuple(x) for x in featureVector)]
			uniqueValues = len(unique_data)
			#print("uniqueValues: {} ".format(uniqueValues))
			flag = 1
			
			if uniqueValues >=4:
				#kmeans = KMeans(n_clusters=4, random_state=0).fit(featureVector)
				aggClustering = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
				aggClustering.fit(featureVector)
				
				#print kmeans.labels_
				#print aggClustering.labels_
				
				
				block0Index = [i for i in range(len(aggClustering.labels_)) if aggClustering.labels_[i] == 0]
				block1Index = [i for i in range(len(aggClustering.labels_)) if aggClustering.labels_[i] == 1]
				block2Index = [i for i in range(len(aggClustering.labels_)) if aggClustering.labels_[i] == 2]
				block3Index = [i for i in range(len(aggClustering.labels_)) if aggClustering.labels_[i] == 3]
				
				
				'''
				block0Index = [i for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 0]
				block1Index = [i for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 1]
				block2Index = [i for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 2]
				block3Index = [i for i in range(len(kmeans.labels_)) if kmeans.labels_[i] == 3]
				'''
				
				
				#print("block0Index: {} ".format(block0Index))
				#print("block1Index: {} ".format(block1Index))
				#print("block2Index: {} ".format(block2Index))
				
				block0= [subCollection[x] for x in block0Index]
				block1 = [subCollection[x] for x in block1Index]
				block2 = [subCollection[x] for x in block2Index]
				block3 = [subCollection[x] for x in block3Index]
				
				'''
				block0= [subCollection[x] for x in block0Index]
				block1 = [subCollection[x] for x in block1Index]
				block2 = [subCollection[x] for x in block2Index]
				block3 = [subCollection[x] for x in block3Index]
				'''
				
				'''
				print("block0: {} ".format(block0))
				print("block1: {} ".format(block1))
				print("block2: {} ".format(block2))
				print("MaxBlock: {} ".format(maxBlock))
				'''
				
				size0 = len(block0)
				size1 = len(block1)
				size2 = len(block2)
				size3 = len(block3)
				sizeMax = len(maxBlock)
				
				prevClassifier0 = prevClassifier.get(block0[0])
				prob0 = [combineProbability(currentProbability.get(i)) for i in block0]
				averageProb0= np.mean(prob0)
				averageUncertainty0 = -averageProb0* np.log2(averageProb0) - (1- averageProb0)* np.log2(1- averageProb0)
				
				
				prevClassifier1 = prevClassifier.get(block1[0])
				prob1 = [combineProbability(currentProbability.get(i)) for i in block1]
				averageProb1= np.mean(prob1)
				averageUncertainty1 = -averageProb1* np.log2(averageProb1) - (1- averageProb1)* np.log2(1- averageProb1)
				
				prevClassifier2 = prevClassifier.get(block2[0])
				prob2 = [combineProbability(currentProbability.get(i)) for i in block2]
				averageProb2= np.mean(prob2)
				averageUncertainty2 = -averageProb2* np.log2(averageProb2) - (1- averageProb2)* np.log2(1- averageProb2)
				
				
				prevClassifier3 = prevClassifier.get(block3[0])							
				prob3 = [combineProbability(currentProbability.get(i)) for i in block3]
				averageProb3= np.mean(prob3)
				averageUncertainty3 = -averageProb3* np.log2(averageProb3) - (1- averageProb3)* np.log2(1- averageProb3)
				
				
				[nextBestClassifier0,deltaUncertainty0] = chooseNextBest(prevClassifier0[0],averageUncertainty0)
				[nextBestClassifier1,deltaUncertainty1] = chooseNextBest(prevClassifier1[0],averageUncertainty1)
				[nextBestClassifier2,deltaUncertainty2] = chooseNextBest(prevClassifier2[0],averageUncertainty2)
				[nextBestClassifier3,deltaUncertainty3] = chooseNextBest(prevClassifier3[0],averageUncertainty3)
				
				
				val0 = float(size0)*float(deltaUncertainty0)
				val1 = float(size1)*float(deltaUncertainty1)
				val2 = float(size2)*float(deltaUncertainty2)
				val3 = float(size3)*float(deltaUncertainty3)				
				
			else:
				
				sizeSubCollection = len(subCollection)
				sizeBlock = sizeSubCollection/4
				print sizeBlock
				if(sizeSubCollection > 200):
					#subset= random.choice(subCollection,sizeBlock, replace=False)
					block0 = subCollection[0:sizeBlock]
					#print block0
				else:				
					block0= subCollection
					
				size0 = float(len(block0))
				prevClassifier0 = prevClassifier.get(block0[0])
				uncertainty0= [currentUncertainty[i] for i in block0]
				averageUncertainty0 = np.mean(uncertainty0)
				[nextBestClassifier0,deltaUncertainty0] = chooseNextBest(prevClassifier0[0],averageUncertainty0)
				
				
				val0 = float(size0)*float(deltaUncertainty0)
				val1=0
				val2=0
				val3=0
				sizeMax = float(len(maxBlock))
				#if flag !=0 and sizeMax !=0:
					#valMax = float(sizeMax)*float(maxDeltaUncertainty)
				'''
				val0 = float(deltaUncertainty0)*cost(nextBestClassifier0)
				val1 = 0
				val2 = 0
				if maxNextBestClassifier !='':
					valMax = float(maxDeltaUncertainty)/cost(maxNextBestClassifier)
				'''
			#print 'deltaUncertainty0'
			#print 'minval for state:%s is :%f'%(state,min(val0,val1,val2))
			if(min(val0,val1,val2,val3) < valMax):
			#if(min(val0,val1,val2) < valMax):
				if val0 < val1 and val0 < val2 and val0<val3:
				#if val0 < val1 and val0 < val2:
					maxNextBestClassifier = nextBestClassifier0
					maxDeltaUncertainty = deltaUncertainty0
					sizeMax = size0
					maxBlock[:]=[]
					maxBlock = block0[:]
					valMax = val0
					print 'block0 selected for state %s'%(str)
				if val1 < val0 and val1 < val2 and val0<val3:
				#if val1 < val0 and val1 < val2:
					maxNextBestClassifier = nextBestClassifier1
					maxDeltaUncertainty = deltaUncertainty1
					sizeMax = size1
					maxBlock[:]=[]
					maxBlock = block1[:]
					valMax = val1
					print 'block1 selected for state %s'%(str)
				if val2 < val1 and val2 < val0 and val0<val3:
				#if val2 < val1 and val2 < val0:
					maxNextBestClassifier = nextBestClassifier2
					maxDeltaUncertainty = deltaUncertainty2
					sizeMax = size2
					maxBlock[:]=[]
					maxBlock = block2[:]
					valMax = val2
					print 'block2 selected for state %s'%(str)
				
				if val3 < val0 and val3 < val1 and val3<val2:
					maxNextBestClassifier = nextBestClassifier3
					maxDeltaUncertainty = deltaUncertainty3
					sizeMax = size3
					maxBlock[:]=[]
					maxBlock = block3[:]
					valMax = val3
					#print 'block3 selected for state %s'%(str)
				
				block0[:]=[]
				block1[:]=[]
				block2[:]=[]
				block3[:]=[]
			print 'valMax:%f selected for state %s'%(valMax,str)
		subCollection[:] = []
		featureVector[:]=[]
		
		
	return [maxNextBestClassifier,maxBlock]
	
def calculateBlockSize(budget, thinkTime,thinkTimePercent):
	costClassifier = float(cost('GNB')+cost('ET')+cost('RF')+cost('SVM'))/4
	print 'costClassifier:%f'%(costClassifier)
	print 'budget:%f'%(budget)
	print 'thinkTime:%f'%(thinkTime)
	thinkBudget = thinkTimePercent * budget
	numIteration = math.floor(float(thinkBudget)/thinkTime)
	blockSize = (1-thinkTimePercent)*thinkTime/(thinkTimePercent*costClassifier)
	return int(blockSize)
	
def cost(classifier):
	cost=0
	'''
	Cost in Muct Dataset
	gnb,et,rf,svm
	[0.029360,0.018030,0.020180,0.790850]

	'''
	#costSet = [0.029360,0.018030,0.020180,0.790850]
	#print 'classifier'
	#print classifier
	if classifier =='LDA':
		cost = 0.018175
	if classifier =='DT':
		cost = 0.029360
	if classifier =='GNB':
		cost = 0.018030
	if classifier =='RF':
		cost = 0.020180
	if classifier =='KNN':
		cost = 0.790850
		
	return cost
	
def combineProbability (probList):
	sumProb = 0
	countProb = 0
	flag = 0
	#print probList
	weights = determineWeights()
	
	for i in range(len(probList[0])):
		if probList[0][i]!=-1:
			sumProb = sumProb+weights[i]*probList[0][i]
			countProb = countProb+weights[i]
			flag = 1
	
	if flag ==1:
		return float(sumProb)/countProb
	else:
		return 0.5
		
	 

def convertToRocProb(prob,operator):
	#print 'In convertToRocProb method, %f'%prob
	#print operator
	clf_thresholds =[]
	clf_fpr =[]
	if operator.__name__== 'genderPredicate1' :
		clf_thresholds = lr_thresholds
		clf_fpr = lr_fprs
	if operator.__name__== 'genderPredicate2' :
		clf_thresholds = et_thresholds
		clf_fpr = et_fprs
	if operator.__name__== 'genderPredicate3' :
		clf_thresholds = rf_thresholds
		clf_fpr = rf_fprs
	if operator.__name__== 'genderPredicate4' :
		clf_thresholds = ab_thresholds
		clf_fpr = ab_fprs
	if operator.__name__== 'genderPredicate5' :
		clf_thresholds = svm_thresholds
		clf_fpr = svm_fprs
	
	thresholdIndex = (np.abs(clf_thresholds - prob)).argmin()
	rocProb = 1- clf_fpr[thresholdIndex]
	return rocProb
	
	
def findUncertainty(prob):
	if prob ==0 or prob == 1:
		return 0
	else :
		return (-prob* np.log2(prob) - (1- prob)* np.log2(1- prob))
	

def findQualityBackup(currentProbability):
	probabilitySet = []
	probDictionary = {}
	for i in range(len(dl)):
		''' For Noisy OR Model
		combinedProbability = 0
		productProbability =1
		'''
		
		sumProb = 0
		countProb = 0		
		flag = 0
		#combinedProbability = combineProbability(currentProbability[i])
		
		for p in currentProbability[i][0]:
			#print>>f1,'current probability: {}'.format(currentProbability[i][0])
			if p!=-1 :
				#productProbability = productProbability*(1-p)
				sumProb = sumProb+p
				countProb = countProb+1
				flag = 1
		if flag==0:
			combinedProbability = 0.5	
		else: 
			#combinedProbability = 1 - productProbability
			combinedProbability = float(sumProb)/countProb
		
		probabilitySet.append(combinedProbability)
		
		key = i
		value = combinedProbability
		probDictionary[key] = [value]
	#probabilitySet.sort(reverse=True)
	sortedProbSet = probabilitySet[:]
	sortedProbSet.sort(reverse=True)
	#x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
	sorted_x = sorted(probDictionary.items(), key=operator.itemgetter(1), reverse = True)
	
	#print 'sorted_x'
	#print sorted_x
	#print 'probabilitySet'
	#print probabilitySet
	#print("probDictionary: {} ".format(sorted_x))
	#print("sorted probabilitySet: {} ".format(sortedProbSet))
	totalSum = sum(sortedProbSet[0:len(sortedProbSet)])
	prevF1 = 0
	precision =0
	recall = 0
	f1Value = 0
	probThreshold = 0
	for i in range(len(sortedProbSet)):
		sizeOfAnswer = i
		sumOfProbability =0
		for j in range(i):
			#probThreshold = sorted_x.get(j)
			sumOfProbability = sumOfProbability + sortedProbSet[j]   #without dictionary
			#sumOfProbability = sumOfProbability + sorted_x.get(j)
		if i>0:
			precision = float(sumOfProbability)/(i)
			if totalSum >0:
				recall = float(sumOfProbability)/totalSum
			else:
				recall = 0
			if (precision+recall) >0:
				f1Value = 2*precision*recall/(precision+recall)
			else:
				f1Value = 0
			#f1Value = 2*float(sumOfProbability)/(totalSum +i)
		#print 'precision Value: %f'%(precision)
		#print 'recall Value: %f'%(recall)
		#print 'f1Value: %f'%(f1Value)
		if f1Value < prevF1 :
			break
		else:
			prevF1 = f1Value
	indexSorted = i
	#print 'indexSorted value : %d'%(indexSorted)
	
	returnedImages = []
	'''
	for j in range(indexSorted):
		#probValue = sortedProbSet[j]
		#indexProbabilitySet = [i for i, x in enumerate(probabilitySet) if x == probValue]
		indexProbabilitySet = [k for k in range(len(probabilitySet)) if probabilitySet[k] == probValue]
		#returnedImages.append(indexProbabilitySet)
		#sorted_x.get(j)
	'''
	
	for key in sorted_x[:indexSorted]:
		returnedImages.append(key[0])
	
	# this part is to eliminate objects which have not gone through any of the classifiers.
	eliminatedImage = []
	for k in range(len(returnedImages)):
		flag1 = 0
		for p in currentProbability[returnedImages[k]][0]:
				if p!=-1 :
					flag1 = 1
		if flag1==0:
			eliminatedImage.append(returnedImages[k])

	selectedImages = [x for x in returnedImages if x not in eliminatedImage]
			
	#return [prevF1,precision, recall]
	#return [prevF1,precision, recall, returnedImages]
	return [prevF1,precision, recall, selectedImages]


def findQuality(currentProbability):
	probabilitySet = []
	probDictionary = {}
	#t1_q=time.time()
	for i in range(len(dl)):		
		combinedProbability = combineProbability(currentProbability[i])
		probabilitySet.append(combinedProbability)

		value = combinedProbability
		probDictionary[i] = [value]
	
	#t2_q=time.time()
	#print 'time init 1: %f'%(t2_q - t1_q)
	
	#probabilitySet.sort(reverse=True)
	#t1_s=time.time()
	sortedProbSet = probabilitySet[:]
	sortedProbSet.sort(reverse=True)
	#t2_s=time.time()
	#print 'time sort 1: %f'%(t2_s - t1_s)
	
	#t1_th=time.time()
	totalSum = sum(sortedProbSet[0:len(sortedProbSet)])
	prevF1 = 0
	precision =0
	recall = 0
	f1Value = 0
	probThreshold = 0
	sumOfProbability =0
	
	for i in range(len(sortedProbSet)):
		sizeOfAnswer = i
		sumOfProbability = sumOfProbability + sortedProbSet[i]
		
		if i>0:
			precision = float(sumOfProbability)/(i)
			if totalSum >0:
				recall = float(sumOfProbability)/totalSum
			else:
				recall = 0 
			if (precision+recall) >0 :
				f1Value = 2*precision*recall/(precision+recall)
			else:
				f1Value = 0
			#f1Value = 2*float(sumOfProbability)/(totalSum +i)
		
		if f1Value < prevF1 :
			break
		else:
			prevF1 = f1Value
	indexSorted = i
	#print sortedProbSet
	probThreshold = sortedProbSet[indexSorted]
	#print 'indexSorted value : %d'%(indexSorted)
	#print 'threshold probability value : %f'%(probThreshold)
	
	#t2_th=time.time()
	#print 'time threshold 1: %f'%(t2_th - t1_th)
	
	returnedImages = []
	outsideImages = []
	
	
	#t1_ret=time.time()
	for i in range(len(probabilitySet)):
		if probabilitySet[i] > probThreshold:
			returnedImages.append(i)
		else:
			outsideImages.append(i)
			
	#t2_ret=time.time()
	#print 'time return 1: %f'%(t2_ret - t1_ret)
	
	return [prevF1,precision, recall, returnedImages, outsideImages]



def findNewQuality(currentProbability,index,newProbabilityValue2):
	# index is the index of the object whose prob is changed
	probabilitySet = []
	probDictionary = {}
	#print 'inside new quality function'
	for i in range(len(dl)):
		
		sumProb = 0
		countProb = 0		
		flag = 0
		'''
		if i != index:
			combinedProbability = combineProbability(currentProbability[i])
		else:
			combinedProbability = newProbabilityValue2
		'''
		'''
		for p in currentProbability[i][0]:
			#print>>f1,'current probability: {}'.format(currentProbability[i][0])
			if p!=-1 :
				#productProbability = productProbability*(1-p)
				sumProb = sumProb+p
				countProb = countProb+1
				flag = 1
		if flag==0:
			combinedProbability = 0.5	
		else: 
			#combinedProbability = 1 - productProbability
			combinedProbability = float(sumProb)/countProb
		'''
		if i != index:
			for p in currentProbability[i][0]:
			#print>>f1,'current probability: {}'.format(currentProbability[i][0])
				if p!=-1 :
					#productProbability = productProbability*(1-p)
					sumProb = sumProb+p
					countProb = countProb+1
					flag = 1
			if flag==0:
				combinedProbability = 0.5	
			else: 
				#combinedProbability = 1 - productProbability
				combinedProbability = float(sumProb)/countProb
		else:
			combinedProbability = newProbabilityValue2
		probabilitySet.append(combinedProbability)
		
		key = i
		value = combinedProbability
		probDictionary[key] = [value]
	#probabilitySet.sort(reverse=True)
	sortedProbSet = probabilitySet[:]
	sortedProbSet.sort(reverse=True)
	#x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
	sorted_x = sorted(probDictionary.items(), key=operator.itemgetter(1), reverse = True)
	
	#print 'sorted_x'
	#print sorted_x
	#print 'probabilitySet'
	#print probabilitySet
	#print("probDictionary: {} ".format(sorted_x))
	#print("sorted probabilitySet: {} ".format(sortedProbSet))
	totalSum = sum(sortedProbSet[0:len(sortedProbSet)])
	prevF1 = 0
	precision =0
	recall = 0
	f1Value = 0
	probThreshold = 0
	for i in range(len(sortedProbSet)):
		sizeOfAnswer = i
		sumOfProbability =0
		for j in range(i):
			#probThreshold = sorted_x.get(j)
			sumOfProbability = sumOfProbability + sortedProbSet[j]   #without dictionary
			#sumOfProbability = sumOfProbability + sorted_x.get(j)
		if i>0:
			precision = float(sumOfProbability)/(i)
			if totalSum >0:
				recall = float(sumOfProbability)/totalSum
			else:
				recall = 0
			if (precision+recall) >0:
				f1Value = 2*precision*recall/(precision+recall)
			else:
				f1Value = 0
			#f1Value = 2*float(sumOfProbability)/(totalSum +i)
		#print 'precision Value: %f'%(precision)
		#print 'recall Value: %f'%(recall)
		#print 'f1Value: %f'%(f1Value)
		if f1Value < prevF1 :
			break
		else:
			prevF1 = f1Value
	indexSorted = i
	#print 'indexSorted value : %d'%(indexSorted)
	
	returnedImages = []
	'''
	for j in range(indexSorted):
		#probValue = sortedProbSet[j]
		#indexProbabilitySet = [i for i, x in enumerate(probabilitySet) if x == probValue]
		indexProbabilitySet = [k for k in range(len(probabilitySet)) if probabilitySet[k] == probValue]
		#returnedImages.append(indexProbabilitySet)
		#sorted_x.get(j)
	'''
	
	for key in sorted_x[:indexSorted]:
		returnedImages.append(key[0])
	
	# this part is to eliminate objects which have not gone through any of the classifiers.
	eliminatedImage = []
	for k in range(len(returnedImages)):
		flag1 = 0
		for p in currentProbability[returnedImages[k]][0]:
				if p!=-1 :
					flag1 = 1
		if flag1==0:
			eliminatedImage.append(returnedImages[k])

	selectedImages = [x for x in returnedImages if x not in eliminatedImage]
			
	#return [prevF1,precision, recall]
	#return [prevF1,precision, recall, returnedImages]
	return [prevF1,precision, recall, selectedImages]


def determineWeights():
	#set = [0.85,0.92,0.92,0.89]
	set = [1,2,2,1]
	
	sumValue = sum(set)
	weightValues = [float(x)/sumValue for x in set]
	return weightValues
	
def findRealF1(imageList):
	sizeAnswer = len(imageList)	
	sizeDataset = len(nl)
	num_ones = (nl==1).sum()
	count = 0
	for i in imageList:
		if nl[i]==1:
			count+=1
	precision = float(count)/sizeAnswer
	recall = float(count)/num_ones
	
	if precision !=0 and recall !=0:
		f1Measure = (2*precision*recall)/(precision+recall)
	else:
		f1Measure = 0
	#print 'precision:%f, recall : %f, f1 measure: %f'%(precision,recall,f1Measure)
	return f1Measure
	
def findStates(outsideObjects,prevClassifier):
	stateCollection = []
	for i in range(len(outsideObjects)):
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = 'init'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '0'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '1'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '2'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '3'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '01'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '02'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '03'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '12'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '13'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '23'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==0 ):
			state = '012'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==0 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '013'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==0 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '023'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==0 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = '123'
		if(prevClassifier.get(outsideObjects[i])[0][0] ==1 and prevClassifier.get(outsideObjects[i])[0][1] ==1 and prevClassifier.get(outsideObjects[i])[0][2]==1 and prevClassifier.get(outsideObjects[i])[0][3]==1 ):
			state = 'NA'
		
		stateCollection.append(state)
	
	
	return stateCollection
	
	
	


def findUnprocessed(currentProbability):
	unprocessedImages = []
	for k in range(len(dl)):
		flag1 = 0
		for p in currentProbability[k][0]:
				if p!=-1 :
					flag1 = 1
		if flag1==0:
			unprocessedImages.append(k)

	return unprocessedImages
	

def adaptiveOrder8(timeBudget,epoch):
	
	f1 = open('queryTestGenderMuct8.txt','w+')

	#lr,et,rf,ab
	
	set = [genderPredicate6,genderPredicate1,genderPredicate3,genderPredicate7]
	
	print>>f1,'total load time :%f'%(t_load)

	print timeBudget
	outsideObjects=[]
	
	
	
	#blockList = [800]
	blockList = [200]
	#blockList = [600]
	
	
	
	executionPerformed = 0
	thinkTimeList = []
	executionTimeList = []
	candidateTimeList = []
	
	listOutsideObjectsLength= []
	#for percent in thinkPercentList:
	realF1List = []
	for block in blockList:
		#totalAllowedExecution = 1000
		executionPerformed = 0
		# The dictionary currentProbability stores the information about the output of previously ran classifiers. Suppose image 20 has gone through c2 and c3 and the output probability was 0.5 and 0.6. 
		# Then the hashmap element of image 20 will be as follows:  20: [0,0.5,0.6,0]. 
		t1 = time.time()
		currentProbability = {}
		for i in range(len(dl)):
			key = i
			value = [-1,-1,-1,-1]
			if key in currentProbability:
				currentProbability[key].append(value)
			else:
				currentProbability[key] = [value]			
		#print currentProbability
		
		
		# The dictionary prevClassifier stores the information about previously ran classifiers. Suppose image 20 has gone through c2 and c3. Then the hashmap element of image 20 will be as follows:  20: [0,1,1,0]. 
		# The bit vector corresponding to classifier 2 and classifier 3 are set.
		prevClassifier = {}
		for i in range(len(dl)):
			key = i
			value = [0,0,0,0]
			if key in prevClassifier:
				prevClassifier[key].append(value)
			else:
				prevClassifier[key] = [value]
				
		#print prevClassifier
		
		
		currentUncertainty = [0.99]*len(dl)
		t2 = time.time()
		
		print>>f1,'initial datastructure creation time :%f'%(t2-t1)
		
		currentF1measure = 0
		#print currentUncertainty
		operator = set[0]
		count = 0
		totalExecutionTime = 0
		totalThinkTime = 0
		
		totalCandidateTime = 0
		totalTripleGenTime = 0
		totalBenefitEstTime = 0
		totalSelectionTriplesTime = 0
		
		
		topKIndexes= [0]
		timeElapsed = 0
		timeList =[]
		f1List =[]
		blockSize = 1	
		executionTime = 0
		
		
		
		stepSize = epoch  #step size of 20 seconds. After every 20 seconds evaluate the quality
		currentTimeBound = epoch
	
		t11 = 0
		t12 = 0
		
		allObjects = list(range(0,len(dl)))
		
		while True:		
			#t11 = time.time()
			#for i in topKIndexes:
		
			if count ==0:
				t1 = time.time()
				operator = set[0]
				#dl_new =  pca.fit_transform(dl)
				#dl_new = np.array(dl_new)
				#print np.array(dl_new[0])
				#probX = operator(np.array(dl_new[0][0]))
				#probX = gender_dt_new.predict_proba([dl_new])
				#print probX
				for i in range(len(dl)):
					#print dl[i]
					#X_transformed = pca.fit_transform(dl[i])
					#print X_transformed
					#probValues = operator(X_transformed)
					#probValues = operator([dl_new[i]])
					#probValues = operator([pca.fit_transform(dl[i])])
					probValues = operator([dl[i]])
					#probValues = operator([dl[i][:850]])
					
					#probValues = probX[i,1]
					#print>>f1,probValues
					'''
					indexClf = set.index(operator)
					tempProb = currentProbability[i][0]
					tempProb[indexClf] = probValues[0]
					'''
					
					indexClf = set.index(operator)
					tempProb = currentProbability[i][0]
					#print probValues[0]
					if probValues[0] > 0.55:
						tempProb[indexClf] = probValues[0]+0.2
					else:
						tempProb[indexClf] = probValues[0]-0.3
					
					# setting the bit for the corresponding classifier
					tempClf = prevClassifier[i][0]
					tempClf[indexClf] = 1
					
					
					# calculating the current cobined probability
					combinedProbability = combineProbability(currentProbability[i])
					
					# using the combined probability value to calculate uncertainty
					uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
					currentUncertainty[i] = uncertainty
					
				t2 = time.time()
				executionTime = executionTime + (t2- t1)
				#set.remove(genderPredicate8)
	
				qualityOfAnswer = findQuality(currentProbability)
				#print 'returned images'
				#print qualityOfAnswer[3]
				#print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
				if len(qualityOfAnswer[3]) > 0:
					realF1 = findRealF1(qualityOfAnswer[3])
				else: 
					realF1 = 0
				#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
				f1measure = realF1
				timeList.append(0)
				f1List.append(f1measure)
				print>>f1,'initial function evaluation time :%f'%(t2-t1)
				#currentTimeBound = currentTimeBound + stepSize
				#print>>f1,'time bound completed:%d'%(currentTimeBound)
	
	
			#t11 = time.time()
			if count >0:
				
				
				for w in range(4):
					tempClfList = ['DT','GNB','RF','KNN']
					#print>>f1,"w = %d"%(w)	
					#print>>f1, tempClfList[w]					
					imageIndex = [item for item in topKIndexes if nextBestClassifier[item] == tempClfList[w]]
					if w!=4:
						operator = set[w]
					#print>>f1,operator
					images = [dl[k] for k in imageIndex]
					
					#print>>f1,"images to be run with this operator : {} ".format(imageIndex)
					
					if len(imageIndex) >0:
						#probValues = operator(images)
						######## Executing the function on all the objects ###########								
						for i1 in range(len(imageIndex)):
							t11 = time.time()		
							#probValues = operator(dl[i])
							#rocProb = probValues
							#rocProb = probValues[i1] //
							
							probValues = operator([images[i1]])
							rocProb = probValues
						
							#print>>f1,"i1 : %d"%(i1)
							#finding index of classifier
							indexClf = w						
							tempProb = currentProbability[imageIndex[i1]][0]
							tempProb[indexClf] = rocProb
							#print>>f1,"image : %d"%(imageIndex[i1])
							#print>>f1,"currentProbability: {}".format(currentProbability[imageIndex[i1]][0])
							
							#print currentProbability[imageIndex[i]]
							#if count !=0:
							#print nextBestClassifier[imageIndex[i]]
							
							# setting the bit for the corresponding classifier
							tempClfList = prevClassifier[imageIndex[i1]][0]
							tempClfList[indexClf] = 1
							#tempClfList2 = prevClassifier.get(outsideObjects[i1])
							#print>>f1,"prev classifier for image : %d"%(imageIndex[i1])
							#print>>f1,"prevClassifier: {}".format(prevClassifier[imageIndex[i1]][0])
							
							# calculating the current cobined probability
							combinedProbability = combineProbability(currentProbability[imageIndex[i1]])
							
						# using the combined probability value to calculate uncertainty
							uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
							currentUncertainty[imageIndex[i1]] = uncertainty
							
							
							t12 = time.time()
							totalExecutionTime = totalExecutionTime + (t12-t11)	
							timeElapsed = timeElapsed +(t12-t11)	
							
							if timeElapsed > currentTimeBound:
								qualityOfAnswer = findQuality(currentProbability)
				
								if len(qualityOfAnswer[3]) > 0:
									realF1 = findRealF1(qualityOfAnswer[3])
								else: 
									realF1 = 0
								#print>>f1,'real F1 : %f'%(realF1)
								#f1measure = qualityOfAnswer[0]
								f1measure = realF1
								timeList.append(timeElapsed)
								f1List.append(f1measure)
								#print>>f1,'F1 list : {}'.format(f1List)
				
								#print 'time bound completed:%d'%(currentTimeBound)	
								#print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
								#print 'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				
								currentTimeBound = currentTimeBound + stepSize
								break 
								
							if timeElapsed > timeBudget:
								break
							
							
					
					#executionTimeList.append(t12-t11)
					
					imageIndex[:]=[]
					images[:]=[]
					
			
			executionTimeList.append(totalExecutionTime)
					
				
			
			nextBestClassifier = [-1]*len(dl)
			deltaUncertainty = [0] *len(dl)
			benefitArray =[float(-10)]* len(dl) 
			topKIndexes = [0] * len(dl)
			
			newUncertaintyValue = 0 #initializing
			
			#### Think phase starts
			# calculating benefit of each objects. Benefit is measured in terms of improvement in f1 measure.
			t21 = time.time()
						
			t_candidate_start = time.time()
			
			# first determining the objects which are not in answer set
			qualityOfAnswer = findQuality(currentProbability)
			currentAnswerSet = qualityOfAnswer[3]
			#allObjects = list(range(0,len(dl)))
			
			#outsideObjects = allObjects
			
			
			if count !=0:
				outsideObjects = [x for x in allObjects if x not in currentAnswerSet]
				
			

			t_candidate_end = time.time()
			
			totalCandidateTime =  totalCandidateTime + (t_candidate_end - t_candidate_start)
			
			
			for j in range(len(outsideObjects)):
				
				t_triple_gen_start = time.time()
				[nextBestClassifier[outsideObjects[j]],deltaUncertainty[outsideObjects[j]]] = chooseNextBest(prevClassifier.get(outsideObjects[j])[0],currentUncertainty[outsideObjects[j]])	
				newUncertaintyValue = currentUncertainty[outsideObjects[j]]  + float(deltaUncertainty[outsideObjects[j]])
				t_triple_gen_end = time.time()
				
				totalTripleGenTime = totalTripleGenTime + (t_triple_gen_end - t_triple_gen_start)
				
				# Benefit Estimation Step	
				t_benefit_est_start = time.time()
				
				newProbabilityValue1 = convertEntropyToProb(newUncertaintyValue)
				#print 'newUncertaintyValue:%f'%(newUncertaintyValue)
				
				
				#finding index of classifier
				#indexTempProbClf = set.index(nextBestClassifier[j])
				if nextBestClassifier[outsideObjects[j]] == 'DT':
					nextBestClassifier[outsideObjects[j]] = 'NA'
				if nextBestClassifier[outsideObjects[j]] == 'GNB':
					indexTempProbClf = 1
				if nextBestClassifier[outsideObjects[j]] == 'RF':
					indexTempProbClf = 2
				if nextBestClassifier[outsideObjects[j]] == 'KNN':
					indexTempProbClf = 3
				
				# higher probability value	
				newProbabilityValue2 = 1 - newProbabilityValue1			
		
				#benefit is (pi * pi_new)/cost(i) 
				
				probability_i = combineProbability(currentProbability[outsideObjects[j]])
				if cost(nextBestClassifier[outsideObjects[j]]) != 0:
					benefit = float((probability_i*newProbabilityValue2)/float(cost(nextBestClassifier[outsideObjects[j]])))
					benefitArray[outsideObjects[j]] = benefit
				else:
					benefitArray[outsideObjects[j]] = -1
				
				t_benefit_est_end = time.time()
			
				totalBenefitEstTime =  totalBenefitEstTime + (t_benefit_est_end - t_benefit_est_start)
			
			
			
			
			t_triple_select_start = time.time()
			if len(outsideObjects) < blockSize :
				topKIndexes = outsideObjects
			else:
				topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			
			t_triple_select_end = time.time()
			
			totalSelectionTriplesTime = totalSelectionTriplesTime + (t_triple_select_end - t_triple_select_start)
	
			t22 = time.time()
			totalThinkTime = totalThinkTime + (t22-t21)
			 
			
			thinkTimeList.append(totalThinkTime)
			
		
			
			if(all(element==0 or element==-1 for element in benefitArray)):
				print 'nothing else to execute which has some benefit values'
				print>>f1,'nothing else to execute which has some benefit values'
				break
			if(len(outsideObjects)==0 and count !=0):
				break
			
			t2 = time.time()
			
			timeElapsed = totalExecutionTime + totalThinkTime
			
			print>>f1,'next images to be run: {}'.format(topKIndexes)
			classifierSet = [nextBestClassifier[item2] for item2 in topKIndexes]
			allClassifierSet = [nextBestClassifier[item3] for item3 in allObjects]
			benefitArray[:] =[]
			classifierSet[:] = []
			
			# block size is determined in this part.
			if count ==0:
				blockSize = block
				topKIndexes[:]= []
				#print 'blockSize: %d'%(blockSize)
			
			
			
			
			###### Time check########
			
			if timeElapsed > currentTimeBound:
				qualityOfAnswer = findQuality(currentProbability)
				
				if len(qualityOfAnswer[3]) > 0:
					realF1 = findRealF1(qualityOfAnswer[3])
				else: 
					realF1 = 0
				#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
				f1measure = realF1
				timeList.append(timeElapsed)
				f1List.append(f1measure)
				'''
				f1measure = qualityOfAnswer[0]
				timeList.append(timeElapsed)
				f1List.append(f1measure)
				'''
				#print 'time bound completed:%d'%(currentTimeBound)	
				#print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				#print 'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				
				currentTimeBound = currentTimeBound + stepSize
				
					
			

			if timeElapsed > timeBudget:
				break
			
			
			count=count+1
			
		plt.title('Quality vs Time Value')
		print>>f1,'total think time :%f'%(totalThinkTime)
		print>>f1,'total candidate selection time :%f'%(totalCandidateTime)
		print>>f1,'total triple generation time :%f'%(totalTripleGenTime)
		print>>f1,'total benefit estimation time:%f'%(totalBenefitEstTime)
		print>>f1,'total triple selection time :%f'%(totalSelectionTriplesTime)

		
		print>>f1,'total execution time :%f'%(totalExecutionTime)
		plt.ylabel('Quality')
		plt.xlabel('time')
		xValue = timeList
		yValue = f1List
		
	
	
	plt.plot(xValue, yValue,'g')
	plt.ylim([0, 1])
	plt.legend(loc="upper left")
	plt.savefig('plotQualityAdaptive8.png')
	#plt.show()
	plt.close()
	return [timeList,f1List]


def adaptiveOrder9(timeBudget):
	#1:Gaussian Naive Bayes
	#2:Extra Tree
	#3:Random Forest
	#4:Adaptive Boosting
	
	f1 = open('queryTestGenderMuct9.txt','w+')

	#lr,et,rf,ab
	
	set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
	
	
	
	print timeBudget
	outsideObjects=[]
	
	#thinkPercentList = [0.001,0.002,0.005,0.007,0.01]
	#thinkPercentList = [0.005,0.006]
	#thinkPercentList = [0.01,0.05,0.1,0.2]
	#thinkPercentList = [0.01]
	#thinkPercentList = [0.0005, 0.006]
	#blockList = [x*50 for x in range(1,10)]
	blockList = [50,100]
	executionPerformed = 0
	
	#for percent in thinkPercentList:
	realF1List = []
	for block in blockList:
		totalAllowedExecution = 3000
		executionPerformed = 0
		thinkTimeList = []
		#######This part is for choosing objects in the first iteration
		probArray = [0] * len(dl)
		operator1 = genderPredicate3
		probArray = operator1(dl)
		print>>f1,probArray
		print>>f1,'block size:%f'%(block)
		print 'block size:%f'%(block)
		lowestTopKProbs = heapq.nsmallest(block, range(len(probArray)), probArray.__getitem__)
		print>>f1,lowestTopKProbs
		print>>f1,probArray[lowestTopKProbs[0]]
		#print>>f1,probArray[lowestTopKProbs[block-1]]
		#print>>f1,probArray[lowestTopKProbs[block-50]]
		###################################################################
		
		# The dictionary currentProbability stores the information about the output of previously ran classifiers. Suppose image 20 has gone through c2 and c3 and the output probability was 0.5 and 0.6. 
		# Then the hashmap element of image 20 will be as follows:  20: [0,0.5,0.6,0]. 
		currentProbability = {}
		for i in range(len(dl)):
			key = i
			value = [-1,-1,-1,-1]
			if key in currentProbability:
				currentProbability[key].append(value)
			else:
				currentProbability[key] = [value]			
		#print currentProbability
		
		
		# The dictionary prevClassifier stores the information about previously ran classifiers. Suppose image 20 has gone through c2 and c3. Then the hashmap element of image 20 will be as follows:  20: [0,1,1,0]. 
		# The bit vector corresponding to classifier 2 and classifier 3 are set.
		prevClassifier = {}
		for i in range(len(dl)):
			key = i
			value = [0,0,0,0]
			if key in prevClassifier:
				prevClassifier[key].append(value)
			else:
				prevClassifier[key] = [value]
				
		#print prevClassifier
		
		#currentUncertainty list stores the information of current uncertainty of all the images.
		
		currentUncertainty = [1]*len(dl)
		currentF1measure = 0
		#print currentUncertainty
		operator = set[0]
		count = 0
		totalExecutionTime = 0
		totalThinkTime = 0
		topKIndexes= [0]
		timeElapsed = 0
		timeList =[]
		f1List =[]
		blockSize = 1	
		executionTime = 0
		stepSize = 20   #step size of 20 seconds. After every 20 seconds evaluate the quality
		currentTimeBound = 20
		t11 = 0
		t12 = 0
		
		t1 = time.time()
		while True:		
			#t11 = time.time()
			#for i in topKIndexes:
			if count !=0:
				tempClfString = ['GNB','ET','RF','SVM']
				for w in range(len(tempClfString)):
					imageIndex = [item for item in topKIndexes if nextBestClassifier[item] == tempClfString[w]]
					print>>f1,'Number of objects:%d, for classifier:%s'%(len(imageIndex),tempClfString[w])
					if w!=4:
						operator = set[w]
					'''
					else:
						if(len(imageIndex)==len(topKIndexes)):    #This implies no more images to be run
							break
					'''
					images = [dl[k] for k in imageIndex]
					if len(imageIndex)!=0:
						t11 = time.time()
						probValues = operator(images)
						t12 = time.time()
						totalExecutionTime = totalExecutionTime + (t12-t11)
						#if(totalExecutionTime +totalThinkTime)>timeBudget:
						#	break
						for i in range(len(imageIndex)):		
							#probValues = operator(dl[i])
							#rocProb = probValues
							rocProb = operator([dl[imageIndex[i]]])
							#rocProb = probValues[i]
							
							#finding index of classifier
							indexClf = set.index(operator)
							tempProb = currentProbability[imageIndex[i]][0]
							tempProb[indexClf] = rocProb
							#print currentProbability[imageIndex[i]]
							#if count !=0:
								#print nextBestClassifier[imageIndex[i]]
							
							# setting the bit for the corresponding classifier
							tempClf = prevClassifier[imageIndex[i]][0]
							tempClf[indexClf] = 1
							
							# calculating the current cobined probability
							combinedProbability = combineProbability(currentProbability[imageIndex[i]])
							
							# using the combined probability value to calculate uncertainty
							uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
							currentUncertainty[imageIndex[i]] = uncertainty						
					
					#imageIndex[:]=[]
					#images[:] =[]
					#topKIndexes[:] =[]
					#probValues[:]=[]
				'''
				if(len(imageIndex)==len(topKIndexes) and w ==4):    #This implies no more images to be run
							break
				'''
			nextBestClassifier = [0]*len(dl)
			deltaUncertainty = [0] *len(dl)
			benefitArray = [0] * len(dl)
			#topKIndexes = [0] * 10000 # initial block size.
			
			#currentTempProbability = copy.deepcopy(currentProbability)
			newUncertaintyValue = 0 #initializing
			
			# calculating benefit of each objects. Benefit is measured in terms of improvement in f1 measure.
			t21 = time.time()
			# first determining the objects which are not in answer set
			qualityOfAnswer = findQuality(currentProbability)
			currentAnswerSet = qualityOfAnswer[3]
			allObjects = list(range(0,len(dl)))
			outsideObjects = [x for x in allObjects if x not in currentAnswerSet]
			print>>f1,"outsideObjects : {} ".format(outsideObjects)
			
			for j in range(len(dl)):
				#print 'deciding for object %d'%(j)
				[nextBestClassifier[j],deltaUncertainty[j]] = chooseNextBest(prevClassifier.get(j)[0],currentUncertainty[j])	
				newUncertaintyValue = currentUncertainty[j]  + float(deltaUncertainty[j])
				newProbabilityValue1 = convertEntropyToProb(newUncertaintyValue)
				#print 'newUncertaintyValue:%f'%(newUncertaintyValue)
				
				
				#finding index of classifier
				#indexTempProbClf = set.index(nextBestClassifier[j])
				if nextBestClassifier[j] == 'GNB':
					indexTempProbClf = 0
				if nextBestClassifier[j] == 'ET':
					indexTempProbClf = 1
				if nextBestClassifier[j] == 'RF':
					indexTempProbClf = 2
				if nextBestClassifier[j] == 'SVM':
					indexTempProbClf = 3
				
				# higher probability value	
				newProbabilityValue2 = 1 - newProbabilityValue1			
		
				#benefit is  (pi * pi_new)/cost(i) 
				probability_i = combineProbability(currentProbability[j])
				#print 'probability_i: %f, new probability : %f, cost : %f'%(probability_i,newProbabilityValue2,cost(nextBestClassifier[j]))
				if cost(nextBestClassifier[j]) != 0:
					benefit = float((probability_i*newProbabilityValue2)/float(cost(nextBestClassifier[j])))
					benefitArray[j] = benefit
				else:
					benefitArray[j] = -1
				
			#seq = sorted(benefitArray)
			#print benefitArray
			#Ordering the objects based on deltaUncertainty Value
			#order = [seq.index(v) for v in benefitArray]
			#topIndex= benefitArray.index(max(benefitArray))
			
			topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			
			#if len(outsideObjects) < blockSize :
			#	topKIndexes = heapq.nlargest(len(outsideObjects), range(len(benefitArray)), benefitArray.__getitem__)
			#else:
				#topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			'''
			if len(outsideObjects) > (blockSize/4):
				topKIndexes = [x for x in topKIndexes if x not in currentAnswerSet]
			else:
				topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			'''
			#print 'top index:%d'%(topIndex)
			
			t22 = time.time()
			totalThinkTime = totalThinkTime + (t22-t21)
	
			thinkTimeList.append(t22-t21)
			#i=topIndex #next image to be run
			t2 = time.time()
			#timeElapsed = timeElapsed+(t2-t11)
			#timeElapsed = timeElapsed + totalExecutionTime+ totalThinkTime 
			timeElapsed = totalExecutionTime + totalThinkTime
			
			#timeList.append(timeElapsed)
			print>>f1,'benefit array: {}'.format(benefitArray)
			print>>f1,'next images to be run: {}'.format(topKIndexes)
			classifierSet = [nextBestClassifier[item2] for item2 in topKIndexes]
			print>>f1,'classifier set: {}'.format(classifierSet)
			#print>>f1,'next best classifier set: {}'.format(nextBestClassifier)
			
			
			print 'round %d completed'%(count)
			print 'time taken %f'%(timeElapsed)
			
			# block size is determined in this part.
			if count ==0:
				#thinkTime = t22-t21
				#thinkTimePercent = percent
				#blockSize = calculateBlockSize(timeBudget, thinkTime,thinkTimePercent)
				blockSize = block
				#topKIndexes[:]= []
				print 'blockSize: %d'%(blockSize)
			
			
			
			
			if timeElapsed > currentTimeBound:
				qualityOfAnswer = findQuality(currentProbability)
				f1measure = qualityOfAnswer[0]
				timeList.append(timeElapsed)
				f1List.append(f1measure)
				print 'time bound completed:%d'%(currentTimeBound)	
				print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				currentTimeBound = currentTimeBound + stepSize
			if timeElapsed > timeBudget:
				break
				
			'''	
			executionPerformed = executionPerformed + 	blockSize	
			if(executionPerformed>totalAllowedExecution):
				qualityOfAnswer = findQuality(currentProbability)
				print 'returned images'
				print qualityOfAnswer[3]
				if len(qualityOfAnswer[3]) > 0 :
					realF1 = findRealF1(qualityOfAnswer[3])
				else:
					realF1 = 0
				print 'real F1 : %f'%(realF1)
				
	
				#f1measure = qualityOfAnswer[0]
				f1measure = realF1
				f1measurePerAction = float(f1measure)/totalAllowedExecution
				timeList.append(timeElapsed)
				f1List.append(f1measurePerAction)
				realF1List.append(f1measure)
				
				print>>f1,'block size:%f'%(blockSize)
				print>>f1,'returned images: {}'.format(qualityOfAnswer[3])
				print>>f1,'length of answer set:%f'%len(qualityOfAnswer[3])
				unprocessedObjects = findUnprocessed(currentProbability)
				
				print>>f1,'unprocessed objects : {} '.format(unprocessedObjects)
				print>>f1,'length of unprocessed objects:%f'%(len(unprocessedObjects))				
				print>>f1,'think time list: {}'.format(thinkTimeList)
				
				break
				'''
			#if count >= 5000:
			#	break
			count=count+1
			
		plt.title('Quality vs Time Value')
		#print>>f1,'percent : %f'%(percent)
		#print>>f1,'block size : %f'%(block)
		#print>>f1,"f1 measures : {} ".format(realF1List)
		#print>>f1,'total think time :%f'%(totalThinkTime)
		#print>>f1,'total execution time :%f'%(totalExecutionTime)
	
		#xValue = timeList
		#yValue = f1List
		

	plt.ylabel('Quality')
	plt.xlabel('Block Size')
	xValue = blockList
	yValue = realF1List
	print>>f1,"x value : {} ".format(xValue)
	print>>f1,"y value : {} ".format(yValue)
	print "x value : {} ".format(xValue)
	print "y value : {} ".format(yValue)
	#yValue = f1measure
	#yValue = f1measurePerAction
	#labelValue = 'Adaptive algorithm(Think='+str(percent)+'%)'
	#labelValue = 'Adaptive algorithm(Block size='+str(block)+')'
	plt.plot(xValue, yValue,'g')
	plt.ylim([0, 1])
	plt.legend(loc="upper left")
	plt.savefig('plotQualityAdaptive9.png')
	#plt.show()
	plt.close()
	return [timeList,f1List]	
	

def adaptiveOrder10(timeBudget,epoch):
	# For measuring the think time and execution time. and also variation of quality with epoch time.
	#1:Gaussian Naive Bayes
	#2:Extra Tree
	#3:Random Forest
	#4:Adaptive Boosting
	
	f1 = open('queryTestGenderMuct10.txt','w+')

	#lr,et,rf,ab
	
	set = [genderPredicate6,genderPredicate1,genderPredicate3,genderPredicate7]
	#set = [genderPredicate17,genderPredicate1,genderPredicate3,genderPredicate7]
	
	print timeBudget
	outsideObjects=[]
	
	#blockList = [200] optimal for bigger dataset.
	#blockList = [100]
	#blockList = [600]
	#blockList = [400]
	#blockList = [200]
	blockList = [60]
	#blockList = [100]
	#blockList = [300]
	#blockList = [90]
	#blockList = [120]
	#blockList = [50]
	
	
	
	executionPerformed = 0
	thinkTimeList = []
	executionTimeList = []
	#for percent in thinkPercentList:
	realF1List = []
	for block in blockList:
		#totalAllowedExecution = 1000
		executionPerformed = 0
		# The dictionary currentProbability stores the information about the output of previously ran classifiers. Suppose image 20 has gone through c2 and c3 and the output probability was 0.5 and 0.6. 
		# Then the hashmap element of image 20 will be as follows:  20: [0,0.5,0.6,0]. 
		currentProbability = {}
		for i in range(len(dl)):
			key = i
			value = [-1,-1,-1,-1]
			if key in currentProbability:
				currentProbability[key].append(value)
			else:
				currentProbability[key] = [value]			
		#print currentProbability
		
		
		# The dictionary prevClassifier stores the information about previously ran classifiers. Suppose image 20 has gone through c2 and c3. Then the hashmap element of image 20 will be as follows:  20: [0,1,1,0]. 
		# The bit vector corresponding to classifier 2 and classifier 3 are set.
		prevClassifier = {}
		for i in range(len(dl)):
			key = i
			value = [0,0,0,0]
			if key in prevClassifier:
				prevClassifier[key].append(value)
			else:
				prevClassifier[key] = [value]
				
		#print prevClassifier
		
		#currentUncertainty list stores the information of current uncertainty of all the images.
		
		currentUncertainty = [0.99]*len(dl)
		currentF1measure = 0
		#print currentUncertainty
		operator = set[0]
		count = 0
		totalExecutionTime = 0
		totalThinkTime = 0
		topKIndexes= [0]
		timeElapsed = 0
		timeList =[]
		f1List =[]
		blockSize = 1	
		executionTime = 0
		
		stepSize =epoch  #step size of 20 seconds. After every 20 seconds evaluate the quality
		currentTimeBound = epoch
	
		t11 = 0
		t12 = 0
		
		pca = PCA()
		'''
		pca = PCA()
	
		clf_uncalibrated = tree.DecisionTreeClassifier()
	
		X_transformed = pca.fit_transform(trainX)
		clf_uncalibrated = clf_uncalibrated.fit(X_transformed,trainY)
		'''
	
		while True:		
			#t11 = time.time()
			#for i in topKIndexes:
		
			if count ==0:
				t1 = time.time()
				operator = set[0]
				#dl_new =  pca.fit_transform(dl)
				#dl_new = np.array(dl_new)
				#print np.array(dl_new[0])
				#probX = operator(np.array(dl_new[0][0]))
				#probX = gender_dt_new.predict_proba([dl_new])
				#print probX
				for i in range(len(dl)):
					#print dl[i]
					#X_transformed = pca.fit_transform(dl[i])
					#print X_transformed
					#probValues = operator(X_transformed)
					#probValues = operator([dl_new[i]])
					#probValues = operator([pca.fit_transform(dl[i])])
					probValues = operator([dl[i]])
					#probValues = operator([dl[i][:850]])
					
					#probValues = probX[i,1]
					#print>>f1,probValues
					'''
					indexClf = set.index(operator)
					tempProb = currentProbability[i][0]
					tempProb[indexClf] = probValues[0]
					'''
					
					indexClf = set.index(operator)
					tempProb = currentProbability[i][0]
					#print probValues[0]
					if probValues[0] > 0.55:
						tempProb[indexClf] = probValues[0]+0.2
					else:
						tempProb[indexClf] = probValues[0]-0.3
					
					# setting the bit for the corresponding classifier
					tempClf = prevClassifier[i][0]
					tempClf[indexClf] = 1
					
					
					# calculating the current cobined probability
					combinedProbability = combineProbability(currentProbability[i])
					
					# using the combined probability value to calculate uncertainty
					uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
					currentUncertainty[i] = uncertainty
					
				t2 = time.time()
				executionTime = executionTime + (t2- t1)
				#set.remove(genderPredicate8)
	
				qualityOfAnswer = findQuality(currentProbability)
				#print 'returned images'
				#print qualityOfAnswer[3]
				#print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
				if len(qualityOfAnswer[3]) > 0:
					realF1 = findRealF1(qualityOfAnswer[3])
				else: 
					realF1 = 0
				#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
				f1measure = realF1
				timeList.append(0)
				f1List.append(f1measure)
				#currentTimeBound = currentTimeBound + stepSize
				#print>>f1,'time bound completed:%d'%(currentTimeBound)
	
	
			#t11 = time.time()
			if count >0:
				
				
				for w in range(4):
					tempClfList = ['DT','GNB','RF','KNN']
					#print>>f1,"w = %d"%(w)	
					#print>>f1, tempClfList[w]					
					imageIndex = [item for item in topKIndexes if nextBestClassifier[item] == tempClfList[w]]
					if w!=4:
						operator = set[w]
					#print>>f1,operator
					images = [dl[k] for k in imageIndex]
					
					#print>>f1,"images to be run with this operator : {} ".format(imageIndex)
					
					if len(imageIndex) >0:
						#probValues = operator(images)
					######## Executing the function on all the objects ###########								
						for i1 in range(len(imageIndex)):
							t11 = time.time()		
							#probValues = operator(dl[i])
							#rocProb = probValues
							#rocProb = probValues[i1] //
							
							probValues = operator([images[i1]])
							rocProb = probValues
						
							#print>>f1,"i1 : %d"%(i1)
							#finding index of classifier
							indexClf = w						
							tempProb = currentProbability[imageIndex[i1]][0]
							tempProb[indexClf] = rocProb
							#print>>f1,"image : %d"%(imageIndex[i1])
							#print>>f1,"currentProbability: {}".format(currentProbability[imageIndex[i1]][0])
							
							#print currentProbability[imageIndex[i]]
							#if count !=0:
							#print nextBestClassifier[imageIndex[i]]
							
							# setting the bit for the corresponding classifier
							tempClfList = prevClassifier[imageIndex[i1]][0]
							tempClfList[indexClf] = 1
							#tempClfList2 = prevClassifier.get(outsideObjects[i1])
							#print>>f1,"prev classifier for image : %d"%(imageIndex[i1])
							#print>>f1,"prevClassifier: {}".format(prevClassifier[imageIndex[i1]][0])
							
							# calculating the current cobined probability
							combinedProbability = combineProbability(currentProbability[imageIndex[i1]])
							
						# using the combined probability value to calculate uncertainty
							uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
							currentUncertainty[imageIndex[i1]] = uncertainty
							
							
							t12 = time.time()
							totalExecutionTime = totalExecutionTime + (t12-t11)	
							timeElapsed = timeElapsed +(t12-t11)	
							
							if timeElapsed > currentTimeBound:
								qualityOfAnswer = findQuality(currentProbability)
				
								if len(qualityOfAnswer[3]) > 0:
									realF1 = findRealF1(qualityOfAnswer[3])
								else: 
									realF1 = 0
								#print>>f1,'real F1 : %f'%(realF1)
								#f1measure = qualityOfAnswer[0]
								f1measure = realF1
								timeList.append(timeElapsed)
								f1List.append(f1measure)
								#print>>f1,'F1 list : {}'.format(f1List)
				
								#print 'time bound completed:%d'%(currentTimeBound)	
								#print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
								#print 'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				
								currentTimeBound = currentTimeBound + stepSize
								break
								
							if timeElapsed > timeBudget:
								break
							
							
					
					#executionTimeList.append(t12-t11)
					
					imageIndex[:]=[]
					images[:]=[]
					#print>>f1,"Outside of inner for loop"
					#continue
							
				#print>>f1,"Finished executing four functions"
					#imageIndex[:]=[]
					#images[:] =[]
					#topKIndexes[:]=[]
					#probValues[:]=[]
			'''
			t12 = time.time()
			totalExecutionTime = totalExecutionTime + (t12-t11)	
			executionTimeList.append(t12-t11)
			'''
			executionTimeList.append(totalExecutionTime)
					
				
			
			nextBestClassifier = [-1]*len(dl)
			deltaUncertainty = [0] *len(dl)
			benefitArray =[float(-10)]* len(dl) 
			topKIndexes = [0] * len(dl) # initial block size.
			
			#currentTempProbability = copy.deepcopy(currentProbability)
			newUncertaintyValue = 0 #initializing
			
			#### Think phase starts
			# calculating benefit of each objects. Benefit is measured in terms of improvement in f1 measure.
			t21 = time.time()
			# first determining the objects which are not in answer set
			qualityOfAnswer = findQuality(currentProbability)
			currentAnswerSet = qualityOfAnswer[3]
			allObjects = list(range(0,len(dl)))
			
			#outsideObjects = allObjects
			
			### Uncomment this part for choosing objects from outside of the answer set.
			
			if count !=0:
				outsideObjects = [x for x in allObjects if x not in currentAnswerSet]
			
			
			#print>>f1,"inside objects : {} ".format(currentAnswerSet)
			#print>>f1,"length of inside objects : %f"%len(currentAnswerSet)
			#stateListInside =[]
			#stateListInside = findStates(currentAnswerSet,prevClassifier)
			
			#print>>f1,"state of inside objects: {}".format(stateListInside)
			
			#print>>f1,"outsideObjects : {} ".format(outsideObjects)
			#print>>f1,"length of outsideObjects : %f"%len(outsideObjects)
			#stateListOutside =[]
			#stateListOutside = findStates(outsideObjects,prevClassifier)
			#print>>f1,"state of outside objects: {}".format(stateListOutside)
			
			
			if(len(outsideObjects)==0 and count !=0):
				break
			
			
			
			for j in range(len(outsideObjects)):
				#print>>f1,'deciding for object %d'%(outsideObjects[j])
				#print>>f1,"currentUncertainty: {}".format(currentUncertainty)
				[nextBestClassifier[outsideObjects[j]],deltaUncertainty[outsideObjects[j]]] = chooseNextBest(prevClassifier.get(outsideObjects[j])[0],currentUncertainty[outsideObjects[j]])	
				newUncertaintyValue = currentUncertainty[outsideObjects[j]]  + float(deltaUncertainty[outsideObjects[j]])
				newProbabilityValue1 = convertEntropyToProb(newUncertaintyValue)
				#print 'newUncertaintyValue:%f'%(newUncertaintyValue)
				
				
				#finding index of classifier
				#indexTempProbClf = set.index(nextBestClassifier[j])
				if nextBestClassifier[outsideObjects[j]] == 'DT':
					nextBestClassifier[outsideObjects[j]] = 'NA'
				if nextBestClassifier[outsideObjects[j]] == 'GNB':
					indexTempProbClf = 1
				if nextBestClassifier[outsideObjects[j]] == 'RF':
					indexTempProbClf = 2
				if nextBestClassifier[outsideObjects[j]] == 'KNN':
					indexTempProbClf = 3
				
				# higher probability value	
				newProbabilityValue2 = 1 - newProbabilityValue1			
		
				#benefit is  (pi * pi_new)/cost(i) 
				probability_i = combineProbability(currentProbability[outsideObjects[j]])
				#print 'probability_i: %f, new probability : %f, cost : %f'%(probability_i,newProbabilityValue2,cost(nextBestClassifier[j]))
				if cost(nextBestClassifier[outsideObjects[j]]) != 0:
					benefit = float((probability_i*newProbabilityValue2)/float(cost(nextBestClassifier[outsideObjects[j]])))
					benefitArray[outsideObjects[j]] = benefit
				else:
					benefitArray[outsideObjects[j]] = -1
				
			#seq = sorted(benefitArray)
			#print benefitArray
			#Ordering the objects based on deltaUncertainty Value
			#order = [seq.index(v) for v in benefitArray]
			#topIndex= benefitArray.index(max(benefitArray))
			
			#topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			
			
			if len(outsideObjects) < blockSize :
				topKIndexes = outsideObjects
				#topKIndexes = heapq.nlargest(len(outsideObjects), range(len(outsideObjects)), benefitArray.__getitem__)
			else:
				topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			
			'''
			if len(outsideObjects) > (blockSize/4):
				topKIndexes = [x for x in topKIndexes if x not in currentAnswerSet]
			else:
				topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			'''
			#print 'top index:%d'%(topIndex)
			
			t22 = time.time()
			totalThinkTime = totalThinkTime + (t22-t21)
			thinkTimeList.append(t22-t21)
			
			stateListInside =[]
			stateListInside = findStates(currentAnswerSet,prevClassifier)
			
			stateListOutside =[]
			stateListOutside = findStates(outsideObjects,prevClassifier)
			
			
			
			print>>f1,"inside objects : {} ".format(currentAnswerSet)
			print>>f1,"state of inside objects: {}".format(stateListInside)
			print>>f1,"outsideObjects : {} ".format(outsideObjects)
			print>>f1,"state of outside objects: {}".format(stateListOutside)
			print>>f1,"think time list: {}".format(thinkTimeList)
			
			'''
			if(all(element==0 or element==-1 for element in benefitArray) and count >20):
				break
			'''
			#i=topIndex #next image to be run
			t2 = time.time()
			
			timeElapsed = totalExecutionTime + totalThinkTime
			
			#timeList.append(timeElapsed)
			#print 'next images to be run'
			#print topKIndexes
			
			#print>>f1,'benefit array: {}'.format(benefitArray)
			#print>>f1,'next images to be run: {}'.format(topKIndexes)
			classifierSet = [nextBestClassifier[item2] for item2 in topKIndexes]
			allClassifierSet = [nextBestClassifier[item3] for item3 in allObjects]
			'''
			if(all(element=='NA' for element in classifierSet) and count > 20):
				break
			'''
			if(all(element=='NA' for element in classifierSet)):
				topKIndexes = allObjects
				#break
			if(all(element=='NA' for element in allClassifierSet) and count > 4):
				break
			
			#print>>f1,'classifier set: {}'.format(classifierSet)
			benefitArray[:] =[]
			classifierSet[:] = []
			
			#print 'round %d completed'%(count)
			#print 'time taken %f'%(timeElapsed)
			
			# block size is determined in this part.
			if count ==0:
				blockSize = block
				topKIndexes[:]= []
				#print 'blockSize: %d'%(blockSize)
			
			
			
			
			###### Time check########
			
			if timeElapsed > currentTimeBound:
				qualityOfAnswer = findQuality(currentProbability)
				
				if len(qualityOfAnswer[3]) > 0:
					realF1 = findRealF1(qualityOfAnswer[3])
				else: 
					realF1 = 0
				#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
				f1measure = realF1
				timeList.append(timeElapsed)
				f1List.append(f1measure)
				'''
				f1measure = qualityOfAnswer[0]
				timeList.append(timeElapsed)
				f1List.append(f1measure)
				'''
				#print 'time bound completed:%d'%(currentTimeBound)	
				#print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				#print 'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				
				currentTimeBound = currentTimeBound + stepSize
				
					
			

			if timeElapsed > timeBudget:
				break
			
			
			'''
			executionPerformed = executionPerformed + 	blockSize	
			if(executionPerformed>totalAllowedExecution):
				qualityOfAnswer = findQuality(currentProbability)
				print 'returned images'
				print qualityOfAnswer[3]
				if len(qualityOfAnswer[3]) > 0 :
					realF1 = findRealF1(qualityOfAnswer[3])
				else:
					realF1 = 0
				print 'real F1 : %f'%(realF1)
				
	
				#f1measure = qualityOfAnswer[0]
				f1measure = realF1
				f1measurePerAction = float(f1measure)/totalAllowedExecution
				timeList.append(timeElapsed)
				f1List.append(f1measurePerAction)
				realF1List.append(f1measure)
				
				print>>f1,'block size:%f'%(blockSize)
				print>>f1,'returned images: {}'.format(qualityOfAnswer[3])
				print>>f1,'length of answer set:%f'%len(qualityOfAnswer[3])
				unprocessedObjects = findUnprocessed(currentProbability)
				print>>f1,'unprocessed objects : {} '.format(unprocessedObjects)
				print>>f1,'length of unprocessed objects:%f'%(len(unprocessedObjects))
				print>>f1,'think time list: {}'.format(thinkTimeList)
				print>>f1,'execution time list: {}'.format(executionTimeList)
				
				break
				'''
			#if count >= 5000:
			#	break
			count=count+1
			
		'''	
		plt.title('Quality vs Time Value')
		#print>>f1,'percent : %f'%(percent)
		#print>>f1,'block size : %f'%(block)
		#print>>f1,"f1 measures : {} ".format(realF1List)
		#print>>f1,'total think time :%f'%(totalThinkTime)
		#print>>f1,'total execution time :%f'%(totalExecutionTime)
		plt.ylabel('Quality')
		plt.xlabel('time')
		xValue = timeList
		yValue = f1List
		#print>>f1,"x value : {} ".format(xValue)
		#print>>f1,"y value : {} ".format(yValue)
		#print "x value : {} ".format(xValue)
		#print "y value : {} ".format(yValue)
	
		plt.plot(xValue, yValue,'b')
		plt.ylim([0, 1])
		plt.legend(loc="upper left")
		#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        #   ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
		plt.savefig('plotQualityAdaptive8'+str(block)+'.eps',format = 'eps')
		plt.title('Quality vs Time for block size = '+str(block))
		'''
		#plt.show()
		#plt.close()
		#xValue = timeList
		#yValue = f1List
		
	'''
	plt.ylabel('Quality')
	plt.xlabel('Block Size')
	xValue = blockList
	yValue = realF1List
	print>>f1,"x value : {} ".format(xValue)
	print>>f1,"y value : {} ".format(yValue)
	print "x value : {} ".format(xValue)
	print "y value : {} ".format(yValue)
	'''
	'''
	plt.ylabel('Quality')
	plt.xlabel('time')
	xValue = timeList
	yValue = f1List
	print>>f1,"x value : {} ".format(xValue)
	print>>f1,"y value : {} ".format(yValue)
	print "x value : {} ".format(xValue)
	print "y value : {} ".format(yValue)
	'''
	#yValue = f1measure
	#yValue = f1measurePerAction
	#labelValue = 'Adaptive algorithm(Think='+str(percent)+'%)'
	#labelValue = 'Adaptive algorithm(Block size='+str(block)+')'
	'''
	#uncomment for plotting
	plt.plot(xValue, yValue,'g')
	plt.ylim([0, 1])
	plt.legend(loc="upper left")
	#plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #       ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
	plt.savefig('plotQualityAdaptive8_comparison.eps',format = 'eps')
	plt.savefig('plotQualityAdaptive8_comparison.png')
	#plt.show()
	plt.close()
	'''
	return [timeList,f1List]



	
	
	
def adaptiveOrder14(timeBudget):
	#1:Gaussian Naive Bayes
	#2:Extra Tree
	#3:Random Forest
	#4:Adaptive Boosting
	# This one we examine the stopping criteria of our approach.
	
	f1 = open('queryTestGenderMuct14.txt','w+')

	#lr,et,rf,ab
	
	set = [genderPredicate6,genderPredicate1,genderPredicate3,genderPredicate7]
	#set = [genderPredicate17,genderPredicate1,genderPredicate3,genderPredicate7]
	
	print timeBudget
	outsideObjects=[]
	
	#thinkPercentList = [0.001,0.002,0.005,0.007,0.01]
	#thinkPercentList = [0.005,0.006]
	#thinkPercentList = [0.01,0.05,0.1,0.2]
	#thinkPercentList = [0.01]
	#thinkPercentList = [0.0005, 0.006]
	#blockList = [1,x * 50 for x in range(1,10)]
	#blockList = [10,100,200,500]
	#blockList = [10,20,50,100,200,500,600]
	#blockList = [100,200]
	#blockList = [200] optimal for bigger dataset.
	#blockList = [100]
	#blockList = [600]
	#blockList = [400]
	#blockList = [200]
	#blockList = [60]
	#blockList = [120]
	#blockList = [30]
	#blockList = [90]
	#blockList = [60]
	
	#blockList = [200]
	blockList = [30]
	#blockList = [600]
	
	
	
	executionPerformed = 0
	thinkTimeList = []
	executionTimeList = []
	listOutsideObjectsLength= []
	#for percent in thinkPercentList:
	realF1List = []
	for block in blockList:
		#totalAllowedExecution = 1000
		executionPerformed = 0
		# The dictionary currentProbability stores the information about the output of previously ran classifiers. Suppose image 20 has gone through c2 and c3 and the output probability was 0.5 and 0.6. 
		# Then the hashmap element of image 20 will be as follows:  20: [0,0.5,0.6,0]. 
		currentProbability = {}
		for i in range(len(dl)):
			key = i
			value = [-1,-1,-1,-1]
			if key in currentProbability:
				currentProbability[key].append(value)
			else:
				currentProbability[key] = [value]			
		#print currentProbability
		
		
		# The dictionary prevClassifier stores the information about previously ran classifiers. Suppose image 20 has gone through c2 and c3. Then the hashmap element of image 20 will be as follows:  20: [0,1,1,0]. 
		# The bit vector corresponding to classifier 2 and classifier 3 are set.
		prevClassifier = {}
		for i in range(len(dl)):
			key = i
			value = [0,0,0,0]
			if key in prevClassifier:
				prevClassifier[key].append(value)
			else:
				prevClassifier[key] = [value]
				
		#print prevClassifier
		
		#currentUncertainty list stores the information of current uncertainty of all the images.
		
		currentUncertainty = [0.99]*len(dl)
		currentF1measure = 0
		#print currentUncertainty
		operator = set[0]
		count = 0
		totalExecutionTime = 0
		totalThinkTime = 0
		topKIndexes= [0]
		timeElapsed = 0
		timeList =[]
		f1List =[]
		blockSize = 1	
		executionTime = 0
		
		stepSize =4  #step size of 20 seconds. After every 20 seconds evaluate the quality
		currentTimeBound = 4
	
		t11 = 0
		t12 = 0
		
		pca = PCA()
		'''
		pca = PCA()
	
		clf_uncalibrated = tree.DecisionTreeClassifier()
	
		X_transformed = pca.fit_transform(trainX)
		clf_uncalibrated = clf_uncalibrated.fit(X_transformed,trainY)
		'''
	
		while True:		
			#t11 = time.time()
			#for i in topKIndexes:
		
			if count ==0:
				t1 = time.time()
				operator = set[0]
				#dl_new =  pca.fit_transform(dl)
				#dl_new = np.array(dl_new)
				#print np.array(dl_new[0])
				#probX = operator(np.array(dl_new[0][0]))
				#probX = gender_dt_new.predict_proba([dl_new])
				#print probX
				for i in range(len(dl)):
					#print dl[i]
					#X_transformed = pca.fit_transform(dl[i])
					#print X_transformed
					#probValues = operator(X_transformed)
					#probValues = operator([dl_new[i]])
					#probValues = operator([pca.fit_transform(dl[i])])
					probValues = operator([dl[i]])
					#probValues = operator([dl[i][:850]])
					
					#probValues = probX[i,1]
					#print>>f1,probValues
					'''
					indexClf = set.index(operator)
					tempProb = currentProbability[i][0]
					tempProb[indexClf] = probValues[0]
					'''
					
					indexClf = set.index(operator)
					tempProb = currentProbability[i][0]
					#print probValues[0]
					if probValues[0] > 0.55:
						tempProb[indexClf] = probValues[0]+0.2
					else:
						tempProb[indexClf] = probValues[0]-0.3
					
					# setting the bit for the corresponding classifier
					tempClf = prevClassifier[i][0]
					tempClf[indexClf] = 1
					
					
					# calculating the current cobined probability
					combinedProbability = combineProbability(currentProbability[i])
					
					# using the combined probability value to calculate uncertainty
					uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
					currentUncertainty[i] = uncertainty
					
				t2 = time.time()
				executionTime = executionTime + (t2- t1)
				#set.remove(genderPredicate8)
	
				qualityOfAnswer = findQuality(currentProbability)
				#print 'returned images'
				#print qualityOfAnswer[3]
				#print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
				if len(qualityOfAnswer[3]) > 0:
					realF1 = findRealF1(qualityOfAnswer[3])
				else: 
					realF1 = 0
				#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
				f1measure = realF1
				timeList.append(0)
				f1List.append(f1measure)
				#currentTimeBound = currentTimeBound + stepSize
				#print>>f1,'time bound completed:%d'%(currentTimeBound)
	
	
			#t11 = time.time()
			if count >0:
				
				
				for w in range(4):
					tempClfList = ['DT','GNB','RF','KNN']
					#print>>f1,"w = %d"%(w)	
					#print>>f1, tempClfList[w]					
					imageIndex = [item for item in topKIndexes if nextBestClassifier[item] == tempClfList[w]]
					if w!=4:
						operator = set[w]
					#print>>f1,operator
					images = [dl[k] for k in imageIndex]
					
					#print>>f1,"images to be run with this operator : {} ".format(imageIndex)
					
					if len(imageIndex) >0:
						#probValues = operator(images)
					######## Executing the function on all the objects ###########								
						for i1 in range(len(imageIndex)):
							t11 = time.time()		
							#probValues = operator(dl[i])
							#rocProb = probValues
							#rocProb = probValues[i1] //
							
							probValues = operator([images[i1]])
							rocProb = probValues
						
							#print>>f1,"i1 : %d"%(i1)
							#finding index of classifier
							indexClf = w						
							tempProb = currentProbability[imageIndex[i1]][0]
							tempProb[indexClf] = rocProb
							#print>>f1,"image : %d"%(imageIndex[i1])
							#print>>f1,"currentProbability: {}".format(currentProbability[imageIndex[i1]][0])
							
							#print currentProbability[imageIndex[i]]
							#if count !=0:
							#print nextBestClassifier[imageIndex[i]]
							
							# setting the bit for the corresponding classifier
							tempClfList = prevClassifier[imageIndex[i1]][0]
							tempClfList[indexClf] = 1
							#tempClfList2 = prevClassifier.get(outsideObjects[i1])
							#print>>f1,"prev classifier for image : %d"%(imageIndex[i1])
							#print>>f1,"prevClassifier: {}".format(prevClassifier[imageIndex[i1]][0])
							
							# calculating the current cobined probability
							combinedProbability = combineProbability(currentProbability[imageIndex[i1]])
							
						# using the combined probability value to calculate uncertainty
							uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
							currentUncertainty[imageIndex[i1]] = uncertainty
							
							
							t12 = time.time()
							totalExecutionTime = totalExecutionTime + (t12-t11)	
							timeElapsed = timeElapsed +(t12-t11)	
							
							if timeElapsed > currentTimeBound:
								qualityOfAnswer = findQuality(currentProbability)
				
								if len(qualityOfAnswer[3]) > 0:
									realF1 = findRealF1(qualityOfAnswer[3])
								else: 
									realF1 = 0
								#print>>f1,'real F1 : %f'%(realF1)
								#f1measure = qualityOfAnswer[0]
								f1measure = realF1
								timeList.append(timeElapsed)
								f1List.append(f1measure)
								#print>>f1,'F1 list : {}'.format(f1List)
				
								#print 'time bound completed:%d'%(currentTimeBound)	
								#print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
								#print 'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				
								currentTimeBound = currentTimeBound + stepSize
								break 
								
							if timeElapsed > timeBudget:
								break
							
							
					
					#executionTimeList.append(t12-t11)
					
					imageIndex[:]=[]
					images[:]=[]
					#print>>f1,"Outside of inner for loop"
					#continue
							
				#print>>f1,"Finished executing four functions"
					#imageIndex[:]=[]
					#images[:] =[]
					#topKIndexes[:]=[]
					#probValues[:]=[]
			'''
			t12 = time.time()
			totalExecutionTime = totalExecutionTime + (t12-t11)	
			executionTimeList.append(t12-t11)
			'''
			executionTimeList.append(totalExecutionTime)
					
				
			
			nextBestClassifier = [-1]*len(dl)
			deltaUncertainty = [0] *len(dl)
			benefitArray =[float(-10)]* len(dl) 
			benefitTopK = [float(-10)]* len(dl)
			topKIndexes = [0] * len(dl) # initial block size.
			
			#currentTempProbability = copy.deepcopy(currentProbability)
			newUncertaintyValue = 0 #initializing
			
			#### Think phase starts
			# calculating benefit of each objects. Benefit is measured in terms of improvement in f1 measure.
			t21 = time.time()
			# first determining the objects which are not in answer set
			qualityOfAnswer = findQuality(currentProbability)
			currentAnswerSet = qualityOfAnswer[3]
			allObjects = list(range(0,len(dl)))
			
			#outsideObjects = allObjects
			
			### Uncomment this part for choosing objects from outside of the answer set.
			
			if count !=0:
				outsideObjects = [x for x in allObjects if x not in currentAnswerSet]
			
			
			
			#print>>f1,"inside objects : {} ".format(currentAnswerSet)
			#print>>f1,"length of inside objects : %f"%len(currentAnswerSet)
			stateListInside =[]
			stateListInside = findStates(currentAnswerSet,prevClassifier)
			
			#print>>f1,"state of inside objects: {}".format(stateListInside)
			
			#print>>f1,"outsideObjects : {} ".format(outsideObjects)
			#print>>f1,"length of outsideObjects : %f"%len(outsideObjects)
			#listOutsideObjectsLength= []
			stateListOutside =[]
			stateListOutside = findStates(outsideObjects,prevClassifier)
			#print>>f1,"state of outside objects: {}".format(stateListOutside)
			
			
			
			
			
			#print>>f1,"outsideObjects : {} ".format(outsideObjects)
			#print 'count=%d'%(count)
			
			for j in range(len(outsideObjects)):
				#print>>f1,'deciding for object %d'%(outsideObjects[j])
				#print>>f1,"currentUncertainty: {}".format(currentUncertainty)
				[nextBestClassifier[outsideObjects[j]],deltaUncertainty[outsideObjects[j]]] = chooseNextBest(prevClassifier.get(outsideObjects[j])[0],currentUncertainty[outsideObjects[j]])	
				newUncertaintyValue = currentUncertainty[outsideObjects[j]]  + float(deltaUncertainty[outsideObjects[j]])
				newProbabilityValue1 = convertEntropyToProb(newUncertaintyValue)
				#print 'newUncertaintyValue:%f'%(newUncertaintyValue)
				
				
				#finding index of classifier
				#indexTempProbClf = set.index(nextBestClassifier[j])
				if nextBestClassifier[outsideObjects[j]] == 'DT':
					nextBestClassifier[outsideObjects[j]] = 'NA'
				if nextBestClassifier[outsideObjects[j]] == 'GNB':
					indexTempProbClf = 1
				if nextBestClassifier[outsideObjects[j]] == 'RF':
					indexTempProbClf = 2
				if nextBestClassifier[outsideObjects[j]] == 'KNN':
					indexTempProbClf = 3
				
				# higher probability value	
				newProbabilityValue2 = 1 - newProbabilityValue1			
		
				#benefit is  (pi * pi_new)/cost(i) 
				probability_i = combineProbability(currentProbability[outsideObjects[j]])
				#print 'probability_i: %f, new probability : %f, cost : %f'%(probability_i,newProbabilityValue2,cost(nextBestClassifier[j]))
				if cost(nextBestClassifier[outsideObjects[j]]) != 0:
					benefit = float((probability_i*newProbabilityValue2)/float(cost(nextBestClassifier[outsideObjects[j]])))
					benefitArray[outsideObjects[j]] = benefit
				else:
					benefitArray[outsideObjects[j]] = -1
				
			#seq = sorted(benefitArray)
			#print benefitArray
			#Ordering the objects based on deltaUncertainty Value
			#order = [seq.index(v) for v in benefitArray]
			#topIndex= benefitArray.index(max(benefitArray))
			
			#topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			
			'''
			if len(outsideObjects) < blockSize :
				topKIndexes = outsideObjects
				#topKIndexes = heapq.nlargest(len(outsideObjects), range(len(outsideObjects)), benefitArray.__getitem__)
			else:
				topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			'''
			topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			
			'''
			if len(outsideObjects) > (blockSize/4):
				topKIndexes = [x for x in topKIndexes if x not in currentAnswerSet]
			else:
				topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			'''
			#print 'top index:%d'%(topIndex)
			
			t22 = time.time()
			totalThinkTime = totalThinkTime + (t22-t21)
			#thinkTimeList.append(t22-t21)
			thinkTimeList.append(totalThinkTime)
			
			
			print>>f1,"inside objects : {} ".format(currentAnswerSet)
			print>>f1,"state of inside objects: {}".format(stateListInside)
			print>>f1,"outsideObjects : {} ".format(outsideObjects)
			print>>f1,"state of outside objects: {}".format(stateListOutside)
			print>>f1,"think time list: {}".format(thinkTimeList)
			print>>f1,"execution time list: {}".format(executionTimeList)
			
			print benefitArray
			benefitTopK = [benefitArray[e] for e in  topKIndexes]
			print benefitTopK
			
			if(all(element==-10 or element==-1 for element in benefitTopK) and count > 5):
				print 'nothing else to execute which has some benefit values'
				print>>f1,'nothing else to execute which has some benefit values'
				break
			benefitTopK[:]=[]
			if(len(outsideObjects)==0 and count !=0):
				break
			
			#i=topIndex #next image to be run
			t2 = time.time()
			
			timeElapsed = totalExecutionTime + totalThinkTime
			
			#timeList.append(timeElapsed)
			#print 'next images to be run'
			#print topKIndexes
			
			print>>f1,'benefit array: {}'.format(benefitArray)
			print>>f1,'next images to be run: {}'.format(topKIndexes)
			classifierSet = [nextBestClassifier[item2] for item2 in topKIndexes]
			allClassifierSet = [nextBestClassifier[item3] for item3 in allObjects]
			'''
			if(all(element=='NA' for element in classifierSet) and count > 20):
				break
			'''
			'''
			#Uncomment this part. For reqular execution.
			if(all(element=='NA' for element in classifierSet) and count > 4):
				topKIndexes = allObjects
				#break
			if(all(element=='NA' for element in allClassifierSet) and count > 4):
				break
			'''
			#print>>f1,'classifier set: {}'.format(classifierSet)
			benefitArray[:] =[]
			classifierSet[:] = []
			
			#print 'round %d completed'%(count)
			#print 'time taken %f'%(timeElapsed)
			
			# block size is determined in this part.
			if count ==0:
				blockSize = block
				topKIndexes[:]= []
				#print 'blockSize: %d'%(blockSize)
			
			
			
			
			###### Time check########
			
			if timeElapsed > currentTimeBound:
				qualityOfAnswer = findQuality(currentProbability)
				
				if len(qualityOfAnswer[3]) > 0:
					realF1 = findRealF1(qualityOfAnswer[3])
				else: 
					realF1 = 0
				#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
				f1measure = realF1
				timeList.append(timeElapsed)
				f1List.append(f1measure)
				'''
				f1measure = qualityOfAnswer[0]
				timeList.append(timeElapsed)
				f1List.append(f1measure)
				'''
				#print 'time bound completed:%d'%(currentTimeBound)	
				#print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				#print 'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				
				currentTimeBound = currentTimeBound + stepSize
				
					
			

			if timeElapsed > timeBudget:
				break
			
			
			'''
			executionPerformed = executionPerformed + 	blockSize	
			if(executionPerformed>totalAllowedExecution):
				qualityOfAnswer = findQuality(currentProbability)
				print 'returned images'
				print qualityOfAnswer[3]
				if len(qualityOfAnswer[3]) > 0 :
					realF1 = findRealF1(qualityOfAnswer[3])
				else:
					realF1 = 0
				print 'real F1 : %f'%(realF1)
				
	
				#f1measure = qualityOfAnswer[0]
				f1measure = realF1
				f1measurePerAction = float(f1measure)/totalAllowedExecution
				timeList.append(timeElapsed)
				f1List.append(f1measurePerAction)
				realF1List.append(f1measure)
				
				print>>f1,'block size:%f'%(blockSize)
				print>>f1,'returned images: {}'.format(qualityOfAnswer[3])
				print>>f1,'length of answer set:%f'%len(qualityOfAnswer[3])
				unprocessedObjects = findUnprocessed(currentProbability)
				print>>f1,'unprocessed objects : {} '.format(unprocessedObjects)
				print>>f1,'length of unprocessed objects:%f'%(len(unprocessedObjects))
				print>>f1,'think time list: {}'.format(thinkTimeList)
				print>>f1,'execution time list: {}'.format(executionTimeList)
				
				break
				'''
			#if count >= 5000:
			#	break
			count=count+1
			
		plt.title('Quality vs Time Value')
		#print>>f1,'percent : %f'%(percent)
		#print>>f1,'block size : %f'%(block)
		#print>>f1,"f1 measures : {} ".format(realF1List)
		#print>>f1,'total think time :%f'%(totalThinkTime)
		#print>>f1,'total execution time :%f'%(totalExecutionTime)
		plt.ylabel('Quality')
		plt.xlabel('time')
		xValue = timeList
		yValue = f1List
		#print>>f1,"x value : {} ".format(xValue)
		#print>>f1,"y value : {} ".format(yValue)
		#print "x value : {} ".format(xValue)
		#print "y value : {} ".format(yValue)
		'''
		plt.plot(xValue, yValue,'b')
		plt.ylim([0, 1])
		plt.legend(loc="upper left")
		plt.savefig('plotQualityAdaptive8'+str(block)+'.eps',format = 'eps')
		plt.title('Quality vs Time for block size = '+str(block))
		'''
		#plt.show()
		#plt.close()
		#xValue = timeList
		#yValue = f1List
		
	'''
	plt.ylabel('Quality')
	plt.xlabel('Block Size')
	xValue = blockList
	yValue = realF1List
	print>>f1,"x value : {} ".format(xValue)
	print>>f1,"y value : {} ".format(yValue)
	print "x value : {} ".format(xValue)
	print "y value : {} ".format(yValue)
	'''
	'''
	plt.ylabel('Quality')
	plt.xlabel('time')
	xValue = timeList
	yValue = f1List
	print>>f1,"x value : {} ".format(xValue)
	print>>f1,"y value : {} ".format(yValue)
	print "x value : {} ".format(xValue)
	print "y value : {} ".format(yValue)
	'''
	#yValue = f1measure
	#yValue = f1measurePerAction
	#labelValue = 'Adaptive algorithm(Think='+str(percent)+'%)'
	#labelValue = 'Adaptive algorithm(Block size='+str(block)+')'
	plt.plot(xValue, yValue,'g')
	plt.ylim([0, 1])
	plt.legend(loc="upper left")
	plt.savefig('plotQualityAdaptive8.png')
	#plt.show()
	plt.close()
	return [timeList,f1List]

	
	
def baseline1(budget):  
	'''
	For this algorithm, one classifier is chosen randomly from a given set of classifiers.
	'''
	f1 = open('QueryExecutionResultMuctBaseline1GenderAverage.txt','w+')
	#gnb,et,rf,svm
	
	totalTime = 0 
	totalQuality= 0
	executionTime = 0
	stepSize = 4   #step size of 20 seconds. After every 20 seconds evaluate the quality
	currentTimeBound =4
	
	timeList =[]
	f1List = []
	
	
	set = [genderPredicate6,genderPredicate1,genderPredicate3,genderPredicate7]
	
	round = 1 
	count = 0 
	currentUncertainty = [1]*len(dl)
	currentProbability = {}
	for i in range(len(dl)):
			key = i
			value = [-1,-1,-1,-1]
			if key in currentProbability:
				currentProbability[key].append(value)
			else:
				currentProbability[key] = [value]	
				
	prevClassifier = {}
	for i in range(len(dl)):
		key = i
		value = [0,0,0,0]
		if key in prevClassifier:
			prevClassifier[key].append(value)
		else:
			prevClassifier[key] = [value]
				
				
	t1 = time.time()
	if count ==0:
		operator = set[0]			
		
				
		for i in range(len(dl)):
			probValues = operator([dl[i]])
			#probValues = operator([dl[i][:850]])
			#print>>f1,probValues
			'''
			indexClf = set.index(operator)
			tempProb = currentProbability[i][0]
			tempProb[indexClf] = probValues[0]
			'''
			#print>>f1,"temp prob : {} ".format(tempProb)
			
			indexClf = set.index(operator)
			tempProb = currentProbability[i][0]
			if probValues[0] > 0.55:
				tempProb[indexClf] = probValues[0]+0.2
			else:
				tempProb[indexClf] = probValues[0]-0.3
			
			# setting the bit for the corresponding classifier
			tempClf = prevClassifier[i][0]
			tempClf[indexClf] = 1
					
					
			# calculating the current cobined probability
			combinedProbability = combineProbability(currentProbability[i])
					
			# using the combined probability value to calculate uncertainty
			uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
			currentUncertainty[i] = uncertainty
			
	
	t2 = time.time()
	executionTime = executionTime + (t2- t1)
	set.remove(genderPredicate6)
	
	qualityOfAnswer = findQuality(currentProbability)
	print 'returned images'
	print qualityOfAnswer[3]
	print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
	if len(qualityOfAnswer[3]) > 0:
		realF1 = findRealF1(qualityOfAnswer[3])
	else: 
		realF1 = 0
	print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
	f1measure = realF1
	timeList.append(0)
	f1List.append(f1measure)
	#currentTimeBound = currentTimeBound + stepSize
	#print>>f1,'time bound completed:%d'%(currentTimeBound)
	
	
	for k in range(0,1):  # number of times this algorithm will be executed
		
				
		t1 = time.time()
		#gnb,et,rf,svm
		#set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
		set = [genderPredicate1,genderPredicate3,genderPredicate7]
		workflow =[]
		round = 1 
		#print workflow
	
		while len(set) >0:
			operator = random.choice(set)
			#probValues = operator(dl)
			workflow.append(operator)
			#rocProb = prob[0]
			for j in range(len(dl)):
				t11 = time.time()
				probValues = operator([dl[j]])
				
				rocProb = probValues[0]
				#print 'image:%d'%(j)
				#print("Roc Prob : {} ".format(rocProb))
					
				#index of classifier
				indexClf = set.index(operator)
				tempProb = currentProbability[j][0]
				tempProb[indexClf+1] = rocProb
				
				t12 = time.time()
				#t12 = time.time()
			
				executionTime = executionTime + (t12- t11)
	
				if executionTime > currentTimeBound:
					qualityOfAnswer = findQuality(currentProbability)
					#print 'returned images'
					#print qualityOfAnswer[3]
					#print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
					if len(qualityOfAnswer[3]) > 0:
						realF1 = findRealF1(qualityOfAnswer[3])
					else: 
						realF1 = 0
					#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
					f1measure = realF1
					print realF1
					timeList.append(currentTimeBound)
					f1List.append(f1measure)
					currentTimeBound = currentTimeBound + stepSize
					#print>>f1,'time bound completed:%d'%(currentTimeBound)
				if executionTime > budget:
					break
				
				

			print 'round %d completed'%(round)
			set.remove(operator)
			round = round + 1
			
				
		
		
		plt.title('Quality vs Time Value for BaseLine 3')
		xValue = timeList
		yValue = f1List
		plt.plot(xValue, yValue)
		plt.ylim([0, 1])
		plt.ylabel('Quality')
		plt.xlabel('Time')	
		plt.savefig('QualityBaseLine1GenderMuct.eps', format='eps')
		#plt.show()
		plt.close()
		
		print>>f1,"Workflow : {} ".format(workflow)
	
	return [timeList,f1List]	
	
	
def baseline2():  
	'''
	For this algorithm, classifiers are ordered based on (AUC)/Cost value.
	'''
	f1 = open('QueryExecutionResultMuctBaseline2Gender.txt','w+')
	
	
	#Initialization step. 
	currentProbability = {}
	
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbability:
			currentProbability[key].append(value)
		else:
			currentProbability[key] = [value]	
			
	t1 = time.time()
	
	#gnb,et,rf,svm
	set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
	aucSet = [0.85,0.92,0.92,0.89]
	#costSet = [0.063052,0.014482,0.015253,1.567327]
	costSet = [0.029360,0.018030,0.020180,0.790850]
	
	benefitSet = [ float(aucSet[i])/costSet[i] for i in range(len(aucSet))]
	print benefitSet
	workflow =[x for y, x in sorted(zip(benefitSet, set),reverse=True)]
	print workflow
	round = 1 
	
	
	for i in range(len(workflow)):
		operator = workflow[i]
		probValues = operator(dl)
		
		for j in range(len(dl)):
			imageProb = probValues[j]
			rocProb = imageProb
			averageProbability = 0;
			#print 'image:%d'%(j)
			#print("Roc Prob : {} ".format(rocProb))
				
			#index of classifier
			indexClf = set.index(operator)
			tempProb = currentProbability[j][0]
			tempProb[indexClf] = rocProb

		print 'round %d completed'%(round)
		set.remove(operator)
		round = round + 1
		
			
	t2 = time.time()
	timeElapsed = t2-t1
	qualityOfAnswer = findQuality(currentProbability)
	print>>f1,'Time taken: %f, f1 measure of the answer set: %f, precision:%f, recall:%f'%(timeElapsed,qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2])
	print("Workflow : {} ".format(workflow))




def baseline3(budget):  
	'''
	For this algorithm, classifiers are chosen based on auc/cost value. But for one classifier, we try to run it on all the images.
	'''
	f1 = open('QueryResultBaseline3GenderMuct.txt','w+')
	#gnb,et,rf,svm
	
	totalTime = 0 
	totalQuality= 0
	timeList =[]
	f1List = []
	
	executionTime = 0
	stepSize = 4   #step size of 20 seconds. After every 20 seconds evaluate the quality
	currentTimeBound =4
		
	#gnb,et,rf,svm
	#set = [genderPredicate6,genderPredicate1,genderPredicate3,genderPredicate7]
	#set = [genderPredicate17,genderPredicate1,genderPredicate3,genderPredicate7]
	set = [genderPredicate6,genderPredicate1,genderPredicate3,genderPredicate7]
	aucSet = [0.52,0.85,0.93,0.80]
	costSet = [0.018175,0.095875,0.023094, 0.866073 ]
	
	
	benefitSet = [ float(aucSet[i])/costSet[i] for i in range(len(aucSet))]
	print benefitSet
	workflow =[x for y, x in sorted(zip(benefitSet, set),reverse=True)]
	print workflow
	round = 1 
	count = 0 
	currentUncertainty = [1]*len(dl)
	currentProbability = {}
	for i in range(len(dl)):
			key = i
			value = [-1,-1,-1,-1]
			if key in currentProbability:
				currentProbability[key].append(value)
			else:
				currentProbability[key] = [value]	
				
	prevClassifier = {}
	for i in range(len(dl)):
		key = i
		value = [0,0,0,0]
		if key in prevClassifier:
			prevClassifier[key].append(value)
		else:
			prevClassifier[key] = [value]
				
				
	t1 = time.time()
	if count ==0:
		operator = set[0]			
		
				
		for i in range(len(dl)):
			probValues = operator([dl[i]])
			#probValues = operator([dl[i][:850]])
			#print>>f1,probValues
			'''
			indexClf = set.index(operator)
			tempProb = currentProbability[i][0]
			tempProb[indexClf] = probValues[0]
			'''
			#print>>f1,"temp prob : {} ".format(tempProb)
			
			indexClf = set.index(operator)
			tempProb = currentProbability[i][0]
			if probValues[0] > 0.55:
				tempProb[indexClf] = probValues[0]+0.2
			else:
				tempProb[indexClf] = probValues[0]-0.3
			
			# setting the bit for the corresponding classifier
			tempClf = prevClassifier[i][0]
			tempClf[indexClf] = 1
					
					
			# calculating the current cobined probability
			combinedProbability = combineProbability(currentProbability[i])
					
			# using the combined probability value to calculate uncertainty
			uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
			currentUncertainty[i] = uncertainty
		'''
		operator = set[1]	
		for i in range((len(dl))//2):
			#print dl[i]
			#X_transformed = pca.fit_transform(dl[i])
			#print X_transformed
			#probValues = operator(X_transformed)
			#probValues = operator([dl_new[i]])
			#probValues = operator([pca.fit_transform(dl[i])])
			probValues = operator([dl[i]])
			#probValues = operator([dl[i][:850]])
			
			#probValues = probX[i,1]
			#print>>f1,probValues
			
			#indexClf = set.index(operator)
			#tempProb = currentProbability[i][0]
			#tempProb[indexClf] = probValues[0]
			
			
			indexClf = set.index(operator)
			tempProb = currentProbability[i][0]
			#print probValues[0]
			if probValues[0] > 0.55:
				tempProb[indexClf] = probValues[0]+0.2
			else:
				tempProb[indexClf] = probValues[0]-0.3
			
			# setting the bit for the corresponding classifier
			tempClf = prevClassifier[i][0]
			tempClf[indexClf] = 1
			
			
			# calculating the current cobined probability
			combinedProbability = combineProbability(currentProbability[i])
			
			# using the combined probability value to calculate uncertainty
			uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
			currentUncertainty[i] = uncertainty
		'''
	t2 = time.time()
	executionTime = executionTime + (t2- t1)
	set.remove(genderPredicate6)
	
	qualityOfAnswer = findQuality(currentProbability)
	print 'returned images'
	print qualityOfAnswer[3]
	print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
	if len(qualityOfAnswer[3]) > 0:
		realF1 = findRealF1(qualityOfAnswer[3])
	else: 
		realF1 = 0
	print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
	f1measure = realF1
	timeList.append(0)
	f1List.append(f1measure)
	#currentTimeBound = currentTimeBound + stepSize
	#print>>f1,'time bound completed:%d'%(currentTimeBound)
	
	
	aucSet = [0.85,0.93,0.80]
	#costSet = [0.095875,0.023094, 0.866073 ]
	costSet = [0.014,0.023094, 0.008 ]
	
	
	print 'size of the dataset:%d'%(len(dl))
	print 'budget:%d'%(budget)
	
	benefitSet = [ float(aucSet[i])/costSet[i] for i in range(len(aucSet))]
	print benefitSet
	workflow =[x for y, x in sorted(zip(benefitSet, set),reverse=True)]
	print workflow
	round = 1 
	
	
	for k in range(0,1):  # number of times this algorithm will be executed
		#Initialization step. 
		
				
		t1 = time.time()
	
		print>>f1,'Workflow : {} '.format(workflow)
		
		for i in range(len(workflow)):
			operator = workflow[i]
			#t11 = time.time()
			#probValues = operator(dl)
			t_sort_start=time.time()
			probabilitySet = []
			probDictionary = {}
			sorted_x={}
			
			for i1 in range(len(dl)):
		
				
				sumProb = 0
				countProb = 0		
				flag = 0
				#combinedProbability = combineProbability(currentProbability[i])
		
				for p in currentProbability[i1][0]:
					#print>>f1,'current probability: {}'.format(currentProbability[i][0])
					if p!=-1 :
					#productProbability = productProbability*(1-p)
						sumProb = sumProb+p
						countProb = countProb+1
						flag = 1
				if flag==0:
					combinedProbability = 0.5	
				else: 
					#combinedProbability = 1 - productProbability
					combinedProbability = float(sumProb)/countProb
		
				probabilitySet.append(combinedProbability)
		
				key1 = i1
				value = combinedProbability
				probDictionary[key1] = value
				
			sortedProbSet = probabilitySet[:]
			sortedProbSet.sort(reverse=True)
			
			#sorted_x = sorted(probDictionary.iteritems(), key=lambda (k,v): (v,k), reverse = True)
			sorted_x = sorted(probDictionary.iteritems(), key=lambda (k,v): (v,k))
					
			#rocProb = prob[0]
			#dl_new = np.random.shuffle(dl)
			'''
			arr = np.arange(len(dl))
			np.random.shuffle(arr) 
			print arr
			'''
			t_sort_end=time.time()
			#executionTime =  executionTime + (t_sort_end-t_sort_start)
			#for j in range(len(dl)):
			for key in sorted_x:
			#for j in arr:
				#imageProb = probValues[j]
				
				
				t11 = time.time()
				imageProb = operator([dl[key[0]]])
				#imageProb = operator([dl[j]])
				
				
				rocProb = imageProb[0]
				

				#print 'image:%d'%(j)
				#print("Roc Prob : {} ".format(rocProb))
					
				#index of classifier
				indexClf = set.index(operator)
				tempProb = currentProbability[key[0]][0]
				#tempProb = currentProbability[j][0]
				tempProb[indexClf+1] = rocProb
				#print>>f1,"temp prob : {} ".format(tempProb)
				
				t12 = time.time()
			#t12 = time.time()
			
				executionTime = executionTime + (t12- t11)
				
				
			
			
			
				if executionTime > currentTimeBound:
					qualityOfAnswer = findQuality(currentProbability)
					#print 'returned images'
					#print qualityOfAnswer[3]
					#print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
					if len(qualityOfAnswer[3]) > 0:
						realF1 = findRealF1(qualityOfAnswer[3])
					else: 
						realF1 = 0
					#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
					f1measure = realF1
					timeList.append(currentTimeBound)
					f1List.append(f1measure)
					currentTimeBound = currentTimeBound + stepSize
					#print>>f1,'time bound completed:%d'%(currentTimeBound)
				if executionTime > budget:
					break
				
				
			print 'round %d completed'%(round)
			
			
			
			round = round + 1
			
				
		t2 = time.time()
		timeElapsed = t2-t1
		qualityOfAnswer = findQuality(currentProbability)
		f1measure = qualityOfAnswer[0]
		
		# store the time values and F1 values
		#print>>f1,"budget values : {} ".format(timeList)
		#print>>f1,"f1 measures : {} ".format(f1List)
			
		#plot quality vs time 
		#timeList = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280] 
		#f1List = [0.6667605490515276, 0.666838374853555, 0.6669078673934243, 0.6670447162904274, 0.6671836282556838, 0.6671978153931125, 0.6671986297399674, 0.6671819086745975, 0.6671767559957542, 0.6672143236162872, 0.6673124457328257, 0.6674186908487334, 0.6674062673780302, 0.6674207092762636] 
		'''
		plt.title('Quality vs Time Value for BaseLine 3')
		xValue = timeList
		yValue = f1List
		plt.plot(xValue, yValue)
		plt.ylim([0, 1])
		plt.ylabel('Quality')
		plt.xlabel('Time')	
		plt.savefig('QualityBaseLine3GenderMuct.eps', format='eps')
		#plt.show()
		plt.close()
		'''
		
		
	print>>f1,"Workflow : {} ".format(workflow)
	print>>f1,'Time taken: %f, f1 measure of the answer set: %f, precision:%f, recall:%f'%(timeElapsed,qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2])
	#return f1measure
	return [timeList,f1List]


def baseline4(budget):  
	'''
	For this algorithm, classifiers are ordered based on auc/cost value. But for each images, we try to run all the classifiers before going to another image.
	'''
	
	#f1 = open('QueryExecutionResultBaseline4GenderAverage.txt','w+')
	f1 = open('QueryResultBaseline4GenderMuct.txt','w+')
	#gnb,et,rf,svm
	
	totalTime = 0 
	totalQuality= 0
	print 'Query budget:%f'%(budget)
	
	timeList =[]
	f1List = []
	executionTime = 0
	stepSize = 4   #step size of 20 seconds. After every 20 seconds evaluate the quality
	currentTimeBound = 4
	
	#gnb,et,rf,svm
	#set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
	#aucSet = [0.85,0.92,0.92,0.89]
	#costSet = [0.063052,0.014482,0.015253,1.567327]
	
	#DT,GNB,RF,KNN
	#LDA,GNB,RF,KNN
	set = [genderPredicate6,genderPredicate1,genderPredicate3,genderPredicate7]
	
	#set = [genderPredicate17,genderPredicate1,genderPredicate3,genderPredicate7]
	aucSet = [0.88,0.85,0.93,0.80]
	#costSet = [0.018175,0.095875,0.023094, 0.866073 ]
	costSet = [0.014,0.023094, 0.008 ]
	currentUncertainty = [1]*len(dl)
	count = 0
	t1 = time.time()
	currentProbability = {}
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbability:
			currentProbability[key].append(value)
		else:
			currentProbability[key] = [value]
			
	prevClassifier = {}
	for i in range(len(dl)):
		key = i
		value = [0,0,0,0]
		if key in prevClassifier:
			prevClassifier[key].append(value)
		else:
			prevClassifier[key] = [value]
	
	operator = set[0]			
		
		
	for i in range(len(dl)):
		#probValues = operator([dl[i]])
		probValues = operator([dl[i]])
		#probValues = operator([dl[i][:850]])
		indexClf = set.index(operator)
		tempProb = currentProbability[i][0]
		if probValues[0] > 0.55:
			tempProb[indexClf] = probValues[0]+0.2
		else:
			tempProb[indexClf] = probValues[0]-0.3
		#print>>f1,probValues
		#indexClf = set.index(operator)
		'''
		tempProb = currentProbability[i][0]
		tempProb[0] = probValues[0]
		'''
		#print>>f1,"temp prob : {} ".format(tempProb)
					
		# setting the bit for the corresponding classifier
		tempClf = prevClassifier[i][0]
		tempClf[0] = 1
	'''	
	operator = set[1]	
	for i in range((len(dl))//2):
		#print dl[i]
		#X_transformed = pca.fit_transform(dl[i])
		#print X_transformed
		#probValues = operator(X_transformed)
		#probValues = operator([dl_new[i]])
		#probValues = operator([pca.fit_transform(dl[i])])
		probValues = operator([dl[i]])
		#probValues = operator([dl[i][:850]])
		
		#probValues = probX[i,1]
		#print>>f1,probValues
		
		#indexClf = set.index(operator)
		#tempProb = currentProbability[i][0]
		#tempProb[indexClf] = probValues[0]
		
		
		indexClf = set.index(operator)
		tempProb = currentProbability[i][0]
		#print probValues[0]
		if probValues[0] > 0.55:
			tempProb[indexClf] = probValues[0]+0.2
		else:
			tempProb[indexClf] = probValues[0]-0.3
		
		# setting the bit for the corresponding classifier
		tempClf = prevClassifier[i][0]
		tempClf[indexClf] = 1
		
		
		# calculating the current cobined probability
		combinedProbability = combineProbability(currentProbability[i])
		
		# using the combined probability value to calculate uncertainty
		uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
		currentUncertainty[i] = uncertainty				
	'''	
	t2 = time.time()
	executionTime = executionTime + (t2- t1)
	set.remove(genderPredicate6)
	
	qualityOfAnswer = findQuality(currentProbability)
	#print 'returned images'
	#print qualityOfAnswer[3]
	#print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
	if len(qualityOfAnswer[3]) > 0:
		realF1 = findRealF1(qualityOfAnswer[3])
	else: 
		realF1 = 0
	#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
	f1measure = realF1
	timeList.append(0)
	f1List.append(f1measure)
	#currentTimeBound = currentTimeBound + stepSize
	#print>>f1,'time bound completed:%d'%(currentTimeBound)
	
	
	#set.remove(genderPredicate1)
	
	aucSet = [0.85,0.93,0.80]
	costSet = [0.095875,0.023094, 0.866073 ]
	
	#print 'size of the dataset:%d'%(len(dl))
	#print 'budget:%d'%(budget)
	
	benefitSet = [ float(aucSet[i])/costSet[i] for i in range(len(aucSet))]
	#print benefitSet
	workflow =[x for y, x in sorted(zip(benefitSet, set),reverse=True)]
	#print workflow
	round = 1 
	
	
	for k in range(0,1):  # number of times this algorithm will be executed
		#Initialization step. 
		
				
		t1 = time.time()
	
		print("Workflow : {} ".format(workflow))
		

		for j in range(len(dl)):
				#imageProb = probValues[j]
			for i in range(len(workflow)):
				operator = workflow[i]
				t11 = time.time()
				imageProb = operator([dl[j]])
				t12 = time.time()
				rocProb = imageProb[0]
				
				#print 'image:%d'%(j)
				#print("Roc Prob : {} ".format(rocProb))
					
				#index of classifier
				indexClf = set.index(operator)
				tempProb = currentProbability[j][0]
				tempProb[indexClf+1] = rocProb
				
				
				
				
				executionTime = executionTime + (t12- t11)
				
				
				if executionTime > budget:
					break
				
				
				if executionTime > currentTimeBound:
					qualityOfAnswer = findQuality(currentProbability)
					#print 'returned images'
					#print qualityOfAnswer[3]
					if len(qualityOfAnswer[3]) > 0:
						realF1 = findRealF1(qualityOfAnswer[3])
					else:
						realF1 = 0
					#print 'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
					f1measure = realF1
					#f1measure = qualityOfAnswer[0]
					timeList.append(currentTimeBound)
					f1List.append(f1measure)
					currentTimeBound = currentTimeBound + stepSize
					#print 'time bound completed:%d'%(currentTimeBound)	
					
				if executionTime > budget:
					break
			
			round = round + 1
			
				
		t2 = time.time()
		timeElapsed = t2-t1
		'''
		qualityOfAnswer = findQuality(currentProbability)
		print 'returned images'
		print qualityOfAnswer[3]
		if len(qualityOfAnswer[3]) > 0:
			realF1 = findRealF1(qualityOfAnswer[3])
		else:
			realF1 = 0
		print 'real F1 : %f'%(realF1)
		#f1measure = qualityOfAnswer[0]
		f1measure = realF1
		#f1measure = qualityOfAnswer[0]
		timeList.append(budget)
		f1List.append(f1measure)
		currentTimeBound = currentTimeBound + stepSize
		print 'time bound completed:%d'%(currentTimeBound)
		'''
		
		# store the time values and F1 values
		#print>>f1,"budget values : {} ".format(timeList)
		#print>>f1,"f1 measures : {} ".format(f1List)
			
		#plot quality vs time 
		#timeList = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280] 
		#f1List = [0.6667605490515276, 0.666838374853555, 0.6669078673934243, 0.6670447162904274, 0.6671836282556838, 0.6671978153931125, 0.6671986297399674, 0.6671819086745975, 0.6671767559957542, 0.6672143236162872, 0.6673124457328257, 0.6674186908487334, 0.6674062673780302, 0.6674207092762636] 
		'''
		plt.title('Quality vs Time Value for BaseLine 4')
		xValue = timeList
		yValue = f1List
		plt.plot(xValue, yValue)
		plt.ylabel('Quality')
		plt.xlabel('Time')	
		plt.ylim([0, 1])
		
		plt.savefig('QualityBaseLine4GenderMuct.eps', format='eps')
		#plt.show()
		plt.close()
		'''
	
	
	#print>>f1,"Workflow : {} ".format(workflow)
	#print>>f1,'Time taken: %f, f1 measure of the answer set: %f, precision:%f, recall:%f'%(timeElapsed,qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2])
	#return f1measure
	return [timeList,f1List]

def baseline5(budget):  
	'''
	For this algorithm, classifiers are ordered based on auc/cost value. But for each images, we try to run all the classifiers before going to another image.
	'''
	
	#f1 = open('QueryExecutionResultBaseline4GenderAverage.txt','w+')
	f1 = open('QueryResultBaseline4GenderMuct.txt','w+')
	#gnb,et,rf,svm
	
	totalTime = 0 
	totalQuality= 0
	print 'Query budget:%f'%(budget)
	
	timeList =[]
	f1List = []
	executionTime = 0
	stepSize = 4   #step size of 20 seconds. After every 20 seconds evaluate the quality
	currentTimeBound = 4
	
	#gnb,et,rf,svm
	#set = [genderPredicate1,genderPredicate2,genderPredicate3,genderPredicate4]
	#aucSet = [0.85,0.92,0.92,0.89]
	#costSet = [0.063052,0.014482,0.015253,1.567327]
	
	#DT,GNB,RF,KNN
	#LDA,GNB,RF,KNN
	set = [genderPredicate6,genderPredicate1,genderPredicate3,genderPredicate7]
	
	#set = [genderPredicate17,genderPredicate1,genderPredicate3,genderPredicate7]
	aucSet = [0.88,0.85,0.93,0.80]
	costSet = [0.018175,0.095875,0.023094, 0.866073 ]
	currentUncertainty = [1]*len(dl)
	count = 0
	t1 = time.time()
	currentProbability = {}
	for i in range(len(dl)):
		key = i
		value = [-1,-1,-1,-1]
		if key in currentProbability:
			currentProbability[key].append(value)
		else:
			currentProbability[key] = [value]
			
	prevClassifier = {}
	for i in range(len(dl)):
		key = i
		value = [0,0,0,0]
		if key in prevClassifier:
			prevClassifier[key].append(value)
		else:
			prevClassifier[key] = [value]
	
	operator = set[0]			
		
		
	for i in range(len(dl)):
		#probValues = operator([dl[i]])
		probValues = operator([dl[i]])
		#probValues = operator([dl[i][:850]])
		indexClf = set.index(operator)
		tempProb = currentProbability[i][0]
		if probValues[0] > 0.55:
			tempProb[indexClf] = probValues[0]+0.2
		else:
			tempProb[indexClf] = probValues[0]-0.3
		#print>>f1,probValues
		#indexClf = set.index(operator)
		'''
		tempProb = currentProbability[i][0]
		tempProb[0] = probValues[0]
		'''
		#print>>f1,"temp prob : {} ".format(tempProb)
					
		# setting the bit for the corresponding classifier
		tempClf = prevClassifier[i][0]
		tempClf[0] = 1
					
					
			
	t2 = time.time()
	executionTime = executionTime + (t2- t1)
	set.remove(genderPredicate6)
	
	qualityOfAnswer = findQuality(currentProbability)
	#print 'returned images'
	#print qualityOfAnswer[3]
	#print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
	if len(qualityOfAnswer[3]) > 0:
		realF1 = findRealF1(qualityOfAnswer[3])
	else: 
		realF1 = 0
	#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
	f1measure = realF1
	timeList.append(0)
	f1List.append(f1measure)
	#currentTimeBound = currentTimeBound + stepSize
	#print>>f1,'time bound completed:%d'%(currentTimeBound)
	
	
	#set.remove(genderPredicate1)
	
	aucSet = [0.85,0.93,0.80]
	costSet = [0.095875,0.023094, 0.866073 ]
	
	#print 'size of the dataset:%d'%(len(dl))
	#print 'budget:%d'%(budget)
	
	benefitSet = [ float(aucSet[i])/costSet[i] for i in range(len(aucSet))]
	#print benefitSet
	workflow =[x for y, x in sorted(zip(benefitSet, set),reverse=True)]
	#print workflow
	round = 1 
	
	
	for k in range(0,1):  # number of times this algorithm will be executed
		#Initialization step. 
		
				
		t1 = time.time()
	
		print("Workflow : {} ".format(workflow))
		
		### Sorting objects based on their probability value.
		probabilitySet = []
		probDictionary = {}
		sorted_x={}
		
		for i1 in range(len(dl)):
	
			
			sumProb = 0
			countProb = 0		
			flag = 0
			#combinedProbability = combineProbability(currentProbability[i])
	
			for p in currentProbability[i1][0]:
				#print>>f1,'current probability: {}'.format(currentProbability[i][0])
				if p!=-1 :
				#productProbability = productProbability*(1-p)
					sumProb = sumProb+p
					countProb = countProb+1
					flag = 1
			if flag==0:
				combinedProbability = 0.5	
			else: 
				#combinedProbability = 1 - productProbability
				combinedProbability = float(sumProb)/countProb
	
			probabilitySet.append(combinedProbability)
	
			key1 = i1
			value = combinedProbability
			probDictionary[key1] = value
			
		sortedProbSet = probabilitySet[:]
		sortedProbSet.sort(reverse=True)
		
		#sorted_x = sorted(probDictionary.iteritems(), key=lambda (k,v): (v,k), reverse = True)
		sorted_x = sorted(probDictionary.iteritems(), key=lambda (k,v): (v,k))
				
		#rocProb = prob[0]
		#dl_new = np.random.shuffle(dl)
		'''
		arr = np.arange(len(dl))
		np.random.shuffle(arr) 
		print arr
		'''
		t_sort_end=time.time()
		#executionTime =  executionTime + (t_sort_end-t_sort_start)
		#for j in range(len(dl)):
		for key in sorted_x:
		#for j in range(len(dl)):
				#imageProb = probValues[j]
			for i in range(len(workflow)):
				operator = workflow[i]
				t11 = time.time()
				imageProb = operator([dl[key[0]]])
				#imageProb = operator([dl[j]])
				t12 = time.time()
				rocProb = imageProb[0]
				
				#print 'image:%d'%(j)
				#print("Roc Prob : {} ".format(rocProb))
					
				#index of classifier
				indexClf = set.index(operator)
				#tempProb = currentProbability[j][0]
				tempProb = currentProbability[key[0]][0]
				tempProb[indexClf+1] = rocProb
				
				
				
				
				executionTime = executionTime + (t12- t11)
				
				
				if executionTime > budget:
					break
				
				
				if executionTime > currentTimeBound:
					qualityOfAnswer = findQuality(currentProbability)
					#print 'returned images'
					#print qualityOfAnswer[3]
					if len(qualityOfAnswer[3]) > 0:
						realF1 = findRealF1(qualityOfAnswer[3])
					else:
						realF1 = 0
					#print 'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
					f1measure = realF1
					#f1measure = qualityOfAnswer[0]
					timeList.append(currentTimeBound)
					f1List.append(f1measure)
					currentTimeBound = currentTimeBound + stepSize
					#print 'time bound completed:%d'%(currentTimeBound)	
					
				if executionTime > budget:
					break
			
			round = round + 1
			
				
		t2 = time.time()
		timeElapsed = t2-t1
		'''
		qualityOfAnswer = findQuality(currentProbability)
		print 'returned images'
		print qualityOfAnswer[3]
		if len(qualityOfAnswer[3]) > 0:
			realF1 = findRealF1(qualityOfAnswer[3])
		else:
			realF1 = 0
		print 'real F1 : %f'%(realF1)
		#f1measure = qualityOfAnswer[0]
		f1measure = realF1
		#f1measure = qualityOfAnswer[0]
		timeList.append(budget)
		f1List.append(f1measure)
		currentTimeBound = currentTimeBound + stepSize
		print 'time bound completed:%d'%(currentTimeBound)
		'''
		
		# store the time values and F1 values
		#print>>f1,"budget values : {} ".format(timeList)
		#print>>f1,"f1 measures : {} ".format(f1List)
			
		#plot quality vs time 
		#timeList = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280] 
		#f1List = [0.6667605490515276, 0.666838374853555, 0.6669078673934243, 0.6670447162904274, 0.6671836282556838, 0.6671978153931125, 0.6671986297399674, 0.6671819086745975, 0.6671767559957542, 0.6672143236162872, 0.6673124457328257, 0.6674186908487334, 0.6674062673780302, 0.6674207092762636] 
		plt.title('Quality vs Time Value for BaseLine 4')
		xValue = timeList
		yValue = f1List
		plt.plot(xValue, yValue)
		plt.ylabel('Quality')
		plt.xlabel('Time')	
		plt.ylim([0, 1])
		
		plt.savefig('QualityBaseLine5GenderMuct.eps', format='eps')
		#plt.show()
		plt.close()
	
	
	#print>>f1,"Workflow : {} ".format(workflow)
	#print>>f1,'Time taken: %f, f1 measure of the answer set: %f, precision:%f, recall:%f'%(timeElapsed,qualityOfAnswer[0],qualityOfAnswer[1],qualityOfAnswer[2])
	#return f1measure
	return [timeList,f1List]

	


def generateMultipleExecutionResult():
	#createSample()
	#createRandomSample()
	
	#q1_all,q2_all,q3_all = [],[],[]
	t1_all,q1_all,t2_all,q2_all,t3_all,q3_all=[],[],[],[],[],[]
	f1 = open('QueryExecutionMuctResultsGenderCompare.txt','w+')
	
	t_init_list = []
	
	for i in range(1):
		global dl,nl 
		#dl,nl =pickle.load(open('5Samples/MuctTrainGender'+str(i)+'_XY.p','rb'))
		
		#imageIndex = [i1 for i1 in sorted(random.sample(xrange(len(dl2)), 100))]
		t_init_pred_start = time.time()
		imageIndex = [i1 for i1 in sorted(random.sample(xrange(len(dl2)), 150))]
		dl_test = [dl2[i1] for i1 in  imageIndex]
		t_init_pred_end = time.time()
		
		t_init = (t_init_pred_end - t_init_pred_start)
		print>>f1,'total load time :%f'%(t_init)
		t_init_list.append(t_init)
		
		nl_test = [nl2[i1] for i1 in imageIndex]
		
		
		
		#uncomment this part after done with the initialization exp.
		dl = np.array(dl_test)
		nl = np. array(nl_test)
		[t3,q3] =adaptiveOrder8(60)
		[t1,q1]=baseline3(60)
		[t2,q2] =baseline4(60)
		
		t1_all.append(t1)
		t2_all.append(t2)
		t3_all.append(t3)
		q1_all.append(q1)
		q2_all.append(q2)
		q3_all.append(q3)
		
		
		print>>f1,'sameple id : %d'%(i)
		print>>f1,"t1 = {} ".format(t1)
		print>>f1,"q1 = {} ".format(q1)
		print>>f1,"t2 = {} ".format(t2)
		print>>f1,"q2 = {} ".format(q2)
		print>>f1,"t3 = {} ".format(t3)
		print>>f1,"q3 = {} ".format(q3)
		
		
		'''
		plt.plot(t1, q1,lw=2,color='green',  label='Baseline1 (Function Based Approach)')
		plt.plot(t2, q2,lw=2,color='orange',  label='Baseline2 (Object Based Approach)')
		plt.plot(t3, q3,lw=2,color='blue', label='Iterative Approach') ##2,000

		plt.ylim([0, 1])
		plt.xlim([0, 20])
		plt.title('Quality vs Cost')
		plt.legend(loc="lower left",fontsize='medium')
		plt.ylabel('F1-measure')
		plt.xlabel('Cost')	
		plt.savefig('PlotQualityComparisonMuctBaseline_gender'+str(i)+'.png', format='png')
		plt.savefig('PlotQualityComparisonMuctBaseline_gender'+str(i)+'.eps', format='eps')
		#plt.show()
		plt.close()
		'''
		print 'iteration :%d completed'%(i)
	
	
	print>>f1,'average initial predicate evaluation time :%f'%(np.mean(t_init_list))
	
	
	
	q1 = [sum(e)/len(e) for e in zip(*q1_all)]
	q2 = [sum(e)/len(e) for e in zip(*q2_all)]
	q3 = [sum(e)/len(e) for e in zip(*q3_all)]
	plt.plot(t1, q1,lw=2,color='green', marker='o', label='Baseline1 (Function Based Approach)')
	plt.plot(t2, q2,lw=2,color='orange', marker='^', label='Baseline2 (Object Based Approach)')
	plt.plot(t3, q3,lw=2,color='blue',marker ='d', label='Iterative Approach') ##2,000

	plt.ylim([0, 1])
	plt.xlim([0, max(max(t1),max(t2),max(t3))])
	plt.title('Quality vs Cost')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
	#plt.ylabel('F1-measure')
	plt.ylabel('Gain')
	plt.xlabel('Cost')	
	plt.savefig('PlotQualityComparisonMuctBaseline_gender_Avg_60percent.png', format='png')
	plt.savefig('PlotQualityComparisonMuctBaseline_gender_Avg_60percent.eps', format='eps')
		#plt.show()
	plt.close()
	
	
	q1_new = np.asarray(q1)
	q2_new = np.asarray(q2)
	q3_new = np.asarray(q3)
	
	t1_new = [sum(e)/len(e) for e in zip(*t1_all)]
	t2_new = [sum(e)/len(e) for e in zip(*t2_all)]
	t3_new = [sum(e)/len(e) for e in zip(*t3_all)]
	
	
	min_val = min(min(q1_new),min(q2_new),min(q3_new))
	max_val = max(max(q1_new),max(q2_new),max(q3_new))
	
	
	
	'''
	plt.plot(t1_new, q1_norm,lw=2,color='green',  label='Baseline1 (Function Based Approach)')
	plt.plot(t2_new, q2_norm,lw=2,color='orange',  label='Baseline2 (Object Based Approach)')
	plt.plot(t3_new, q3_norm,lw=2,color='blue', label='Iterative Approach') ##2,000
	'''
	
	
	q1_norm = (q1_new-min_val)/(max_val - min_val)
	q2_norm = (q2_new-min_val)/(max_val - min_val)
	q3_norm = (q3_new-min_val)/(max_val - min_val)
	
	
	plt.plot(t1_new, q1_norm,lw=2,color='green', marker='o', label='Baseline1 (Function Based Approach)')
	plt.plot(t2_new, q2_norm,lw=2,color='orange',marker='^',  label='Baseline2 (Object Based Approach)')
	plt.plot(t3_new, q3_norm,lw=2,color='blue',marker ='d', label='Iterative Approach') ##2,000
	
	

	
	plt.title('Quality vs Cost')
	#plt.legend(loc="upper left",fontsize='medium')
	plt.xlim([0, max(max(t1_new),max(t2_new),max(t3_new))])
	#plt.ylabel('F1-measure')
	plt.ylabel('Gain')
	plt.xlabel('Cost')
	#plt.ylim([0, 1])
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
	plt.xlim([0, max(max(t1_new),max(t2_new),max(t3_new))])	
	plt.savefig('Image_Muct_Gender_10percent.png', format='png')
	plt.savefig('Image_Muct_Gender_10percent.eps', format='eps')
		#plt.show()
	plt.close()




	
	
	
	
	
def generateDifferentStrategyResults():
	#createSample()
	#createRandomSample()
	
	#q1_all,q2_all,q3_all = [],[],[]
	t1_all,q1_all,t2_all,q2_all,t3_all,q3_all=[],[],[],[],[],[]
	f1 = open('MuctResultsStrategyVariation.txt','w+')
	
	for i in range(1):
		global dl,nl 
		#dl,nl =pickle.load(open('5Samples/MuctTrainGender'+str(i)+'_XY.p','rb'))
		
		imageIndex = [i1 for i1 in sorted(random.sample(xrange(len(dl2)), 400))]
		dl_test = [dl2[i1] for i1 in  imageIndex]
		nl_test = [nl2[i1] for i1 in imageIndex]
	
		
		
		dl = np.array(dl_test)
		nl = np. array(nl_test)
		
		[t1,q1] =adaptiveOrder10(60,4)
		[t2,q2] =adaptiveOrder11(60)
		
		
		t1_all.append(t1)
		t2_all.append(t2)
		q1_all.append(q1)
		q2_all.append(q2)
		
		
		print>>f1,'sameple id : %d'%(i)
		print>>f1,"t1 = {} ".format(t1)
		print>>f1,"q1 = {} ".format(q1)
		print>>f1,"t2 = {} ".format(t2)
		print>>f1,"q2 = {} ".format(q2)
		'''
		plt.plot(t1, q1,lw=2,color='green',  label='Baseline1 (Function Based Approach)')
		plt.plot(t2, q2,lw=2,color='orange',  label='Baseline2 (Object Based Approach)')
		plt.plot(t3, q3,lw=2,color='blue', label='Iterative Approach') ##2,000

		plt.ylim([0, 1])
		plt.xlim([0, 20])
		plt.title('Quality vs Cost')
		plt.legend(loc="lower left",fontsize='medium')
		plt.ylabel('F1-measure')
		plt.xlabel('Cost')	
		plt.savefig('PlotQualityComparisonMuctBaseline_gender'+str(i)+'.png', format='png')
		plt.savefig('PlotQualityComparisonMuctBaseline_gender'+str(i)+'.eps', format='eps')
		#plt.show()
		plt.close()
		'''
		print 'iteration :%d completed'%(i)
	q1 = [sum(e)/len(e) for e in zip(*q1_all)]
	q2 = [sum(e)/len(e) for e in zip(*q2_all)]
	plt.plot(t1, q1,lw=2,color='green', marker='o', label='Strategy(choosing from outside)')
	plt.plot(t2, q2,lw=2,color='blue', marker='d', label='Strategy(choosing from all)')
	
	plt.ylim([0, 1])
	plt.xlim([0, max(max(t1),max(t2))])
	plt.title('Quality vs Cost')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
	plt.ylabel('F1-measure')
	#plt.ylabel('Gain')
	plt.xlabel('Cost')	
	plt.savefig('Strategy Comparison(all vs outside).png', format='png')
	plt.savefig('Strategy Comparison(all vs outside).eps', format='eps')
		#plt.show()
	plt.close()
	
	
	q1_new = np.asarray(q1)
	q2_new = np.asarray(q2)
	#q3_new = np.asarray(q3)
	
	t1_new = [sum(e)/len(e) for e in zip(*t1_all)]
	t2_new = [sum(e)/len(e) for e in zip(*t2_all)]
	#t3_new = [sum(e)/len(e) for e in zip(*t3_all)]
	
	
	min_val = min(min(q1_new),min(q2_new))
	max_val = max(max(q1_new),max(q2_new))
	
	
	
	'''
	plt.plot(t1_new, q1_norm,lw=2,color='green',  label='Baseline1 (Function Based Approach)')
	plt.plot(t2_new, q2_norm,lw=2,color='orange',  label='Baseline2 (Object Based Approach)')
	plt.plot(t3_new, q3_norm,lw=2,color='blue', label='Iterative Approach') ##2,000
	'''
	
	
	q1_norm = (q1_new-min_val)/(max_val - min_val)
	q2_norm = (q2_new-min_val)/(max_val - min_val)
	#q3_norm = (q3_new-min_val)/(max_val - min_val)
	
	
	plt.plot(t1_new, q1_norm,lw=2,color='green', marker='o', label='Strategy(choosing from outside)_100obj')
	plt.plot(t2_new, q2_norm,lw=2,color='blue',marker='^',  label='Strategy(choosing from all)_100obj')
	
	

	
	plt.title('Quality vs Cost')
	#plt.legend(loc="upper left",fontsize='medium')
	plt.xlim([0, max(max(t1_new),max(t2_new))])
	#plt.ylabel('F1-measure')
	plt.ylabel('Gain')
	plt.xlabel('Cost')
	#plt.ylim([0, 1])
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
	plt.xlim([0, max(max(t1_new),max(t2_new))])	
	plt.savefig('Strategy(choosing from outside)100obj_gain.png', format='png')
	plt.savefig('Strategy(choosing from outside)100obj_gain.eps', format='eps')
		#plt.show()
	plt.close()



def generateDifferentStrategyResultsUsingProgressiveScore():
	#createSample()
	#createRandomSample()
	
	#q1_all,q2_all,q3_all = [],[],[]
	t1_all,q1_all,t2_all,q2_all,t3_all,q3_all=[],[],[],[],[],[]
	f1 = open('MuctResultsStrategyVariationUsingProgressive.txt','w+')
	
	budget = 23
	for i in range(4):
		global dl,nl 
		
		imageIndex = [i1 for i1 in sorted(random.sample(xrange(len(dl2)), 50))]
		dl_test = [dl2[i1] for i1 in  imageIndex]
		nl_test = [nl2[i1] for i1 in imageIndex]
	
		dl = np.array(dl_test)
		nl = np. array(nl_test)
		
		[t1,q1] =adaptiveOrder10(budget,float((1*budget)/100))
		[t2,q2] =adaptiveOrder11(budget,float((1*budget)/100))
		
		
		t1_all.append(t1)
		t2_all.append(t2)
		q1_all.append(q1)
		q2_all.append(q2)
		
		
		print>>f1,'sameple id : %d'%(i)
		print>>f1,"t1 = {} ".format(t1)
		print>>f1,"q1 = {} ".format(q1)
		print>>f1,"t2 = {} ".format(t2)
		print>>f1,"q2 = {} ".format(q2)
		print 'iteration :%d completed'%(i)
		
		
	q1 = [sum(e)/len(e) for e in zip(*q1_all)]
	q2 = [sum(e)/len(e) for e in zip(*q2_all)]
	
	epoch_list = [1,2,3,4,5,6,7,8,9,10] 
	score_list = []
	
	t1_new = [sum(e)/len(e) for e in zip(*t1_all)]
	t2_new = [sum(e)/len(e) for e in zip(*t2_all)]
	
	
	q1_new = np.asarray(q1)
	q2_new = np.asarray(q2)
	
	t1_list = [t1_new,t2_new]
	q1_list = [q1_new,q2_new]
	
	
	for i1 in range(len(t1_list)):
		t1_2 = t1_list[i1]
		t1_2 = t1_2[1:]
		q1_2 = q1_list[i1]
		weight_t1 = [max(1-float(element - 1)/budget,0) for element in t1_2]
		improv_q1 = [x - q1_2[i - 1] for i, x in enumerate(q1_2) if i > 0]
		print weight_t1
		print improv_q1
		a1 = np.dot(weight_t1,improv_q1)
		print a1
		score_list.append(a1)
	print>>f1,"epoch_list = {} ".format(epoch_list)
	print>>f1,"score_list = {} ".format(score_list)	
	
	
	
	


def generateMultipleFunctionResult():
	f1 = open('queryTestGenderMuct14.txt','w+')

	#lr,et,rf,ab
	
	#set = [genderPredicate6,genderPredicate1,genderPredicate3,genderPredicate7] # default one
	#set = [genderPredicate17,genderPredicate1,genderPredicate3,genderPredicate7]
	set = [genderPredicate6,genderPredicate1,genderPredicate3,genderPredicate7,
	genderPredicate2,genderPredicate4,genderPredicate5,genderPredicate8,
	genderPredicate9,genderPredicate10,genderPredicate17]
	
	print timeBudget
	outsideObjects=[]
	
	
	#blockList = [200] optimal for bigger dataset.
	#blockList = [100]
	#blockList = [200]
	#blockList = [300]
	blockList = [60]
	#blockList = [800]
	
	
	
	executionPerformed = 0
	thinkTimeList = []
	executionTimeList = []
	#for percent in thinkPercentList:
	realF1List = []
	for block in blockList:
		#totalAllowedExecution = 1000
		executionPerformed = 0
		# The dictionary currentProbability stores the information about the output of previously ran classifiers. Suppose image 20 has gone through c2 and c3 and the output probability was 0.5 and 0.6. 
		# Then the hashmap element of image 20 will be as follows:  20: [0,0.5,0.6,0]. 
		currentProbability = {}
		for i in range(len(dl)):
			key = i
			#value = [-1,-1,-1,-1]
			value = [-1]*len(set)
			print value
			if key in currentProbability:
				currentProbability[key].append(value)
			else:
				currentProbability[key] = [value]			
		#print currentProbability
		
		
		# The dictionary prevClassifier stores the information about previously ran classifiers. Suppose image 20 has gone through c2 and c3. Then the hashmap element of image 20 will be as follows:  20: [0,1,1,0]. 
		# The bit vector corresponding to classifier 2 and classifier 3 are set.
		prevClassifier = {}
		for i in range(len(dl)):
			key = i
			#value = [0,0,0,0]
			value = [0]*len(set)
			if key in prevClassifier:
				prevClassifier[key].append(value)
			else:
				prevClassifier[key] = [value]
				
		#print prevClassifier
		
		#currentUncertainty list stores the information of current uncertainty of all the images.
		
		currentUncertainty = [0.99]*len(dl)
		currentF1measure = 0
		#print currentUncertainty
		operator = set[0]
		count = 0
		totalExecutionTime = 0
		totalThinkTime = 0
		topKIndexes= [0]
		timeElapsed = 0
		timeList =[]
		f1List =[]
		blockSize = 1	
		executionTime = 0
		
		stepSize =4  #step size of 20 seconds. After every 20 seconds evaluate the quality
		currentTimeBound = 4
	
		t11 = 0
		t12 = 0
		
		while True:		
			#t11 = time.time()
			#for i in topKIndexes:
		
			if count ==0:
				t1 = time.time()
				operator = set[0]
				#dl_new =  pca.fit_transform(dl)
				#dl_new = np.array(dl_new)
				#print np.array(dl_new[0])
				#probX = operator(np.array(dl_new[0][0]))
				#probX = gender_dt_new.predict_proba([dl_new])
				#print probX
				for i in range(len(dl)):
					probValues = operator([dl[i]])
					'''
					indexClf = set.index(operator)
					tempProb = currentProbability[i][0]
					tempProb[indexClf] = probValues[0]
					'''
					
					indexClf = set.index(operator)
					tempProb = currentProbability[i][0]
					#print probValues[0]
					if probValues[0] > 0.55:
						tempProb[indexClf] = probValues[0]+0.2
					else:
						tempProb[indexClf] = probValues[0]-0.3
					
					# setting the bit for the corresponding classifier
					tempClf = prevClassifier[i][0]
					tempClf[indexClf] = 1
					
					
					# calculating the current cobined probability
					combinedProbability = combineProbability(currentProbability[i])
					
					# using the combined probability value to calculate uncertainty
					uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
					currentUncertainty[i] = uncertainty
					
				t2 = time.time()
				executionTime = executionTime + (t2- t1)
				#set.remove(genderPredicate8)
	
				qualityOfAnswer = findQuality(currentProbability)
				#print 'returned images'
				#print qualityOfAnswer[3]
				#print>>f1,'size of answer set : %d'%(len(qualityOfAnswer[3]))
				if len(qualityOfAnswer[3]) > 0:
					realF1 = findRealF1(qualityOfAnswer[3])
				else: 
					realF1 = 0
				#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
				f1measure = realF1
				timeList.append(0)
				f1List.append(f1measure)
				#currentTimeBound = currentTimeBound + stepSize
				#print>>f1,'time bound completed:%d'%(currentTimeBound)
	
	
			#t11 = time.time()
			if count >0:
				
				
				for w in range(4):
					tempClfList = ['DT','GNB','RF','KNN']
					#print>>f1,"w = %d"%(w)	
					#print>>f1, tempClfList[w]					
					imageIndex = [item for item in topKIndexes if nextBestClassifier[item] == tempClfList[w]]
					if w!=4:
						operator = set[w]
					#print>>f1,operator
					images = [dl[k] for k in imageIndex]
					
					#print>>f1,"images to be run with this operator : {} ".format(imageIndex)
					
					if len(imageIndex) >0:
						#probValues = operator(images)
					######## Executing the function on all the objects ###########								
						for i1 in range(len(imageIndex)):
							t11 = time.time()		
							#probValues = operator(dl[i])
							#rocProb = probValues
							#rocProb = probValues[i1] //
							
							probValues = operator([images[i1]])
							rocProb = probValues
						
							#print>>f1,"i1 : %d"%(i1)
							#finding index of classifier
							indexClf = w						
							tempProb = currentProbability[imageIndex[i1]][0]
							tempProb[indexClf] = rocProb
							#print>>f1,"image : %d"%(imageIndex[i1])
							#print>>f1,"currentProbability: {}".format(currentProbability[imageIndex[i1]][0])
							
							#print currentProbability[imageIndex[i]]
							#if count !=0:
							#print nextBestClassifier[imageIndex[i]]
							
							# setting the bit for the corresponding classifier
							tempClfList = prevClassifier[imageIndex[i1]][0]
							tempClfList[indexClf] = 1
							#tempClfList2 = prevClassifier.get(outsideObjects[i1])
							#print>>f1,"prev classifier for image : %d"%(imageIndex[i1])
							#print>>f1,"prevClassifier: {}".format(prevClassifier[imageIndex[i1]][0])
							
							# calculating the current cobined probability
							combinedProbability = combineProbability(currentProbability[imageIndex[i1]])
							
						# using the combined probability value to calculate uncertainty
							uncertainty = -combinedProbability* np.log2(combinedProbability) - (1- combinedProbability)* np.log2(1- combinedProbability)
							currentUncertainty[imageIndex[i1]] = uncertainty
							
							
							t12 = time.time()
							totalExecutionTime = totalExecutionTime + (t12-t11)	
							timeElapsed = timeElapsed +(t12-t11)	
							
							if timeElapsed > currentTimeBound:
								qualityOfAnswer = findQuality(currentProbability)
				
								if len(qualityOfAnswer[3]) > 0:
									realF1 = findRealF1(qualityOfAnswer[3])
								else: 
									realF1 = 0
								#print>>f1,'real F1 : %f'%(realF1)
								#f1measure = qualityOfAnswer[0]
								f1measure = realF1
								timeList.append(timeElapsed)
								f1List.append(f1measure)
								#print>>f1,'F1 list : {}'.format(f1List)
				
								#print 'time bound completed:%d'%(currentTimeBound)	
								#print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
								#print 'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				
								currentTimeBound = currentTimeBound + stepSize
								break 
								
							if timeElapsed > timeBudget:
								break
							
							
					
					#executionTimeList.append(t12-t11)
					
					imageIndex[:]=[]
					images[:]=[]
					#print>>f1,"Outside of inner for loop"
					#continue
							
				#print>>f1,"Finished executing four functions"
					#imageIndex[:]=[]
					#images[:] =[]
					#topKIndexes[:]=[]
					#probValues[:]=[]
			'''
			t12 = time.time()
			totalExecutionTime = totalExecutionTime + (t12-t11)	
			executionTimeList.append(t12-t11)
			'''
			executionTimeList.append(totalExecutionTime)
					
				
			
			nextBestClassifier = [-1]*len(dl)
			deltaUncertainty = [0] *len(dl)
			benefitArray =[float(-10)]* len(dl) 
			topKIndexes = [0] * len(dl) # initial block size.
			
			#currentTempProbability = copy.deepcopy(currentProbability)
			newUncertaintyValue = 0 #initializing
			
			#### Think phase starts
			# calculating benefit of each objects. Benefit is measured in terms of improvement in f1 measure.
			t21 = time.time()
			# first determining the objects which are not in answer set
			qualityOfAnswer = findQuality(currentProbability)
			currentAnswerSet = qualityOfAnswer[3]
			allObjects = list(range(0,len(dl)))
			
			#outsideObjects = allObjects
			
			### Uncomment this part for choosing objects from outside of the answer set.
			
			if count !=0:
				outsideObjects = [x for x in allObjects if x not in currentAnswerSet]
			
			
			
			#print>>f1,"inside objects : {} ".format(currentAnswerSet)
			#print>>f1,"length of inside objects : %f"%len(currentAnswerSet)
			stateListInside =[]
			stateListInside = findStates(currentAnswerSet,prevClassifier)
			
			#print>>f1,"state of inside objects: {}".format(stateListInside)
			
			#print>>f1,"outsideObjects : {} ".format(outsideObjects)
			#print>>f1,"length of outsideObjects : %f"%len(outsideObjects)
			stateListOutside =[]
			stateListOutside = findStates(outsideObjects,prevClassifier)
			#print>>f1,"state of outside objects: {}".format(stateListOutside)
			
			
			if(len(outsideObjects)==0 and count !=0):
				break
			
			
			#print>>f1,"outsideObjects : {} ".format(outsideObjects)
			#print 'count=%d'%(count)
			
			for j in range(len(outsideObjects)):
				#print>>f1,'deciding for object %d'%(outsideObjects[j])
				#print>>f1,"currentUncertainty: {}".format(currentUncertainty)
				[nextBestClassifier[outsideObjects[j]],deltaUncertainty[outsideObjects[j]]] = chooseNextBest(prevClassifier.get(outsideObjects[j])[0],currentUncertainty[outsideObjects[j]])	
				newUncertaintyValue = currentUncertainty[outsideObjects[j]]  + float(deltaUncertainty[outsideObjects[j]])
				newProbabilityValue1 = convertEntropyToProb(newUncertaintyValue)
				#print 'newUncertaintyValue:%f'%(newUncertaintyValue)
				
				
				#finding index of classifier
				#indexTempProbClf = set.index(nextBestClassifier[j])
				if nextBestClassifier[outsideObjects[j]] == 'DT':
					nextBestClassifier[outsideObjects[j]] = 'NA'
				if nextBestClassifier[outsideObjects[j]] == 'GNB':
					indexTempProbClf = 1
				if nextBestClassifier[outsideObjects[j]] == 'RF':
					indexTempProbClf = 2
				if nextBestClassifier[outsideObjects[j]] == 'KNN':
					indexTempProbClf = 3
				
				# higher probability value	
				newProbabilityValue2 = 1 - newProbabilityValue1			
		
				#benefit is  (pi * pi_new)/cost(i) 
				probability_i = combineProbability(currentProbability[outsideObjects[j]])
				#print 'probability_i: %f, new probability : %f, cost : %f'%(probability_i,newProbabilityValue2,cost(nextBestClassifier[j]))
				if cost(nextBestClassifier[outsideObjects[j]]) != 0:
					benefit = float((probability_i*newProbabilityValue2)/float(cost(nextBestClassifier[outsideObjects[j]])))
					benefitArray[outsideObjects[j]] = benefit
				else:
					benefitArray[outsideObjects[j]] = -1
				
			#seq = sorted(benefitArray)
			#print benefitArray
			#Ordering the objects based on deltaUncertainty Value
			#order = [seq.index(v) for v in benefitArray]
			#topIndex= benefitArray.index(max(benefitArray))
			
			#topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			
			
			if len(outsideObjects) < blockSize :
				topKIndexes = outsideObjects
				#topKIndexes = heapq.nlargest(len(outsideObjects), range(len(outsideObjects)), benefitArray.__getitem__)
			else:
				topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			
			'''
			if len(outsideObjects) > (blockSize/4):
				topKIndexes = [x for x in topKIndexes if x not in currentAnswerSet]
			else:
				topKIndexes = heapq.nlargest(blockSize, range(len(benefitArray)), benefitArray.__getitem__)
			'''
			#print 'top index:%d'%(topIndex)
			
			t22 = time.time()
			totalThinkTime = totalThinkTime + (t22-t21)
			thinkTimeList.append(t22-t21)
			
			print>>f1,"inside objects : {} ".format(currentAnswerSet)
			print>>f1,"state of inside objects: {}".format(stateListInside)
			print>>f1,"outsideObjects : {} ".format(outsideObjects)
			print>>f1,"state of outside objects: {}".format(stateListOutside)
			
			'''
			if(all(element==0 or element==-1 for element in benefitArray) and count >20):
				break
			'''
			#i=topIndex #next image to be run
			t2 = time.time()
			
			timeElapsed = totalExecutionTime + totalThinkTime
			
			#timeList.append(timeElapsed)
			#print 'next images to be run'
			#print topKIndexes
			
			print>>f1,'benefit array: {}'.format(benefitArray)
			print>>f1,'next images to be run: {}'.format(topKIndexes)
			classifierSet = [nextBestClassifier[item2] for item2 in topKIndexes]
			allClassifierSet = [nextBestClassifier[item3] for item3 in allObjects]
			'''
			if(all(element=='NA' for element in classifierSet) and count > 20):
				break
			'''
			if(all(element=='NA' for element in classifierSet) and count > 20):
				topKIndexes = allObjects
				#break
			if(all(element=='NA' for element in allClassifierSet) and count > 20):
				break
			#print>>f1,'classifier set: {}'.format(classifierSet)
			benefitArray[:] =[]
			classifierSet[:] = []
			
			#print 'round %d completed'%(count)
			#print 'time taken %f'%(timeElapsed)
			
			# block size is determined in this part.
			if count ==0:
				blockSize = block
				topKIndexes[:]= []
				#print 'blockSize: %d'%(blockSize)
			
			
			
			
			###### Time check########
			
			if timeElapsed > currentTimeBound:
				qualityOfAnswer = findQuality(currentProbability)
				
				if len(qualityOfAnswer[3]) > 0:
					realF1 = findRealF1(qualityOfAnswer[3])
				else: 
					realF1 = 0
				#print>>f1,'real F1 : %f'%(realF1)
					#f1measure = qualityOfAnswer[0]
				f1measure = realF1
				timeList.append(timeElapsed)
				f1List.append(f1measure)
				'''
				f1measure = qualityOfAnswer[0]
				timeList.append(timeElapsed)
				f1List.append(f1measure)
				'''
				#print 'time bound completed:%d'%(currentTimeBound)	
				#print>>f1,'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				#print 'f1 measure of the answer set: %f, precision:%f, recall:%f, executionTime:%f, thinkTime:%f, timeElapsed:%f '%(f1measure,qualityOfAnswer[1],qualityOfAnswer[2],totalExecutionTime,totalThinkTime,timeElapsed)
				
				currentTimeBound = currentTimeBound + stepSize
				
					
			

			if timeElapsed > timeBudget:
				break
			
			
			'''
			executionPerformed = executionPerformed + 	blockSize	
			if(executionPerformed>totalAllowedExecution):
				qualityOfAnswer = findQuality(currentProbability)
				print 'returned images'
				print qualityOfAnswer[3]
				if len(qualityOfAnswer[3]) > 0 :
					realF1 = findRealF1(qualityOfAnswer[3])
				else:
					realF1 = 0
				print 'real F1 : %f'%(realF1)
				
	
				#f1measure = qualityOfAnswer[0]
				f1measure = realF1
				f1measurePerAction = float(f1measure)/totalAllowedExecution
				timeList.append(timeElapsed)
				f1List.append(f1measurePerAction)
				realF1List.append(f1measure)
				
				print>>f1,'block size:%f'%(blockSize)
				print>>f1,'returned images: {}'.format(qualityOfAnswer[3])
				print>>f1,'length of answer set:%f'%len(qualityOfAnswer[3])
				unprocessedObjects = findUnprocessed(currentProbability)
				print>>f1,'unprocessed objects : {} '.format(unprocessedObjects)
				print>>f1,'length of unprocessed objects:%f'%(len(unprocessedObjects))
				print>>f1,'think time list: {}'.format(thinkTimeList)
				print>>f1,'execution time list: {}'.format(executionTimeList)
				
				break
				'''
			#if count >= 5000:
			#	break
			count=count+1
			
		plt.title('Quality vs Time Value')
		#print>>f1,'percent : %f'%(percent)
		#print>>f1,'block size : %f'%(block)
		#print>>f1,"f1 measures : {} ".format(realF1List)
		#print>>f1,'total think time :%f'%(totalThinkTime)
		#print>>f1,'total execution time :%f'%(totalExecutionTime)
		plt.ylabel('Quality')
		plt.xlabel('time')
		xValue = timeList
		yValue = f1List
		#print>>f1,"x value : {} ".format(xValue)
		#print>>f1,"y value : {} ".format(yValue)
		#print "x value : {} ".format(xValue)
		#print "y value : {} ".format(yValue)
	
		plt.plot(xValue, yValue,'b')
		plt.ylim([0, 1])
		plt.legend(loc="upper left")
		plt.savefig('plotQualityAdaptive8'+str(block)+'.eps',format = 'eps')
		plt.title('Quality vs Time for block size = '+str(block))
		#plt.show()
		plt.close()
		#xValue = timeList
		#yValue = f1List
		
	'''
	plt.ylabel('Quality')
	plt.xlabel('Block Size')
	xValue = blockList
	yValue = realF1List
	print>>f1,"x value : {} ".format(xValue)
	print>>f1,"y value : {} ".format(yValue)
	print "x value : {} ".format(xValue)
	print "y value : {} ".format(yValue)
	'''
	'''
	plt.ylabel('Quality')
	plt.xlabel('time')
	xValue = timeList
	yValue = f1List
	print>>f1,"x value : {} ".format(xValue)
	print>>f1,"y value : {} ".format(yValue)
	print "x value : {} ".format(xValue)
	print "y value : {} ".format(yValue)
	'''
	#yValue = f1measure
	#yValue = f1measurePerAction
	#labelValue = 'Adaptive algorithm(Think='+str(percent)+'%)'
	#labelValue = 'Adaptive algorithm(Block size='+str(block)+')'
	plt.plot(xValue, yValue,'g')
	plt.ylim([0, 1])
	plt.legend(loc="upper left")
	plt.savefig('plotQualityAdaptive14.png')
	#plt.show()
	plt.close()
	return [timeList,f1List]




def generateDifferentStrategyResults_benefit_estimation():
	
	t1_all,q1_all,t2_all,q2_all,t3_all,q3_all=[],[],[],[],[],[]
	f1 = open('MuctResultsStrategyVariation_benefit_estimation.txt','w+')
	
	for i in range(4):
		global dl,nl 
		#dl,nl =pickle.load(open('5Samples/MuctTrainGender'+str(i)+'_XY.p','rb'))
		
		
		imageIndex = [i1 for i1 in sorted(random.sample(xrange(len(dl2)), 50))]
		dl_test = [dl2[i1] for i1 in  imageIndex]
		nl_test = [nl2[i1] for i1 in imageIndex]
	
		
		
		dl = np.array(dl_test)
		nl = np. array(nl_test)
		
		[t1,q1] =adaptiveOrder12(60,4)
		[t2,q2] =adaptiveOrder8(60,4)
		
		
		t1_all.append(t1)
		t2_all.append(t2)
		q1_all.append(q1)
		q2_all.append(q2)
		
		
		print>>f1,'sameple id : %d'%(i)
		print>>f1,"t1 = {} ".format(t1)
		print>>f1,"q1 = {} ".format(q1)
		print>>f1,"t2 = {} ".format(t2)
		print>>f1,"q2 = {} ".format(q2)
		'''
		plt.plot(t1, q1,lw=2,color='green',  label='Baseline1 (Function Based Approach)')
		plt.plot(t2, q2,lw=2,color='orange',  label='Baseline2 (Object Based Approach)')
		plt.plot(t3, q3,lw=2,color='blue', label='Iterative Approach') ##2,000

		plt.ylim([0, 1])
		plt.xlim([0, 20])
		plt.title('Quality vs Cost')
		plt.legend(loc="lower left",fontsize='medium')
		plt.ylabel('F1-measure')
		plt.xlabel('Cost')	
		plt.savefig('PlotQualityComparisonMuctBaseline_gender'+str(i)+'.png', format='png')
		plt.savefig('PlotQualityComparisonMuctBaseline_gender'+str(i)+'.eps', format='eps')
		#plt.show()
		plt.close()
		'''
		print 'iteration :%d completed'%(i)
	q1 = [sum(e)/len(e) for e in zip(*q1_all)]
	q2 = [sum(e)/len(e) for e in zip(*q2_all)]
	
	
	plt.plot(t1, q1,lw=2,color='green', marker='o', label='Benefit Estimation(without optimization)')
	plt.plot(t2, q2,lw=2,color='blue', marker='d', label='Benefit Estimation(with optimization)')
	
	plt.ylim([0, 1])
	plt.xlim([0, max(max(t1),max(t2))])
	plt.title('Quality vs Cost')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.,fontsize='medium')
	plt.ylabel('F1-measure')
	#plt.ylabel('Gain')
	plt.xlabel('Cost')	
	plt.savefig('Strategy Comparison F1 measure(benefit estimation).png', format='png')
	plt.savefig('Strategy Comparison F1 measure(benefit estimation).eps', format='eps')
		#plt.show()
	plt.close()
	
	
	q1_new = np.asarray(q1)
	q2_new = np.asarray(q2)
	#q3_new = np.asarray(q3)
	
	t1_new = [sum(e)/len(e) for e in zip(*t1_all)]
	t2_new = [sum(e)/len(e) for e in zip(*t2_all)]
	#t3_new = [sum(e)/len(e) for e in zip(*t3_all)]
	
	
	min_val = min(min(q1_new),min(q2_new))
	max_val = max(max(q1_new),max(q2_new))
	
	
	
	'''
	plt.plot(t1_new, q1_norm,lw=2,color='green',  label='Baseline1 (Function Based Approach)')
	plt.plot(t2_new, q2_norm,lw=2,color='orange',  label='Baseline2 (Object Based Approach)')
	plt.plot(t3_new, q3_norm,lw=2,color='blue', label='Iterative Approach') ##2,000
	'''
	
	
	q1_norm = (q1_new-min_val)/(max_val - min_val)
	q2_norm = (q2_new-min_val)/(max_val - min_val)
	#q3_norm = (q3_new-min_val)/(max_val - min_val)
	
	
	
	plt.plot(t2_new, q2_norm,lw=2,color='blue',marker='d',  label='Benefit Estimation(with optimization)')
	plt.plot(t1_new, q1_norm,lw=2,color='green', marker='o', label='Benefit Estimation(without optimization)')
	
	

	
	plt.title('Quality vs Cost')
	#plt.legend(loc="upper left",fontsize='medium')
	plt.xlim([0, max(max(t1_new),max(t2_new))])
	#plt.ylabel('F1-measure')
	plt.ylabel('Gain')
	plt.xlabel('Cost')
	#plt.ylim([0, 1])
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.,fontsize='medium')
	plt.xlim([0, max(max(t1_new),max(t2_new))])	
	plt.savefig('Strategy Comparison Gain(benefit estimation).png', format='png')
	plt.savefig('Strategy Comparison Gain(benefit estimation).eps', format='eps')
		#plt.show()
	plt.close()






def generateDifferentStrategyResults_benefit_estimation_progressive_score():
	#createSample()
	#createRandomSample()
	
	#q1_all,q2_all,q3_all = [],[],[]
	t1_all,q1_all,t2_all,q2_all,t3_all,q3_all=[],[],[],[],[],[]
	f1 = open('MuctResultsStrategyVariation_benefit_estimation.txt','w+')
	
	budget = 90
	
	for i in range(10):
		global dl,nl 
		
		imageIndex = [i1 for i1 in sorted(random.sample(xrange(len(dl2)), 200))]
		dl_test = [dl2[i1] for i1 in  imageIndex]
		nl_test = [nl2[i1] for i1 in imageIndex]
	
		
		
		dl = np.array(dl_test)
		nl = np. array(nl_test)
		
		
		
		[t1,q1] =adaptiveOrder12(budget,float((5*budget)/100))
		[t2,q2] =adaptiveOrder10(budget,float((5*budget)/100))
		
				
		t1_all.append(t1)
		t2_all.append(t2)
		q1_all.append(q1)
		q2_all.append(q2)
		
		
		print>>f1,'sameple id : %d'%(i)
		print>>f1,"t1 = {} ".format(t1)
		print>>f1,"q1 = {} ".format(q1)
		print>>f1,"t2 = {} ".format(t2)
		print>>f1,"q2 = {} ".format(q2)
		print 'iteration :%d completed'%(i)
	
	
	q1 = [sum(e)/len(e) for e in zip(*q1_all)]
	q2 = [sum(e)/len(e) for e in zip(*q2_all)]
	
	epoch_list = [1,2,3,4,5,6,7,8,9,10] 
	score_list = []
	
	t1_new = [sum(e)/len(e) for e in zip(*t1_all)]
	t2_new = [sum(e)/len(e) for e in zip(*t2_all)]
	
	
	q1_new = np.asarray(q1)
	q2_new = np.asarray(q2)
	
	t1_list = [t1_new,t2_new]
	q1_list = [q1_new,q2_new]
	
	
	for i1 in range(len(t1_list)):
		t1_2 = t1_list[i1]
		t1_2 = t1_2[1:]
		q1_2 = q1_list[i1]
		weight_t1 = [max(1-float(element - 1)/budget,0) for element in t1_2]
		improv_q1 = [x - q1_2[i - 1] for i, x in enumerate(q1_2) if i > 0]
		print weight_t1
		print improv_q1
		a1 = np.dot(weight_t1,improv_q1)
		print a1
		score_list.append(a1)
	print>>f1,"epoch_list = {} ".format(epoch_list)
	print>>f1,"score_list = {} ".format(score_list)	
	
	


def plotResults2():
	
	t1_all,q1_all,t2_all,q2_all,t3_all,q3_all=[],[],[],[],[],[]
	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.6621621621621622, 0.6713286713286712, 0.6906474820143884, 0.7121212121212123, 0.7286821705426357, 0.7642276422764227, 0.7966101694915255, 0.8245614035087718, 0.8392857142857143, 0.8440366972477065] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.6621621621621622, 0.6621621621621622, 0.6575342465753425, 0.6620689655172414, 0.6713286713286712, 0.6857142857142856, 0.6906474820143884, 0.7058823529411765, 0.7058823529411765] 
	t3 = [0.24915099143981934, 4.967166900634766, 10.780376195907593, 14.043654203414917, 21.566923141479492] 
	q3 = [0.6621621621621622, 0.8440366972477065, 0.8440366972477065, 0.8440366972477065, 0.8440366972477065] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)



	t1 = [0, 4, 6, 8, 10, 12, 14, 16] 
	q1 = [0.6438356164383562, 0.6714285714285714, 0.6762589928057554, 0.6917293233082706, 0.7076923076923076, 0.7244094488188976, 0.7301587301587302, 0.7301587301587302] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.6438356164383562, 0.6482758620689655, 0.6527777777777778, 0.6619718309859155, 0.6714285714285714, 0.6762589928057554, 0.6762589928057554, 0.6762589928057554, 0.6762589928057554] 
	t3 = [0.20136594772338867, 6.235433340072632, 12.083542346954346, 15.582733154296875, 22.79766321182251] 
	q3 = [0.6438356164383562, 0.8571428571428572, 0.8490566037735849, 0.8490566037735849, 0.8411214953271028] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)

	
	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.6344827586206897, 0.6618705035971223, 0.676470588235294, 0.6917293233082707, 0.7244094488188977, 0.7666666666666667, 0.7719298245614036, 0.822429906542056, 0.8282828282828283, 0.8571428571428572] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.6344827586206897, 0.6344827586206897, 0.6388888888888888, 0.6433566433566433, 0.6524822695035462, 0.6571428571428571, 0.6618705035971223, 0.6618705035971223, 0.6666666666666666] 
	t3 = [0.19667506217956543, 7.405038595199585, 11.393654584884644, 16.14687442779541, 24.095147371292114] 
	q3 = [0.6344827586206897, 0.8316831683168316, 0.8235294117647057, 0.8235294117647057, 0.8155339805825242] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)

	
	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.6973684210526315, 0.6986301369863013, 0.7285714285714284, 0.75, 0.7518796992481204, 0.7596899224806201, 0.7903225806451613, 0.8099173553719008, 0.8035714285714285, 0.7818181818181817] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.6973684210526315, 0.6973684210526315, 0.7114093959731543, 0.7162162162162162, 0.7210884353741497, 0.7260273972602741, 0.7310344827586206, 0.7310344827586206, 0.736111111111111] 
	t3 = [0.2693510055541992, 10.178794622421265, 15.578510522842407, 24.171244382858276] 
	q3 = [0.6973684210526315, 0.7678571428571428, 0.7678571428571428, 0.7787610619469026] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)


	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.6344827586206897, 0.6666666666666667, 0.6766917293233083, 0.6923076923076923, 0.7200000000000001, 0.7457627118644068, 0.7652173913043478, 0.7889908256880734, 0.8037383177570093, 0.8076923076923076] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.6344827586206897, 0.6388888888888888, 0.647887323943662, 0.6428571428571429, 0.6474820143884893, 0.6521739130434782, 0.6666666666666667, 0.6716417910447762, 0.6766917293233083] 
	t3 = [0.19946002960205078, 5.001859188079834, 10.481369018554688, 13.79622197151184, 21.454269886016846] 
	q3 = [0.6344827586206897, 0.8037383177570093, 0.8037383177570093, 0.8037383177570093, 0.7962962962962964] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)


	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.6711409395973155, 0.6756756756756758, 0.6896551724137931, 0.6857142857142858, 0.7164179104477612, 0.7441860465116279, 0.7741935483870968, 0.7899159663865546, 0.8035714285714286, 0.822429906542056] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.6711409395973155, 0.6577181208053691, 0.6577181208053691, 0.6577181208053691, 0.6621621621621622, 0.6621621621621622, 0.6666666666666666, 0.6666666666666666, 0.6758620689655174] 
	t3 = [0.1973259449005127, 10.52013087272644, 12.00377106666565, 20.853950023651123] 
	q3 = [0.6711409395973155, 0.8073394495412843, 0.8148148148148148, 0.7927927927927928] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)



	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.6799999999999999, 0.7034482758620689, 0.7132867132867133, 0.7285714285714285, 0.7555555555555554, 0.7874015748031497, 0.8196721311475409, 0.8596491228070176, 0.9074074074074074, 0.9056603773584905] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.6799999999999999, 0.6891891891891893, 0.6938775510204082, 0.6986301369863014, 0.7083333333333333, 0.7183098591549296, 0.7234042553191489, 0.7234042553191489, 0.7338129496402878] 
	t3 = [0.2005159854888916, 6.001028537750244, 11.585296630859375, 15.19298791885376, 23.67618989944458] 
	q3 = [0.6799999999999999, 0.897196261682243, 0.8888888888888888, 0.897196261682243, 0.8807339449541284] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)


	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.625, 0.6428571428571429, 0.6521739130434783, 0.6870229007633588, 0.6929133858267716, 0.7096774193548387, 0.7333333333333334, 0.7521367521367521, 0.8000000000000002, 0.8190476190476191] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.625, 0.6338028169014084, 0.6338028169014084, 0.6382978723404256, 0.6428571428571429, 0.6521739130434783, 0.6569343065693432, 0.6666666666666666, 0.6716417910447761] 
	t3 = [0.20403695106506348, 5.179411888122559, 10.935384511947632, 14.012794733047485, 21.956069707870483] 
	q3 = [0.625, 0.8076923076923076, 0.8076923076923076, 0.8076923076923076, 0.8076923076923076] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)

	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.6799999999999999, 0.6666666666666666, 0.6857142857142857, 0.6811594202898551, 0.7121212121212122, 0.7187500000000001, 0.7479674796747967, 0.773109243697479, 0.7927927927927927, 0.8148148148148149] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16] 
	q2 = [0.6799999999999999, 0.6756756756756755, 0.6620689655172413, 0.6666666666666666, 0.676056338028169, 0.6808510638297872, 0.6857142857142857, 0.6906474820143884] 
	t3 = [0.22991514205932617, 11.435917854309082, 14.277423620223999, 22.449156522750854] 
	q3 = [0.6799999999999999, 0.7826086956521738, 0.7826086956521738, 0.7826086956521738] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)

	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.6711409395973155, 0.6857142857142858, 0.711111111111111, 0.717557251908397, 0.7580645161290321, 0.7666666666666667, 0.7758620689655172, 0.8181818181818182, 0.822429906542056, 0.826923076923077] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.6711409395973155, 0.684931506849315, 0.6896551724137931, 0.6805555555555556, 0.6713286713286712, 0.6808510638297873, 0.6857142857142858, 0.6762589928057553, 0.681159420289855] 
	t3 = [0.197221040725708, 6.169342041015625, 10.570740222930908, 14.289059162139893, 21.557047128677368] 
	q3 = [0.6711409395973155, 0.8349514563106797, 0.8349514563106797, 0.8349514563106797, 0.826923076923077] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)

	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.6344827586206897, 0.661764705882353, 0.6515151515151515, 0.6771653543307087, 0.7049180327868853, 0.7350427350427351, 0.7678571428571428, 0.7889908256880734, 0.826923076923077, 0.826923076923077] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.6344827586206897, 0.6388888888888888, 0.6382978723404256, 0.6330935251798562, 0.6423357664233577, 0.6518518518518518, 0.6567164179104478, 0.6666666666666667, 0.6717557251908397] 
	t3 = [0.19465398788452148, 6.043243885040283, 10.410318851470947, 14.057048797607422, 21.22432589530945] 
	q3 = [0.6344827586206897, 0.826923076923077, 0.8113207547169811, 0.8113207547169811, 0.8037383177570093] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)

	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.6621621621621622, 0.6805555555555556, 0.6861313868613138, 0.7067669172932332, 0.7460317460317459, 0.7768595041322315, 0.8141592920353982, 0.8333333333333333, 0.8301886792452831, 0.838095238095238] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.6621621621621622, 0.6666666666666666, 0.6712328767123288, 0.6712328767123288, 0.6712328767123288, 0.6666666666666666, 0.6760563380281689, 0.6760563380281689, 0.6808510638297872] 
	t3 = [0.2078099250793457, 7.92710542678833, 10.462743282318115, 15.899851083755493, 24.107229948043823] 
	q3 = [0.6621621621621622, 0.822429906542056, 0.822429906542056, 0.8333333333333333, 0.8440366972477065] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)

	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.7225806451612903, 0.7320261437908496, 0.7368421052631579, 0.7516778523489933, 0.7801418439716312, 0.8029197080291971, 0.8270676691729324, 0.8437499999999999, 0.8571428571428571, 0.8376068376068376] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.7225806451612903, 0.7225806451612903, 0.7225806451612903, 0.7272727272727273, 0.7320261437908496, 0.7417218543046357, 0.7516778523489933, 0.7432432432432433, 0.7432432432432433] 
	t3 = [0.22031021118164062, 11.140376567840576, 13.095251560211182, 21.30521559715271] 
	q3 = [0.7225806451612903, 0.85, 0.85, 0.85] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)

	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.6621621621621622, 0.6805555555555556, 0.6950354609929079, 0.7205882352941176, 0.732824427480916, 0.7619047619047619, 0.793388429752066, 0.8245614035087718, 0.8392857142857143, 0.8301886792452831] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.6621621621621622, 0.6621621621621622, 0.6621621621621622, 0.6666666666666666, 0.6758620689655173, 0.6805555555555556, 0.6853146853146853, 0.6950354609929079, 0.7000000000000001] 
	t3 = [0.2124640941619873, 7.486321926116943, 10.964564085006714, 15.895547866821289, 23.19960594177246] 
	q3 = [0.6621621621621622, 0.8333333333333333, 0.8333333333333333, 0.8333333333333333, 0.8256880733944955] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)

	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.6799999999999999, 0.684931506849315, 0.7092198581560284, 0.7111111111111111, 0.7328244274809161, 0.7619047619047621, 0.7804878048780487, 0.8034188034188035, 0.8256880733944953, 0.8333333333333333] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.6799999999999999, 0.6845637583892618, 0.6802721088435374, 0.689655172413793, 0.6944444444444443, 0.6901408450704225, 0.6950354609929078, 0.7000000000000001, 0.7101449275362318] 
	t3 = [0.19323301315307617, 7.06549596786499, 12.481868982315063, 15.995145082473755, 23.69118309020996] 
	q3 = [0.6799999999999999, 0.8108108108108107, 0.8108108108108107, 0.8035714285714285, 0.8035714285714285] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)

	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.7777777777777778, 0.759493670886076, 0.7741935483870968, 0.7682119205298014, 0.7837837837837838, 0.8028169014084506, 0.8175182481751826, 0.8527131782945736, 0.864, 0.864] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.7777777777777778, 0.7749999999999998, 0.7749999999999998, 0.7848101265822786, 0.7870967741935483, 0.7870967741935483, 0.7792207792207793, 0.7792207792207793, 0.7792207792207793] 
	t3 = [0.20943999290466309, 5.06219482421875, 11.56827187538147, 13.94556474685669, 20.09172296524048] 
	q3 = [0.7777777777777778, 0.864, 0.864, 0.864, 0.864] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)

	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.6621621621621622, 0.6950354609929079, 0.7205882352941176, 0.7313432835820896, 0.7272727272727273, 0.7384615384615384, 0.752, 0.7500000000000001, 0.8, 0.7850467289719627] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.6621621621621622, 0.6666666666666666, 0.6758620689655173, 0.6758620689655173, 0.6573426573426573, 0.6619718309859155, 0.6666666666666666, 0.6762589928057555, 0.6861313868613138] 
	t3 = [0.20877599716186523, 10.945090055465698, 13.700244188308716, 21.975874423980713] 
	q3 = [0.6621621621621622, 0.7706422018348624, 0.7706422018348624, 0.75] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)

	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.653061224489796, 0.6527777777777778, 0.6714285714285714, 0.6911764705882353, 0.7067669172932329, 0.7401574803149605, 0.7642276422764227, 0.8034188034188035, 0.8103448275862069, 0.8392857142857143] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.653061224489796, 0.6575342465753424, 0.6482758620689655, 0.6527777777777778, 0.6527777777777778, 0.6573426573426573, 0.6573426573426573, 0.6619718309859155, 0.6714285714285714] 
	t3 = [0.21004199981689453, 9.689432144165039, 11.696299076080322, 19.51404309272766, 28.143835067749023] 
	q3 = [0.653061224489796, 0.8392857142857143, 0.8392857142857143, 0.8392857142857143, 0.8288288288288289] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)

	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.7307692307692308, 0.7466666666666668, 0.7777777777777777, 0.7832167832167831, 0.7714285714285714, 0.7883211678832116, 0.8030303030303029, 0.8346456692913385, 0.8688524590163934, 0.8717948717948718] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.7307692307692308, 0.7307692307692308, 0.7354838709677418, 0.7402597402597402, 0.7549668874172186, 0.76, 0.7651006711409396, 0.767123287671233, 0.767123287671233] 
	t3 = [0.20418500900268555, 11.033854961395264, 11.816263914108276, 19.456294059753418, 27.16227626800537] 
	q3 = [0.7307692307692308, 0.8666666666666667, 0.8739495798319329, 0.8666666666666667, 0.8760330578512396] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)

	t1 = [0, 4, 6, 8, 10, 12, 14, 16, 18, 20] 
	q1 = [0.7058823529411764, 0.7248322147651006, 0.7210884353741497, 0.7464788732394366, 0.7737226277372263, 0.7851851851851851, 0.8091603053435115, 0.832, 0.8455284552845528, 0.8666666666666665] 
	t2 = [0, 4, 6, 8, 10, 12, 14, 16, 18] 
	q2 = [0.7058823529411764, 0.7152317880794702, 0.7152317880794702, 0.72, 0.7248322147651006, 0.7297297297297298, 0.7297297297297298, 0.7297297297297298, 0.7346938775510204] 
	t3 = [0.21091818809509277, 4.308044195175171, 11.289819240570068, 12.865328073501587, 21.110344171524048] 
	q3 = [0.7058823529411764, 0.859504132231405, 0.859504132231405, 0.859504132231405, 0.8524590163934426] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	
	t1_new = [sum(e)/len(e) for e in zip(*t1_all)]
	q1_new = [sum(e)/len(e) for e in zip(*q1_all)]
	t2_new = [sum(e)/len(e) for e in zip(*t2_all)]
	q2_new = [sum(e)/len(e) for e in zip(*q2_all)]	
	t3_new = [sum(e)/len(e) for e in zip(*t3_all)]
	q3_new = [sum(e)/len(e) for e in zip(*q3_all)]
	print t1_new
	print q1_new
	print t2_new
	print q2_new
	print t3_new
	print q3_new
	q2_new.append(q2_new[len(q2_new)-1])
	t2_new.append(20)
	q1_new = np.asarray(q1_new)
	q2_new = np.asarray(q2_new)
	q3_new = np.asarray(q3_new)
	
	
	'''
	#q1_new = preprocessing.normalize(q1_new)
	#q2_new = preprocessing.normalize(q2_new)
	#q3_new = preprocessing.normalize(q3_new)
	#q1_norm = q1_new / np.linalg.norm(q1_new)
	#q2_norm = q2_new / np.linalg.norm(q2_new)
	#q3_norm = q3_new / np.linalg.norm(q3_new)
	'''
	min_val = min(min(q1_new),min(q2_new),min(q3_new))
	max_val = max(max(q1_new),max(q2_new),max(q3_new))
	'''
	#q1_norm = (q1_new-min(q1_new))/(max(q1_new)-min(q1_new))
	#q2_norm = (q2_new-min(q2_new))/(max(q2_new)-min(q2_new))
	#q3_norm = (q3_new-min(q3_new))/(max(q3_new)-min(q3_new))
	'''
	q1_norm = (q1_new-min_val)/(max_val - min_val)
	q2_norm = (q2_new-min_val)/(max_val - min_val)
	q3_norm = (q3_new-min_val)/(max_val - min_val)
	
	
	plt.plot(t1_new, q1_norm,lw=2,color='green',  label='Baseline1 (Function Based Approach)')
	plt.plot(t2_new, q2_norm,lw=2,color='orange',  label='Baseline2 (Object Based Approach)')
	plt.plot(t3_new, q3_norm,lw=2,color='blue', label='Iterative Approach') ##2,000
	
	#plt.plot(t1, q1,lw=2,color='green', marker ='d', label='Baseline1 (Function Based Approach)')
	#plt.plot(t2, q2,lw=2,color='orange', marker ='o', label='Baseline2 (Object Based Approach)')
	#plt.plot(t3, q3,lw=2,color='blue',marker ='^',  label='Iterative Approach') ##2,000
	
	
	'''
	#plt.plot(t1_new, q1_new,lw=2,color='green',  label='Baseline1 (Function Based Approach)')
	#plt.plot(t2_new, q2_new,lw=2,color='orange',  label='Baseline2 (Object Based Approach)')
	#plt.plot(t3_new, q3_new,lw=2,color='blue', label='Iterative Approach') ##2,000
	'''

	
	plt.title('Quality vs Cost')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
	#plt.legend(bbox_to_anchor=(0, 1),loc="upper left",fontsize='xx-small')
	#bbox_to_anchor=(0, 1), loc='upper left', ncol=1
	plt.ylabel('F1-measure')
	plt.xlabel('Cost')
	plt.ylim([0, 1])
	plt.xlim([0, 20])	
	plt.savefig('PlotQualityComparisonMuctBaseline_gender_Avg_norm_5percent.png', format='png')
	plt.savefig('PlotQualityComparisonMuctBaseline_gender_Avg_norm_5percent.eps', format='eps')
		#plt.show()
	plt.close()
	

def plotOptimalEpoch():
	epoch_list = [1,2,3,4,5,6,7,8,9,10]
	t1_all,q1_all,t2_all,q2_all,t3_all,q3_all=[],[],[],[],[],[]
	t4_all,q4_all,t5_all,q5_all,t6_all,q6_all=[],[],[],[],[],[]
	t7_all,q7_all,t8_all,q8_all,t9_all,q9_all=[],[],[],[],[],[]	
	t10_all,q10_all=[],[]
	# Plotting epoch for 1000 objects.
	
	f1 = open('PlotEpoch.txt','w+')
	budget = 60
	t1 = [0, 1.1421301364898682, 2.07663631439209, 3.0818979740142822, 4.08338737487793, 5.099615097045898, 6.107669353485107, 7.094089508056641, 8.08315372467041, 9.108853816986084, 10.095873832702637, 11.027074337005615, 12.003092288970947, 13.005414485931396, 14.013357162475586, 15.098936080932617, 16.068843126296997, 17.063591241836548, 18.040178060531616, 19.13695192337036, 20.120185136795044, 21.12131643295288, 22.07657551765442, 23.114760875701904, 24.069523811340332, 25.049437999725342, 26.020316123962402, 27.042701482772827, 28.05931520462036, 29.072198152542114, 30.043521404266357, 31.03858733177185, 32.12394857406616, 33.07915925979614, 34.05141496658325, 35.015876054763794, 36.00778341293335, 37.10968017578125, 38.06522750854492, 39.005422592163086, 40.14305543899536, 41.0788733959198, 42.05105662345886, 43.01772713661194, 44.126885414123535, 45.12817454338074, 46.015957832336426, 47.09357523918152, 48.07171106338501, 49.02590227127075, 50.13283610343933, 51.12456941604614, 52.088176250457764, 53.0419499874115, 54.130590200424194, 55.10443305969238, 56.1087384223938, 57.07093143463135, 58.03224325180054, 59.10783624649048, 60.10301423072815] 
	q1 = [0.36419753086419754, 0.37116564417177916, 0.3780487804878049, 0.3897280966767372, 0.3963963963963964, 0.40834575260804773, 0.41715976331360943, 0.4258443465491924, 0.43045387994143486, 0.4366812227074236, 0.44283646888567296, 0.4511494252873563, 0.4549356223175965, 0.4580369843527739, 0.4661016949152543, 0.4733893557422969, 0.4735376044568246, 0.48342541436464087, 0.4814305364511692, 0.48907103825136616, 0.4945652173913044, 0.5067385444743935, 0.5087483176312247, 0.516042780748663, 0.5232403718459496, 0.5317460317460317, 0.5421052631578948, 0.54640522875817, 0.5539661898569571, 0.5607235142118863, 0.5692307692307692, 0.5739795918367347, 0.5865992414664982, 0.5937106918238995, 0.5947302383939774, 0.6027397260273972, 0.607940446650124, 0.6182266009852218, 0.625, 0.6317073170731707, 0.6383495145631068, 0.6457831325301205, 0.6491017964071857, 0.6563614744351962, 0.6603550295857987, 0.6658823529411765, 0.6721311475409836, 0.675990675990676, 0.679814385150812, 0.6851211072664359, 0.6881472957422323, 0.6911595866819747, 0.6926605504587156, 0.6948571428571428, 0.6947608200455581, 0.6969353007945517, 0.6990950226244345, 0.701912260967379, 0.7025813692480359, 0.7083798882681565, 0.7104677060133631] 
	t2 = [0, 2.1238903999328613, 4.110911846160889, 6.091003179550171, 8.084938764572144, 10.012205123901367, 12.140867710113525, 14.058226585388184, 16.022084712982178, 18.071865797042847, 20.132930040359497, 22.03125500679016, 24.11094331741333, 26.003081560134888, 28.105425596237183, 30.034626007080078, 32.01245093345642, 34.138784646987915, 36.06335973739624, 38.0503294467926, 40.068623542785645, 42.065184593200684, 44.13629341125488, 46.13137221336365, 48.02508807182312, 50.04358744621277, 52.03864145278931, 54.01003336906433, 56.04631781578064, 58.084187030792236, 60.134846448898315] 
	q2 = [0.36419753086419754, 0.38239757207890746, 0.4059701492537313, 0.4235294117647059, 0.43440233236151604, 0.4511494252873563, 0.46022727272727276, 0.4727272727272728, 0.4806629834254144, 0.48840381991814463, 0.5087483176312247, 0.5173333333333333, 0.5375494071146245, 0.5528031290743155, 0.5681233933161953, 0.5830164765525983, 0.5947302383939774, 0.6104218362282878, 0.625, 0.6383495145631068, 0.6507177033492824, 0.6603550295857987, 0.6728971962616822, 0.6813441483198147, 0.689655172413793, 0.6941580756013747, 0.6977272727272728, 0.701240135287486, 0.7069351230425057, 0.7111111111111111, 0.7160220994475138] 
	t3 = [0, 3.0074496269226074, 6.014394998550415, 9.030627489089966, 12.127566576004028, 15.093461751937866, 18.01807188987732, 21.092859029769897, 24.094733715057373, 27.088606119155884, 30.007619619369507, 33.02085256576538, 36.10740900039673, 39.13725924491882, 42.031683921813965, 45.07506704330444, 48.00115609169006, 51.113260984420776, 54.11562442779541, 57.04399299621582, 60.13231015205383] 
	q3 = [0.36419753086419754, 0.39457831325301207, 0.4235294117647059, 0.44283646888567296, 0.4624113475177305, 0.4798890429958391, 0.49046321525885556, 0.5140562248995983, 0.5421052631578948, 0.5588615782664942, 0.5865992414664982, 0.6052303860523038, 0.63003663003663, 0.6474820143884892, 0.6611570247933884, 0.6782810685249709, 0.6911595866819747, 0.6947608200455581, 0.7027027027027026, 0.7126948775055679, 0.7174392935982341] 
	t4 = [0, 4.0333921909332275, 8.08641767501831, 12.087337493896484, 16.11101269721985, 20.076168060302734, 24.09618878364563, 28.033162355422974, 32.11526155471802, 36.08677673339844, 40.05938243865967, 44.038376808166504, 48.098082304000854, 52.122355937957764, 56.09919619560242, 60.10752868652344] 
	q4 = [0.36419753086419754, 0.4059701492537313, 0.4366812227074236, 0.4624113475177305, 0.4793388429752066, 0.510752688172043, 0.5447368421052631, 0.5699614890885751, 0.6, 0.63003663003663, 0.6547619047619048, 0.6744457409568261, 0.6911595866819747, 0.6984126984126985, 0.7083798882681565, 0.7174392935982341] 
	t5 = [0, 5.069773197174072, 10.007799863815308, 15.086117029190063, 20.05875873565674, 25.138978719711304, 30.047862768173218, 35.11377477645874, 40.00264000892639, 45.00557541847229, 50.113765478134155, 55.09384083747864, 60.107484579086304] 
	q5 = [0.36419753086419754, 0.4148148148148148, 0.4505021520803443, 0.479224376731302, 0.510752688172043, 0.5497382198952879, 0.5873417721518986, 0.6257668711656442, 0.6547619047619048, 0.6782810685249709, 0.6940639269406392, 0.7077267637178052, 0.7174392935982341] 
	t6 = [0, 6.001662254333496, 12.136298179626465, 18.005889415740967, 24.111199617385864, 30.122830629348755, 36.10718774795532, 42.012521743774414, 48.13635873794556, 54.08355522155762, 60.12863516807556] 
	q6 = [0.36419753086419754, 0.4235294117647059, 0.4624113475177305, 0.49046321525885556, 0.5447368421052631, 0.5891276864728192, 0.6324786324786326, 0.6627358490566038, 0.6911595866819747, 0.701912260967379, 0.7182320441988951] 
	t7 = [0, 7.034785032272339, 14.098866939544678, 21.102428197860718, 28.065725564956665, 35.05711913108826, 42.00697612762451, 49.04806089401245, 56.01845026016235] 
	q7 = [0.36419753086419754, 0.43045387994143486, 0.4727272727272728, 0.516042780748663, 0.5732647814910026, 0.6257668711656442, 0.6643109540636042, 0.6926605504587156, 0.7120535714285715] 
	t8 = [0, 8.04780101776123, 16.07738494873047, 24.126025915145874, 32.110719203948975, 40.05675172805786, 48.058434009552, 56.024736404418945] 
	q8 = [0.36419753086419754, 0.4366812227074236, 0.48, 0.5466491458607096, 0.6042446941323346, 0.6563614744351962, 0.689655172413793, 0.7120535714285715] 
	t9 = [0, 9.097555875778198, 18.09144139289856, 27.099843502044678, 36.14071989059448, 45.023149251937866, 54.13927960395813] 
	q9 = [0.36419753086419754, 0.4450867052023122, 0.4931880108991825, 0.567741935483871, 0.6324786324786326, 0.679814385150812, 0.7033707865168538] 
	t10 = [0, 10.140229940414429, 20.12906837463379, 30.120646953582764, 40.1150803565979, 50.11060428619385, 60.04435110092163] 
	q10 = [0.36419753086419754, 0.45272206303724927, 0.5087483176312247, 0.5924050632911393, 0.6587395957193817, 0.6902857142857143, 0.7190265486725663] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
		
	t1 = [0, 1.145557165145874, 2.0167829990386963, 3.070876359939575, 4.095244407653809, 5.039327144622803, 6.054346323013306, 7.114323139190674, 8.128237247467041, 9.018351316452026, 10.064985752105713, 11.14625859260559, 12.000388622283936, 13.135489225387573, 14.135303258895874, 15.10997223854065, 16.040486335754395, 17.064093589782715, 18.001842975616455, 19.12989616394043, 20.09576153755188, 21.053289651870728, 22.026889324188232, 23.127421379089355, 24.05494737625122, 25.038881301879883, 26.030174255371094, 27.06503415107727, 28.059198141098022, 29.07521915435791, 30.044228076934814, 31.004374265670776, 32.10909628868103, 33.093196868896484, 34.02763271331787, 35.00947594642639, 36.085076093673706, 37.03457832336426, 38.12325406074524, 39.089613914489746, 40.06298303604126, 41.023584604263306, 42.02283501625061, 43.00399208068848, 44.00881552696228, 45.129900217056274, 46.06376910209656, 47.009850025177, 48.09582734107971, 49.080073595047, 50.06532311439514, 51.01998162269592, 52.00944495201111, 53.11290001869202, 54.04799437522888, 55.02340841293335, 56.016642570495605, 57.113085985183716, 58.05186343193054, 59.07329225540161, 60.03421139717102] 
	q1 = [0.34633385335413414, 0.3478260869565218, 0.35802469135802467, 0.36447166921898927, 0.36946564885496186, 0.3793626707132018, 0.3861236802413273, 0.39461883408071746, 0.40118870728083206, 0.41124260355029585, 0.41826215022091306, 0.4269005847953216, 0.430232558139535, 0.43352601156069365, 0.43678160919540227, 0.4457142857142857, 0.45014245014245013, 0.4604519774011299, 0.47124824684431976, 0.48189415041782724, 0.48753462603878117, 0.49103448275862066, 0.4979480164158688, 0.5101763907734056, 0.5175202156334232, 0.5301204819277109, 0.5352862849533955, 0.5411140583554377, 0.5488126649076517, 0.5590551181102362, 0.5677083333333334, 0.572538860103627, 0.577319587628866, 0.5838668373879642, 0.5903307888040713, 0.5984848484848485, 0.6082603254067585, 0.6134663341645885, 0.6212871287128713, 0.626387176325524, 0.6340269277845777, 0.6406820950060901, 0.6472727272727272, 0.6530120481927711, 0.6610778443113772, 0.6682520808561236, 0.6729857819905214, 0.6784452296819788, 0.687719298245614, 0.6923076923076924, 0.6968641114982579, 0.6998841251448435, 0.7028901734104046, 0.7065592635212888, 0.7080459770114942, 0.7087155963302753, 0.7123287671232876, 0.7144482366325371, 0.7188208616780045, 0.7217194570135747, 0.7223476297968396] 
	t2 = [0, 2.1159257888793945, 4.078946352005005, 6.093810558319092, 8.089213371276855, 10.125604629516602, 12.091827154159546, 14.082260370254517, 16.102725505828857, 18.050812005996704, 20.040732860565186, 22.05283236503601, 24.04650902748108, 26.1421537399292, 28.10599994659424, 30.12622880935669, 32.13990497589111, 34.000248670578, 36.10694766044617, 38.10371923446655, 40.02092146873474, 42.082624435424805, 44.08614158630371, 46.05184555053711, 48.125869035720825, 50.01502251625061, 52.119728088378906, 54.04425001144409, 56.01954436302185, 58.078699827194214, 60.11314392089844] 
	q2 = [0.34633385335413414, 0.36251920122887865, 0.37689969604863227, 0.39461883408071746, 0.413589364844904, 0.4308588064046579, 0.4357864357864358, 0.45014245014245013, 0.4669479606188467, 0.4854368932038835, 0.49657064471879286, 0.5182186234817814, 0.5352862849533955, 0.5514511873350924, 0.5677083333333334, 0.577319587628866, 0.5913705583756345, 0.6082603254067585, 0.6229913473423981, 0.6356968215158924, 0.648910411622276, 0.6634844868735084, 0.6768867924528302, 0.6892523364485981, 0.6998841251448435, 0.707373271889401, 0.7087155963302753, 0.7144482366325371, 0.7202718006795016, 0.7266591676040495, 0.7307262569832403] 
	t3 = [0, 3.0270166397094727, 6.038617849349976, 9.047117471694946, 12.122978448867798, 15.139983415603638, 18.04870867729187, 21.136324405670166, 24.01557469367981, 27.06640386581421, 30.037589073181152, 33.092204332351685, 36.144267559051514, 39.030481576919556, 42.11472535133362, 45.09366011619568, 48.108840227127075, 51.092257499694824, 54.14133095741272, 57.07763338088989, 60.13263559341431] 
	q3 = [0.34633385335413414, 0.36447166921898927, 0.39461883408071746, 0.4252199413489737, 0.4380403458213256, 0.4582743988684582, 0.48753462603878117, 0.5128900949796472, 0.5352862849533955, 0.56282722513089, 0.5791505791505791, 0.6047678795483061, 0.6246913580246913, 0.6472727272727272, 0.669833729216152, 0.687719298245614, 0.6998841251448435, 0.7080459770114942, 0.7167235494880546, 0.7229729729729729, 0.73355629877369] 
	t4 = [0, 4.053027153015137, 8.003569841384888, 12.111596822738647, 16.11741328239441, 20.12563443183899, 24.087968587875366, 28.023900270462036, 32.06928253173828, 36.012227058410645, 40.09925889968872, 44.14327096939087, 48.060954093933105, 52.10019135475159, 56.096956968307495, 60.11146521568298] 
	q4 = [0.34633385335413414, 0.37689969604863227, 0.413589364844904, 0.4380403458213256, 0.4691011235955057, 0.5027322404371585, 0.5372340425531915, 0.5714285714285715, 0.5974683544303798, 0.6246913580246913, 0.654632972322503, 0.6830985915492958, 0.6983758700696056, 0.711670480549199, 0.7209039548022599, 0.7313266443701227] 
	t5 = [0, 5.085408926010132, 10.004542112350464, 15.06087851524353, 20.00646162033081, 25.033860683441162, 30.091560125350952, 35.11940670013428, 40.135600328445435, 45.12726306915283, 50.143064737319946, 55.073288679122925, 60.05191087722778] 
	q5 = [0.34633385335413414, 0.3831070889894419, 0.4308588064046579, 0.45957446808510644, 0.5006839945280437, 0.5444887118193892, 0.5809768637532133, 0.6212871287128713, 0.654632972322503, 0.6892523364485981, 0.7065592635212888, 0.7210884353741497, 0.7321428571428571] 
	t6 = [0, 6.126628637313843, 12.133405685424805, 18.090950965881348, 24.047316312789917, 30.04759693145752, 36.05911183357239, 42.08700394630432, 48.09595227241516, 54.02335548400879, 60.12561225891113] 
	q6 = [0.34633385335413414, 0.39461883408071746, 0.4380403458213256, 0.48753462603878117, 0.5391766268260293, 0.5853658536585366, 0.6280788177339901, 0.669833729216152, 0.6998841251448435, 0.7196367763904653, 0.732739420935412] 
	t7 = [0, 7.053351163864136, 14.09169626235962, 21.13970947265625, 28.08936858177185, 35.111246824264526, 42.13700008392334, 49.01611375808716, 56.00352501869202] 
	q7 = [0.34633385335413414, 0.4035608308605341, 0.4536376604850214, 0.516914749661705, 0.5740259740259741, 0.6220570012391574, 0.6714116251482799, 0.7035755478662054, 0.7209039548022599] 
	t8 = [0, 8.08163046836853, 16.09730887413025, 24.07810115814209, 32.02726769447327, 40.10657262802124, 48.05740284919739, 56.1164391040802] 
	q8 = [0.34633385335413414, 0.413589364844904, 0.47042253521126765, 0.5398936170212767, 0.5992414664981036, 0.6562499999999999, 0.6983758700696056, 0.7209039548022599] 
	t9 = [0, 9.10825777053833, 18.06796908378601, 27.08391547203064, 36.03549361228943, 45.04699349403381, 54.09232974052429] 
	q9 = [0.34633385335413414, 0.424597364568082, 0.4888888888888889, 0.5673202614379086, 0.6280788177339901, 0.6892523364485981, 0.7210884353741497] 
	t10 = [0, 10.12211561203003, 20.07104229927063, 30.090749740600586, 40.12814259529114, 50.110151529312134, 60.13711476325989] 
	q10 = [0.34633385335413414, 0.430232558139535, 0.505464480874317, 0.5879332477535303, 0.6594724220623501, 0.7050691244239632, 0.7321428571428571] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.0982134342193604, 2.0271573066711426, 3.0070149898529053, 4.099928140640259, 5.049942255020142, 6.12503981590271, 7.073804140090942, 8.03342890739441, 9.07119083404541, 10.005188703536987, 11.098406553268433, 12.056205987930298, 13.03326416015625, 14.112760543823242, 15.059711456298828, 16.020415782928467, 17.14259696006775, 18.123124361038208, 19.06395673751831, 20.026848077774048, 21.03738236427307, 22.13356876373291, 23.110565662384033, 24.08210277557373, 25.057913541793823, 26.01348042488098, 27.001391887664795, 28.082525730133057, 29.030044555664062, 30.13262915611267, 31.103212118148804, 32.02882671356201, 33.1457622051239, 34.11554574966431, 35.05463480949402, 36.12956643104553, 37.14094567298889, 38.101282596588135, 39.07185339927673, 40.05462312698364, 41.146098136901855, 42.098687171936035, 43.03933882713318, 44.11916923522949, 45.093573331832886, 46.023224115371704, 47.12252712249756, 48.07660937309265, 49.05007076263428, 50.00642132759094, 51.023642778396606, 52.12154674530029, 53.09384298324585, 54.03245162963867, 55.00962233543396, 56.11182236671448, 57.12128949165344, 58.07777547836304, 59.024311542510986, 60.09576725959778] 
	q1 = [0.37181409295352325, 0.37797619047619047, 0.3846153846153846, 0.3929618768328446, 0.4011627906976744, 0.40810419681620835, 0.4195402298850574, 0.4234620886981402, 0.43304843304843305, 0.4428772919605077, 0.45314685314685316, 0.45769764216366154, 0.46344827586206894, 0.46849315068493147, 0.48303934871099047, 0.490566037735849, 0.49329758713136723, 0.49866666666666665, 0.5059602649006623, 0.5157894736842105, 0.5190039318479684, 0.5260416666666666, 0.5388601036269429, 0.5463917525773196, 0.5549872122762147, 0.5623409669211196, 0.5670886075949367, 0.5753768844221105, 0.5853051058530511, 0.594059405940594, 0.6002460024600246, 0.6063569682151588, 0.6114494518879415, 0.619105199516324, 0.625, 0.629940119760479, 0.6348448687350835, 0.6389548693586697, 0.6422668240850059, 0.6447058823529411, 0.6510538641686183, 0.6565774155995343, 0.6635838150289017, 0.6697353279631761, 0.6765714285714285, 0.6810933940774487, 0.6848072562358276, 0.6892655367231639, 0.6914414414414415, 0.6958473625140291, 0.6973094170403588, 0.6972067039106146, 0.6985539488320357, 0.6984478935698448, 0.6991150442477876, 0.6983425414364641, 0.701212789415656, 0.701098901098901, 0.7039473684210528, 0.7039473684210528, 0.7039473684210528] 
	t2 = [0, 2.1234614849090576, 4.134967565536499, 6.101630926132202, 8.096796035766602, 10.040756940841675, 12.00066614151001, 14.032771348953247, 16.00125503540039, 18.053857803344727, 20.10259199142456, 22.021680116653442, 24.025604963302612, 26.056764841079712, 28.10480546951294, 30.06917667388916, 32.08786940574646, 34.11571478843689, 36.086474895477295, 38.00291085243225, 40.09352135658264, 42.12463307380676, 44.12114238739014, 46.120280265808105, 48.01631546020508, 50.02686810493469, 52.09063720703125, 54.130378007888794, 56.00871801376343, 58.067572832107544, 60.04050397872925] 
	q2 = [0.37181409295352325, 0.3870014771048744, 0.4034833091436865, 0.4195402298850574, 0.4375, 0.45746164574616455, 0.46556473829201106, 0.48443843031123135, 0.49732620320855614, 0.5099075297225892, 0.5254901960784314, 0.544516129032258, 0.5586734693877551, 0.5735849056603773, 0.594059405940594, 0.6070991432068543, 0.6198547215496367, 0.629940119760479, 0.6389548693586697, 0.6447058823529411, 0.6565774155995343, 0.6712643678160919, 0.6810933940774487, 0.6892655367231639, 0.6958473625140291, 0.6964285714285713, 0.6984478935698448, 0.6997792494481236, 0.701098901098901, 0.7039473684210528, 0.7066521264994546] 
	t3 = [0, 3.0206782817840576, 6.056391477584839, 9.029143810272217, 12.020671367645264, 15.101492166519165, 18.121720790863037, 21.1143159866333, 24.07135796546936, 27.048752784729004, 30.0882568359375, 33.000900745391846, 36.07956147193909, 39.097012758255005, 42.03049612045288, 45.057536363601685, 48.088013887405396, 51.058796882629395, 54.00988817214966, 57.12224578857422, 60.008975982666016] 
	q3 = [0.37181409295352325, 0.39238653001464124, 0.4195402298850574, 0.44788732394366193, 0.4676753782668501, 0.49193548387096775, 0.5138339920948617, 0.5376623376623376, 0.5605095541401274, 0.5853051058530511, 0.6063569682151588, 0.6274038461538463, 0.6405693950177936, 0.6534422403733955, 0.6735395189003436, 0.6892655367231639, 0.6973094170403588, 0.696329254727475, 0.701212789415656, 0.7039473684210528, 0.7080610021786493] 
	t4 = [0, 4.051710605621338, 8.117942571640015, 12.120754957199097, 16.093992948532104, 20.07853412628174, 24.126720905303955, 28.115028619766235, 32.10870575904846, 36.05734705924988, 40.12150740623474, 44.100762605667114, 48.01778745651245, 52.032161235809326, 56.03925943374634, 60.04773163795471] 
	q4 = [0.37181409295352325, 0.4011627906976744, 0.43971631205673756, 0.4676753782668501, 0.49933244325767684, 0.5267275097783573, 0.5623409669211196, 0.5933250927070457, 0.6231884057971014, 0.6421800947867299, 0.662037037037037, 0.6848072562358276, 0.6973094170403588, 0.6962305986696231, 0.7047200878155873, 0.7080610021786493] 
	t5 = [0, 5.073621988296509, 10.014706134796143, 15.109547853469849, 20.106592178344727, 25.098326206207275, 30.068031072616577, 35.05814456939697, 40.034183979034424, 45.05451250076294, 50.03579497337341, 55.04458951950073, 60.09641790390015] 
	q5 = [0.37181409295352325, 0.41040462427745666, 0.45810055865921784, 0.49395973154362416, 0.5286458333333333, 0.5689001264222504, 0.6063569682151588, 0.6357142857142858, 0.660486674391657, 0.6892655367231639, 0.6941964285714286, 0.7018701870187017, 0.7080610021786493] 
	t6 = [0, 6.107774019241333, 12.121619939804077, 18.10396957397461, 24.06871795654297, 30.127647638320923, 36.060991525650024, 42.039698362350464, 48.13990616798401, 54.125638008117676, 60.045559883117676] 
	q6 = [0.37181409295352325, 0.4195402298850574, 0.47107438016528924, 0.5164690382081687, 0.564885496183206, 0.608058608058608, 0.6437869822485207, 0.6765714285714285, 0.6973094170403588, 0.701212789415656, 0.7080610021786493] 
	t7 = [0, 7.070720434188843, 14.008076429367065, 21.099876165390015, 28.054440021514893, 35.06156802177429, 42.10144782066345, 49.083322286605835, 56.08134579658508] 
	q7 = [0.37181409295352325, 0.4234620886981402, 0.489851150202977, 0.5421530479896238, 0.5933250927070457, 0.6364719904648392, 0.6780821917808219, 0.6949720670391062, 0.7054945054945054] 
	t8 = [0, 8.060140371322632, 16.00146174430847, 24.046562433242798, 32.14047455787659, 40.054476261138916, 48.01254987716675, 56.07000136375427] 
	q8 = [0.37181409295352325, 0.43971631205673756, 0.5, 0.564885496183206, 0.6224366706875754, 0.662037037037037, 0.6973094170403588, 0.7040704070407041] 
	t9 = [0, 9.08663296699524, 18.023170948028564, 27.044793128967285, 36.075560331344604, 45.01198148727417, 54.10168242454529] 
	q9 = [0.37181409295352325, 0.45007032348804504, 0.5171503957783641, 0.5895522388059701, 0.6437869822485207, 0.6892655367231639, 0.6997792494481236] 
	t10 = [0, 10.10966944694519, 20.082568407058716, 30.116431713104248, 40.004812479019165, 50.06934332847595, 60.07201528549194] 
	q10 = [0.37181409295352325, 0.45746164574616455, 0.53125, 0.6097560975609756, 0.6635838150289017, 0.6948775055679287, 0.710239651416122] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.1001362800598145, 2.0257747173309326, 3.1155920028686523, 4.046116590499878, 5.140592098236084, 6.0668439865112305, 7.0228111743927, 8.095128059387207, 9.068950414657593, 10.034108877182007, 11.013028144836426, 12.034619331359863, 13.122985363006592, 14.053006887435913, 15.14228343963623, 16.070087909698486, 17.047923803329468, 18.125110626220703, 19.075464725494385, 20.048821449279785, 21.12105631828308, 22.10868525505066, 23.002615213394165, 24.09656810760498, 25.10692310333252, 26.063931465148926, 27.09914541244507, 28.05281352996826, 29.043031454086304, 30.127859354019165, 31.07285761833191, 32.00256562232971, 33.08895826339722, 34.015806913375854, 35.14701962471008, 36.10264873504639, 37.056440591812134, 38.01314330101013, 39.13337802886963, 40.097792625427246, 41.04257369041443, 42.11353874206543, 43.0881290435791, 44.05413317680359, 45.033711194992065, 46.09653830528259, 47.07114315032959, 48.0214478969574, 49.124330043792725, 50.10848784446716, 51.056782722473145, 52.124613761901855, 53.129966735839844, 54.08676266670227, 55.026957750320435, 56.09111976623535, 57.09119987487793, 58.01413416862488, 59.13751244544983, 60.11647987365723] 
	q1 = [0.3758169934640523, 0.3766233766233766, 0.38647342995169087, 0.4044585987261146, 0.4088748019017433, 0.42006269592476486, 0.43167701863354035, 0.4382716049382716, 0.445468509984639, 0.45259938837920494, 0.4589665653495441, 0.46827794561933545, 0.4730538922155689, 0.47774480712166173, 0.48153618906942397, 0.4897360703812316, 0.49635036496350365, 0.49782923299565845, 0.5, 0.5064377682403434, 0.5149359886201992, 0.5225988700564972, 0.5260196905766527, 0.5322128851540616, 0.5355648535564853, 0.5436893203883495, 0.5517241379310345, 0.5616438356164384, 0.5694822888283378, 0.5745257452574526, 0.5879194630872483, 0.5935828877005348, 0.602921646746348, 0.6103038309114927, 0.6149802890932983, 0.6223958333333334, 0.6304909560723514, 0.6367137355584083, 0.6454081632653061, 0.648854961832061, 0.6522842639593909, 0.6574307304785894, 0.6616541353383459, 0.6674968866749689, 0.6707920792079207, 0.6757090012330457, 0.6781326781326782, 0.6805385556915544, 0.6877278250303767, 0.6893203883495145, 0.694074969770254, 0.6987951807228916, 0.701923076923077, 0.704326923076923, 0.7050359712230215, 0.7048984468339308, 0.7095238095238096, 0.7131050767414403, 0.7137809187279153, 0.7144535840188014, 0.7151230949589684] 
	t2 = [0, 2.138166666030884, 4.134227514266968, 6.111835956573486, 8.11059284210205, 10.00384521484375, 12.071480512619019, 14.108927726745605, 16.087451934814453, 18.093556880950928, 20.06174921989441, 22.120155811309814, 24.036341667175293, 26.078498601913452, 28.043982982635498, 30.048627376556396, 32.02179312705994, 34.138437271118164, 36.0263671875, 38.028295278549194, 40.053093910217285, 42.05509042739868, 44.066654443740845, 46.0929217338562, 48.132158517837524, 50.00965166091919, 52.06183457374573, 54.06657648086548, 56.068347692489624, 58.10250902175903, 60.151220083236694] 
	q2 = [0.3758169934640523, 0.3890675241157556, 0.4113924050632911, 0.434108527131783, 0.44785276073619634, 0.4620060790273556, 0.47832585949177875, 0.4837758112094395, 0.4956268221574344, 0.5, 0.5163120567375886, 0.5329593267882188, 0.5436893203883495, 0.5576923076923077, 0.5745257452574526, 0.5909090909090908, 0.6084656084656085, 0.623207301173403, 0.6367137355584083, 0.648854961832061, 0.6574307304785894, 0.6691542288557214, 0.6765067650676506, 0.6821515892420538, 0.6909090909090909, 0.6987951807228916, 0.7058823529411765, 0.7064439140811457, 0.7131050767414403, 0.7159624413145539, 0.7217694994179279] 
	t3 = [0, 3.0234758853912354, 6.023982763290405, 9.03132176399231, 12.120656251907349, 15.113349676132202, 18.118199586868286, 21.023839712142944, 24.144083976745605, 27.015464067459106, 30.023484230041504, 33.036829710006714, 36.07786202430725, 39.080613136291504, 42.03210735321045, 45.02494478225708, 48.040480852127075, 51.12202477455139, 54.04757642745972, 57.098737716674805, 60.0611686706543] 
	q3 = [0.3758169934640523, 0.4019138755980861, 0.434108527131783, 0.4542682926829269, 0.48059701492537316, 0.4941520467836257, 0.5050215208034433, 0.5260196905766527, 0.5464632454923717, 0.5694822888283378, 0.595460614152203, 0.6186107470511141, 0.6402048655569783, 0.652338811630847, 0.6708074534161491, 0.6773006134969324, 0.6933333333333334, 0.704326923076923, 0.7095238095238096, 0.7144535840188014, 0.7232558139534884] 
	t4 = [0, 4.058889865875244, 8.120739459991455, 12.071294069290161, 16.118106603622437, 20.057714700698853, 24.051478624343872, 28.139758110046387, 32.05834364891052, 36.13153696060181, 40.01208019256592, 44.115389585494995, 48.09235429763794, 52.084632396698, 56.07250761985779, 60.053340673446655] 
	q4 = [0.3758169934640523, 0.4113924050632911, 0.44785276073619634, 0.48059701492537316, 0.4970930232558139, 0.5205091937765204, 0.5464632454923717, 0.5802968960863698, 0.6113306982872201, 0.6427656850192062, 0.6608040201005025, 0.6781326781326782, 0.6933333333333334, 0.7041916167664671, 0.7154663518299881, 0.7247386759581881] 
	t5 = [0, 5.100014925003052, 10.037231683731079, 15.12788200378418, 20.09188222885132, 25.011066436767578, 30.058258056640625, 35.04966449737549, 40.09329533576965, 45.00634026527405, 50.06484365463257, 55.005627155303955, 60.034812211990356] 
	q5 = [0.3758169934640523, 0.41940532081377147, 0.464339908952959, 0.49707602339181284, 0.5233380480905233, 0.5544827586206896, 0.5973333333333334, 0.6357786357786358, 0.6599749058971142, 0.6805385556915544, 0.700361010830325, 0.7139479905437353, 0.7247386759581881] 
	t6 = [0, 6.131175756454468, 12.055875539779663, 18.051613569259644, 24.076584815979004, 30.021467208862305, 36.09533166885376, 42.01819086074829, 48.14582347869873, 54.103989601135254, 60.06463384628296] 
	q6 = [0.3758169934640523, 0.43167701863354035, 0.4798807749627422, 0.5071633237822349, 0.5484764542936288, 0.5992010652463383, 0.6445012787723785, 0.6707920792079207, 0.6949152542372881, 0.7117437722419929, 0.7247386759581881] 
	t7 = [0, 7.041127681732178, 14.093735933303833, 21.10298490524292, 28.142921686172485, 35.126272201538086, 42.060739517211914, 49.06997895240784, 56.07425498962402] 
	q7 = [0.3758169934640523, 0.4382716049382716, 0.4889543446244477, 0.5288326300984529, 0.582210242587601, 0.6357786357786358, 0.6724351050679853, 0.6980676328502415, 0.7161366313309777] 
	t8 = [0, 8.0776047706604, 16.00696063041687, 24.07401466369629, 32.10411286354065, 40.008439779281616, 48.01470232009888, 56.089744329452515] 
	q8 = [0.3758169934640523, 0.44785276073619634, 0.5007278020378457, 0.5484764542936288, 0.613157894736842, 0.6616541353383459, 0.6964933494558646, 0.7161366313309777] 
	t9 = [0, 9.063295364379883, 18.037996292114258, 27.11362051963806, 36.07821440696716, 45.13530468940735, 54.089356422424316] 
	q9 = [0.3758169934640523, 0.4542682926829269, 0.5078909612625538, 0.5733695652173914, 0.6462324393358876, 0.6837606837606838, 0.7117437722419929] 
	t10 = [0, 10.10849928855896, 20.08853316307068, 30.03764033317566, 40.04952549934387, 50.056931495666504, 60.14640235900879] 
	q10 = [0.3758169934640523, 0.4666666666666667, 0.5261669024045261, 0.6018641810918774, 0.6641604010025063, 0.7012048192771085, 0.727061556329849] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.1036937236785889, 2.0421485900878906, 3.1463005542755127, 4.070799350738525, 5.0141212940216064, 6.082498550415039, 7.041187763214111, 8.143524169921875, 9.125368118286133, 10.085347175598145, 11.032130479812622, 12.106890678405762, 13.053857564926147, 14.04327392578125, 15.046581506729126, 16.002153396606445, 17.101946592330933, 18.036097764968872, 19.051058769226074, 20.1235671043396, 21.09740138053894, 22.05537223815918, 23.04268455505371, 24.108784437179565, 25.08716583251953, 26.038914680480957, 27.138009309768677, 28.130600452423096, 29.10536766052246, 30.0871684551239, 31.05995774269104, 32.047144174575806, 33.1333384513855, 34.0896213054657, 35.03307843208313, 36.134894371032715, 37.09963297843933, 38.030702114105225, 39.12320518493652, 40.04478931427002, 41.129491329193115, 42.080036878585815, 43.03163170814514, 44.114206075668335, 45.0665020942688, 46.075607776641846, 47.05487394332886, 48.131059408187866, 49.10225248336792, 50.02528166770935, 51.139819622039795, 52.05547642707825, 53.05552840232849, 54.11835765838623, 55.128966331481934, 56.04790425300598, 57.002782106399536, 58.06781005859375, 59.04686713218689, 60.144492864608765] 
	q1 = [0.3540372670807453, 0.36419753086419754, 0.37366003062787134, 0.38543247344461307, 0.39457831325301207, 0.4011976047904192, 0.4154302670623146, 0.4194977843426883, 0.4304538799414348, 0.4366812227074236, 0.4444444444444444, 0.45114942528735624, 0.4564907275320971, 0.4680851063829787, 0.47457627118644063, 0.4775280898876405, 0.48189415041782724, 0.48962655601659744, 0.4896836313617607, 0.4924760601915184, 0.4959128065395095, 0.5060893098782139, 0.5134408602150538, 0.5206942590120159, 0.5251989389920425, 0.531578947368421, 0.538562091503268, 0.5468749999999999, 0.5536869340232858, 0.5611325611325612, 0.5710627400768247, 0.5790816326530612, 0.5822784810126583, 0.5886792452830188, 0.593984962406015, 0.6, 0.6096654275092938, 0.6165228113440198, 0.624235006119951, 0.6309378806333739, 0.639225181598063, 0.6393244873341375, 0.6474820143884892, 0.65, 0.6548463356973996, 0.6619552414605419, 0.6690058479532164, 0.6767441860465115, 0.6843930635838151, 0.6904487917146145, 0.6926605504587156, 0.6978335233751426, 0.7007963594994312, 0.7037457434733259, 0.7089467723669308, 0.7089467723669308, 0.7096045197740112, 0.708803611738149, 0.7109111361079865, 0.7152466367713005, 0.7171492204899778] 
	t2 = [0, 2.1277592182159424, 4.124479532241821, 6.11326789855957, 8.11266016960144, 10.140102863311768, 12.147414922714233, 14.006836414337158, 16.101924657821655, 18.09603261947632, 20.01952052116394, 22.03916835784912, 24.0695858001709, 26.11408805847168, 28.093900442123413, 30.040034770965576, 32.000590801239014, 34.02296686172485, 36.03435397148132, 38.012572288513184, 40.00057935714722, 42.003469944000244, 44.12698554992676, 46.03630495071411, 48.02866983413696, 50.039350509643555, 52.07472562789917, 54.120752811431885, 56.017640352249146, 58.00991940498352, 60.01560735702515] 
	q2 = [0.3540372670807453, 0.37308868501529047, 0.39457831325301207, 0.4154302670623146, 0.4304538799414348, 0.4473304473304473, 0.4615384615384615, 0.47672778561354023, 0.48611111111111105, 0.49108367626886146, 0.5, 0.5194109772423026, 0.5264550264550264, 0.5449804432855281, 0.5592783505154639, 0.5761843790012804, 0.587641866330391, 0.6, 0.6165228113440198, 0.6292682926829269, 0.6393244873341375, 0.6507747318235997, 0.6619552414605419, 0.6767441860465115, 0.6904487917146145, 0.6986301369863013, 0.7022727272727273, 0.7089467723669308, 0.708803611738149, 0.7144456886898096, 0.7228381374722839] 
	t3 = [0, 3.025162935256958, 6.037459373474121, 9.041959762573242, 12.148163795471191, 15.068365335464478, 18.036813259124756, 21.034237146377563, 24.12300705909729, 27.09169864654541, 30.05574321746826, 33.04982256889343, 36.12638545036316, 39.053731203079224, 42.03224015235901, 45.085891246795654, 48.053919553756714, 51.11542224884033, 54.05753207206726, 57.09004735946655, 60.04073643684387] 
	q3 = [0.3540372670807453, 0.3829787234042553, 0.4154302670623146, 0.43831640058055155, 0.4637268847795163, 0.4832402234636871, 0.49108367626886146, 0.5094339622641509, 0.5303430079155673, 0.5544041450777202, 0.5798212005108557, 0.593984962406015, 0.6182266009852216, 0.639225181598063, 0.653206650831354, 0.6705607476635514, 0.6911595866819749, 0.7015945330296127, 0.7104072398190044, 0.7138047138047138, 0.7228381374722839] 
	t4 = [0, 4.0291969776153564, 8.111737966537476, 12.13054084777832, 16.113022089004517, 20.012109994888306, 24.14451813697815, 28.035300970077515, 32.08951139450073, 36.1209192276001, 40.06881380081177, 44.13494563102722, 48.14090895652771, 52.14700698852539, 56.135873556137085, 60.11486339569092] 
	q4 = [0.3540372670807453, 0.39457831325301207, 0.4298245614035088, 0.46438746438746437, 0.48821081830790564, 0.5, 0.5322793148880105, 0.5611325611325612, 0.5901639344262295, 0.6199261992619927, 0.6425992779783393, 0.664319248826291, 0.6926605504587156, 0.7052154195011338, 0.7109111361079865, 0.7250554323725056] 
	t5 = [0, 5.082035779953003, 10.03375506401062, 15.081164121627808, 20.06731343269348, 25.032811641693115, 30.095516443252563, 35.01844263076782, 40.07487607002258, 45.06933307647705, 50.108596324920654, 55.049222469329834, 60.050447940826416] 
	q5 = [0.3540372670807453, 0.4035874439461884, 0.44956772334293943, 0.4825662482566248, 0.5020352781546811, 0.5392670157068062, 0.5834394904458599, 0.6113861386138614, 0.6425992779783393, 0.6721120186697783, 0.7001140250855188, 0.708803611738149, 0.7250554323725056] 
	t6 = [0, 6.019402027130127, 12.129953622817993, 18.0791494846344, 24.02921199798584, 30.003316640853882, 36.143982887268066, 42.08425807952881, 48.01435899734497, 54.06137299537659, 60.03826713562012] 
	q6 = [0.3540372670807453, 0.4154302670623146, 0.4665718349928877, 0.49108367626886146, 0.5303430079155673, 0.5816326530612245, 0.6199261992619927, 0.6524317912218268, 0.6896551724137931, 0.7081447963800904, 0.7250554323725056] 
	t7 = [0, 7.0436341762542725, 14.019613265991211, 21.069591522216797, 28.106915950775146, 35.076765298843384, 42.08849811553955, 49.105310916900635, 56.1301212310791] 
	q7 = [0.3540372670807453, 0.4188790560471976, 0.476056338028169, 0.5114401076716015, 0.5655526992287918, 0.6163366336633663, 0.6532544378698225, 0.6971428571428571, 0.7109111361079865] 
	t8 = [0, 8.063727617263794, 16.018582344055176, 24.105071544647217, 32.07906699180603, 40.03816866874695, 48.06624627113342, 56.08234691619873] 
	q8 = [0.3540372670807453, 0.4298245614035088, 0.4888888888888889, 0.5322793148880105, 0.5944584382871536, 0.6442307692307693, 0.6926605504587156, 0.7094594594594593] 
	t9 = [0, 9.080902576446533, 18.079935550689697, 27.168427228927612, 36.09164524078369, 45.09276723861694, 54.138832569122314] 
	q9 = [0.3540372670807453, 0.43831640058055155, 0.49108367626886146, 0.5599999999999999, 0.6248462484624846, 0.6767441860465115, 0.7089467723669308] 
	t10 = [0, 10.11781620979309, 20.055987119674683, 30.108210563659668, 40.12049055099487, 50.06593942642212, 60.00060296058655] 
	q10 = [0.3540372670807453, 0.4473304473304473, 0.5040650406504065, 0.5852417302798982, 0.6474820143884892, 0.7001140250855188, 0.726467331118494] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.1030092239379883, 2.063704252243042, 3.0482475757598877, 4.123750925064087, 5.082269906997681, 6.017186641693115, 7.145286560058594, 8.101431608200073, 9.140265941619873, 10.067831754684448, 11.079705715179443, 12.013649225234985, 13.10286545753479, 14.058751583099365, 15.040968656539917, 16.10922074317932, 17.090441465377808, 18.054452657699585, 19.039921283721924, 20.022618055343628, 21.041590213775635, 22.122039079666138, 23.134217023849487, 24.063437700271606, 25.016008377075195, 26.08121633529663, 27.0536892414093, 28.017062187194824, 29.019588470458984, 30.116769790649414, 31.059498071670532, 32.124194860458374, 33.095258951187134, 34.0118408203125, 35.13607692718506, 36.07550072669983, 37.01475667953491, 38.129560708999634, 39.10048317909241, 40.01898264884949, 41.13329768180847, 42.05424761772156, 43.0554473400116, 44.13135552406311, 45.11852312088013, 46.077088356018066, 47.023446559906006, 48.131181478500366, 49.08042335510254, 50.063321590423584, 51.0069625377655, 52.13643550872803, 53.10966348648071, 54.048784494400024, 55.01206564903259, 56.08349418640137, 57.05999255180359, 58.01492404937744, 59.14613223075867, 60.09772539138794] 
	q1 = [0.3723076923076923, 0.3822629969418961, 0.3860182370820669, 0.3981900452488688, 0.40657698056801195, 0.41777777777777775, 0.4264705882352941, 0.4333821376281113, 0.4402332361516034, 0.4434782608695652, 0.4502164502164502, 0.4606580829756795, 0.471590909090909, 0.4745762711864407, 0.47752808988764045, 0.48603351955307256, 0.49722222222222223, 0.5013850415512465, 0.5082417582417582, 0.5163934426229508, 0.5217391304347827, 0.5277401894451963, 0.5349462365591399, 0.5401069518716577, 0.5444887118193891, 0.554089709762533, 0.5654450261780105, 0.5721716514954487, 0.5777202072538861, 0.5861182519280206, 0.5925925925925927, 0.5989847715736041, 0.6120906801007556, 0.619047619047619, 0.6217228464419476, 0.630407911001236, 0.638036809815951, 0.6430317848410758, 0.6455542021924482, 0.6504854368932039, 0.6553808948004837, 0.6618705035971223, 0.6682577565632458, 0.6729857819905213, 0.673733804475854, 0.6853801169590643, 0.68997668997669, 0.6960556844547564, 0.7020785219399539, 0.7057471264367816, 0.7100456621004566, 0.7129840546697038, 0.7150964812712827, 0.7165532879818595, 0.7209039548022599, 0.7235955056179774, 0.7250280583613916, 0.7262569832402234, 0.7290969899665553, 0.7347391786903441, 0.7367256637168141] 
	t2 = [0, 2.150472402572632, 4.028414964675903, 6.004720211029053, 8.015043497085571, 10.115280151367188, 12.120786428451538, 14.115406274795532, 16.137782096862793, 18.135822057724, 20.106375694274902, 22.130178689956665, 24.013665199279785, 26.128973484039307, 28.120365142822266, 30.085080862045288, 32.04311203956604, 34.052002906799316, 36.0557804107666, 38.04646039009094, 40.050944328308105, 42.047563314437866, 44.07305359840393, 46.10812497138977, 48.1051139831543, 50.01018738746643, 52.05086922645569, 54.053693771362305, 56.022727727890015, 58.02956509590149, 60.130239725112915] 
	q2 = [0.3723076923076923, 0.3884673748103187, 0.40657698056801195, 0.4264705882352941, 0.4402332361516034, 0.4546762589928058, 0.4724186704384724, 0.482468443197756, 0.49930651872399445, 0.5157750342935529, 0.5257452574525745, 0.538152610441767, 0.5502645502645502, 0.5729166666666666, 0.5868725868725869, 0.5972045743329097, 0.617314930991217, 0.6294919454770755, 0.6430317848410758, 0.6504854368932039, 0.6618705035971223, 0.6745562130177515, 0.6838407494145199, 0.699074074074074, 0.7064220183486238, 0.7121729237770194, 0.7202718006795016, 0.7250280583613916, 0.7262569832402234, 0.7333333333333335, 0.7378854625550662] 
	t3 = [0, 3.0224995613098145, 6.099388360977173, 9.025258302688599, 12.01068639755249, 15.111251592636108, 18.02826499938965, 21.028308629989624, 24.021811723709106, 27.027323722839355, 30.084917306900024, 33.11915826797485, 36.04474854469299, 39.09536051750183, 42.14131808280945, 45.078678369522095, 48.00451850891113, 51.094791650772095, 54.03166389465332, 57.02432608604431, 60.015491247177124] 
	q3 = [0.3723076923076923, 0.3981900452488688, 0.4264705882352941, 0.4486251808972504, 0.4724186704384724, 0.49303621169916434, 0.5150684931506849, 0.5336927223719676, 0.5521796565389696, 0.5784695201037614, 0.5972045743329097, 0.6217228464419476, 0.6430317848410758, 0.6545893719806763, 0.6745562130177515, 0.6915017462165308, 0.7064220183486238, 0.7165532879818595, 0.726457399103139, 0.731924360400445, 0.7392739273927392] 
	t4 = [0, 4.0444440841674805, 8.129435300827026, 12.138908386230469, 16.13841462135315, 20.11960196495056, 24.018762588500977, 28.118215084075928, 32.020986557006836, 36.09224605560303, 40.055668115615845, 44.08931040763855, 48.052552700042725, 52.01267409324646, 56.14071869850159, 60.01145267486572] 
	q4 = [0.3723076923076923, 0.40895522388059696, 0.43959243085880634, 0.4724186704384724, 0.5013850415512465, 0.5277401894451963, 0.5559947299077733, 0.5905006418485237, 0.6182728410513142, 0.6446886446886447, 0.6650717703349283, 0.6884480746791132, 0.7108571428571427, 0.7200902934537247, 0.7305122494432071, 0.7406593406593408] 
	t5 = [0, 5.0669567584991455, 10.047478437423706, 15.007227897644043, 20.017988920211792, 25.029479265213013, 30.058659553527832, 35.030277252197266, 40.01895093917847, 45.099376916885376, 50.10574412345886, 55.064976930618286, 60.03094458580017] 
	q5 = [0.3723076923076923, 0.41949778434268836, 0.45689655172413796, 0.49513212795549383, 0.5284552845528456, 0.5673202614379085, 0.605830164765526, 0.6404907975460123, 0.6650717703349283, 0.6945412311265969, 0.7159090909090909, 0.7262569832402234, 0.7400881057268723] 
	t6 = [0, 6.097805023193359, 12.00475025177002, 18.129655361175537, 24.10369563102722, 30.107364892959595, 36.09259915351868, 42.130199670791626, 48.10068941116333, 54.04025149345398, 60.01785373687744] 
	q6 = [0.3723076923076923, 0.4287812041116006, 0.47308781869688393, 0.5170998632010944, 0.5578947368421052, 0.6065989847715736, 0.6446886446886447, 0.6745283018867925, 0.7108571428571427, 0.7250280583613916, 0.7414741474147414] 
	t7 = [0, 7.052329063415527, 14.040235757827759, 21.052995681762695, 28.052836179733276, 35.04178047180176, 42.00074481964111, 49.14455699920654, 56.13566279411316] 
	q7 = [0.3723076923076923, 0.4333821376281113, 0.484593837535014, 0.5329744279946164, 0.5905006418485237, 0.642156862745098, 0.6745283018867925, 0.7129840546697038, 0.731924360400445] 
	t8 = [0, 8.06373119354248, 16.007511615753174, 24.071488857269287, 32.022393465042114, 40.11180400848389, 48.021960973739624, 56.10156226158142] 
	q8 = [0.3723076923076923, 0.43731778425655976, 0.5020804438280166, 0.5559947299077733, 0.6207759699624531, 0.6682577565632458, 0.7093821510297483, 0.731924360400445] 
	t9 = [0, 9.138057947158813, 18.02794885635376, 27.144920825958252, 36.058006286621094, 45.0930540561676, 54.06155443191528] 
	q9 = [0.3723076923076923, 0.44637681159420295, 0.5178082191780822, 0.5839793281653747, 0.6446886446886447, 0.6960556844547564, 0.7272727272727273] 
	t10 = [0, 10.102262020111084, 20.038676261901855, 30.13820481300354, 40.0678277015686, 50.02318811416626, 60.10319185256958] 
	q10 = [0.3723076923076923, 0.4591104734576758, 0.5311653116531166, 0.610126582278481, 0.6690561529271207, 0.7159090909090909, 0.7400881057268723] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.127983570098877, 2.09053111076355, 3.044337034225464, 4.133875846862793, 5.088654518127441, 6.054192781448364, 7.008846998214722, 8.084167242050171, 9.034786939620972, 10.054585456848145, 11.028083324432373, 12.102537393569946, 13.088044166564941, 14.051056146621704, 15.143431186676025, 16.098470449447632, 17.042389392852783, 18.0047025680542, 19.10271191596985, 20.053636074066162, 21.008501052856445, 22.12075424194336, 23.081130266189575, 24.007304191589355, 25.041085243225098, 26.140921115875244, 27.036065340042114, 28.13520050048828, 29.095775842666626, 30.02222990989685, 31.106432914733887, 32.04249835014343, 33.02940917015076, 34.13755559921265, 35.107558488845825, 36.025333881378174, 37.141228675842285, 38.09767556190491, 39.04420256614685, 40.11772871017456, 41.091843605041504, 42.013683795928955, 43.04573392868042, 44.04503107070923, 45.1362042427063, 46.11585593223572, 47.05588102340698, 48.00792670249939, 49.00928497314453, 50.08400297164917, 51.02763605117798, 52.1260781288147, 53.102423429489136, 54.03173542022705, 55.130013942718506, 56.08111381530762, 57.069029808044434, 58.0178599357605, 59.10632252693176, 60.086450815200806] 
	q1 = [0.3385579937304075, 0.3483670295489891, 0.35802469135802467, 0.3619631901840491, 0.37442922374429227, 0.3812405446293495, 0.3903903903903904, 0.3970149253731343, 0.4065281899109793, 0.41124260355029585, 0.4222873900293256, 0.42794759825327516, 0.4370477568740955, 0.44316546762589926, 0.4507845934379458, 0.4589235127478754, 0.4641350210970464, 0.4671328671328671, 0.4658298465829846, 0.4736842105263158, 0.484181568088033, 0.4876712328767124, 0.49389416553595655, 0.5020242914979758, 0.510752688172043, 0.520694259012016, 0.5272969374167776, 0.5358090185676393, 0.5485564304461942, 0.5572916666666666, 0.565891472868217, 0.5750962772785623, 0.5805626598465472, 0.5888324873096445, 0.5959595959595959, 0.6047678795483062, 0.6117353308364545, 0.6127023661270237, 0.6205191594561187, 0.6282208588957056, 0.6324786324786325, 0.6366950182260025, 0.6424242424242426, 0.6481927710843374, 0.6555023923444977, 0.661904761904762, 0.6690307328605201, 0.6713780918727915, 0.6791569086651054, 0.6837209302325581, 0.6867749419953597, 0.6906141367323292, 0.6936416184971098, 0.6943483275663207, 0.695752009184845, 0.694954128440367, 0.6979405034324944, 0.6970387243735763, 0.6977272727272726, 0.705084745762712, 0.7094594594594595] 
	t2 = [0, 2.119194507598877, 4.127839803695679, 6.115299701690674, 8.148690938949585, 10.085306882858276, 12.148921966552734, 14.021759033203125, 16.044113159179688, 18.095803260803223, 20.097209215164185, 22.0673770904541, 24.067487955093384, 26.020296573638916, 28.11257791519165, 30.08832597732544, 32.06473731994629, 34.09242582321167, 36.098392724990845, 38.13139986991882, 40.107399463653564, 42.117276430130005, 44.08612513542175, 46.12070417404175, 48.02367448806763, 50.02621912956238, 52.02880620956421, 54.01546263694763, 56.02493929862976, 58.0898711681366, 60.12642693519592] 
	q2 = [0.3385579937304075, 0.35802469135802467, 0.3768996960486322, 0.39280359820089955, 0.4065281899109793, 0.424597364568082, 0.4370477568740955, 0.45584045584045585, 0.4641350210970464, 0.4700973574408901, 0.48559670781893005, 0.5006765899864682, 0.5167336010709505, 0.5338645418326693, 0.5535248041775458, 0.5732647814910025, 0.5877862595419847, 0.6030150753768844, 0.6127023661270237, 0.6282208588957056, 0.6366950182260025, 0.6457831325301205, 0.661904761904762, 0.6713780918727915, 0.6852497096399536, 0.6906141367323292, 0.6943483275663207, 0.6964490263459335, 0.6993166287015945, 0.705084745762712, 0.7109111361079864] 
	t3 = [0, 3.0263638496398926, 6.052764415740967, 9.06777024269104, 12.050041913986206, 15.005897998809814, 18.06861925125122, 21.014825582504272, 24.057015419006348, 27.051006317138672, 30.013750076293945, 33.009965896606445, 36.108182430267334, 39.06739139556885, 42.09634804725647, 45.122315645217896, 48.07561159133911, 51.14479899406433, 54.04135060310364, 57.1062228679657, 60.063403606414795] 
	q3 = [0.3385579937304075, 0.36447166921898927, 0.39280359820089955, 0.4153166421207658, 0.4370477568740955, 0.4611032531824611, 0.47434119278779474, 0.49318801089918257, 0.520694259012016, 0.5492772667542707, 0.5750962772785623, 0.5984848484848484, 0.6161490683229813, 0.6341463414634148, 0.6506602641056423, 0.6706021251475797, 0.6875725900116144, 0.6951501154734411, 0.6979405034324944, 0.705084745762712, 0.7166853303471444] 
	t4 = [0, 4.068777084350586, 8.044251203536987, 12.009530782699585, 16.037070989608765, 20.003596305847168, 24.11369824409485, 28.05394458770752, 32.02086615562439, 36.125585079193115, 40.093491554260254, 44.0425922870636, 48.094433546066284, 52.06624436378479, 56.02646040916443, 60.074283599853516] 
	q4 = [0.3385579937304075, 0.37442922374429227, 0.4065281899109793, 0.4370477568740955, 0.46844319775596066, 0.4876712328767124, 0.5213903743315508, 0.5598958333333333, 0.5931558935361216, 0.6178660049627792, 0.6424242424242426, 0.6658767772511848, 0.686046511627907, 0.695752009184845, 0.7000000000000001, 0.7166853303471444] 
	t5 = [0, 5.076750993728638, 10.062618255615234, 15.117804527282715, 20.148205757141113, 25.058228015899658, 30.035404682159424, 35.052016735076904, 40.098143339157104, 45.1015522480011, 50.125993728637695, 55.03433084487915, 60.04161596298218] 
	q5 = [0.3385579937304075, 0.38310708898944196, 0.4298245614035087, 0.4667609618104668, 0.4876712328767124, 0.5292553191489362, 0.5776636713735558, 0.6142322097378278, 0.6424242424242426, 0.6698113207547169, 0.6944444444444445, 0.6978335233751425, 0.7152466367713006] 
	t6 = [0, 6.102589130401611, 12.119754791259766, 18.11874294281006, 24.038565635681152, 30.079294681549072, 36.13208532333374, 42.09480357170105, 48.074111461639404, 54.03696870803833, 60.00298833847046] 
	q6 = [0.3385579937304075, 0.39280359820089955, 0.44219653179190754, 0.47434119278779474, 0.5213903743315508, 0.5776636713735558, 0.6212871287128713, 0.6555023923444977, 0.6875725900116144, 0.6986301369863014, 0.7166853303471444] 
	t7 = [0, 7.035925626754761, 14.093968629837036, 21.138636112213135, 28.131900548934937, 35.07059717178345, 42.05191159248352, 49.10951280593872, 56.13604164123535] 
	q7 = [0.3385579937304075, 0.39940387481371087, 0.46088193456614507, 0.4972826086956521, 0.5654993514915694, 0.6134663341645885, 0.6555023923444977, 0.691415313225058, 0.7029478458049886] 
	t8 = [0, 8.042822122573853, 16.101650714874268, 24.13732123374939, 32.119710206985474, 40.01584768295288, 48.069557189941406, 56.059210777282715] 
	q8 = [0.3385579937304075, 0.4065281899109793, 0.4691011235955056, 0.5233644859813084, 0.5974683544303797, 0.6424242424242426, 0.6898954703832753, 0.7029478458049886] 
	t9 = [0, 9.090478897094727, 18.021793127059937, 27.16399049758911, 36.06161594390869, 45.139002084732056, 54.08287262916565] 
	q9 = [0.3385579937304075, 0.41764705882352937, 0.4756606397774687, 0.5549738219895288, 0.6220570012391573, 0.6729411764705882, 0.6978335233751425] 
	t10 = [0, 10.137898206710815, 20.123607397079468, 30.106169939041138, 40.030160903930664, 50.11410641670227, 60.041616439819336] 
	q10 = [0.3385579937304075, 0.42627737226277373, 0.49175824175824173, 0.5794871794871795, 0.6424242424242426, 0.6944444444444445, 0.7174887892376681] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.0997211933135986, 2.0291857719421387, 3.1347126960754395, 4.07358193397522, 5.026141166687012, 6.098368167877197, 7.048664093017578, 8.131555080413818, 9.022128105163574, 10.006364345550537, 11.122247457504272, 12.053514957427979, 13.073466300964355, 14.0400972366333, 15.131026983261108, 16.061623573303223, 17.045853853225708, 18.12671947479248, 19.13994288444519, 20.0796160697937, 21.087053775787354, 22.074949026107788, 23.09592056274414, 24.058574438095093, 25.004663705825806, 26.10650396347046, 27.115460872650146, 28.0721378326416, 29.02433943748474, 30.120976209640503, 31.078723430633545, 32.00052213668823, 33.10694098472595, 34.03584146499634, 35.13521957397461, 36.061317443847656, 37.03960204124451, 38.116780281066895, 39.058403730392456, 40.13403677940369, 41.074589014053345, 42.14087247848511, 43.009055376052856, 44.08847975730896, 45.048038721084595, 46.02589511871338, 47.03127908706665, 48.09587287902832, 49.03518199920654, 50.10567855834961, 51.07545852661133, 52.000449657440186, 53.13817644119263, 54.11677145957947, 55.054314374923706, 56.13048553466797, 57.10110402107239, 58.08264780044556, 59.05221748352051, 60.14639687538147] 
	q1 = [0.3192771084337349, 0.33183856502242154, 0.3387815750371471, 0.3480825958702065, 0.3601756954612006, 0.37209302325581395, 0.37518037518037517, 0.3839541547277937, 0.39142857142857146, 0.3977272727272727, 0.4067796610169492, 0.41678321678321684, 0.4200278164116829, 0.4281767955801105, 0.43347050754458166, 0.44414168937329696, 0.44565217391304346, 0.45283018867924524, 0.45783132530120485, 0.4674634794156706, 0.47354497354497355, 0.4815789473684211, 0.49214659685863876, 0.49479166666666674, 0.5038759689922481, 0.5134788189987163, 0.5242966751918158, 0.5304568527918782, 0.5415617128463477, 0.5506883604505632, 0.5607940446650124, 0.5696670776818743, 0.580171358629131, 0.5888077858880778, 0.593939393939394, 0.6040914560770156, 0.6083832335329342, 0.6159334126040428, 0.6241134751773049, 0.6305882352941177, 0.6330597889800703, 0.640279394644936, 0.6465816917728853, 0.6528258362168398, 0.6597938144329896, 0.6628571428571428, 0.6666666666666666, 0.671201814058957, 0.6749435665914221, 0.6786516853932585, 0.6816143497757846, 0.6823266219239374, 0.6837988826815643, 0.688195991091314, 0.6903440621531632, 0.6902654867256638, 0.6923925027563397, 0.6938325991189429, 0.6981339187705817, 0.7016393442622951, 0.7022900763358779] 
	t2 = [0, 2.1319332122802734, 4.107113838195801, 6.083642959594727, 8.056230783462524, 10.115917682647705, 12.025886058807373, 14.058969736099243, 16.062345266342163, 18.08285093307495, 20.124064683914185, 22.04505681991577, 24.110080003738403, 26.144450187683105, 28.061836004257202, 30.06149911880493, 32.03038311004639, 34.13989806175232, 36.12132453918457, 38.14855480194092, 40.13193964958191, 42.12902069091797, 44.01773285865784, 46.049163818359375, 48.08924198150635, 50.09294414520264, 52.09969449043274, 54.139509439468384, 56.005587339401245, 58.043723583221436, 60.09546947479248] 
	q2 = [0.3192771084337349, 0.34124629080118696, 0.36257309941520466, 0.377521613832853, 0.39372325249643364, 0.4135021097046413, 0.4222222222222222, 0.43775649794801635, 0.44986449864498645, 0.4640000000000001, 0.4775725593667547, 0.49608355091383816, 0.5122265122265123, 0.529262086513995, 0.5488721804511277, 0.5679012345679013, 0.587088915956151, 0.6040914560770156, 0.6175771971496437, 0.6298472385428907, 0.641860465116279, 0.6543778801843317, 0.6643835616438356, 0.6704416761041903, 0.6786516853932585, 0.6823266219239374, 0.688195991091314, 0.6917127071823204, 0.6967032967032967, 0.7016393442622951, 0.7079261672095548] 
	t3 = [0, 3.141888380050659, 6.0203163623809814, 9.05074954032898, 12.100919723510742, 15.016130924224854, 18.063554763793945, 21.004321575164795, 24.0409197807312, 27.127579927444458, 30.098626375198364, 33.1444833278656, 36.13036775588989, 39.02939987182617, 42.048367738723755, 45.10089039802551, 48.06086540222168, 51.12092185020447, 54.10667586326599, 57.00876498222351, 60.152602672576904] 
	q3 = [0.3192771084337349, 0.35051546391752575, 0.377521613832853, 0.4028368794326242, 0.4244105409153952, 0.44565217391304346, 0.4640000000000001, 0.4901703800786369, 0.5141388174807199, 0.5422446406052964, 0.5714285714285715, 0.5956416464891041, 0.6192170818505337, 0.6362573099415204, 0.6574712643678161, 0.6696935300794551, 0.6816143497757846, 0.688195991091314, 0.6931567328918322, 0.6987951807228916, 0.7121212121212122] 
	t4 = [0, 4.056415557861328, 8.101706981658936, 12.142591953277588, 16.03187084197998, 20.1338312625885, 24.05565619468689, 28.100095987319946, 32.10418462753296, 36.002543210983276, 40.089178800582886, 44.05491614341736, 48.09125876426697, 52.01987957954407, 56.137874364852905, 60.16447615623474] 
	q4 = [0.3192771084337349, 0.36257309941520466, 0.39372325249643364, 0.4271844660194174, 0.4519621109607578, 0.47957839262187085, 0.5148005148005147, 0.5506883604505632, 0.5888077858880778, 0.6192170818505337, 0.6450116009280742, 0.6651480637813212, 0.6816143497757846, 0.6903440621531632, 0.6981339187705817, 0.7121212121212122] 
	t5 = [0, 5.059758186340332, 10.00617504119873, 15.11117434501648, 20.064963340759277, 25.11788272857666, 30.123286485671997, 35.121108293533325, 40.09102535247803, 45.06090021133423, 50.003679513931274, 55.01674795150757, 60.09431004524231] 
	q5 = [0.3192771084337349, 0.3691860465116279, 0.4140845070422535, 0.44625850340136053, 0.48021108179419525, 0.5249679897567222, 0.5731857318573186, 0.6109785202863962, 0.6450116009280742, 0.671201814058957, 0.6837988826815643, 0.6952695269526953, 0.7121212121212122] 
	t6 = [0, 6.1251842975616455, 12.090412616729736, 18.067492961883545, 24.122305393218994, 30.05314564704895, 36.143540143966675, 42.006526947021484, 48.055180311203, 54.023173093795776, 60.11029267311096] 
	q6 = [0.3192771084337349, 0.3798561151079136, 0.4277777777777778, 0.4660452729693742, 0.519280205655527, 0.5738916256157636, 0.6208530805687204, 0.658256880733945, 0.6816143497757846, 0.6923925027563397, 0.7107258938244853] 
	t7 = [0, 7.027250289916992, 14.05916166305542, 21.123109579086304, 28.090202569961548, 35.133984327316284, 42.018184423446655, 49.078848123550415, 56.104140281677246] 
	q7 = [0.3192771084337349, 0.38626609442060084, 0.4453551912568306, 0.4927916120576671, 0.5549999999999999, 0.6142857142857143, 0.6597938144329896, 0.6823266219239374, 0.6987951807228916] 
	t8 = [0, 8.019148349761963, 16.0737042427063, 24.090128183364868, 32.14428901672363, 40.03809309005737, 48.00721049308777, 56.02180886268616] 
	q8 = [0.3192771084337349, 0.3908701854493581, 0.45466847090663054, 0.5173745173745173, 0.5888077858880778, 0.6450116009280742, 0.6816143497757846, 0.6973684210526316] 
	t9 = [0, 9.095239162445068, 18.05953812599182, 27.000991344451904, 36.108206033706665, 45.095961570739746, 54.0476233959198] 
	q9 = [0.3192771084337349, 0.4028368794326242, 0.4666666666666666, 0.5491183879093199, 0.6224852071005917, 0.671945701357466, 0.6923925027563397] 
	t10 = [0, 10.128945350646973, 20.110642194747925, 30.026047945022583, 40.109129428863525, 50.02420735359192, 60.0335488319397] 
	q10 = [0.3192771084337349, 0.4135021097046413, 0.4848484848484848, 0.5788177339901477, 0.6465816917728853, 0.6867335562987736, 0.7099567099567099] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.095921277999878, 2.0218379497528076, 3.006847858428955, 4.083143949508667, 5.062376022338867, 6.136995077133179, 7.103214263916016, 8.033205270767212, 9.128327369689941, 10.113569736480713, 11.144624471664429, 12.079001665115356, 13.025989532470703, 14.099304676055908, 15.074772834777832, 16.066891193389893, 17.010645389556885, 18.093052864074707, 19.034507513046265, 20.103458404541016, 21.119258642196655, 22.041120767593384, 23.04458999633789, 24.023241996765137, 25.141289472579956, 26.121192693710327, 27.021522045135498, 28.133565664291382, 29.022092580795288, 30.096306324005127, 31.08388614654541, 32.009833335876465, 33.105005502700806, 34.06264352798462, 35.00613713264465, 36.11026477813721, 37.06058692932129, 38.017828702926636, 39.142805337905884, 40.081618309020996, 41.05019426345825, 42.134809255599976, 43.12118411064148, 44.04377102851868, 45.133915424346924, 46.11336898803711, 47.088860750198364, 48.04309391975403, 49.018256187438965, 50.00775742530823, 51.13337707519531, 52.05008387565613, 53.02048873901367, 54.080490589141846, 55.05051589012146, 56.1165189743042, 57.122166872024536, 58.11564803123474, 59.09161448478699, 60.04397487640381] 
	q1 = [0.3644859813084112, 0.3678516228748068, 0.3742331288343558, 0.3835616438356165, 0.3933434190620272, 0.4047976011994004, 0.4160475482912333, 0.4300441826215022, 0.44152046783625737, 0.446064139941691, 0.45217391304347826, 0.4604316546762591, 0.4670487106017192, 0.4744318181818182, 0.48169014084507034, 0.48314606741573035, 0.4825662482566248, 0.49307479224376727, 0.5013774104683196, 0.5048010973936901, 0.510231923601637, 0.5149051490514905, 0.5209176788124157, 0.5315436241610739, 0.5367156208277704, 0.5464190981432361, 0.554089709762533, 0.5583224115334208, 0.5658409387222947, 0.5751295336787565, 0.5842985842985843, 0.5907928388746803, 0.5972045743329097, 0.605296343001261, 0.6140350877192983, 0.6192259675405742, 0.630407911001236, 0.6346863468634687, 0.6355828220858896, 0.6414634146341464, 0.6496969696969698, 0.6545893719806763, 0.6618705035971223, 0.6698450536352801, 0.6753554502369669, 0.6768867924528302, 0.6830985915492958, 0.6892523364485982, 0.6953488372093023, 0.701388888888889, 0.7035755478662055, 0.7072330654420208, 0.7101947308132875, 0.7160775370581528, 0.7173666288308741, 0.7194570135746606, 0.7200902934537247, 0.7237880496054114, 0.7280898876404495, 0.7309417040358744, 0.7301231802911534] 
	t2 = [0, 2.1183416843414307, 4.131761789321899, 6.0023486614227295, 8.127097845077515, 10.117730617523193, 12.00234580039978, 14.131154537200928, 16.02101469039917, 18.036929607391357, 20.00442624092102, 22.045623064041138, 24.115000247955322, 26.014843940734863, 28.02685022354126, 30.090629816055298, 32.07964062690735, 34.08499526977539, 36.09969758987427, 38.09765839576721, 40.09112620353699, 42.052077531814575, 44.05199217796326, 46.094200134277344, 48.12712907791138, 50.041494607925415, 52.0360107421875, 54.058895111083984, 56.08159112930298, 58.1441764831543, 60.06178545951843] 
	q2 = [0.3644859813084112, 0.3767228177641654, 0.3957703927492447, 0.4160475482912333, 0.44152046783625737, 0.4544138929088278, 0.4714285714285714, 0.48169014084507034, 0.4867872044506259, 0.5034387895460798, 0.5115646258503401, 0.5295698924731184, 0.5444887118193891, 0.5559947299077733, 0.5732814526588845, 0.58898847631242, 0.6060606060606061, 0.6192259675405742, 0.6346863468634687, 0.6414634146341464, 0.6562123039806996, 0.6714285714285714, 0.6784452296819788, 0.6923076923076923, 0.7028901734104047, 0.7072330654420208, 0.7175398633257404, 0.7194570135746606, 0.7266591676040495, 0.7295173961840629, 0.7321428571428571] 
	t3 = [0, 3.041562557220459, 6.060063362121582, 9.077831029891968, 12.010854959487915, 15.09192180633545, 18.063382148742676, 21.01209855079651, 24.004645109176636, 27.004866123199463, 30.095933198928833, 33.02318334579468, 36.07372045516968, 39.029513359069824, 42.03345608711243, 45.09254693984985, 48.06186628341675, 51.027145862579346, 54.11053824424744, 57.033785820007324, 60.006670236587524] 
	q3 = [0.3644859813084112, 0.3835616438356165, 0.41839762611275966, 0.4483260553129549, 0.470756062767475, 0.48391608391608393, 0.5027472527472527, 0.5169147496617049, 0.5444887118193891, 0.5635648754914809, 0.5907928388746803, 0.6140350877192983, 0.6354679802955666, 0.6513317191283293, 0.6714285714285714, 0.6861826697892273, 0.7028901734104047, 0.7131428571428571, 0.7186440677966102, 0.7280898876404495, 0.7321428571428571] 
	t4 = [0, 4.063344955444336, 8.132242202758789, 12.045550346374512, 16.01170516014099, 20.01059603691101, 24.096121311187744, 28.06563115119934, 32.03517937660217, 36.10768413543701, 40.06373453140259, 44.00367546081543, 48.0032172203064, 52.058335304260254, 56.0301308631897, 60.106767416000366] 
	q4 = [0.3644859813084112, 0.3957703927492447, 0.44152046783625737, 0.47293447293447294, 0.4923504867872044, 0.5108695652173914, 0.5490716180371353, 0.5777202072538861, 0.6070528967254408, 0.6371463714637146, 0.6586538461538461, 0.6830985915492958, 0.7043879907621245, 0.7173666288308741, 0.7266591676040495, 0.7363737486095661] 
	t5 = [0, 5.044769763946533, 10.047955751419067, 15.049553155899048, 20.08365774154663, 25.065622806549072, 30.13967752456665, 35.04891848564148, 40.071205377578735, 45.06496000289917, 50.13587260246277, 55.05809259414673, 60.01681590080261] 
	q5 = [0.3644859813084112, 0.40956651718983555, 0.45664739884393063, 0.48391608391608393, 0.5128900949796472, 0.5548216644649935, 0.5936305732484076, 0.6320987654320989, 0.6602641056422569, 0.6907817969661609, 0.7124856815578465, 0.7266591676040495, 0.7357859531772576] 
	t6 = [0, 6.105729341506958, 12.131676435470581, 18.041372299194336, 24.11858367919922, 30.088322401046753, 36.009644746780396, 42.104498624801636, 48.12012314796448, 54.029640436172485, 60.00212740898132] 
	q6 = [0.3644859813084112, 0.4207407407407408, 0.47226173541963024, 0.5041322314049587, 0.547144754316069, 0.5915492957746479, 0.6371463714637146, 0.6737841043890866, 0.7043879907621245, 0.7223476297968396, 0.734375] 
	t7 = [0, 7.018527984619141, 14.071648836135864, 21.107969760894775, 28.07283043861389, 35.117830753326416, 42.08781051635742, 49.01253581047058, 56.09098029136658] 
	q7 = [0.3644859813084112, 0.43235294117647055, 0.48382559774964845, 0.5209176788124157, 0.5829015544041452, 0.6337854500616522, 0.6745562130177515, 0.7080459770114942, 0.7280898876404495] 
	t8 = [0, 8.022206783294678, 16.141350984573364, 24.057955741882324, 32.056875228881836, 40.05914235115051, 48.069348096847534, 56.06524419784546] 
	q8 = [0.3644859813084112, 0.44152046783625737, 0.4965325936199722, 0.5490716180371353, 0.6095717884130983, 0.6626650660264105, 0.7058823529411765, 0.7280898876404495] 
	t9 = [0, 9.081738948822021, 18.136460304260254, 27.040815591812134, 36.138492584228516, 45.09400653839111, 54.002793312072754] 
	q9 = [0.3644859813084112, 0.45058139534883723, 0.5061898211829435, 0.5680628272251309, 0.6363636363636364, 0.6907817969661609, 0.7217194570135747] 
	t10 = [0, 10.109366416931152, 20.08702802658081, 30.04377841949463, 40.01071500778198, 50.13956260681152, 60.07237458229065] 
	q10 = [0.3644859813084112, 0.45887445887445893, 0.5163043478260869, 0.5959079283887468, 0.6626650660264105, 0.7139588100686499, 0.7357859531772576] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.1021363735198975, 2.0337777137756348, 3.0457565784454346, 4.125756025314331, 5.079651355743408, 6.004600286483765, 7.098105430603027, 8.027682304382324, 9.043455362319946, 10.006484270095825, 11.102197170257568, 12.056313753128052, 13.002444744110107, 14.11413311958313, 15.087849140167236, 16.027483701705933, 17.123146057128906, 18.05207324028015, 19.074176788330078, 20.026988983154297, 21.039709329605103, 22.10688543319702, 23.09085202217102, 24.019688606262207, 25.1070237159729, 26.090818881988525, 27.032421588897705, 28.13608741760254, 29.083937644958496, 30.051395893096924, 31.139194011688232, 32.124836921691895, 33.1118061542511, 34.068374156951904, 35.00940465927124, 36.07642602920532, 37.01623773574829, 38.08641695976257, 39.079596757888794, 40.01655697822571, 41.102413177490234, 42.05302596092224, 43.003331899642944, 44.07516694068909, 45.05829954147339, 46.01524043083191, 47.13799071311951, 48.00720143318176, 49.107354164123535, 50.13586688041687, 51.116915225982666, 52.04781150817871, 53.06621265411377, 54.14353561401367, 55.0838418006897, 56.0042359828949, 57.1026177406311, 58.11903524398804, 59.06618690490723, 60.137808084487915] 
	q1 = [0.34359805510534847, 0.35048231511254024, 0.3584, 0.3682539682539683, 0.380503144654088, 0.3875, 0.3950233281493002, 0.404320987654321, 0.4030769230769231, 0.40916030534351144, 0.4127465857359636, 0.42232277526395173, 0.42750373692077726, 0.43154761904761907, 0.43722304283604135, 0.44640234948604995, 0.45321637426900585, 0.4608695652173913, 0.4639769452449568, 0.4721030042918455, 0.47578347578347585, 0.4887005649717514, 0.4943820224719101, 0.4979020979020979, 0.5083333333333333, 0.5158620689655173, 0.5273224043715847, 0.5326086956521738, 0.5390835579514824, 0.5495978552278821, 0.5539280958721705, 0.5604249667994687, 0.570673712021136, 0.5763157894736842, 0.5886990801576872, 0.59375, 0.6020671834625323, 0.6110397946084724, 0.6163682864450127, 0.621656050955414, 0.629582806573957, 0.6347607052896727, 0.6390977443609022, 0.6450809464508095, 0.6526576019777504, 0.6592865928659286, 0.6625766871165644, 0.6699266503667483, 0.6731470230862697, 0.6747572815533981, 0.6755447941888619, 0.6787439613526569, 0.6803377563329313, 0.6787003610108304, 0.6842105263157895, 0.6865315852205006, 0.688836104513064, 0.6934911242603551, 0.6972909305064782, 0.6980023501762632, 0.7010550996483] 
	t2 = [0, 2.1128640174865723, 4.127187728881836, 6.142874479293823, 8.120839595794678, 10.03728723526001, 12.031187295913696, 14.016697645187378, 16.064066171646118, 18.036361932754517, 20.07677984237671, 22.105300903320312, 24.13757300376892, 26.000742435455322, 28.027035236358643, 30.030681371688843, 32.03976130485535, 34.08851981163025, 36.0409414768219, 38.01898646354675, 40.03942060470581, 42.06024742126465, 44.06583333015442, 46.0820734500885, 48.034005641937256, 50.102317810058594, 52.13940501213074, 54.14320993423462, 56.127285957336426, 58.04141116142273, 60.083534240722656] 
	q2 = [0.34359805510534847, 0.3578274760383387, 0.38304552590266877, 0.4, 0.40245775729646704, 0.4157814871016691, 0.429210134128167, 0.4395280235988201, 0.45772594752186585, 0.4677187948350072, 0.48725212464589235, 0.4964936886395512, 0.5138121546961325, 0.5286103542234333, 0.5456989247311828, 0.5604249667994687, 0.5744400527009222, 0.5953002610966057, 0.6110397946084724, 0.621656050955414, 0.6347607052896727, 0.6450809464508095, 0.6592865928659286, 0.6699266503667483, 0.6739393939393941, 0.6803377563329313, 0.6818727490996398, 0.688095238095238, 0.6965761511216055, 0.6995305164319249, 0.7101280558789289] 
	t3 = [0, 3.0333023071289062, 6.064619779586792, 9.09124207496643, 12.049109697341919, 15.10826563835144, 18.013675689697266, 21.09786581993103, 24.050692081451416, 27.098994970321655, 30.031482458114624, 33.081886529922485, 36.02719807624817, 39.05768704414368, 42.11901330947876, 45.00439715385437, 48.12615394592285, 51.12316823005676, 54.08450222015381, 57.081881046295166, 60.042174339294434] 
	q3 = [0.34359805510534847, 0.3734177215189874, 0.4, 0.40916030534351144, 0.43154761904761907, 0.4509516837481699, 0.4677187948350072, 0.4950773558368496, 0.5158620689655173, 0.5378378378378378, 0.5611702127659575, 0.5879265091863517, 0.6128205128205128, 0.629582806573957, 0.6459627329192547, 0.6650306748466258, 0.6755447941888619, 0.6795180722891566, 0.689655172413793, 0.6972909305064782, 0.7116279069767443] 
	t4 = [0, 4.0422186851501465, 8.116942405700684, 12.146333932876587, 16.081167936325073, 20.017083406448364, 24.046795129776, 28.049817323684692, 32.00948667526245, 36.06121635437012, 40.08676028251648, 44.012388706207275, 48.15566611289978, 52.092230558395386, 56.021135568618774, 60.025044679641724] 
	q4 = [0.34359805510534847, 0.38304552590266877, 0.40490797546012275, 0.43154761904761907, 0.4593023255813953, 0.48939179632248936, 0.5158620689655173, 0.5495978552278821, 0.580814717477004, 0.6161745827984596, 0.6398996235884568, 0.6592865928659286, 0.6763636363636363, 0.6842105263157895, 0.6972909305064782, 0.7116279069767443] 
	t5 = [0, 5.050650358200073, 10.003708600997925, 15.078511238098145, 20.101256132125854, 25.11835026741028, 30.00827407836914, 35.06631636619568, 40.05002808570862, 45.069358825683594, 50.0657172203064, 55.03491497039795, 60.080002546310425] 
	q5 = [0.34359805510534847, 0.390015600624025, 0.4181818181818181, 0.45321637426900585, 0.48939179632248936, 0.5232876712328768, 0.5611702127659575, 0.6090322580645161, 0.6398996235884568, 0.6666666666666667, 0.6803377563329313, 0.6950354609929078, 0.7116279069767443] 
	t6 = [0, 6.141274690628052, 12.134759664535522, 18.131950616836548, 24.059369564056396, 30.10676336288452, 36.12246131896973, 42.14112377166748, 48.04063081741333, 54.13890862464905, 60.05724763870239] 
	q6 = [0.34359805510534847, 0.40247678018575844, 0.43219076005961254, 0.46991404011461324, 0.5179063360881543, 0.5630810092961488, 0.617948717948718, 0.650990099009901, 0.6747572815533981, 0.690391459074733, 0.7124563445867287] 
	t7 = [0, 7.018432855606079, 14.047771453857422, 21.075472593307495, 28.090876579284668, 35.06330442428589, 42.12503004074097, 49.04861092567444, 56.093751668930054] 
	q7 = [0.34359805510534847, 0.4030769230769231, 0.4470588235294118, 0.4950773558368496, 0.5476510067114094, 0.6126126126126127, 0.6526576019777504, 0.6795646916565901, 0.6972909305064782] 
	t8 = [0, 8.022361755371094, 16.047178983688354, 24.04736018180847, 32.11922788619995, 40.09229874610901, 48.06780457496643, 56.106467485427856] 
	q8 = [0.34359805510534847, 0.40490797546012275, 0.4615384615384615, 0.5179063360881543, 0.5860709592641261, 0.6424090338770388, 0.6747572815533981, 0.6972909305064782] 
	t9 = [0, 9.073846578598022, 18.071701765060425, 27.11428689956665, 36.038053035736084, 45.05504035949707, 54.05288505554199] 
	q9 = [0.34359805510534847, 0.4140030441400305, 0.46991404011461324, 0.5390835579514824, 0.6187419768934531, 0.6674816625916871, 0.6919431279620852] 
	t10 = [0, 10.136391401290894, 20.10960817337036, 30.015212059020996, 40.01065754890442, 50.10661220550537, 60.02687382698059] 
	q10 = [0.34359805510534847, 0.4229607250755287, 0.4915254237288135, 0.5611702127659575, 0.6424090338770388, 0.6771463119709794, 0.7109557109557109] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.10691499710083, 2.0331573486328125, 3.1260108947753906, 4.053690195083618, 5.008774280548096, 6.074951648712158, 7.017907381057739, 8.007038354873657, 9.13399624824524, 10.134021759033203, 11.076713800430298, 12.004766941070557, 13.143914937973022, 14.083129644393921, 15.036052942276001, 16.10716199874878, 17.117481231689453, 18.109880685806274, 19.083284854888916, 20.078341245651245, 21.049858331680298, 22.001556396484375, 23.105364561080933, 24.05507779121399, 25.0669207572937, 26.02262258529663, 27.14206099510193, 28.09810709953308, 29.03854274749756, 30.12613272666931, 31.086509704589844, 32.009252071380615, 33.138001918792725, 34.10091829299927, 35.05338740348816, 36.012754917144775, 37.11260485649109, 38.06571412086487, 39.03800582885742, 40.11324405670166, 41.00377631187439, 42.1116726398468, 43.0882408618927, 44.0727903842926, 45.03059935569763, 46.12618923187256, 47.06898546218872, 48.030195474624634, 49.004427433013916, 50.070019483566284, 51.04375171661377, 52.11831998825073, 53.05792689323425, 54.12048006057739, 55.119428873062134, 56.03648114204407, 57.0135293006897, 58.09001970291138, 59.06879210472107, 60.02909803390503] 
	q1 = [0.36196319018404904, 0.36585365853658536, 0.38066465256797577, 0.3898050974512744, 0.39821693907875183, 0.4070796460176991, 0.4164222873900293, 0.4256559766763848, 0.4267053701015965, 0.43352601156069365, 0.44189383070301286, 0.4479315263908702, 0.4517045454545454, 0.45915492957746484, 0.46993006993006997, 0.4707520891364902, 0.47790055248618785, 0.4862637362637363, 0.49041095890410963, 0.4959128065395096, 0.503382949932341, 0.5080645161290323, 0.5173333333333333, 0.5298013245033114, 0.5375494071146245, 0.5490196078431373, 0.561038961038961, 0.5658914728682171, 0.5725288831835686, 0.5798212005108557, 0.5888324873096447, 0.5959595959595959, 0.5997490589711417, 0.6034912718204489, 0.607940446650124, 0.6131025957972805, 0.6182266009852216, 0.6266829865361078, 0.6333739342265531, 0.64, 0.644927536231884, 0.6506602641056424, 0.6610978520286396, 0.6634958382877527, 0.6713947990543736, 0.6776470588235294, 0.6869158878504672, 0.68997668997669, 0.694541231126597, 0.7005780346820809, 0.7050691244239631, 0.706559263521289, 0.7064220183486238, 0.7077625570776256, 0.7107061503416856, 0.7113636363636363, 0.7128263337116912, 0.7171945701357465, 0.7163841807909603, 0.7192784667418264, 0.7199100112485939] 
	t2 = [0, 2.1282999515533447, 4.109192848205566, 6.087310075759888, 8.058830499649048, 10.144147634506226, 12.007411241531372, 14.015113592147827, 16.137948274612427, 18.049834489822388, 20.088574409484863, 22.118174076080322, 24.025065660476685, 26.09152388572693, 28.14452314376831, 30.118102550506592, 32.086930990219116, 34.13556241989136, 36.1473548412323, 38.139522075653076, 40.003475189208984, 42.10318565368652, 44.018898487091064, 46.00933241844177, 48.01317834854126, 50.014808177948, 52.00349998474121, 54.118754863739014, 56.00791096687317, 58.0130569934845, 60.04548740386963] 
	q2 = [0.36196319018404904, 0.38009049773755654, 0.400593471810089, 0.4187408491947291, 0.4267053701015965, 0.4441260744985673, 0.453257790368272, 0.4692737430167598, 0.4827586206896552, 0.49180327868852464, 0.5087483176312247, 0.5258964143426296, 0.5433070866141733, 0.5647668393782384, 0.5780051150895141, 0.5959595959595959, 0.6017478152309613, 0.6131025957972805, 0.6266829865361078, 0.64, 0.6506602641056424, 0.665083135391924, 0.6776470588235294, 0.6915017462165309, 0.7020785219399538, 0.7080459770114942, 0.709236031927024, 0.7136363636363636, 0.7171945701357465, 0.7199100112485939, 0.7256438969764839] 
	t3 = [0, 3.035181760787964, 6.02358865737915, 9.024139404296875, 12.061882972717285, 15.130398988723755, 18.137641191482544, 21.036713361740112, 24.096121549606323, 27.071366548538208, 30.000226974487305, 33.003772497177124, 36.08245229721069, 39.01823830604553, 42.021350383758545, 45.12887620925903, 48.002575635910034, 51.109079122543335, 54.10653376579285, 57.04115438461304, 60.10735511779785] 
	q3 = [0.36196319018404904, 0.38680659670164913, 0.4187408491947291, 0.4380403458213257, 0.4554455445544555, 0.4792243767313019, 0.49180327868852464, 0.5160427807486632, 0.5471204188481675, 0.5725288831835686, 0.5959595959595959, 0.607940446650124, 0.63003663003663, 0.646562123039807, 0.6682464454976305, 0.6884480746791132, 0.7020785219399538, 0.7085714285714285, 0.7136363636363636, 0.7192784667418264, 0.7270693512304249] 
	t4 = [0, 4.037683010101318, 8.07369589805603, 12.11129641532898, 16.052417278289795, 20.049153804779053, 24.08671998977661, 28.135047435760498, 32.09221148490906, 36.041375398635864, 40.00469779968262, 44.06150794029236, 48.072017431259155, 52.045923709869385, 56.08645582199097, 60.09875798225403] 
	q4 = [0.36196319018404904, 0.400593471810089, 0.4267053701015965, 0.4554455445544555, 0.48484848484848475, 0.5087483176312247, 0.5471204188481675, 0.5798212005108557, 0.6034912718204489, 0.63003663003663, 0.6555023923444976, 0.6822977725674092, 0.7035755478662054, 0.7107061503416856, 0.7171945701357465, 0.7270693512304249] 
	t5 = [0, 5.084929704666138, 10.142705917358398, 15.064712285995483, 20.05142903327942, 25.08529496192932, 30.018601655960083, 35.129807472229004, 40.03750681877136, 45.00649428367615, 50.093533515930176, 55.10061740875244, 60.11560249328613] 
	q5 = [0.36196319018404904, 0.40942562592047127, 0.4434907010014306, 0.4792243767313019, 0.506056527590848, 0.561038961038961, 0.5994962216624685, 0.6233128834355829, 0.6594982078853047, 0.6884480746791132, 0.7064220183486238, 0.7180067950169874, 0.7270693512304249] 
	t6 = [0, 6.126925230026245, 12.088501691818237, 18.07441520690918, 24.067620038986206, 30.09966516494751, 36.144057512283325, 42.08488941192627, 48.014814376831055, 54.074190855026245, 60.125730752944946] 
	q6 = [0.36196319018404904, 0.42105263157894735, 0.4576271186440678, 0.49386084583901774, 0.5516339869281046, 0.6012578616352201, 0.6317073170731707, 0.6698224852071005, 0.7050691244239631, 0.7150964812712827, 0.7270693512304249] 
	t7 = [0, 7.013762712478638, 14.088975429534912, 21.112998962402344, 28.051905393600464, 35.07333493232727, 42.04901909828186, 49.10675024986267, 56.08526039123535] 
	q7 = [0.36196319018404904, 0.42274052478134116, 0.4720670391061453, 0.5180240320427236, 0.5798212005108557, 0.6216216216216216, 0.6698224852071005, 0.7050691244239631, 0.7186440677966102] 
	t8 = [0, 8.07319712638855, 16.125107049942017, 24.030827522277832, 32.05551815032959, 40.099063873291016, 48.0954430103302, 56.15415549278259] 
	q8 = [0.36196319018404904, 0.4267053701015965, 0.4876033057851239, 0.5516339869281046, 0.6044776119402986, 0.6594982078853047, 0.7035755478662054, 0.7200902934537247] 
	t9 = [0, 9.115409851074219, 18.02003002166748, 27.1347234249115, 36.04389834403992, 45.06472873687744, 54.14420962333679] 
	q9 = [0.36196319018404904, 0.4351585014409221, 0.49453551912568305, 0.5750962772785623, 0.6317073170731707, 0.68997668997669, 0.7180067950169874] 
	t10 = [0, 10.094748973846436, 20.055281162261963, 30.006645917892456, 40.007834672927856, 50.09927201271057, 60.10679745674133] 
	q10 = [0.36196319018404904, 0.44126074498567336, 0.5067385444743935, 0.5994962216624685, 0.6594982078853047, 0.7026406429391505, 0.7293064876957495] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.0998401641845703, 2.0374042987823486, 3.14408540725708, 4.076573371887207, 5.0238823890686035, 6.09792160987854, 7.052130699157715, 8.011599779129028, 9.003006219863892, 10.108438968658447, 11.09995985031128, 12.064262628555298, 13.066782474517822, 14.143784523010254, 15.086788415908813, 16.013402938842773, 17.13958716392517, 18.060953378677368, 19.030399322509766, 20.016411781311035, 21.02458882331848, 22.131459712982178, 23.103347063064575, 24.029974460601807, 25.033129692077637, 26.102672338485718, 27.047095775604248, 28.143844842910767, 29.103252410888672, 30.02474856376648, 31.129640340805054, 32.0805504322052, 33.022727489471436, 34.09340143203735, 35.03726863861084, 36.10426902770996, 37.103726863861084, 38.018969774246216, 39.131147146224976, 40.094038248062134, 41.03647232055664, 42.14428496360779, 43.003658056259155, 44.078498125076294, 45.07732319831848, 46.09712815284729, 47.06604623794556, 48.01229786872864, 49.1361358165741, 50.09096169471741, 51.036396741867065, 52.13699960708618, 53.083893060684204, 54.010818004608154, 55.135703563690186, 56.09234070777893, 57.04170370101929, 58.06517028808594, 59.011375427246094, 60.0745792388916] 
	q1 = [0.36645962732919257, 0.3784615384615384, 0.3877862595419848, 0.3939393939393939, 0.4, 0.40718562874251496, 0.41604754829123325, 0.4183976261127597, 0.42709867452135497, 0.43631039531478766, 0.4476744186046511, 0.455988455988456, 0.45755395683453237, 0.46, 0.4730878186968838, 0.476056338028169, 0.4811188811188811, 0.490984743411928, 0.4965517241379311, 0.5054945054945055, 0.5081967213114754, 0.5108695652173914, 0.5148247978436657, 0.5267379679144385, 0.5326231691078562, 0.5408970976253299, 0.5523560209424084, 0.5617685305591678, 0.5710594315245477, 0.5750962772785623, 0.5816326530612245, 0.5870393900889453, 0.5959595959595959, 0.6040100250626566, 0.6127023661270237, 0.6178660049627791, 0.6271604938271605, 0.628992628992629, 0.6341463414634146, 0.6424242424242426, 0.6481927710843374, 0.6538922155688623, 0.6547619047619048, 0.6627218934911243, 0.669811320754717, 0.6729411764705883, 0.6814469078179697, 0.6837209302325582, 0.6867749419953596, 0.6905311778290992, 0.6935483870967741, 0.6950517836593786, 0.6979405034324943, 0.6963470319634704, 0.6993166287015945, 0.7006802721088435, 0.7028248587570621, 0.7042889390519187, 0.7093153759820428, 0.70996640537514, 0.7157190635451505] 
	t2 = [0, 2.128361225128174, 4.116765260696411, 6.085350513458252, 8.048367738723755, 10.121778726577759, 12.115488767623901, 14.065097093582153, 16.030658960342407, 18.033262014389038, 20.058342695236206, 22.100314617156982, 24.13896131515503, 26.017054557800293, 28.14254331588745, 30.00066113471985, 32.11311745643616, 34.121649503707886, 36.10608720779419, 38.14350748062134, 40.00578045845032, 42.01620435714722, 44.09283089637756, 46.04654359817505, 48.09320569038391, 50.09701371192932, 52.10418891906738, 54.12094211578369, 56.13403844833374, 58.047669887542725, 60.04720115661621] 
	q2 = [0.36645962732919257, 0.3871951219512195, 0.40240240240240244, 0.41604754829123325, 0.4294117647058823, 0.44992743105950656, 0.45689655172413796, 0.4752475247524753, 0.4874651810584958, 0.5013774104683195, 0.5115646258503401, 0.5234899328859061, 0.5396825396825397, 0.5598958333333333, 0.5732647814910027, 0.5870393900889453, 0.6047678795483061, 0.6178660049627791, 0.628992628992629, 0.6424242424242426, 0.6538922155688623, 0.6650887573964498, 0.6745005875440659, 0.6845168800931315, 0.6905311778290992, 0.6965517241379311, 0.6963470319634704, 0.7021517553793885, 0.7080045095828635, 0.7128491620111732, 0.7177777777777777] 
	t3 = [0, 3.0159473419189453, 6.02655553817749, 9.040404796600342, 12.044955015182495, 15.002565860748291, 18.14611554145813, 21.13670516014099, 24.112451791763306, 27.045019388198853, 30.084656238555908, 33.09537625312805, 36.09443688392639, 39.02977466583252, 42.041786432266235, 45.02014207839966, 48.01997137069702, 51.141185998916626, 54.0358464717865, 57.077136754989624, 60.07401084899902] 
	q3 = [0.36645962732919257, 0.3939393939393939, 0.41604754829123325, 0.43859649122807026, 0.45689655172413796, 0.48246844319775595, 0.5054945054945055, 0.5162162162162163, 0.5416116248348746, 0.5692108667529107, 0.5870393900889453, 0.6144278606965174, 0.6323529411764706, 0.6514423076923077, 0.6682408500590319, 0.6837806301050176, 0.6905311778290992, 0.6963470319634704, 0.7036199095022625, 0.7093153759820428, 0.7184035476718403] 
	t4 = [0, 4.128469228744507, 8.079154253005981, 12.064065933227539, 16.09325408935547, 20.02936100959778, 24.06037712097168, 28.075227737426758, 32.12510681152344, 36.042699337005615, 40.14673638343811, 44.011563539505005, 48.08779692649841, 52.05584406852722, 56.01351881027222, 60.04970407485962] 
	q4 = [0.36645962732919257, 0.40240240240240244, 0.4294117647058823, 0.45755395683453237, 0.4895688456189151, 0.5115646258503401, 0.5423280423280423, 0.5750962772785623, 0.6082603254067583, 0.6323529411764706, 0.6547192353643966, 0.6776084407971864, 0.6920415224913495, 0.6993166287015945, 0.7086614173228347, 0.7198228128460686] 
	t5 = [0, 5.063287734985352, 10.120372533798218, 15.07022738456726, 20.0629563331604, 25.031073331832886, 30.02994465827942, 35.034594774246216, 40.06280159950256, 45.027238845825195, 50.05190825462341, 55.08036422729492, 60.13243532180786] 
	q5 = [0.36645962732919257, 0.40956651718983555, 0.45217391304347826, 0.48179271708683474, 0.5115646258503401, 0.5549738219895288, 0.5913705583756346, 0.6273062730627307, 0.6539379474940334, 0.6837806301050176, 0.6964490263459335, 0.7065462753950339, 0.7190265486725663] 
	t6 = [0, 6.0282371044158936, 12.149386882781982, 18.013516902923584, 24.090599298477173, 30.07762837409973, 36.10650587081909, 42.08544659614563, 48.08410620689392, 54.00246047973633, 60.09515166282654] 
	q6 = [0.36645962732919257, 0.41604754829123325, 0.4597701149425287, 0.5054945054945055, 0.5442536327608982, 0.5913705583756346, 0.6308068459657702, 0.669811320754717, 0.6921296296296297, 0.7036199095022625, 0.7212389380530974] 
	t7 = [0, 7.104924201965332, 14.13019871711731, 21.003659963607788, 28.05250096321106, 35.00258278846741, 42.07862210273743, 49.0933141708374, 56.04063272476196] 
	q7 = [0.36645962732919257, 0.42011834319526625, 0.4787535410764873, 0.5162162162162163, 0.5794871794871795, 0.6280788177339902, 0.669811320754717, 0.6966551326412919, 0.7093153759820428] 
	t8 = [0, 8.0494384765625, 16.02113652229309, 24.053143739700317, 32.061474561691284, 40.0086088180542, 48.08625054359436, 56.07836937904358] 
	q8 = [0.36645962732919257, 0.4294117647058823, 0.4909344490934449, 0.546895640686922, 0.61, 0.6555423122765197, 0.6921296296296297, 0.7101123595505618] 
	t9 = [0, 9.091446161270142, 18.130075454711914, 27.13011932373047, 36.11638045310974, 45.12909412384033, 54.107309103012085] 
	q9 = [0.36645962732919257, 0.4408759124087591, 0.5061898211829436, 0.5710594315245477, 0.6332518337408313, 0.6853146853146853, 0.7036199095022625] 
	t10 = [0, 10.0839102268219, 20.074047803878784, 30.11776328086853, 40.10847806930542, 50.11785054206848, 60.06519794464111] 
	q10 = [0.36645962732919257, 0.45217391304347826, 0.5115646258503401, 0.5939086294416244, 0.6547619047619048, 0.695752009184845, 0.7220376522702104] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.111732006072998, 2.0540409088134766, 3.009596347808838, 4.110177278518677, 5.095278739929199, 6.025933027267456, 7.1214377880096436, 8.086282968521118, 9.078488826751709, 10.008196115493774, 11.135892629623413, 12.089895248413086, 13.033798217773438, 14.134340524673462, 15.077495336532593, 16.14575719833374, 17.094417333602905, 18.02178454399109, 19.142338514328003, 20.07369041442871, 21.058788537979126, 22.13124656677246, 23.130287170410156, 24.04973030090332, 25.041513919830322, 26.11517596244812, 27.008347034454346, 28.12648296356201, 29.11474323272705, 30.05662250518799, 31.02090358734131, 32.085676431655884, 33.029916286468506, 34.13520789146423, 35.07693004608154, 36.143659830093384, 37.08102989196777, 38.05128717422485, 39.14684796333313, 40.06751322746277, 41.010462284088135, 42.08274841308594, 43.067506313323975, 44.046725273132324, 45.02040243148804, 46.11469006538391, 47.088518142700195, 48.01489043235779, 49.00406813621521, 50.07902932167053, 51.076945543289185, 52.002851247787476, 53.12336230278015, 54.0501127243042, 55.1336715221405, 56.08935332298279, 57.10186195373535, 58.06578278541565, 59.01576805114746, 60.11576318740845] 
	q1 = [0.37694704049844235, 0.38639876352395675, 0.39263803680981596, 0.3981623277182236, 0.40364188163884673, 0.41327300150829566, 0.42514970059880236, 0.43684992570579495, 0.4460856720827178, 0.4493392070484582, 0.456140350877193, 0.46444121915820025, 0.4668587896253602, 0.47058823529411764, 0.4779516358463727, 0.48870056497175146, 0.49719101123595505, 0.5013927576601671, 0.5117565698478561, 0.5212620027434842, 0.5266030013642564, 0.5358592692828146, 0.5410497981157469, 0.5519999999999999, 0.5596816976127321, 0.570673712021136, 0.5774278215223096, 0.5796344647519581, 0.5844155844155845, 0.5909677419354838, 0.6, 0.604591836734694, 0.6134347275031685, 0.6204287515762926, 0.6299999999999999, 0.6377171215880894, 0.6428571428571429, 0.6519607843137254, 0.6536585365853659, 0.6610169491525424, 0.6690734055354994, 0.6738351254480287, 0.6809015421115066, 0.6863207547169812, 0.6948356807511737, 0.6994152046783625, 0.7038327526132403, 0.7113163972286374, 0.7149425287356321, 0.7191780821917808, 0.7206385404789054, 0.7212741751990899, 0.7235494880546074, 0.7264472190692395, 0.72686230248307, 0.7282976324689967, 0.7289088863892013, 0.7301231802911534, 0.7327394209354119, 0.7369589345172032, 0.7425414364640884] 
	t2 = [0, 2.13702654838562, 4.1344428062438965, 6.0372703075408936, 8.017540216445923, 10.048243522644043, 12.068020820617676, 14.115105390548706, 16.093766450881958, 18.070254802703857, 20.07160997390747, 22.07372808456421, 24.114463806152344, 26.140140771865845, 28.048720121383667, 30.054147243499756, 32.03888177871704, 34.05140709877014, 36.02678060531616, 38.03650212287903, 40.00381350517273, 42.13364839553833, 44.0545711517334, 46.132351875305176, 48.11448550224304, 50.03055429458618, 52.001322746276855, 54.14182925224304, 56.0037522315979, 58.06678557395935, 60.07226514816284] 
	q2 = [0.37694704049844235, 0.39263803680981596, 0.40364188163884673, 0.4275037369207773, 0.4460856720827178, 0.45839416058394167, 0.4697406340057637, 0.4822695035460992, 0.49859943977591037, 0.5138121546961326, 0.532608695652174, 0.5469168900804289, 0.5653896961690885, 0.5796344647519581, 0.5891472868217055, 0.604591836734694, 0.6204287515762926, 0.6377171215880894, 0.6519607843137254, 0.6610169491525424, 0.6754176610978521, 0.6878680800942285, 0.7009345794392523, 0.7104959630911187, 0.7200000000000001, 0.7212741751990899, 0.7278911564625851, 0.7266591676040495, 0.735195530726257, 0.738359201773836, 0.743109151047409] 
	t3 = [0, 3.077807664871216, 6.034039497375488, 9.107590198516846, 12.087350606918335, 15.032663106918335, 18.010827779769897, 21.07312297821045, 24.009502172470093, 27.10159420967102, 30.076831817626953, 33.07246804237366, 36.133437395095825, 39.03732466697693, 42.056344747543335, 45.03371000289917, 48.12085008621216, 51.09833383560181, 54.14247274398804, 57.09210205078125, 60.045010805130005] 
	q3 = [0.37694704049844235, 0.3969465648854962, 0.4275037369207773, 0.4516129032258065, 0.4697406340057637, 0.4908321579689704, 0.5138121546961326, 0.5378378378378378, 0.5653896961690885, 0.5844155844155845, 0.6063694267515924, 0.6317103620474407, 0.6528117359413202, 0.6722689075630252, 0.6894117647058823, 0.7083333333333334, 0.7214611872146119, 0.725, 0.7289088863892013, 0.7369589345172032, 0.7458745874587458] 
	t4 = [0, 4.027574300765991, 8.091228723526001, 12.083025932312012, 16.0591721534729, 20.096832275390625, 24.06049609184265, 28.134953498840332, 32.07884645462036, 36.144378423690796, 40.06410765647888, 44.08945631980896, 48.02746224403381, 52.119338512420654, 56.056594371795654, 60.09865760803223] 
	q4 = [0.37694704049844235, 0.40606060606060607, 0.4454277286135693, 0.4697406340057637, 0.5, 0.5338753387533874, 0.570673712021136, 0.5945945945945947, 0.6273525721455457, 0.6536585365853659, 0.6809015421115066, 0.7038327526132403, 0.7214611872146119, 0.729119638826185, 0.7363737486095662, 0.7499999999999999] 
	t5 = [0, 5.11507773399353, 10.054290771484375, 15.11628007888794, 20.122812747955322, 25.13736844062805, 30.1292884349823, 35.117300510406494, 40.10908555984497, 45.067524671554565, 50.0397424697876, 55.13328409194946, 60.063607931137085] 
	q5 = [0.37694704049844235, 0.42042042042042044, 0.462882096069869, 0.4943502824858757, 0.5338753387533874, 0.5800524934383202, 0.6099110546378652, 0.6486486486486487, 0.6809015421115066, 0.7098265895953758, 0.7243735763097949, 0.735195530726257, 0.7472527472527472] 
	t6 = [0, 6.100088357925415, 12.008552551269531, 18.12270212173462, 24.095973014831543, 30.13725709915161, 36.03084468841553, 42.06811547279358, 48.12669277191162, 54.114787578582764, 60.10802507400513] 
	q6 = [0.37694704049844235, 0.43219076005961254, 0.4697406340057637, 0.5171939477303988, 0.5733157199471598, 0.6116751269035533, 0.6536585365853659, 0.6948356807511737, 0.7229190421892817, 0.7295173961840627, 0.7486278814489573] 
	t7 = [0, 7.018277883529663, 14.06328010559082, 21.054181575775146, 28.056635856628418, 35.07434964179993, 42.13855767250061, 49.12484860420227, 56.113914012908936] 
	q7 = [0.37694704049844235, 0.4385185185185185, 0.48794326241134744, 0.5390835579514826, 0.5971685971685973, 0.6503067484662577, 0.6948356807511737, 0.7243735763097949, 0.7377777777777779] 
	t8 = [0, 8.058363437652588, 16.04835033416748, 24.043442249298096, 32.00349545478821, 40.03200602531433, 48.07910633087158, 56.07204341888428] 
	q8 = [0.37694704049844235, 0.4477172312223859, 0.5006993006993007, 0.5714285714285715, 0.6273525721455457, 0.6809015421115066, 0.7214611872146119, 0.7377777777777779] 
	t9 = [0, 9.085291385650635, 18.12288546562195, 27.011159658432007, 36.06889295578003, 45.031240701675415, 54.149090051651] 
	q9 = [0.37694704049844235, 0.4538799414348462, 0.5206611570247933, 0.5888456549935148, 0.6536585365853659, 0.7104959630911187, 0.7309417040358744] 
	t10 = [0, 10.10382080078125, 20.06842803955078, 30.12014412879944, 40.0424165725708, 50.12991690635681, 60.06808638572693] 
	q10 = [0.37694704049844235, 0.46444121915820025, 0.5353260869565217, 0.6149936467598475, 0.6809015421115066, 0.7258248009101251, 0.7494505494505495] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.094465732574463, 2.0263113975524902, 3.0085794925689697, 4.0882627964019775, 5.039682626724243, 6.1090216636657715, 7.072022438049316, 8.004124402999878, 9.100784063339233, 10.058046579360962, 11.03628945350647, 12.140800476074219, 13.138168573379517, 14.089877367019653, 15.096659421920776, 16.020294189453125, 17.126732349395752, 18.05276370048523, 19.037261486053467, 20.000524759292603, 21.006611108779907, 22.078212022781372, 23.053704023361206, 24.131324291229248, 25.083258390426636, 26.06491446495056, 27.110350370407104, 28.061400890350342, 29.014323711395264, 30.12614393234253, 31.070946216583252, 32.14673113822937, 33.12237215042114, 34.08597421646118, 35.0397162437439, 36.11349415779114, 37.09611630439758, 38.01977562904358, 39.106988191604614, 40.070231676101685, 41.05124831199646, 42.000521421432495, 43.01438331604004, 44.10611081123352, 45.051305532455444, 46.037617444992065, 47.129162549972534, 48.12314534187317, 49.09778642654419, 50.096620321273804, 51.03468465805054, 52.142568826675415, 53.094605445861816, 54.05004668235779, 55.03825783729553, 56.11950993537903, 57.12469506263733, 58.07978940010071, 59.034218549728394, 60.13368320465088] 
	q1 = [0.35104669887278583, 0.3578274760383387, 0.36190476190476195, 0.36850393700787404, 0.3837753510140406, 0.3869969040247678, 0.3981623277182236, 0.4079147640791477, 0.41452344931921337, 0.41867469879518066, 0.42514970059880236, 0.4338781575037147, 0.4460856720827179, 0.44868035190615835, 0.45547445255474456, 0.4602026049204052, 0.4668587896253602, 0.47428571428571437, 0.4801136363636364, 0.48587570621468923, 0.4943820224719101, 0.504881450488145, 0.5076282940360609, 0.517193947730399, 0.529331514324693, 0.5326086956521738, 0.5425101214574899, 0.5483870967741936, 0.5527369826435248, 0.5577689243027888, 0.5718050065876152, 0.579292267365662, 0.5867014341590613, 0.5950840879689522, 0.5971685971685972, 0.6025641025641025, 0.6096938775510204, 0.6142131979695431, 0.6194690265486726, 0.6231155778894473, 0.6307884856070087, 0.6343283582089552, 0.6386138613861386, 0.6437346437346437, 0.6536585365853659, 0.6585662211421629, 0.6626360338573155, 0.6674698795180724, 0.6746411483253588, 0.6793802145411204, 0.6856465005931199, 0.6872037914691943, 0.6903073286052009, 0.6949352179034158, 0.6933019976498238, 0.6940211019929661, 0.6923976608187135, 0.6915017462165309, 0.694541231126597, 0.697566628041715, 0.6997690531177829] 
	t2 = [0, 2.1280927658081055, 4.137872934341431, 6.131240129470825, 8.107521533966064, 10.096288681030273, 12.117680311203003, 14.024646520614624, 16.097209215164185, 18.06818175315857, 20.085444450378418, 22.123040437698364, 24.0083429813385, 26.01446294784546, 28.000694274902344, 30.01547932624817, 32.13578987121582, 34.12792468070984, 36.13577127456665, 38.131404399871826, 40.12653422355652, 42.01861596107483, 44.07557415962219, 46.10474371910095, 48.126832008361816, 50.02908205986023, 52.03127336502075, 54.06614804267883, 56.062957525253296, 58.100534200668335, 60.00689435005188] 
	q2 = [0.35104669887278583, 0.36450079239302696, 0.38317757009345793, 0.3981623277182236, 0.41452344931921337, 0.42985074626865677, 0.44477172312223856, 0.45481049562682213, 0.46839080459770116, 0.48158640226628896, 0.5006993006993007, 0.5131034482758622, 0.5306122448979592, 0.5444743935309972, 0.5539280958721704, 0.5774278215223096, 0.5958549222797928, 0.6025641025641025, 0.6142131979695431, 0.6231155778894473, 0.63681592039801, 0.6420664206642066, 0.6585662211421629, 0.6690734055354993, 0.6793802145411204, 0.6872037914691943, 0.6949352179034158, 0.6923976608187135, 0.6938300349243307, 0.6990740740740742, 0.7004608294930874] 
	t3 = [0, 3.0244221687316895, 6.063011169433594, 9.067220211029053, 12.103476762771606, 15.07009243965149, 18.035884141921997, 21.071895360946655, 24.00851821899414, 27.12488603591919, 30.09357261657715, 33.10118865966797, 36.034170389175415, 39.08879852294922, 42.001729011535645, 45.12495803833008, 48.05642223358154, 51.0242657661438, 54.024263858795166, 57.120705127716064, 60.088704109191895] 
	q3 = [0.35104669887278583, 0.36850393700787404, 0.3981623277182236, 0.4210526315789474, 0.4477172312223859, 0.46531791907514447, 0.48158640226628896, 0.5083333333333334, 0.5326086956521738, 0.553475935828877, 0.581151832460733, 0.5997425997425998, 0.6159695817490494, 0.6317103620474408, 0.6470588235294117, 0.6658624849215923, 0.6833333333333333, 0.6933962264150944, 0.6923976608187135, 0.6983758700696056, 0.7034482758620689] 
	t4 = [0, 4.04699182510376, 8.126071691513062, 12.031235694885254, 16.041598081588745, 20.015295028686523, 24.015971183776855, 28.0488338470459, 32.10017704963684, 36.05625081062317, 40.103055238723755, 44.10433006286621, 48.01648163795471, 52.046170711517334, 56.04439067840576, 60.036704778671265] 
	q4 = [0.35104669887278583, 0.3800623052959502, 0.41389728096676737, 0.4477172312223859, 0.47345767575322817, 0.5027932960893855, 0.5326086956521738, 0.5630810092961487, 0.5968992248062016, 0.6159695817490494, 0.6394052044609666, 0.6626360338573155, 0.6833333333333333, 0.6933019976498238, 0.6953488372093023, 0.7042577675489068] 
	t5 = [0, 5.062410831451416, 10.002973794937134, 15.038542985916138, 20.097474098205566, 25.145358085632324, 30.126606702804565, 35.03649830818176, 40.026992082595825, 45.10632801055908, 50.09710931777954, 55.02753186225891, 60.09917068481445] 
	q5 = [0.35104669887278583, 0.3919753086419753, 0.42985074626865677, 0.4659913169319827, 0.504881450488145, 0.5405405405405405, 0.581913499344692, 0.6132315521628499, 0.6394052044609666, 0.6674698795180724, 0.6887573964497041, 0.6938300349243307, 0.7042577675489068] 
	t6 = [0, 6.1309545040130615, 12.120427131652832, 18.1036274433136, 24.04248070716858, 30.016013145446777, 36.015692472457886, 42.03795146942139, 48.04333782196045, 54.1353645324707, 60.06390333175659] 
	q6 = [0.35104669887278583, 0.3981623277182236, 0.4477172312223859, 0.48441926345609054, 0.5345997286295794, 0.581913499344692, 0.6159695817490494, 0.6528117359413204, 0.6848989298454221, 0.6931155192532088, 0.7042577675489068] 
	t7 = [0, 7.009806394577026, 14.039416790008545, 21.104503870010376, 28.109842538833618, 35.05502700805664, 42.10902142524719, 49.127866983413696, 56.06065535545349] 
	q7 = [0.35104669887278583, 0.4079147640791477, 0.45997088791848617, 0.5076282940360609, 0.5630810092961487, 0.6157760814249363, 0.6544566544566545, 0.6880189798339266, 0.6968641114982579] 
	t8 = [0, 8.053046941757202, 16.099292755126953, 24.070423364639282, 32.04911947250366, 40.1374294757843, 48.09230351448059, 56.0658016204834] 
	q8 = [0.35104669887278583, 0.4114977307110439, 0.47632711621233864, 0.5326086956521738, 0.5976714100905564, 0.6386138613861386, 0.6848989298454221, 0.6968641114982579] 
	t9 = [0, 9.08794903755188, 18.124085664749146, 27.14286184310913, 36.04793930053711, 45.07295203208923, 54.04603457450867] 
	q9 = [0.35104669887278583, 0.4210526315789474, 0.48725212464589235, 0.5508021390374331, 0.6185044359949303, 0.6682750301568156, 0.691588785046729] 
	t10 = [0, 10.101624965667725, 20.0464346408844, 30.05961322784424, 40.124568700790405, 50.01358485221863, 60.05589246749878] 
	q10 = [0.35104669887278583, 0.43219076005961254, 0.5062937062937062, 0.581913499344692, 0.6386138613861386, 0.6911242603550296, 0.7057471264367816] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.0976812839508057, 2.022731304168701, 3.143202543258667, 4.0652854442596436, 5.016737699508667, 6.092763900756836, 7.040043115615845, 8.137825965881348, 9.115129232406616, 10.071775913238525, 11.014064073562622, 12.119301795959473, 13.065923929214478, 14.021507024765015, 15.143095016479492, 16.110904455184937, 17.062516450881958, 18.130756855010986, 19.073512077331543, 20.14120864868164, 21.11528706550598, 22.03804612159729, 23.14327120780945, 24.126070022583008, 25.12706756591797, 26.098607540130615, 27.10707926750183, 28.094767332077026, 29.043405771255493, 30.030181169509888, 31.035634994506836, 32.024232149124146, 33.000967502593994, 34.07464933395386, 35.026015758514404, 36.108393907547, 37.04845833778381, 38.001479387283325, 39.096651554107666, 40.05663752555847, 41.00530028343201, 42.078362464904785, 43.05864763259888, 44.02706742286682, 45.15581727027893, 46.08682417869568, 47.086899280548096, 48.01524353027344, 49.11236357688904, 50.03777861595154, 51.01551294326782, 52.04149532318115, 53.01022553443909, 54.120025396347046, 55.07568168640137, 56.03099703788757, 57.12489366531372, 58.0871102809906, 59.074249267578125, 60.036991596221924] 
	q1 = [0.34782608695652173, 0.35747303543913717, 0.3644716692189893, 0.3708206686930091, 0.37821482602117995, 0.3898050974512744, 0.3988095238095238, 0.4053254437869822, 0.4140969162995594, 0.42043795620437957, 0.42608695652173917, 0.43227665706051877, 0.4444444444444445, 0.45106382978723397, 0.4535211267605634, 0.4587412587412587, 0.4666666666666667, 0.4772413793103449, 0.48285322359396443, 0.4884038199181447, 0.49526387009472256, 0.4952893674293405, 0.5013404825737265, 0.511318242343542, 0.5218543046357615, 0.5243741765480895, 0.5340314136125656, 0.5416666666666666, 0.5510996119016818, 0.5592783505154638, 0.5659411011523687, 0.5728900255754477, 0.579415501905972, 0.5876418663303908, 0.5939849624060151, 0.6044776119402985, 0.6131025957972805, 0.6199261992619927, 0.6268292682926829, 0.6343825665859564, 0.6394230769230769, 0.6475507765830346, 0.6539833531510106, 0.6595744680851063, 0.6650998824911868, 0.6697782963827305, 0.6744186046511628, 0.6789838337182448, 0.6850574712643679, 0.6902857142857144, 0.69327251995439, 0.6977272727272728, 0.7028248587570621, 0.7072072072072071, 0.7115600448933782, 0.7144456886898097, 0.71731843575419, 0.7216035634743876, 0.7236403995560488, 0.7278761061946902, 0.725468577728776] 
	t2 = [0, 2.1214852333068848, 4.093465089797974, 6.103769779205322, 8.075255393981934, 10.111451864242554, 12.107527732849121, 14.122554540634155, 16.05121660232544, 18.02030920982361, 20.13490915298462, 22.13786268234253, 24.017449855804443, 26.073341131210327, 28.022058486938477, 30.06842064857483, 32.01020956039429, 34.007073163986206, 36.13250017166138, 38.00342535972595, 40.12632441520691, 42.13124203681946, 44.01606225967407, 46.026917695999146, 48.0637104511261, 50.04529547691345, 52.141674518585205, 54.03055143356323, 56.03434777259827, 58.04462647438049, 60.07271862030029] 
	q2 = [0.34782608695652173, 0.36391437308868496, 0.377643504531722, 0.40118870728083206, 0.4164222873900293, 0.4312590448625181, 0.4472934472934473, 0.45569620253164556, 0.4687933425797504, 0.4821917808219178, 0.4966261808367071, 0.5053475935828877, 0.5238095238095238, 0.5378590078328982, 0.5548387096774194, 0.5728900255754477, 0.5858585858585859, 0.6027397260273972, 0.6199261992619927, 0.6343825665859564, 0.649164677804296, 0.6611570247933884, 0.6697782963827305, 0.6789838337182448, 0.6917808219178082, 0.6992054483541429, 0.7086614173228347, 0.7158836689038031, 0.7216035634743876, 0.72707182320442, 0.7260726072607261] 
	t3 = [0, 3.01993465423584, 6.0241539478302, 9.064539194107056, 12.026251077651978, 15.071160793304443, 18.01972985267639, 21.04291582107544, 24.060818433761597, 27.095415115356445, 30.071070909500122, 33.10256576538086, 36.0158588886261, 39.06734371185303, 42.09215426445007, 45.020148277282715, 48.00302767753601, 51.11703872680664, 54.14499378204346, 57.03033947944641, 60.005964040756226] 
	q3 = [0.34782608695652173, 0.3732928679817906, 0.40118870728083206, 0.4250363901018923, 0.4472934472934473, 0.46582984658298465, 0.4821917808219178, 0.49932885906040264, 0.523117569352708, 0.5510996119016818, 0.5728900255754477, 0.5957446808510638, 0.6216216216216216, 0.642685851318945, 0.6643109540636042, 0.6743916570104287, 0.6917808219178082, 0.7036199095022624, 0.71731843575419, 0.7278761061946902, 0.7260726072607261] 
	t4 = [0, 4.065378665924072, 8.00470232963562, 12.128142356872559, 16.065332412719727, 20.0320827960968, 24.116058826446533, 28.011955499649048, 32.094332695007324, 36.145864725112915, 40.083399534225464, 44.05436182022095, 48.01028060913086, 52.130191802978516, 56.05590510368347, 60.04714751243591] 
	q4 = [0.34782608695652173, 0.38009049773755654, 0.4164222873900293, 0.4472934472934473, 0.47368421052631576, 0.4959568733153638, 0.525065963060686, 0.5585585585585585, 0.5919395465994962, 0.625, 0.6507747318235995, 0.6744186046511628, 0.69327251995439, 0.7080045095828637, 0.7236403995560488, 0.7260726072607261] 
	t5 = [0, 5.075068235397339, 10.13399624824524, 15.065855979919434, 20.119234323501587, 25.106404781341553, 30.08947253227234, 35.11366844177246, 40.067827463150024, 45.10824799537659, 50.0185911655426, 55.136189222335815, 60.10468053817749] 
	q5 = [0.34782608695652173, 0.39162929745889385, 0.4335260115606937, 0.4651810584958217, 0.4959568733153638, 0.5373525557011797, 0.576530612244898, 0.6165228113440197, 0.6531585220500595, 0.6789838337182448, 0.7006802721088435, 0.7216035634743876, 0.7268722466960352] 
	t6 = [0, 6.1422600746154785, 12.132311820983887, 18.085434198379517, 24.005265951156616, 30.02324080467224, 36.146008014678955, 42.04682683944702, 48.14171385765076, 54.055763483047485, 60.046095848083496] 
	q6 = [0.34782608695652173, 0.400593471810089, 0.4472934472934473, 0.48700410396716826, 0.525065963060686, 0.576530612244898, 0.6266829865361077, 0.6658823529411765, 0.69327251995439, 0.7166853303471444, 0.7268722466960352] 
	t7 = [0, 7.029880523681641, 14.092094421386719, 21.143258810043335, 28.057887315750122, 35.08368539810181, 42.13649272918701, 49.09513545036316, 56.09516358375549] 
	q7 = [0.34782608695652173, 0.4076809453471197, 0.46132208157524623, 0.5013404825737265, 0.5637065637065638, 0.6172839506172839, 0.6650998824911868, 0.6992054483541429, 0.7250554323725054] 
	t8 = [0, 8.037404775619507, 16.006443738937378, 24.08418107032776, 32.167306900024414, 40.08230185508728, 48.082916498184204, 56.01757287979126] 
	q8 = [0.34782608695652173, 0.4158125915080527, 0.477115117891817, 0.5270092226613966, 0.592964824120603, 0.6531585220500595, 0.69327251995439, 0.725860155382908] 
	t9 = [0, 9.106136560440063, 18.009929180145264, 27.119649410247803, 36.050872802734375, 45.02659773826599, 54.10005569458008] 
	q9 = [0.34782608695652173, 0.4250363901018923, 0.4876712328767123, 0.55627425614489, 0.6266829865361077, 0.6789838337182448, 0.7181208053691275] 
	t10 = [0, 10.137340068817139, 20.106638431549072, 30.057682991027832, 40.044904470443726, 50.00606369972229, 60.11399054527283] 
	q10 = [0.34782608695652173, 0.43290043290043295, 0.4959568733153638, 0.578005115089514, 0.6531585220500595, 0.7021517553793886, 0.7282728272827284] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.1138641834259033, 2.0380332469940186, 3.130195140838623, 4.097579717636108, 5.079263925552368, 6.012392044067383, 7.122800827026367, 8.150636672973633, 9.03694772720337, 10.000097513198853, 11.129636287689209, 12.070924758911133, 13.020346879959106, 14.012717962265015, 15.003509044647217, 16.091875076293945, 17.052412033081055, 18.124932050704956, 19.09995698928833, 20.025551319122314, 21.122864246368408, 22.0851149559021, 23.074989557266235, 24.040486812591553, 25.095537662506104, 26.029088973999023, 27.00832509994507, 28.037664890289307, 29.02405858039856, 30.129234790802002, 31.075700759887695, 32.03019571304321, 33.13155674934387, 34.06425380706787, 35.020331621170044, 36.09846544265747, 37.08231520652771, 38.06845211982727, 39.02423596382141, 40.10674715042114, 41.108957052230835, 42.02915930747986, 43.145711183547974, 44.071587562561035, 45.020918130874634, 46.03841853141785, 47.02767729759216, 48.12577414512634, 49.07183027267456, 50.03271746635437, 51.01138162612915, 52.09278678894043, 53.071288108825684, 54.00050640106201, 55.12897610664368, 56.05230093002319, 57.02279806137085, 58.089330434799194, 59.06661057472229, 60.032607555389404] 
	q1 = [0.37617554858934166, 0.3862928348909657, 0.3931888544891641, 0.4000000000000001, 0.4042879019908117, 0.4127465857359635, 0.4223227752639517, 0.431784107946027, 0.4411326378539493, 0.44674556213017746, 0.4552129221732746, 0.4635568513119534, 0.46888567293777134, 0.4697406340057637, 0.47564469914040114, 0.4801136363636363, 0.4908321579689704, 0.49579831932773116, 0.5027777777777779, 0.5082872928176796, 0.5116918844566712, 0.5211459754433834, 0.5311653116531165, 0.534412955465587, 0.5403225806451613, 0.5500667556742322, 0.557029177718833, 0.570673712021136, 0.5781865965834428, 0.5830065359477125, 0.5914396887159533, 0.5979381443298969, 0.6051282051282052, 0.6132315521628499, 0.6212121212121213, 0.6256281407035176, 0.6334164588528678, 0.6385093167701863, 0.6427688504326329, 0.6511056511056511, 0.6585067319461445, 0.6634146341463414, 0.6690909090909091, 0.6778846153846154, 0.6842105263157895, 0.6865315852205006, 0.6887573964497041, 0.6988235294117647, 0.702576112412178, 0.7079439252336449, 0.710128055878929, 0.7137891077636153, 0.7152777777777778, 0.7174163783160323, 0.7188940092165897, 0.7218390804597701, 0.7262313860252004, 0.7237442922374429, 0.7258248009101251, 0.7256235827664399, 0.727683615819209] 
	t2 = [0, 2.1312530040740967, 4.001946926116943, 6.019499778747559, 8.011498928070068, 10.089561223983765, 12.058786153793335, 14.064106225967407, 16.10086989402771, 18.070881128311157, 20.071370124816895, 22.043593406677246, 24.10584044456482, 26.027830600738525, 28.118648767471313, 30.00965189933777, 32.02215814590454, 34.00218749046326, 36.12162518501282, 38.03956484794617, 40.056365728378296, 42.06614875793457, 44.06532025337219, 46.14266872406006, 48.010520219802856, 50.06203055381775, 52.07627749443054, 54.07681059837341, 56.085203647613525, 58.082834005355835, 60.124544620513916] 
	q2 = [0.37617554858934166, 0.39009287925696595, 0.4042879019908117, 0.42469879518072295, 0.4434523809523809, 0.4545454545454545, 0.46753246753246747, 0.47863247863247854, 0.4964936886395512, 0.5089903181189488, 0.5191256830601092, 0.534412955465587, 0.5500667556742322, 0.5687830687830688, 0.5848563968668408, 0.5979381443298969, 0.6132315521628499, 0.6273525721455457, 0.6385093167701863, 0.6535626535626536, 0.6642335766423357, 0.6794717887154862, 0.6872770511296077, 0.7018779342723006, 0.710955710955711, 0.7152777777777778, 0.7188940092165897, 0.7233065442020666, 0.7258248009101251, 0.7285067873303167, 0.7328072153325816] 
	t3 = [0, 3.0280635356903076, 6.056605577468872, 9.095627784729004, 12.071170091629028, 15.119091033935547, 18.046999216079712, 21.09674334526062, 24.01699161529541, 27.051212072372437, 30.13049578666687, 33.00156259536743, 36.013317346572876, 39.11115288734436, 42.025060415267944, 45.05809283256531, 48.04843831062317, 51.12119174003601, 54.156821727752686, 57.04525589942932, 60.1300892829895] 
	q3 = [0.37617554858934166, 0.3969230769230769, 0.4270676691729323, 0.45132743362831856, 0.4697406340057637, 0.4887005649717514, 0.5089903181189488, 0.5311653116531165, 0.5527369826435247, 0.5797101449275363, 0.5997425997425997, 0.624685138539043, 0.640198511166253, 0.6617826617826618, 0.6826347305389222, 0.6972909305064783, 0.710955710955711, 0.7174163783160323, 0.725400457665904, 0.7270668176670442, 0.7342342342342342] 
	t4 = [0, 4.069843292236328, 8.001073122024536, 12.013843059539795, 16.1048424243927, 20.064512968063354, 24.029237508773804, 28.118285655975342, 32.03703999519348, 36.107566118240356, 40.112783670425415, 44.09555196762085, 48.102057218551636, 52.1080687046051, 56.055463790893555, 60.040846824645996] 
	q4 = [0.37617554858934166, 0.40672782874617736, 0.4427934621099554, 0.4697406340057637, 0.49579831932773116, 0.5211459754433834, 0.5527369826435247, 0.5900783289817233, 0.6159695817490494, 0.6410891089108911, 0.6690909090909091, 0.6887573964497041, 0.7124563445867288, 0.720368239355581, 0.7264472190692395, 0.735658042744657] 
	t5 = [0, 5.099141836166382, 10.102050542831421, 15.121031522750854, 20.14081358909607, 25.00423288345337, 30.012704849243164, 35.05651926994324, 40.09401750564575, 45.13507795333862, 50.07413649559021, 55.11108613014221, 60.0166437625885] 
	q5 = [0.37617554858934166, 0.41515151515151516, 0.4597364568081991, 0.4915254237288136, 0.5231607629427792, 0.56158940397351, 0.6038709677419356, 0.6385093167701863, 0.6707021791767555, 0.6988235294117647, 0.7161066048667439, 0.7258248009101251, 0.7379077615298087] 
	t6 = [0, 6.098129749298096, 12.142407894134521, 18.007213354110718, 24.100626707077026, 30.128790616989136, 36.10999608039856, 42.02890872955322, 48.11532258987427, 54.019392251968384, 60.06450009346008] 
	q6 = [0.37617554858934166, 0.4270676691729323, 0.47041847041847046, 0.5082872928176796, 0.5546666666666666, 0.6056701030927836, 0.6427688504326329, 0.6833930704898448, 0.710955710955711, 0.7245714285714285, 0.7379077615298087] 
	t7 = [0, 7.030995845794678, 14.136730194091797, 21.12796688079834, 28.068866729736328, 35.07735085487366, 42.0537314414978, 49.00497913360596, 56.118295431137085] 
	q7 = [0.37617554858934166, 0.4341317365269461, 0.48295454545454547, 0.530446549391069, 0.5908496732026144, 0.6385093167701863, 0.684964200477327, 0.7131242740998838, 0.7256235827664399] 
	t8 = [0, 8.025344371795654, 16.092523097991943, 24.089942693710327, 32.068989753723145, 40.000372648239136, 48.07531952857971, 56.06777286529541] 
	q8 = [0.37617554858934166, 0.4427934621099554, 0.49859943977591037, 0.5565912117177098, 0.6235741444866921, 0.6723095525997582, 0.7124563445867288, 0.7256235827664399] 
	t9 = [0, 9.072764873504639, 18.12762761116028, 27.0376193523407, 36.10818028450012, 45.00552272796631, 54.025389671325684] 
	q9 = [0.37617554858934166, 0.45360824742268047, 0.5082872928176796, 0.5815789473684211, 0.6427688504326329, 0.6988235294117647, 0.7239404352806414] 
	t10 = [0, 10.135797262191772, 20.07272505760193, 30.00538969039917, 40.02286100387573, 50.020885705947876, 60.068278312683105] 
	q10 = [0.37617554858934166, 0.4619883040935672, 0.5258855585831064, 0.6056701030927836, 0.6739130434782609, 0.7152777777777778, 0.738496071829405] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.097196102142334, 2.022329092025757, 3.1148998737335205, 4.05005955696106, 5.011842250823975, 6.082965612411499, 7.024119138717651, 8.124852895736694, 9.072354793548584, 10.03190803527832, 11.133162498474121, 12.067205429077148, 13.015037298202515, 14.020829200744629, 15.111647129058838, 16.037176609039307, 17.000354766845703, 18.08002281188965, 19.055608987808228, 20.123398065567017, 21.083953619003296, 22.009601831436157, 23.024182319641113, 24.10521650314331, 25.139465808868408, 26.131520748138428, 27.135499238967896, 28.05672526359558, 29.026901721954346, 30.121910572052002, 31.067631244659424, 32.02129411697388, 33.112988233566284, 34.07165503501892, 35.01579308509827, 36.11867427825928, 37.102083921432495, 38.02965784072876, 39.1226646900177, 40.07625770568848, 41.02479910850525, 42.01642084121704, 43.107876777648926, 44.069793462753296, 45.05361580848694, 46.01192355155945, 47.11571955680847, 48.0702018737793, 49.02152967453003, 50.011735916137695, 51.108304023742676, 52.06402039527893, 53.038421630859375, 54.144160985946655, 55.08818292617798, 56.04669189453125, 57.139322996139526, 58.07072377204895, 59.08492684364319, 60.012048959732056] 
	q1 = [0.3630769230769231, 0.37251908396946565, 0.37936267071320184, 0.3855421686746988, 0.3916292974588939, 0.39821693907875183, 0.406480117820324, 0.4181286549707602, 0.42608695652173906, 0.43578643578643583, 0.44412607449856734, 0.4488636363636364, 0.4535211267605634, 0.4600280504908836, 0.4673157162726008, 0.47790055248618785, 0.48626373626373626, 0.4918032786885246, 0.4993215739484396, 0.5094339622641509, 0.518716577540107, 0.5285524568393095, 0.5389696169088507, 0.5504587155963303, 0.556135770234987, 0.5673575129533679, 0.5710594315245479, 0.576923076923077, 0.5798212005108557, 0.5877862595419847, 0.5952080706179067, 0.6015037593984962, 0.6119402985074628, 0.6163366336633663, 0.6257668711656442, 0.6275946275946277, 0.6318347509113001, 0.6384522370012092, 0.6409638554216868, 0.6434573829531812, 0.6507747318235997, 0.653206650831354, 0.657210401891253, 0.6619718309859155, 0.6682242990654205, 0.6736353077816493, 0.6789838337182448, 0.6865671641791045, 0.6909920182440137, 0.6939704209328782, 0.6946651532349603, 0.6976217440543602, 0.6990950226244345, 0.7042889390519187, 0.7040358744394617, 0.7054871220604704, 0.7098214285714286, 0.7119021134593992, 0.7147613762486126, 0.7168141592920354, 0.7210584343991181] 
	t2 = [0, 2.1252405643463135, 4.100848436355591, 6.077343463897705, 8.068849325180054, 10.07423210144043, 12.08484959602356, 14.093456506729126, 16.101609230041504, 18.089890718460083, 20.09753441810608, 22.060423135757446, 24.10635995864868, 26.045998096466064, 28.109291315078735, 30.13111186027527, 32.12455940246582, 34.120627880096436, 36.131030321121216, 38.143468141555786, 40.134361743927, 42.029518365859985, 44.02036952972412, 46.03888177871704, 48.05144715309143, 50.079527139663696, 52.122045040130615, 54.14456367492676, 56.000900745391846, 58.12388324737549, 60.00145769119263] 
	q2 = [0.3630769230769231, 0.3787878787878788, 0.3916292974588939, 0.40882352941176475, 0.42608695652173906, 0.44571428571428573, 0.4556962025316456, 0.47222222222222227, 0.48834019204389567, 0.503382949932341, 0.5226666666666667, 0.5466491458607096, 0.5617685305591678, 0.5758354755784062, 0.5877862595419847, 0.5997490589711417, 0.6163366336633663, 0.6275946275946277, 0.6384522370012092, 0.6434573829531812, 0.653206650831354, 0.6619718309859155, 0.6751740139211136, 0.6880733944954128, 0.6939704209328782, 0.6976217440543602, 0.7049549549549549, 0.7083798882681563, 0.7119021134593992, 0.7182320441988951, 0.723076923076923] 
	t3 = [0, 3.035460948944092, 6.030137062072754, 9.018876791000366, 12.083927869796753, 15.125515460968018, 18.00448751449585, 21.050411701202393, 24.10017967224121, 27.004648685455322, 30.09871244430542, 33.14282488822937, 36.07195854187012, 39.11585974693298, 42.05152177810669, 45.06480383872986, 48.024455070495605, 51.080620765686035, 54.04455018043518, 57.094693422317505, 60.1411988735199] 
	q3 = [0.3630769230769231, 0.3855421686746988, 0.40882352941176475, 0.437410071942446, 0.4556962025316456, 0.4848484848484848, 0.503382949932341, 0.537037037037037, 0.5636363636363636, 0.5805626598465473, 0.6040100250626567, 0.6282208588957056, 0.6400966183574881, 0.6523809523809524, 0.6666666666666666, 0.6835443037974682, 0.6954545454545454, 0.7042889390519187, 0.7098214285714286, 0.7153931339977853, 0.7244785949506037] 
	t4 = [0, 4.04047966003418, 8.100138902664185, 12.072512865066528, 16.004514694213867, 20.089962005615234, 24.032937049865723, 28.12493133544922, 32.14277911186218, 36.08337712287903, 40.060601472854614, 44.02926540374756, 48.00304317474365, 52.16368365287781, 56.11098384857178, 60.08417797088623] 
	q4 = [0.3630769230769231, 0.3916292974588939, 0.4289855072463768, 0.4556962025316456, 0.49108367626886146, 0.5246338215712384, 0.5636363636363636, 0.5877862595419847, 0.619753086419753, 0.6417370325693607, 0.6563981042654029, 0.6782407407407406, 0.6962457337883959, 0.7048260381593715, 0.7147613762486126, 0.7244785949506037] 
	t5 = [0, 5.091423749923706, 10.03400444984436, 15.017385721206665, 20.00395154953003, 25.13224768638611, 30.060240745544434, 35.07656764984131, 40.1022789478302, 45.13351011276245, 50.05141830444336, 55.115553855895996, 60.11063575744629] 
	q5 = [0.3630769230769231, 0.40296296296296297, 0.44571428571428573, 0.4848484848484848, 0.5246338215712384, 0.5684754521963824, 0.6057571964956195, 0.6359223300970874, 0.655621301775148, 0.6865671641791045, 0.6990950226244345, 0.7126948775055679, 0.7258771929824561] 
	t6 = [0, 6.1411755084991455, 12.111478328704834, 18.03827977180481, 24.023260831832886, 30.085563898086548, 36.00496578216553, 42.07619595527649, 48.05122947692871, 54.00123715400696, 60.09445309638977] 
	q6 = [0.3630769230769231, 0.4111600587371513, 0.4556962025316456, 0.5054054054054054, 0.5636363636363636, 0.6082603254067586, 0.6441495778045839, 0.6682242990654205, 0.6954545454545454, 0.7112597547380156, 0.7258771929824561] 
	t7 = [0, 7.036041498184204, 14.075426816940308, 21.01758885383606, 28.127086877822876, 35.079171657562256, 42.12902641296387, 49.11196851730347, 56.12738347053528] 
	q7 = [0.3630769230769231, 0.41982507288629733, 0.4743411927877948, 0.537037037037037, 0.5877862595419847, 0.6375757575757576, 0.6682242990654205, 0.6984126984126984, 0.7169811320754718] 
	t8 = [0, 8.045032262802124, 16.008735179901123, 24.09971022605896, 32.073882818222046, 40.12342190742493, 48.051974058151245, 56.126595973968506] 
	q8 = [0.3630769230769231, 0.4289855072463768, 0.49041095890410963, 0.5617685305591678, 0.6246913580246913, 0.657210401891253, 0.6969353007945516, 0.7169811320754718] 
	t9 = [0, 9.10850477218628, 18.1184344291687, 27.028462886810303, 36.09555625915527, 45.141925573349, 54.01051092147827] 
	q9 = [0.3630769230769231, 0.437410071942446, 0.508108108108108, 0.5831202046035805, 0.6441495778045839, 0.6872852233676976, 0.7112597547380156] 
	t10 = [0, 10.144946098327637, 20.06795334815979, 30.110475301742554, 40.072598695755005, 50.06623554229736, 60.01343274116516] 
	q10 = [0.3630769230769231, 0.4450784593437946, 0.5272969374167776, 0.6082603254067586, 0.6579881656804735, 0.7006802721088435, 0.7258771929824561] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.102147102355957, 2.026860237121582, 3.126110553741455, 4.0542893409729, 5.012324571609497, 6.094643831253052, 7.0495686531066895, 8.128922462463379, 9.104789733886719, 10.096739768981934, 11.04269289970398, 12.115821838378906, 13.02171516418457, 14.09221887588501, 15.041043996810913, 16.140684366226196, 17.1157968044281, 18.04026985168457, 19.12572431564331, 20.113125562667847, 21.090335369110107, 22.048487901687622, 23.028244256973267, 24.0943021774292, 25.095622301101685, 26.08089852333069, 27.025846481323242, 28.014349699020386, 29.107850074768066, 30.034731149673462, 31.045907974243164, 32.00403571128845, 33.133880376815796, 34.092859506607056, 35.032591819763184, 36.13468360900879, 37.08379030227661, 38.00591993331909, 39.09456491470337, 40.05419111251831, 41.01280117034912, 42.00748062133789, 43.01753377914429, 44.09010362625122, 45.061519622802734, 46.011436462402344, 47.11884021759033, 48.0483078956604, 49.14489817619324, 50.132590532302856, 51.08686876296997, 52.0110068321228, 53.14532279968262, 54.10403823852539, 55.0527184009552, 56.12460732460022, 57.12618970870972, 58.047460079193115, 59.13808822631836, 60.096633434295654] 
	q1 = [0.3563402889245586, 0.3688394276629571, 0.37914691943127965, 0.3836477987421384, 0.39313572542901715, 0.4031007751937984, 0.41230769230769226, 0.4213740458015268, 0.4279210925644917, 0.43504531722054385, 0.4347826086956522, 0.44576523031203563, 0.4503703703703704, 0.45227606461086645, 0.4635568513119534, 0.4680232558139535, 0.47330447330447334, 0.48068669527896996, 0.4864864864864865, 0.49152542372881364, 0.5021037868162693, 0.5076708507670852, 0.5172890733056708, 0.52400548696845, 0.5300546448087431, 0.5447154471544716, 0.5537634408602151, 0.5622489959839356, 0.5683930942895087, 0.5770750988142292, 0.5842105263157896, 0.5942408376963352, 0.5945241199478487, 0.6036269430051814, 0.6118251928020566, 0.615581098339719, 0.6243654822335025, 0.6279949558638083, 0.6314465408805031, 0.64, 0.6442786069651741, 0.6493184634448576, 0.6576354679802955, 0.6658506731946144, 0.6691086691086692, 0.6747572815533982, 0.6803377563329313, 0.6835138387484958, 0.6913875598086124, 0.6952380952380953, 0.6975088967971531, 0.7005917159763314, 0.7021276595744681, 0.7051886792452831, 0.7089201877934272, 0.7072599531615925, 0.7126168224299065, 0.7147846332945286, 0.7169373549883991, 0.7190751445086705, 0.7220299884659745] 
	t2 = [0, 2.1244049072265625, 4.09131932258606, 6.069143533706665, 8.057498693466187, 10.08558440208435, 12.095053672790527, 14.040898084640503, 16.01480221748352, 18.061940670013428, 20.05306363105774, 22.09302592277527, 24.125102281570435, 26.042352199554443, 28.112648010253906, 30.09538960456848, 32.02397179603577, 34.05580735206604, 36.0743408203125, 38.05797505378723, 40.05713677406311, 42.08039307594299, 44.11313819885254, 46.00201439857483, 48.12603163719177, 50.04587149620056, 52.05002164840698, 54.05085802078247, 56.04077219963074, 58.06239604949951, 60.058653831481934] 
	q2 = [0.3563402889245586, 0.37914691943127965, 0.3956386292834891, 0.4147465437788019, 0.4279210925644917, 0.43712574850299407, 0.4526627218934911, 0.4635568513119534, 0.47619047619047616, 0.4879432624113475, 0.5083798882681564, 0.5206611570247933, 0.5387755102040815, 0.5583892617449664, 0.5759577278731837, 0.5923984272608126, 0.6033810143042914, 0.6163682864450128, 0.6262626262626263, 0.64, 0.6493184634448576, 0.667481662591687, 0.6763636363636363, 0.6835138387484958, 0.6952380952380953, 0.7005917159763314, 0.7067137809187278, 0.711111111111111, 0.7162790697674418, 0.7220299884659745, 0.7241379310344828] 
	t3 = [0, 3.044874906539917, 6.0492753982543945, 9.043349504470825, 12.142759799957275, 15.023291826248169, 18.04384994506836, 21.111170768737793, 24.0769464969635, 27.103217363357544, 30.166226625442505, 33.07996129989624, 36.02122783660889, 39.08811712265015, 42.08465576171875, 45.119582414627075, 48.026434659957886, 51.05115509033203, 54.08905863761902, 57.05806064605713, 60.053725719451904] 
	q3 = [0.3563402889245586, 0.3836477987421384, 0.4147465437788019, 0.43373493975903615, 0.45199409158050224, 0.47024673439767783, 0.49078014184397173, 0.518005540166205, 0.5387755102040815, 0.5691489361702127, 0.5942408376963352, 0.6143958868894601, 0.6279949558638083, 0.6459627329192548, 0.667481662591687, 0.6819277108433734, 0.6967895362663495, 0.7036599763872491, 0.7126168224299065, 0.7175925925925928, 0.7233065442020665] 
	t4 = [0, 4.074731349945068, 8.11109972000122, 12.0523841381073, 16.11091375350952, 20.08427095413208, 24.085190773010254, 28.144587993621826, 32.01080584526062, 36.042418003082275, 40.135952949523926, 44.01150631904602, 48.138705253601074, 52.131598472595215, 56.05581307411194, 60.14145565032959] 
	q4 = [0.3563402889245586, 0.3956386292834891, 0.43030303030303035, 0.4526627218934911, 0.4776978417266186, 0.5076708507670852, 0.5434782608695652, 0.5804749340369393, 0.6070038910505837, 0.6279949558638083, 0.651851851851852, 0.6771463119709795, 0.6975088967971531, 0.7089201877934272, 0.7169373549883991, 0.7233065442020665] 
	t5 = [0, 5.096698999404907, 10.057219505310059, 15.06477689743042, 20.038793325424194, 25.0942542552948, 30.117836475372314, 35.034220933914185, 40.06381440162659, 45.13260531425476, 50.007598876953125, 55.03224802017212, 60.09026050567627] 
	q5 = [0.3563402889245586, 0.4024767801857585, 0.4417910447761194, 0.47383720930232565, 0.5076708507670852, 0.5513513513513514, 0.5942408376963352, 0.6286438529784537, 0.651851851851852, 0.6835138387484958, 0.7021276595744681, 0.7162790697674418, 0.7233065442020665] 
	t6 = [0, 6.122634410858154, 12.123998165130615, 18.11178755760193, 24.053311824798584, 30.111255407333374, 36.06766200065613, 42.102887868881226, 48.122185468673706, 54.02820611000061, 60.1159086227417] 
	q6 = [0.3563402889245586, 0.4171779141104295, 0.4542772861356933, 0.49291784702549574, 0.5434782608695652, 0.5968586387434556, 0.6305170239596469, 0.667481662591687, 0.6975088967971531, 0.7126168224299065, 0.724770642201835] 
	t7 = [0, 7.02576470375061, 14.022400140762329, 21.030001878738403, 28.118833541870117, 35.03116536140442, 42.04196333885193, 49.06503248214722, 56.14851641654968] 
	q7 = [0.3563402889245586, 0.42378048780487804, 0.4700729927007299, 0.518005540166205, 0.582010582010582, 0.6286438529784537, 0.667481662591687, 0.7005917159763314, 0.7175925925925928] 
	t8 = [0, 8.060060739517212, 16.00044846534729, 24.148401737213135, 32.0048406124115, 40.13556408882141, 48.034701347351074, 56.04485082626343] 
	q8 = [0.3563402889245586, 0.43030303030303035, 0.481962481962482, 0.5434782608695652, 0.6113989637305699, 0.655980271270037, 0.6975088967971531, 0.7175925925925928] 
	t9 = [0, 9.095786809921265, 18.08463764190674, 27.16002607345581, 36.09102487564087, 45.01347517967224, 54.025593996047974] 
	q9 = [0.3563402889245586, 0.43308270676691735, 0.4943181818181818, 0.574468085106383, 0.6313131313131314, 0.6835138387484958, 0.7117852975495915] 
	t10 = [0, 10.097745895385742, 20.083782196044922, 30.03555917739868, 40.139747619628906, 50.028400897979736, 60.13423418998718] 
	q10 = [0.3563402889245586, 0.444113263785395, 0.5146853146853148, 0.5931758530183727, 0.6600985221674877, 0.7021276595744681, 0.7262313860252004] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.106161117553711, 2.041574001312256, 3.130689859390259, 4.061336517333984, 5.015340805053711, 6.089142799377441, 7.038455963134766, 8.110889434814453, 9.060950517654419, 10.05217456817627, 11.06612491607666, 12.005393981933594, 13.129602670669556, 14.083700895309448, 15.037629842758179, 16.114276885986328, 17.09664559364319, 18.08433723449707, 19.03763699531555, 20.023579835891724, 21.010253190994263, 22.087918519973755, 23.06469750404358, 24.02039361000061, 25.114712238311768, 26.135636568069458, 27.063472747802734, 28.048314332962036, 29.021892547607422, 30.089534521102905, 31.032411813735962, 32.13587546348572, 33.09555244445801, 34.01784348487854, 35.10680961608887, 36.0604944229126, 37.03734564781189, 38.14916753768921, 39.09720587730408, 40.06001353263855, 41.07053565979004, 42.009459495544434, 43.00130891799927, 44.07763886451721, 45.088666915893555, 46.010825634002686, 47.13494372367859, 48.122910499572754, 49.071046352386475, 50.02315831184387, 51.14164924621582, 52.07299304008484, 53.020588636398315, 54.12973093986511, 55.076911211013794, 56.03643321990967, 57.074236154556274, 58.00260376930237, 59.129244804382324, 60.12090039253235] 
	q1 = [0.3508771929824561, 0.36075949367088606, 0.36792452830188677, 0.378125, 0.3875968992248062, 0.39445300462249616, 0.40916030534351144, 0.4109589041095891, 0.4205748865355522, 0.42706766917293243, 0.4334828101644246, 0.4437869822485207, 0.4552129221732746, 0.462882096069869, 0.4704184704184704, 0.479196556671449, 0.4857142857142857, 0.4907801418439717, 0.49647390691114246, 0.4985915492957746, 0.5049088359046282, 0.5153203342618385, 0.5200553250345782, 0.5274725274725275, 0.5395095367847411, 0.5508819538670285, 0.5587044534412955, 0.5606469002695418, 0.570281124497992, 0.5771276595744681, 0.5820105820105821, 0.5876152832674573, 0.5968586387434556, 0.5997392438070405, 0.6062176165803109, 0.6143958868894601, 0.617948717948718, 0.6257982120051085, 0.6319796954314721, 0.638888888888889, 0.6457286432160803, 0.6499999999999999, 0.654228855721393, 0.6617283950617284, 0.6699386503067485, 0.6731946144430844, 0.6772228989037758, 0.6812121212121214, 0.6859903381642513, 0.6891566265060242, 0.6907340553549941, 0.6946107784431138, 0.6977299880525687, 0.698450536352801, 0.6998813760379596, 0.7021276595744681, 0.7021276595744681, 0.7058823529411764, 0.7080890973036342, 0.7072599531615925, 0.7117852975495917] 
	t2 = [0, 2.1337735652923584, 4.120574235916138, 6.097674131393433, 8.095181226730347, 10.071941375732422, 12.025575399398804, 14.032455444335938, 16.04077386856079, 18.046571731567383, 20.076629877090454, 22.134722471237183, 24.015944004058838, 26.134218454360962, 28.021512985229492, 30.069171667099, 32.05283570289612, 34.015743017196655, 36.0187509059906, 38.060590744018555, 40.06318950653076, 42.08872723579407, 44.09150576591492, 46.1205792427063, 48.02008128166199, 50.028095960617065, 52.0411958694458, 54.04839038848877, 56.05842590332031, 58.01862835884094, 60.04942464828491] 
	q2 = [0.3508771929824561, 0.37048665620094196, 0.3869969040247678, 0.40916030534351144, 0.42296072507552873, 0.43815201192250375, 0.4574780058651026, 0.4733044733044733, 0.49002849002848997, 0.4936530324400564, 0.5132496513249651, 0.5234159779614326, 0.5469387755102041, 0.5606469002695418, 0.576, 0.5857519788918205, 0.597911227154047, 0.6108247422680413, 0.6257982120051085, 0.6371681415929203, 0.6499999999999999, 0.6633785450061652, 0.6723716381418092, 0.6828087167070218, 0.6891566265060242, 0.6946107784431138, 0.7, 0.7021276595744681, 0.7058823529411764, 0.7072599531615925, 0.7177700348432055] 
	t3 = [0, 3.03108286857605, 6.043783664703369, 9.045530796051025, 12.087451696395874, 15.103045225143433, 18.143128395080566, 21.07740330696106, 24.058784008026123, 27.065943479537964, 30.13758134841919, 33.143959283828735, 36.01783275604248, 39.12882375717163, 42.055989027023315, 45.01876354217529, 48.03656077384949, 51.104888677597046, 54.04873991012573, 57.02463459968567, 60.1381516456604] 
	q3 = [0.3508771929824561, 0.378125, 0.40916030534351144, 0.43413173652694614, 0.4604105571847507, 0.48137535816618904, 0.49577464788732395, 0.5180055401662049, 0.5489130434782609, 0.5721925133689839, 0.5876152832674573, 0.6098191214470284, 0.6292993630573249, 0.6466165413533834, 0.6650246305418719, 0.6812121212121214, 0.6891566265060242, 0.698450536352801, 0.7021276595744681, 0.7072599531615925, 0.7192575406032483] 
	t4 = [0, 4.037462472915649, 8.07499384880066, 12.018518447875977, 16.06703209877014, 20.023467540740967, 24.07206153869629, 28.10285472869873, 32.08688831329346, 36.02943253517151, 40.01304221153259, 44.02487373352051, 48.148022174835205, 52.02765941619873, 56.09780263900757, 60.14418077468872] 
	q4 = [0.3508771929824561, 0.3869969040247678, 0.42296072507552873, 0.4604105571847507, 0.492176386913229, 0.5132496513249651, 0.5508819538670285, 0.5797872340425532, 0.6015625, 0.6292993630573249, 0.6533665835411471, 0.675609756097561, 0.6907340553549941, 0.6998813760379596, 0.7065727699530516, 0.7177700348432055] 
	t5 = [0, 5.065036773681641, 10.004190921783447, 15.041524887084961, 20.106062650680542, 25.14052724838257, 30.00446319580078, 35.10369324684143, 40.14169096946716, 45.05695819854736, 50.0652437210083, 55.09424543380737, 60.038342237472534] 
	q5 = [0.3508771929824561, 0.39938556067588327, 0.4398216939078752, 0.48493543758967, 0.5132496513249651, 0.5575101488497971, 0.5883905013192612, 0.6205128205128205, 0.6533665835411471, 0.6812121212121214, 0.6961722488038278, 0.7058823529411764, 0.7177700348432055] 
	t6 = [0, 6.097728490829468, 12.073375940322876, 18.012506008148193, 24.14640474319458, 30.042481184005737, 36.05923295021057, 42.13750433921814, 48.020532846450806, 54.144569635391235, 60.092238903045654] 
	q6 = [0.3508771929824561, 0.40916030534351144, 0.46198830409356717, 0.49647390691114246, 0.5528455284552846, 0.5939553219448095, 0.6310432569974554, 0.6699386503067485, 0.6907340553549941, 0.705188679245283, 0.7192575406032483] 
	t7 = [0, 7.011035680770874, 14.03197717666626, 21.048449754714966, 28.14438772201538, 35.063064098358154, 42.003331422805786, 49.1047465801239, 56.0024254322052] 
	q7 = [0.3508771929824561, 0.4157814871016692, 0.4812680115273776, 0.5187239944521498, 0.5805592543275633, 0.6205128205128205, 0.6650246305418719, 0.6923076923076923, 0.7067137809187277] 
	t8 = [0, 8.026752948760986, 16.101381301879883, 24.095938205718994, 32.147852659225464, 40.11970901489258, 48.10551571846008, 56.11998152732849] 
	q8 = [0.3508771929824561, 0.42296072507552873, 0.49358059914407987, 0.55359565807327, 0.6059817945383615, 0.654228855721393, 0.6907340553549941, 0.7065727699530516] 
	t9 = [0, 9.108036756515503, 18.03778386116028, 27.078929662704468, 36.01093602180481, 45.088767766952515, 54.012919187545776] 
	q9 = [0.3508771929824561, 0.43413173652694614, 0.4978783592644979, 0.5748663101604277, 0.6310432569974554, 0.6828087167070218, 0.705188679245283] 
	t10 = [0, 10.017174243927002, 20.092089891433716, 30.06848978996277, 40.14373207092285, 50.0165798664093, 60.08505868911743] 
	q10 = [0.3508771929824561, 0.4398216939078752, 0.5146853146853148, 0.5947368421052632, 0.6559006211180124, 0.6953405017921148, 0.7186046511627907] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.1131680011749268, 2.081968307495117, 3.044325351715088, 4.130446195602417, 5.091939449310303, 6.026788234710693, 7.13524055480957, 8.127904653549194, 9.003080606460571, 10.115276575088501, 11.071208715438843, 12.014889001846313, 13.000268459320068, 14.113631248474121, 15.12148666381836, 16.088942766189575, 17.039878129959106, 18.00766396522522, 19.002767086029053, 20.079989433288574, 21.062535524368286, 22.137298583984375, 23.095422983169556, 24.044930934906006, 25.141643524169922, 26.07441234588623, 27.11812448501587, 28.017154216766357, 29.003698110580444, 30.117691040039062, 31.134000301361084, 32.07374691963196, 33.05577206611633, 34.13040375709534, 35.0797438621521, 36.038947105407715, 37.13652849197388, 38.10695719718933, 39.05296540260315, 40.13206696510315, 41.1234450340271, 42.052645206451416, 43.031299352645874, 44.127753496170044, 45.101879596710205, 46.05184459686279, 47.13158130645752, 48.055177211761475, 49.04847264289856, 50.114792346954346, 51.068862438201904, 52.025211811065674, 53.131267786026, 54.100544929504395, 55.044121980667114, 56.01380395889282, 57.135034799575806, 58.07572650909424, 59.027575731277466, 60.10999321937561] 
	q1 = [0.3443708609271523, 0.35255354200988465, 0.3562091503267974, 0.366288492706645, 0.37620578778135044, 0.3859649122807018, 0.39873417721518983, 0.4025157232704403, 0.4043887147335423, 0.41368584758942456, 0.4283513097072419, 0.437308868501529, 0.4407294832826748, 0.44478063540090773, 0.44879518072289165, 0.45739910313901344, 0.4629080118694362, 0.4690265486725664, 0.47368421052631576, 0.4760522496371553, 0.47976878612716767, 0.4878048780487805, 0.4978662873399715, 0.5014164305949008, 0.5112359550561798, 0.5257301808066759, 0.5318559556786704, 0.543956043956044, 0.5499316005471956, 0.5597826086956522, 0.5675675675675675, 0.576043068640646, 0.5836680053547523, 0.5922974767596282, 0.5978835978835979, 0.605263157894737, 0.6117647058823529, 0.6173800259403373, 0.6245161290322581, 0.6290115532734275, 0.6360153256704981, 0.6404066073697585, 0.6463878326996199, 0.649056603773585, 0.6550000000000001, 0.6592039800995025, 0.6600496277915632, 0.665024630541872, 0.6731946144430844, 0.6780487804878048, 0.6820388349514563, 0.6884057971014492, 0.6899879372738239, 0.6915662650602409, 0.6931407942238267, 0.6954436450839329, 0.6970059880239521, 0.6977299880525686, 0.6992840095465394, 0.703923900118906, 0.7069988137603795] 
	t2 = [0, 2.1363673210144043, 4.006890535354614, 6.141085386276245, 8.149322509765625, 10.098950147628784, 12.101900100708008, 14.135984897613525, 16.038990259170532, 18.090503454208374, 20.09657573699951, 22.091712474822998, 24.07362151145935, 26.064329147338867, 28.12151312828064, 30.03212833404541, 32.06897044181824, 34.06677293777466, 36.07188630104065, 38.0839958190918, 40.04870057106018, 42.046655893325806, 44.07250714302063, 46.13230633735657, 48.0008659362793, 50.1449658870697, 52.145432472229004, 54.01702284812927, 56.01152276992798, 58.019492626190186, 60.05024528503418] 
	q2 = [0.3443708609271523, 0.3588907014681892, 0.37620578778135044, 0.40126382306477093, 0.40937500000000004, 0.43076923076923074, 0.44309559939301973, 0.45045045045045046, 0.46814814814814815, 0.47743813682678304, 0.48345323741007196, 0.49929078014184397, 0.5216178521617851, 0.5399449035812672, 0.5617367706919946, 0.576043068640646, 0.5922974767596282, 0.605263157894737, 0.6191709844559585, 0.6307692307692309, 0.6429479034307497, 0.650753768844221, 0.6592039800995025, 0.6715686274509803, 0.6804374240583232, 0.6899879372738239, 0.6931407942238267, 0.6946107784431138, 0.7008343265792609, 0.7069988137603795, 0.7068557919621749] 
	t3 = [0, 3.0232043266296387, 6.079526662826538, 9.060011863708496, 12.090225458145142, 15.032531261444092, 18.136489391326904, 21.078742265701294, 24.100038766860962, 27.151359796524048, 30.004623889923096, 33.097328901290894, 36.02083158493042, 39.092851877212524, 42.102081298828125, 45.047688007354736, 48.00118350982666, 51.065979957580566, 54.1130211353302, 57.04249906539917, 60.11865854263306] 
	q3 = [0.3443708609271523, 0.366288492706645, 0.40126382306477093, 0.42170542635658914, 0.44309559939301973, 0.4649776453055142, 0.47674418604651164, 0.4978662873399715, 0.5257301808066759, 0.5491803278688524, 0.5771812080536913, 0.5997357992073976, 0.6191709844559585, 0.638676844783715, 0.655819774718398, 0.6633785450061652, 0.6820388349514563, 0.6915662650602409, 0.6977299880525686, 0.7054631828978624, 0.705188679245283] 
	t4 = [0, 4.036273956298828, 8.058112859725952, 12.015103816986084, 16.01531410217285, 20.04654812812805, 24.03914523124695, 28.112401008605957, 32.07137084007263, 36.04985237121582, 40.00992250442505, 44.11551809310913, 48.066184997558594, 52.13998293876648, 56.002018451690674, 60.056047439575195] 
	q4 = [0.3443708609271523, 0.3756019261637239, 0.40937500000000004, 0.44309559939301973, 0.46814814814814815, 0.48563218390804597, 0.5236768802228412, 0.5578231292517006, 0.5941644562334217, 0.6199740596627757, 0.6446700507614213, 0.6600496277915632, 0.6820388349514563, 0.6931407942238267, 0.7023809523809524, 0.706021251475797] 
	t5 = [0, 5.088858604431152, 10.053570985794067, 15.011133432388306, 20.100388050079346, 25.119269609451294, 30.102707862854004, 35.11375546455383, 40.113196849823, 45.11367583274841, 50.079392194747925, 55.107120752334595, 60.12010645866394] 
	q5 = [0.3443708609271523, 0.3885350318471338, 0.4362519201228878, 0.4649776453055142, 0.49067431850789095, 0.5331491712707183, 0.5817694369973191, 0.6163849154746424, 0.6481012658227848, 0.6683046683046683, 0.6915662650602409, 0.6992840095465394, 0.706021251475797] 
	t6 = [0, 6.135974884033203, 12.000401496887207, 18.02518582344055, 24.00853204727173, 30.01848840713501, 36.13330626487732, 42.028191566467285, 48.109615325927734, 54.11171317100525, 60.05997371673584] 
	q6 = [0.3443708609271523, 0.40126382306477093, 0.4461305007587253, 0.478134110787172, 0.5277777777777778, 0.5779569892473119, 0.6253229974160207, 0.6541822721598002, 0.6852300242130751, 0.6977299880525686, 0.706021251475797] 
	t7 = [0, 7.058560371398926, 14.0061616897583, 21.110791206359863, 28.005306243896484, 35.06766867637634, 42.05753827095032, 49.04044532775879, 56.047449350357056] 
	q7 = [0.3443708609271523, 0.39937106918238996, 0.45645645645645644, 0.5014245014245015, 0.5597826086956522, 0.6163849154746424, 0.6541822721598002, 0.6884057971014492, 0.703923900118906] 
	t8 = [0, 8.058050394058228, 16.11241364479065, 24.067901134490967, 32.027310371398926, 40.01008343696594, 48.05183219909668, 56.0083429813385] 
	q8 = [0.3443708609271523, 0.40937500000000004, 0.47337278106508873, 0.5277777777777778, 0.5960264900662251, 0.6481012658227848, 0.6852300242130751, 0.7054631828978624] 
	t9 = [0, 9.117651224136353, 18.067994832992554, 27.07478356361389, 36.09255409240723, 45.14388942718506, 54.00503158569336] 
	q9 = [0.3443708609271523, 0.4241486068111455, 0.47883211678832116, 0.5538881309686221, 0.6253229974160207, 0.6699386503067486, 0.6977299880525686] 
	t10 = [0, 10.142205238342285, 20.09775710105896, 30.018369436264038, 40.11847424507141, 50.05768394470215, 60.14109683036804] 
	q10 = [0.3443708609271523, 0.4355828220858896, 0.49712643678160917, 0.5779569892473119, 0.6481012658227848, 0.6899879372738239, 0.7075471698113207] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.1010448932647705, 2.047410726547241, 3.0023083686828613, 4.1383562088012695, 5.098572731018066, 6.029729127883911, 7.132946252822876, 8.077301263809204, 9.026063203811646, 10.1049222946167, 11.005573511123657, 12.104968309402466, 13.091115713119507, 14.015803575515747, 15.003775119781494, 16.087137937545776, 17.042934894561768, 18.043201446533203, 19.056795120239258, 20.13051676750183, 21.004130125045776, 22.08757734298706, 23.065107345581055, 24.043614864349365, 25.145179271697998, 26.114416360855103, 27.081247091293335, 28.084733486175537, 29.094794988632202, 30.022883892059326, 31.127846240997314, 32.09704923629761, 33.07796096801758, 34.04410099983215, 35.00170278549194, 36.10288596153259, 37.086241722106934, 38.01927089691162, 39.11022639274597, 40.03222894668579, 41.121365785598755, 42.08177375793457, 43.055617809295654, 44.13862061500549, 45.03065037727356, 46.1030592918396, 47.118611097335815, 48.076128244400024, 49.03431582450867, 50.1022686958313, 51.12954068183899, 52.09244990348816, 53.04214954376221, 54.12418866157532, 55.106329917907715, 56.06636071205139, 57.062668800354004, 58.0265474319458, 59.13364100456238, 60.06458926200867] 
	q1 = [0.37403400309119017, 0.38343558282208584, 0.3926940639269407, 0.3969696969696969, 0.4012066365007541, 0.4095665171898356, 0.42136498516320475, 0.4247787610619469, 0.4363103953147877, 0.44250363901018924, 0.4486251808972504, 0.4546762589928058, 0.4679029957203995, 0.4752475247524752, 0.47605633802816905, 0.4797768479776848, 0.4854368932038835, 0.49103448275862066, 0.49589041095890407, 0.5027173913043478, 0.5060893098782139, 0.5148247978436657, 0.5301204819277108, 0.536, 0.5411140583554377, 0.5526315789473684, 0.5602094240837696, 0.5677083333333334, 0.574385510996119, 0.5817245817245819, 0.5856777493606139, 0.5928753180661577, 0.6017699115044247, 0.6045340050377834, 0.6132665832290363, 0.616729088639201, 0.6195786864931846, 0.6273062730627306, 0.6323529411764706, 0.6399026763990268, 0.6440677966101694, 0.6450060168471721, 0.6531100478468901, 0.6587395957193818, 0.6650887573964497, 0.6737089201877934, 0.6814469078179696, 0.6875725900116144, 0.6929316338354579, 0.697459584295612, 0.7019562715765246, 0.706422018348624, 0.7123287671232877, 0.7137970353477765, 0.7181818181818183, 0.7188208616780044, 0.7186440677966102, 0.7192784667418263, 0.7207207207207207, 0.7248322147651006, 0.7262569832402234] 
	t2 = [0, 2.1288700103759766, 4.141336917877197, 6.133907079696655, 8.094048261642456, 10.072678804397583, 12.126893997192383, 14.010316848754883, 16.021150827407837, 18.029217958450317, 20.09649085998535, 22.140910625457764, 24.02909803390503, 26.03221893310547, 28.040793657302856, 30.112870454788208, 32.12216877937317, 34.01262021064758, 36.01882362365723, 38.019166231155396, 40.13928437232971, 42.00225639343262, 44.1344838142395, 46.05499768257141, 48.09057354927063, 50.06366324424744, 52.13050198554993, 54.12344932556152, 56.13980150222778, 58.033385276794434, 60.000871896743774] 
	q2 = [0.37403400309119017, 0.3896499238964992, 0.40361445783132527, 0.4237037037037037, 0.44087591240875906, 0.4508670520231214, 0.4694167852062589, 0.47685834502103785, 0.4868603042876903, 0.4986376021798365, 0.5148247978436657, 0.5340453938584779, 0.5488126649076517, 0.5677083333333334, 0.5798969072164948, 0.5928753180661577, 0.6062893081761006, 0.616729088639201, 0.6273062730627306, 0.6391251518833536, 0.6482593037214885, 0.6619217081850534, 0.6791569086651054, 0.6890951276102089, 0.7019562715765246, 0.7079037800687286, 0.7152619589977219, 0.7186440677966102, 0.7207207207207207, 0.7248322147651006, 0.7296996662958843] 
	t3 = [0, 3.0224881172180176, 6.06633186340332, 9.049225807189941, 12.112856149673462, 15.042391538619995, 18.089457511901855, 21.096078872680664, 24.037396907806396, 27.111366271972656, 30.104167222976685, 33.01316046714783, 36.09318399429321, 39.033714056015015, 42.0438232421875, 45.1417977809906, 48.08783459663391, 51.021470069885254, 54.0593695640564, 57.103535652160645, 60.02041506767273] 
	q3 = [0.37403400309119017, 0.3969696969696969, 0.4237037037037037, 0.4447674418604651, 0.47159090909090906, 0.4861111111111111, 0.5013623978201635, 0.5301204819277108, 0.5526315789473684, 0.5751295336787564, 0.5972045743329097, 0.6132665832290363, 0.6289926289926291, 0.644927536231884, 0.6635071090047394, 0.6883720930232559, 0.7019562715765246, 0.7137970353477765, 0.7186440677966102, 0.7248322147651006, 0.7339246119733924] 
	t4 = [0, 4.0394158363342285, 8.128147840499878, 12.040130138397217, 16.06835103034973, 20.098246812820435, 24.07271671295166, 28.08804225921631, 32.1143593788147, 36.11296367645264, 40.098270654678345, 44.02955341339111, 48.07508993148804, 52.08301377296448, 56.056835889816284, 60.01524782180786] 
	q4 = [0.37403400309119017, 0.4006024096385542, 0.44087591240875906, 0.47226173541963024, 0.49239280774550476, 0.5148247978436657, 0.5526315789473684, 0.5817245817245819, 0.6097867001254705, 0.6297662976629765, 0.651497005988024, 0.6807017543859649, 0.7019562715765246, 0.7167235494880546, 0.7227833894500562, 0.7339246119733924] 
	t5 = [0, 5.110252141952515, 10.086124897003174, 15.113806247711182, 20.134462594985962, 25.01404571533203, 30.023170232772827, 35.10960602760315, 40.034074544906616, 45.03104639053345, 50.12840437889099, 55.041611671447754, 60.130385398864746] 
	q5 = [0.37403400309119017, 0.4119402985074627, 0.4508670520231214, 0.4867872044506259, 0.5182186234817813, 0.5620915032679739, 0.5946632782719187, 0.626387176325524, 0.651497005988024, 0.6883720930232559, 0.710857142857143, 0.7207207207207207, 0.7367256637168144] 
	t6 = [0, 6.111771821975708, 12.026349544525146, 18.127495765686035, 24.08974528312683, 30.071245193481445, 36.05458188056946, 42.12602686882019, 48.09903073310852, 54.077861070632935, 60.01031279563904] 
	q6 = [0.37403400309119017, 0.4237037037037037, 0.47226173541963024, 0.5020463847203275, 0.5545335085413929, 0.5964467005076143, 0.6331288343558283, 0.6666666666666667, 0.7049368541905855, 0.7178329571106095, 0.7367256637168144] 
	t7 = [0, 7.028509616851807, 14.151544332504272, 21.00577735900879, 28.085854291915894, 35.136972188949585, 42.14608550071716, 49.01156210899353, 56.06867432594299] 
	q7 = [0.37403400309119017, 0.4270986745213549, 0.48179271708683474, 0.5288590604026846, 0.5817245817245819, 0.626387176325524, 0.6650887573964497, 0.7079037800687286, 0.7242152466367713] 
	t8 = [0, 8.018670082092285, 16.006091117858887, 24.163590669631958, 32.08022952079773, 40.0339412689209, 48.11566972732544, 56.126102924346924] 
	q8 = [0.37403400309119017, 0.44087591240875906, 0.4930747922437673, 0.5526315789473684, 0.6115288220551378, 0.6538922155688623, 0.7042577675489068, 0.7256438969764837] 
	t9 = [0, 9.12815237045288, 18.081674337387085, 27.09061598777771, 36.039167404174805, 45.096312284469604, 54.12681317329407] 
	q9 = [0.37403400309119017, 0.444121915820029, 0.5027322404371585, 0.5788113695090439, 0.6331288343558283, 0.6883720930232559, 0.7178329571106095] 
	t10 = [0, 10.031468152999878, 20.064070224761963, 30.025954008102417, 40.05621647834778, 50.05963063240051, 60.08484721183777] 
	q10 = [0.37403400309119017, 0.4486251808972504, 0.516914749661705, 0.5946632782719187, 0.6538922155688623, 0.7101947308132875, 0.7367256637168144] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.101938247680664, 2.0239853858947754, 3.1155686378479004, 4.040267705917358, 5.138702869415283, 6.0744948387146, 7.0322980880737305, 8.134954690933228, 9.140938758850098, 10.06505274772644, 11.020910739898682, 12.098358869552612, 13.05238676071167, 14.123389959335327, 15.064783096313477, 16.031972885131836, 17.135899543762207, 18.123639822006226, 19.132036924362183, 20.061027765274048, 21.040717124938965, 22.028293132781982, 23.130792140960693, 24.093733549118042, 25.102888584136963, 26.12629461288452, 27.089402437210083, 28.011473894119263, 29.127923011779785, 30.047308683395386, 31.12977385520935, 32.057119846343994, 33.00165891647339, 34.069151878356934, 35.00842595100403, 36.10526251792908, 37.06686544418335, 38.001787185668945, 39.11982798576355, 40.042449712753296, 41.08730721473694, 42.00753092765808, 43.101284980773926, 44.06340503692627, 45.01974081993103, 46.0118510723114, 47.12735104560852, 48.03529095649719, 49.00753903388977, 50.08384299278259, 51.08598184585571, 52.0047881603241, 53.11735701560974, 54.07131505012512, 55.070379972457886, 56.02020215988159, 57.00270438194275, 58.10942840576172, 59.086233377456665, 60.00375056266785] 
	q1 = [0.37519872813990457, 0.3848580441640379, 0.39184952978056425, 0.40372670807453415, 0.41294298921417566, 0.42266462480857586, 0.4244274809160305, 0.4370257966616085, 0.4457831325301205, 0.45603576751117736, 0.46449704142011833, 0.46989720998531564, 0.4832605531295488, 0.489855072463768, 0.4956772334293948, 0.49785407725321884, 0.5014245014245015, 0.5070821529745042, 0.5126760563380282, 0.5230769230769231, 0.5271966527196652, 0.5325936199722608, 0.5379310344827587, 0.5479452054794521, 0.5538881309686221, 0.5578231292517007, 0.5660377358490567, 0.5710455764075066, 0.5748663101604279, 0.5827814569536424, 0.5902503293807642, 0.599476439790576, 0.6067708333333333, 0.610608020698577, 0.617948717948718, 0.6232439335887611, 0.6294416243654822, 0.6330390920554856, 0.6356783919597989, 0.6384039900249378, 0.6434782608695652, 0.6535141800246608, 0.660122699386503, 0.6682926829268293, 0.6699147381242387, 0.6763636363636363, 0.6843373493975904, 0.6875, 0.6906474820143885, 0.6945107398568019, 0.6960667461263408, 0.6976190476190476, 0.6998813760379597, 0.7028301886792452, 0.7043580683156654, 0.7080890973036342, 0.707943925233645, 0.7124563445867288, 0.7146171693735498, 0.7190751445086706, 0.7174163783160323] 
	t2 = [0, 2.1381983757019043, 4.123997926712036, 6.10398530960083, 8.07624340057373, 10.142881870269775, 12.12722659111023, 14.090037822723389, 16.06607723236084, 18.09518313407898, 20.032270908355713, 22.089327573776245, 24.133957862854004, 26.079107522964478, 28.086363554000854, 30.10582423210144, 32.109551191329956, 34.09653663635254, 36.076819896698, 38.09244966506958, 40.0985472202301, 42.02136993408203, 44.028021574020386, 46.04331684112549, 48.12906885147095, 50.01741814613342, 52.022167921066284, 54.047659397125244, 56.110639810562134, 58.13707423210144, 60.12678003311157] 
	q2 = [0.37519872813990457, 0.3912363067292645, 0.4153846153846154, 0.42682926829268286, 0.4481203007518797, 0.4660766961651917, 0.48546511627906974, 0.49712643678160917, 0.5028409090909092, 0.5189340813464236, 0.5292479108635098, 0.544704264099037, 0.5578231292517007, 0.5698924731182795, 0.5797872340425531, 0.5976408912188729, 0.6088082901554404, 0.6232439335887611, 0.6313131313131314, 0.6392009987515606, 0.6518518518518518, 0.6666666666666666, 0.6763636363636363, 0.6875, 0.6929510155316607, 0.6991676575505351, 0.7028301886792452, 0.7096018735362999, 0.713953488372093, 0.7182448036951502, 0.7262313860252004] 
	t3 = [0, 3.035891532897949, 6.025666952133179, 9.028151750564575, 12.002849102020264, 15.022248029708862, 18.059723138809204, 21.03236675262451, 24.00170373916626, 27.070401430130005, 30.10902428627014, 33.1250102519989, 36.000237464904785, 39.06951403617859, 42.07342076301575, 45.12553906440735, 48.13673663139343, 51.160090923309326, 54.06191349029541, 57.04451584815979, 60.02617907524109] 
	q3 = [0.37519872813990457, 0.40372670807453415, 0.42682926829268286, 0.4567164179104477, 0.48546511627906974, 0.5021398002853067, 0.5210084033613445, 0.5366528354080221, 0.5578231292517007, 0.5748663101604279, 0.599476439790576, 0.619718309859155, 0.6347607052896725, 0.6468401486988848, 0.6682926829268293, 0.6843373493975904, 0.6913875598086124, 0.7012987012987013, 0.7087719298245615, 0.7175925925925926, 0.7276887871853547] 
	t4 = [0, 4.02997088432312, 8.130019187927246, 12.135754108428955, 16.090012788772583, 20.04225182533264, 24.10250163078308, 28.055654764175415, 32.01403284072876, 36.066054821014404, 40.13443326950073, 44.0116331577301, 48.14356088638306, 52.09885048866272, 56.071598291397095, 60.109200954437256] 
	q4 = [0.37519872813990457, 0.41294298921417566, 0.4481203007518797, 0.4876632801161103, 0.5028409090909092, 0.5292479108635098, 0.5578231292517007, 0.5816733067729084, 0.610608020698577, 0.6347607052896725, 0.6568265682656828, 0.6779661016949152, 0.6929510155316607, 0.7043580683156654, 0.7162790697674419, 0.7276887871853547] 
	t5 = [0, 5.088334321975708, 10.01634168624878, 15.059004068374634, 20.037113904953003, 25.05411696434021, 30.0395450592041, 35.01818323135376, 40.00049614906311, 45.07085204124451, 50.07010531425476, 55.023261070251465, 60.13237953186035] 
	q5 = [0.37519872813990457, 0.42266462480857586, 0.46971935007385524, 0.5014245014245015, 0.5333333333333333, 0.5668016194331984, 0.600262123197903, 0.6286438529784538, 0.6568265682656828, 0.6843373493975904, 0.697508896797153, 0.7147846332945286, 0.7291428571428571] 
	t6 = [0, 6.121213674545288, 12.091781616210938, 18.10853147506714, 24.05995750427246, 30.043653964996338, 36.010000705718994, 42.007421255111694, 48.067615270614624, 54.02027106285095, 60.03602313995361] 
	q6 = [0.37519872813990457, 0.42987804878048785, 0.4890829694323144, 0.5230769230769231, 0.5597826086956521, 0.600262123197903, 0.6339622641509434, 0.6682926829268293, 0.6913875598086124, 0.7102803738317757, 0.7291428571428571] 
	t7 = [0, 7.144281387329102, 14.08816933631897, 21.03427004814148, 28.13923144340515, 35.049208879470825, 42.07224249839783, 49.0912446975708, 56.0911340713501] 
	q7 = [0.37519872813990457, 0.43939393939393934, 0.4992826398852224, 0.5406896551724137, 0.5880794701986755, 0.6278481012658228, 0.6682926829268293, 0.6960667461263408, 0.7184241019698726] 
	t8 = [0, 8.03703761100769, 16.127163648605347, 24.103349208831787, 32.04409837722778, 40.10138177871704, 48.00355553627014, 56.05518698692322] 
	q8 = [0.37519872813990457, 0.4481203007518797, 0.504964539007092, 0.5617367706919946, 0.6159793814432991, 0.6584766584766585, 0.6913875598086124, 0.7184241019698726] 
	t9 = [0, 9.086281776428223, 18.097144603729248, 27.026808738708496, 36.00250959396362, 45.11552810668945, 54.02472639083862] 
	q9 = [0.37519872813990457, 0.45901639344262296, 0.5230769230769231, 0.5813333333333334, 0.6339622641509434, 0.6859205776173286, 0.7102803738317757] 
	t10 = [0, 10.111899375915527, 20.057449340820312, 30.04202437400818, 40.045703172683716, 50.00696301460266, 60.132972955703735] 
	q10 = [0.37519872813990457, 0.4667651403249631, 0.5333333333333333, 0.6028833551769333, 0.6584766584766585, 0.697508896797153, 0.7308132875143184] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.0930488109588623, 2.0570836067199707, 3.0131263732910156, 4.088042736053467, 5.046553134918213, 6.130492687225342, 7.084993362426758, 8.01526141166687, 9.107081651687622, 10.06040072441101, 11.133731126785278, 12.060111045837402, 13.013535499572754, 14.086800336837769, 15.062686443328857, 16.140974521636963, 17.01014471054077, 18.095457315444946, 19.06942629814148, 20.02385926246643, 21.032758235931396, 22.13722825050354, 23.082924365997314, 24.020068645477295, 25.006637573242188, 26.124789476394653, 27.078633308410645, 28.00889801979065, 29.13191056251526, 30.087984323501587, 31.039066553115845, 32.14558148384094, 33.12651801109314, 34.11893677711487, 35.09168076515198, 36.030848026275635, 37.12612056732178, 38.07809281349182, 39.01931285858154, 40.09470200538635, 41.03460907936096, 42.12979698181152, 43.1320538520813, 44.05354952812195, 45.094924449920654, 46.025657415390015, 47.03536105155945, 48.10782718658447, 49.11588954925537, 50.07689428329468, 51.05577874183655, 52.01220417022705, 53.14121341705322, 54.092413663864136, 55.06313467025757, 56.140331745147705, 57.08850145339966, 58.01693058013916, 59.143205404281616, 60.10206985473633] 
	q1 = [0.35499207606973054, 0.35962145110410093, 0.3667711598746081, 0.37325038880248834, 0.3858024691358025, 0.3950995405819296, 0.4018264840182648, 0.40785498489425975, 0.41141141141141147, 0.4149253731343284, 0.4172876304023844, 0.42540620384047273, 0.4294117647058824, 0.43795620437956206, 0.44927536231884063, 0.45533141210374634, 0.4655172413793104, 0.4714285714285715, 0.4759206798866855, 0.4795486600846262, 0.4859550561797753, 0.496513249651325, 0.5041551246537396, 0.5116918844566712, 0.5191256830601092, 0.5271739130434783, 0.5383580080753702, 0.5508021390374331, 0.5565912117177096, 0.5608465608465609, 0.5684210526315789, 0.5777777777777778, 0.5859375, 0.5966277561608301, 0.6030927835051546, 0.6069142125480154, 0.6175349428208387, 0.624525916561315, 0.6306532663316583, 0.6408977556109725, 0.6476426799007443, 0.654320987654321, 0.6609336609336609, 0.665036674816626, 0.6674786845310596, 0.6787439613526571, 0.6835138387484957, 0.6866746698679472, 0.6929510155316606, 0.6967895362663495, 0.6998813760379596, 0.7014218009478673, 0.7021276595744681, 0.7051886792452831, 0.7067137809187279, 0.7104337631887455, 0.7101280558789289, 0.710801393728223, 0.7106481481481481, 0.7119815668202765, 0.7164179104477612] 
	t2 = [0, 2.127568483352661, 4.138633728027344, 6.119072914123535, 8.095943927764893, 10.069459915161133, 12.047832250595093, 14.011573553085327, 16.012410879135132, 18.040239572525024, 20.03983974456787, 22.109993934631348, 24.10592746734619, 26.112743854522705, 28.129066467285156, 30.142618417739868, 32.042094707489014, 34.083914041519165, 36.11548328399658, 38.104763984680176, 40.09643292427063, 42.10772228240967, 44.02969002723694, 46.055389165878296, 48.121466875076294, 50.01715803146362, 52.04415035247803, 54.07514762878418, 56.069010734558105, 58.07556939125061, 60.120662450790405] 
	q2 = [0.35499207606973054, 0.36619718309859156, 0.3858024691358025, 0.4012158054711247, 0.41379310344827586, 0.41901931649331353, 0.43401759530791795, 0.45151953690303903, 0.46704871060171926, 0.47807637906647804, 0.49230769230769234, 0.5075862068965517, 0.5271739130434783, 0.5469168900804289, 0.5608465608465609, 0.5777777777777778, 0.5966277561608301, 0.6076923076923078, 0.624525916561315, 0.6408977556109725, 0.655980271270037, 0.665036674816626, 0.6787439613526571, 0.689075630252101, 0.7007125890736342, 0.6990521327014219, 0.7067137809187279, 0.7087719298245614, 0.7099767981438516, 0.7149425287356321, 0.7170675830469644] 
	t3 = [0, 3.015209436416626, 6.0559046268463135, 9.087048053741455, 12.043460369110107, 15.115318059921265, 18.029727458953857, 21.110271453857422, 24.064196586608887, 27.027228116989136, 30.068256616592407, 33.123130083084106, 36.1192843914032, 39.026123046875, 42.141592264175415, 45.04181456565857, 48.12510323524475, 51.037301778793335, 54.02263617515564, 57.08010792732239, 60.0001175403595] 
	q3 = [0.35499207606973054, 0.37577639751552794, 0.4012158054711247, 0.417910447761194, 0.43401759530791795, 0.4626436781609195, 0.47807637906647804, 0.5020804438280166, 0.5271739130434783, 0.5565912117177096, 0.5796344647519581, 0.6056701030927835, 0.6262626262626262, 0.650990099009901, 0.6642246642246641, 0.6866746698679472, 0.7007125890736342, 0.7036599763872492, 0.7086247086247086, 0.7128027681660899, 0.7185354691075515] 
	t4 = [0, 4.049229860305786, 8.13135051727295, 12.010992765426636, 16.065282583236694, 20.04565191268921, 24.064995050430298, 28.092522859573364, 32.05543303489685, 36.07101607322693, 40.04587125778198, 44.047059774398804, 48.119465827941895, 52.13628792762756, 56.00375175476074, 60.10439968109131] 
	q4 = [0.35499207606973054, 0.3858024691358025, 0.4161676646706587, 0.43401759530791795, 0.46924177396280403, 0.4972067039106145, 0.5291723202170964, 0.5627476882430646, 0.5958549222797928, 0.6262626262626262, 0.6576354679802955, 0.6803377563329313, 0.7007125890736342, 0.708235294117647, 0.7106481481481481, 0.7177142857142857] 
	t5 = [0, 5.087143421173096, 10.061480522155762, 15.026299238204956, 20.035414218902588, 25.00526714324951, 30.048299074172974, 35.099504709243774, 40.00250554084778, 45.076226472854614, 50.119075536727905, 55.1063506603241, 60.00830340385437] 
	q5 = [0.35499207606973054, 0.39755351681957185, 0.4213649851632047, 0.4626436781609195, 0.4972067039106145, 0.536388140161725, 0.5814863102998696, 0.6226175349428209, 0.6576354679802955, 0.6850961538461539, 0.7014218009478673, 0.7099767981438516, 0.7177142857142857] 
	t6 = [0, 6.0126166343688965, 12.063181400299072, 18.061530590057373, 24.08365273475647, 30.081021547317505, 36.02536058425903, 42.03826594352722, 48.02005362510681, 54.03974175453186, 60.13544178009033] 
	q6 = [0.35499207606973054, 0.3987823439878234, 0.43401759530791795, 0.480225988700565, 0.5291723202170964, 0.5814863102998696, 0.6270543615676358, 0.6642246642246641, 0.7007125890736342, 0.7101280558789289, 0.7185354691075515] 
	t7 = [0, 7.022958517074585, 14.084367513656616, 21.143394947052002, 28.009997129440308, 35.01716494560242, 42.00258803367615, 49.02683997154236, 56.03936576843262] 
	q7 = [0.35499207606973054, 0.40483383685800606, 0.45375722543352603, 0.5069252077562326, 0.5608465608465609, 0.6243654822335025, 0.6674786845310596, 0.6998813760379596, 0.7136258660508082] 
	t8 = [0, 8.050017595291138, 16.123989582061768, 24.11247158050537, 32.14229464530945, 40.12683916091919, 48.13609290122986, 56.070091009140015] 
	q8 = [0.35499207606973054, 0.41379310344827586, 0.4714285714285715, 0.5311653116531165, 0.6020671834625323, 0.6609336609336609, 0.6991676575505351, 0.7136258660508082] 
	t9 = [0, 9.11162519454956, 18.017802715301514, 27.02516007423401, 36.14172291755676, 45.00130248069763, 54.02278542518616] 
	q9 = [0.35499207606973054, 0.41430700447093893, 0.480225988700565, 0.5577689243027888, 0.6322418136020151, 0.6874999999999999, 0.7094515752625438] 
	t10 = [0, 10.139702320098877, 20.117785215377808, 30.00936770439148, 40.144179344177246, 50.023319721221924, 60.071545362472534] 
	q10 = [0.35499207606973054, 0.4213649851632047, 0.5013927576601671, 0.5814863102998696, 0.6642066420664207, 0.7014218009478673, 0.7214611872146119] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.105396032333374, 2.0341031551361084, 3.1361451148986816, 4.058877229690552, 5.018264055252075, 6.09617805480957, 7.039422988891602, 8.111954689025879, 9.120322704315186, 10.07436752319336, 11.018353939056396, 12.087844610214233, 13.062521934509277, 14.04723596572876, 15.14335012435913, 16.084330558776855, 17.059581756591797, 18.01764678955078, 19.001455545425415, 20.076237440109253, 21.06319546699524, 22.022682905197144, 23.118497133255005, 24.10544490814209, 25.082632780075073, 26.012868881225586, 27.04499125480652, 28.120994329452515, 29.070351362228394, 30.1441433429718, 31.13248038291931, 32.05651235580444, 33.00190234184265, 34.080766439437866, 35.038079500198364, 36.10858345031738, 37.08911490440369, 38.0153431892395, 39.11099600791931, 40.06217908859253, 41.004875898361206, 42.08633089065552, 43.131447315216064, 44.05978345870972, 45.08751368522644, 46.07198357582092, 47.05776405334473, 48.012197494506836, 49.10037660598755, 50.04939675331116, 51.13539218902588, 52.11511015892029, 53.05706763267517, 54.03721809387207, 55.13610029220581, 56.06233072280884, 57.04061460494995, 58.0091598033905, 59.1087749004364, 60.09017276763916] 
	q1 = [0.37288135593220334, 0.382262996941896, 0.39150227617602423, 0.40060240963855415, 0.41017964071856283, 0.42136498516320475, 0.4323529411764706, 0.43923865300146414, 0.4476744186046512, 0.45151953690303914, 0.46197991391678617, 0.46723646723646717, 0.47175141242937846, 0.47471910112359544, 0.4818941504178273, 0.4903047091412742, 0.49518569463548834, 0.4972677595628415, 0.5034013605442177, 0.5135135135135135, 0.5281501340482573, 0.5326231691078561, 0.5396825396825397, 0.5447368421052631, 0.5542483660130719, 0.5610389610389611, 0.5647668393782384, 0.5740025740025739, 0.5823754789272031, 0.5895806861499364, 0.5984848484848485, 0.6037735849056605, 0.6142322097378278, 0.6220570012391574, 0.6280788177339902, 0.6381418092909535, 0.6447688564476886, 0.6481257557436517, 0.6530120481927711, 0.6586826347305389, 0.6682520808561236, 0.6721893491124261, 0.6737338044758541, 0.6830409356725147, 0.6876456876456877, 0.6929316338354576, 0.697459584295612, 0.7049368541905855, 0.7056128293241696, 0.7085714285714286, 0.7121729237770194, 0.7136363636363637, 0.7180067950169876, 0.7209039548022599, 0.7192784667418264, 0.7191011235955056, 0.7174887892376681, 0.7181208053691275, 0.7193763919821825, 0.7222222222222223, 0.7228381374722838] 
	t2 = [0, 2.115309953689575, 4.091801643371582, 6.064861536026001, 8.044912338256836, 10.126590490341187, 12.097769737243652, 14.13512372970581, 16.145100831985474, 18.030102729797363, 20.03739356994629, 22.063610076904297, 24.106041431427002, 26.00474214553833, 28.07466173171997, 30.062309503555298, 32.07235383987427, 34.03933596611023, 36.0242383480072, 38.03650403022766, 40.042585611343384, 42.0089635848999, 44.05466938018799, 46.10690689086914, 48.1442813873291, 50.00834131240845, 52.04083442687988, 54.07758688926697, 56.05551719665527, 58.08813977241516, 60.131431579589844] 
	q2 = [0.37288135593220334, 0.393939393939394, 0.41255605381165916, 0.4346549192364171, 0.4454148471615721, 0.4641833810888252, 0.47175141242937846, 0.48821081830790575, 0.4972527472527472, 0.5074626865671642, 0.5294117647058822, 0.5408970976253298, 0.5580182529335072, 0.5729032258064517, 0.5877862595419847, 0.6037735849056605, 0.6203473945409429, 0.6364749082007344, 0.6464891041162227, 0.6586826347305389, 0.6721893491124261, 0.6814988290398126, 0.6944444444444445, 0.703448275862069, 0.7115165336374003, 0.7150964812712827, 0.7192784667418264, 0.718294051627385, 0.7209821428571429, 0.7236403995560488, 0.7262693156732892] 
	t3 = [0, 3.0202276706695557, 6.008338451385498, 9.008557558059692, 12.096707344055176, 15.034079313278198, 18.105550050735474, 21.026362895965576, 24.077561140060425, 27.056297063827515, 30.004307746887207, 33.03225064277649, 36.06149888038635, 39.10972571372986, 42.12131476402283, 45.12062096595764, 48.09198236465454, 51.11790490150452, 54.08126711845398, 57.097399950027466, 60.0522518157959] 
	q3 = [0.37288135593220334, 0.40060240963855415, 0.4346549192364171, 0.45598845598845594, 0.47390691114245415, 0.49171270718232046, 0.5094850948509485, 0.5391766268260292, 0.5617685305591678, 0.5823754789272031, 0.6037735849056605, 0.6297662976629766, 0.6497584541062802, 0.6690391459074734, 0.6861143523920653, 0.7019562715765246, 0.7107061503416855, 0.7209039548022599, 0.7174887892376681, 0.7236403995560488, 0.7262693156732892] 
	t4 = [0, 4.013110160827637, 8.113893032073975, 12.1431725025177, 16.10986566543579, 20.123037815093994, 24.139151573181152, 28.002012729644775, 32.125502824783325, 36.05805730819702, 40.140734910964966, 44.13845705986023, 48.03626489639282, 52.142367362976074, 56.00721979141235, 60.11630630493164] 
	q4 = [0.37288135593220334, 0.41255605381165916, 0.4476744186046512, 0.47390691114245415, 0.49931412894375854, 0.5313751668891855, 0.5617685305591678, 0.5895806861499364, 0.6254635352286775, 0.6497584541062802, 0.6729634002361276, 0.697459584295612, 0.7107061503416855, 0.7192784667418264, 0.7216035634743875, 0.72707182320442] 
	t5 = [0, 5.070100784301758, 10.126105070114136, 15.02666687965393, 20.065263271331787, 25.127948999404907, 30.091843366622925, 35.06996726989746, 40.02262544631958, 45.064833879470825, 50.046581983566284, 55.07582664489746, 60.13709855079651] 
	q5 = [0.37288135593220334, 0.42370370370370375, 0.46285714285714286, 0.494475138121547, 0.5313751668891855, 0.5692108667529107, 0.6072772898368883, 0.6480582524271844, 0.6721698113207547, 0.703448275862069, 0.7165532879818594, 0.7209821428571429, 0.72707182320442] 
	t6 = [0, 6.145990610122681, 12.077425003051758, 18.09232807159424, 24.001799821853638, 30.12868642807007, 36.05230355262756, 42.03847646713257, 48.04779005050659, 54.11248826980591, 60.099921226501465] 
	q6 = [0.37288135593220334, 0.4369501466275659, 0.47390691114245415, 0.5121951219512195, 0.5632333767926988, 0.6090225563909775, 0.6497584541062802, 0.6861143523920653, 0.7107061503416855, 0.7189249720044794, 0.728476821192053] 
	t7 = [0, 7.1425323486328125, 14.030640840530396, 21.06299877166748, 28.092475414276123, 35.08972787857056, 42.142934799194336, 49.03164577484131, 56.09701633453369] 
	q7 = [0.37288135593220334, 0.44152046783625737, 0.4909847434119278, 0.5398936170212767, 0.5939086294416244, 0.6480582524271844, 0.6861143523920653, 0.7121729237770194, 0.7230255839822024] 
	t8 = [0, 8.034449338912964, 16.01709270477295, 24.04648518562317, 32.1293740272522, 40.0816867351532, 48.09252405166626, 56.046470642089844] 
	q8 = [0.37288135593220334, 0.4476744186046512, 0.49931412894375854, 0.5632333767926988, 0.6288532675709002, 0.6737338044758541, 0.7107061503416855, 0.7230255839822024] 
	t9 = [0, 9.031269073486328, 18.059983015060425, 27.165496826171875, 36.04305052757263, 45.03026509284973, 54.0455276966095] 
	q9 = [0.37288135593220334, 0.45598845598845594, 0.5128900949796472, 0.5882352941176471, 0.6497584541062802, 0.7019562715765246, 0.7166853303471444] 
	t10 = [0, 10.154910326004028, 20.05607843399048, 30.098641872406006, 40.11123251914978, 50.07568669319153, 60.021727085113525] 
	q10 = [0.37288135593220334, 0.46285714285714286, 0.53475935828877, 0.6115288220551379, 0.6737338044758541, 0.7165532879818594, 0.729281767955801] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.1064467430114746, 2.035010814666748, 3.129431962966919, 4.062012195587158, 5.0223400592803955, 6.099046945571899, 7.079965353012085, 8.042138576507568, 9.135159492492676, 10.05894136428833, 11.03333568572998, 12.139721155166626, 13.001505374908447, 14.109533071517944, 15.084399938583374, 16.01625418663025, 17.105311632156372, 18.055317401885986, 19.001739263534546, 20.071877241134644, 21.017168283462524, 22.124931573867798, 23.078190803527832, 24.04305601119995, 25.052572011947632, 26.131969213485718, 27.028899431228638, 28.12657141685486, 29.101203680038452, 30.022965669631958, 31.110990285873413, 32.0654673576355, 33.018310546875, 34.12055802345276, 35.128334283828735, 36.048454999923706, 37.03532600402832, 38.106497287750244, 39.051616191864014, 40.12023687362671, 41.10403299331665, 42.056941509246826, 43.00245261192322, 44.12377405166626, 45.10116624832153, 46.02752494812012, 47.02899193763733, 48.09424185752869, 49.13314986228943, 50.084173917770386, 51.02277898788452, 52.115294456481934, 53.086445569992065, 54.04914355278015, 55.037580728530884, 56.01762771606445, 57.11313843727112, 58.07342267036438, 59.02252244949341, 60.09731864929199] 
	q1 = [0.35968992248062015, 0.3717357910906298, 0.37251908396946565, 0.3768996960486322, 0.38612368024132726, 0.3892215568862275, 0.3970149253731343, 0.4035608308605341, 0.4123711340206186, 0.4233576642335767, 0.43251088534107407, 0.43578643578643583, 0.44189383070301286, 0.4457142857142857, 0.4517045454545454, 0.45698166431593795, 0.4628330995792426, 0.4700973574408901, 0.47513812154696133, 0.48010973936899853, 0.49046321525885556, 0.4959349593495935, 0.5080645161290323, 0.5087014725568942, 0.5146666666666666, 0.5245033112582782, 0.5329815303430079, 0.5399737876802098, 0.5513654096228869, 0.5607235142118863, 0.5681233933161953, 0.5761843790012804, 0.5790816326530612, 0.5862944162436549, 0.592686002522068, 0.6, 0.6044776119402986, 0.6106304079110012, 0.6174661746617466, 0.6275946275946275, 0.629404617253949, 0.6360338573155985, 0.6410564225690276, 0.6459330143540669, 0.6539833531510107, 0.6579881656804734, 0.6643109540636043, 0.6705744431418522, 0.6752336448598131, 0.6782810685249708, 0.6828703703703703, 0.6874279123414072, 0.6949541284403669, 0.6979405034324944, 0.6986301369863014, 0.7030716723549487, 0.7030716723549487, 0.7104072398190044, 0.7102593010146562, 0.7115600448933783, 0.7122060470324747] 
	t2 = [0, 2.1139209270477295, 4.075960159301758, 6.049591302871704, 8.064018249511719, 10.100325107574463, 12.133961200714111, 14.054795503616333, 16.066738843917847, 18.041176080703735, 20.044790983200073, 22.004240036010742, 24.04075789451599, 26.06483817100525, 28.132484197616577, 30.02994728088379, 32.021918535232544, 34.132394313812256, 36.01193833351135, 38.023810148239136, 40.010658264160156, 42.04464364051819, 44.04640340805054, 46.07134938240051, 48.11340284347534, 50.035059213638306, 52.021910667419434, 54.07770848274231, 56.12495565414429, 58.136791706085205, 60.008063554763794] 
	q2 = [0.35968992248062015, 0.375, 0.3855421686746988, 0.3994038748137108, 0.4147058823529412, 0.43188405797101453, 0.44476327116212333, 0.45609065155807366, 0.4664804469273742, 0.47867950481430543, 0.49660786974219806, 0.5093833780160858, 0.5225464190981433, 0.5380577427821522, 0.5588615782664942, 0.5725288831835686, 0.5862944162436549, 0.6007509386733417, 0.6106304079110012, 0.6275946275946275, 0.6360338573155985, 0.6475507765830346, 0.6579881656804734, 0.6705744431418522, 0.6813441483198146, 0.6889400921658986, 0.6979405034324944, 0.7030716723549487, 0.708803611738149, 0.7130044843049328, 0.7157190635451505] 
	t3 = [0, 3.015021562576294, 6.009306907653809, 9.031051874160767, 12.103087663650513, 15.080584049224854, 18.133150815963745, 21.049447059631348, 24.06956195831299, 27.02647590637207, 30.011606454849243, 33.04646348953247, 36.003668785095215, 39.06670117378235, 42.11131429672241, 45.07789969444275, 48.14567422866821, 51.00379824638367, 54.069950580596924, 57.0634868144989, 60.096243143081665] 
	q3 = [0.35968992248062015, 0.3768996960486322, 0.3994038748137108, 0.4256559766763848, 0.4454022988505747, 0.45915492957746484, 0.47802197802197804, 0.5040431266846362, 0.5245033112582782, 0.5494791666666667, 0.5761843790012804, 0.5944584382871537, 0.6140567200986435, 0.6310679611650486, 0.6491646778042959, 0.6690140845070421, 0.6828703703703703, 0.6964490263459335, 0.7015945330296126, 0.7115600448933783, 0.7171492204899778] 
	t4 = [0, 4.06152606010437, 8.118896245956421, 12.088172197341919, 16.143869876861572, 20.071171283721924, 24.021873950958252, 28.044050931930542, 32.101563930511475, 36.063748836517334, 40.02901601791382, 44.014066219329834, 48.067699670791626, 52.10304260253906, 56.14354944229126, 60.01222538948059] 
	q4 = [0.35968992248062015, 0.38253012048192764, 0.41703377386196766, 0.4454022988505747, 0.4686192468619247, 0.4959349593495935, 0.5245033112582782, 0.5607235142118863, 0.5855513307984791, 0.6157635467980296, 0.636144578313253, 0.6611570247933886, 0.6828703703703703, 0.6971428571428571, 0.7117117117117118, 0.7171492204899778] 
	t5 = [0, 5.110489130020142, 10.030514240264893, 15.11983585357666, 20.068899393081665, 25.049203634262085, 30.037685871124268, 35.10281562805176, 40.13005352020264, 45.14083504676819, 50.06133580207825, 55.083794355392456, 60.13676333427429] 
	q5 = [0.35968992248062015, 0.3892215568862275, 0.43188405797101453, 0.4619718309859155, 0.4959349593495935, 0.5349143610013175, 0.5754475703324808, 0.607940446650124, 0.6377858002406739, 0.6705744431418522, 0.69345579793341, 0.7081447963800905, 0.7185761957730812] 
	t6 = [0, 6.100279331207275, 12.079814195632935, 18.14011001586914, 24.054452657699585, 30.11324644088745, 36.12600612640381, 42.0411434173584, 48.1314377784729, 54.13880920410156, 60.0360963344574] 
	q6 = [0.35968992248062015, 0.3994038748137108, 0.44476327116212333, 0.47867950481430543, 0.5245033112582782, 0.5736235595390525, 0.6165228113440198, 0.6507747318235996, 0.684393063583815, 0.7045454545454545, 0.7185761957730812] 
	t7 = [0, 7.031321048736572, 14.120330095291138, 21.132346630096436, 28.051488876342773, 35.00629806518555, 42.11590647697449, 49.10604953765869, 56.051589488983154] 
	q7 = [0.35968992248062015, 0.4053254437869822, 0.4589235127478754, 0.5080645161290323, 0.5625806451612904, 0.6096654275092938, 0.6539833531510107, 0.6904487917146145, 0.7109111361079866] 
	t8 = [0, 8.04356074333191, 16.00500750541687, 24.14188289642334, 32.00775861740112, 40.10834288597107, 48.01223587989807, 56.018314599990845] 
	q8 = [0.35968992248062015, 0.4140969162995595, 0.46993006993006997, 0.5264550264550265, 0.5873417721518988, 0.6410564225690276, 0.684393063583815, 0.7094594594594594] 
	t9 = [0, 9.095318078994751, 18.0178062915802, 27.137033939361572, 36.09387278556824, 45.007344245910645, 54.117467641830444] 
	q9 = [0.35968992248062015, 0.4256559766763848, 0.48000000000000004, 0.5513654096228869, 0.6165228113440198, 0.6705744431418522, 0.7045454545454545] 
	t10 = [0, 10.127776861190796, 20.030063152313232, 30.053297758102417, 40.114386320114136, 50.1410448551178, 60.10496139526367] 
	q10 = [0.35968992248062015, 0.43125904486251815, 0.4993215739484396, 0.5754475703324808, 0.6434573829531813, 0.6964490263459335, 0.7200000000000001] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.10384202003479, 2.0351593494415283, 3.137641429901123, 4.075869560241699, 5.024151563644409, 6.120245456695557, 7.076704025268555, 8.029924154281616, 9.137207508087158, 10.009272575378418, 11.140170335769653, 12.069201231002808, 13.029557943344116, 14.106544494628906, 15.08406662940979, 16.023269653320312, 17.12897300720215, 18.052266836166382, 19.03107523918152, 20.104134559631348, 21.090724229812622, 22.05825424194336, 23.01513910293579, 24.016384840011597, 25.143568992614746, 26.139830112457275, 27.115766525268555, 28.101880311965942, 29.13631296157837, 30.11775541305542, 31.090200424194336, 32.00910425186157, 33.11112332344055, 34.074912548065186, 35.03836512565613, 36.11360216140747, 37.08753800392151, 38.04151487350464, 39.14384913444519, 40.06503701210022, 41.06697106361389, 42.01611542701721, 43.11570954322815, 44.0768027305603, 45.02319526672363, 46.10261607170105, 47.05084013938904, 48.0080361366272, 49.095025062561035, 50.07468318939209, 51.01627826690674, 52.028071641922, 53.11772608757019, 54.066482067108154, 55.02294111251831, 56.10400629043579, 57.111265897750854, 58.04350304603577, 59.02650737762451, 60.09620976448059] 
	q1 = [0.3572567783094099, 0.36708860759493667, 0.3679245283018868, 0.38317757009345793, 0.38948995363214833, 0.3987730061349693, 0.4054878048780487, 0.41274658573596357, 0.4175491679273827, 0.4270676691729323, 0.4328358208955224, 0.4414814814814815, 0.4477172312223859, 0.45839416058394167, 0.46153846153846156, 0.4726224783861671, 0.48068669527896996, 0.4886363636363637, 0.4992947813822285, 0.5, 0.505586592178771, 0.5159500693481276, 0.5261707988980716, 0.5302197802197802, 0.5340599455040872, 0.5447154471544715, 0.5572005383580081, 0.5622489959839357, 0.5660881174899867, 0.5725699067909454, 0.5812417437252312, 0.5842105263157895, 0.5934640522875817, 0.5999999999999999, 0.6072351421188631, 0.6143958868894601, 0.619718309859155, 0.6226175349428209, 0.6287878787878788, 0.6365914786967418, 0.64, 0.6450809464508096, 0.650185414091471, 0.656019656019656, 0.6593406593406593, 0.6642424242424242, 0.6730769230769231, 0.6794258373205742, 0.6833333333333333, 0.6864608076009502, 0.6949352179034157, 0.6980023501762633, 0.7017543859649121, 0.704784130688448, 0.7062937062937062, 0.7023255813953487, 0.7045191193511008, 0.7037037037037038, 0.7043879907621247, 0.7072330654420206, 0.7116704805491991] 
	t2 = [0, 2.1268553733825684, 4.0992207527160645, 6.064614534378052, 8.078210353851318, 10.123908758163452, 12.056236267089844, 14.018037796020508, 16.037100315093994, 18.000027179718018, 20.01196265220642, 22.04668927192688, 24.072707176208496, 26.063589572906494, 28.053837299346924, 30.02805256843567, 32.03729224205017, 34.14395332336426, 36.142974615097046, 38.04996943473816, 40.02554512023926, 42.09470057487488, 44.09435296058655, 46.076205253601074, 48.099783420562744, 50.137906312942505, 52.05716872215271, 54.076382637023926, 56.08646559715271, 58.11348271369934, 60.05897283554077] 
	q2 = [0.3572567783094099, 0.3704866562009419, 0.39197530864197533, 0.4079147640791477, 0.41993957703927487, 0.4398216939078751, 0.4522760646108664, 0.46599131693198254, 0.4821683309557774, 0.4978902953586498, 0.5097493036211699, 0.5302197802197802, 0.5420054200542005, 0.5603217158176944, 0.5725699067909454, 0.5842105263157895, 0.5999999999999999, 0.6143958868894601, 0.6234096692111959, 0.6365914786967418, 0.6467661691542289, 0.6576687116564417, 0.6650544135429262, 0.6801909307875895, 0.6895734597156399, 0.6971830985915494, 0.704784130688448, 0.7038327526132404, 0.7043879907621247, 0.7087155963302754, 0.7189988623435722] 
	t3 = [0, 3.0305254459381104, 6.0463783740997314, 9.05952525138855, 12.0929434299469, 15.123893976211548, 18.14662265777588, 21.062455892562866, 24.107335805892944, 27.137301683425903, 30.053099155426025, 33.116496562957764, 36.04202389717102, 39.13483381271362, 42.08266806602478, 45.13493299484253, 48.01117992401123, 51.08022689819336, 54.06973171234131, 57.1374773979187, 60.066750288009644] 
	q3 = [0.3572567783094099, 0.3806552262090484, 0.4079147640791477, 0.42942942942942935, 0.4545454545454546, 0.4763271162123386, 0.4978902953586498, 0.5241379310344827, 0.5454545454545455, 0.5641711229946524, 0.5868421052631578, 0.6090322580645161, 0.625158831003812, 0.6416978776529338, 0.6585067319461444, 0.6762589928057554, 0.6926713947990544, 0.7032710280373832, 0.7053364269141531, 0.7072330654420206, 0.7227272727272727] 
	t4 = [0, 4.044890403747559, 8.11553406715393, 12.135585069656372, 16.107906341552734, 20.059551000595093, 24.041853427886963, 28.07602882385254, 32.06644129753113, 36.01296353340149, 40.1181435585022, 44.09707188606262, 48.000900983810425, 52.137513160705566, 56.13203954696655, 60.11619830131531] 
	q4 = [0.3572567783094099, 0.39197530864197533, 0.41993957703927487, 0.4538799414348463, 0.4843304843304843, 0.513888888888889, 0.5454545454545455, 0.5718085106382979, 0.6036269430051814, 0.6243654822335025, 0.6485148514851484, 0.671480144404332, 0.6957547169811321, 0.7039627039627039, 0.7065592635212887, 0.72562358276644] 
	t5 = [0, 5.065242767333984, 10.018216133117676, 15.111172199249268, 20.081200122833252, 25.13726043701172, 30.12035346031189, 35.11190366744995, 40.09964346885681, 45.01341772079468, 50.00719404220581, 55.02365183830261, 60.07996988296509] 
	q5 = [0.3572567783094099, 0.3987730061349693, 0.4398216939078751, 0.47851002865329506, 0.5159500693481276, 0.5572005383580081, 0.5923984272608126, 0.621656050955414, 0.650185414091471, 0.6794258373205742, 0.7025761124121779, 0.7052023121387284, 0.72562358276644] 
	t6 = [0, 6.125445604324341, 12.030800342559814, 18.14254069328308, 24.039398431777954, 30.091728925704956, 36.0382399559021, 42.10997986793518, 48.075047731399536, 54.02006483078003, 60.04683303833008] 
	q6 = [0.3572567783094099, 0.4054878048780487, 0.4545454545454546, 0.5, 0.5474254742547426, 0.5905511811023622, 0.6261089987325729, 0.6593406593406593, 0.6957547169811321, 0.703016241299304, 0.72562358276644] 
	t7 = [0, 7.043611764907837, 14.084587574005127, 21.094029188156128, 28.056665420532227, 35.03165125846863, 42.09549832344055, 49.08416175842285, 56.11940574645996] 
	q7 = [0.3572567783094099, 0.41274658573596357, 0.4733044733044733, 0.5281980742778541, 0.5763612217795485, 0.621656050955414, 0.6609756097560975, 0.6980023501762633, 0.7057471264367815] 
	t8 = [0, 8.06710171699524, 16.06169080734253, 24.053307056427002, 32.03597950935364, 40.068986892700195, 48.1207160949707, 56.073609352111816] 
	q8 = [0.3572567783094099, 0.41993957703927487, 0.4864864864864865, 0.5501355013550137, 0.6044098573281451, 0.6526576019777504, 0.6957547169811321, 0.7057471264367815] 
	t9 = [0, 9.070756196975708, 18.14772367477417, 27.045846939086914, 36.00628137588501, 45.143860816955566, 54.05282545089722] 
	q9 = [0.3572567783094099, 0.4287856071964018, 0.5, 0.5687583444592791, 0.6278481012658228, 0.6817640047675804, 0.703016241299304] 
	t10 = [0, 10.119614601135254, 20.072014570236206, 30.056801557540894, 40.01114058494568, 50.11689043045044, 60.01628375053406] 
	q10 = [0.3572567783094099, 0.4375, 0.5180055401662049, 0.5931758530183726, 0.6526576019777504, 0.7040935672514619, 0.7270668176670442] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.0937399864196777, 2.024665355682373, 3.128225564956665, 4.091376543045044, 5.044295310974121, 6.1500184535980225, 7.095489501953125, 8.052555799484253, 9.1453378200531, 10.098939657211304, 11.04302453994751, 12.12613821029663, 13.106080055236816, 14.103241205215454, 15.054578304290771, 16.036682605743408, 17.027039766311646, 18.128664255142212, 19.07500433921814, 20.059596300125122, 21.007156133651733, 22.12272620201111, 23.06769847869873, 24.144907236099243, 25.11882472038269, 26.10021162033081, 27.079021692276, 28.038919687271118, 29.138776779174805, 30.123178720474243, 31.06489372253418, 32.02567386627197, 33.00534701347351, 34.1030912399292, 35.054394245147705, 36.12489604949951, 37.097174644470215, 38.02302050590515, 39.11128497123718, 40.09047317504883, 41.03409481048584, 42.10985517501831, 43.00080871582031, 44.07011818885803, 45.04128384590149, 46.00635576248169, 47.12755513191223, 48.060126066207886, 49.01610565185547, 50.0957190990448, 51.06446313858032, 52.156355142593384, 53.13068389892578, 54.10469031333923, 55.062453508377075, 56.0511040687561, 57.13641691207886, 58.098862648010254, 59.077165842056274, 60.14951777458191] 
	q1 = [0.40120663650075417, 0.4131736526946108, 0.42199108469539376, 0.4283604135893649, 0.4375917767988252, 0.4408759124087591, 0.4515195369030391, 0.46043165467625896, 0.4663805436337626, 0.4744318181818182, 0.4851904090267982, 0.48807854137447393, 0.4951321279554937, 0.5006915629322268, 0.5034387895460798, 0.5102880658436214, 0.5109289617486339, 0.5135869565217391, 0.5202156334231806, 0.5254691689008043, 0.5372340425531915, 0.5403973509933775, 0.5454545454545454, 0.5549738219895288, 0.564369310793238, 0.5717981888745148, 0.5791505791505792, 0.5838668373879642, 0.5885350318471337, 0.5931558935361217, 0.6002522068095838, 0.6047678795483061, 0.6109725685785536, 0.6178660049627792, 0.623921085080148, 0.6323529411764707, 0.6365853658536585, 0.6424242424242425, 0.6457831325301205, 0.6522781774580335, 0.6563245823389021, 0.660332541567696, 0.6619385342789599, 0.6698002350176262, 0.6760233918128654, 0.6813953488372092, 0.6829268292682927, 0.6897347174163783, 0.6942528735632183, 0.695752009184845, 0.6994285714285714, 0.7009132420091325, 0.7, 0.699205448354143, 0.7021517553793885, 0.7058823529411764, 0.705749718151071, 0.7064116985376828, 0.7056179775280899, 0.7142857142857143, 0.7171492204899778] 
	t2 = [0, 2.1186115741729736, 4.114548206329346, 6.100404500961304, 8.095241785049438, 10.15027141571045, 12.117229700088501, 14.037127017974854, 16.042845010757446, 18.098935842514038, 20.11106014251709, 22.034787893295288, 24.00568914413452, 26.026455402374268, 28.10825490951538, 30.079814195632935, 32.14102339744568, 34.01567554473877, 36.130645751953125, 38.142467975616455, 40.03009629249573, 42.00344181060791, 44.05484676361084, 46.08697485923767, 48.087137937545776, 50.00480246543884, 52.11048078536987, 54.11179184913635, 56.01320219039917, 58.044594526290894, 60.01976752281189] 
	q2 = [0.40120663650075417, 0.41666666666666674, 0.439882697947214, 0.453757225433526, 0.4685714285714286, 0.48450704225352104, 0.497913769123783, 0.5061898211829436, 0.5122615803814713, 0.5234899328859061, 0.5403973509933775, 0.5511811023622046, 0.5662337662337662, 0.582798459563543, 0.5931558935361217, 0.6047678795483061, 0.6178660049627792, 0.6306748466257669, 0.6424242424242425, 0.6522781774580335, 0.660332541567696, 0.6698002350176262, 0.6813953488372092, 0.6882217090069285, 0.6972477064220183, 0.7023945267958951, 0.7006802721088435, 0.7065462753950338, 0.7064116985376828, 0.7157190635451505, 0.7192008879023307] 
	t3 = [0, 3.015751361846924, 6.054513454437256, 9.083383083343506, 12.025583028793335, 15.003837585449219, 18.090383291244507, 21.043514490127563, 24.121602773666382, 27.10456609725952, 30.04248881340027, 33.02653241157532, 36.13175439834595, 39.039167404174805, 42.13355302810669, 45.0157573223114, 48.06096172332764, 51.03585886955261, 54.09101438522339, 57.110790967941284, 60.05733847618103] 
	q3 = [0.40120663650075417, 0.4260355029585799, 0.453757225433526, 0.47875354107648727, 0.497913769123783, 0.5095890410958904, 0.5234899328859061, 0.5408970976253298, 0.5680933852140078, 0.5892857142857143, 0.6047678795483061, 0.623921085080148, 0.6424242424242425, 0.6555423122765197, 0.6713615023474178, 0.685979142526072, 0.6987399770904925, 0.699205448354143, 0.7065462753950338, 0.7128491620111732, 0.7220376522702104] 
	t4 = [0, 4.070405006408691, 8.125277996063232, 12.130149841308594, 16.11780595779419, 20.116803407669067, 24.01075839996338, 28.137305736541748, 32.01231622695923, 36.10742115974426, 40.05721044540405, 44.08110857009888, 48.0751953125, 52.112550258636475, 56.09188222885132, 60.082128047943115] 
	q4 = [0.40120663650075417, 0.4369501466275659, 0.4707560627674751, 0.5, 0.5135869565217391, 0.5403973509933775, 0.5680933852140078, 0.5939086294416245, 0.6195786864931847, 0.6457073760580412, 0.6635071090047393, 0.6813953488372092, 0.6964490263459335, 0.7021517553793885, 0.7078651685393258, 0.7226519337016576] 
	t5 = [0, 5.118979215621948, 10.113738298416138, 15.097191572189331, 20.063658237457275, 25.142924070358276, 30.045031785964966, 35.09521007537842, 40.10499620437622, 45.015477418899536, 50.08897924423218, 55.10871171951294, 60.03741717338562] 
	q5 = [0.40120663650075417, 0.44314868804664725, 0.48450704225352104, 0.5102880658436214, 0.5403973509933775, 0.577319587628866, 0.6047678795483061, 0.637469586374696, 0.6635071090047393, 0.6851851851851852, 0.6985210466439136, 0.7072072072072073, 0.7212389380530974] 
	t6 = [0, 6.116238594055176, 12.105875730514526, 18.100828886032104, 24.078404903411865, 30.038821935653687, 36.04976296424866, 42.123486280441284, 48.024662494659424, 54.14381790161133, 60.09356951713562] 
	q6 = [0.40120663650075417, 0.453757225433526, 0.5, 0.5254691689008043, 0.5717981888745148, 0.6090225563909774, 0.644927536231884, 0.6744730679156907, 0.6979405034324944, 0.705749718151071, 0.7226519337016576] 
	t7 = [0, 7.046228647232056, 14.112504482269287, 21.13772988319397, 28.161747932434082, 35.07621622085571, 42.02634358406067, 49.06788110733032, 56.080103397369385] 
	q7 = [0.40120663650075417, 0.46043165467625896, 0.5089408528198074, 0.5473684210526316, 0.5956907477820025, 0.6399026763990269, 0.6744730679156907, 0.6986301369863013, 0.7122060470324749] 
	t8 = [0, 8.05454158782959, 16.007837057113647, 24.082568407058716, 32.051949977874756, 40.12674856185913, 48.047961473464966, 56.13893699645996] 
	q8 = [0.40120663650075417, 0.4685714285714286, 0.5149863760217984, 0.5699481865284974, 0.6195786864931847, 0.6627218934911243, 0.6964490263459335, 0.7107623318385651] 
	t9 = [0, 9.11591911315918, 18.147902965545654, 27.010030508041382, 36.01348376274109, 45.1370313167572, 54.017486810684204] 
	q9 = [0.40120663650075417, 0.47875354107648727, 0.5301204819277108, 0.5903307888040712, 0.644927536231884, 0.6867052023121387, 0.7034949267192785] 
	t10 = [0, 10.06468677520752, 20.034788608551025, 30.047411918640137, 40.001662731170654, 50.04589033126831, 60.097463846206665] 
	q10 = [0.40120663650075417, 0.4838255977496484, 0.5423280423280423, 0.6072772898368882, 0.6627218934911243, 0.6947608200455581, 0.7234513274336285] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	
	q1 = [sum(e)/len(e) for e in zip(*q1_all)]
	q2 = [sum(e)/len(e) for e in zip(*q2_all)]
	q3 = [sum(e)/len(e) for e in zip(*q3_all)]
	q4 = [sum(e)/len(e) for e in zip(*q4_all)]
	q5 = [sum(e)/len(e) for e in zip(*q5_all)]
	q6 = [sum(e)/len(e) for e in zip(*q6_all)]
	q7 = [sum(e)/len(e) for e in zip(*q7_all)]
	q8 = [sum(e)/len(e) for e in zip(*q8_all)]
	q9 = [sum(e)/len(e) for e in zip(*q9_all)]
	q10 = [sum(e)/len(e) for e in zip(*q10_all)]
	
	
	'''
	plt.plot(t1, q1,lw=2,color='green',marker='o',  label='Epoch size(small)')
	plt.plot(t2, q2,lw=2,color='orange',marker='^',  label='Epoch size(large)')
	plt.plot(t3, q3,lw=2,color='blue',marker ='d', label='Epoch size(medium)') ##2,000
	'''
	plt.plot(t1, q1,lw=2,color='blue',marker='o',  label='Iterative Approach(epoch=1)')
	plt.plot(t2, q2,lw=2,color='green',marker='^',  label='Iterative Approach(epoch=2)')
	plt.plot(t3, q3,lw=2,color='orange',marker ='d', label='Iterative Approach(epoch=3)') ##2,000
	plt.plot(t4, q4,lw=2,color='yellow',marker='o',  label='Iterative Approach(epoch=4)')
	plt.plot(t5, q5,lw=2,color='black',marker='^',  label='Iterative Approach(epoch=5)')
	plt.plot(t6, q6,lw=2,color='cyan',marker ='d', label='Iterative Approach(epoch=6)') ##2,000
	
	
	
	'''
	plt.plot(t4, q4,lw=2,color='cyan',marker='o',  label='Iterative Approach')
	plt.plot(t5, q5,lw=2,color='yellow',marker='^',  label='Baseline1 (Function Based Approach)')
	plt.plot(t6, q6,lw=2,color='black',marker ='d', label='Baseline2 (Object Based Approach)') ##2,000
	'''
	
	plt.ylim([0, 1])
	plt.xlim([0, max(max(t1),max(t2),max(t3))])
	plt.title('Quality vs Cost')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .11), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='x-small')
	plt.ylabel('F1-measure')
	#plt.ylabel('Gain')
	plt.xlabel('Cost')	
	plt.savefig('Muct_epoch_size_variation_gender_1000_epoch1.png', format='png')
	plt.savefig('Muct_epoch_size_variation_gender_1000_epoch1.eps', format='eps')
		#plt.show()
	plt.close()
	
	
	q1_new = np.asarray(q1)
	q2_new = np.asarray(q2)
	q3_new = np.asarray(q3)
	q4_new = np.asarray(q4)
	q5_new = np.asarray(q5)
	q6_new = np.asarray(q6)
	q7_new = np.asarray(q7)
	q8_new = np.asarray(q8)
	q9_new = np.asarray(q9)
	q10_new = np.asarray(q10)
	
	t1_new = [sum(e)/len(e) for e in zip(*t1_all)]
	t2_new = [sum(e)/len(e) for e in zip(*t2_all)]
	t3_new = [sum(e)/len(e) for e in zip(*t3_all)]
	t4_new = [sum(e)/len(e) for e in zip(*t4_all)]
	t5_new = [sum(e)/len(e) for e in zip(*t5_all)]
	t6_new = [sum(e)/len(e) for e in zip(*t6_all)]
	t7_new = [sum(e)/len(e) for e in zip(*t7_all)]
	t8_new = [sum(e)/len(e) for e in zip(*t8_all)]
	t9_new = [sum(e)/len(e) for e in zip(*t9_all)]
	t10_new = [sum(e)/len(e) for e in zip(*t10_all)]
	
	
	
	t1_list = [t1_new,t2_new,t3_new,t4_new,t5_new,t6_new,t7_new,t8_new,t9_new,t10_new]
	q1_list = [q1_new,q2_new,q3_new,q4_new,q5_new,q6_new,q7_new,q8_new,q9_new,q10_new]
	#epoch_list = [1,2,4,6,8,10]
	epoch_list = [1,2,3,4,5,6,7,8,9,10]
	score_list = []
	
	for i1 in range(len(t1_list)):
		t1_2 = t1_list[i1]
		t1_2 = t1_2[1:]
		q1_2 = q1_list[i1]
		weight_t1 = [max(1-float(element - 1)/budget,0) for element in t1_2]
		improv_q1 = [x - q1_2[i - 1] for i, x in enumerate(q1_2) if i > 0]
		print weight_t1
		print improv_q1
		a1 = np.dot(weight_t1,improv_q1)
		print a1
		score_list.append(a1)
	print>>f1,"epoch_list = {} ".format(epoch_list)
	print>>f1,"score_list = {} ".format(score_list)	
	plt.plot(epoch_list, score_list,lw=2,color='blue',marker='o',  label='Iterative Approach')
	
	
	#plt.ylim([0, 1])
	plt.xlim([0, max(epoch_list)])
	plt.title('AUC Score vs Epoch Size')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .11), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
	plt.ylabel('AUC Score')
	#plt.ylabel('Gain')
	plt.xlabel('Epoch Size')	
	plt.savefig('EpochSize_AUC_Plot_500.png', format='png')
	plt.savefig('EpochSize_AUC_Plot_500.eps', format='eps')
		#plt.show()
	plt.close()	
	
	##### Plotting with setting the ylim #######
	plt.plot(epoch_list, score_list,lw=2,color='blue',marker='o',  label='Iterative Approach')
	
	
	plt.ylim([0, 1])
	plt.xlim([0, max(epoch_list)])
	plt.title('AUC Score vs Epoch Size')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .11), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
	plt.ylabel('AUC Score')
	#plt.ylabel('Gain')
	plt.xlabel('Epoch Size')	
	plt.savefig('EpochSize_AUC_Plot_ylim_1000.png', format='png')
	plt.savefig('EpochSize_AUC_Plot_ylim_1000.eps', format='eps')
		#plt.show()
	plt.close()




def plotOptimalEpochDifferentSelectivity():
	epoch_list = [1,2,3,4,5,6,7,8,9,10]
	t1_all,q1_all,t2_all,q2_all,t3_all,q3_all=[],[],[],[],[],[]
	t4_all,q4_all,t5_all,q5_all,t6_all,q6_all=[],[],[],[],[],[]
	t7_all,q7_all,t8_all,q8_all,t9_all,q9_all=[],[],[],[],[],[]	
	t10_all,q10_all=[],[]
	# Plotting epoch for 1000 objects.
	
	f1 = open('PlotEpoch.txt','w+')
	budget = 60
	t1 = [0, 1.1421301364898682, 2.07663631439209, 3.0818979740142822, 4.08338737487793, 5.099615097045898, 6.107669353485107, 7.094089508056641, 8.08315372467041, 9.108853816986084, 10.095873832702637, 11.027074337005615, 12.003092288970947, 13.005414485931396, 14.013357162475586, 15.098936080932617, 16.068843126296997, 17.063591241836548, 18.040178060531616, 19.13695192337036, 20.120185136795044, 21.12131643295288, 22.07657551765442, 23.114760875701904, 24.069523811340332, 25.049437999725342, 26.020316123962402, 27.042701482772827, 28.05931520462036, 29.072198152542114, 30.043521404266357, 31.03858733177185, 32.12394857406616, 33.07915925979614, 34.05141496658325, 35.015876054763794, 36.00778341293335, 37.10968017578125, 38.06522750854492, 39.005422592163086, 40.14305543899536, 41.0788733959198, 42.05105662345886, 43.01772713661194, 44.126885414123535, 45.12817454338074, 46.015957832336426, 47.09357523918152, 48.07171106338501, 49.02590227127075, 50.13283610343933, 51.12456941604614, 52.088176250457764, 53.0419499874115, 54.130590200424194, 55.10443305969238, 56.1087384223938, 57.07093143463135, 58.03224325180054, 59.10783624649048, 60.10301423072815] 
	q1 = [0.36419753086419754, 0.37116564417177916, 0.3780487804878049, 0.3897280966767372, 0.3963963963963964, 0.40834575260804773, 0.41715976331360943, 0.4258443465491924, 0.43045387994143486, 0.4366812227074236, 0.44283646888567296, 0.4511494252873563, 0.4549356223175965, 0.4580369843527739, 0.4661016949152543, 0.4733893557422969, 0.4735376044568246, 0.48342541436464087, 0.4814305364511692, 0.48907103825136616, 0.4945652173913044, 0.5067385444743935, 0.5087483176312247, 0.516042780748663, 0.5232403718459496, 0.5317460317460317, 0.5421052631578948, 0.54640522875817, 0.5539661898569571, 0.5607235142118863, 0.5692307692307692, 0.5739795918367347, 0.5865992414664982, 0.5937106918238995, 0.5947302383939774, 0.6027397260273972, 0.607940446650124, 0.6182266009852218, 0.625, 0.6317073170731707, 0.6383495145631068, 0.6457831325301205, 0.6491017964071857, 0.6563614744351962, 0.6603550295857987, 0.6658823529411765, 0.6721311475409836, 0.675990675990676, 0.679814385150812, 0.6851211072664359, 0.6881472957422323, 0.6911595866819747, 0.6926605504587156, 0.6948571428571428, 0.6947608200455581, 0.6969353007945517, 0.6990950226244345, 0.701912260967379, 0.7025813692480359, 0.7083798882681565, 0.7104677060133631] 
	t2 = [0, 2.1238903999328613, 4.110911846160889, 6.091003179550171, 8.084938764572144, 10.012205123901367, 12.140867710113525, 14.058226585388184, 16.022084712982178, 18.071865797042847, 20.132930040359497, 22.03125500679016, 24.11094331741333, 26.003081560134888, 28.105425596237183, 30.034626007080078, 32.01245093345642, 34.138784646987915, 36.06335973739624, 38.0503294467926, 40.068623542785645, 42.065184593200684, 44.13629341125488, 46.13137221336365, 48.02508807182312, 50.04358744621277, 52.03864145278931, 54.01003336906433, 56.04631781578064, 58.084187030792236, 60.134846448898315] 
	q2 = [0.36419753086419754, 0.38239757207890746, 0.4059701492537313, 0.4235294117647059, 0.43440233236151604, 0.4511494252873563, 0.46022727272727276, 0.4727272727272728, 0.4806629834254144, 0.48840381991814463, 0.5087483176312247, 0.5173333333333333, 0.5375494071146245, 0.5528031290743155, 0.5681233933161953, 0.5830164765525983, 0.5947302383939774, 0.6104218362282878, 0.625, 0.6383495145631068, 0.6507177033492824, 0.6603550295857987, 0.6728971962616822, 0.6813441483198147, 0.689655172413793, 0.6941580756013747, 0.6977272727272728, 0.701240135287486, 0.7069351230425057, 0.7111111111111111, 0.7160220994475138] 
	t3 = [0, 3.0074496269226074, 6.014394998550415, 9.030627489089966, 12.127566576004028, 15.093461751937866, 18.01807188987732, 21.092859029769897, 24.094733715057373, 27.088606119155884, 30.007619619369507, 33.02085256576538, 36.10740900039673, 39.13725924491882, 42.031683921813965, 45.07506704330444, 48.00115609169006, 51.113260984420776, 54.11562442779541, 57.04399299621582, 60.13231015205383] 
	q3 = [0.36419753086419754, 0.39457831325301207, 0.4235294117647059, 0.44283646888567296, 0.4624113475177305, 0.4798890429958391, 0.49046321525885556, 0.5140562248995983, 0.5421052631578948, 0.5588615782664942, 0.5865992414664982, 0.6052303860523038, 0.63003663003663, 0.6474820143884892, 0.6611570247933884, 0.6782810685249709, 0.6911595866819747, 0.6947608200455581, 0.7027027027027026, 0.7126948775055679, 0.7174392935982341] 
	t4 = [0, 4.0333921909332275, 8.08641767501831, 12.087337493896484, 16.11101269721985, 20.076168060302734, 24.09618878364563, 28.033162355422974, 32.11526155471802, 36.08677673339844, 40.05938243865967, 44.038376808166504, 48.098082304000854, 52.122355937957764, 56.09919619560242, 60.10752868652344] 
	q4 = [0.36419753086419754, 0.4059701492537313, 0.4366812227074236, 0.4624113475177305, 0.4793388429752066, 0.510752688172043, 0.5447368421052631, 0.5699614890885751, 0.6, 0.63003663003663, 0.6547619047619048, 0.6744457409568261, 0.6911595866819747, 0.6984126984126985, 0.7083798882681565, 0.7174392935982341] 
	t5 = [0, 5.069773197174072, 10.007799863815308, 15.086117029190063, 20.05875873565674, 25.138978719711304, 30.047862768173218, 35.11377477645874, 40.00264000892639, 45.00557541847229, 50.113765478134155, 55.09384083747864, 60.107484579086304] 
	q5 = [0.36419753086419754, 0.4148148148148148, 0.4505021520803443, 0.479224376731302, 0.510752688172043, 0.5497382198952879, 0.5873417721518986, 0.6257668711656442, 0.6547619047619048, 0.6782810685249709, 0.6940639269406392, 0.7077267637178052, 0.7174392935982341] 
	t6 = [0, 6.001662254333496, 12.136298179626465, 18.005889415740967, 24.111199617385864, 30.122830629348755, 36.10718774795532, 42.012521743774414, 48.13635873794556, 54.08355522155762, 60.12863516807556] 
	q6 = [0.36419753086419754, 0.4235294117647059, 0.4624113475177305, 0.49046321525885556, 0.5447368421052631, 0.5891276864728192, 0.6324786324786326, 0.6627358490566038, 0.6911595866819747, 0.701912260967379, 0.7182320441988951] 
	t7 = [0, 7.034785032272339, 14.098866939544678, 21.102428197860718, 28.065725564956665, 35.05711913108826, 42.00697612762451, 49.04806089401245, 56.01845026016235] 
	q7 = [0.36419753086419754, 0.43045387994143486, 0.4727272727272728, 0.516042780748663, 0.5732647814910026, 0.6257668711656442, 0.6643109540636042, 0.6926605504587156, 0.7120535714285715] 
	t8 = [0, 8.04780101776123, 16.07738494873047, 24.126025915145874, 32.110719203948975, 40.05675172805786, 48.058434009552, 56.024736404418945] 
	q8 = [0.36419753086419754, 0.4366812227074236, 0.48, 0.5466491458607096, 0.6042446941323346, 0.6563614744351962, 0.689655172413793, 0.7120535714285715] 
	t9 = [0, 9.097555875778198, 18.09144139289856, 27.099843502044678, 36.14071989059448, 45.023149251937866, 54.13927960395813] 
	q9 = [0.36419753086419754, 0.4450867052023122, 0.4931880108991825, 0.567741935483871, 0.6324786324786326, 0.679814385150812, 0.7033707865168538] 
	t10 = [0, 10.140229940414429, 20.12906837463379, 30.120646953582764, 40.1150803565979, 50.11060428619385, 60.04435110092163] 
	q10 = [0.36419753086419754, 0.45272206303724927, 0.5087483176312247, 0.5924050632911393, 0.6587395957193817, 0.6902857142857143, 0.7190265486725663] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
		
	t1 = [0, 1.145557165145874, 2.0167829990386963, 3.070876359939575, 4.095244407653809, 5.039327144622803, 6.054346323013306, 7.114323139190674, 8.128237247467041, 9.018351316452026, 10.064985752105713, 11.14625859260559, 12.000388622283936, 13.135489225387573, 14.135303258895874, 15.10997223854065, 16.040486335754395, 17.064093589782715, 18.001842975616455, 19.12989616394043, 20.09576153755188, 21.053289651870728, 22.026889324188232, 23.127421379089355, 24.05494737625122, 25.038881301879883, 26.030174255371094, 27.06503415107727, 28.059198141098022, 29.07521915435791, 30.044228076934814, 31.004374265670776, 32.10909628868103, 33.093196868896484, 34.02763271331787, 35.00947594642639, 36.085076093673706, 37.03457832336426, 38.12325406074524, 39.089613914489746, 40.06298303604126, 41.023584604263306, 42.02283501625061, 43.00399208068848, 44.00881552696228, 45.129900217056274, 46.06376910209656, 47.009850025177, 48.09582734107971, 49.080073595047, 50.06532311439514, 51.01998162269592, 52.00944495201111, 53.11290001869202, 54.04799437522888, 55.02340841293335, 56.016642570495605, 57.113085985183716, 58.05186343193054, 59.07329225540161, 60.03421139717102] 
	q1 = [0.34633385335413414, 0.3478260869565218, 0.35802469135802467, 0.36447166921898927, 0.36946564885496186, 0.3793626707132018, 0.3861236802413273, 0.39461883408071746, 0.40118870728083206, 0.41124260355029585, 0.41826215022091306, 0.4269005847953216, 0.430232558139535, 0.43352601156069365, 0.43678160919540227, 0.4457142857142857, 0.45014245014245013, 0.4604519774011299, 0.47124824684431976, 0.48189415041782724, 0.48753462603878117, 0.49103448275862066, 0.4979480164158688, 0.5101763907734056, 0.5175202156334232, 0.5301204819277109, 0.5352862849533955, 0.5411140583554377, 0.5488126649076517, 0.5590551181102362, 0.5677083333333334, 0.572538860103627, 0.577319587628866, 0.5838668373879642, 0.5903307888040713, 0.5984848484848485, 0.6082603254067585, 0.6134663341645885, 0.6212871287128713, 0.626387176325524, 0.6340269277845777, 0.6406820950060901, 0.6472727272727272, 0.6530120481927711, 0.6610778443113772, 0.6682520808561236, 0.6729857819905214, 0.6784452296819788, 0.687719298245614, 0.6923076923076924, 0.6968641114982579, 0.6998841251448435, 0.7028901734104046, 0.7065592635212888, 0.7080459770114942, 0.7087155963302753, 0.7123287671232876, 0.7144482366325371, 0.7188208616780045, 0.7217194570135747, 0.7223476297968396] 
	t2 = [0, 2.1159257888793945, 4.078946352005005, 6.093810558319092, 8.089213371276855, 10.125604629516602, 12.091827154159546, 14.082260370254517, 16.102725505828857, 18.050812005996704, 20.040732860565186, 22.05283236503601, 24.04650902748108, 26.1421537399292, 28.10599994659424, 30.12622880935669, 32.13990497589111, 34.000248670578, 36.10694766044617, 38.10371923446655, 40.02092146873474, 42.082624435424805, 44.08614158630371, 46.05184555053711, 48.125869035720825, 50.01502251625061, 52.119728088378906, 54.04425001144409, 56.01954436302185, 58.078699827194214, 60.11314392089844] 
	q2 = [0.34633385335413414, 0.36251920122887865, 0.37689969604863227, 0.39461883408071746, 0.413589364844904, 0.4308588064046579, 0.4357864357864358, 0.45014245014245013, 0.4669479606188467, 0.4854368932038835, 0.49657064471879286, 0.5182186234817814, 0.5352862849533955, 0.5514511873350924, 0.5677083333333334, 0.577319587628866, 0.5913705583756345, 0.6082603254067585, 0.6229913473423981, 0.6356968215158924, 0.648910411622276, 0.6634844868735084, 0.6768867924528302, 0.6892523364485981, 0.6998841251448435, 0.707373271889401, 0.7087155963302753, 0.7144482366325371, 0.7202718006795016, 0.7266591676040495, 0.7307262569832403] 
	t3 = [0, 3.0270166397094727, 6.038617849349976, 9.047117471694946, 12.122978448867798, 15.139983415603638, 18.04870867729187, 21.136324405670166, 24.01557469367981, 27.06640386581421, 30.037589073181152, 33.092204332351685, 36.144267559051514, 39.030481576919556, 42.11472535133362, 45.09366011619568, 48.108840227127075, 51.092257499694824, 54.14133095741272, 57.07763338088989, 60.13263559341431] 
	q3 = [0.34633385335413414, 0.36447166921898927, 0.39461883408071746, 0.4252199413489737, 0.4380403458213256, 0.4582743988684582, 0.48753462603878117, 0.5128900949796472, 0.5352862849533955, 0.56282722513089, 0.5791505791505791, 0.6047678795483061, 0.6246913580246913, 0.6472727272727272, 0.669833729216152, 0.687719298245614, 0.6998841251448435, 0.7080459770114942, 0.7167235494880546, 0.7229729729729729, 0.73355629877369] 
	t4 = [0, 4.053027153015137, 8.003569841384888, 12.111596822738647, 16.11741328239441, 20.12563443183899, 24.087968587875366, 28.023900270462036, 32.06928253173828, 36.012227058410645, 40.09925889968872, 44.14327096939087, 48.060954093933105, 52.10019135475159, 56.096956968307495, 60.11146521568298] 
	q4 = [0.34633385335413414, 0.37689969604863227, 0.413589364844904, 0.4380403458213256, 0.4691011235955057, 0.5027322404371585, 0.5372340425531915, 0.5714285714285715, 0.5974683544303798, 0.6246913580246913, 0.654632972322503, 0.6830985915492958, 0.6983758700696056, 0.711670480549199, 0.7209039548022599, 0.7313266443701227] 
	t5 = [0, 5.085408926010132, 10.004542112350464, 15.06087851524353, 20.00646162033081, 25.033860683441162, 30.091560125350952, 35.11940670013428, 40.135600328445435, 45.12726306915283, 50.143064737319946, 55.073288679122925, 60.05191087722778] 
	q5 = [0.34633385335413414, 0.3831070889894419, 0.4308588064046579, 0.45957446808510644, 0.5006839945280437, 0.5444887118193892, 0.5809768637532133, 0.6212871287128713, 0.654632972322503, 0.6892523364485981, 0.7065592635212888, 0.7210884353741497, 0.7321428571428571] 
	t6 = [0, 6.126628637313843, 12.133405685424805, 18.090950965881348, 24.047316312789917, 30.04759693145752, 36.05911183357239, 42.08700394630432, 48.09595227241516, 54.02335548400879, 60.12561225891113] 
	q6 = [0.34633385335413414, 0.39461883408071746, 0.4380403458213256, 0.48753462603878117, 0.5391766268260293, 0.5853658536585366, 0.6280788177339901, 0.669833729216152, 0.6998841251448435, 0.7196367763904653, 0.732739420935412] 
	t7 = [0, 7.053351163864136, 14.09169626235962, 21.13970947265625, 28.08936858177185, 35.111246824264526, 42.13700008392334, 49.01611375808716, 56.00352501869202] 
	q7 = [0.34633385335413414, 0.4035608308605341, 0.4536376604850214, 0.516914749661705, 0.5740259740259741, 0.6220570012391574, 0.6714116251482799, 0.7035755478662054, 0.7209039548022599] 
	t8 = [0, 8.08163046836853, 16.09730887413025, 24.07810115814209, 32.02726769447327, 40.10657262802124, 48.05740284919739, 56.1164391040802] 
	q8 = [0.34633385335413414, 0.413589364844904, 0.47042253521126765, 0.5398936170212767, 0.5992414664981036, 0.6562499999999999, 0.6983758700696056, 0.7209039548022599] 
	t9 = [0, 9.10825777053833, 18.06796908378601, 27.08391547203064, 36.03549361228943, 45.04699349403381, 54.09232974052429] 
	q9 = [0.34633385335413414, 0.424597364568082, 0.4888888888888889, 0.5673202614379086, 0.6280788177339901, 0.6892523364485981, 0.7210884353741497] 
	t10 = [0, 10.12211561203003, 20.07104229927063, 30.090749740600586, 40.12814259529114, 50.110151529312134, 60.13711476325989] 
	q10 = [0.34633385335413414, 0.430232558139535, 0.505464480874317, 0.5879332477535303, 0.6594724220623501, 0.7050691244239632, 0.7321428571428571] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.0982134342193604, 2.0271573066711426, 3.0070149898529053, 4.099928140640259, 5.049942255020142, 6.12503981590271, 7.073804140090942, 8.03342890739441, 9.07119083404541, 10.005188703536987, 11.098406553268433, 12.056205987930298, 13.03326416015625, 14.112760543823242, 15.059711456298828, 16.020415782928467, 17.14259696006775, 18.123124361038208, 19.06395673751831, 20.026848077774048, 21.03738236427307, 22.13356876373291, 23.110565662384033, 24.08210277557373, 25.057913541793823, 26.01348042488098, 27.001391887664795, 28.082525730133057, 29.030044555664062, 30.13262915611267, 31.103212118148804, 32.02882671356201, 33.1457622051239, 34.11554574966431, 35.05463480949402, 36.12956643104553, 37.14094567298889, 38.101282596588135, 39.07185339927673, 40.05462312698364, 41.146098136901855, 42.098687171936035, 43.03933882713318, 44.11916923522949, 45.093573331832886, 46.023224115371704, 47.12252712249756, 48.07660937309265, 49.05007076263428, 50.00642132759094, 51.023642778396606, 52.12154674530029, 53.09384298324585, 54.03245162963867, 55.00962233543396, 56.11182236671448, 57.12128949165344, 58.07777547836304, 59.024311542510986, 60.09576725959778] 
	q1 = [0.37181409295352325, 0.37797619047619047, 0.3846153846153846, 0.3929618768328446, 0.4011627906976744, 0.40810419681620835, 0.4195402298850574, 0.4234620886981402, 0.43304843304843305, 0.4428772919605077, 0.45314685314685316, 0.45769764216366154, 0.46344827586206894, 0.46849315068493147, 0.48303934871099047, 0.490566037735849, 0.49329758713136723, 0.49866666666666665, 0.5059602649006623, 0.5157894736842105, 0.5190039318479684, 0.5260416666666666, 0.5388601036269429, 0.5463917525773196, 0.5549872122762147, 0.5623409669211196, 0.5670886075949367, 0.5753768844221105, 0.5853051058530511, 0.594059405940594, 0.6002460024600246, 0.6063569682151588, 0.6114494518879415, 0.619105199516324, 0.625, 0.629940119760479, 0.6348448687350835, 0.6389548693586697, 0.6422668240850059, 0.6447058823529411, 0.6510538641686183, 0.6565774155995343, 0.6635838150289017, 0.6697353279631761, 0.6765714285714285, 0.6810933940774487, 0.6848072562358276, 0.6892655367231639, 0.6914414414414415, 0.6958473625140291, 0.6973094170403588, 0.6972067039106146, 0.6985539488320357, 0.6984478935698448, 0.6991150442477876, 0.6983425414364641, 0.701212789415656, 0.701098901098901, 0.7039473684210528, 0.7039473684210528, 0.7039473684210528] 
	t2 = [0, 2.1234614849090576, 4.134967565536499, 6.101630926132202, 8.096796035766602, 10.040756940841675, 12.00066614151001, 14.032771348953247, 16.00125503540039, 18.053857803344727, 20.10259199142456, 22.021680116653442, 24.025604963302612, 26.056764841079712, 28.10480546951294, 30.06917667388916, 32.08786940574646, 34.11571478843689, 36.086474895477295, 38.00291085243225, 40.09352135658264, 42.12463307380676, 44.12114238739014, 46.120280265808105, 48.01631546020508, 50.02686810493469, 52.09063720703125, 54.130378007888794, 56.00871801376343, 58.067572832107544, 60.04050397872925] 
	q2 = [0.37181409295352325, 0.3870014771048744, 0.4034833091436865, 0.4195402298850574, 0.4375, 0.45746164574616455, 0.46556473829201106, 0.48443843031123135, 0.49732620320855614, 0.5099075297225892, 0.5254901960784314, 0.544516129032258, 0.5586734693877551, 0.5735849056603773, 0.594059405940594, 0.6070991432068543, 0.6198547215496367, 0.629940119760479, 0.6389548693586697, 0.6447058823529411, 0.6565774155995343, 0.6712643678160919, 0.6810933940774487, 0.6892655367231639, 0.6958473625140291, 0.6964285714285713, 0.6984478935698448, 0.6997792494481236, 0.701098901098901, 0.7039473684210528, 0.7066521264994546] 
	t3 = [0, 3.0206782817840576, 6.056391477584839, 9.029143810272217, 12.020671367645264, 15.101492166519165, 18.121720790863037, 21.1143159866333, 24.07135796546936, 27.048752784729004, 30.0882568359375, 33.000900745391846, 36.07956147193909, 39.097012758255005, 42.03049612045288, 45.057536363601685, 48.088013887405396, 51.058796882629395, 54.00988817214966, 57.12224578857422, 60.008975982666016] 
	q3 = [0.37181409295352325, 0.39238653001464124, 0.4195402298850574, 0.44788732394366193, 0.4676753782668501, 0.49193548387096775, 0.5138339920948617, 0.5376623376623376, 0.5605095541401274, 0.5853051058530511, 0.6063569682151588, 0.6274038461538463, 0.6405693950177936, 0.6534422403733955, 0.6735395189003436, 0.6892655367231639, 0.6973094170403588, 0.696329254727475, 0.701212789415656, 0.7039473684210528, 0.7080610021786493] 
	t4 = [0, 4.051710605621338, 8.117942571640015, 12.120754957199097, 16.093992948532104, 20.07853412628174, 24.126720905303955, 28.115028619766235, 32.10870575904846, 36.05734705924988, 40.12150740623474, 44.100762605667114, 48.01778745651245, 52.032161235809326, 56.03925943374634, 60.04773163795471] 
	q4 = [0.37181409295352325, 0.4011627906976744, 0.43971631205673756, 0.4676753782668501, 0.49933244325767684, 0.5267275097783573, 0.5623409669211196, 0.5933250927070457, 0.6231884057971014, 0.6421800947867299, 0.662037037037037, 0.6848072562358276, 0.6973094170403588, 0.6962305986696231, 0.7047200878155873, 0.7080610021786493] 
	t5 = [0, 5.073621988296509, 10.014706134796143, 15.109547853469849, 20.106592178344727, 25.098326206207275, 30.068031072616577, 35.05814456939697, 40.034183979034424, 45.05451250076294, 50.03579497337341, 55.04458951950073, 60.09641790390015] 
	q5 = [0.37181409295352325, 0.41040462427745666, 0.45810055865921784, 0.49395973154362416, 0.5286458333333333, 0.5689001264222504, 0.6063569682151588, 0.6357142857142858, 0.660486674391657, 0.6892655367231639, 0.6941964285714286, 0.7018701870187017, 0.7080610021786493] 
	t6 = [0, 6.107774019241333, 12.121619939804077, 18.10396957397461, 24.06871795654297, 30.127647638320923, 36.060991525650024, 42.039698362350464, 48.13990616798401, 54.125638008117676, 60.045559883117676] 
	q6 = [0.37181409295352325, 0.4195402298850574, 0.47107438016528924, 0.5164690382081687, 0.564885496183206, 0.608058608058608, 0.6437869822485207, 0.6765714285714285, 0.6973094170403588, 0.701212789415656, 0.7080610021786493] 
	t7 = [0, 7.070720434188843, 14.008076429367065, 21.099876165390015, 28.054440021514893, 35.06156802177429, 42.10144782066345, 49.083322286605835, 56.08134579658508] 
	q7 = [0.37181409295352325, 0.4234620886981402, 0.489851150202977, 0.5421530479896238, 0.5933250927070457, 0.6364719904648392, 0.6780821917808219, 0.6949720670391062, 0.7054945054945054] 
	t8 = [0, 8.060140371322632, 16.00146174430847, 24.046562433242798, 32.14047455787659, 40.054476261138916, 48.01254987716675, 56.07000136375427] 
	q8 = [0.37181409295352325, 0.43971631205673756, 0.5, 0.564885496183206, 0.6224366706875754, 0.662037037037037, 0.6973094170403588, 0.7040704070407041] 
	t9 = [0, 9.08663296699524, 18.023170948028564, 27.044793128967285, 36.075560331344604, 45.01198148727417, 54.10168242454529] 
	q9 = [0.37181409295352325, 0.45007032348804504, 0.5171503957783641, 0.5895522388059701, 0.6437869822485207, 0.6892655367231639, 0.6997792494481236] 
	t10 = [0, 10.10966944694519, 20.082568407058716, 30.116431713104248, 40.004812479019165, 50.06934332847595, 60.07201528549194] 
	q10 = [0.37181409295352325, 0.45746164574616455, 0.53125, 0.6097560975609756, 0.6635838150289017, 0.6948775055679287, 0.710239651416122] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.1001362800598145, 2.0257747173309326, 3.1155920028686523, 4.046116590499878, 5.140592098236084, 6.0668439865112305, 7.0228111743927, 8.095128059387207, 9.068950414657593, 10.034108877182007, 11.013028144836426, 12.034619331359863, 13.122985363006592, 14.053006887435913, 15.14228343963623, 16.070087909698486, 17.047923803329468, 18.125110626220703, 19.075464725494385, 20.048821449279785, 21.12105631828308, 22.10868525505066, 23.002615213394165, 24.09656810760498, 25.10692310333252, 26.063931465148926, 27.09914541244507, 28.05281352996826, 29.043031454086304, 30.127859354019165, 31.07285761833191, 32.00256562232971, 33.08895826339722, 34.015806913375854, 35.14701962471008, 36.10264873504639, 37.056440591812134, 38.01314330101013, 39.13337802886963, 40.097792625427246, 41.04257369041443, 42.11353874206543, 43.0881290435791, 44.05413317680359, 45.033711194992065, 46.09653830528259, 47.07114315032959, 48.0214478969574, 49.124330043792725, 50.10848784446716, 51.056782722473145, 52.124613761901855, 53.129966735839844, 54.08676266670227, 55.026957750320435, 56.09111976623535, 57.09119987487793, 58.01413416862488, 59.13751244544983, 60.11647987365723] 
	q1 = [0.3758169934640523, 0.3766233766233766, 0.38647342995169087, 0.4044585987261146, 0.4088748019017433, 0.42006269592476486, 0.43167701863354035, 0.4382716049382716, 0.445468509984639, 0.45259938837920494, 0.4589665653495441, 0.46827794561933545, 0.4730538922155689, 0.47774480712166173, 0.48153618906942397, 0.4897360703812316, 0.49635036496350365, 0.49782923299565845, 0.5, 0.5064377682403434, 0.5149359886201992, 0.5225988700564972, 0.5260196905766527, 0.5322128851540616, 0.5355648535564853, 0.5436893203883495, 0.5517241379310345, 0.5616438356164384, 0.5694822888283378, 0.5745257452574526, 0.5879194630872483, 0.5935828877005348, 0.602921646746348, 0.6103038309114927, 0.6149802890932983, 0.6223958333333334, 0.6304909560723514, 0.6367137355584083, 0.6454081632653061, 0.648854961832061, 0.6522842639593909, 0.6574307304785894, 0.6616541353383459, 0.6674968866749689, 0.6707920792079207, 0.6757090012330457, 0.6781326781326782, 0.6805385556915544, 0.6877278250303767, 0.6893203883495145, 0.694074969770254, 0.6987951807228916, 0.701923076923077, 0.704326923076923, 0.7050359712230215, 0.7048984468339308, 0.7095238095238096, 0.7131050767414403, 0.7137809187279153, 0.7144535840188014, 0.7151230949589684] 
	t2 = [0, 2.138166666030884, 4.134227514266968, 6.111835956573486, 8.11059284210205, 10.00384521484375, 12.071480512619019, 14.108927726745605, 16.087451934814453, 18.093556880950928, 20.06174921989441, 22.120155811309814, 24.036341667175293, 26.078498601913452, 28.043982982635498, 30.048627376556396, 32.02179312705994, 34.138437271118164, 36.0263671875, 38.028295278549194, 40.053093910217285, 42.05509042739868, 44.066654443740845, 46.0929217338562, 48.132158517837524, 50.00965166091919, 52.06183457374573, 54.06657648086548, 56.068347692489624, 58.10250902175903, 60.151220083236694] 
	q2 = [0.3758169934640523, 0.3890675241157556, 0.4113924050632911, 0.434108527131783, 0.44785276073619634, 0.4620060790273556, 0.47832585949177875, 0.4837758112094395, 0.4956268221574344, 0.5, 0.5163120567375886, 0.5329593267882188, 0.5436893203883495, 0.5576923076923077, 0.5745257452574526, 0.5909090909090908, 0.6084656084656085, 0.623207301173403, 0.6367137355584083, 0.648854961832061, 0.6574307304785894, 0.6691542288557214, 0.6765067650676506, 0.6821515892420538, 0.6909090909090909, 0.6987951807228916, 0.7058823529411765, 0.7064439140811457, 0.7131050767414403, 0.7159624413145539, 0.7217694994179279] 
	t3 = [0, 3.0234758853912354, 6.023982763290405, 9.03132176399231, 12.120656251907349, 15.113349676132202, 18.118199586868286, 21.023839712142944, 24.144083976745605, 27.015464067459106, 30.023484230041504, 33.036829710006714, 36.07786202430725, 39.080613136291504, 42.03210735321045, 45.02494478225708, 48.040480852127075, 51.12202477455139, 54.04757642745972, 57.098737716674805, 60.0611686706543] 
	q3 = [0.3758169934640523, 0.4019138755980861, 0.434108527131783, 0.4542682926829269, 0.48059701492537316, 0.4941520467836257, 0.5050215208034433, 0.5260196905766527, 0.5464632454923717, 0.5694822888283378, 0.595460614152203, 0.6186107470511141, 0.6402048655569783, 0.652338811630847, 0.6708074534161491, 0.6773006134969324, 0.6933333333333334, 0.704326923076923, 0.7095238095238096, 0.7144535840188014, 0.7232558139534884] 
	t4 = [0, 4.058889865875244, 8.120739459991455, 12.071294069290161, 16.118106603622437, 20.057714700698853, 24.051478624343872, 28.139758110046387, 32.05834364891052, 36.13153696060181, 40.01208019256592, 44.115389585494995, 48.09235429763794, 52.084632396698, 56.07250761985779, 60.053340673446655] 
	q4 = [0.3758169934640523, 0.4113924050632911, 0.44785276073619634, 0.48059701492537316, 0.4970930232558139, 0.5205091937765204, 0.5464632454923717, 0.5802968960863698, 0.6113306982872201, 0.6427656850192062, 0.6608040201005025, 0.6781326781326782, 0.6933333333333334, 0.7041916167664671, 0.7154663518299881, 0.7247386759581881] 
	t5 = [0, 5.100014925003052, 10.037231683731079, 15.12788200378418, 20.09188222885132, 25.011066436767578, 30.058258056640625, 35.04966449737549, 40.09329533576965, 45.00634026527405, 50.06484365463257, 55.005627155303955, 60.034812211990356] 
	q5 = [0.3758169934640523, 0.41940532081377147, 0.464339908952959, 0.49707602339181284, 0.5233380480905233, 0.5544827586206896, 0.5973333333333334, 0.6357786357786358, 0.6599749058971142, 0.6805385556915544, 0.700361010830325, 0.7139479905437353, 0.7247386759581881] 
	t6 = [0, 6.131175756454468, 12.055875539779663, 18.051613569259644, 24.076584815979004, 30.021467208862305, 36.09533166885376, 42.01819086074829, 48.14582347869873, 54.103989601135254, 60.06463384628296] 
	q6 = [0.3758169934640523, 0.43167701863354035, 0.4798807749627422, 0.5071633237822349, 0.5484764542936288, 0.5992010652463383, 0.6445012787723785, 0.6707920792079207, 0.6949152542372881, 0.7117437722419929, 0.7247386759581881] 
	t7 = [0, 7.041127681732178, 14.093735933303833, 21.10298490524292, 28.142921686172485, 35.126272201538086, 42.060739517211914, 49.06997895240784, 56.07425498962402] 
	q7 = [0.3758169934640523, 0.4382716049382716, 0.4889543446244477, 0.5288326300984529, 0.582210242587601, 0.6357786357786358, 0.6724351050679853, 0.6980676328502415, 0.7161366313309777] 
	t8 = [0, 8.0776047706604, 16.00696063041687, 24.07401466369629, 32.10411286354065, 40.008439779281616, 48.01470232009888, 56.089744329452515] 
	q8 = [0.3758169934640523, 0.44785276073619634, 0.5007278020378457, 0.5484764542936288, 0.613157894736842, 0.6616541353383459, 0.6964933494558646, 0.7161366313309777] 
	t9 = [0, 9.063295364379883, 18.037996292114258, 27.11362051963806, 36.07821440696716, 45.13530468940735, 54.089356422424316] 
	q9 = [0.3758169934640523, 0.4542682926829269, 0.5078909612625538, 0.5733695652173914, 0.6462324393358876, 0.6837606837606838, 0.7117437722419929] 
	t10 = [0, 10.10849928855896, 20.08853316307068, 30.03764033317566, 40.04952549934387, 50.056931495666504, 60.14640235900879] 
	q10 = [0.3758169934640523, 0.4666666666666667, 0.5261669024045261, 0.6018641810918774, 0.6641604010025063, 0.7012048192771085, 0.727061556329849] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.1036937236785889, 2.0421485900878906, 3.1463005542755127, 4.070799350738525, 5.0141212940216064, 6.082498550415039, 7.041187763214111, 8.143524169921875, 9.125368118286133, 10.085347175598145, 11.032130479812622, 12.106890678405762, 13.053857564926147, 14.04327392578125, 15.046581506729126, 16.002153396606445, 17.101946592330933, 18.036097764968872, 19.051058769226074, 20.1235671043396, 21.09740138053894, 22.05537223815918, 23.04268455505371, 24.108784437179565, 25.08716583251953, 26.038914680480957, 27.138009309768677, 28.130600452423096, 29.10536766052246, 30.0871684551239, 31.05995774269104, 32.047144174575806, 33.1333384513855, 34.0896213054657, 35.03307843208313, 36.134894371032715, 37.09963297843933, 38.030702114105225, 39.12320518493652, 40.04478931427002, 41.129491329193115, 42.080036878585815, 43.03163170814514, 44.114206075668335, 45.0665020942688, 46.075607776641846, 47.05487394332886, 48.131059408187866, 49.10225248336792, 50.02528166770935, 51.139819622039795, 52.05547642707825, 53.05552840232849, 54.11835765838623, 55.128966331481934, 56.04790425300598, 57.002782106399536, 58.06781005859375, 59.04686713218689, 60.144492864608765] 
	q1 = [0.3540372670807453, 0.36419753086419754, 0.37366003062787134, 0.38543247344461307, 0.39457831325301207, 0.4011976047904192, 0.4154302670623146, 0.4194977843426883, 0.4304538799414348, 0.4366812227074236, 0.4444444444444444, 0.45114942528735624, 0.4564907275320971, 0.4680851063829787, 0.47457627118644063, 0.4775280898876405, 0.48189415041782724, 0.48962655601659744, 0.4896836313617607, 0.4924760601915184, 0.4959128065395095, 0.5060893098782139, 0.5134408602150538, 0.5206942590120159, 0.5251989389920425, 0.531578947368421, 0.538562091503268, 0.5468749999999999, 0.5536869340232858, 0.5611325611325612, 0.5710627400768247, 0.5790816326530612, 0.5822784810126583, 0.5886792452830188, 0.593984962406015, 0.6, 0.6096654275092938, 0.6165228113440198, 0.624235006119951, 0.6309378806333739, 0.639225181598063, 0.6393244873341375, 0.6474820143884892, 0.65, 0.6548463356973996, 0.6619552414605419, 0.6690058479532164, 0.6767441860465115, 0.6843930635838151, 0.6904487917146145, 0.6926605504587156, 0.6978335233751426, 0.7007963594994312, 0.7037457434733259, 0.7089467723669308, 0.7089467723669308, 0.7096045197740112, 0.708803611738149, 0.7109111361079865, 0.7152466367713005, 0.7171492204899778] 
	t2 = [0, 2.1277592182159424, 4.124479532241821, 6.11326789855957, 8.11266016960144, 10.140102863311768, 12.147414922714233, 14.006836414337158, 16.101924657821655, 18.09603261947632, 20.01952052116394, 22.03916835784912, 24.0695858001709, 26.11408805847168, 28.093900442123413, 30.040034770965576, 32.000590801239014, 34.02296686172485, 36.03435397148132, 38.012572288513184, 40.00057935714722, 42.003469944000244, 44.12698554992676, 46.03630495071411, 48.02866983413696, 50.039350509643555, 52.07472562789917, 54.120752811431885, 56.017640352249146, 58.00991940498352, 60.01560735702515] 
	q2 = [0.3540372670807453, 0.37308868501529047, 0.39457831325301207, 0.4154302670623146, 0.4304538799414348, 0.4473304473304473, 0.4615384615384615, 0.47672778561354023, 0.48611111111111105, 0.49108367626886146, 0.5, 0.5194109772423026, 0.5264550264550264, 0.5449804432855281, 0.5592783505154639, 0.5761843790012804, 0.587641866330391, 0.6, 0.6165228113440198, 0.6292682926829269, 0.6393244873341375, 0.6507747318235997, 0.6619552414605419, 0.6767441860465115, 0.6904487917146145, 0.6986301369863013, 0.7022727272727273, 0.7089467723669308, 0.708803611738149, 0.7144456886898096, 0.7228381374722839] 
	t3 = [0, 3.025162935256958, 6.037459373474121, 9.041959762573242, 12.148163795471191, 15.068365335464478, 18.036813259124756, 21.034237146377563, 24.12300705909729, 27.09169864654541, 30.05574321746826, 33.04982256889343, 36.12638545036316, 39.053731203079224, 42.03224015235901, 45.085891246795654, 48.053919553756714, 51.11542224884033, 54.05753207206726, 57.09004735946655, 60.04073643684387] 
	q3 = [0.3540372670807453, 0.3829787234042553, 0.4154302670623146, 0.43831640058055155, 0.4637268847795163, 0.4832402234636871, 0.49108367626886146, 0.5094339622641509, 0.5303430079155673, 0.5544041450777202, 0.5798212005108557, 0.593984962406015, 0.6182266009852216, 0.639225181598063, 0.653206650831354, 0.6705607476635514, 0.6911595866819749, 0.7015945330296127, 0.7104072398190044, 0.7138047138047138, 0.7228381374722839] 
	t4 = [0, 4.0291969776153564, 8.111737966537476, 12.13054084777832, 16.113022089004517, 20.012109994888306, 24.14451813697815, 28.035300970077515, 32.08951139450073, 36.1209192276001, 40.06881380081177, 44.13494563102722, 48.14090895652771, 52.14700698852539, 56.135873556137085, 60.11486339569092] 
	q4 = [0.3540372670807453, 0.39457831325301207, 0.4298245614035088, 0.46438746438746437, 0.48821081830790564, 0.5, 0.5322793148880105, 0.5611325611325612, 0.5901639344262295, 0.6199261992619927, 0.6425992779783393, 0.664319248826291, 0.6926605504587156, 0.7052154195011338, 0.7109111361079865, 0.7250554323725056] 
	t5 = [0, 5.082035779953003, 10.03375506401062, 15.081164121627808, 20.06731343269348, 25.032811641693115, 30.095516443252563, 35.01844263076782, 40.07487607002258, 45.06933307647705, 50.108596324920654, 55.049222469329834, 60.050447940826416] 
	q5 = [0.3540372670807453, 0.4035874439461884, 0.44956772334293943, 0.4825662482566248, 0.5020352781546811, 0.5392670157068062, 0.5834394904458599, 0.6113861386138614, 0.6425992779783393, 0.6721120186697783, 0.7001140250855188, 0.708803611738149, 0.7250554323725056] 
	t6 = [0, 6.019402027130127, 12.129953622817993, 18.0791494846344, 24.02921199798584, 30.003316640853882, 36.143982887268066, 42.08425807952881, 48.01435899734497, 54.06137299537659, 60.03826713562012] 
	q6 = [0.3540372670807453, 0.4154302670623146, 0.4665718349928877, 0.49108367626886146, 0.5303430079155673, 0.5816326530612245, 0.6199261992619927, 0.6524317912218268, 0.6896551724137931, 0.7081447963800904, 0.7250554323725056] 
	t7 = [0, 7.0436341762542725, 14.019613265991211, 21.069591522216797, 28.106915950775146, 35.076765298843384, 42.08849811553955, 49.105310916900635, 56.1301212310791] 
	q7 = [0.3540372670807453, 0.4188790560471976, 0.476056338028169, 0.5114401076716015, 0.5655526992287918, 0.6163366336633663, 0.6532544378698225, 0.6971428571428571, 0.7109111361079865] 
	t8 = [0, 8.063727617263794, 16.018582344055176, 24.105071544647217, 32.07906699180603, 40.03816866874695, 48.06624627113342, 56.08234691619873] 
	q8 = [0.3540372670807453, 0.4298245614035088, 0.4888888888888889, 0.5322793148880105, 0.5944584382871536, 0.6442307692307693, 0.6926605504587156, 0.7094594594594593] 
	t9 = [0, 9.080902576446533, 18.079935550689697, 27.168427228927612, 36.09164524078369, 45.09276723861694, 54.138832569122314] 
	q9 = [0.3540372670807453, 0.43831640058055155, 0.49108367626886146, 0.5599999999999999, 0.6248462484624846, 0.6767441860465115, 0.7089467723669308] 
	t10 = [0, 10.11781620979309, 20.055987119674683, 30.108210563659668, 40.12049055099487, 50.06593942642212, 60.00060296058655] 
	q10 = [0.3540372670807453, 0.4473304473304473, 0.5040650406504065, 0.5852417302798982, 0.6474820143884892, 0.7001140250855188, 0.726467331118494] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.1030092239379883, 2.063704252243042, 3.0482475757598877, 4.123750925064087, 5.082269906997681, 6.017186641693115, 7.145286560058594, 8.101431608200073, 9.140265941619873, 10.067831754684448, 11.079705715179443, 12.013649225234985, 13.10286545753479, 14.058751583099365, 15.040968656539917, 16.10922074317932, 17.090441465377808, 18.054452657699585, 19.039921283721924, 20.022618055343628, 21.041590213775635, 22.122039079666138, 23.134217023849487, 24.063437700271606, 25.016008377075195, 26.08121633529663, 27.0536892414093, 28.017062187194824, 29.019588470458984, 30.116769790649414, 31.059498071670532, 32.124194860458374, 33.095258951187134, 34.0118408203125, 35.13607692718506, 36.07550072669983, 37.01475667953491, 38.129560708999634, 39.10048317909241, 40.01898264884949, 41.13329768180847, 42.05424761772156, 43.0554473400116, 44.13135552406311, 45.11852312088013, 46.077088356018066, 47.023446559906006, 48.131181478500366, 49.08042335510254, 50.063321590423584, 51.0069625377655, 52.13643550872803, 53.10966348648071, 54.048784494400024, 55.01206564903259, 56.08349418640137, 57.05999255180359, 58.01492404937744, 59.14613223075867, 60.09772539138794] 
	q1 = [0.3723076923076923, 0.3822629969418961, 0.3860182370820669, 0.3981900452488688, 0.40657698056801195, 0.41777777777777775, 0.4264705882352941, 0.4333821376281113, 0.4402332361516034, 0.4434782608695652, 0.4502164502164502, 0.4606580829756795, 0.471590909090909, 0.4745762711864407, 0.47752808988764045, 0.48603351955307256, 0.49722222222222223, 0.5013850415512465, 0.5082417582417582, 0.5163934426229508, 0.5217391304347827, 0.5277401894451963, 0.5349462365591399, 0.5401069518716577, 0.5444887118193891, 0.554089709762533, 0.5654450261780105, 0.5721716514954487, 0.5777202072538861, 0.5861182519280206, 0.5925925925925927, 0.5989847715736041, 0.6120906801007556, 0.619047619047619, 0.6217228464419476, 0.630407911001236, 0.638036809815951, 0.6430317848410758, 0.6455542021924482, 0.6504854368932039, 0.6553808948004837, 0.6618705035971223, 0.6682577565632458, 0.6729857819905213, 0.673733804475854, 0.6853801169590643, 0.68997668997669, 0.6960556844547564, 0.7020785219399539, 0.7057471264367816, 0.7100456621004566, 0.7129840546697038, 0.7150964812712827, 0.7165532879818595, 0.7209039548022599, 0.7235955056179774, 0.7250280583613916, 0.7262569832402234, 0.7290969899665553, 0.7347391786903441, 0.7367256637168141] 
	t2 = [0, 2.150472402572632, 4.028414964675903, 6.004720211029053, 8.015043497085571, 10.115280151367188, 12.120786428451538, 14.115406274795532, 16.137782096862793, 18.135822057724, 20.106375694274902, 22.130178689956665, 24.013665199279785, 26.128973484039307, 28.120365142822266, 30.085080862045288, 32.04311203956604, 34.052002906799316, 36.0557804107666, 38.04646039009094, 40.050944328308105, 42.047563314437866, 44.07305359840393, 46.10812497138977, 48.1051139831543, 50.01018738746643, 52.05086922645569, 54.053693771362305, 56.022727727890015, 58.02956509590149, 60.130239725112915] 
	q2 = [0.3723076923076923, 0.3884673748103187, 0.40657698056801195, 0.4264705882352941, 0.4402332361516034, 0.4546762589928058, 0.4724186704384724, 0.482468443197756, 0.49930651872399445, 0.5157750342935529, 0.5257452574525745, 0.538152610441767, 0.5502645502645502, 0.5729166666666666, 0.5868725868725869, 0.5972045743329097, 0.617314930991217, 0.6294919454770755, 0.6430317848410758, 0.6504854368932039, 0.6618705035971223, 0.6745562130177515, 0.6838407494145199, 0.699074074074074, 0.7064220183486238, 0.7121729237770194, 0.7202718006795016, 0.7250280583613916, 0.7262569832402234, 0.7333333333333335, 0.7378854625550662] 
	t3 = [0, 3.0224995613098145, 6.099388360977173, 9.025258302688599, 12.01068639755249, 15.111251592636108, 18.02826499938965, 21.028308629989624, 24.021811723709106, 27.027323722839355, 30.084917306900024, 33.11915826797485, 36.04474854469299, 39.09536051750183, 42.14131808280945, 45.078678369522095, 48.00451850891113, 51.094791650772095, 54.03166389465332, 57.02432608604431, 60.015491247177124] 
	q3 = [0.3723076923076923, 0.3981900452488688, 0.4264705882352941, 0.4486251808972504, 0.4724186704384724, 0.49303621169916434, 0.5150684931506849, 0.5336927223719676, 0.5521796565389696, 0.5784695201037614, 0.5972045743329097, 0.6217228464419476, 0.6430317848410758, 0.6545893719806763, 0.6745562130177515, 0.6915017462165308, 0.7064220183486238, 0.7165532879818595, 0.726457399103139, 0.731924360400445, 0.7392739273927392] 
	t4 = [0, 4.0444440841674805, 8.129435300827026, 12.138908386230469, 16.13841462135315, 20.11960196495056, 24.018762588500977, 28.118215084075928, 32.020986557006836, 36.09224605560303, 40.055668115615845, 44.08931040763855, 48.052552700042725, 52.01267409324646, 56.14071869850159, 60.01145267486572] 
	q4 = [0.3723076923076923, 0.40895522388059696, 0.43959243085880634, 0.4724186704384724, 0.5013850415512465, 0.5277401894451963, 0.5559947299077733, 0.5905006418485237, 0.6182728410513142, 0.6446886446886447, 0.6650717703349283, 0.6884480746791132, 0.7108571428571427, 0.7200902934537247, 0.7305122494432071, 0.7406593406593408] 
	t5 = [0, 5.0669567584991455, 10.047478437423706, 15.007227897644043, 20.017988920211792, 25.029479265213013, 30.058659553527832, 35.030277252197266, 40.01895093917847, 45.099376916885376, 50.10574412345886, 55.064976930618286, 60.03094458580017] 
	q5 = [0.3723076923076923, 0.41949778434268836, 0.45689655172413796, 0.49513212795549383, 0.5284552845528456, 0.5673202614379085, 0.605830164765526, 0.6404907975460123, 0.6650717703349283, 0.6945412311265969, 0.7159090909090909, 0.7262569832402234, 0.7400881057268723] 
	t6 = [0, 6.097805023193359, 12.00475025177002, 18.129655361175537, 24.10369563102722, 30.107364892959595, 36.09259915351868, 42.130199670791626, 48.10068941116333, 54.04025149345398, 60.01785373687744] 
	q6 = [0.3723076923076923, 0.4287812041116006, 0.47308781869688393, 0.5170998632010944, 0.5578947368421052, 0.6065989847715736, 0.6446886446886447, 0.6745283018867925, 0.7108571428571427, 0.7250280583613916, 0.7414741474147414] 
	t7 = [0, 7.052329063415527, 14.040235757827759, 21.052995681762695, 28.052836179733276, 35.04178047180176, 42.00074481964111, 49.14455699920654, 56.13566279411316] 
	q7 = [0.3723076923076923, 0.4333821376281113, 0.484593837535014, 0.5329744279946164, 0.5905006418485237, 0.642156862745098, 0.6745283018867925, 0.7129840546697038, 0.731924360400445] 
	t8 = [0, 8.06373119354248, 16.007511615753174, 24.071488857269287, 32.022393465042114, 40.11180400848389, 48.021960973739624, 56.10156226158142] 
	q8 = [0.3723076923076923, 0.43731778425655976, 0.5020804438280166, 0.5559947299077733, 0.6207759699624531, 0.6682577565632458, 0.7093821510297483, 0.731924360400445] 
	t9 = [0, 9.138057947158813, 18.02794885635376, 27.144920825958252, 36.058006286621094, 45.0930540561676, 54.06155443191528] 
	q9 = [0.3723076923076923, 0.44637681159420295, 0.5178082191780822, 0.5839793281653747, 0.6446886446886447, 0.6960556844547564, 0.7272727272727273] 
	t10 = [0, 10.102262020111084, 20.038676261901855, 30.13820481300354, 40.0678277015686, 50.02318811416626, 60.10319185256958] 
	q10 = [0.3723076923076923, 0.4591104734576758, 0.5311653116531166, 0.610126582278481, 0.6690561529271207, 0.7159090909090909, 0.7400881057268723] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.127983570098877, 2.09053111076355, 3.044337034225464, 4.133875846862793, 5.088654518127441, 6.054192781448364, 7.008846998214722, 8.084167242050171, 9.034786939620972, 10.054585456848145, 11.028083324432373, 12.102537393569946, 13.088044166564941, 14.051056146621704, 15.143431186676025, 16.098470449447632, 17.042389392852783, 18.0047025680542, 19.10271191596985, 20.053636074066162, 21.008501052856445, 22.12075424194336, 23.081130266189575, 24.007304191589355, 25.041085243225098, 26.140921115875244, 27.036065340042114, 28.13520050048828, 29.095775842666626, 30.02222990989685, 31.106432914733887, 32.04249835014343, 33.02940917015076, 34.13755559921265, 35.107558488845825, 36.025333881378174, 37.141228675842285, 38.09767556190491, 39.04420256614685, 40.11772871017456, 41.091843605041504, 42.013683795928955, 43.04573392868042, 44.04503107070923, 45.1362042427063, 46.11585593223572, 47.05588102340698, 48.00792670249939, 49.00928497314453, 50.08400297164917, 51.02763605117798, 52.1260781288147, 53.102423429489136, 54.03173542022705, 55.130013942718506, 56.08111381530762, 57.069029808044434, 58.0178599357605, 59.10632252693176, 60.086450815200806] 
	q1 = [0.3385579937304075, 0.3483670295489891, 0.35802469135802467, 0.3619631901840491, 0.37442922374429227, 0.3812405446293495, 0.3903903903903904, 0.3970149253731343, 0.4065281899109793, 0.41124260355029585, 0.4222873900293256, 0.42794759825327516, 0.4370477568740955, 0.44316546762589926, 0.4507845934379458, 0.4589235127478754, 0.4641350210970464, 0.4671328671328671, 0.4658298465829846, 0.4736842105263158, 0.484181568088033, 0.4876712328767124, 0.49389416553595655, 0.5020242914979758, 0.510752688172043, 0.520694259012016, 0.5272969374167776, 0.5358090185676393, 0.5485564304461942, 0.5572916666666666, 0.565891472868217, 0.5750962772785623, 0.5805626598465472, 0.5888324873096445, 0.5959595959595959, 0.6047678795483062, 0.6117353308364545, 0.6127023661270237, 0.6205191594561187, 0.6282208588957056, 0.6324786324786325, 0.6366950182260025, 0.6424242424242426, 0.6481927710843374, 0.6555023923444977, 0.661904761904762, 0.6690307328605201, 0.6713780918727915, 0.6791569086651054, 0.6837209302325581, 0.6867749419953597, 0.6906141367323292, 0.6936416184971098, 0.6943483275663207, 0.695752009184845, 0.694954128440367, 0.6979405034324944, 0.6970387243735763, 0.6977272727272726, 0.705084745762712, 0.7094594594594595] 
	t2 = [0, 2.119194507598877, 4.127839803695679, 6.115299701690674, 8.148690938949585, 10.085306882858276, 12.148921966552734, 14.021759033203125, 16.044113159179688, 18.095803260803223, 20.097209215164185, 22.0673770904541, 24.067487955093384, 26.020296573638916, 28.11257791519165, 30.08832597732544, 32.06473731994629, 34.09242582321167, 36.098392724990845, 38.13139986991882, 40.107399463653564, 42.117276430130005, 44.08612513542175, 46.12070417404175, 48.02367448806763, 50.02621912956238, 52.02880620956421, 54.01546263694763, 56.02493929862976, 58.0898711681366, 60.12642693519592] 
	q2 = [0.3385579937304075, 0.35802469135802467, 0.3768996960486322, 0.39280359820089955, 0.4065281899109793, 0.424597364568082, 0.4370477568740955, 0.45584045584045585, 0.4641350210970464, 0.4700973574408901, 0.48559670781893005, 0.5006765899864682, 0.5167336010709505, 0.5338645418326693, 0.5535248041775458, 0.5732647814910025, 0.5877862595419847, 0.6030150753768844, 0.6127023661270237, 0.6282208588957056, 0.6366950182260025, 0.6457831325301205, 0.661904761904762, 0.6713780918727915, 0.6852497096399536, 0.6906141367323292, 0.6943483275663207, 0.6964490263459335, 0.6993166287015945, 0.705084745762712, 0.7109111361079864] 
	t3 = [0, 3.0263638496398926, 6.052764415740967, 9.06777024269104, 12.050041913986206, 15.005897998809814, 18.06861925125122, 21.014825582504272, 24.057015419006348, 27.051006317138672, 30.013750076293945, 33.009965896606445, 36.108182430267334, 39.06739139556885, 42.09634804725647, 45.122315645217896, 48.07561159133911, 51.14479899406433, 54.04135060310364, 57.1062228679657, 60.063403606414795] 
	q3 = [0.3385579937304075, 0.36447166921898927, 0.39280359820089955, 0.4153166421207658, 0.4370477568740955, 0.4611032531824611, 0.47434119278779474, 0.49318801089918257, 0.520694259012016, 0.5492772667542707, 0.5750962772785623, 0.5984848484848484, 0.6161490683229813, 0.6341463414634148, 0.6506602641056423, 0.6706021251475797, 0.6875725900116144, 0.6951501154734411, 0.6979405034324944, 0.705084745762712, 0.7166853303471444] 
	t4 = [0, 4.068777084350586, 8.044251203536987, 12.009530782699585, 16.037070989608765, 20.003596305847168, 24.11369824409485, 28.05394458770752, 32.02086615562439, 36.125585079193115, 40.093491554260254, 44.0425922870636, 48.094433546066284, 52.06624436378479, 56.02646040916443, 60.074283599853516] 
	q4 = [0.3385579937304075, 0.37442922374429227, 0.4065281899109793, 0.4370477568740955, 0.46844319775596066, 0.4876712328767124, 0.5213903743315508, 0.5598958333333333, 0.5931558935361216, 0.6178660049627792, 0.6424242424242426, 0.6658767772511848, 0.686046511627907, 0.695752009184845, 0.7000000000000001, 0.7166853303471444] 
	t5 = [0, 5.076750993728638, 10.062618255615234, 15.117804527282715, 20.148205757141113, 25.058228015899658, 30.035404682159424, 35.052016735076904, 40.098143339157104, 45.1015522480011, 50.125993728637695, 55.03433084487915, 60.04161596298218] 
	q5 = [0.3385579937304075, 0.38310708898944196, 0.4298245614035087, 0.4667609618104668, 0.4876712328767124, 0.5292553191489362, 0.5776636713735558, 0.6142322097378278, 0.6424242424242426, 0.6698113207547169, 0.6944444444444445, 0.6978335233751425, 0.7152466367713006] 
	t6 = [0, 6.102589130401611, 12.119754791259766, 18.11874294281006, 24.038565635681152, 30.079294681549072, 36.13208532333374, 42.09480357170105, 48.074111461639404, 54.03696870803833, 60.00298833847046] 
	q6 = [0.3385579937304075, 0.39280359820089955, 0.44219653179190754, 0.47434119278779474, 0.5213903743315508, 0.5776636713735558, 0.6212871287128713, 0.6555023923444977, 0.6875725900116144, 0.6986301369863014, 0.7166853303471444] 
	t7 = [0, 7.035925626754761, 14.093968629837036, 21.138636112213135, 28.131900548934937, 35.07059717178345, 42.05191159248352, 49.10951280593872, 56.13604164123535] 
	q7 = [0.3385579937304075, 0.39940387481371087, 0.46088193456614507, 0.4972826086956521, 0.5654993514915694, 0.6134663341645885, 0.6555023923444977, 0.691415313225058, 0.7029478458049886] 
	t8 = [0, 8.042822122573853, 16.101650714874268, 24.13732123374939, 32.119710206985474, 40.01584768295288, 48.069557189941406, 56.059210777282715] 
	q8 = [0.3385579937304075, 0.4065281899109793, 0.4691011235955056, 0.5233644859813084, 0.5974683544303797, 0.6424242424242426, 0.6898954703832753, 0.7029478458049886] 
	t9 = [0, 9.090478897094727, 18.021793127059937, 27.16399049758911, 36.06161594390869, 45.139002084732056, 54.08287262916565] 
	q9 = [0.3385579937304075, 0.41764705882352937, 0.4756606397774687, 0.5549738219895288, 0.6220570012391573, 0.6729411764705882, 0.6978335233751425] 
	t10 = [0, 10.137898206710815, 20.123607397079468, 30.106169939041138, 40.030160903930664, 50.11410641670227, 60.041616439819336] 
	q10 = [0.3385579937304075, 0.42627737226277373, 0.49175824175824173, 0.5794871794871795, 0.6424242424242426, 0.6944444444444445, 0.7174887892376681] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.0997211933135986, 2.0291857719421387, 3.1347126960754395, 4.07358193397522, 5.026141166687012, 6.098368167877197, 7.048664093017578, 8.131555080413818, 9.022128105163574, 10.006364345550537, 11.122247457504272, 12.053514957427979, 13.073466300964355, 14.0400972366333, 15.131026983261108, 16.061623573303223, 17.045853853225708, 18.12671947479248, 19.13994288444519, 20.0796160697937, 21.087053775787354, 22.074949026107788, 23.09592056274414, 24.058574438095093, 25.004663705825806, 26.10650396347046, 27.115460872650146, 28.0721378326416, 29.02433943748474, 30.120976209640503, 31.078723430633545, 32.00052213668823, 33.10694098472595, 34.03584146499634, 35.13521957397461, 36.061317443847656, 37.03960204124451, 38.116780281066895, 39.058403730392456, 40.13403677940369, 41.074589014053345, 42.14087247848511, 43.009055376052856, 44.08847975730896, 45.048038721084595, 46.02589511871338, 47.03127908706665, 48.09587287902832, 49.03518199920654, 50.10567855834961, 51.07545852661133, 52.000449657440186, 53.13817644119263, 54.11677145957947, 55.054314374923706, 56.13048553466797, 57.10110402107239, 58.08264780044556, 59.05221748352051, 60.14639687538147] 
	q1 = [0.3192771084337349, 0.33183856502242154, 0.3387815750371471, 0.3480825958702065, 0.3601756954612006, 0.37209302325581395, 0.37518037518037517, 0.3839541547277937, 0.39142857142857146, 0.3977272727272727, 0.4067796610169492, 0.41678321678321684, 0.4200278164116829, 0.4281767955801105, 0.43347050754458166, 0.44414168937329696, 0.44565217391304346, 0.45283018867924524, 0.45783132530120485, 0.4674634794156706, 0.47354497354497355, 0.4815789473684211, 0.49214659685863876, 0.49479166666666674, 0.5038759689922481, 0.5134788189987163, 0.5242966751918158, 0.5304568527918782, 0.5415617128463477, 0.5506883604505632, 0.5607940446650124, 0.5696670776818743, 0.580171358629131, 0.5888077858880778, 0.593939393939394, 0.6040914560770156, 0.6083832335329342, 0.6159334126040428, 0.6241134751773049, 0.6305882352941177, 0.6330597889800703, 0.640279394644936, 0.6465816917728853, 0.6528258362168398, 0.6597938144329896, 0.6628571428571428, 0.6666666666666666, 0.671201814058957, 0.6749435665914221, 0.6786516853932585, 0.6816143497757846, 0.6823266219239374, 0.6837988826815643, 0.688195991091314, 0.6903440621531632, 0.6902654867256638, 0.6923925027563397, 0.6938325991189429, 0.6981339187705817, 0.7016393442622951, 0.7022900763358779] 
	t2 = [0, 2.1319332122802734, 4.107113838195801, 6.083642959594727, 8.056230783462524, 10.115917682647705, 12.025886058807373, 14.058969736099243, 16.062345266342163, 18.08285093307495, 20.124064683914185, 22.04505681991577, 24.110080003738403, 26.144450187683105, 28.061836004257202, 30.06149911880493, 32.03038311004639, 34.13989806175232, 36.12132453918457, 38.14855480194092, 40.13193964958191, 42.12902069091797, 44.01773285865784, 46.049163818359375, 48.08924198150635, 50.09294414520264, 52.09969449043274, 54.139509439468384, 56.005587339401245, 58.043723583221436, 60.09546947479248] 
	q2 = [0.3192771084337349, 0.34124629080118696, 0.36257309941520466, 0.377521613832853, 0.39372325249643364, 0.4135021097046413, 0.4222222222222222, 0.43775649794801635, 0.44986449864498645, 0.4640000000000001, 0.4775725593667547, 0.49608355091383816, 0.5122265122265123, 0.529262086513995, 0.5488721804511277, 0.5679012345679013, 0.587088915956151, 0.6040914560770156, 0.6175771971496437, 0.6298472385428907, 0.641860465116279, 0.6543778801843317, 0.6643835616438356, 0.6704416761041903, 0.6786516853932585, 0.6823266219239374, 0.688195991091314, 0.6917127071823204, 0.6967032967032967, 0.7016393442622951, 0.7079261672095548] 
	t3 = [0, 3.141888380050659, 6.0203163623809814, 9.05074954032898, 12.100919723510742, 15.016130924224854, 18.063554763793945, 21.004321575164795, 24.0409197807312, 27.127579927444458, 30.098626375198364, 33.1444833278656, 36.13036775588989, 39.02939987182617, 42.048367738723755, 45.10089039802551, 48.06086540222168, 51.12092185020447, 54.10667586326599, 57.00876498222351, 60.152602672576904] 
	q3 = [0.3192771084337349, 0.35051546391752575, 0.377521613832853, 0.4028368794326242, 0.4244105409153952, 0.44565217391304346, 0.4640000000000001, 0.4901703800786369, 0.5141388174807199, 0.5422446406052964, 0.5714285714285715, 0.5956416464891041, 0.6192170818505337, 0.6362573099415204, 0.6574712643678161, 0.6696935300794551, 0.6816143497757846, 0.688195991091314, 0.6931567328918322, 0.6987951807228916, 0.7121212121212122] 
	t4 = [0, 4.056415557861328, 8.101706981658936, 12.142591953277588, 16.03187084197998, 20.1338312625885, 24.05565619468689, 28.100095987319946, 32.10418462753296, 36.002543210983276, 40.089178800582886, 44.05491614341736, 48.09125876426697, 52.01987957954407, 56.137874364852905, 60.16447615623474] 
	q4 = [0.3192771084337349, 0.36257309941520466, 0.39372325249643364, 0.4271844660194174, 0.4519621109607578, 0.47957839262187085, 0.5148005148005147, 0.5506883604505632, 0.5888077858880778, 0.6192170818505337, 0.6450116009280742, 0.6651480637813212, 0.6816143497757846, 0.6903440621531632, 0.6981339187705817, 0.7121212121212122] 
	t5 = [0, 5.059758186340332, 10.00617504119873, 15.11117434501648, 20.064963340759277, 25.11788272857666, 30.123286485671997, 35.121108293533325, 40.09102535247803, 45.06090021133423, 50.003679513931274, 55.01674795150757, 60.09431004524231] 
	q5 = [0.3192771084337349, 0.3691860465116279, 0.4140845070422535, 0.44625850340136053, 0.48021108179419525, 0.5249679897567222, 0.5731857318573186, 0.6109785202863962, 0.6450116009280742, 0.671201814058957, 0.6837988826815643, 0.6952695269526953, 0.7121212121212122] 
	t6 = [0, 6.1251842975616455, 12.090412616729736, 18.067492961883545, 24.122305393218994, 30.05314564704895, 36.143540143966675, 42.006526947021484, 48.055180311203, 54.023173093795776, 60.11029267311096] 
	q6 = [0.3192771084337349, 0.3798561151079136, 0.4277777777777778, 0.4660452729693742, 0.519280205655527, 0.5738916256157636, 0.6208530805687204, 0.658256880733945, 0.6816143497757846, 0.6923925027563397, 0.7107258938244853] 
	t7 = [0, 7.027250289916992, 14.05916166305542, 21.123109579086304, 28.090202569961548, 35.133984327316284, 42.018184423446655, 49.078848123550415, 56.104140281677246] 
	q7 = [0.3192771084337349, 0.38626609442060084, 0.4453551912568306, 0.4927916120576671, 0.5549999999999999, 0.6142857142857143, 0.6597938144329896, 0.6823266219239374, 0.6987951807228916] 
	t8 = [0, 8.019148349761963, 16.0737042427063, 24.090128183364868, 32.14428901672363, 40.03809309005737, 48.00721049308777, 56.02180886268616] 
	q8 = [0.3192771084337349, 0.3908701854493581, 0.45466847090663054, 0.5173745173745173, 0.5888077858880778, 0.6450116009280742, 0.6816143497757846, 0.6973684210526316] 
	t9 = [0, 9.095239162445068, 18.05953812599182, 27.000991344451904, 36.108206033706665, 45.095961570739746, 54.0476233959198] 
	q9 = [0.3192771084337349, 0.4028368794326242, 0.4666666666666666, 0.5491183879093199, 0.6224852071005917, 0.671945701357466, 0.6923925027563397] 
	t10 = [0, 10.128945350646973, 20.110642194747925, 30.026047945022583, 40.109129428863525, 50.02420735359192, 60.0335488319397] 
	q10 = [0.3192771084337349, 0.4135021097046413, 0.4848484848484848, 0.5788177339901477, 0.6465816917728853, 0.6867335562987736, 0.7099567099567099] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.095921277999878, 2.0218379497528076, 3.006847858428955, 4.083143949508667, 5.062376022338867, 6.136995077133179, 7.103214263916016, 8.033205270767212, 9.128327369689941, 10.113569736480713, 11.144624471664429, 12.079001665115356, 13.025989532470703, 14.099304676055908, 15.074772834777832, 16.066891193389893, 17.010645389556885, 18.093052864074707, 19.034507513046265, 20.103458404541016, 21.119258642196655, 22.041120767593384, 23.04458999633789, 24.023241996765137, 25.141289472579956, 26.121192693710327, 27.021522045135498, 28.133565664291382, 29.022092580795288, 30.096306324005127, 31.08388614654541, 32.009833335876465, 33.105005502700806, 34.06264352798462, 35.00613713264465, 36.11026477813721, 37.06058692932129, 38.017828702926636, 39.142805337905884, 40.081618309020996, 41.05019426345825, 42.134809255599976, 43.12118411064148, 44.04377102851868, 45.133915424346924, 46.11336898803711, 47.088860750198364, 48.04309391975403, 49.018256187438965, 50.00775742530823, 51.13337707519531, 52.05008387565613, 53.02048873901367, 54.080490589141846, 55.05051589012146, 56.1165189743042, 57.122166872024536, 58.11564803123474, 59.09161448478699, 60.04397487640381] 
	q1 = [0.3644859813084112, 0.3678516228748068, 0.3742331288343558, 0.3835616438356165, 0.3933434190620272, 0.4047976011994004, 0.4160475482912333, 0.4300441826215022, 0.44152046783625737, 0.446064139941691, 0.45217391304347826, 0.4604316546762591, 0.4670487106017192, 0.4744318181818182, 0.48169014084507034, 0.48314606741573035, 0.4825662482566248, 0.49307479224376727, 0.5013774104683196, 0.5048010973936901, 0.510231923601637, 0.5149051490514905, 0.5209176788124157, 0.5315436241610739, 0.5367156208277704, 0.5464190981432361, 0.554089709762533, 0.5583224115334208, 0.5658409387222947, 0.5751295336787565, 0.5842985842985843, 0.5907928388746803, 0.5972045743329097, 0.605296343001261, 0.6140350877192983, 0.6192259675405742, 0.630407911001236, 0.6346863468634687, 0.6355828220858896, 0.6414634146341464, 0.6496969696969698, 0.6545893719806763, 0.6618705035971223, 0.6698450536352801, 0.6753554502369669, 0.6768867924528302, 0.6830985915492958, 0.6892523364485982, 0.6953488372093023, 0.701388888888889, 0.7035755478662055, 0.7072330654420208, 0.7101947308132875, 0.7160775370581528, 0.7173666288308741, 0.7194570135746606, 0.7200902934537247, 0.7237880496054114, 0.7280898876404495, 0.7309417040358744, 0.7301231802911534] 
	t2 = [0, 2.1183416843414307, 4.131761789321899, 6.0023486614227295, 8.127097845077515, 10.117730617523193, 12.00234580039978, 14.131154537200928, 16.02101469039917, 18.036929607391357, 20.00442624092102, 22.045623064041138, 24.115000247955322, 26.014843940734863, 28.02685022354126, 30.090629816055298, 32.07964062690735, 34.08499526977539, 36.09969758987427, 38.09765839576721, 40.09112620353699, 42.052077531814575, 44.05199217796326, 46.094200134277344, 48.12712907791138, 50.041494607925415, 52.0360107421875, 54.058895111083984, 56.08159112930298, 58.1441764831543, 60.06178545951843] 
	q2 = [0.3644859813084112, 0.3767228177641654, 0.3957703927492447, 0.4160475482912333, 0.44152046783625737, 0.4544138929088278, 0.4714285714285714, 0.48169014084507034, 0.4867872044506259, 0.5034387895460798, 0.5115646258503401, 0.5295698924731184, 0.5444887118193891, 0.5559947299077733, 0.5732814526588845, 0.58898847631242, 0.6060606060606061, 0.6192259675405742, 0.6346863468634687, 0.6414634146341464, 0.6562123039806996, 0.6714285714285714, 0.6784452296819788, 0.6923076923076923, 0.7028901734104047, 0.7072330654420208, 0.7175398633257404, 0.7194570135746606, 0.7266591676040495, 0.7295173961840629, 0.7321428571428571] 
	t3 = [0, 3.041562557220459, 6.060063362121582, 9.077831029891968, 12.010854959487915, 15.09192180633545, 18.063382148742676, 21.01209855079651, 24.004645109176636, 27.004866123199463, 30.095933198928833, 33.02318334579468, 36.07372045516968, 39.029513359069824, 42.03345608711243, 45.09254693984985, 48.06186628341675, 51.027145862579346, 54.11053824424744, 57.033785820007324, 60.006670236587524] 
	q3 = [0.3644859813084112, 0.3835616438356165, 0.41839762611275966, 0.4483260553129549, 0.470756062767475, 0.48391608391608393, 0.5027472527472527, 0.5169147496617049, 0.5444887118193891, 0.5635648754914809, 0.5907928388746803, 0.6140350877192983, 0.6354679802955666, 0.6513317191283293, 0.6714285714285714, 0.6861826697892273, 0.7028901734104047, 0.7131428571428571, 0.7186440677966102, 0.7280898876404495, 0.7321428571428571] 
	t4 = [0, 4.063344955444336, 8.132242202758789, 12.045550346374512, 16.01170516014099, 20.01059603691101, 24.096121311187744, 28.06563115119934, 32.03517937660217, 36.10768413543701, 40.06373453140259, 44.00367546081543, 48.0032172203064, 52.058335304260254, 56.0301308631897, 60.106767416000366] 
	q4 = [0.3644859813084112, 0.3957703927492447, 0.44152046783625737, 0.47293447293447294, 0.4923504867872044, 0.5108695652173914, 0.5490716180371353, 0.5777202072538861, 0.6070528967254408, 0.6371463714637146, 0.6586538461538461, 0.6830985915492958, 0.7043879907621245, 0.7173666288308741, 0.7266591676040495, 0.7363737486095661] 
	t5 = [0, 5.044769763946533, 10.047955751419067, 15.049553155899048, 20.08365774154663, 25.065622806549072, 30.13967752456665, 35.04891848564148, 40.071205377578735, 45.06496000289917, 50.13587260246277, 55.05809259414673, 60.01681590080261] 
	q5 = [0.3644859813084112, 0.40956651718983555, 0.45664739884393063, 0.48391608391608393, 0.5128900949796472, 0.5548216644649935, 0.5936305732484076, 0.6320987654320989, 0.6602641056422569, 0.6907817969661609, 0.7124856815578465, 0.7266591676040495, 0.7357859531772576] 
	t6 = [0, 6.105729341506958, 12.131676435470581, 18.041372299194336, 24.11858367919922, 30.088322401046753, 36.009644746780396, 42.104498624801636, 48.12012314796448, 54.029640436172485, 60.00212740898132] 
	q6 = [0.3644859813084112, 0.4207407407407408, 0.47226173541963024, 0.5041322314049587, 0.547144754316069, 0.5915492957746479, 0.6371463714637146, 0.6737841043890866, 0.7043879907621245, 0.7223476297968396, 0.734375] 
	t7 = [0, 7.018527984619141, 14.071648836135864, 21.107969760894775, 28.07283043861389, 35.117830753326416, 42.08781051635742, 49.01253581047058, 56.09098029136658] 
	q7 = [0.3644859813084112, 0.43235294117647055, 0.48382559774964845, 0.5209176788124157, 0.5829015544041452, 0.6337854500616522, 0.6745562130177515, 0.7080459770114942, 0.7280898876404495] 
	t8 = [0, 8.022206783294678, 16.141350984573364, 24.057955741882324, 32.056875228881836, 40.05914235115051, 48.069348096847534, 56.06524419784546] 
	q8 = [0.3644859813084112, 0.44152046783625737, 0.4965325936199722, 0.5490716180371353, 0.6095717884130983, 0.6626650660264105, 0.7058823529411765, 0.7280898876404495] 
	t9 = [0, 9.081738948822021, 18.136460304260254, 27.040815591812134, 36.138492584228516, 45.09400653839111, 54.002793312072754] 
	q9 = [0.3644859813084112, 0.45058139534883723, 0.5061898211829435, 0.5680628272251309, 0.6363636363636364, 0.6907817969661609, 0.7217194570135747] 
	t10 = [0, 10.109366416931152, 20.08702802658081, 30.04377841949463, 40.01071500778198, 50.13956260681152, 60.07237458229065] 
	q10 = [0.3644859813084112, 0.45887445887445893, 0.5163043478260869, 0.5959079283887468, 0.6626650660264105, 0.7139588100686499, 0.7357859531772576] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.1021363735198975, 2.0337777137756348, 3.0457565784454346, 4.125756025314331, 5.079651355743408, 6.004600286483765, 7.098105430603027, 8.027682304382324, 9.043455362319946, 10.006484270095825, 11.102197170257568, 12.056313753128052, 13.002444744110107, 14.11413311958313, 15.087849140167236, 16.027483701705933, 17.123146057128906, 18.05207324028015, 19.074176788330078, 20.026988983154297, 21.039709329605103, 22.10688543319702, 23.09085202217102, 24.019688606262207, 25.1070237159729, 26.090818881988525, 27.032421588897705, 28.13608741760254, 29.083937644958496, 30.051395893096924, 31.139194011688232, 32.124836921691895, 33.1118061542511, 34.068374156951904, 35.00940465927124, 36.07642602920532, 37.01623773574829, 38.08641695976257, 39.079596757888794, 40.01655697822571, 41.102413177490234, 42.05302596092224, 43.003331899642944, 44.07516694068909, 45.05829954147339, 46.01524043083191, 47.13799071311951, 48.00720143318176, 49.107354164123535, 50.13586688041687, 51.116915225982666, 52.04781150817871, 53.06621265411377, 54.14353561401367, 55.0838418006897, 56.0042359828949, 57.1026177406311, 58.11903524398804, 59.06618690490723, 60.137808084487915] 
	q1 = [0.34359805510534847, 0.35048231511254024, 0.3584, 0.3682539682539683, 0.380503144654088, 0.3875, 0.3950233281493002, 0.404320987654321, 0.4030769230769231, 0.40916030534351144, 0.4127465857359636, 0.42232277526395173, 0.42750373692077726, 0.43154761904761907, 0.43722304283604135, 0.44640234948604995, 0.45321637426900585, 0.4608695652173913, 0.4639769452449568, 0.4721030042918455, 0.47578347578347585, 0.4887005649717514, 0.4943820224719101, 0.4979020979020979, 0.5083333333333333, 0.5158620689655173, 0.5273224043715847, 0.5326086956521738, 0.5390835579514824, 0.5495978552278821, 0.5539280958721705, 0.5604249667994687, 0.570673712021136, 0.5763157894736842, 0.5886990801576872, 0.59375, 0.6020671834625323, 0.6110397946084724, 0.6163682864450127, 0.621656050955414, 0.629582806573957, 0.6347607052896727, 0.6390977443609022, 0.6450809464508095, 0.6526576019777504, 0.6592865928659286, 0.6625766871165644, 0.6699266503667483, 0.6731470230862697, 0.6747572815533981, 0.6755447941888619, 0.6787439613526569, 0.6803377563329313, 0.6787003610108304, 0.6842105263157895, 0.6865315852205006, 0.688836104513064, 0.6934911242603551, 0.6972909305064782, 0.6980023501762632, 0.7010550996483] 
	t2 = [0, 2.1128640174865723, 4.127187728881836, 6.142874479293823, 8.120839595794678, 10.03728723526001, 12.031187295913696, 14.016697645187378, 16.064066171646118, 18.036361932754517, 20.07677984237671, 22.105300903320312, 24.13757300376892, 26.000742435455322, 28.027035236358643, 30.030681371688843, 32.03976130485535, 34.08851981163025, 36.0409414768219, 38.01898646354675, 40.03942060470581, 42.06024742126465, 44.06583333015442, 46.0820734500885, 48.034005641937256, 50.102317810058594, 52.13940501213074, 54.14320993423462, 56.127285957336426, 58.04141116142273, 60.083534240722656] 
	q2 = [0.34359805510534847, 0.3578274760383387, 0.38304552590266877, 0.4, 0.40245775729646704, 0.4157814871016691, 0.429210134128167, 0.4395280235988201, 0.45772594752186585, 0.4677187948350072, 0.48725212464589235, 0.4964936886395512, 0.5138121546961325, 0.5286103542234333, 0.5456989247311828, 0.5604249667994687, 0.5744400527009222, 0.5953002610966057, 0.6110397946084724, 0.621656050955414, 0.6347607052896727, 0.6450809464508095, 0.6592865928659286, 0.6699266503667483, 0.6739393939393941, 0.6803377563329313, 0.6818727490996398, 0.688095238095238, 0.6965761511216055, 0.6995305164319249, 0.7101280558789289] 
	t3 = [0, 3.0333023071289062, 6.064619779586792, 9.09124207496643, 12.049109697341919, 15.10826563835144, 18.013675689697266, 21.09786581993103, 24.050692081451416, 27.098994970321655, 30.031482458114624, 33.081886529922485, 36.02719807624817, 39.05768704414368, 42.11901330947876, 45.00439715385437, 48.12615394592285, 51.12316823005676, 54.08450222015381, 57.081881046295166, 60.042174339294434] 
	q3 = [0.34359805510534847, 0.3734177215189874, 0.4, 0.40916030534351144, 0.43154761904761907, 0.4509516837481699, 0.4677187948350072, 0.4950773558368496, 0.5158620689655173, 0.5378378378378378, 0.5611702127659575, 0.5879265091863517, 0.6128205128205128, 0.629582806573957, 0.6459627329192547, 0.6650306748466258, 0.6755447941888619, 0.6795180722891566, 0.689655172413793, 0.6972909305064782, 0.7116279069767443] 
	t4 = [0, 4.0422186851501465, 8.116942405700684, 12.146333932876587, 16.081167936325073, 20.017083406448364, 24.046795129776, 28.049817323684692, 32.00948667526245, 36.06121635437012, 40.08676028251648, 44.012388706207275, 48.15566611289978, 52.092230558395386, 56.021135568618774, 60.025044679641724] 
	q4 = [0.34359805510534847, 0.38304552590266877, 0.40490797546012275, 0.43154761904761907, 0.4593023255813953, 0.48939179632248936, 0.5158620689655173, 0.5495978552278821, 0.580814717477004, 0.6161745827984596, 0.6398996235884568, 0.6592865928659286, 0.6763636363636363, 0.6842105263157895, 0.6972909305064782, 0.7116279069767443] 
	t5 = [0, 5.050650358200073, 10.003708600997925, 15.078511238098145, 20.101256132125854, 25.11835026741028, 30.00827407836914, 35.06631636619568, 40.05002808570862, 45.069358825683594, 50.0657172203064, 55.03491497039795, 60.080002546310425] 
	q5 = [0.34359805510534847, 0.390015600624025, 0.4181818181818181, 0.45321637426900585, 0.48939179632248936, 0.5232876712328768, 0.5611702127659575, 0.6090322580645161, 0.6398996235884568, 0.6666666666666667, 0.6803377563329313, 0.6950354609929078, 0.7116279069767443] 
	t6 = [0, 6.141274690628052, 12.134759664535522, 18.131950616836548, 24.059369564056396, 30.10676336288452, 36.12246131896973, 42.14112377166748, 48.04063081741333, 54.13890862464905, 60.05724763870239] 
	q6 = [0.34359805510534847, 0.40247678018575844, 0.43219076005961254, 0.46991404011461324, 0.5179063360881543, 0.5630810092961488, 0.617948717948718, 0.650990099009901, 0.6747572815533981, 0.690391459074733, 0.7124563445867287] 
	t7 = [0, 7.018432855606079, 14.047771453857422, 21.075472593307495, 28.090876579284668, 35.06330442428589, 42.12503004074097, 49.04861092567444, 56.093751668930054] 
	q7 = [0.34359805510534847, 0.4030769230769231, 0.4470588235294118, 0.4950773558368496, 0.5476510067114094, 0.6126126126126127, 0.6526576019777504, 0.6795646916565901, 0.6972909305064782] 
	t8 = [0, 8.022361755371094, 16.047178983688354, 24.04736018180847, 32.11922788619995, 40.09229874610901, 48.06780457496643, 56.106467485427856] 
	q8 = [0.34359805510534847, 0.40490797546012275, 0.4615384615384615, 0.5179063360881543, 0.5860709592641261, 0.6424090338770388, 0.6747572815533981, 0.6972909305064782] 
	t9 = [0, 9.073846578598022, 18.071701765060425, 27.11428689956665, 36.038053035736084, 45.05504035949707, 54.05288505554199] 
	q9 = [0.34359805510534847, 0.4140030441400305, 0.46991404011461324, 0.5390835579514824, 0.6187419768934531, 0.6674816625916871, 0.6919431279620852] 
	t10 = [0, 10.136391401290894, 20.10960817337036, 30.015212059020996, 40.01065754890442, 50.10661220550537, 60.02687382698059] 
	q10 = [0.34359805510534847, 0.4229607250755287, 0.4915254237288135, 0.5611702127659575, 0.6424090338770388, 0.6771463119709794, 0.7109557109557109] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.10691499710083, 2.0331573486328125, 3.1260108947753906, 4.053690195083618, 5.008774280548096, 6.074951648712158, 7.017907381057739, 8.007038354873657, 9.13399624824524, 10.134021759033203, 11.076713800430298, 12.004766941070557, 13.143914937973022, 14.083129644393921, 15.036052942276001, 16.10716199874878, 17.117481231689453, 18.109880685806274, 19.083284854888916, 20.078341245651245, 21.049858331680298, 22.001556396484375, 23.105364561080933, 24.05507779121399, 25.0669207572937, 26.02262258529663, 27.14206099510193, 28.09810709953308, 29.03854274749756, 30.12613272666931, 31.086509704589844, 32.009252071380615, 33.138001918792725, 34.10091829299927, 35.05338740348816, 36.012754917144775, 37.11260485649109, 38.06571412086487, 39.03800582885742, 40.11324405670166, 41.00377631187439, 42.1116726398468, 43.0882408618927, 44.0727903842926, 45.03059935569763, 46.12618923187256, 47.06898546218872, 48.030195474624634, 49.004427433013916, 50.070019483566284, 51.04375171661377, 52.11831998825073, 53.05792689323425, 54.12048006057739, 55.119428873062134, 56.03648114204407, 57.0135293006897, 58.09001970291138, 59.06879210472107, 60.02909803390503] 
	q1 = [0.36196319018404904, 0.36585365853658536, 0.38066465256797577, 0.3898050974512744, 0.39821693907875183, 0.4070796460176991, 0.4164222873900293, 0.4256559766763848, 0.4267053701015965, 0.43352601156069365, 0.44189383070301286, 0.4479315263908702, 0.4517045454545454, 0.45915492957746484, 0.46993006993006997, 0.4707520891364902, 0.47790055248618785, 0.4862637362637363, 0.49041095890410963, 0.4959128065395096, 0.503382949932341, 0.5080645161290323, 0.5173333333333333, 0.5298013245033114, 0.5375494071146245, 0.5490196078431373, 0.561038961038961, 0.5658914728682171, 0.5725288831835686, 0.5798212005108557, 0.5888324873096447, 0.5959595959595959, 0.5997490589711417, 0.6034912718204489, 0.607940446650124, 0.6131025957972805, 0.6182266009852216, 0.6266829865361078, 0.6333739342265531, 0.64, 0.644927536231884, 0.6506602641056424, 0.6610978520286396, 0.6634958382877527, 0.6713947990543736, 0.6776470588235294, 0.6869158878504672, 0.68997668997669, 0.694541231126597, 0.7005780346820809, 0.7050691244239631, 0.706559263521289, 0.7064220183486238, 0.7077625570776256, 0.7107061503416856, 0.7113636363636363, 0.7128263337116912, 0.7171945701357465, 0.7163841807909603, 0.7192784667418264, 0.7199100112485939] 
	t2 = [0, 2.1282999515533447, 4.109192848205566, 6.087310075759888, 8.058830499649048, 10.144147634506226, 12.007411241531372, 14.015113592147827, 16.137948274612427, 18.049834489822388, 20.088574409484863, 22.118174076080322, 24.025065660476685, 26.09152388572693, 28.14452314376831, 30.118102550506592, 32.086930990219116, 34.13556241989136, 36.1473548412323, 38.139522075653076, 40.003475189208984, 42.10318565368652, 44.018898487091064, 46.00933241844177, 48.01317834854126, 50.014808177948, 52.00349998474121, 54.118754863739014, 56.00791096687317, 58.0130569934845, 60.04548740386963] 
	q2 = [0.36196319018404904, 0.38009049773755654, 0.400593471810089, 0.4187408491947291, 0.4267053701015965, 0.4441260744985673, 0.453257790368272, 0.4692737430167598, 0.4827586206896552, 0.49180327868852464, 0.5087483176312247, 0.5258964143426296, 0.5433070866141733, 0.5647668393782384, 0.5780051150895141, 0.5959595959595959, 0.6017478152309613, 0.6131025957972805, 0.6266829865361078, 0.64, 0.6506602641056424, 0.665083135391924, 0.6776470588235294, 0.6915017462165309, 0.7020785219399538, 0.7080459770114942, 0.709236031927024, 0.7136363636363636, 0.7171945701357465, 0.7199100112485939, 0.7256438969764839] 
	t3 = [0, 3.035181760787964, 6.02358865737915, 9.024139404296875, 12.061882972717285, 15.130398988723755, 18.137641191482544, 21.036713361740112, 24.096121549606323, 27.071366548538208, 30.000226974487305, 33.003772497177124, 36.08245229721069, 39.01823830604553, 42.021350383758545, 45.12887620925903, 48.002575635910034, 51.109079122543335, 54.10653376579285, 57.04115438461304, 60.10735511779785] 
	q3 = [0.36196319018404904, 0.38680659670164913, 0.4187408491947291, 0.4380403458213257, 0.4554455445544555, 0.4792243767313019, 0.49180327868852464, 0.5160427807486632, 0.5471204188481675, 0.5725288831835686, 0.5959595959595959, 0.607940446650124, 0.63003663003663, 0.646562123039807, 0.6682464454976305, 0.6884480746791132, 0.7020785219399538, 0.7085714285714285, 0.7136363636363636, 0.7192784667418264, 0.7270693512304249] 
	t4 = [0, 4.037683010101318, 8.07369589805603, 12.11129641532898, 16.052417278289795, 20.049153804779053, 24.08671998977661, 28.135047435760498, 32.09221148490906, 36.041375398635864, 40.00469779968262, 44.06150794029236, 48.072017431259155, 52.045923709869385, 56.08645582199097, 60.09875798225403] 
	q4 = [0.36196319018404904, 0.400593471810089, 0.4267053701015965, 0.4554455445544555, 0.48484848484848475, 0.5087483176312247, 0.5471204188481675, 0.5798212005108557, 0.6034912718204489, 0.63003663003663, 0.6555023923444976, 0.6822977725674092, 0.7035755478662054, 0.7107061503416856, 0.7171945701357465, 0.7270693512304249] 
	t5 = [0, 5.084929704666138, 10.142705917358398, 15.064712285995483, 20.05142903327942, 25.08529496192932, 30.018601655960083, 35.129807472229004, 40.03750681877136, 45.00649428367615, 50.093533515930176, 55.10061740875244, 60.11560249328613] 
	q5 = [0.36196319018404904, 0.40942562592047127, 0.4434907010014306, 0.4792243767313019, 0.506056527590848, 0.561038961038961, 0.5994962216624685, 0.6233128834355829, 0.6594982078853047, 0.6884480746791132, 0.7064220183486238, 0.7180067950169874, 0.7270693512304249] 
	t6 = [0, 6.126925230026245, 12.088501691818237, 18.07441520690918, 24.067620038986206, 30.09966516494751, 36.144057512283325, 42.08488941192627, 48.014814376831055, 54.074190855026245, 60.125730752944946] 
	q6 = [0.36196319018404904, 0.42105263157894735, 0.4576271186440678, 0.49386084583901774, 0.5516339869281046, 0.6012578616352201, 0.6317073170731707, 0.6698224852071005, 0.7050691244239631, 0.7150964812712827, 0.7270693512304249] 
	t7 = [0, 7.013762712478638, 14.088975429534912, 21.112998962402344, 28.051905393600464, 35.07333493232727, 42.04901909828186, 49.10675024986267, 56.08526039123535] 
	q7 = [0.36196319018404904, 0.42274052478134116, 0.4720670391061453, 0.5180240320427236, 0.5798212005108557, 0.6216216216216216, 0.6698224852071005, 0.7050691244239631, 0.7186440677966102] 
	t8 = [0, 8.07319712638855, 16.125107049942017, 24.030827522277832, 32.05551815032959, 40.099063873291016, 48.0954430103302, 56.15415549278259] 
	q8 = [0.36196319018404904, 0.4267053701015965, 0.4876033057851239, 0.5516339869281046, 0.6044776119402986, 0.6594982078853047, 0.7035755478662054, 0.7200902934537247] 
	t9 = [0, 9.115409851074219, 18.02003002166748, 27.1347234249115, 36.04389834403992, 45.06472873687744, 54.14420962333679] 
	q9 = [0.36196319018404904, 0.4351585014409221, 0.49453551912568305, 0.5750962772785623, 0.6317073170731707, 0.68997668997669, 0.7180067950169874] 
	t10 = [0, 10.094748973846436, 20.055281162261963, 30.006645917892456, 40.007834672927856, 50.09927201271057, 60.10679745674133] 
	q10 = [0.36196319018404904, 0.44126074498567336, 0.5067385444743935, 0.5994962216624685, 0.6594982078853047, 0.7026406429391505, 0.7293064876957495] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.0998401641845703, 2.0374042987823486, 3.14408540725708, 4.076573371887207, 5.0238823890686035, 6.09792160987854, 7.052130699157715, 8.011599779129028, 9.003006219863892, 10.108438968658447, 11.09995985031128, 12.064262628555298, 13.066782474517822, 14.143784523010254, 15.086788415908813, 16.013402938842773, 17.13958716392517, 18.060953378677368, 19.030399322509766, 20.016411781311035, 21.02458882331848, 22.131459712982178, 23.103347063064575, 24.029974460601807, 25.033129692077637, 26.102672338485718, 27.047095775604248, 28.143844842910767, 29.103252410888672, 30.02474856376648, 31.129640340805054, 32.0805504322052, 33.022727489471436, 34.09340143203735, 35.03726863861084, 36.10426902770996, 37.103726863861084, 38.018969774246216, 39.131147146224976, 40.094038248062134, 41.03647232055664, 42.14428496360779, 43.003658056259155, 44.078498125076294, 45.07732319831848, 46.09712815284729, 47.06604623794556, 48.01229786872864, 49.1361358165741, 50.09096169471741, 51.036396741867065, 52.13699960708618, 53.083893060684204, 54.010818004608154, 55.135703563690186, 56.09234070777893, 57.04170370101929, 58.06517028808594, 59.011375427246094, 60.0745792388916] 
	q1 = [0.36645962732919257, 0.3784615384615384, 0.3877862595419848, 0.3939393939393939, 0.4, 0.40718562874251496, 0.41604754829123325, 0.4183976261127597, 0.42709867452135497, 0.43631039531478766, 0.4476744186046511, 0.455988455988456, 0.45755395683453237, 0.46, 0.4730878186968838, 0.476056338028169, 0.4811188811188811, 0.490984743411928, 0.4965517241379311, 0.5054945054945055, 0.5081967213114754, 0.5108695652173914, 0.5148247978436657, 0.5267379679144385, 0.5326231691078562, 0.5408970976253299, 0.5523560209424084, 0.5617685305591678, 0.5710594315245477, 0.5750962772785623, 0.5816326530612245, 0.5870393900889453, 0.5959595959595959, 0.6040100250626566, 0.6127023661270237, 0.6178660049627791, 0.6271604938271605, 0.628992628992629, 0.6341463414634146, 0.6424242424242426, 0.6481927710843374, 0.6538922155688623, 0.6547619047619048, 0.6627218934911243, 0.669811320754717, 0.6729411764705883, 0.6814469078179697, 0.6837209302325582, 0.6867749419953596, 0.6905311778290992, 0.6935483870967741, 0.6950517836593786, 0.6979405034324943, 0.6963470319634704, 0.6993166287015945, 0.7006802721088435, 0.7028248587570621, 0.7042889390519187, 0.7093153759820428, 0.70996640537514, 0.7157190635451505] 
	t2 = [0, 2.128361225128174, 4.116765260696411, 6.085350513458252, 8.048367738723755, 10.121778726577759, 12.115488767623901, 14.065097093582153, 16.030658960342407, 18.033262014389038, 20.058342695236206, 22.100314617156982, 24.13896131515503, 26.017054557800293, 28.14254331588745, 30.00066113471985, 32.11311745643616, 34.121649503707886, 36.10608720779419, 38.14350748062134, 40.00578045845032, 42.01620435714722, 44.09283089637756, 46.04654359817505, 48.09320569038391, 50.09701371192932, 52.10418891906738, 54.12094211578369, 56.13403844833374, 58.047669887542725, 60.04720115661621] 
	q2 = [0.36645962732919257, 0.3871951219512195, 0.40240240240240244, 0.41604754829123325, 0.4294117647058823, 0.44992743105950656, 0.45689655172413796, 0.4752475247524753, 0.4874651810584958, 0.5013774104683195, 0.5115646258503401, 0.5234899328859061, 0.5396825396825397, 0.5598958333333333, 0.5732647814910027, 0.5870393900889453, 0.6047678795483061, 0.6178660049627791, 0.628992628992629, 0.6424242424242426, 0.6538922155688623, 0.6650887573964498, 0.6745005875440659, 0.6845168800931315, 0.6905311778290992, 0.6965517241379311, 0.6963470319634704, 0.7021517553793885, 0.7080045095828635, 0.7128491620111732, 0.7177777777777777] 
	t3 = [0, 3.0159473419189453, 6.02655553817749, 9.040404796600342, 12.044955015182495, 15.002565860748291, 18.14611554145813, 21.13670516014099, 24.112451791763306, 27.045019388198853, 30.084656238555908, 33.09537625312805, 36.09443688392639, 39.02977466583252, 42.041786432266235, 45.02014207839966, 48.01997137069702, 51.141185998916626, 54.0358464717865, 57.077136754989624, 60.07401084899902] 
	q3 = [0.36645962732919257, 0.3939393939393939, 0.41604754829123325, 0.43859649122807026, 0.45689655172413796, 0.48246844319775595, 0.5054945054945055, 0.5162162162162163, 0.5416116248348746, 0.5692108667529107, 0.5870393900889453, 0.6144278606965174, 0.6323529411764706, 0.6514423076923077, 0.6682408500590319, 0.6837806301050176, 0.6905311778290992, 0.6963470319634704, 0.7036199095022625, 0.7093153759820428, 0.7184035476718403] 
	t4 = [0, 4.128469228744507, 8.079154253005981, 12.064065933227539, 16.09325408935547, 20.02936100959778, 24.06037712097168, 28.075227737426758, 32.12510681152344, 36.042699337005615, 40.14673638343811, 44.011563539505005, 48.08779692649841, 52.05584406852722, 56.01351881027222, 60.04970407485962] 
	q4 = [0.36645962732919257, 0.40240240240240244, 0.4294117647058823, 0.45755395683453237, 0.4895688456189151, 0.5115646258503401, 0.5423280423280423, 0.5750962772785623, 0.6082603254067583, 0.6323529411764706, 0.6547192353643966, 0.6776084407971864, 0.6920415224913495, 0.6993166287015945, 0.7086614173228347, 0.7198228128460686] 
	t5 = [0, 5.063287734985352, 10.120372533798218, 15.07022738456726, 20.0629563331604, 25.031073331832886, 30.02994465827942, 35.034594774246216, 40.06280159950256, 45.027238845825195, 50.05190825462341, 55.08036422729492, 60.13243532180786] 
	q5 = [0.36645962732919257, 0.40956651718983555, 0.45217391304347826, 0.48179271708683474, 0.5115646258503401, 0.5549738219895288, 0.5913705583756346, 0.6273062730627307, 0.6539379474940334, 0.6837806301050176, 0.6964490263459335, 0.7065462753950339, 0.7190265486725663] 
	t6 = [0, 6.0282371044158936, 12.149386882781982, 18.013516902923584, 24.090599298477173, 30.07762837409973, 36.10650587081909, 42.08544659614563, 48.08410620689392, 54.00246047973633, 60.09515166282654] 
	q6 = [0.36645962732919257, 0.41604754829123325, 0.4597701149425287, 0.5054945054945055, 0.5442536327608982, 0.5913705583756346, 0.6308068459657702, 0.669811320754717, 0.6921296296296297, 0.7036199095022625, 0.7212389380530974] 
	t7 = [0, 7.104924201965332, 14.13019871711731, 21.003659963607788, 28.05250096321106, 35.00258278846741, 42.07862210273743, 49.0933141708374, 56.04063272476196] 
	q7 = [0.36645962732919257, 0.42011834319526625, 0.4787535410764873, 0.5162162162162163, 0.5794871794871795, 0.6280788177339902, 0.669811320754717, 0.6966551326412919, 0.7093153759820428] 
	t8 = [0, 8.0494384765625, 16.02113652229309, 24.053143739700317, 32.061474561691284, 40.0086088180542, 48.08625054359436, 56.07836937904358] 
	q8 = [0.36645962732919257, 0.4294117647058823, 0.4909344490934449, 0.546895640686922, 0.61, 0.6555423122765197, 0.6921296296296297, 0.7101123595505618] 
	t9 = [0, 9.091446161270142, 18.130075454711914, 27.13011932373047, 36.11638045310974, 45.12909412384033, 54.107309103012085] 
	q9 = [0.36645962732919257, 0.4408759124087591, 0.5061898211829436, 0.5710594315245477, 0.6332518337408313, 0.6853146853146853, 0.7036199095022625] 
	t10 = [0, 10.0839102268219, 20.074047803878784, 30.11776328086853, 40.10847806930542, 50.11785054206848, 60.06519794464111] 
	q10 = [0.36645962732919257, 0.45217391304347826, 0.5115646258503401, 0.5939086294416244, 0.6547619047619048, 0.695752009184845, 0.7220376522702104] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.111732006072998, 2.0540409088134766, 3.009596347808838, 4.110177278518677, 5.095278739929199, 6.025933027267456, 7.1214377880096436, 8.086282968521118, 9.078488826751709, 10.008196115493774, 11.135892629623413, 12.089895248413086, 13.033798217773438, 14.134340524673462, 15.077495336532593, 16.14575719833374, 17.094417333602905, 18.02178454399109, 19.142338514328003, 20.07369041442871, 21.058788537979126, 22.13124656677246, 23.130287170410156, 24.04973030090332, 25.041513919830322, 26.11517596244812, 27.008347034454346, 28.12648296356201, 29.11474323272705, 30.05662250518799, 31.02090358734131, 32.085676431655884, 33.029916286468506, 34.13520789146423, 35.07693004608154, 36.143659830093384, 37.08102989196777, 38.05128717422485, 39.14684796333313, 40.06751322746277, 41.010462284088135, 42.08274841308594, 43.067506313323975, 44.046725273132324, 45.02040243148804, 46.11469006538391, 47.088518142700195, 48.01489043235779, 49.00406813621521, 50.07902932167053, 51.076945543289185, 52.002851247787476, 53.12336230278015, 54.0501127243042, 55.1336715221405, 56.08935332298279, 57.10186195373535, 58.06578278541565, 59.01576805114746, 60.11576318740845] 
	q1 = [0.37694704049844235, 0.38639876352395675, 0.39263803680981596, 0.3981623277182236, 0.40364188163884673, 0.41327300150829566, 0.42514970059880236, 0.43684992570579495, 0.4460856720827178, 0.4493392070484582, 0.456140350877193, 0.46444121915820025, 0.4668587896253602, 0.47058823529411764, 0.4779516358463727, 0.48870056497175146, 0.49719101123595505, 0.5013927576601671, 0.5117565698478561, 0.5212620027434842, 0.5266030013642564, 0.5358592692828146, 0.5410497981157469, 0.5519999999999999, 0.5596816976127321, 0.570673712021136, 0.5774278215223096, 0.5796344647519581, 0.5844155844155845, 0.5909677419354838, 0.6, 0.604591836734694, 0.6134347275031685, 0.6204287515762926, 0.6299999999999999, 0.6377171215880894, 0.6428571428571429, 0.6519607843137254, 0.6536585365853659, 0.6610169491525424, 0.6690734055354994, 0.6738351254480287, 0.6809015421115066, 0.6863207547169812, 0.6948356807511737, 0.6994152046783625, 0.7038327526132403, 0.7113163972286374, 0.7149425287356321, 0.7191780821917808, 0.7206385404789054, 0.7212741751990899, 0.7235494880546074, 0.7264472190692395, 0.72686230248307, 0.7282976324689967, 0.7289088863892013, 0.7301231802911534, 0.7327394209354119, 0.7369589345172032, 0.7425414364640884] 
	t2 = [0, 2.13702654838562, 4.1344428062438965, 6.0372703075408936, 8.017540216445923, 10.048243522644043, 12.068020820617676, 14.115105390548706, 16.093766450881958, 18.070254802703857, 20.07160997390747, 22.07372808456421, 24.114463806152344, 26.140140771865845, 28.048720121383667, 30.054147243499756, 32.03888177871704, 34.05140709877014, 36.02678060531616, 38.03650212287903, 40.00381350517273, 42.13364839553833, 44.0545711517334, 46.132351875305176, 48.11448550224304, 50.03055429458618, 52.001322746276855, 54.14182925224304, 56.0037522315979, 58.06678557395935, 60.07226514816284] 
	q2 = [0.37694704049844235, 0.39263803680981596, 0.40364188163884673, 0.4275037369207773, 0.4460856720827178, 0.45839416058394167, 0.4697406340057637, 0.4822695035460992, 0.49859943977591037, 0.5138121546961326, 0.532608695652174, 0.5469168900804289, 0.5653896961690885, 0.5796344647519581, 0.5891472868217055, 0.604591836734694, 0.6204287515762926, 0.6377171215880894, 0.6519607843137254, 0.6610169491525424, 0.6754176610978521, 0.6878680800942285, 0.7009345794392523, 0.7104959630911187, 0.7200000000000001, 0.7212741751990899, 0.7278911564625851, 0.7266591676040495, 0.735195530726257, 0.738359201773836, 0.743109151047409] 
	t3 = [0, 3.077807664871216, 6.034039497375488, 9.107590198516846, 12.087350606918335, 15.032663106918335, 18.010827779769897, 21.07312297821045, 24.009502172470093, 27.10159420967102, 30.076831817626953, 33.07246804237366, 36.133437395095825, 39.03732466697693, 42.056344747543335, 45.03371000289917, 48.12085008621216, 51.09833383560181, 54.14247274398804, 57.09210205078125, 60.045010805130005] 
	q3 = [0.37694704049844235, 0.3969465648854962, 0.4275037369207773, 0.4516129032258065, 0.4697406340057637, 0.4908321579689704, 0.5138121546961326, 0.5378378378378378, 0.5653896961690885, 0.5844155844155845, 0.6063694267515924, 0.6317103620474407, 0.6528117359413202, 0.6722689075630252, 0.6894117647058823, 0.7083333333333334, 0.7214611872146119, 0.725, 0.7289088863892013, 0.7369589345172032, 0.7458745874587458] 
	t4 = [0, 4.027574300765991, 8.091228723526001, 12.083025932312012, 16.0591721534729, 20.096832275390625, 24.06049609184265, 28.134953498840332, 32.07884645462036, 36.144378423690796, 40.06410765647888, 44.08945631980896, 48.02746224403381, 52.119338512420654, 56.056594371795654, 60.09865760803223] 
	q4 = [0.37694704049844235, 0.40606060606060607, 0.4454277286135693, 0.4697406340057637, 0.5, 0.5338753387533874, 0.570673712021136, 0.5945945945945947, 0.6273525721455457, 0.6536585365853659, 0.6809015421115066, 0.7038327526132403, 0.7214611872146119, 0.729119638826185, 0.7363737486095662, 0.7499999999999999] 
	t5 = [0, 5.11507773399353, 10.054290771484375, 15.11628007888794, 20.122812747955322, 25.13736844062805, 30.1292884349823, 35.117300510406494, 40.10908555984497, 45.067524671554565, 50.0397424697876, 55.13328409194946, 60.063607931137085] 
	q5 = [0.37694704049844235, 0.42042042042042044, 0.462882096069869, 0.4943502824858757, 0.5338753387533874, 0.5800524934383202, 0.6099110546378652, 0.6486486486486487, 0.6809015421115066, 0.7098265895953758, 0.7243735763097949, 0.735195530726257, 0.7472527472527472] 
	t6 = [0, 6.100088357925415, 12.008552551269531, 18.12270212173462, 24.095973014831543, 30.13725709915161, 36.03084468841553, 42.06811547279358, 48.12669277191162, 54.114787578582764, 60.10802507400513] 
	q6 = [0.37694704049844235, 0.43219076005961254, 0.4697406340057637, 0.5171939477303988, 0.5733157199471598, 0.6116751269035533, 0.6536585365853659, 0.6948356807511737, 0.7229190421892817, 0.7295173961840627, 0.7486278814489573] 
	t7 = [0, 7.018277883529663, 14.06328010559082, 21.054181575775146, 28.056635856628418, 35.07434964179993, 42.13855767250061, 49.12484860420227, 56.113914012908936] 
	q7 = [0.37694704049844235, 0.4385185185185185, 0.48794326241134744, 0.5390835579514826, 0.5971685971685973, 0.6503067484662577, 0.6948356807511737, 0.7243735763097949, 0.7377777777777779] 
	t8 = [0, 8.058363437652588, 16.04835033416748, 24.043442249298096, 32.00349545478821, 40.03200602531433, 48.07910633087158, 56.07204341888428] 
	q8 = [0.37694704049844235, 0.4477172312223859, 0.5006993006993007, 0.5714285714285715, 0.6273525721455457, 0.6809015421115066, 0.7214611872146119, 0.7377777777777779] 
	t9 = [0, 9.085291385650635, 18.12288546562195, 27.011159658432007, 36.06889295578003, 45.031240701675415, 54.149090051651] 
	q9 = [0.37694704049844235, 0.4538799414348462, 0.5206611570247933, 0.5888456549935148, 0.6536585365853659, 0.7104959630911187, 0.7309417040358744] 
	t10 = [0, 10.10382080078125, 20.06842803955078, 30.12014412879944, 40.0424165725708, 50.12991690635681, 60.06808638572693] 
	q10 = [0.37694704049844235, 0.46444121915820025, 0.5353260869565217, 0.6149936467598475, 0.6809015421115066, 0.7258248009101251, 0.7494505494505495] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.094465732574463, 2.0263113975524902, 3.0085794925689697, 4.0882627964019775, 5.039682626724243, 6.1090216636657715, 7.072022438049316, 8.004124402999878, 9.100784063339233, 10.058046579360962, 11.03628945350647, 12.140800476074219, 13.138168573379517, 14.089877367019653, 15.096659421920776, 16.020294189453125, 17.126732349395752, 18.05276370048523, 19.037261486053467, 20.000524759292603, 21.006611108779907, 22.078212022781372, 23.053704023361206, 24.131324291229248, 25.083258390426636, 26.06491446495056, 27.110350370407104, 28.061400890350342, 29.014323711395264, 30.12614393234253, 31.070946216583252, 32.14673113822937, 33.12237215042114, 34.08597421646118, 35.0397162437439, 36.11349415779114, 37.09611630439758, 38.01977562904358, 39.106988191604614, 40.070231676101685, 41.05124831199646, 42.000521421432495, 43.01438331604004, 44.10611081123352, 45.051305532455444, 46.037617444992065, 47.129162549972534, 48.12314534187317, 49.09778642654419, 50.096620321273804, 51.03468465805054, 52.142568826675415, 53.094605445861816, 54.05004668235779, 55.03825783729553, 56.11950993537903, 57.12469506263733, 58.07978940010071, 59.034218549728394, 60.13368320465088] 
	q1 = [0.35104669887278583, 0.3578274760383387, 0.36190476190476195, 0.36850393700787404, 0.3837753510140406, 0.3869969040247678, 0.3981623277182236, 0.4079147640791477, 0.41452344931921337, 0.41867469879518066, 0.42514970059880236, 0.4338781575037147, 0.4460856720827179, 0.44868035190615835, 0.45547445255474456, 0.4602026049204052, 0.4668587896253602, 0.47428571428571437, 0.4801136363636364, 0.48587570621468923, 0.4943820224719101, 0.504881450488145, 0.5076282940360609, 0.517193947730399, 0.529331514324693, 0.5326086956521738, 0.5425101214574899, 0.5483870967741936, 0.5527369826435248, 0.5577689243027888, 0.5718050065876152, 0.579292267365662, 0.5867014341590613, 0.5950840879689522, 0.5971685971685972, 0.6025641025641025, 0.6096938775510204, 0.6142131979695431, 0.6194690265486726, 0.6231155778894473, 0.6307884856070087, 0.6343283582089552, 0.6386138613861386, 0.6437346437346437, 0.6536585365853659, 0.6585662211421629, 0.6626360338573155, 0.6674698795180724, 0.6746411483253588, 0.6793802145411204, 0.6856465005931199, 0.6872037914691943, 0.6903073286052009, 0.6949352179034158, 0.6933019976498238, 0.6940211019929661, 0.6923976608187135, 0.6915017462165309, 0.694541231126597, 0.697566628041715, 0.6997690531177829] 
	t2 = [0, 2.1280927658081055, 4.137872934341431, 6.131240129470825, 8.107521533966064, 10.096288681030273, 12.117680311203003, 14.024646520614624, 16.097209215164185, 18.06818175315857, 20.085444450378418, 22.123040437698364, 24.0083429813385, 26.01446294784546, 28.000694274902344, 30.01547932624817, 32.13578987121582, 34.12792468070984, 36.13577127456665, 38.131404399871826, 40.12653422355652, 42.01861596107483, 44.07557415962219, 46.10474371910095, 48.126832008361816, 50.02908205986023, 52.03127336502075, 54.06614804267883, 56.062957525253296, 58.100534200668335, 60.00689435005188] 
	q2 = [0.35104669887278583, 0.36450079239302696, 0.38317757009345793, 0.3981623277182236, 0.41452344931921337, 0.42985074626865677, 0.44477172312223856, 0.45481049562682213, 0.46839080459770116, 0.48158640226628896, 0.5006993006993007, 0.5131034482758622, 0.5306122448979592, 0.5444743935309972, 0.5539280958721704, 0.5774278215223096, 0.5958549222797928, 0.6025641025641025, 0.6142131979695431, 0.6231155778894473, 0.63681592039801, 0.6420664206642066, 0.6585662211421629, 0.6690734055354993, 0.6793802145411204, 0.6872037914691943, 0.6949352179034158, 0.6923976608187135, 0.6938300349243307, 0.6990740740740742, 0.7004608294930874] 
	t3 = [0, 3.0244221687316895, 6.063011169433594, 9.067220211029053, 12.103476762771606, 15.07009243965149, 18.035884141921997, 21.071895360946655, 24.00851821899414, 27.12488603591919, 30.09357261657715, 33.10118865966797, 36.034170389175415, 39.08879852294922, 42.001729011535645, 45.12495803833008, 48.05642223358154, 51.0242657661438, 54.024263858795166, 57.120705127716064, 60.088704109191895] 
	q3 = [0.35104669887278583, 0.36850393700787404, 0.3981623277182236, 0.4210526315789474, 0.4477172312223859, 0.46531791907514447, 0.48158640226628896, 0.5083333333333334, 0.5326086956521738, 0.553475935828877, 0.581151832460733, 0.5997425997425998, 0.6159695817490494, 0.6317103620474408, 0.6470588235294117, 0.6658624849215923, 0.6833333333333333, 0.6933962264150944, 0.6923976608187135, 0.6983758700696056, 0.7034482758620689] 
	t4 = [0, 4.04699182510376, 8.126071691513062, 12.031235694885254, 16.041598081588745, 20.015295028686523, 24.015971183776855, 28.0488338470459, 32.10017704963684, 36.05625081062317, 40.103055238723755, 44.10433006286621, 48.01648163795471, 52.046170711517334, 56.04439067840576, 60.036704778671265] 
	q4 = [0.35104669887278583, 0.3800623052959502, 0.41389728096676737, 0.4477172312223859, 0.47345767575322817, 0.5027932960893855, 0.5326086956521738, 0.5630810092961487, 0.5968992248062016, 0.6159695817490494, 0.6394052044609666, 0.6626360338573155, 0.6833333333333333, 0.6933019976498238, 0.6953488372093023, 0.7042577675489068] 
	t5 = [0, 5.062410831451416, 10.002973794937134, 15.038542985916138, 20.097474098205566, 25.145358085632324, 30.126606702804565, 35.03649830818176, 40.026992082595825, 45.10632801055908, 50.09710931777954, 55.02753186225891, 60.09917068481445] 
	q5 = [0.35104669887278583, 0.3919753086419753, 0.42985074626865677, 0.4659913169319827, 0.504881450488145, 0.5405405405405405, 0.581913499344692, 0.6132315521628499, 0.6394052044609666, 0.6674698795180724, 0.6887573964497041, 0.6938300349243307, 0.7042577675489068] 
	t6 = [0, 6.1309545040130615, 12.120427131652832, 18.1036274433136, 24.04248070716858, 30.016013145446777, 36.015692472457886, 42.03795146942139, 48.04333782196045, 54.1353645324707, 60.06390333175659] 
	q6 = [0.35104669887278583, 0.3981623277182236, 0.4477172312223859, 0.48441926345609054, 0.5345997286295794, 0.581913499344692, 0.6159695817490494, 0.6528117359413204, 0.6848989298454221, 0.6931155192532088, 0.7042577675489068] 
	t7 = [0, 7.009806394577026, 14.039416790008545, 21.104503870010376, 28.109842538833618, 35.05502700805664, 42.10902142524719, 49.127866983413696, 56.06065535545349] 
	q7 = [0.35104669887278583, 0.4079147640791477, 0.45997088791848617, 0.5076282940360609, 0.5630810092961487, 0.6157760814249363, 0.6544566544566545, 0.6880189798339266, 0.6968641114982579] 
	t8 = [0, 8.053046941757202, 16.099292755126953, 24.070423364639282, 32.04911947250366, 40.1374294757843, 48.09230351448059, 56.0658016204834] 
	q8 = [0.35104669887278583, 0.4114977307110439, 0.47632711621233864, 0.5326086956521738, 0.5976714100905564, 0.6386138613861386, 0.6848989298454221, 0.6968641114982579] 
	t9 = [0, 9.08794903755188, 18.124085664749146, 27.14286184310913, 36.04793930053711, 45.07295203208923, 54.04603457450867] 
	q9 = [0.35104669887278583, 0.4210526315789474, 0.48725212464589235, 0.5508021390374331, 0.6185044359949303, 0.6682750301568156, 0.691588785046729] 
	t10 = [0, 10.101624965667725, 20.0464346408844, 30.05961322784424, 40.124568700790405, 50.01358485221863, 60.05589246749878] 
	q10 = [0.35104669887278583, 0.43219076005961254, 0.5062937062937062, 0.581913499344692, 0.6386138613861386, 0.6911242603550296, 0.7057471264367816] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.0976812839508057, 2.022731304168701, 3.143202543258667, 4.0652854442596436, 5.016737699508667, 6.092763900756836, 7.040043115615845, 8.137825965881348, 9.115129232406616, 10.071775913238525, 11.014064073562622, 12.119301795959473, 13.065923929214478, 14.021507024765015, 15.143095016479492, 16.110904455184937, 17.062516450881958, 18.130756855010986, 19.073512077331543, 20.14120864868164, 21.11528706550598, 22.03804612159729, 23.14327120780945, 24.126070022583008, 25.12706756591797, 26.098607540130615, 27.10707926750183, 28.094767332077026, 29.043405771255493, 30.030181169509888, 31.035634994506836, 32.024232149124146, 33.000967502593994, 34.07464933395386, 35.026015758514404, 36.108393907547, 37.04845833778381, 38.001479387283325, 39.096651554107666, 40.05663752555847, 41.00530028343201, 42.078362464904785, 43.05864763259888, 44.02706742286682, 45.15581727027893, 46.08682417869568, 47.086899280548096, 48.01524353027344, 49.11236357688904, 50.03777861595154, 51.01551294326782, 52.04149532318115, 53.01022553443909, 54.120025396347046, 55.07568168640137, 56.03099703788757, 57.12489366531372, 58.0871102809906, 59.074249267578125, 60.036991596221924] 
	q1 = [0.34782608695652173, 0.35747303543913717, 0.3644716692189893, 0.3708206686930091, 0.37821482602117995, 0.3898050974512744, 0.3988095238095238, 0.4053254437869822, 0.4140969162995594, 0.42043795620437957, 0.42608695652173917, 0.43227665706051877, 0.4444444444444445, 0.45106382978723397, 0.4535211267605634, 0.4587412587412587, 0.4666666666666667, 0.4772413793103449, 0.48285322359396443, 0.4884038199181447, 0.49526387009472256, 0.4952893674293405, 0.5013404825737265, 0.511318242343542, 0.5218543046357615, 0.5243741765480895, 0.5340314136125656, 0.5416666666666666, 0.5510996119016818, 0.5592783505154638, 0.5659411011523687, 0.5728900255754477, 0.579415501905972, 0.5876418663303908, 0.5939849624060151, 0.6044776119402985, 0.6131025957972805, 0.6199261992619927, 0.6268292682926829, 0.6343825665859564, 0.6394230769230769, 0.6475507765830346, 0.6539833531510106, 0.6595744680851063, 0.6650998824911868, 0.6697782963827305, 0.6744186046511628, 0.6789838337182448, 0.6850574712643679, 0.6902857142857144, 0.69327251995439, 0.6977272727272728, 0.7028248587570621, 0.7072072072072071, 0.7115600448933782, 0.7144456886898097, 0.71731843575419, 0.7216035634743876, 0.7236403995560488, 0.7278761061946902, 0.725468577728776] 
	t2 = [0, 2.1214852333068848, 4.093465089797974, 6.103769779205322, 8.075255393981934, 10.111451864242554, 12.107527732849121, 14.122554540634155, 16.05121660232544, 18.02030920982361, 20.13490915298462, 22.13786268234253, 24.017449855804443, 26.073341131210327, 28.022058486938477, 30.06842064857483, 32.01020956039429, 34.007073163986206, 36.13250017166138, 38.00342535972595, 40.12632441520691, 42.13124203681946, 44.01606225967407, 46.026917695999146, 48.0637104511261, 50.04529547691345, 52.141674518585205, 54.03055143356323, 56.03434777259827, 58.04462647438049, 60.07271862030029] 
	q2 = [0.34782608695652173, 0.36391437308868496, 0.377643504531722, 0.40118870728083206, 0.4164222873900293, 0.4312590448625181, 0.4472934472934473, 0.45569620253164556, 0.4687933425797504, 0.4821917808219178, 0.4966261808367071, 0.5053475935828877, 0.5238095238095238, 0.5378590078328982, 0.5548387096774194, 0.5728900255754477, 0.5858585858585859, 0.6027397260273972, 0.6199261992619927, 0.6343825665859564, 0.649164677804296, 0.6611570247933884, 0.6697782963827305, 0.6789838337182448, 0.6917808219178082, 0.6992054483541429, 0.7086614173228347, 0.7158836689038031, 0.7216035634743876, 0.72707182320442, 0.7260726072607261] 
	t3 = [0, 3.01993465423584, 6.0241539478302, 9.064539194107056, 12.026251077651978, 15.071160793304443, 18.01972985267639, 21.04291582107544, 24.060818433761597, 27.095415115356445, 30.071070909500122, 33.10256576538086, 36.0158588886261, 39.06734371185303, 42.09215426445007, 45.020148277282715, 48.00302767753601, 51.11703872680664, 54.14499378204346, 57.03033947944641, 60.005964040756226] 
	q3 = [0.34782608695652173, 0.3732928679817906, 0.40118870728083206, 0.4250363901018923, 0.4472934472934473, 0.46582984658298465, 0.4821917808219178, 0.49932885906040264, 0.523117569352708, 0.5510996119016818, 0.5728900255754477, 0.5957446808510638, 0.6216216216216216, 0.642685851318945, 0.6643109540636042, 0.6743916570104287, 0.6917808219178082, 0.7036199095022624, 0.71731843575419, 0.7278761061946902, 0.7260726072607261] 
	t4 = [0, 4.065378665924072, 8.00470232963562, 12.128142356872559, 16.065332412719727, 20.0320827960968, 24.116058826446533, 28.011955499649048, 32.094332695007324, 36.145864725112915, 40.083399534225464, 44.05436182022095, 48.01028060913086, 52.130191802978516, 56.05590510368347, 60.04714751243591] 
	q4 = [0.34782608695652173, 0.38009049773755654, 0.4164222873900293, 0.4472934472934473, 0.47368421052631576, 0.4959568733153638, 0.525065963060686, 0.5585585585585585, 0.5919395465994962, 0.625, 0.6507747318235995, 0.6744186046511628, 0.69327251995439, 0.7080045095828637, 0.7236403995560488, 0.7260726072607261] 
	t5 = [0, 5.075068235397339, 10.13399624824524, 15.065855979919434, 20.119234323501587, 25.106404781341553, 30.08947253227234, 35.11366844177246, 40.067827463150024, 45.10824799537659, 50.0185911655426, 55.136189222335815, 60.10468053817749] 
	q5 = [0.34782608695652173, 0.39162929745889385, 0.4335260115606937, 0.4651810584958217, 0.4959568733153638, 0.5373525557011797, 0.576530612244898, 0.6165228113440197, 0.6531585220500595, 0.6789838337182448, 0.7006802721088435, 0.7216035634743876, 0.7268722466960352] 
	t6 = [0, 6.1422600746154785, 12.132311820983887, 18.085434198379517, 24.005265951156616, 30.02324080467224, 36.146008014678955, 42.04682683944702, 48.14171385765076, 54.055763483047485, 60.046095848083496] 
	q6 = [0.34782608695652173, 0.400593471810089, 0.4472934472934473, 0.48700410396716826, 0.525065963060686, 0.576530612244898, 0.6266829865361077, 0.6658823529411765, 0.69327251995439, 0.7166853303471444, 0.7268722466960352] 
	t7 = [0, 7.029880523681641, 14.092094421386719, 21.143258810043335, 28.057887315750122, 35.08368539810181, 42.13649272918701, 49.09513545036316, 56.09516358375549] 
	q7 = [0.34782608695652173, 0.4076809453471197, 0.46132208157524623, 0.5013404825737265, 0.5637065637065638, 0.6172839506172839, 0.6650998824911868, 0.6992054483541429, 0.7250554323725054] 
	t8 = [0, 8.037404775619507, 16.006443738937378, 24.08418107032776, 32.167306900024414, 40.08230185508728, 48.082916498184204, 56.01757287979126] 
	q8 = [0.34782608695652173, 0.4158125915080527, 0.477115117891817, 0.5270092226613966, 0.592964824120603, 0.6531585220500595, 0.69327251995439, 0.725860155382908] 
	t9 = [0, 9.106136560440063, 18.009929180145264, 27.119649410247803, 36.050872802734375, 45.02659773826599, 54.10005569458008] 
	q9 = [0.34782608695652173, 0.4250363901018923, 0.4876712328767123, 0.55627425614489, 0.6266829865361077, 0.6789838337182448, 0.7181208053691275] 
	t10 = [0, 10.137340068817139, 20.106638431549072, 30.057682991027832, 40.044904470443726, 50.00606369972229, 60.11399054527283] 
	q10 = [0.34782608695652173, 0.43290043290043295, 0.4959568733153638, 0.578005115089514, 0.6531585220500595, 0.7021517553793886, 0.7282728272827284] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.1138641834259033, 2.0380332469940186, 3.130195140838623, 4.097579717636108, 5.079263925552368, 6.012392044067383, 7.122800827026367, 8.150636672973633, 9.03694772720337, 10.000097513198853, 11.129636287689209, 12.070924758911133, 13.020346879959106, 14.012717962265015, 15.003509044647217, 16.091875076293945, 17.052412033081055, 18.124932050704956, 19.09995698928833, 20.025551319122314, 21.122864246368408, 22.0851149559021, 23.074989557266235, 24.040486812591553, 25.095537662506104, 26.029088973999023, 27.00832509994507, 28.037664890289307, 29.02405858039856, 30.129234790802002, 31.075700759887695, 32.03019571304321, 33.13155674934387, 34.06425380706787, 35.020331621170044, 36.09846544265747, 37.08231520652771, 38.06845211982727, 39.02423596382141, 40.10674715042114, 41.108957052230835, 42.02915930747986, 43.145711183547974, 44.071587562561035, 45.020918130874634, 46.03841853141785, 47.02767729759216, 48.12577414512634, 49.07183027267456, 50.03271746635437, 51.01138162612915, 52.09278678894043, 53.071288108825684, 54.00050640106201, 55.12897610664368, 56.05230093002319, 57.02279806137085, 58.089330434799194, 59.06661057472229, 60.032607555389404] 
	q1 = [0.37617554858934166, 0.3862928348909657, 0.3931888544891641, 0.4000000000000001, 0.4042879019908117, 0.4127465857359635, 0.4223227752639517, 0.431784107946027, 0.4411326378539493, 0.44674556213017746, 0.4552129221732746, 0.4635568513119534, 0.46888567293777134, 0.4697406340057637, 0.47564469914040114, 0.4801136363636363, 0.4908321579689704, 0.49579831932773116, 0.5027777777777779, 0.5082872928176796, 0.5116918844566712, 0.5211459754433834, 0.5311653116531165, 0.534412955465587, 0.5403225806451613, 0.5500667556742322, 0.557029177718833, 0.570673712021136, 0.5781865965834428, 0.5830065359477125, 0.5914396887159533, 0.5979381443298969, 0.6051282051282052, 0.6132315521628499, 0.6212121212121213, 0.6256281407035176, 0.6334164588528678, 0.6385093167701863, 0.6427688504326329, 0.6511056511056511, 0.6585067319461445, 0.6634146341463414, 0.6690909090909091, 0.6778846153846154, 0.6842105263157895, 0.6865315852205006, 0.6887573964497041, 0.6988235294117647, 0.702576112412178, 0.7079439252336449, 0.710128055878929, 0.7137891077636153, 0.7152777777777778, 0.7174163783160323, 0.7188940092165897, 0.7218390804597701, 0.7262313860252004, 0.7237442922374429, 0.7258248009101251, 0.7256235827664399, 0.727683615819209] 
	t2 = [0, 2.1312530040740967, 4.001946926116943, 6.019499778747559, 8.011498928070068, 10.089561223983765, 12.058786153793335, 14.064106225967407, 16.10086989402771, 18.070881128311157, 20.071370124816895, 22.043593406677246, 24.10584044456482, 26.027830600738525, 28.118648767471313, 30.00965189933777, 32.02215814590454, 34.00218749046326, 36.12162518501282, 38.03956484794617, 40.056365728378296, 42.06614875793457, 44.06532025337219, 46.14266872406006, 48.010520219802856, 50.06203055381775, 52.07627749443054, 54.07681059837341, 56.085203647613525, 58.082834005355835, 60.124544620513916] 
	q2 = [0.37617554858934166, 0.39009287925696595, 0.4042879019908117, 0.42469879518072295, 0.4434523809523809, 0.4545454545454545, 0.46753246753246747, 0.47863247863247854, 0.4964936886395512, 0.5089903181189488, 0.5191256830601092, 0.534412955465587, 0.5500667556742322, 0.5687830687830688, 0.5848563968668408, 0.5979381443298969, 0.6132315521628499, 0.6273525721455457, 0.6385093167701863, 0.6535626535626536, 0.6642335766423357, 0.6794717887154862, 0.6872770511296077, 0.7018779342723006, 0.710955710955711, 0.7152777777777778, 0.7188940092165897, 0.7233065442020666, 0.7258248009101251, 0.7285067873303167, 0.7328072153325816] 
	t3 = [0, 3.0280635356903076, 6.056605577468872, 9.095627784729004, 12.071170091629028, 15.119091033935547, 18.046999216079712, 21.09674334526062, 24.01699161529541, 27.051212072372437, 30.13049578666687, 33.00156259536743, 36.013317346572876, 39.11115288734436, 42.025060415267944, 45.05809283256531, 48.04843831062317, 51.12119174003601, 54.156821727752686, 57.04525589942932, 60.1300892829895] 
	q3 = [0.37617554858934166, 0.3969230769230769, 0.4270676691729323, 0.45132743362831856, 0.4697406340057637, 0.4887005649717514, 0.5089903181189488, 0.5311653116531165, 0.5527369826435247, 0.5797101449275363, 0.5997425997425997, 0.624685138539043, 0.640198511166253, 0.6617826617826618, 0.6826347305389222, 0.6972909305064783, 0.710955710955711, 0.7174163783160323, 0.725400457665904, 0.7270668176670442, 0.7342342342342342] 
	t4 = [0, 4.069843292236328, 8.001073122024536, 12.013843059539795, 16.1048424243927, 20.064512968063354, 24.029237508773804, 28.118285655975342, 32.03703999519348, 36.107566118240356, 40.112783670425415, 44.09555196762085, 48.102057218551636, 52.1080687046051, 56.055463790893555, 60.040846824645996] 
	q4 = [0.37617554858934166, 0.40672782874617736, 0.4427934621099554, 0.4697406340057637, 0.49579831932773116, 0.5211459754433834, 0.5527369826435247, 0.5900783289817233, 0.6159695817490494, 0.6410891089108911, 0.6690909090909091, 0.6887573964497041, 0.7124563445867288, 0.720368239355581, 0.7264472190692395, 0.735658042744657] 
	t5 = [0, 5.099141836166382, 10.102050542831421, 15.121031522750854, 20.14081358909607, 25.00423288345337, 30.012704849243164, 35.05651926994324, 40.09401750564575, 45.13507795333862, 50.07413649559021, 55.11108613014221, 60.0166437625885] 
	q5 = [0.37617554858934166, 0.41515151515151516, 0.4597364568081991, 0.4915254237288136, 0.5231607629427792, 0.56158940397351, 0.6038709677419356, 0.6385093167701863, 0.6707021791767555, 0.6988235294117647, 0.7161066048667439, 0.7258248009101251, 0.7379077615298087] 
	t6 = [0, 6.098129749298096, 12.142407894134521, 18.007213354110718, 24.100626707077026, 30.128790616989136, 36.10999608039856, 42.02890872955322, 48.11532258987427, 54.019392251968384, 60.06450009346008] 
	q6 = [0.37617554858934166, 0.4270676691729323, 0.47041847041847046, 0.5082872928176796, 0.5546666666666666, 0.6056701030927836, 0.6427688504326329, 0.6833930704898448, 0.710955710955711, 0.7245714285714285, 0.7379077615298087] 
	t7 = [0, 7.030995845794678, 14.136730194091797, 21.12796688079834, 28.068866729736328, 35.07735085487366, 42.0537314414978, 49.00497913360596, 56.118295431137085] 
	q7 = [0.37617554858934166, 0.4341317365269461, 0.48295454545454547, 0.530446549391069, 0.5908496732026144, 0.6385093167701863, 0.684964200477327, 0.7131242740998838, 0.7256235827664399] 
	t8 = [0, 8.025344371795654, 16.092523097991943, 24.089942693710327, 32.068989753723145, 40.000372648239136, 48.07531952857971, 56.06777286529541] 
	q8 = [0.37617554858934166, 0.4427934621099554, 0.49859943977591037, 0.5565912117177098, 0.6235741444866921, 0.6723095525997582, 0.7124563445867288, 0.7256235827664399] 
	t9 = [0, 9.072764873504639, 18.12762761116028, 27.0376193523407, 36.10818028450012, 45.00552272796631, 54.025389671325684] 
	q9 = [0.37617554858934166, 0.45360824742268047, 0.5082872928176796, 0.5815789473684211, 0.6427688504326329, 0.6988235294117647, 0.7239404352806414] 
	t10 = [0, 10.135797262191772, 20.07272505760193, 30.00538969039917, 40.02286100387573, 50.020885705947876, 60.068278312683105] 
	q10 = [0.37617554858934166, 0.4619883040935672, 0.5258855585831064, 0.6056701030927836, 0.6739130434782609, 0.7152777777777778, 0.738496071829405] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.097196102142334, 2.022329092025757, 3.1148998737335205, 4.05005955696106, 5.011842250823975, 6.082965612411499, 7.024119138717651, 8.124852895736694, 9.072354793548584, 10.03190803527832, 11.133162498474121, 12.067205429077148, 13.015037298202515, 14.020829200744629, 15.111647129058838, 16.037176609039307, 17.000354766845703, 18.08002281188965, 19.055608987808228, 20.123398065567017, 21.083953619003296, 22.009601831436157, 23.024182319641113, 24.10521650314331, 25.139465808868408, 26.131520748138428, 27.135499238967896, 28.05672526359558, 29.026901721954346, 30.121910572052002, 31.067631244659424, 32.02129411697388, 33.112988233566284, 34.07165503501892, 35.01579308509827, 36.11867427825928, 37.102083921432495, 38.02965784072876, 39.1226646900177, 40.07625770568848, 41.02479910850525, 42.01642084121704, 43.107876777648926, 44.069793462753296, 45.05361580848694, 46.01192355155945, 47.11571955680847, 48.0702018737793, 49.02152967453003, 50.011735916137695, 51.108304023742676, 52.06402039527893, 53.038421630859375, 54.144160985946655, 55.08818292617798, 56.04669189453125, 57.139322996139526, 58.07072377204895, 59.08492684364319, 60.012048959732056] 
	q1 = [0.3630769230769231, 0.37251908396946565, 0.37936267071320184, 0.3855421686746988, 0.3916292974588939, 0.39821693907875183, 0.406480117820324, 0.4181286549707602, 0.42608695652173906, 0.43578643578643583, 0.44412607449856734, 0.4488636363636364, 0.4535211267605634, 0.4600280504908836, 0.4673157162726008, 0.47790055248618785, 0.48626373626373626, 0.4918032786885246, 0.4993215739484396, 0.5094339622641509, 0.518716577540107, 0.5285524568393095, 0.5389696169088507, 0.5504587155963303, 0.556135770234987, 0.5673575129533679, 0.5710594315245479, 0.576923076923077, 0.5798212005108557, 0.5877862595419847, 0.5952080706179067, 0.6015037593984962, 0.6119402985074628, 0.6163366336633663, 0.6257668711656442, 0.6275946275946277, 0.6318347509113001, 0.6384522370012092, 0.6409638554216868, 0.6434573829531812, 0.6507747318235997, 0.653206650831354, 0.657210401891253, 0.6619718309859155, 0.6682242990654205, 0.6736353077816493, 0.6789838337182448, 0.6865671641791045, 0.6909920182440137, 0.6939704209328782, 0.6946651532349603, 0.6976217440543602, 0.6990950226244345, 0.7042889390519187, 0.7040358744394617, 0.7054871220604704, 0.7098214285714286, 0.7119021134593992, 0.7147613762486126, 0.7168141592920354, 0.7210584343991181] 
	t2 = [0, 2.1252405643463135, 4.100848436355591, 6.077343463897705, 8.068849325180054, 10.07423210144043, 12.08484959602356, 14.093456506729126, 16.101609230041504, 18.089890718460083, 20.09753441810608, 22.060423135757446, 24.10635995864868, 26.045998096466064, 28.109291315078735, 30.13111186027527, 32.12455940246582, 34.120627880096436, 36.131030321121216, 38.143468141555786, 40.134361743927, 42.029518365859985, 44.02036952972412, 46.03888177871704, 48.05144715309143, 50.079527139663696, 52.122045040130615, 54.14456367492676, 56.000900745391846, 58.12388324737549, 60.00145769119263] 
	q2 = [0.3630769230769231, 0.3787878787878788, 0.3916292974588939, 0.40882352941176475, 0.42608695652173906, 0.44571428571428573, 0.4556962025316456, 0.47222222222222227, 0.48834019204389567, 0.503382949932341, 0.5226666666666667, 0.5466491458607096, 0.5617685305591678, 0.5758354755784062, 0.5877862595419847, 0.5997490589711417, 0.6163366336633663, 0.6275946275946277, 0.6384522370012092, 0.6434573829531812, 0.653206650831354, 0.6619718309859155, 0.6751740139211136, 0.6880733944954128, 0.6939704209328782, 0.6976217440543602, 0.7049549549549549, 0.7083798882681563, 0.7119021134593992, 0.7182320441988951, 0.723076923076923] 
	t3 = [0, 3.035460948944092, 6.030137062072754, 9.018876791000366, 12.083927869796753, 15.125515460968018, 18.00448751449585, 21.050411701202393, 24.10017967224121, 27.004648685455322, 30.09871244430542, 33.14282488822937, 36.07195854187012, 39.11585974693298, 42.05152177810669, 45.06480383872986, 48.024455070495605, 51.080620765686035, 54.04455018043518, 57.094693422317505, 60.1411988735199] 
	q3 = [0.3630769230769231, 0.3855421686746988, 0.40882352941176475, 0.437410071942446, 0.4556962025316456, 0.4848484848484848, 0.503382949932341, 0.537037037037037, 0.5636363636363636, 0.5805626598465473, 0.6040100250626567, 0.6282208588957056, 0.6400966183574881, 0.6523809523809524, 0.6666666666666666, 0.6835443037974682, 0.6954545454545454, 0.7042889390519187, 0.7098214285714286, 0.7153931339977853, 0.7244785949506037] 
	t4 = [0, 4.04047966003418, 8.100138902664185, 12.072512865066528, 16.004514694213867, 20.089962005615234, 24.032937049865723, 28.12493133544922, 32.14277911186218, 36.08337712287903, 40.060601472854614, 44.02926540374756, 48.00304317474365, 52.16368365287781, 56.11098384857178, 60.08417797088623] 
	q4 = [0.3630769230769231, 0.3916292974588939, 0.4289855072463768, 0.4556962025316456, 0.49108367626886146, 0.5246338215712384, 0.5636363636363636, 0.5877862595419847, 0.619753086419753, 0.6417370325693607, 0.6563981042654029, 0.6782407407407406, 0.6962457337883959, 0.7048260381593715, 0.7147613762486126, 0.7244785949506037] 
	t5 = [0, 5.091423749923706, 10.03400444984436, 15.017385721206665, 20.00395154953003, 25.13224768638611, 30.060240745544434, 35.07656764984131, 40.1022789478302, 45.13351011276245, 50.05141830444336, 55.115553855895996, 60.11063575744629] 
	q5 = [0.3630769230769231, 0.40296296296296297, 0.44571428571428573, 0.4848484848484848, 0.5246338215712384, 0.5684754521963824, 0.6057571964956195, 0.6359223300970874, 0.655621301775148, 0.6865671641791045, 0.6990950226244345, 0.7126948775055679, 0.7258771929824561] 
	t6 = [0, 6.1411755084991455, 12.111478328704834, 18.03827977180481, 24.023260831832886, 30.085563898086548, 36.00496578216553, 42.07619595527649, 48.05122947692871, 54.00123715400696, 60.09445309638977] 
	q6 = [0.3630769230769231, 0.4111600587371513, 0.4556962025316456, 0.5054054054054054, 0.5636363636363636, 0.6082603254067586, 0.6441495778045839, 0.6682242990654205, 0.6954545454545454, 0.7112597547380156, 0.7258771929824561] 
	t7 = [0, 7.036041498184204, 14.075426816940308, 21.01758885383606, 28.127086877822876, 35.079171657562256, 42.12902641296387, 49.11196851730347, 56.12738347053528] 
	q7 = [0.3630769230769231, 0.41982507288629733, 0.4743411927877948, 0.537037037037037, 0.5877862595419847, 0.6375757575757576, 0.6682242990654205, 0.6984126984126984, 0.7169811320754718] 
	t8 = [0, 8.045032262802124, 16.008735179901123, 24.09971022605896, 32.073882818222046, 40.12342190742493, 48.051974058151245, 56.126595973968506] 
	q8 = [0.3630769230769231, 0.4289855072463768, 0.49041095890410963, 0.5617685305591678, 0.6246913580246913, 0.657210401891253, 0.6969353007945516, 0.7169811320754718] 
	t9 = [0, 9.10850477218628, 18.1184344291687, 27.028462886810303, 36.09555625915527, 45.141925573349, 54.01051092147827] 
	q9 = [0.3630769230769231, 0.437410071942446, 0.508108108108108, 0.5831202046035805, 0.6441495778045839, 0.6872852233676976, 0.7112597547380156] 
	t10 = [0, 10.144946098327637, 20.06795334815979, 30.110475301742554, 40.072598695755005, 50.06623554229736, 60.01343274116516] 
	q10 = [0.3630769230769231, 0.4450784593437946, 0.5272969374167776, 0.6082603254067586, 0.6579881656804735, 0.7006802721088435, 0.7258771929824561] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.102147102355957, 2.026860237121582, 3.126110553741455, 4.0542893409729, 5.012324571609497, 6.094643831253052, 7.0495686531066895, 8.128922462463379, 9.104789733886719, 10.096739768981934, 11.04269289970398, 12.115821838378906, 13.02171516418457, 14.09221887588501, 15.041043996810913, 16.140684366226196, 17.1157968044281, 18.04026985168457, 19.12572431564331, 20.113125562667847, 21.090335369110107, 22.048487901687622, 23.028244256973267, 24.0943021774292, 25.095622301101685, 26.08089852333069, 27.025846481323242, 28.014349699020386, 29.107850074768066, 30.034731149673462, 31.045907974243164, 32.00403571128845, 33.133880376815796, 34.092859506607056, 35.032591819763184, 36.13468360900879, 37.08379030227661, 38.00591993331909, 39.09456491470337, 40.05419111251831, 41.01280117034912, 42.00748062133789, 43.01753377914429, 44.09010362625122, 45.061519622802734, 46.011436462402344, 47.11884021759033, 48.0483078956604, 49.14489817619324, 50.132590532302856, 51.08686876296997, 52.0110068321228, 53.14532279968262, 54.10403823852539, 55.0527184009552, 56.12460732460022, 57.12618970870972, 58.047460079193115, 59.13808822631836, 60.096633434295654] 
	q1 = [0.3563402889245586, 0.3688394276629571, 0.37914691943127965, 0.3836477987421384, 0.39313572542901715, 0.4031007751937984, 0.41230769230769226, 0.4213740458015268, 0.4279210925644917, 0.43504531722054385, 0.4347826086956522, 0.44576523031203563, 0.4503703703703704, 0.45227606461086645, 0.4635568513119534, 0.4680232558139535, 0.47330447330447334, 0.48068669527896996, 0.4864864864864865, 0.49152542372881364, 0.5021037868162693, 0.5076708507670852, 0.5172890733056708, 0.52400548696845, 0.5300546448087431, 0.5447154471544716, 0.5537634408602151, 0.5622489959839356, 0.5683930942895087, 0.5770750988142292, 0.5842105263157896, 0.5942408376963352, 0.5945241199478487, 0.6036269430051814, 0.6118251928020566, 0.615581098339719, 0.6243654822335025, 0.6279949558638083, 0.6314465408805031, 0.64, 0.6442786069651741, 0.6493184634448576, 0.6576354679802955, 0.6658506731946144, 0.6691086691086692, 0.6747572815533982, 0.6803377563329313, 0.6835138387484958, 0.6913875598086124, 0.6952380952380953, 0.6975088967971531, 0.7005917159763314, 0.7021276595744681, 0.7051886792452831, 0.7089201877934272, 0.7072599531615925, 0.7126168224299065, 0.7147846332945286, 0.7169373549883991, 0.7190751445086705, 0.7220299884659745] 
	t2 = [0, 2.1244049072265625, 4.09131932258606, 6.069143533706665, 8.057498693466187, 10.08558440208435, 12.095053672790527, 14.040898084640503, 16.01480221748352, 18.061940670013428, 20.05306363105774, 22.09302592277527, 24.125102281570435, 26.042352199554443, 28.112648010253906, 30.09538960456848, 32.02397179603577, 34.05580735206604, 36.0743408203125, 38.05797505378723, 40.05713677406311, 42.08039307594299, 44.11313819885254, 46.00201439857483, 48.12603163719177, 50.04587149620056, 52.05002164840698, 54.05085802078247, 56.04077219963074, 58.06239604949951, 60.058653831481934] 
	q2 = [0.3563402889245586, 0.37914691943127965, 0.3956386292834891, 0.4147465437788019, 0.4279210925644917, 0.43712574850299407, 0.4526627218934911, 0.4635568513119534, 0.47619047619047616, 0.4879432624113475, 0.5083798882681564, 0.5206611570247933, 0.5387755102040815, 0.5583892617449664, 0.5759577278731837, 0.5923984272608126, 0.6033810143042914, 0.6163682864450128, 0.6262626262626263, 0.64, 0.6493184634448576, 0.667481662591687, 0.6763636363636363, 0.6835138387484958, 0.6952380952380953, 0.7005917159763314, 0.7067137809187278, 0.711111111111111, 0.7162790697674418, 0.7220299884659745, 0.7241379310344828] 
	t3 = [0, 3.044874906539917, 6.0492753982543945, 9.043349504470825, 12.142759799957275, 15.023291826248169, 18.04384994506836, 21.111170768737793, 24.0769464969635, 27.103217363357544, 30.166226625442505, 33.07996129989624, 36.02122783660889, 39.08811712265015, 42.08465576171875, 45.119582414627075, 48.026434659957886, 51.05115509033203, 54.08905863761902, 57.05806064605713, 60.053725719451904] 
	q3 = [0.3563402889245586, 0.3836477987421384, 0.4147465437788019, 0.43373493975903615, 0.45199409158050224, 0.47024673439767783, 0.49078014184397173, 0.518005540166205, 0.5387755102040815, 0.5691489361702127, 0.5942408376963352, 0.6143958868894601, 0.6279949558638083, 0.6459627329192548, 0.667481662591687, 0.6819277108433734, 0.6967895362663495, 0.7036599763872491, 0.7126168224299065, 0.7175925925925928, 0.7233065442020665] 
	t4 = [0, 4.074731349945068, 8.11109972000122, 12.0523841381073, 16.11091375350952, 20.08427095413208, 24.085190773010254, 28.144587993621826, 32.01080584526062, 36.042418003082275, 40.135952949523926, 44.01150631904602, 48.138705253601074, 52.131598472595215, 56.05581307411194, 60.14145565032959] 
	q4 = [0.3563402889245586, 0.3956386292834891, 0.43030303030303035, 0.4526627218934911, 0.4776978417266186, 0.5076708507670852, 0.5434782608695652, 0.5804749340369393, 0.6070038910505837, 0.6279949558638083, 0.651851851851852, 0.6771463119709795, 0.6975088967971531, 0.7089201877934272, 0.7169373549883991, 0.7233065442020665] 
	t5 = [0, 5.096698999404907, 10.057219505310059, 15.06477689743042, 20.038793325424194, 25.0942542552948, 30.117836475372314, 35.034220933914185, 40.06381440162659, 45.13260531425476, 50.007598876953125, 55.03224802017212, 60.09026050567627] 
	q5 = [0.3563402889245586, 0.4024767801857585, 0.4417910447761194, 0.47383720930232565, 0.5076708507670852, 0.5513513513513514, 0.5942408376963352, 0.6286438529784537, 0.651851851851852, 0.6835138387484958, 0.7021276595744681, 0.7162790697674418, 0.7233065442020665] 
	t6 = [0, 6.122634410858154, 12.123998165130615, 18.11178755760193, 24.053311824798584, 30.111255407333374, 36.06766200065613, 42.102887868881226, 48.122185468673706, 54.02820611000061, 60.1159086227417] 
	q6 = [0.3563402889245586, 0.4171779141104295, 0.4542772861356933, 0.49291784702549574, 0.5434782608695652, 0.5968586387434556, 0.6305170239596469, 0.667481662591687, 0.6975088967971531, 0.7126168224299065, 0.724770642201835] 
	t7 = [0, 7.02576470375061, 14.022400140762329, 21.030001878738403, 28.118833541870117, 35.03116536140442, 42.04196333885193, 49.06503248214722, 56.14851641654968] 
	q7 = [0.3563402889245586, 0.42378048780487804, 0.4700729927007299, 0.518005540166205, 0.582010582010582, 0.6286438529784537, 0.667481662591687, 0.7005917159763314, 0.7175925925925928] 
	t8 = [0, 8.060060739517212, 16.00044846534729, 24.148401737213135, 32.0048406124115, 40.13556408882141, 48.034701347351074, 56.04485082626343] 
	q8 = [0.3563402889245586, 0.43030303030303035, 0.481962481962482, 0.5434782608695652, 0.6113989637305699, 0.655980271270037, 0.6975088967971531, 0.7175925925925928] 
	t9 = [0, 9.095786809921265, 18.08463764190674, 27.16002607345581, 36.09102487564087, 45.01347517967224, 54.025593996047974] 
	q9 = [0.3563402889245586, 0.43308270676691735, 0.4943181818181818, 0.574468085106383, 0.6313131313131314, 0.6835138387484958, 0.7117852975495915] 
	t10 = [0, 10.097745895385742, 20.083782196044922, 30.03555917739868, 40.139747619628906, 50.028400897979736, 60.13423418998718] 
	q10 = [0.3563402889245586, 0.444113263785395, 0.5146853146853148, 0.5931758530183727, 0.6600985221674877, 0.7021276595744681, 0.7262313860252004] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.106161117553711, 2.041574001312256, 3.130689859390259, 4.061336517333984, 5.015340805053711, 6.089142799377441, 7.038455963134766, 8.110889434814453, 9.060950517654419, 10.05217456817627, 11.06612491607666, 12.005393981933594, 13.129602670669556, 14.083700895309448, 15.037629842758179, 16.114276885986328, 17.09664559364319, 18.08433723449707, 19.03763699531555, 20.023579835891724, 21.010253190994263, 22.087918519973755, 23.06469750404358, 24.02039361000061, 25.114712238311768, 26.135636568069458, 27.063472747802734, 28.048314332962036, 29.021892547607422, 30.089534521102905, 31.032411813735962, 32.13587546348572, 33.09555244445801, 34.01784348487854, 35.10680961608887, 36.0604944229126, 37.03734564781189, 38.14916753768921, 39.09720587730408, 40.06001353263855, 41.07053565979004, 42.009459495544434, 43.00130891799927, 44.07763886451721, 45.088666915893555, 46.010825634002686, 47.13494372367859, 48.122910499572754, 49.071046352386475, 50.02315831184387, 51.14164924621582, 52.07299304008484, 53.020588636398315, 54.12973093986511, 55.076911211013794, 56.03643321990967, 57.074236154556274, 58.00260376930237, 59.129244804382324, 60.12090039253235] 
	q1 = [0.3508771929824561, 0.36075949367088606, 0.36792452830188677, 0.378125, 0.3875968992248062, 0.39445300462249616, 0.40916030534351144, 0.4109589041095891, 0.4205748865355522, 0.42706766917293243, 0.4334828101644246, 0.4437869822485207, 0.4552129221732746, 0.462882096069869, 0.4704184704184704, 0.479196556671449, 0.4857142857142857, 0.4907801418439717, 0.49647390691114246, 0.4985915492957746, 0.5049088359046282, 0.5153203342618385, 0.5200553250345782, 0.5274725274725275, 0.5395095367847411, 0.5508819538670285, 0.5587044534412955, 0.5606469002695418, 0.570281124497992, 0.5771276595744681, 0.5820105820105821, 0.5876152832674573, 0.5968586387434556, 0.5997392438070405, 0.6062176165803109, 0.6143958868894601, 0.617948717948718, 0.6257982120051085, 0.6319796954314721, 0.638888888888889, 0.6457286432160803, 0.6499999999999999, 0.654228855721393, 0.6617283950617284, 0.6699386503067485, 0.6731946144430844, 0.6772228989037758, 0.6812121212121214, 0.6859903381642513, 0.6891566265060242, 0.6907340553549941, 0.6946107784431138, 0.6977299880525687, 0.698450536352801, 0.6998813760379596, 0.7021276595744681, 0.7021276595744681, 0.7058823529411764, 0.7080890973036342, 0.7072599531615925, 0.7117852975495917] 
	t2 = [0, 2.1337735652923584, 4.120574235916138, 6.097674131393433, 8.095181226730347, 10.071941375732422, 12.025575399398804, 14.032455444335938, 16.04077386856079, 18.046571731567383, 20.076629877090454, 22.134722471237183, 24.015944004058838, 26.134218454360962, 28.021512985229492, 30.069171667099, 32.05283570289612, 34.015743017196655, 36.0187509059906, 38.060590744018555, 40.06318950653076, 42.08872723579407, 44.09150576591492, 46.1205792427063, 48.02008128166199, 50.028095960617065, 52.0411958694458, 54.04839038848877, 56.05842590332031, 58.01862835884094, 60.04942464828491] 
	q2 = [0.3508771929824561, 0.37048665620094196, 0.3869969040247678, 0.40916030534351144, 0.42296072507552873, 0.43815201192250375, 0.4574780058651026, 0.4733044733044733, 0.49002849002848997, 0.4936530324400564, 0.5132496513249651, 0.5234159779614326, 0.5469387755102041, 0.5606469002695418, 0.576, 0.5857519788918205, 0.597911227154047, 0.6108247422680413, 0.6257982120051085, 0.6371681415929203, 0.6499999999999999, 0.6633785450061652, 0.6723716381418092, 0.6828087167070218, 0.6891566265060242, 0.6946107784431138, 0.7, 0.7021276595744681, 0.7058823529411764, 0.7072599531615925, 0.7177700348432055] 
	t3 = [0, 3.03108286857605, 6.043783664703369, 9.045530796051025, 12.087451696395874, 15.103045225143433, 18.143128395080566, 21.07740330696106, 24.058784008026123, 27.065943479537964, 30.13758134841919, 33.143959283828735, 36.01783275604248, 39.12882375717163, 42.055989027023315, 45.01876354217529, 48.03656077384949, 51.104888677597046, 54.04873991012573, 57.02463459968567, 60.1381516456604] 
	q3 = [0.3508771929824561, 0.378125, 0.40916030534351144, 0.43413173652694614, 0.4604105571847507, 0.48137535816618904, 0.49577464788732395, 0.5180055401662049, 0.5489130434782609, 0.5721925133689839, 0.5876152832674573, 0.6098191214470284, 0.6292993630573249, 0.6466165413533834, 0.6650246305418719, 0.6812121212121214, 0.6891566265060242, 0.698450536352801, 0.7021276595744681, 0.7072599531615925, 0.7192575406032483] 
	t4 = [0, 4.037462472915649, 8.07499384880066, 12.018518447875977, 16.06703209877014, 20.023467540740967, 24.07206153869629, 28.10285472869873, 32.08688831329346, 36.02943253517151, 40.01304221153259, 44.02487373352051, 48.148022174835205, 52.02765941619873, 56.09780263900757, 60.14418077468872] 
	q4 = [0.3508771929824561, 0.3869969040247678, 0.42296072507552873, 0.4604105571847507, 0.492176386913229, 0.5132496513249651, 0.5508819538670285, 0.5797872340425532, 0.6015625, 0.6292993630573249, 0.6533665835411471, 0.675609756097561, 0.6907340553549941, 0.6998813760379596, 0.7065727699530516, 0.7177700348432055] 
	t5 = [0, 5.065036773681641, 10.004190921783447, 15.041524887084961, 20.106062650680542, 25.14052724838257, 30.00446319580078, 35.10369324684143, 40.14169096946716, 45.05695819854736, 50.0652437210083, 55.09424543380737, 60.038342237472534] 
	q5 = [0.3508771929824561, 0.39938556067588327, 0.4398216939078752, 0.48493543758967, 0.5132496513249651, 0.5575101488497971, 0.5883905013192612, 0.6205128205128205, 0.6533665835411471, 0.6812121212121214, 0.6961722488038278, 0.7058823529411764, 0.7177700348432055] 
	t6 = [0, 6.097728490829468, 12.073375940322876, 18.012506008148193, 24.14640474319458, 30.042481184005737, 36.05923295021057, 42.13750433921814, 48.020532846450806, 54.144569635391235, 60.092238903045654] 
	q6 = [0.3508771929824561, 0.40916030534351144, 0.46198830409356717, 0.49647390691114246, 0.5528455284552846, 0.5939553219448095, 0.6310432569974554, 0.6699386503067485, 0.6907340553549941, 0.705188679245283, 0.7192575406032483] 
	t7 = [0, 7.011035680770874, 14.03197717666626, 21.048449754714966, 28.14438772201538, 35.063064098358154, 42.003331422805786, 49.1047465801239, 56.0024254322052] 
	q7 = [0.3508771929824561, 0.4157814871016692, 0.4812680115273776, 0.5187239944521498, 0.5805592543275633, 0.6205128205128205, 0.6650246305418719, 0.6923076923076923, 0.7067137809187277] 
	t8 = [0, 8.026752948760986, 16.101381301879883, 24.095938205718994, 32.147852659225464, 40.11970901489258, 48.10551571846008, 56.11998152732849] 
	q8 = [0.3508771929824561, 0.42296072507552873, 0.49358059914407987, 0.55359565807327, 0.6059817945383615, 0.654228855721393, 0.6907340553549941, 0.7065727699530516] 
	t9 = [0, 9.108036756515503, 18.03778386116028, 27.078929662704468, 36.01093602180481, 45.088767766952515, 54.012919187545776] 
	q9 = [0.3508771929824561, 0.43413173652694614, 0.4978783592644979, 0.5748663101604277, 0.6310432569974554, 0.6828087167070218, 0.705188679245283] 
	t10 = [0, 10.017174243927002, 20.092089891433716, 30.06848978996277, 40.14373207092285, 50.0165798664093, 60.08505868911743] 
	q10 = [0.3508771929824561, 0.4398216939078752, 0.5146853146853148, 0.5947368421052632, 0.6559006211180124, 0.6953405017921148, 0.7186046511627907] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.1131680011749268, 2.081968307495117, 3.044325351715088, 4.130446195602417, 5.091939449310303, 6.026788234710693, 7.13524055480957, 8.127904653549194, 9.003080606460571, 10.115276575088501, 11.071208715438843, 12.014889001846313, 13.000268459320068, 14.113631248474121, 15.12148666381836, 16.088942766189575, 17.039878129959106, 18.00766396522522, 19.002767086029053, 20.079989433288574, 21.062535524368286, 22.137298583984375, 23.095422983169556, 24.044930934906006, 25.141643524169922, 26.07441234588623, 27.11812448501587, 28.017154216766357, 29.003698110580444, 30.117691040039062, 31.134000301361084, 32.07374691963196, 33.05577206611633, 34.13040375709534, 35.0797438621521, 36.038947105407715, 37.13652849197388, 38.10695719718933, 39.05296540260315, 40.13206696510315, 41.1234450340271, 42.052645206451416, 43.031299352645874, 44.127753496170044, 45.101879596710205, 46.05184459686279, 47.13158130645752, 48.055177211761475, 49.04847264289856, 50.114792346954346, 51.068862438201904, 52.025211811065674, 53.131267786026, 54.100544929504395, 55.044121980667114, 56.01380395889282, 57.135034799575806, 58.07572650909424, 59.027575731277466, 60.10999321937561] 
	q1 = [0.3443708609271523, 0.35255354200988465, 0.3562091503267974, 0.366288492706645, 0.37620578778135044, 0.3859649122807018, 0.39873417721518983, 0.4025157232704403, 0.4043887147335423, 0.41368584758942456, 0.4283513097072419, 0.437308868501529, 0.4407294832826748, 0.44478063540090773, 0.44879518072289165, 0.45739910313901344, 0.4629080118694362, 0.4690265486725664, 0.47368421052631576, 0.4760522496371553, 0.47976878612716767, 0.4878048780487805, 0.4978662873399715, 0.5014164305949008, 0.5112359550561798, 0.5257301808066759, 0.5318559556786704, 0.543956043956044, 0.5499316005471956, 0.5597826086956522, 0.5675675675675675, 0.576043068640646, 0.5836680053547523, 0.5922974767596282, 0.5978835978835979, 0.605263157894737, 0.6117647058823529, 0.6173800259403373, 0.6245161290322581, 0.6290115532734275, 0.6360153256704981, 0.6404066073697585, 0.6463878326996199, 0.649056603773585, 0.6550000000000001, 0.6592039800995025, 0.6600496277915632, 0.665024630541872, 0.6731946144430844, 0.6780487804878048, 0.6820388349514563, 0.6884057971014492, 0.6899879372738239, 0.6915662650602409, 0.6931407942238267, 0.6954436450839329, 0.6970059880239521, 0.6977299880525686, 0.6992840095465394, 0.703923900118906, 0.7069988137603795] 
	t2 = [0, 2.1363673210144043, 4.006890535354614, 6.141085386276245, 8.149322509765625, 10.098950147628784, 12.101900100708008, 14.135984897613525, 16.038990259170532, 18.090503454208374, 20.09657573699951, 22.091712474822998, 24.07362151145935, 26.064329147338867, 28.12151312828064, 30.03212833404541, 32.06897044181824, 34.06677293777466, 36.07188630104065, 38.0839958190918, 40.04870057106018, 42.046655893325806, 44.07250714302063, 46.13230633735657, 48.0008659362793, 50.1449658870697, 52.145432472229004, 54.01702284812927, 56.01152276992798, 58.019492626190186, 60.05024528503418] 
	q2 = [0.3443708609271523, 0.3588907014681892, 0.37620578778135044, 0.40126382306477093, 0.40937500000000004, 0.43076923076923074, 0.44309559939301973, 0.45045045045045046, 0.46814814814814815, 0.47743813682678304, 0.48345323741007196, 0.49929078014184397, 0.5216178521617851, 0.5399449035812672, 0.5617367706919946, 0.576043068640646, 0.5922974767596282, 0.605263157894737, 0.6191709844559585, 0.6307692307692309, 0.6429479034307497, 0.650753768844221, 0.6592039800995025, 0.6715686274509803, 0.6804374240583232, 0.6899879372738239, 0.6931407942238267, 0.6946107784431138, 0.7008343265792609, 0.7069988137603795, 0.7068557919621749] 
	t3 = [0, 3.0232043266296387, 6.079526662826538, 9.060011863708496, 12.090225458145142, 15.032531261444092, 18.136489391326904, 21.078742265701294, 24.100038766860962, 27.151359796524048, 30.004623889923096, 33.097328901290894, 36.02083158493042, 39.092851877212524, 42.102081298828125, 45.047688007354736, 48.00118350982666, 51.065979957580566, 54.1130211353302, 57.04249906539917, 60.11865854263306] 
	q3 = [0.3443708609271523, 0.366288492706645, 0.40126382306477093, 0.42170542635658914, 0.44309559939301973, 0.4649776453055142, 0.47674418604651164, 0.4978662873399715, 0.5257301808066759, 0.5491803278688524, 0.5771812080536913, 0.5997357992073976, 0.6191709844559585, 0.638676844783715, 0.655819774718398, 0.6633785450061652, 0.6820388349514563, 0.6915662650602409, 0.6977299880525686, 0.7054631828978624, 0.705188679245283] 
	t4 = [0, 4.036273956298828, 8.058112859725952, 12.015103816986084, 16.01531410217285, 20.04654812812805, 24.03914523124695, 28.112401008605957, 32.07137084007263, 36.04985237121582, 40.00992250442505, 44.11551809310913, 48.066184997558594, 52.13998293876648, 56.002018451690674, 60.056047439575195] 
	q4 = [0.3443708609271523, 0.3756019261637239, 0.40937500000000004, 0.44309559939301973, 0.46814814814814815, 0.48563218390804597, 0.5236768802228412, 0.5578231292517006, 0.5941644562334217, 0.6199740596627757, 0.6446700507614213, 0.6600496277915632, 0.6820388349514563, 0.6931407942238267, 0.7023809523809524, 0.706021251475797] 
	t5 = [0, 5.088858604431152, 10.053570985794067, 15.011133432388306, 20.100388050079346, 25.119269609451294, 30.102707862854004, 35.11375546455383, 40.113196849823, 45.11367583274841, 50.079392194747925, 55.107120752334595, 60.12010645866394] 
	q5 = [0.3443708609271523, 0.3885350318471338, 0.4362519201228878, 0.4649776453055142, 0.49067431850789095, 0.5331491712707183, 0.5817694369973191, 0.6163849154746424, 0.6481012658227848, 0.6683046683046683, 0.6915662650602409, 0.6992840095465394, 0.706021251475797] 
	t6 = [0, 6.135974884033203, 12.000401496887207, 18.02518582344055, 24.00853204727173, 30.01848840713501, 36.13330626487732, 42.028191566467285, 48.109615325927734, 54.11171317100525, 60.05997371673584] 
	q6 = [0.3443708609271523, 0.40126382306477093, 0.4461305007587253, 0.478134110787172, 0.5277777777777778, 0.5779569892473119, 0.6253229974160207, 0.6541822721598002, 0.6852300242130751, 0.6977299880525686, 0.706021251475797] 
	t7 = [0, 7.058560371398926, 14.0061616897583, 21.110791206359863, 28.005306243896484, 35.06766867637634, 42.05753827095032, 49.04044532775879, 56.047449350357056] 
	q7 = [0.3443708609271523, 0.39937106918238996, 0.45645645645645644, 0.5014245014245015, 0.5597826086956522, 0.6163849154746424, 0.6541822721598002, 0.6884057971014492, 0.703923900118906] 
	t8 = [0, 8.058050394058228, 16.11241364479065, 24.067901134490967, 32.027310371398926, 40.01008343696594, 48.05183219909668, 56.0083429813385] 
	q8 = [0.3443708609271523, 0.40937500000000004, 0.47337278106508873, 0.5277777777777778, 0.5960264900662251, 0.6481012658227848, 0.6852300242130751, 0.7054631828978624] 
	t9 = [0, 9.117651224136353, 18.067994832992554, 27.07478356361389, 36.09255409240723, 45.14388942718506, 54.00503158569336] 
	q9 = [0.3443708609271523, 0.4241486068111455, 0.47883211678832116, 0.5538881309686221, 0.6253229974160207, 0.6699386503067486, 0.6977299880525686] 
	t10 = [0, 10.142205238342285, 20.09775710105896, 30.018369436264038, 40.11847424507141, 50.05768394470215, 60.14109683036804] 
	q10 = [0.3443708609271523, 0.4355828220858896, 0.49712643678160917, 0.5779569892473119, 0.6481012658227848, 0.6899879372738239, 0.7075471698113207] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.1010448932647705, 2.047410726547241, 3.0023083686828613, 4.1383562088012695, 5.098572731018066, 6.029729127883911, 7.132946252822876, 8.077301263809204, 9.026063203811646, 10.1049222946167, 11.005573511123657, 12.104968309402466, 13.091115713119507, 14.015803575515747, 15.003775119781494, 16.087137937545776, 17.042934894561768, 18.043201446533203, 19.056795120239258, 20.13051676750183, 21.004130125045776, 22.08757734298706, 23.065107345581055, 24.043614864349365, 25.145179271697998, 26.114416360855103, 27.081247091293335, 28.084733486175537, 29.094794988632202, 30.022883892059326, 31.127846240997314, 32.09704923629761, 33.07796096801758, 34.04410099983215, 35.00170278549194, 36.10288596153259, 37.086241722106934, 38.01927089691162, 39.11022639274597, 40.03222894668579, 41.121365785598755, 42.08177375793457, 43.055617809295654, 44.13862061500549, 45.03065037727356, 46.1030592918396, 47.118611097335815, 48.076128244400024, 49.03431582450867, 50.1022686958313, 51.12954068183899, 52.09244990348816, 53.04214954376221, 54.12418866157532, 55.106329917907715, 56.06636071205139, 57.062668800354004, 58.0265474319458, 59.13364100456238, 60.06458926200867] 
	q1 = [0.37403400309119017, 0.38343558282208584, 0.3926940639269407, 0.3969696969696969, 0.4012066365007541, 0.4095665171898356, 0.42136498516320475, 0.4247787610619469, 0.4363103953147877, 0.44250363901018924, 0.4486251808972504, 0.4546762589928058, 0.4679029957203995, 0.4752475247524752, 0.47605633802816905, 0.4797768479776848, 0.4854368932038835, 0.49103448275862066, 0.49589041095890407, 0.5027173913043478, 0.5060893098782139, 0.5148247978436657, 0.5301204819277108, 0.536, 0.5411140583554377, 0.5526315789473684, 0.5602094240837696, 0.5677083333333334, 0.574385510996119, 0.5817245817245819, 0.5856777493606139, 0.5928753180661577, 0.6017699115044247, 0.6045340050377834, 0.6132665832290363, 0.616729088639201, 0.6195786864931846, 0.6273062730627306, 0.6323529411764706, 0.6399026763990268, 0.6440677966101694, 0.6450060168471721, 0.6531100478468901, 0.6587395957193818, 0.6650887573964497, 0.6737089201877934, 0.6814469078179696, 0.6875725900116144, 0.6929316338354579, 0.697459584295612, 0.7019562715765246, 0.706422018348624, 0.7123287671232877, 0.7137970353477765, 0.7181818181818183, 0.7188208616780044, 0.7186440677966102, 0.7192784667418263, 0.7207207207207207, 0.7248322147651006, 0.7262569832402234] 
	t2 = [0, 2.1288700103759766, 4.141336917877197, 6.133907079696655, 8.094048261642456, 10.072678804397583, 12.126893997192383, 14.010316848754883, 16.021150827407837, 18.029217958450317, 20.09649085998535, 22.140910625457764, 24.02909803390503, 26.03221893310547, 28.040793657302856, 30.112870454788208, 32.12216877937317, 34.01262021064758, 36.01882362365723, 38.019166231155396, 40.13928437232971, 42.00225639343262, 44.1344838142395, 46.05499768257141, 48.09057354927063, 50.06366324424744, 52.13050198554993, 54.12344932556152, 56.13980150222778, 58.033385276794434, 60.000871896743774] 
	q2 = [0.37403400309119017, 0.3896499238964992, 0.40361445783132527, 0.4237037037037037, 0.44087591240875906, 0.4508670520231214, 0.4694167852062589, 0.47685834502103785, 0.4868603042876903, 0.4986376021798365, 0.5148247978436657, 0.5340453938584779, 0.5488126649076517, 0.5677083333333334, 0.5798969072164948, 0.5928753180661577, 0.6062893081761006, 0.616729088639201, 0.6273062730627306, 0.6391251518833536, 0.6482593037214885, 0.6619217081850534, 0.6791569086651054, 0.6890951276102089, 0.7019562715765246, 0.7079037800687286, 0.7152619589977219, 0.7186440677966102, 0.7207207207207207, 0.7248322147651006, 0.7296996662958843] 
	t3 = [0, 3.0224881172180176, 6.06633186340332, 9.049225807189941, 12.112856149673462, 15.042391538619995, 18.089457511901855, 21.096078872680664, 24.037396907806396, 27.111366271972656, 30.104167222976685, 33.01316046714783, 36.09318399429321, 39.033714056015015, 42.0438232421875, 45.1417977809906, 48.08783459663391, 51.021470069885254, 54.0593695640564, 57.103535652160645, 60.02041506767273] 
	q3 = [0.37403400309119017, 0.3969696969696969, 0.4237037037037037, 0.4447674418604651, 0.47159090909090906, 0.4861111111111111, 0.5013623978201635, 0.5301204819277108, 0.5526315789473684, 0.5751295336787564, 0.5972045743329097, 0.6132665832290363, 0.6289926289926291, 0.644927536231884, 0.6635071090047394, 0.6883720930232559, 0.7019562715765246, 0.7137970353477765, 0.7186440677966102, 0.7248322147651006, 0.7339246119733924] 
	t4 = [0, 4.0394158363342285, 8.128147840499878, 12.040130138397217, 16.06835103034973, 20.098246812820435, 24.07271671295166, 28.08804225921631, 32.1143593788147, 36.11296367645264, 40.098270654678345, 44.02955341339111, 48.07508993148804, 52.08301377296448, 56.056835889816284, 60.01524782180786] 
	q4 = [0.37403400309119017, 0.4006024096385542, 0.44087591240875906, 0.47226173541963024, 0.49239280774550476, 0.5148247978436657, 0.5526315789473684, 0.5817245817245819, 0.6097867001254705, 0.6297662976629765, 0.651497005988024, 0.6807017543859649, 0.7019562715765246, 0.7167235494880546, 0.7227833894500562, 0.7339246119733924] 
	t5 = [0, 5.110252141952515, 10.086124897003174, 15.113806247711182, 20.134462594985962, 25.01404571533203, 30.023170232772827, 35.10960602760315, 40.034074544906616, 45.03104639053345, 50.12840437889099, 55.041611671447754, 60.130385398864746] 
	q5 = [0.37403400309119017, 0.4119402985074627, 0.4508670520231214, 0.4867872044506259, 0.5182186234817813, 0.5620915032679739, 0.5946632782719187, 0.626387176325524, 0.651497005988024, 0.6883720930232559, 0.710857142857143, 0.7207207207207207, 0.7367256637168144] 
	t6 = [0, 6.111771821975708, 12.026349544525146, 18.127495765686035, 24.08974528312683, 30.071245193481445, 36.05458188056946, 42.12602686882019, 48.09903073310852, 54.077861070632935, 60.01031279563904] 
	q6 = [0.37403400309119017, 0.4237037037037037, 0.47226173541963024, 0.5020463847203275, 0.5545335085413929, 0.5964467005076143, 0.6331288343558283, 0.6666666666666667, 0.7049368541905855, 0.7178329571106095, 0.7367256637168144] 
	t7 = [0, 7.028509616851807, 14.151544332504272, 21.00577735900879, 28.085854291915894, 35.136972188949585, 42.14608550071716, 49.01156210899353, 56.06867432594299] 
	q7 = [0.37403400309119017, 0.4270986745213549, 0.48179271708683474, 0.5288590604026846, 0.5817245817245819, 0.626387176325524, 0.6650887573964497, 0.7079037800687286, 0.7242152466367713] 
	t8 = [0, 8.018670082092285, 16.006091117858887, 24.163590669631958, 32.08022952079773, 40.0339412689209, 48.11566972732544, 56.126102924346924] 
	q8 = [0.37403400309119017, 0.44087591240875906, 0.4930747922437673, 0.5526315789473684, 0.6115288220551378, 0.6538922155688623, 0.7042577675489068, 0.7256438969764837] 
	t9 = [0, 9.12815237045288, 18.081674337387085, 27.09061598777771, 36.039167404174805, 45.096312284469604, 54.12681317329407] 
	q9 = [0.37403400309119017, 0.444121915820029, 0.5027322404371585, 0.5788113695090439, 0.6331288343558283, 0.6883720930232559, 0.7178329571106095] 
	t10 = [0, 10.031468152999878, 20.064070224761963, 30.025954008102417, 40.05621647834778, 50.05963063240051, 60.08484721183777] 
	q10 = [0.37403400309119017, 0.4486251808972504, 0.516914749661705, 0.5946632782719187, 0.6538922155688623, 0.7101947308132875, 0.7367256637168144] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.101938247680664, 2.0239853858947754, 3.1155686378479004, 4.040267705917358, 5.138702869415283, 6.0744948387146, 7.0322980880737305, 8.134954690933228, 9.140938758850098, 10.06505274772644, 11.020910739898682, 12.098358869552612, 13.05238676071167, 14.123389959335327, 15.064783096313477, 16.031972885131836, 17.135899543762207, 18.123639822006226, 19.132036924362183, 20.061027765274048, 21.040717124938965, 22.028293132781982, 23.130792140960693, 24.093733549118042, 25.102888584136963, 26.12629461288452, 27.089402437210083, 28.011473894119263, 29.127923011779785, 30.047308683395386, 31.12977385520935, 32.057119846343994, 33.00165891647339, 34.069151878356934, 35.00842595100403, 36.10526251792908, 37.06686544418335, 38.001787185668945, 39.11982798576355, 40.042449712753296, 41.08730721473694, 42.00753092765808, 43.101284980773926, 44.06340503692627, 45.01974081993103, 46.0118510723114, 47.12735104560852, 48.03529095649719, 49.00753903388977, 50.08384299278259, 51.08598184585571, 52.0047881603241, 53.11735701560974, 54.07131505012512, 55.070379972457886, 56.02020215988159, 57.00270438194275, 58.10942840576172, 59.086233377456665, 60.00375056266785] 
	q1 = [0.37519872813990457, 0.3848580441640379, 0.39184952978056425, 0.40372670807453415, 0.41294298921417566, 0.42266462480857586, 0.4244274809160305, 0.4370257966616085, 0.4457831325301205, 0.45603576751117736, 0.46449704142011833, 0.46989720998531564, 0.4832605531295488, 0.489855072463768, 0.4956772334293948, 0.49785407725321884, 0.5014245014245015, 0.5070821529745042, 0.5126760563380282, 0.5230769230769231, 0.5271966527196652, 0.5325936199722608, 0.5379310344827587, 0.5479452054794521, 0.5538881309686221, 0.5578231292517007, 0.5660377358490567, 0.5710455764075066, 0.5748663101604279, 0.5827814569536424, 0.5902503293807642, 0.599476439790576, 0.6067708333333333, 0.610608020698577, 0.617948717948718, 0.6232439335887611, 0.6294416243654822, 0.6330390920554856, 0.6356783919597989, 0.6384039900249378, 0.6434782608695652, 0.6535141800246608, 0.660122699386503, 0.6682926829268293, 0.6699147381242387, 0.6763636363636363, 0.6843373493975904, 0.6875, 0.6906474820143885, 0.6945107398568019, 0.6960667461263408, 0.6976190476190476, 0.6998813760379597, 0.7028301886792452, 0.7043580683156654, 0.7080890973036342, 0.707943925233645, 0.7124563445867288, 0.7146171693735498, 0.7190751445086706, 0.7174163783160323] 
	t2 = [0, 2.1381983757019043, 4.123997926712036, 6.10398530960083, 8.07624340057373, 10.142881870269775, 12.12722659111023, 14.090037822723389, 16.06607723236084, 18.09518313407898, 20.032270908355713, 22.089327573776245, 24.133957862854004, 26.079107522964478, 28.086363554000854, 30.10582423210144, 32.109551191329956, 34.09653663635254, 36.076819896698, 38.09244966506958, 40.0985472202301, 42.02136993408203, 44.028021574020386, 46.04331684112549, 48.12906885147095, 50.01741814613342, 52.022167921066284, 54.047659397125244, 56.110639810562134, 58.13707423210144, 60.12678003311157] 
	q2 = [0.37519872813990457, 0.3912363067292645, 0.4153846153846154, 0.42682926829268286, 0.4481203007518797, 0.4660766961651917, 0.48546511627906974, 0.49712643678160917, 0.5028409090909092, 0.5189340813464236, 0.5292479108635098, 0.544704264099037, 0.5578231292517007, 0.5698924731182795, 0.5797872340425531, 0.5976408912188729, 0.6088082901554404, 0.6232439335887611, 0.6313131313131314, 0.6392009987515606, 0.6518518518518518, 0.6666666666666666, 0.6763636363636363, 0.6875, 0.6929510155316607, 0.6991676575505351, 0.7028301886792452, 0.7096018735362999, 0.713953488372093, 0.7182448036951502, 0.7262313860252004] 
	t3 = [0, 3.035891532897949, 6.025666952133179, 9.028151750564575, 12.002849102020264, 15.022248029708862, 18.059723138809204, 21.03236675262451, 24.00170373916626, 27.070401430130005, 30.10902428627014, 33.1250102519989, 36.000237464904785, 39.06951403617859, 42.07342076301575, 45.12553906440735, 48.13673663139343, 51.160090923309326, 54.06191349029541, 57.04451584815979, 60.02617907524109] 
	q3 = [0.37519872813990457, 0.40372670807453415, 0.42682926829268286, 0.4567164179104477, 0.48546511627906974, 0.5021398002853067, 0.5210084033613445, 0.5366528354080221, 0.5578231292517007, 0.5748663101604279, 0.599476439790576, 0.619718309859155, 0.6347607052896725, 0.6468401486988848, 0.6682926829268293, 0.6843373493975904, 0.6913875598086124, 0.7012987012987013, 0.7087719298245615, 0.7175925925925926, 0.7276887871853547] 
	t4 = [0, 4.02997088432312, 8.130019187927246, 12.135754108428955, 16.090012788772583, 20.04225182533264, 24.10250163078308, 28.055654764175415, 32.01403284072876, 36.066054821014404, 40.13443326950073, 44.0116331577301, 48.14356088638306, 52.09885048866272, 56.071598291397095, 60.109200954437256] 
	q4 = [0.37519872813990457, 0.41294298921417566, 0.4481203007518797, 0.4876632801161103, 0.5028409090909092, 0.5292479108635098, 0.5578231292517007, 0.5816733067729084, 0.610608020698577, 0.6347607052896725, 0.6568265682656828, 0.6779661016949152, 0.6929510155316607, 0.7043580683156654, 0.7162790697674419, 0.7276887871853547] 
	t5 = [0, 5.088334321975708, 10.01634168624878, 15.059004068374634, 20.037113904953003, 25.05411696434021, 30.0395450592041, 35.01818323135376, 40.00049614906311, 45.07085204124451, 50.07010531425476, 55.023261070251465, 60.13237953186035] 
	q5 = [0.37519872813990457, 0.42266462480857586, 0.46971935007385524, 0.5014245014245015, 0.5333333333333333, 0.5668016194331984, 0.600262123197903, 0.6286438529784538, 0.6568265682656828, 0.6843373493975904, 0.697508896797153, 0.7147846332945286, 0.7291428571428571] 
	t6 = [0, 6.121213674545288, 12.091781616210938, 18.10853147506714, 24.05995750427246, 30.043653964996338, 36.010000705718994, 42.007421255111694, 48.067615270614624, 54.02027106285095, 60.03602313995361] 
	q6 = [0.37519872813990457, 0.42987804878048785, 0.4890829694323144, 0.5230769230769231, 0.5597826086956521, 0.600262123197903, 0.6339622641509434, 0.6682926829268293, 0.6913875598086124, 0.7102803738317757, 0.7291428571428571] 
	t7 = [0, 7.144281387329102, 14.08816933631897, 21.03427004814148, 28.13923144340515, 35.049208879470825, 42.07224249839783, 49.0912446975708, 56.0911340713501] 
	q7 = [0.37519872813990457, 0.43939393939393934, 0.4992826398852224, 0.5406896551724137, 0.5880794701986755, 0.6278481012658228, 0.6682926829268293, 0.6960667461263408, 0.7184241019698726] 
	t8 = [0, 8.03703761100769, 16.127163648605347, 24.103349208831787, 32.04409837722778, 40.10138177871704, 48.00355553627014, 56.05518698692322] 
	q8 = [0.37519872813990457, 0.4481203007518797, 0.504964539007092, 0.5617367706919946, 0.6159793814432991, 0.6584766584766585, 0.6913875598086124, 0.7184241019698726] 
	t9 = [0, 9.086281776428223, 18.097144603729248, 27.026808738708496, 36.00250959396362, 45.11552810668945, 54.02472639083862] 
	q9 = [0.37519872813990457, 0.45901639344262296, 0.5230769230769231, 0.5813333333333334, 0.6339622641509434, 0.6859205776173286, 0.7102803738317757] 
	t10 = [0, 10.111899375915527, 20.057449340820312, 30.04202437400818, 40.045703172683716, 50.00696301460266, 60.132972955703735] 
	q10 = [0.37519872813990457, 0.4667651403249631, 0.5333333333333333, 0.6028833551769333, 0.6584766584766585, 0.697508896797153, 0.7308132875143184] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.0930488109588623, 2.0570836067199707, 3.0131263732910156, 4.088042736053467, 5.046553134918213, 6.130492687225342, 7.084993362426758, 8.01526141166687, 9.107081651687622, 10.06040072441101, 11.133731126785278, 12.060111045837402, 13.013535499572754, 14.086800336837769, 15.062686443328857, 16.140974521636963, 17.01014471054077, 18.095457315444946, 19.06942629814148, 20.02385926246643, 21.032758235931396, 22.13722825050354, 23.082924365997314, 24.020068645477295, 25.006637573242188, 26.124789476394653, 27.078633308410645, 28.00889801979065, 29.13191056251526, 30.087984323501587, 31.039066553115845, 32.14558148384094, 33.12651801109314, 34.11893677711487, 35.09168076515198, 36.030848026275635, 37.12612056732178, 38.07809281349182, 39.01931285858154, 40.09470200538635, 41.03460907936096, 42.12979698181152, 43.1320538520813, 44.05354952812195, 45.094924449920654, 46.025657415390015, 47.03536105155945, 48.10782718658447, 49.11588954925537, 50.07689428329468, 51.05577874183655, 52.01220417022705, 53.14121341705322, 54.092413663864136, 55.06313467025757, 56.140331745147705, 57.08850145339966, 58.01693058013916, 59.143205404281616, 60.10206985473633] 
	q1 = [0.35499207606973054, 0.35962145110410093, 0.3667711598746081, 0.37325038880248834, 0.3858024691358025, 0.3950995405819296, 0.4018264840182648, 0.40785498489425975, 0.41141141141141147, 0.4149253731343284, 0.4172876304023844, 0.42540620384047273, 0.4294117647058824, 0.43795620437956206, 0.44927536231884063, 0.45533141210374634, 0.4655172413793104, 0.4714285714285715, 0.4759206798866855, 0.4795486600846262, 0.4859550561797753, 0.496513249651325, 0.5041551246537396, 0.5116918844566712, 0.5191256830601092, 0.5271739130434783, 0.5383580080753702, 0.5508021390374331, 0.5565912117177096, 0.5608465608465609, 0.5684210526315789, 0.5777777777777778, 0.5859375, 0.5966277561608301, 0.6030927835051546, 0.6069142125480154, 0.6175349428208387, 0.624525916561315, 0.6306532663316583, 0.6408977556109725, 0.6476426799007443, 0.654320987654321, 0.6609336609336609, 0.665036674816626, 0.6674786845310596, 0.6787439613526571, 0.6835138387484957, 0.6866746698679472, 0.6929510155316606, 0.6967895362663495, 0.6998813760379596, 0.7014218009478673, 0.7021276595744681, 0.7051886792452831, 0.7067137809187279, 0.7104337631887455, 0.7101280558789289, 0.710801393728223, 0.7106481481481481, 0.7119815668202765, 0.7164179104477612] 
	t2 = [0, 2.127568483352661, 4.138633728027344, 6.119072914123535, 8.095943927764893, 10.069459915161133, 12.047832250595093, 14.011573553085327, 16.012410879135132, 18.040239572525024, 20.03983974456787, 22.109993934631348, 24.10592746734619, 26.112743854522705, 28.129066467285156, 30.142618417739868, 32.042094707489014, 34.083914041519165, 36.11548328399658, 38.104763984680176, 40.09643292427063, 42.10772228240967, 44.02969002723694, 46.055389165878296, 48.121466875076294, 50.01715803146362, 52.04415035247803, 54.07514762878418, 56.069010734558105, 58.07556939125061, 60.120662450790405] 
	q2 = [0.35499207606973054, 0.36619718309859156, 0.3858024691358025, 0.4012158054711247, 0.41379310344827586, 0.41901931649331353, 0.43401759530791795, 0.45151953690303903, 0.46704871060171926, 0.47807637906647804, 0.49230769230769234, 0.5075862068965517, 0.5271739130434783, 0.5469168900804289, 0.5608465608465609, 0.5777777777777778, 0.5966277561608301, 0.6076923076923078, 0.624525916561315, 0.6408977556109725, 0.655980271270037, 0.665036674816626, 0.6787439613526571, 0.689075630252101, 0.7007125890736342, 0.6990521327014219, 0.7067137809187279, 0.7087719298245614, 0.7099767981438516, 0.7149425287356321, 0.7170675830469644] 
	t3 = [0, 3.015209436416626, 6.0559046268463135, 9.087048053741455, 12.043460369110107, 15.115318059921265, 18.029727458953857, 21.110271453857422, 24.064196586608887, 27.027228116989136, 30.068256616592407, 33.123130083084106, 36.1192843914032, 39.026123046875, 42.141592264175415, 45.04181456565857, 48.12510323524475, 51.037301778793335, 54.02263617515564, 57.08010792732239, 60.0001175403595] 
	q3 = [0.35499207606973054, 0.37577639751552794, 0.4012158054711247, 0.417910447761194, 0.43401759530791795, 0.4626436781609195, 0.47807637906647804, 0.5020804438280166, 0.5271739130434783, 0.5565912117177096, 0.5796344647519581, 0.6056701030927835, 0.6262626262626262, 0.650990099009901, 0.6642246642246641, 0.6866746698679472, 0.7007125890736342, 0.7036599763872492, 0.7086247086247086, 0.7128027681660899, 0.7185354691075515] 
	t4 = [0, 4.049229860305786, 8.13135051727295, 12.010992765426636, 16.065282583236694, 20.04565191268921, 24.064995050430298, 28.092522859573364, 32.05543303489685, 36.07101607322693, 40.04587125778198, 44.047059774398804, 48.119465827941895, 52.13628792762756, 56.00375175476074, 60.10439968109131] 
	q4 = [0.35499207606973054, 0.3858024691358025, 0.4161676646706587, 0.43401759530791795, 0.46924177396280403, 0.4972067039106145, 0.5291723202170964, 0.5627476882430646, 0.5958549222797928, 0.6262626262626262, 0.6576354679802955, 0.6803377563329313, 0.7007125890736342, 0.708235294117647, 0.7106481481481481, 0.7177142857142857] 
	t5 = [0, 5.087143421173096, 10.061480522155762, 15.026299238204956, 20.035414218902588, 25.00526714324951, 30.048299074172974, 35.099504709243774, 40.00250554084778, 45.076226472854614, 50.119075536727905, 55.1063506603241, 60.00830340385437] 
	q5 = [0.35499207606973054, 0.39755351681957185, 0.4213649851632047, 0.4626436781609195, 0.4972067039106145, 0.536388140161725, 0.5814863102998696, 0.6226175349428209, 0.6576354679802955, 0.6850961538461539, 0.7014218009478673, 0.7099767981438516, 0.7177142857142857] 
	t6 = [0, 6.0126166343688965, 12.063181400299072, 18.061530590057373, 24.08365273475647, 30.081021547317505, 36.02536058425903, 42.03826594352722, 48.02005362510681, 54.03974175453186, 60.13544178009033] 
	q6 = [0.35499207606973054, 0.3987823439878234, 0.43401759530791795, 0.480225988700565, 0.5291723202170964, 0.5814863102998696, 0.6270543615676358, 0.6642246642246641, 0.7007125890736342, 0.7101280558789289, 0.7185354691075515] 
	t7 = [0, 7.022958517074585, 14.084367513656616, 21.143394947052002, 28.009997129440308, 35.01716494560242, 42.00258803367615, 49.02683997154236, 56.03936576843262] 
	q7 = [0.35499207606973054, 0.40483383685800606, 0.45375722543352603, 0.5069252077562326, 0.5608465608465609, 0.6243654822335025, 0.6674786845310596, 0.6998813760379596, 0.7136258660508082] 
	t8 = [0, 8.050017595291138, 16.123989582061768, 24.11247158050537, 32.14229464530945, 40.12683916091919, 48.13609290122986, 56.070091009140015] 
	q8 = [0.35499207606973054, 0.41379310344827586, 0.4714285714285715, 0.5311653116531165, 0.6020671834625323, 0.6609336609336609, 0.6991676575505351, 0.7136258660508082] 
	t9 = [0, 9.11162519454956, 18.017802715301514, 27.02516007423401, 36.14172291755676, 45.00130248069763, 54.02278542518616] 
	q9 = [0.35499207606973054, 0.41430700447093893, 0.480225988700565, 0.5577689243027888, 0.6322418136020151, 0.6874999999999999, 0.7094515752625438] 
	t10 = [0, 10.139702320098877, 20.117785215377808, 30.00936770439148, 40.144179344177246, 50.023319721221924, 60.071545362472534] 
	q10 = [0.35499207606973054, 0.4213649851632047, 0.5013927576601671, 0.5814863102998696, 0.6642066420664207, 0.7014218009478673, 0.7214611872146119] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.105396032333374, 2.0341031551361084, 3.1361451148986816, 4.058877229690552, 5.018264055252075, 6.09617805480957, 7.039422988891602, 8.111954689025879, 9.120322704315186, 10.07436752319336, 11.018353939056396, 12.087844610214233, 13.062521934509277, 14.04723596572876, 15.14335012435913, 16.084330558776855, 17.059581756591797, 18.01764678955078, 19.001455545425415, 20.076237440109253, 21.06319546699524, 22.022682905197144, 23.118497133255005, 24.10544490814209, 25.082632780075073, 26.012868881225586, 27.04499125480652, 28.120994329452515, 29.070351362228394, 30.1441433429718, 31.13248038291931, 32.05651235580444, 33.00190234184265, 34.080766439437866, 35.038079500198364, 36.10858345031738, 37.08911490440369, 38.0153431892395, 39.11099600791931, 40.06217908859253, 41.004875898361206, 42.08633089065552, 43.131447315216064, 44.05978345870972, 45.08751368522644, 46.07198357582092, 47.05776405334473, 48.012197494506836, 49.10037660598755, 50.04939675331116, 51.13539218902588, 52.11511015892029, 53.05706763267517, 54.03721809387207, 55.13610029220581, 56.06233072280884, 57.04061460494995, 58.0091598033905, 59.1087749004364, 60.09017276763916] 
	q1 = [0.37288135593220334, 0.382262996941896, 0.39150227617602423, 0.40060240963855415, 0.41017964071856283, 0.42136498516320475, 0.4323529411764706, 0.43923865300146414, 0.4476744186046512, 0.45151953690303914, 0.46197991391678617, 0.46723646723646717, 0.47175141242937846, 0.47471910112359544, 0.4818941504178273, 0.4903047091412742, 0.49518569463548834, 0.4972677595628415, 0.5034013605442177, 0.5135135135135135, 0.5281501340482573, 0.5326231691078561, 0.5396825396825397, 0.5447368421052631, 0.5542483660130719, 0.5610389610389611, 0.5647668393782384, 0.5740025740025739, 0.5823754789272031, 0.5895806861499364, 0.5984848484848485, 0.6037735849056605, 0.6142322097378278, 0.6220570012391574, 0.6280788177339902, 0.6381418092909535, 0.6447688564476886, 0.6481257557436517, 0.6530120481927711, 0.6586826347305389, 0.6682520808561236, 0.6721893491124261, 0.6737338044758541, 0.6830409356725147, 0.6876456876456877, 0.6929316338354576, 0.697459584295612, 0.7049368541905855, 0.7056128293241696, 0.7085714285714286, 0.7121729237770194, 0.7136363636363637, 0.7180067950169876, 0.7209039548022599, 0.7192784667418264, 0.7191011235955056, 0.7174887892376681, 0.7181208053691275, 0.7193763919821825, 0.7222222222222223, 0.7228381374722838] 
	t2 = [0, 2.115309953689575, 4.091801643371582, 6.064861536026001, 8.044912338256836, 10.126590490341187, 12.097769737243652, 14.13512372970581, 16.145100831985474, 18.030102729797363, 20.03739356994629, 22.063610076904297, 24.106041431427002, 26.00474214553833, 28.07466173171997, 30.062309503555298, 32.07235383987427, 34.03933596611023, 36.0242383480072, 38.03650403022766, 40.042585611343384, 42.0089635848999, 44.05466938018799, 46.10690689086914, 48.1442813873291, 50.00834131240845, 52.04083442687988, 54.07758688926697, 56.05551719665527, 58.08813977241516, 60.131431579589844] 
	q2 = [0.37288135593220334, 0.393939393939394, 0.41255605381165916, 0.4346549192364171, 0.4454148471615721, 0.4641833810888252, 0.47175141242937846, 0.48821081830790575, 0.4972527472527472, 0.5074626865671642, 0.5294117647058822, 0.5408970976253298, 0.5580182529335072, 0.5729032258064517, 0.5877862595419847, 0.6037735849056605, 0.6203473945409429, 0.6364749082007344, 0.6464891041162227, 0.6586826347305389, 0.6721893491124261, 0.6814988290398126, 0.6944444444444445, 0.703448275862069, 0.7115165336374003, 0.7150964812712827, 0.7192784667418264, 0.718294051627385, 0.7209821428571429, 0.7236403995560488, 0.7262693156732892] 
	t3 = [0, 3.0202276706695557, 6.008338451385498, 9.008557558059692, 12.096707344055176, 15.034079313278198, 18.105550050735474, 21.026362895965576, 24.077561140060425, 27.056297063827515, 30.004307746887207, 33.03225064277649, 36.06149888038635, 39.10972571372986, 42.12131476402283, 45.12062096595764, 48.09198236465454, 51.11790490150452, 54.08126711845398, 57.097399950027466, 60.0522518157959] 
	q3 = [0.37288135593220334, 0.40060240963855415, 0.4346549192364171, 0.45598845598845594, 0.47390691114245415, 0.49171270718232046, 0.5094850948509485, 0.5391766268260292, 0.5617685305591678, 0.5823754789272031, 0.6037735849056605, 0.6297662976629766, 0.6497584541062802, 0.6690391459074734, 0.6861143523920653, 0.7019562715765246, 0.7107061503416855, 0.7209039548022599, 0.7174887892376681, 0.7236403995560488, 0.7262693156732892] 
	t4 = [0, 4.013110160827637, 8.113893032073975, 12.1431725025177, 16.10986566543579, 20.123037815093994, 24.139151573181152, 28.002012729644775, 32.125502824783325, 36.05805730819702, 40.140734910964966, 44.13845705986023, 48.03626489639282, 52.142367362976074, 56.00721979141235, 60.11630630493164] 
	q4 = [0.37288135593220334, 0.41255605381165916, 0.4476744186046512, 0.47390691114245415, 0.49931412894375854, 0.5313751668891855, 0.5617685305591678, 0.5895806861499364, 0.6254635352286775, 0.6497584541062802, 0.6729634002361276, 0.697459584295612, 0.7107061503416855, 0.7192784667418264, 0.7216035634743875, 0.72707182320442] 
	t5 = [0, 5.070100784301758, 10.126105070114136, 15.02666687965393, 20.065263271331787, 25.127948999404907, 30.091843366622925, 35.06996726989746, 40.02262544631958, 45.064833879470825, 50.046581983566284, 55.07582664489746, 60.13709855079651] 
	q5 = [0.37288135593220334, 0.42370370370370375, 0.46285714285714286, 0.494475138121547, 0.5313751668891855, 0.5692108667529107, 0.6072772898368883, 0.6480582524271844, 0.6721698113207547, 0.703448275862069, 0.7165532879818594, 0.7209821428571429, 0.72707182320442] 
	t6 = [0, 6.145990610122681, 12.077425003051758, 18.09232807159424, 24.001799821853638, 30.12868642807007, 36.05230355262756, 42.03847646713257, 48.04779005050659, 54.11248826980591, 60.099921226501465] 
	q6 = [0.37288135593220334, 0.4369501466275659, 0.47390691114245415, 0.5121951219512195, 0.5632333767926988, 0.6090225563909775, 0.6497584541062802, 0.6861143523920653, 0.7107061503416855, 0.7189249720044794, 0.728476821192053] 
	t7 = [0, 7.1425323486328125, 14.030640840530396, 21.06299877166748, 28.092475414276123, 35.08972787857056, 42.142934799194336, 49.03164577484131, 56.09701633453369] 
	q7 = [0.37288135593220334, 0.44152046783625737, 0.4909847434119278, 0.5398936170212767, 0.5939086294416244, 0.6480582524271844, 0.6861143523920653, 0.7121729237770194, 0.7230255839822024] 
	t8 = [0, 8.034449338912964, 16.01709270477295, 24.04648518562317, 32.1293740272522, 40.0816867351532, 48.09252405166626, 56.046470642089844] 
	q8 = [0.37288135593220334, 0.4476744186046512, 0.49931412894375854, 0.5632333767926988, 0.6288532675709002, 0.6737338044758541, 0.7107061503416855, 0.7230255839822024] 
	t9 = [0, 9.031269073486328, 18.059983015060425, 27.165496826171875, 36.04305052757263, 45.03026509284973, 54.0455276966095] 
	q9 = [0.37288135593220334, 0.45598845598845594, 0.5128900949796472, 0.5882352941176471, 0.6497584541062802, 0.7019562715765246, 0.7166853303471444] 
	t10 = [0, 10.154910326004028, 20.05607843399048, 30.098641872406006, 40.11123251914978, 50.07568669319153, 60.021727085113525] 
	q10 = [0.37288135593220334, 0.46285714285714286, 0.53475935828877, 0.6115288220551379, 0.6737338044758541, 0.7165532879818594, 0.729281767955801] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.1064467430114746, 2.035010814666748, 3.129431962966919, 4.062012195587158, 5.0223400592803955, 6.099046945571899, 7.079965353012085, 8.042138576507568, 9.135159492492676, 10.05894136428833, 11.03333568572998, 12.139721155166626, 13.001505374908447, 14.109533071517944, 15.084399938583374, 16.01625418663025, 17.105311632156372, 18.055317401885986, 19.001739263534546, 20.071877241134644, 21.017168283462524, 22.124931573867798, 23.078190803527832, 24.04305601119995, 25.052572011947632, 26.131969213485718, 27.028899431228638, 28.12657141685486, 29.101203680038452, 30.022965669631958, 31.110990285873413, 32.0654673576355, 33.018310546875, 34.12055802345276, 35.128334283828735, 36.048454999923706, 37.03532600402832, 38.106497287750244, 39.051616191864014, 40.12023687362671, 41.10403299331665, 42.056941509246826, 43.00245261192322, 44.12377405166626, 45.10116624832153, 46.02752494812012, 47.02899193763733, 48.09424185752869, 49.13314986228943, 50.084173917770386, 51.02277898788452, 52.115294456481934, 53.086445569992065, 54.04914355278015, 55.037580728530884, 56.01762771606445, 57.11313843727112, 58.07342267036438, 59.02252244949341, 60.09731864929199] 
	q1 = [0.35968992248062015, 0.3717357910906298, 0.37251908396946565, 0.3768996960486322, 0.38612368024132726, 0.3892215568862275, 0.3970149253731343, 0.4035608308605341, 0.4123711340206186, 0.4233576642335767, 0.43251088534107407, 0.43578643578643583, 0.44189383070301286, 0.4457142857142857, 0.4517045454545454, 0.45698166431593795, 0.4628330995792426, 0.4700973574408901, 0.47513812154696133, 0.48010973936899853, 0.49046321525885556, 0.4959349593495935, 0.5080645161290323, 0.5087014725568942, 0.5146666666666666, 0.5245033112582782, 0.5329815303430079, 0.5399737876802098, 0.5513654096228869, 0.5607235142118863, 0.5681233933161953, 0.5761843790012804, 0.5790816326530612, 0.5862944162436549, 0.592686002522068, 0.6, 0.6044776119402986, 0.6106304079110012, 0.6174661746617466, 0.6275946275946275, 0.629404617253949, 0.6360338573155985, 0.6410564225690276, 0.6459330143540669, 0.6539833531510107, 0.6579881656804734, 0.6643109540636043, 0.6705744431418522, 0.6752336448598131, 0.6782810685249708, 0.6828703703703703, 0.6874279123414072, 0.6949541284403669, 0.6979405034324944, 0.6986301369863014, 0.7030716723549487, 0.7030716723549487, 0.7104072398190044, 0.7102593010146562, 0.7115600448933783, 0.7122060470324747] 
	t2 = [0, 2.1139209270477295, 4.075960159301758, 6.049591302871704, 8.064018249511719, 10.100325107574463, 12.133961200714111, 14.054795503616333, 16.066738843917847, 18.041176080703735, 20.044790983200073, 22.004240036010742, 24.04075789451599, 26.06483817100525, 28.132484197616577, 30.02994728088379, 32.021918535232544, 34.132394313812256, 36.01193833351135, 38.023810148239136, 40.010658264160156, 42.04464364051819, 44.04640340805054, 46.07134938240051, 48.11340284347534, 50.035059213638306, 52.021910667419434, 54.07770848274231, 56.12495565414429, 58.136791706085205, 60.008063554763794] 
	q2 = [0.35968992248062015, 0.375, 0.3855421686746988, 0.3994038748137108, 0.4147058823529412, 0.43188405797101453, 0.44476327116212333, 0.45609065155807366, 0.4664804469273742, 0.47867950481430543, 0.49660786974219806, 0.5093833780160858, 0.5225464190981433, 0.5380577427821522, 0.5588615782664942, 0.5725288831835686, 0.5862944162436549, 0.6007509386733417, 0.6106304079110012, 0.6275946275946275, 0.6360338573155985, 0.6475507765830346, 0.6579881656804734, 0.6705744431418522, 0.6813441483198146, 0.6889400921658986, 0.6979405034324944, 0.7030716723549487, 0.708803611738149, 0.7130044843049328, 0.7157190635451505] 
	t3 = [0, 3.015021562576294, 6.009306907653809, 9.031051874160767, 12.103087663650513, 15.080584049224854, 18.133150815963745, 21.049447059631348, 24.06956195831299, 27.02647590637207, 30.011606454849243, 33.04646348953247, 36.003668785095215, 39.06670117378235, 42.11131429672241, 45.07789969444275, 48.14567422866821, 51.00379824638367, 54.069950580596924, 57.0634868144989, 60.096243143081665] 
	q3 = [0.35968992248062015, 0.3768996960486322, 0.3994038748137108, 0.4256559766763848, 0.4454022988505747, 0.45915492957746484, 0.47802197802197804, 0.5040431266846362, 0.5245033112582782, 0.5494791666666667, 0.5761843790012804, 0.5944584382871537, 0.6140567200986435, 0.6310679611650486, 0.6491646778042959, 0.6690140845070421, 0.6828703703703703, 0.6964490263459335, 0.7015945330296126, 0.7115600448933783, 0.7171492204899778] 
	t4 = [0, 4.06152606010437, 8.118896245956421, 12.088172197341919, 16.143869876861572, 20.071171283721924, 24.021873950958252, 28.044050931930542, 32.101563930511475, 36.063748836517334, 40.02901601791382, 44.014066219329834, 48.067699670791626, 52.10304260253906, 56.14354944229126, 60.01222538948059] 
	q4 = [0.35968992248062015, 0.38253012048192764, 0.41703377386196766, 0.4454022988505747, 0.4686192468619247, 0.4959349593495935, 0.5245033112582782, 0.5607235142118863, 0.5855513307984791, 0.6157635467980296, 0.636144578313253, 0.6611570247933886, 0.6828703703703703, 0.6971428571428571, 0.7117117117117118, 0.7171492204899778] 
	t5 = [0, 5.110489130020142, 10.030514240264893, 15.11983585357666, 20.068899393081665, 25.049203634262085, 30.037685871124268, 35.10281562805176, 40.13005352020264, 45.14083504676819, 50.06133580207825, 55.083794355392456, 60.13676333427429] 
	q5 = [0.35968992248062015, 0.3892215568862275, 0.43188405797101453, 0.4619718309859155, 0.4959349593495935, 0.5349143610013175, 0.5754475703324808, 0.607940446650124, 0.6377858002406739, 0.6705744431418522, 0.69345579793341, 0.7081447963800905, 0.7185761957730812] 
	t6 = [0, 6.100279331207275, 12.079814195632935, 18.14011001586914, 24.054452657699585, 30.11324644088745, 36.12600612640381, 42.0411434173584, 48.1314377784729, 54.13880920410156, 60.0360963344574] 
	q6 = [0.35968992248062015, 0.3994038748137108, 0.44476327116212333, 0.47867950481430543, 0.5245033112582782, 0.5736235595390525, 0.6165228113440198, 0.6507747318235996, 0.684393063583815, 0.7045454545454545, 0.7185761957730812] 
	t7 = [0, 7.031321048736572, 14.120330095291138, 21.132346630096436, 28.051488876342773, 35.00629806518555, 42.11590647697449, 49.10604953765869, 56.051589488983154] 
	q7 = [0.35968992248062015, 0.4053254437869822, 0.4589235127478754, 0.5080645161290323, 0.5625806451612904, 0.6096654275092938, 0.6539833531510107, 0.6904487917146145, 0.7109111361079866] 
	t8 = [0, 8.04356074333191, 16.00500750541687, 24.14188289642334, 32.00775861740112, 40.10834288597107, 48.01223587989807, 56.018314599990845] 
	q8 = [0.35968992248062015, 0.4140969162995595, 0.46993006993006997, 0.5264550264550265, 0.5873417721518988, 0.6410564225690276, 0.684393063583815, 0.7094594594594594] 
	t9 = [0, 9.095318078994751, 18.0178062915802, 27.137033939361572, 36.09387278556824, 45.007344245910645, 54.117467641830444] 
	q9 = [0.35968992248062015, 0.4256559766763848, 0.48000000000000004, 0.5513654096228869, 0.6165228113440198, 0.6705744431418522, 0.7045454545454545] 
	t10 = [0, 10.127776861190796, 20.030063152313232, 30.053297758102417, 40.114386320114136, 50.1410448551178, 60.10496139526367] 
	q10 = [0.35968992248062015, 0.43125904486251815, 0.4993215739484396, 0.5754475703324808, 0.6434573829531813, 0.6964490263459335, 0.7200000000000001] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.10384202003479, 2.0351593494415283, 3.137641429901123, 4.075869560241699, 5.024151563644409, 6.120245456695557, 7.076704025268555, 8.029924154281616, 9.137207508087158, 10.009272575378418, 11.140170335769653, 12.069201231002808, 13.029557943344116, 14.106544494628906, 15.08406662940979, 16.023269653320312, 17.12897300720215, 18.052266836166382, 19.03107523918152, 20.104134559631348, 21.090724229812622, 22.05825424194336, 23.01513910293579, 24.016384840011597, 25.143568992614746, 26.139830112457275, 27.115766525268555, 28.101880311965942, 29.13631296157837, 30.11775541305542, 31.090200424194336, 32.00910425186157, 33.11112332344055, 34.074912548065186, 35.03836512565613, 36.11360216140747, 37.08753800392151, 38.04151487350464, 39.14384913444519, 40.06503701210022, 41.06697106361389, 42.01611542701721, 43.11570954322815, 44.0768027305603, 45.02319526672363, 46.10261607170105, 47.05084013938904, 48.0080361366272, 49.095025062561035, 50.07468318939209, 51.01627826690674, 52.028071641922, 53.11772608757019, 54.066482067108154, 55.02294111251831, 56.10400629043579, 57.111265897750854, 58.04350304603577, 59.02650737762451, 60.09620976448059] 
	q1 = [0.3572567783094099, 0.36708860759493667, 0.3679245283018868, 0.38317757009345793, 0.38948995363214833, 0.3987730061349693, 0.4054878048780487, 0.41274658573596357, 0.4175491679273827, 0.4270676691729323, 0.4328358208955224, 0.4414814814814815, 0.4477172312223859, 0.45839416058394167, 0.46153846153846156, 0.4726224783861671, 0.48068669527896996, 0.4886363636363637, 0.4992947813822285, 0.5, 0.505586592178771, 0.5159500693481276, 0.5261707988980716, 0.5302197802197802, 0.5340599455040872, 0.5447154471544715, 0.5572005383580081, 0.5622489959839357, 0.5660881174899867, 0.5725699067909454, 0.5812417437252312, 0.5842105263157895, 0.5934640522875817, 0.5999999999999999, 0.6072351421188631, 0.6143958868894601, 0.619718309859155, 0.6226175349428209, 0.6287878787878788, 0.6365914786967418, 0.64, 0.6450809464508096, 0.650185414091471, 0.656019656019656, 0.6593406593406593, 0.6642424242424242, 0.6730769230769231, 0.6794258373205742, 0.6833333333333333, 0.6864608076009502, 0.6949352179034157, 0.6980023501762633, 0.7017543859649121, 0.704784130688448, 0.7062937062937062, 0.7023255813953487, 0.7045191193511008, 0.7037037037037038, 0.7043879907621247, 0.7072330654420206, 0.7116704805491991] 
	t2 = [0, 2.1268553733825684, 4.0992207527160645, 6.064614534378052, 8.078210353851318, 10.123908758163452, 12.056236267089844, 14.018037796020508, 16.037100315093994, 18.000027179718018, 20.01196265220642, 22.04668927192688, 24.072707176208496, 26.063589572906494, 28.053837299346924, 30.02805256843567, 32.03729224205017, 34.14395332336426, 36.142974615097046, 38.04996943473816, 40.02554512023926, 42.09470057487488, 44.09435296058655, 46.076205253601074, 48.099783420562744, 50.137906312942505, 52.05716872215271, 54.076382637023926, 56.08646559715271, 58.11348271369934, 60.05897283554077] 
	q2 = [0.3572567783094099, 0.3704866562009419, 0.39197530864197533, 0.4079147640791477, 0.41993957703927487, 0.4398216939078751, 0.4522760646108664, 0.46599131693198254, 0.4821683309557774, 0.4978902953586498, 0.5097493036211699, 0.5302197802197802, 0.5420054200542005, 0.5603217158176944, 0.5725699067909454, 0.5842105263157895, 0.5999999999999999, 0.6143958868894601, 0.6234096692111959, 0.6365914786967418, 0.6467661691542289, 0.6576687116564417, 0.6650544135429262, 0.6801909307875895, 0.6895734597156399, 0.6971830985915494, 0.704784130688448, 0.7038327526132404, 0.7043879907621247, 0.7087155963302754, 0.7189988623435722] 
	t3 = [0, 3.0305254459381104, 6.0463783740997314, 9.05952525138855, 12.0929434299469, 15.123893976211548, 18.14662265777588, 21.062455892562866, 24.107335805892944, 27.137301683425903, 30.053099155426025, 33.116496562957764, 36.04202389717102, 39.13483381271362, 42.08266806602478, 45.13493299484253, 48.01117992401123, 51.08022689819336, 54.06973171234131, 57.1374773979187, 60.066750288009644] 
	q3 = [0.3572567783094099, 0.3806552262090484, 0.4079147640791477, 0.42942942942942935, 0.4545454545454546, 0.4763271162123386, 0.4978902953586498, 0.5241379310344827, 0.5454545454545455, 0.5641711229946524, 0.5868421052631578, 0.6090322580645161, 0.625158831003812, 0.6416978776529338, 0.6585067319461444, 0.6762589928057554, 0.6926713947990544, 0.7032710280373832, 0.7053364269141531, 0.7072330654420206, 0.7227272727272727] 
	t4 = [0, 4.044890403747559, 8.11553406715393, 12.135585069656372, 16.107906341552734, 20.059551000595093, 24.041853427886963, 28.07602882385254, 32.06644129753113, 36.01296353340149, 40.1181435585022, 44.09707188606262, 48.000900983810425, 52.137513160705566, 56.13203954696655, 60.11619830131531] 
	q4 = [0.3572567783094099, 0.39197530864197533, 0.41993957703927487, 0.4538799414348463, 0.4843304843304843, 0.513888888888889, 0.5454545454545455, 0.5718085106382979, 0.6036269430051814, 0.6243654822335025, 0.6485148514851484, 0.671480144404332, 0.6957547169811321, 0.7039627039627039, 0.7065592635212887, 0.72562358276644] 
	t5 = [0, 5.065242767333984, 10.018216133117676, 15.111172199249268, 20.081200122833252, 25.13726043701172, 30.12035346031189, 35.11190366744995, 40.09964346885681, 45.01341772079468, 50.00719404220581, 55.02365183830261, 60.07996988296509] 
	q5 = [0.3572567783094099, 0.3987730061349693, 0.4398216939078751, 0.47851002865329506, 0.5159500693481276, 0.5572005383580081, 0.5923984272608126, 0.621656050955414, 0.650185414091471, 0.6794258373205742, 0.7025761124121779, 0.7052023121387284, 0.72562358276644] 
	t6 = [0, 6.125445604324341, 12.030800342559814, 18.14254069328308, 24.039398431777954, 30.091728925704956, 36.0382399559021, 42.10997986793518, 48.075047731399536, 54.02006483078003, 60.04683303833008] 
	q6 = [0.3572567783094099, 0.4054878048780487, 0.4545454545454546, 0.5, 0.5474254742547426, 0.5905511811023622, 0.6261089987325729, 0.6593406593406593, 0.6957547169811321, 0.703016241299304, 0.72562358276644] 
	t7 = [0, 7.043611764907837, 14.084587574005127, 21.094029188156128, 28.056665420532227, 35.03165125846863, 42.09549832344055, 49.08416175842285, 56.11940574645996] 
	q7 = [0.3572567783094099, 0.41274658573596357, 0.4733044733044733, 0.5281980742778541, 0.5763612217795485, 0.621656050955414, 0.6609756097560975, 0.6980023501762633, 0.7057471264367815] 
	t8 = [0, 8.06710171699524, 16.06169080734253, 24.053307056427002, 32.03597950935364, 40.068986892700195, 48.1207160949707, 56.073609352111816] 
	q8 = [0.3572567783094099, 0.41993957703927487, 0.4864864864864865, 0.5501355013550137, 0.6044098573281451, 0.6526576019777504, 0.6957547169811321, 0.7057471264367815] 
	t9 = [0, 9.070756196975708, 18.14772367477417, 27.045846939086914, 36.00628137588501, 45.143860816955566, 54.05282545089722] 
	q9 = [0.3572567783094099, 0.4287856071964018, 0.5, 0.5687583444592791, 0.6278481012658228, 0.6817640047675804, 0.703016241299304] 
	t10 = [0, 10.119614601135254, 20.072014570236206, 30.056801557540894, 40.01114058494568, 50.11689043045044, 60.01628375053406] 
	q10 = [0.3572567783094099, 0.4375, 0.5180055401662049, 0.5931758530183726, 0.6526576019777504, 0.7040935672514619, 0.7270668176670442] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 1.0937399864196777, 2.024665355682373, 3.128225564956665, 4.091376543045044, 5.044295310974121, 6.1500184535980225, 7.095489501953125, 8.052555799484253, 9.1453378200531, 10.098939657211304, 11.04302453994751, 12.12613821029663, 13.106080055236816, 14.103241205215454, 15.054578304290771, 16.036682605743408, 17.027039766311646, 18.128664255142212, 19.07500433921814, 20.059596300125122, 21.007156133651733, 22.12272620201111, 23.06769847869873, 24.144907236099243, 25.11882472038269, 26.10021162033081, 27.079021692276, 28.038919687271118, 29.138776779174805, 30.123178720474243, 31.06489372253418, 32.02567386627197, 33.00534701347351, 34.1030912399292, 35.054394245147705, 36.12489604949951, 37.097174644470215, 38.02302050590515, 39.11128497123718, 40.09047317504883, 41.03409481048584, 42.10985517501831, 43.00080871582031, 44.07011818885803, 45.04128384590149, 46.00635576248169, 47.12755513191223, 48.060126066207886, 49.01610565185547, 50.0957190990448, 51.06446313858032, 52.156355142593384, 53.13068389892578, 54.10469031333923, 55.062453508377075, 56.0511040687561, 57.13641691207886, 58.098862648010254, 59.077165842056274, 60.14951777458191] 
	q1 = [0.40120663650075417, 0.4131736526946108, 0.42199108469539376, 0.4283604135893649, 0.4375917767988252, 0.4408759124087591, 0.4515195369030391, 0.46043165467625896, 0.4663805436337626, 0.4744318181818182, 0.4851904090267982, 0.48807854137447393, 0.4951321279554937, 0.5006915629322268, 0.5034387895460798, 0.5102880658436214, 0.5109289617486339, 0.5135869565217391, 0.5202156334231806, 0.5254691689008043, 0.5372340425531915, 0.5403973509933775, 0.5454545454545454, 0.5549738219895288, 0.564369310793238, 0.5717981888745148, 0.5791505791505792, 0.5838668373879642, 0.5885350318471337, 0.5931558935361217, 0.6002522068095838, 0.6047678795483061, 0.6109725685785536, 0.6178660049627792, 0.623921085080148, 0.6323529411764707, 0.6365853658536585, 0.6424242424242425, 0.6457831325301205, 0.6522781774580335, 0.6563245823389021, 0.660332541567696, 0.6619385342789599, 0.6698002350176262, 0.6760233918128654, 0.6813953488372092, 0.6829268292682927, 0.6897347174163783, 0.6942528735632183, 0.695752009184845, 0.6994285714285714, 0.7009132420091325, 0.7, 0.699205448354143, 0.7021517553793885, 0.7058823529411764, 0.705749718151071, 0.7064116985376828, 0.7056179775280899, 0.7142857142857143, 0.7171492204899778] 
	t2 = [0, 2.1186115741729736, 4.114548206329346, 6.100404500961304, 8.095241785049438, 10.15027141571045, 12.117229700088501, 14.037127017974854, 16.042845010757446, 18.098935842514038, 20.11106014251709, 22.034787893295288, 24.00568914413452, 26.026455402374268, 28.10825490951538, 30.079814195632935, 32.14102339744568, 34.01567554473877, 36.130645751953125, 38.142467975616455, 40.03009629249573, 42.00344181060791, 44.05484676361084, 46.08697485923767, 48.087137937545776, 50.00480246543884, 52.11048078536987, 54.11179184913635, 56.01320219039917, 58.044594526290894, 60.01976752281189] 
	q2 = [0.40120663650075417, 0.41666666666666674, 0.439882697947214, 0.453757225433526, 0.4685714285714286, 0.48450704225352104, 0.497913769123783, 0.5061898211829436, 0.5122615803814713, 0.5234899328859061, 0.5403973509933775, 0.5511811023622046, 0.5662337662337662, 0.582798459563543, 0.5931558935361217, 0.6047678795483061, 0.6178660049627792, 0.6306748466257669, 0.6424242424242425, 0.6522781774580335, 0.660332541567696, 0.6698002350176262, 0.6813953488372092, 0.6882217090069285, 0.6972477064220183, 0.7023945267958951, 0.7006802721088435, 0.7065462753950338, 0.7064116985376828, 0.7157190635451505, 0.7192008879023307] 
	t3 = [0, 3.015751361846924, 6.054513454437256, 9.083383083343506, 12.025583028793335, 15.003837585449219, 18.090383291244507, 21.043514490127563, 24.121602773666382, 27.10456609725952, 30.04248881340027, 33.02653241157532, 36.13175439834595, 39.039167404174805, 42.13355302810669, 45.0157573223114, 48.06096172332764, 51.03585886955261, 54.09101438522339, 57.110790967941284, 60.05733847618103] 
	q3 = [0.40120663650075417, 0.4260355029585799, 0.453757225433526, 0.47875354107648727, 0.497913769123783, 0.5095890410958904, 0.5234899328859061, 0.5408970976253298, 0.5680933852140078, 0.5892857142857143, 0.6047678795483061, 0.623921085080148, 0.6424242424242425, 0.6555423122765197, 0.6713615023474178, 0.685979142526072, 0.6987399770904925, 0.699205448354143, 0.7065462753950338, 0.7128491620111732, 0.7220376522702104] 
	t4 = [0, 4.070405006408691, 8.125277996063232, 12.130149841308594, 16.11780595779419, 20.116803407669067, 24.01075839996338, 28.137305736541748, 32.01231622695923, 36.10742115974426, 40.05721044540405, 44.08110857009888, 48.0751953125, 52.112550258636475, 56.09188222885132, 60.082128047943115] 
	q4 = [0.40120663650075417, 0.4369501466275659, 0.4707560627674751, 0.5, 0.5135869565217391, 0.5403973509933775, 0.5680933852140078, 0.5939086294416245, 0.6195786864931847, 0.6457073760580412, 0.6635071090047393, 0.6813953488372092, 0.6964490263459335, 0.7021517553793885, 0.7078651685393258, 0.7226519337016576] 
	t5 = [0, 5.118979215621948, 10.113738298416138, 15.097191572189331, 20.063658237457275, 25.142924070358276, 30.045031785964966, 35.09521007537842, 40.10499620437622, 45.015477418899536, 50.08897924423218, 55.10871171951294, 60.03741717338562] 
	q5 = [0.40120663650075417, 0.44314868804664725, 0.48450704225352104, 0.5102880658436214, 0.5403973509933775, 0.577319587628866, 0.6047678795483061, 0.637469586374696, 0.6635071090047393, 0.6851851851851852, 0.6985210466439136, 0.7072072072072073, 0.7212389380530974] 
	t6 = [0, 6.116238594055176, 12.105875730514526, 18.100828886032104, 24.078404903411865, 30.038821935653687, 36.04976296424866, 42.123486280441284, 48.024662494659424, 54.14381790161133, 60.09356951713562] 
	q6 = [0.40120663650075417, 0.453757225433526, 0.5, 0.5254691689008043, 0.5717981888745148, 0.6090225563909774, 0.644927536231884, 0.6744730679156907, 0.6979405034324944, 0.705749718151071, 0.7226519337016576] 
	t7 = [0, 7.046228647232056, 14.112504482269287, 21.13772988319397, 28.161747932434082, 35.07621622085571, 42.02634358406067, 49.06788110733032, 56.080103397369385] 
	q7 = [0.40120663650075417, 0.46043165467625896, 0.5089408528198074, 0.5473684210526316, 0.5956907477820025, 0.6399026763990269, 0.6744730679156907, 0.6986301369863013, 0.7122060470324749] 
	t8 = [0, 8.05454158782959, 16.007837057113647, 24.082568407058716, 32.051949977874756, 40.12674856185913, 48.047961473464966, 56.13893699645996] 
	q8 = [0.40120663650075417, 0.4685714285714286, 0.5149863760217984, 0.5699481865284974, 0.6195786864931847, 0.6627218934911243, 0.6964490263459335, 0.7107623318385651] 
	t9 = [0, 9.11591911315918, 18.147902965545654, 27.010030508041382, 36.01348376274109, 45.1370313167572, 54.017486810684204] 
	q9 = [0.40120663650075417, 0.47875354107648727, 0.5301204819277108, 0.5903307888040712, 0.644927536231884, 0.6867052023121387, 0.7034949267192785] 
	t10 = [0, 10.06468677520752, 20.034788608551025, 30.047411918640137, 40.001662731170654, 50.04589033126831, 60.097463846206665] 
	q10 = [0.40120663650075417, 0.4838255977496484, 0.5423280423280423, 0.6072772898368882, 0.6627218934911243, 0.6947608200455581, 0.7234513274336285] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	
	q1 = [sum(e)/len(e) for e in zip(*q1_all)]
	q2 = [sum(e)/len(e) for e in zip(*q2_all)]
	q3 = [sum(e)/len(e) for e in zip(*q3_all)]
	q4 = [sum(e)/len(e) for e in zip(*q4_all)]
	q5 = [sum(e)/len(e) for e in zip(*q5_all)]
	q6 = [sum(e)/len(e) for e in zip(*q6_all)]
	q7 = [sum(e)/len(e) for e in zip(*q7_all)]
	q8 = [sum(e)/len(e) for e in zip(*q8_all)]
	q9 = [sum(e)/len(e) for e in zip(*q9_all)]
	q10 = [sum(e)/len(e) for e in zip(*q10_all)]
	
	
	'''
	plt.plot(t1, q1,lw=2,color='green',marker='o',  label='Epoch size(small)')
	plt.plot(t2, q2,lw=2,color='orange',marker='^',  label='Epoch size(large)')
	plt.plot(t3, q3,lw=2,color='blue',marker ='d', label='Epoch size(medium)') ##2,000
	'''
	plt.plot(t1, q1,lw=2,color='blue',marker='o',  label='Iterative Approach(epoch=1)')
	plt.plot(t2, q2,lw=2,color='green',marker='^',  label='Iterative Approach(epoch=2)')
	plt.plot(t3, q3,lw=2,color='orange',marker ='d', label='Iterative Approach(epoch=3)') ##2,000
	plt.plot(t4, q4,lw=2,color='yellow',marker='o',  label='Iterative Approach(epoch=4)')
	plt.plot(t5, q5,lw=2,color='black',marker='^',  label='Iterative Approach(epoch=5)')
	plt.plot(t6, q6,lw=2,color='cyan',marker ='d', label='Iterative Approach(epoch=6)') ##2,000
	
	
	
	'''
	plt.plot(t4, q4,lw=2,color='cyan',marker='o',  label='Iterative Approach')
	plt.plot(t5, q5,lw=2,color='yellow',marker='^',  label='Baseline1 (Function Based Approach)')
	plt.plot(t6, q6,lw=2,color='black',marker ='d', label='Baseline2 (Object Based Approach)') ##2,000
	'''
	
	plt.ylim([0, 1])
	plt.xlim([0, max(max(t1),max(t2),max(t3))])
	plt.title('Quality vs Cost')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .11), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='x-small')
	plt.ylabel('F1-measure')
	#plt.ylabel('Gain')
	plt.xlabel('Cost')	
	plt.savefig('Muct_epoch_size_variation_gender_1000_epoch1.png', format='png')
	plt.savefig('Muct_epoch_size_variation_gender_1000_epoch1.eps', format='eps')
		#plt.show()
	plt.close()
	
	
	q1_new = np.asarray(q1)
	q2_new = np.asarray(q2)
	q3_new = np.asarray(q3)
	q4_new = np.asarray(q4)
	q5_new = np.asarray(q5)
	q6_new = np.asarray(q6)
	q7_new = np.asarray(q7)
	q8_new = np.asarray(q8)
	q9_new = np.asarray(q9)
	q10_new = np.asarray(q10)
	
	t1_new = [sum(e)/len(e) for e in zip(*t1_all)]
	t2_new = [sum(e)/len(e) for e in zip(*t2_all)]
	t3_new = [sum(e)/len(e) for e in zip(*t3_all)]
	t4_new = [sum(e)/len(e) for e in zip(*t4_all)]
	t5_new = [sum(e)/len(e) for e in zip(*t5_all)]
	t6_new = [sum(e)/len(e) for e in zip(*t6_all)]
	t7_new = [sum(e)/len(e) for e in zip(*t7_all)]
	t8_new = [sum(e)/len(e) for e in zip(*t8_all)]
	t9_new = [sum(e)/len(e) for e in zip(*t9_all)]
	t10_new = [sum(e)/len(e) for e in zip(*t10_all)]
	
	
	
	t1_list = [t1_new,t2_new,t3_new,t4_new,t5_new,t6_new,t7_new,t8_new,t9_new,t10_new]
	q1_list = [q1_new,q2_new,q3_new,q4_new,q5_new,q6_new,q7_new,q8_new,q9_new,q10_new]
	#epoch_list = [1,2,4,6,8,10]
	epoch_list = [1,2,3,4,5,6,7,8,9,10]
	score_list = []
	
	for i1 in range(len(t1_list)):
		t1_2 = t1_list[i1]
		t1_2 = t1_2[1:]
		q1_2 = q1_list[i1]
		weight_t1 = [max(1-float(element - 1)/budget,0) for element in t1_2]
		improv_q1 = [x - q1_2[i - 1] for i, x in enumerate(q1_2) if i > 0]
		print weight_t1
		print improv_q1
		a1 = np.dot(weight_t1,improv_q1)
		print a1
		score_list.append(a1)
	print>>f1,"epoch_list = {} ".format(epoch_list)
	print>>f1,"score_list = {} ".format(score_list)	
	plt.plot(epoch_list, score_list,lw=2,color='blue',marker='o',  label='Iterative Approach')
	
	
	#plt.ylim([0, 1])
	plt.xlim([0, max(epoch_list)])
	plt.title('AUC Score vs Epoch Size')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .11), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
	plt.ylabel('AUC Score')
	#plt.ylabel('Gain')
	plt.xlabel('Epoch Size')	
	plt.savefig('EpochSize_AUC_Plot_500.png', format='png')
	plt.savefig('EpochSize_AUC_Plot_500.eps', format='eps')
		#plt.show()
	plt.close()	
	
	##### Plotting with setting the ylim #######
	plt.plot(epoch_list, score_list,lw=2,color='blue',marker='o',  label='Iterative Approach')
	
	
	plt.ylim([0, 1])
	plt.xlim([0, max(epoch_list)])
	plt.title('AUC Score vs Epoch Size')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .11), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
	plt.ylabel('AUC Score')
	#plt.ylabel('Gain')
	plt.xlabel('Epoch Size')	
	plt.savefig('EpochSize_AUC_Plot_ylim_1000.png', format='png')
	plt.savefig('EpochSize_AUC_Plot_ylim_1000.eps', format='eps')
		#plt.show()
	plt.close()





def plotResult4():
	# for 40% selectivity result.
	t1_all,q1_all,t2_all,q2_all,t3_all,q3_all=[],[],[],[],[],[]
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.6593959731543625, 0.6593959731543625, 0.6621392190152802, 0.6638152266894781, 0.6666666666666666, 0.67012987012987, 0.6856634016028496, 0.6939146230699363, 0.6988847583643124, 0.7054337464251669, 0.7402206619859579, 0.7705263157894735, 0.7891304347826087, 0.8022222222222223, 0.8125727590221188, 0.8116279069767443] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.6593959731543625, 0.6638513513513513, 0.6677994902293968, 0.6678023850085179, 0.6689536878216122, 0.672428694900605, 0.6742160278745646, 0.6795434591747146, 0.6831421006178288, 0.6826241134751774, 0.6880570409982174, 0.6917562724014337, 0.6948694869486948, 0.6968325791855203, 0.7006369426751592] 
	t3 = [0, 20.118134021759033, 30.04462218284607, 40.06691646575928, 50.00772166252136, 60.077696323394775, 70.10619330406189, 80.01071882247925, 90.0089852809906, 100.01583313941956, 110.14146709442139, 120.05490827560425, 130.12726998329163, 140.25864958763123, 150.1883065700531, 160.08340644836426] 
	q3 = [0.6593959731543625, 0.6588235294117648, 0.6621392190152802, 0.6655290102389079, 0.6660944206008583, 0.6724738675958188, 0.6887298747763864, 0.6916058394160582, 0.70533208606174, 0.7062146892655367, 0.7303822937625755, 0.758985200845666, 0.7950727883538634, 0.8132875143184422, 0.8124999999999999, 0.8106235565819861] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.6446140797285835, 0.6446140797285835, 0.6490179333902648, 0.6540447504302926, 0.6557093425605537, 0.663167104111986, 0.6731967943009795, 0.6893382352941176, 0.7045454545454547, 0.7149853085210577, 0.7492260061919503, 0.7817589576547233, 0.7991021324354658, 0.8168674698795181, 0.8139255702280913, 0.8139255702280913] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.6446140797285835, 0.6523605150214593, 0.6563039723661486, 0.6568201563857515, 0.6596675415573053, 0.6625659050966608, 0.663716814159292, 0.6708407871198568, 0.6696588868940755, 0.6726780883678989, 0.6757246376811595, 0.6763636363636364, 0.6788321167883211, 0.6801099908340972, 0.6851338873499538] 
	t3 = [0, 20.09520983695984, 30.06803822517395, 40.07307457923889, 50.092041015625, 60.11441421508789, 70.14920711517334, 80.12093663215637, 90.08944272994995, 100.01961994171143, 110.03683471679688, 120.03084135055542, 130.00976610183716, 140.34721994400024, 150.17124843597412, 160.0190794467926] 
	q3 = [0.6446140797285835, 0.6451612903225806, 0.6484641638225256, 0.6500857632933105, 0.6545768566493955, 0.663167104111986, 0.6749999999999999, 0.6881324747010118, 0.7020872865275142, 0.7018216682646214, 0.7213114754098361, 0.7567567567567567, 0.7940503432494279, 0.8127208480565371, 0.8117647058823529, 0.8127962085308056] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.6571428571428571, 0.6576955424726662, 0.6621507197290432, 0.6655290102389079, 0.6712564543889845, 0.6777583187390542, 0.6917562724014338, 0.7014652014652014, 0.7086466165413534, 0.7207729468599033, 0.7482305358948433, 0.782608695652174, 0.8008705114254625, 0.8235294117647057, 0.8321678321678321, 0.8321678321678321] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.6571428571428571, 0.6593406593406593, 0.6638297872340426, 0.6638078902229846, 0.664944013781223, 0.6695576756287944, 0.6742358078602622, 0.6778169014084506, 0.6790450928381964, 0.6844919786096256, 0.6857142857142858, 0.6894075403949732, 0.6871055004508567, 0.6877828054298644, 0.6927985414767548] 
	t3 = [0, 20.13560461997986, 30.09698796272278, 40.054455041885376, 50.00636529922485, 60.11031699180603, 70.01649475097656, 80.03450226783752, 90.03429460525513, 100.01489472389221, 110.14949083328247, 120.14600872993469, 130.08983421325684, 140.08009314537048, 150.01108026504517, 160.15542101860046] 
	q3 = [0.6571428571428571, 0.6582491582491583, 0.661590524534687, 0.6666666666666666, 0.6695427092320967, 0.6789797713280563, 0.6924460431654677, 0.6985294117647058, 0.7118644067796609, 0.715370018975332, 0.7401415571284126, 0.7716701902748416, 0.8066666666666665, 0.8251428571428573, 0.8251428571428573, 0.8287037037037036] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.6537489469250211, 0.6537489469250211, 0.6570458404074703, 0.657580919931857, 0.6609294320137694, 0.6660854402789887, 0.6821844225604297, 0.695970695970696, 0.7074317968015053, 0.7248062015503876, 0.7489878542510121, 0.777422790202343, 0.8, 0.8030973451327433, 0.8225616921269095, 0.8215962441314554] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.6537489469250211, 0.6570458404074703, 0.6592844974446337, 0.6615120274914089, 0.6666666666666667, 0.6724890829694322, 0.6760316066725197, 0.6772767462422634, 0.6803205699020481, 0.6845878136200718, 0.6876687668766878, 0.6895306859205776, 0.6884650317892824, 0.686703096539162, 0.6917431192660551] 
	t3 = [0, 20.027994632720947, 30.11117649078369, 40.03334951400757, 50.112826108932495, 60.09465003013611, 70.07070541381836, 80.01008152961731, 90.11030197143555, 100.00873970985413, 110.05124974250793, 120.01671552658081, 130.13173937797546, 140.16552639007568, 150.11632251739502, 160.18084692955017] 
	q3 = [0.6537489469250211, 0.6537489469250211, 0.6559322033898305, 0.658703071672355, 0.6592082616179001, 0.668416447944007, 0.6858685868586858, 0.6974169741697417, 0.7160493827160493, 0.7215311004784688, 0.7446153846153847, 0.7720430107526882, 0.8068181818181819, 0.8175519630484988, 0.815668202764977, 0.8186046511627906] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.661641541038526, 0.6599664991624791, 0.6655405405405405, 0.6672340425531914, 0.6729613733905578, 0.6829694323144105, 0.695111111111111, 0.707916287534122, 0.7194780987884436, 0.737247353224254, 0.7615230460921844, 0.7916230366492146, 0.8064516129032258, 0.8184892897406989, 0.8203939745075319, 0.8217592592592593] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.661641541038526, 0.6666666666666666, 0.6689303904923599, 0.6712328767123288, 0.6747195858498706, 0.6753472222222223, 0.6782911944202266, 0.6801051709027169, 0.6813380281690142, 0.6849557522123895, 0.6874443455031166, 0.6905187835420393, 0.6929982046678635, 0.6992753623188407, 0.7024567788899] 
	t3 = [0, 20.00779938697815, 30.04100251197815, 40.08920454978943, 50.07970952987671, 60.083906412124634, 70.01527094841003, 80.15688896179199, 90.10660886764526, 100.01903581619263, 110.08233976364136, 120.0172472000122, 130.14466404914856, 140.38808012008667, 150.27239608764648, 160.00833821296692] 
	q3 = [0.661641541038526, 0.6621961441743504, 0.6661031276415892, 0.6695059625212948, 0.6706689536878216, 0.6794425087108015, 0.6945681211041852, 0.7093235831809872, 0.7247191011235955, 0.7262464722483537, 0.7522750252780587, 0.7809523809523808, 0.8128460686600222, 0.824858757062147, 0.8229988726042842, 0.8216340621403914] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.6782464846980977, 0.6782464846980977, 0.6804979253112033, 0.6839464882943144, 0.6873949579831934, 0.6963216424294268, 0.7091703056768559, 0.7191413237924866, 0.7247706422018348, 0.7354596622889304, 0.7669616519174041, 0.7876923076923078, 0.8000000000000002, 0.8129175946547884, 0.8094701240135288, 0.8085585585585585] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.6782464846980977, 0.680532445923461, 0.6817420435510887, 0.6846543001686342, 0.6847457627118644, 0.6859574468085108, 0.6912521440823326, 0.6907216494845361, 0.693171996542783, 0.6961805555555556, 0.6979982593559617, 0.699912510936133, 0.7005253940455342, 0.7012302284710018, 0.7031802120141343] 
	t3 = [0, 20.104210376739502, 30.0433828830719, 40.14008855819702, 50.032832860946655, 60.0274395942688, 70.0275022983551, 80.00914168357849, 90.14088654518127, 100.00825810432434, 110.04000997543335, 120.10678100585938, 130.15100407600403, 140.2864944934845, 150.1378927230835, 160.08136248588562] 
	q3 = [0.6782464846980977, 0.6782464846980977, 0.6799667497921862, 0.6828046744574291, 0.6863406408094436, 0.6928999144568007, 0.7092819614711032, 0.720575022461815, 0.7259395050412465, 0.7287822878228781, 0.7433628318584071, 0.7615148413510746, 0.7982740021574972, 0.8114663726571114, 0.8109890109890111, 0.8089385474860336] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.6480541455160744, 0.6491525423728813, 0.6507666098807496, 0.6563573883161512, 0.6614718614718615, 0.6761061946902656, 0.6920289855072463, 0.7017543859649122, 0.7160493827160493, 0.736426456071076, 0.7689119170984456, 0.7995666305525462, 0.8074807480748074, 0.8284023668639052, 0.8284023668639052, 0.8288075560802833] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.6480541455160744, 0.6541417591801878, 0.6569468267581475, 0.6597582037996546, 0.6649260226283724, 0.666083916083916, 0.6696035242290749, 0.6702033598585322, 0.6720142602495544, 0.6756272401433693, 0.6792792792792792, 0.6823956442831215, 0.6842584167424932, 0.6886446886446885, 0.6930875576036867] 
	t3 = [0, 20.04122281074524, 30.108298301696777, 40.07076573371887, 50.13337254524231, 60.03663372993469, 70.05999755859375, 80.01693224906921, 90.00757098197937, 100.14552593231201, 110.05179619789124, 120.0283772945404, 130.6724464893341, 140.11721086502075, 150.23111987113953, 160.37542414665222] 
	q3 = [0.6480541455160744, 0.6497031382527566, 0.6513213981244672, 0.655793991416309, 0.6608996539792387, 0.6737213403880071, 0.6927272727272727, 0.7038068709377903, 0.7136060894386298, 0.7219999999999999, 0.7500000000000001, 0.7833698030634574, 0.823121387283237, 0.8202764976958525, 0.8262910798122067, 0.8243559718969554] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.6627615062761506, 0.6627615062761506, 0.6683544303797468, 0.6712095400340716, 0.6758147512864493, 0.6846689895470384, 0.6981300089047195, 0.708029197080292, 0.722326454033771, 0.7364341085271319, 0.7687626774847869, 0.7961783439490445, 0.8012958963282937, 0.8119266055045872, 0.8119266055045872, 0.8119266055045872] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.6627615062761506, 0.6672283066554338, 0.6694843617920541, 0.6746166950596253, 0.6798283261802575, 0.682800345721694, 0.6852173913043479, 0.6870095902353968, 0.6924428822495607, 0.694273127753304, 0.6974267968056789, 0.697508896797153, 0.7000000000000001, 0.7031390134529149, 0.7038703870387039] 
	t3 = [0, 20.015430212020874, 30.124768495559692, 40.04687571525574, 50.14430809020996, 60.071430921554565, 70.05005764961243, 80.14409828186035, 90.00479435920715, 100.02768564224243, 110.05739974975586, 120.01935362815857, 130.10311651229858, 140.0409243106842, 150.02025866508484, 160.31752610206604] 
	q3 = [0.6627615062761506, 0.6627615062761506, 0.6683544303797468, 0.6728971962616822, 0.676419965576592, 0.6864628820960699, 0.7000000000000001, 0.7099725526075022, 0.7208216619981326, 0.7290448343079923, 0.7469635627530364, 0.7728237791932059, 0.8080808080808082, 0.8116591928251121, 0.8127853881278538, 0.8118586088939567] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.6582703610411419, 0.6576955424726662, 0.6627118644067796, 0.6666666666666666, 0.6689536878216124, 0.6759581881533101, 0.6940966010733453, 0.710091743119266, 0.7218045112781954, 0.7276264591439688, 0.7638603696098561, 0.7952586206896552, 0.8039430449069003, 0.8171296296296297, 0.8171296296296297, 0.8171296296296297] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.6582703610411419, 0.6666666666666666, 0.667235494880546, 0.6718213058419243, 0.6747404844290656, 0.6765475152571926, 0.6830985915492958, 0.6831421006178288, 0.6862222222222222, 0.6887298747763863, 0.6900269541778975, 0.6907775768535264, 0.6914027149321268, 0.6964448495897902, 0.6996336996336997] 
	t3 = [0, 20.101510047912598, 30.084825038909912, 40.07823729515076, 50.05051565170288, 60.158740758895874, 70.14650535583496, 80.09487462043762, 90.00447797775269, 100.02077651023865, 110.03921341896057, 120.0342149734497, 130.0287322998047, 140.35934472084045, 150.33171463012695, 160.23448395729065] 
	q3 = [0.6582703610411419, 0.6593591905564924, 0.6632739609838847, 0.666098807495741, 0.6701030927835051, 0.6794759825327511, 0.6977578475336323, 0.711520737327189, 0.7200754005655042, 0.7283464566929134, 0.7352342158859471, 0.7698924731182796, 0.8127853881278538, 0.8145620022753128, 0.815668202764977, 0.8183908045977011] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.6503378378378378, 0.6514382402707276, 0.65587734241908, 0.6580976863753214, 0.6592082616179001, 0.6707638279192274, 0.6852517985611509, 0.6967741935483871, 0.7084520417853751, 0.7181996086105676, 0.7487179487179486, 0.7789699570815449, 0.7835497835497836, 0.813953488372093, 0.8120649651972157, 0.8129330254041571] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.6503378378378378, 0.6547619047619047, 0.6569965870307167, 0.6597938144329897, 0.6643598615916956, 0.6655052264808363, 0.6707638279192274, 0.6725352112676056, 0.6779059449866902, 0.6821428571428572, 0.6845878136200717, 0.6846361185983828, 0.6871055004508566, 0.6933575978161967, 0.6972477064220183] 
	t3 = [0, 20.080305814743042, 30.039318084716797, 40.007121562957764, 50.06631064414978, 60.151249170303345, 70.10536313056946, 80.07410454750061, 90.01839518547058, 100.01481986045837, 110.06234049797058, 120.05330896377563, 130.13200116157532, 140.02992582321167, 150.21690607070923, 160.34520769119263] 
	q3 = [0.6503378378378378, 0.650887573964497, 0.6553191489361703, 0.6580976863753214, 0.6620570440795159, 0.6701754385964912, 0.6846846846846847, 0.6962142197599261, 0.704438149197356, 0.7186574531095755, 0.7397540983606558, 0.7682403433476395, 0.8009049773755655, 0.802721088435374, 0.8114942528735631, 0.8105625717566016] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.6571428571428571, 0.6571428571428571, 0.6610312764158918, 0.6638297872340426, 0.6672384219554031, 0.6695576756287944, 0.688670829616414, 0.7007299270072994, 0.711992445703494, 0.7283349561830574, 0.7641606591143151, 0.7991313789359391, 0.8026315789473685, 0.8329411764705882, 0.8329411764705882, 0.8309859154929577] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.6571428571428571, 0.6598984771573604, 0.6615646258503401, 0.6643835616438355, 0.6672413793103449, 0.6724587315377932, 0.6771929824561403, 0.6795774647887325, 0.6814159292035399, 0.6844444444444444, 0.6851520572450804, 0.6858168761220826, 0.6907775768535263, 0.6946216955332726, 0.7003676470588235] 
	t3 = [0, 20.082131147384644, 30.02110481262207, 40.1231005191803, 50.07911014556885, 60.008206367492676, 70.06771945953369, 80.04179763793945, 90.02430462837219, 100.01150560379028, 110.0997519493103, 120.00254464149475, 130.22374296188354, 140.34639859199524, 150.17247986793518, 160.35442447662354] 
	q3 = [0.6571428571428571, 0.6571428571428571, 0.6598984771573604, 0.6632566069906224, 0.6643776824034335, 0.6742160278745645, 0.6923766816143498, 0.7046918123275069, 0.7105263157894737, 0.7297830374753452, 0.7520661157024794, 0.7882096069868996, 0.8267898383371823, 0.8271889400921659, 0.830409356725146, 0.830409356725146] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.6341880341880342, 0.6335616438356164, 0.6355785837651122, 0.6371527777777778, 0.6409807355516637, 0.6512042818911686, 0.6672777268560953, 0.6804511278195489, 0.6905916585838991, 0.7144298688193744, 0.7518636847710329, 0.7803790412486066, 0.7855530474040632, 0.8100961538461539, 0.8100961538461539, 0.8100961538461539] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.6341880341880342, 0.6373815676141258, 0.6412478336221837, 0.6457242582897033, 0.6490765171503957, 0.6519043401240036, 0.6565295169946332, 0.6588658865886589, 0.6618444846292948, 0.6648451730418944, 0.6697247706422018, 0.6728110599078342, 0.6753246753246753, 0.6778398510242086, 0.6803738317757009] 
	t3 = [0, 20.14240789413452, 30.121273279190063, 40.064773082733154, 50.133097887039185, 60.101014852523804, 70.00947403907776, 80.10059261322021, 90.01083827018738, 100.00390219688416, 110.01407766342163, 120.10705780982971, 130.51196455955505, 140.24993300437927, 150.02138423919678, 160.29328894615173] 
	q3 = [0.6341880341880342, 0.6352739726027398, 0.6377816291161178, 0.64, 0.6409130816505707, 0.6540880503144654, 0.6709677419354838, 0.6824644549763034, 0.6964112512124152, 0.7126903553299493, 0.7309322033898304, 0.760845383759733, 0.806146572104019, 0.8056537102473498, 0.8056537102473498, 0.8052568697729987] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.6605196982397318, 0.6588432523051131, 0.6627218934911242, 0.6660988074957411, 0.6689536878216125, 0.6777003484320557, 0.6916221033868093, 0.7032967032967034, 0.7173708920187792, 0.7380720545277507, 0.7681307456588355, 0.7944325481798715, 0.8052230685527749, 0.8227114716106605, 0.8208092485549132, 0.8208092485549132] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.6605196982397318, 0.6644182124789207, 0.6661031276415892, 0.6695059625212947, 0.6712328767123288, 0.6712564543889846, 0.6765217391304348, 0.6771378708551484, 0.6819383259911894, 0.6855624446412754, 0.6874443455031166, 0.6875559534467323, 0.69009009009009, 0.6950998185117968, 0.697632058287796] 
	t3 = [0, 20.067198514938354, 30.050883293151855, 40.06103539466858, 50.02644348144531, 60.131364583969116, 70.0238287448883, 80.1122477054596, 90.01094913482666, 100.01088166236877, 110.06689476966858, 120.03923273086548, 130.02351140975952, 140.12976551055908, 150.1640284061432, 160.36321330070496] 
	q3 = [0.6605196982397318, 0.6593959731543624, 0.6627218934911242, 0.6644010195412065, 0.671244635193133, 0.6753698868581376, 0.692927484333035, 0.7063129002744739, 0.7185741088180113, 0.7362204724409448, 0.7517875383043922, 0.7832618025751072, 0.8217934165720772, 0.8217934165720772, 0.8225806451612904, 0.8216340621403911] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.6560134566862911, 0.6560134566862911, 0.6621392190152802, 0.6672369546621043, 0.6706896551724139, 0.6782911944202267, 0.6934763181411975, 0.7044830741079597, 0.7209737827715356, 0.7395121951219512, 0.7741273100616016, 0.8056155507559396, 0.814489571899012, 0.8358556461001163, 0.8372093023255814, 0.8372093023255814] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.6560134566862911, 0.6598812553011028, 0.6632566069906223, 0.668384879725086, 0.6735930735930736, 0.6765475152571926, 0.6824978012313105, 0.6849073256840248, 0.6843971631205673, 0.6886708296164138, 0.6911369740376007, 0.696122633002705, 0.6987295825771326, 0.7007299270072993, 0.7065317387304508] 
	t3 = [0, 20.039540767669678, 30.131874561309814, 40.0993378162384, 50.054763317108154, 60.012309551239014, 70.14023542404175, 80.09489750862122, 90.01692461967468, 100.03440237045288, 110.04713654518127, 120.01352906227112, 130.14023971557617, 140.34830927848816, 150.16938734054565, 160.1133954524994] 
	q3 = [0.6560134566862911, 0.6565656565656565, 0.6604572396274344, 0.6649572649572649, 0.6689595872742906, 0.6783216783216783, 0.6935483870967742, 0.7100917431192659, 0.7212806026365348, 0.7371541501976285, 0.7584789311408018, 0.79004329004329, 0.8344827586206897, 0.8325688073394496, 0.8310502283105022, 0.8346820809248555] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.6560134566862911, 0.6560134566862911, 0.6587436332767402, 0.6626712328767123, 0.6649395509499136, 0.6731107205623902, 0.6864864864864866, 0.699443413729128, 0.7106017191977078, 0.7278106508875739, 0.7614107883817427, 0.7874186550976139, 0.7938257993384785, 0.8089622641509434, 0.8103651354534748, 0.8084606345475909] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.6560134566862911, 0.6604414261460102, 0.6643894107600342, 0.6620808254514187, 0.6637856525496976, 0.6672473867595818, 0.6695880806310254, 0.6696035242290749, 0.6719717064544649, 0.6755793226381461, 0.6779964221824687, 0.6828828828828828, 0.6854034451495921, 0.6854545454545454, 0.6904761904761905] 
	t3 = [0, 20.019575834274292, 30.140846967697144, 40.12386631965637, 50.119998931884766, 60.0968861579895, 70.02924752235413, 80.09384250640869, 90.01773738861084, 100.00572991371155, 110.02675271034241, 120.11914587020874, 130.1084017753601, 140.2413568496704, 150.122980594635, 160.22633266448975] 
	q3 = [0.6560134566862911, 0.6565656565656565, 0.6587436332767402, 0.6615384615384615, 0.6632213608957795, 0.6695880806310254, 0.6815742397137746, 0.6918123275068997, 0.7064393939393938, 0.7085714285714286, 0.7244897959183673, 0.7521367521367521, 0.7913832199546486, 0.8023121387283236, 0.7995391705069125, 0.8028004667444574] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.6434634974533108, 0.6434042553191489, 0.6472103004291846, 0.6511226252158895, 0.6556521739130435, 0.6660761736049601, 0.6823956442831215, 0.6952469711090401, 0.7004784688995215, 0.7176938369781314, 0.7518324607329843, 0.7828947368421054, 0.7884828349944629, 0.8047058823529412, 0.8042203985932005, 0.8046783625730993] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.6434634974533108, 0.6500857632933105, 0.6528854435831181, 0.655680832610581, 0.6596675415573053, 0.6631393298059964, 0.6672597864768685, 0.6702508960573477, 0.6714542190305207, 0.6763110307414105, 0.6787989080982711, 0.6812785388127853, 0.6838235294117646, 0.6869806094182825, 0.6915191053122088] 
	t3 = [0, 20.014386892318726, 30.127787590026855, 40.10811400413513, 50.02231502532959, 60.00843644142151, 70.12809586524963, 80.12967705726624, 90.01709294319153, 100.02437734603882, 110.11729550361633, 120.08345532417297, 130.01514434814453, 140.3658640384674, 150.28685879707336, 160.0374038219452] 
	q3 = [0.6434634974533108, 0.6428571428571428, 0.6472103004291846, 0.6511226252158895, 0.6539130434782608, 0.6672582076308785, 0.6818596171376482, 0.694756554307116, 0.7045235803657363, 0.7184079601990049, 0.7432712215320911, 0.7717391304347826, 0.805045871559633, 0.805491990846682, 0.8051341890315051, 0.8055878928987193] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.6661101836393989, 0.664440734557596, 0.6672254819782062, 0.6700421940928271, 0.6723404255319149, 0.6816608996539792, 0.6902654867256638, 0.7078039927404719, 0.7184284377923292, 0.736231884057971, 0.7670396744659207, 0.7970244420828905, 0.8004314994606258, 0.8252873563218391, 0.8243398392652124, 0.823394495412844] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.6661101836393989, 0.6722830665543387, 0.6722972972972973, 0.6751700680272109, 0.6775021385799829, 0.6833477135461604, 0.6863084922010397, 0.6904969485614646, 0.6917030567685589, 0.6977973568281938, 0.6996466431095405, 0.7046263345195729, 0.7053571428571428, 0.7091561938958708, 0.7155797101449275] 
	t3 = [0, 20.0622239112854, 30.12395739555359, 40.049232482910156, 50.02587151527405, 60.1431565284729, 70.03676342964172, 80.05208015441895, 90.00175738334656, 100.10791850090027, 110.02815246582031, 120.04647040367126, 130.0432207584381, 140.27113723754883, 150.02191758155823, 160.0764079093933] 
	q3 = [0.6661101836393989, 0.6655518394648829, 0.6683459277917715, 0.669484361792054, 0.6723404255319149, 0.6793760831889081, 0.6927175843694494, 0.7097361237488626, 0.7164179104477612, 0.7283349561830574, 0.7497467071935158, 0.7876200640341516, 0.8179775280898877, 0.8183856502242153, 0.8183856502242153, 0.8205714285714285] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.680429397192403, 0.6804635761589405, 0.683903252710592, 0.6873949579831933, 0.6892464013547842, 0.6940170940170941, 0.7032201914708441, 0.716577540106952, 0.7274384685505926, 0.7422680412371134, 0.774384236453202, 0.8061855670103092, 0.8125654450261779, 0.843159065628476, 0.8422222222222222, 0.8403547671840355] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.680429397192403, 0.6838602329450915, 0.68671679197995, 0.6862910008410429, 0.689189189189189, 0.6927659574468085, 0.6963979416809606, 0.7, 0.7012987012987014, 0.705574912891986, 0.7086614173228346, 0.7111501316944688, 0.7113857016769638, 0.7140319715808171, 0.719785138764548] 
	t3 = [0, 20.024445056915283, 30.084110736846924, 40.03909158706665, 50.13450908660889, 60.09648275375366, 70.03057718276978, 80.05329275131226, 90.01710367202759, 100.0191559791565, 110.13059282302856, 120.12920761108398, 131.40507316589355, 140.16322779655457, 150.02228713035583, 160.29448699951172] 
	q3 = [0.680429397192403, 0.6804635761589405, 0.684474123539232, 0.6873949579831933, 0.6886632825719119, 0.6964746345657781, 0.7070175438596492, 0.7179946284691137, 0.7272727272727273, 0.7471482889733839, 0.7652859960552268, 0.8012422360248448, 0.836244541484716, 0.8353326063249727, 0.8403547671840355, 0.8384955752212391] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	'''
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.6491969568892646, 0.649746192893401, 0.6541095890410958, 0.658599827139153, 0.6620330147697655, 0.6725507502206531, 0.6847826086956523, 0.6938775510204082, 0.7100478468899523, 0.7285291214215203, 0.7590486039296794, 0.7921653971708378, 0.7969094922737306, 0.8276670574443141, 0.8266978922716629, 0.8266978922716629] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100] 
	q2 = [0.6491969568892646, 0.6524701873935264, 0.6569468267581475, 0.657487091222031, 0.6602953953084274, 0.6643356643356643, 0.6696035242290749, 0.6755555555555555, 0.6779964221824688, 0.6810810810810812] 
	t3 = [0, 20.077027559280396, 30.017561197280884, 40.04065942764282, 50.002402782440186, 60.045904874801636, 70.14121961593628, 80.11852025985718, 90.02785539627075, 100.08745169639587, 110.01771855354309, 120.1226634979248, 130.06640219688416, 140.24255418777466, 150.0133571624756, 160.2586555480957] 
	q3 = [0.6491969568892646, 0.6508474576271187, 0.6546700942587832, 0.6586206896551725, 0.662608695652174, 0.6707964601769911, 0.6842584167424931, 0.700280112044818, 0.7061068702290076, 0.7225548902195609, 0.7466110531803961, 0.7828947368421053, 0.8190255220417634, 0.8208092485549132, 0.8189158016147635, 0.8242142025611175] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	'''
	
	t1 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160] 
	q1 = [0.646909398814564, 0.646909398814564, 0.6518771331058021, 0.6546080964685616, 0.6585577758470895, 0.664323374340949, 0.6744186046511628, 0.6867579908675798, 0.699812382739212, 0.7058823529411765, 0.7256281407035176, 0.7536842105263158, 0.7828004410143329, 0.792368125701459, 0.8139255702280911, 0.8129496402877698] 
	t2 = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150] 
	q2 = [0.646909398814564, 0.6552315608919382, 0.6569217540842648, 0.6597402597402597, 0.6614447345517841, 0.6666666666666666, 0.6654898499558694, 0.6708185053380783, 0.6702412868632708, 0.6696508504923904, 0.6750902527075812, 0.6757000903342366, 0.67574931880109, 0.680073126142596, 0.6844526218951242] 
	t3 = [0, 20.02149248123169, 30.053501844406128, 40.1067578792572, 50.08737111091614, 60.088374376297, 70.13886880874634, 80.04632234573364, 90.07422733306885, 100.00177836418152, 110.02486658096313, 120.14555740356445, 130.08788681030273, 140.0103418827057, 150.26353979110718, 160.08070421218872] 
	q3 = [0.646909398814564, 0.6485568760611207, 0.6507258753202392, 0.6574633304572908, 0.6585365853658537, 0.6649076517150396, 0.6762331838565022, 0.6862925482980681, 0.7005649717514125, 0.7064485081809433, 0.7236180904522613, 0.7424400417101147, 0.7702407002188183, 0.8022988505747126, 0.813599062133646, 0.8116959064327485] 
	q1_all.append(q1)
	t1_all.append(t1)
	q2_all.append(q2)
	t2_all.append(t2)
	q3_all.append(q3)
	t3_all.append(t3)
	
	t1_new = [sum(e)/len(e) for e in zip(*t1_all)]
	q1_new = [sum(e)/len(e) for e in zip(*q1_all)]
	t2_new = [sum(e)/len(e) for e in zip(*t2_all)]
	q2_new = [sum(e)/len(e) for e in zip(*q2_all)]	
	t3_new = [sum(e)/len(e) for e in zip(*t3_all)]
	q3_new = [sum(e)/len(e) for e in zip(*q3_all)]
	print t1_new
	print q1_new
	print t2_new
	print q2_new
	print t3_new
	print q3_new
	q2_new.append(q2_new[len(q2_new)-1])
	t2_new.append(t2_new[len(t2_new)-1])
	q1_new = np.asarray(q1_new)
	q2_new = np.asarray(q2_new)
	q3_new = np.asarray(q3_new)
	
	
	min_val = min(min(q1_new),min(q2_new),min(q3_new))
	max_val = max(max(q1_new),max(q2_new),max(q3_new))
	
	q1_norm = (q1_new-min_val)/(max_val - min_val)
	q2_norm = (q2_new-min_val)/(max_val - min_val)
	q3_norm = (q3_new-min_val)/(max_val - min_val)
	
	
	plt.plot(t1_new, q1_norm,lw=2,color='green',  label='Baseline1 (Function Based Approach)')
	plt.plot(t2_new, q2_norm,lw=2,color='orange',  label='Baseline2 (Object Based Approach)')
	plt.plot(t3_new, q3_norm,lw=2,color='blue', label='Iterative Approach') ##2,000
	
	#plt.plot(t1, q1,lw=2,color='green', marker ='d', label='Baseline1 (Function Based Approach)')
	#plt.plot(t2, q2,lw=2,color='orange', marker ='o', label='Baseline2 (Object Based Approach)')
	#plt.plot(t3, q3,lw=2,color='blue',marker ='^',  label='Iterative Approach') ##2,000
	
	
	'''
	#plt.plot(t1_new, q1_new,lw=2,color='green',  label='Baseline1 (Function Based Approach)')
	#plt.plot(t2_new, q2_new,lw=2,color='orange',  label='Baseline2 (Object Based Approach)')
	#plt.plot(t3_new, q3_new,lw=2,color='blue', label='Iterative Approach') ##2,000
	'''

	
	plt.title('Quality vs Cost')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
	#plt.legend(bbox_to_anchor=(0, 1),loc="upper left",fontsize='xx-small')
	#bbox_to_anchor=(0, 1), loc='upper left', ncol=1
	plt.ylabel('F1-measure')
	plt.xlabel('Cost')
	plt.ylim([0, 1])
	plt.xlim([0, max(max(t1_new),max(t2_new),max(t3_new))])	
	plt.savefig('ImageMuctBaseline_gender_40percent.png', format='png')
	plt.savefig('ImageMuctBaseline_gender_40percent.eps', format='eps')
		#plt.show()
	plt.close()
	

def plotOptimalEpoch2():
	#50objects. budget 50 seconds
	budget = 20
	epoch_list = [1,2,3,4,5,6,7,8,9,10]
	t1_all,q1_all,t2_all,q2_all,t3_all,q3_all=[],[],[],[],[],[]
	t4_all,q4_all,t5_all,q5_all,t6_all,q6_all=[],[],[],[],[],[]
	t7_all,q7_all,t8_all,q8_all,t9_all,q9_all=[],[],[],[],[],[]	
	t10_all,q10_all=[],[]
	# Plotting epoch for 1000 objects.
	
	f1 = open('PlotEpoch.txt','w+')
	
	t1 = [0, 10.162250518798828, 20.000030279159546, 30.000805616378784, 40.00080370903015, 50.00060558319092] 
	q1 = [0.39999999999999997, 0.8461538461538461, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906] 
	t2 = [0, 5.081209421157837, 10.027287483215332, 15.179351091384888, 20.001975774765015, 25.000702381134033, 30.001688718795776, 35.00017070770264, 40.00057530403137, 45.0002543926239, 50.0007541179657] 
	q2 = [0.39999999999999997, 0.7755102040816326, 0.8235294117647058, 0.8679245283018868, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906] 
	t3 = [0, 3.3775858879089355, 6.788924217224121, 10.06868314743042, 13.642565727233887, 16.65484929084778, 19.980329513549805, 23.3102548122406, 26.640275478363037, 29.971835613250732, 33.30031871795654, 36.63188576698303, 39.96054935455322, 43.29100036621094, 46.62035322189331, 49.950392961502075] 
	q3 = [0.39999999999999997, 0.7111111111111111, 0.8235294117647058, 0.8461538461538461, 0.8461538461538461, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906] 
	t4 = [0, 2.5759031772613525, 5.097277402877808, 7.5826427936553955, 10.218758821487427, 12.582567691802979, 15.055986881256104, 17.501088857650757, 20.001384496688843, 22.50028681755066, 25.001933574676514, 27.501731634140015, 30.000816583633423, 32.50015473365784, 35.001487731933594, 37.50114417076111, 40.00128126144409, 42.500712156295776, 45.00265717506409, 47.50079965591431, 50.00157070159912] 
	q4 = [0.39999999999999997, 0.6511627906976744, 0.8, 0.8461538461538461, 0.8461538461538461, 0.8461538461538461, 0.8461538461538461, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906] 
	t5 = [0, 2.0719211101531982, 4.113810777664185, 6.019743204116821, 8.252767562866211, 10.109732627868652, 12.119037628173828, 14.176136016845703, 16.19379425048828, 18.000348806381226, 20.00095558166504, 22.000022411346436, 24.001147985458374, 26.00086498260498, 28.001439809799194, 30.00001096725464, 32.000054597854614, 34.00208783149719, 36.00186729431152, 38.00044918060303, 40.000234842300415, 42.0010769367218, 44.00081944465637, 46.00172781944275, 48.00171780586243, 50.00075912475586] 
	q5 = [0.39999999999999997, 0.6666666666666667, 0.7916666666666666, 0.8, 0.8235294117647058, 0.8461538461538461, 0.8461538461538461, 0.8461538461538461, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906] 
	t6 = [0, 1.7532010078430176, 3.496068000793457, 5.118725776672363, 6.990796804428101, 8.571522235870361, 10.481756925582886, 12.14781928062439, 13.819683074951172, 15.478880167007446, 17.14009118080139, 18.700833559036255, 20.401718139648438, 22.100791454315186, 23.800947666168213, 25.501822233200073, 27.200392723083496, 28.901028633117676, 30.600394248962402, 32.30020260810852, 34.00024676322937, 35.70176362991333, 37.400548696517944, 39.10201168060303, 40.80012655258179, 42.50049138069153, 44.20102143287659, 45.90201425552368, 47.60056281089783, 49.30034065246582] 
	q6 = [0.39999999999999997, 0.6666666666666667, 0.7391304347826088, 0.8235294117647058, 0.8461538461538461, 0.8461538461538461, 0.8461538461538461, 0.8461538461538461, 0.8461538461538461, 0.8679245283018868, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906] 
	t7 = [0, 1.4596595764160156, 2.927792549133301, 4.23039174079895, 5.730862617492676, 7.1351237297058105, 8.647031784057617, 10.167954683303833, 11.352604627609253, 12.945295810699463, 14.217852592468262, 15.444825649261475, 17.10300326347351, 18.202100038528442, 19.601412057876587, 21.00075936317444, 22.4015851020813, 23.802058696746826, 25.200913667678833, 26.600324153900146, 28.000121116638184, 29.401500940322876, 30.800566911697388, 32.20008587837219, 33.60205936431885, 35.00100302696228, 36.40033793449402, 37.80199337005615, 39.20096564292908, 40.60201406478882, 42.000150203704834, 43.40018820762634, 44.8009979724884, 46.20188498497009, 47.60093426704407, 49.00194239616394] 
	q7 = [0.39999999999999997, 0.65, 0.7111111111111111, 0.7916666666666666, 0.8163265306122449, 0.8400000000000001, 0.8400000000000001, 0.8461538461538461, 0.8461538461538461, 0.8461538461538461, 0.8461538461538461, 0.8679245283018868, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906] 
	t8 = [0, 1.2017884254455566, 2.5093231201171875, 3.651594400405884, 4.806089639663696, 6.124413728713989, 7.254056692123413, 8.407309770584106, 9.913915872573853, 10.993629932403564, 12.172123193740845, 13.42047119140625, 14.639875173568726, 15.865515232086182, 17.108323335647583, 18.00024676322937, 19.200916528701782, 20.400768756866455, 21.60045862197876, 22.80148458480835, 24.00142812728882, 25.20045757293701, 26.400627851486206, 27.600317001342773, 28.801777362823486, 30.002086877822876, 31.2000994682312, 32.401071548461914, 33.60002303123474, 34.80094504356384, 36.001561641693115, 37.20142149925232, 38.40134787559509, 39.60097908973694, 40.80061721801758, 42.00172400474548, 43.20059132575989, 44.40003705024719, 45.601993799209595, 46.801286458969116, 48.0003707408905, 49.20103454589844] 
	q8 = [0.39999999999999997, 0.5789473684210527, 0.7272727272727274, 0.7659574468085107, 0.7916666666666666, 0.8163265306122449, 0.8163265306122449, 0.8235294117647058, 0.8235294117647058, 0.8461538461538461, 0.8461538461538461, 0.8461538461538461, 0.8461538461538461, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906] 
	t9 = [0, 1.186737298965454, 2.203334093093872, 3.4158570766448975, 4.519383192062378, 5.634154319763184, 6.729808330535889, 7.851110458374023, 8.827726602554321, 9.966395616531372, 11.378159761428833, 12.238505840301514, 13.455636024475098, 14.668580770492554, 15.473875522613525, 16.680780172348022, 17.60088539123535, 18.701591968536377, 19.801489114761353, 20.900184869766235, 22.000631093978882, 23.100927352905273, 24.201025009155273, 25.30065107345581, 26.40105891227722, 27.500097513198853, 28.600011110305786, 29.7009539604187, 30.80097770690918, 31.901602745056152, 33.000861406326294, 34.10047721862793, 35.20106220245361, 36.301323652267456, 37.40068817138672, 38.50168251991272, 39.60171031951904, 40.70082879066467, 41.80085015296936, 42.900574922561646, 44.00003957748413, 45.101245641708374, 46.20110607147217, 47.30042505264282, 48.40101885795593, 49.50180149078369] 
	q9 = [0.39999999999999997, 0.5789473684210527, 0.7272727272727274, 0.7391304347826088, 0.7659574468085107, 0.7659574468085107, 0.8163265306122449, 0.8163265306122449, 0.8235294117647058, 0.8235294117647058, 0.8461538461538461, 0.8461538461538461, 0.8461538461538461, 0.8679245283018868, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906] 
	t10 = [0, 1.0422289371490479, 2.0545995235443115, 3.082270622253418, 4.113217830657959, 5.234966993331909, 6.031819105148315, 7.1831889152526855, 8.004319429397583, 9.14261531829834, 10.096538543701172, 11.21221661567688, 12.18901252746582, 13.015729665756226, 14.214338541030884, 15.066236972808838, 16.205472230911255, 17.04154133796692, 18.000993251800537, 19.000235557556152, 20.002079963684082, 21.001527547836304, 22.000244140625, 23.001240968704224, 24.001994132995605, 25.00014615058899, 26.00155520439148, 27.000599145889282, 28.000842094421387, 29.000457763671875, 30.00163960456848, 31.00046420097351, 32.00044393539429, 33.00064444541931, 34.000948905944824, 35.001405000686646, 36.000312089920044, 37.001691818237305, 38.00114369392395, 39.001558780670166, 40.000242948532104, 41.001442432403564, 42.0019211769104, 43.002076864242554, 44.000797271728516, 45.00154137611389, 46.00175595283508, 47.00085234642029, 48.00188326835632, 49.00096130371094, 50.00067734718323] 
	q10 = [0.39999999999999997, 0.5405405405405405, 0.744186046511628, 0.7391304347826088, 0.7916666666666666, 0.8163265306122449, 0.8163265306122449, 0.8163265306122449, 0.8400000000000001, 0.8400000000000001, 0.8461538461538461, 0.8461538461538461, 0.8461538461538461, 0.8461538461538461, 0.8461538461538461, 0.8679245283018868, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906, 0.9056603773584906] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 10.312074899673462, 20.000189304351807, 30.000185012817383, 40.001893043518066, 50.000247955322266] 
	q1 = [0.39999999999999997, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492] 
	t2 = [0, 5.0379486083984375, 10.19209909439087, 15.000298500061035, 20.00081467628479, 25.001317262649536, 30.00164246559143, 35.00035309791565, 40.00125789642334, 45.00027084350586, 50.000123262405396] 
	q2 = [0.39999999999999997, 0.7777777777777779, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492] 
	t3 = [0, 3.432643175125122, 6.715429782867432, 10.232470512390137, 13.492409706115723, 16.65052843093872, 19.98038673400879, 23.31145143508911, 26.64059329032898, 29.971335649490356, 33.301687479019165, 36.631564140319824, 39.961933612823486, 43.290645599365234, 46.621158838272095, 49.95029592514038] 
	q3 = [0.39999999999999997, 0.7755102040816326, 0.7719298245614036, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492] 
	t4 = [0, 2.6531474590301514, 5.065294027328491, 7.53371000289917, 10.219407796859741, 12.684108257293701, 15.000852346420288, 17.500640869140625, 20.00199031829834, 22.500741958618164, 25.000629425048828, 27.50091290473938, 30.000867128372192, 32.5013222694397, 35.00006413459778, 37.50092577934265, 40.00185704231262, 42.501240491867065, 45.00148963928223, 47.502050161361694, 50.00110054016113] 
	q4 = [0.39999999999999997, 0.7234042553191489, 0.7169811320754716, 0.7719298245614036, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492] 
	t5 = [0, 2.0681467056274414, 4.134716749191284, 6.039471626281738, 8.26473879814148, 10.147222995758057, 12.087022304534912, 14.058668375015259, 16.001611709594727, 18.000410079956055, 20.00122094154358, 22.000128507614136, 24.00054097175598, 26.001020908355713, 28.001078844070435, 30.00013256072998, 32.001850605010986, 34.00120401382446, 36.00144124031067, 38.00012159347534, 40.001734495162964, 42.00004959106445, 44.00179433822632, 46.00143003463745, 48.000494956970215, 50.000025272369385] 
	q5 = [0.39999999999999997, 0.6666666666666666, 0.7547169811320754, 0.75, 0.7857142857142856, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492] 
	t6 = [0, 1.78776216506958, 3.423877477645874, 5.23775315284729, 7.150282382965088, 8.778302192687988, 10.487633228302002, 12.07313585281372, 13.666507005691528, 15.300833225250244, 17.00078248977661, 18.700584411621094, 20.400267601013184, 22.100110292434692, 23.801133394241333, 25.50145196914673, 1028.054433107376] 
	q6 = [0.39999999999999997, 0.6818181818181818, 0.76, 0.7636363636363636, 0.7636363636363636, 0.7719298245614036, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492] 
	t7 = [0, 1.4329466819763184, 2.877927780151367, 4.2360615730285645, 5.727700710296631, 7.261279582977295, 8.46252727508545, 9.845360040664673, 11.388906002044678, 12.711138248443604, 14.329393148422241, 15.401966571807861, 16.800289392471313, 18.201136350631714, 19.600903749465942, 21.00020480155945, 22.40203070640564, 23.800354480743408, 25.200918436050415, 26.600946187973022, 28.001142978668213, 29.4019935131073, 30.800812005996704, 32.200830936431885, 33.60161519050598, 35.00084161758423, 36.40064597129822, 37.8003363609314, 39.20042371749878, 40.60226321220398, 42.000481605529785, 43.40209364891052, 44.80087232589722, 46.20044541358948, 47.60086941719055, 49.001511573791504] 
	q7 = [0.39999999999999997, 0.6, 0.6521739130434783, 0.76, 0.7450980392156863, 0.7843137254901961, 0.8, 0.8, 0.8, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492] 
	t8 = [0, 1.3129851818084717, 2.5677683353424072, 3.75063157081604, 4.856112718582153, 6.087971925735474, 7.562877178192139, 8.501811504364014, 9.807243585586548, 10.919890880584717, 12.163658380508423, 13.405515193939209, 14.56020212173462, 15.984567403793335, 17.020774602890015, 18.001214265823364, 19.201324701309204, 20.400300979614258, 21.601231813430786, 22.800805807113647, 24.00141954421997, 25.201891899108887, 26.40191102027893, 27.60068678855896, 28.800033807754517, 30.00045371055603, 31.20060443878174, 32.401739835739136, 33.601109743118286, 34.80094265937805, 36.001092195510864, 37.20049810409546, 38.40007543563843, 39.60149097442627, 40.80205225944519, 42.00049948692322, 43.202889919281006, 44.40076422691345, 45.60164761543274, 46.80011034011841, 48.000603914260864, 49.200000286102295] 
	q8 = [0.39999999999999997, 0.6, 0.6521739130434783, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7307692307692306, 0.7692307692307693, 0.7636363636363636, 0.7636363636363636, 0.7636363636363636, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492] 
	t9 = [0, 1.1269559860229492, 2.2915215492248535, 3.3199875354766846, 4.485226631164551, 5.534660577774048, 6.790222644805908, 7.729616403579712, 8.992323398590088, 10.058528423309326, 11.076828241348267, 12.37883448600769, 13.584960699081421, 14.394184589385986, 15.624340772628784, 16.501193284988403, 17.601855278015137, 18.70001983642578, 19.803242444992065, 20.90006399154663, 22.001495361328125, 23.10167384147644, 24.20053243637085, 25.30068850517273, 26.401144981384277, 27.500718355178833, 28.60046625137329, 29.700648307800293, 30.801021099090576, 31.90014910697937, 33.001930475234985, 34.100605487823486, 35.2011923789978, 36.300458669662476, 37.40071702003479, 38.50191259384155, 39.60023760795593, 40.70003843307495, 41.80049228668213, 42.90173602104187, 44.00012993812561, 45.10136032104492, 46.20081305503845, 47.30067443847656, 48.40192484855652, 49.50049018859863] 
	q9 = [0.39999999999999997, 0.6, 0.6521739130434783, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7307692307692306, 0.7692307692307693, 0.7636363636363636, 0.7636363636363636, 0.7636363636363636, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492] 
	t10 = [0, 1.094799518585205, 2.076753616333008, 3.0699574947357178, 4.124170303344727, 5.00595235824585, 6.0268120765686035, 7.309505939483643, 8.118312120437622, 9.334235191345215, 10.086370944976807, 11.354444026947021, 12.3177809715271, 13.139668941497803, 14.052525281906128, 15.392062187194824, 16.001317024230957, 17.0011727809906, 18.00134587287903, 19.000356435775757, 20.000438690185547, 21.00126075744629, 22.00143575668335, 23.00060796737671, 24.001925230026245, 25.000006437301636, 26.0009446144104, 27.00138282775879, 28.001688480377197, 29.001607418060303, 30.002020597457886, 31.001783847808838, 32.00183463096619, 33.00189423561096, 34.001492500305176, 35.0001015663147, 36.00097370147705, 37.00071430206299, 38.001981019973755, 39.000173807144165, 40.0007688999176, 41.00129747390747, 42.0012469291687, 43.00166392326355, 44.00128746032715, 45.00119113922119, 46.00118374824524, 47.00141382217407, 48.00077199935913, 49.00091600418091, 50.0020215511322] 
	q10 = [0.39999999999999997, 0.6, 0.6666666666666666, 0.7083333333333333, 0.693877551020408, 0.693877551020408, 0.6923076923076923, 0.6923076923076923, 0.7037037037037038, 0.7407407407407408, 0.7857142857142856, 0.7857142857142856, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492, 0.8070175438596492] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 10.269145488739014, 20.00137186050415, 30.000910997390747, 40.00266075134277, 50.00177621841431] 
	q1 = [0.30303030303030304, 0.8, 0.8, 0.8, 0.8, 0.8] 
	t2 = [0, 5.092446565628052, 10.317523717880249, 15.237910747528076, 20.001931190490723, 25.000745058059692, 30.001857042312622, 35.00045299530029, 40.00152254104614, 45.0008442401886, 50.000284910202026] 
	q2 = [0.30303030303030304, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8] 
	t3 = [0, 3.4855690002441406, 6.81181788444519, 10.095379829406738, 13.589863777160645, 16.651232957839966, 19.98193097114563, 23.310386896133423, 26.640029907226562, 29.97044324874878, 33.30084753036499, 36.63039565086365, 39.960421323776245, 43.29079270362854, 46.6211884021759, 49.950453758239746] 
	q3 = [0.30303030303030304, 0.6666666666666666, 0.7755102040816326, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8] 
	t4 = [0, 2.5808281898498535, 5.145214557647705, 7.559507131576538, 10.123995780944824, 12.696956634521484, 15.293839454650879, 17.500336170196533, 20.002568006515503, 22.501282691955566, 25.001639127731323, 27.501828908920288, 30.000961780548096, 32.50212097167969, 35.00003981590271, 37.5011842250824, 40.001012086868286, 42.501424074172974, 45.000035762786865, 47.50184726715088, 50.00204920768738] 
	q4 = [0.30303030303030304, 0.6511627906976744, 0.7916666666666666, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8] 
	t5 = [0, 2.0855929851531982, 4.058976888656616, 6.014848232269287, 8.41013240814209, 10.004129648208618, 12.349090576171875, 14.356125116348267, 16.002429723739624, 18.000134706497192, 20.00169038772583, 22.00082278251648, 24.001540660858154, 26.000752925872803, 28.001385927200317, 30.00184154510498, 32.00155448913574, 34.0005156993866, 36.00146746635437, 38.00179314613342, 40.00274133682251, 42.000417709350586, 44.00105881690979, 46.000232458114624, 48.00152897834778, 50.000834465026855] 
	q5 = [0.30303030303030304, 0.5714285714285713, 0.7500000000000001, 0.7755102040816326, 0.7755102040816326, 0.7755102040816326, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8] 
	t6 = [0, 1.762740135192871, 3.5377039909362793, 5.2256386280059814, 7.161351919174194, 8.51001787185669, 10.467618703842163, 11.926623821258545, 13.680457592010498, 15.349893569946289, 17.000427722930908, 18.701268434524536, 20.400949716567993, 22.1013286113739, 23.80077624320984, 25.500732898712158, 27.200888633728027, 28.90074634552002, 30.60023832321167, 32.30058288574219, 34.00076627731323, 35.70060920715332, 37.40027928352356, 39.10071539878845, 40.801143169403076, 42.50153994560242, 44.20053672790527, 45.90251970291138, 47.601969957351685, 49.30030822753906] 
	q6 = [0.30303030303030304, 0.5365853658536586, 0.6956521739130435, 0.7755102040816326, 0.7755102040816326, 0.7755102040816326, 0.7755102040816326, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8] 
	t7 = [0, 1.5281188488006592, 2.884916305541992, 4.274824380874634, 5.826769590377808, 7.111544370651245, 8.662806034088135, 10.124940872192383, 11.345507144927979, 12.946555137634277, 14.206793069839478, 15.401091575622559, 16.800691843032837, 18.200865507125854, 19.600139379501343, 21.00010347366333, 22.40139102935791, 23.800081253051758, 25.200612545013428, 26.60187578201294, 28.001035928726196, 29.400440454483032, 30.801114559173584, 32.2019100189209, 33.60022020339966, 35.00027585029602, 36.40194010734558, 37.80170464515686, 39.20115876197815, 40.6007764339447, 42.001282691955566, 43.40189266204834, 44.80027103424072, 46.20213294029236, 47.60108923912048, 49.00118923187256] 
	q7 = [0.30303030303030304, 0.55, 0.6818181818181818, 0.7659574468085107, 0.7659574468085107, 0.7916666666666666, 0.7916666666666666, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8] 
	t8 = [0, 1.3398041725158691, 2.537283420562744, 3.7322824001312256, 4.852254390716553, 6.109037637710571, 7.398058652877808, 8.454770803451538, 9.784266233444214, 11.072913408279419, 12.410075664520264, 13.294018745422363, 14.566730976104736, 15.931215047836304, 16.801912784576416, 18.00145149230957, 19.201057195663452, 20.401118278503418, 21.600219249725342, 22.800177335739136, 24.001237154006958, 25.202089309692383, 26.401795148849487, 27.600414752960205, 28.80093288421631, 30.00125479698181, 31.20187497138977, 32.401625633239746, 33.60188269615173, 34.80045032501221, 36.0011305809021, 37.20167851448059, 38.401111125946045, 39.60064959526062, 40.800074100494385, 42.00080895423889, 43.20194053649902, 44.40137553215027, 45.60128355026245, 46.80041170120239, 48.00200271606445, 49.201902866363525] 
	q8 = [0.30303030303030304, 0.5128205128205129, 0.6511627906976744, 0.7391304347826088, 0.7391304347826088, 0.7659574468085107, 0.7659574468085107, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8] 
	t9 = [0, 1.2012739181518555, 2.2607004642486572, 3.3310842514038086, 4.495560884475708, 5.647918224334717, 6.83689022064209, 7.709635019302368, 8.887338638305664, 9.920870542526245, 11.065221309661865, 12.301941394805908, 13.551956415176392, 14.416513681411743, 15.402029275894165, 16.501070737838745, 17.600769519805908, 18.701420545578003, 19.80044174194336, 20.900799989700317, 22.000593423843384, 23.100077390670776, 24.201221704483032, 25.301201343536377, 26.40195322036743, 27.50160241127014, 28.60130262374878, 29.70096731185913, 30.80025362968445, 31.90094757080078, 33.00113320350647, 34.101572036743164, 35.20109510421753, 36.301668882369995, 37.40126848220825, 38.50117373466492, 39.60047101974487, 40.701706886291504, 41.801368713378906, 42.90166115760803, 44.00000524520874, 45.10226130485535, 46.20113968849182, 47.30107593536377, 48.400230407714844, 49.50064730644226] 
	q9 = [0.30303030303030304, 0.47368421052631576, 0.6046511627906977, 0.6956521739130435, 0.6956521739130435, 0.7659574468085107, 0.7659574468085107, 0.7916666666666666, 0.7916666666666666, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8] 
	t10 = [0, 1.0536129474639893, 2.11438250541687, 3.015953302383423, 4.2100372314453125, 5.116594314575195, 6.3460752964019775, 7.07118034362793, 8.225223064422607, 9.059080123901367, 10.23698616027832, 11.109402418136597, 12.318187475204468, 13.382107257843018, 14.247299194335938, 15.171314001083374, 16.00083589553833, 17.00057601928711, 18.00155758857727, 19.000691890716553, 20.001808404922485, 21.001514434814453, 22.001920223236084, 23.00053381919861, 24.000230073928833, 25.001813411712646, 26.000057458877563, 27.001723527908325, 28.000972270965576, 29.00057888031006, 30.00193166732788, 31.001967668533325, 32.00016188621521, 33.000550270080566, 34.001895904541016, 35.000547647476196, 36.001511335372925, 37.00054359436035, 38.00231146812439, 39.00058841705322, 40.0004243850708, 41.00032567977905, 42.00178599357605, 43.000786542892456, 44.00062680244446, 45.00077533721924, 46.00158953666687, 47.00063514709473, 48.00045728683472, 49.0004608631134, 50.00150108337402] 
	q10 = [0.30303030303030304, 0.43243243243243246, 0.5714285714285713, 0.6818181818181818, 0.6818181818181818, 0.7659574468085107, 0.7659574468085107, 0.7916666666666666, 0.7916666666666666, 0.7916666666666666, 0.7916666666666666, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 10.221467018127441, 20.00096893310547, 30.00006341934204, 40.001888036727905, 50.001389026641846] 
	q1 = [0.39999999999999997, 0.8181818181818182, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107] 
	t2 = [0, 5.008875608444214, 10.253434181213379, 15.388815879821777, 20.001872301101685, 25.000802278518677, 30.001941204071045, 35.00023865699768, 40.00116682052612, 45.00103545188904, 50.00161004066467] 
	q2 = [0.39999999999999997, 0.7906976744186046, 0.7906976744186046, 0.7391304347826088, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107] 
	t3 = [0, 3.39780592918396, 6.7425923347473145, 10.332369565963745, 13.331730842590332, 16.659629821777344, 19.981464385986328, 23.310558557510376, 26.641584873199463, 29.97102975845337, 33.30157446861267, 36.630773067474365, 39.961488485336304, 43.291590213775635, 46.62224507331848, 49.951624393463135] 
	q3 = [0.39999999999999997, 0.7499999999999999, 0.8372093023255814, 0.8372093023255814, 0.7391304347826088, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107] 
	t4 = [0, 2.6165738105773926, 5.119232654571533, 7.542290925979614, 10.359544038772583, 12.819106101989746, 15.152028560638428, 17.530495643615723, 20.00196886062622, 22.50030827522278, 25.0015926361084, 27.500237464904785, 30.00195026397705, 32.50119757652283, 35.00161027908325, 37.50158071517944, 40.000290393829346, 42.50210762023926, 45.00022482872009, 47.50103235244751, 50.00219392776489] 
	q4 = [0.39999999999999997, 0.717948717948718, 0.8095238095238095, 0.7555555555555555, 0.711111111111111, 0.7391304347826088, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107] 
	t5 = [0, 2.039421319961548, 4.101512432098389, 6.051023483276367, 8.33684229850769, 10.150225400924683, 12.165772199630737, 14.15445876121521, 16.227983713150024, 18.000469207763672, 20.000392198562622, 22.00074791908264, 24.00029969215393, 26.00205421447754, 28.000843048095703, 30.00113010406494, 32.000083208084106, 34.000914335250854, 36.000295639038086, 38.00172233581543, 40.001856088638306, 42.0009765625, 44.00194048881531, 46.000821590423584, 48.001535415649414, 50.00002145767212] 
	q5 = [0.39999999999999997, 0.6486486486486486, 0.7317073170731707, 0.8181818181818182, 0.7727272727272727, 0.7391304347826088, 0.7391304347826088, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107] 
	t6 = [0, 1.773573875427246, 3.5456178188323975, 5.217348575592041, 7.1397154331207275, 8.515506744384766, 10.442442893981934, 12.262138843536377, 13.928962469100952, 15.570094585418701, 17.224018096923828, 18.70144510269165, 20.401330947875977, 22.10117793083191, 23.800236463546753, 25.50210928916931, 27.20050311088562, 28.900424242019653, 30.600402355194092, 32.3003511428833, 34.00161147117615, 35.70160984992981, 37.400052070617676, 39.10003733634949, 40.80198526382446, 42.50071835517883, 44.20013403892517, 45.90150165557861, 47.60133934020996, 49.30012083053589] 
	q6 = [0.39999999999999997, 0.6111111111111112, 0.7499999999999999, 0.7906976744186046, 0.7441860465116279, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107] 
	t7 = [0, 1.5001778602600098, 2.8438849449157715, 4.317057371139526, 5.865539073944092, 7.03904128074646, 8.538156032562256, 10.143162250518799, 11.341567516326904, 12.943393230438232, 14.149440288543701, 15.768304109573364, 17.009265661239624, 18.201749801635742, 19.602209091186523, 21.00214910507202, 22.400795698165894, 23.80132246017456, 25.200935125350952, 26.60158133506775, 28.000271797180176, 29.401203870773315, 30.800190687179565, 32.20091676712036, 33.602195262908936, 35.00047755241394, 36.4012131690979, 37.80146408081055, 39.200159549713135, 40.60076642036438, 42.00030589103699, 43.402085304260254, 44.800041913986206, 46.20091247558594, 47.60019540786743, 49.00187706947327] 
	q7 = [0.39999999999999997, 0.5714285714285714, 0.717948717948718, 0.7317073170731707, 0.7317073170731707, 0.711111111111111, 0.711111111111111, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107] 
	t8 = [0, 1.2443439960479736, 2.475437641143799, 3.7375805377960205, 4.882893800735474, 6.128599405288696, 7.413721323013306, 8.424153089523315, 9.95197606086731, 10.813767194747925, 12.134058952331543, 13.437448978424072, 14.7455894947052, 15.96155333518982, 16.83120632171631, 18.03169345855713, 19.200713634490967, 20.4008150100708, 21.60166358947754, 22.80001211166382, 24.002049207687378, 25.2001850605011, 26.402103424072266, 27.601652145385742, 28.802082300186157, 30.00171709060669, 31.200062036514282, 32.40127968788147, 33.60053086280823, 34.80172109603882, 36.00188732147217, 37.200100898742676, 38.40051078796387, 39.600157499313354, 40.80077314376831, 42.00075626373291, 43.201096534729004, 44.40061974525452, 45.60017657279968, 46.800443172454834, 48.0022087097168, 49.200724840164185] 
	q8 = [0.39999999999999997, 0.5294117647058824, 0.6842105263157896, 0.7499999999999999, 0.7499999999999999, 0.7272727272727273, 0.7272727272727273, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107] 
	t9 = [0, 1.2208020687103271, 2.2854413986206055, 3.3624978065490723, 4.555538177490234, 5.783010721206665, 6.684861421585083, 7.903846263885498, 8.902173280715942, 10.087557315826416, 11.127801418304443, 12.319262504577637, 13.529436111450195, 14.361067771911621, 15.577087879180908, 16.82802724838257, 17.66559886932373, 18.70159673690796, 19.801458835601807, 20.901986122131348, 22.00046730041504, 23.101091623306274, 24.200390815734863, 25.301722526550293, 26.401766777038574, 27.50124454498291, 28.600146293640137, 29.70221734046936, 30.8015296459198, 31.900521755218506, 33.00029110908508, 34.101922273635864, 35.20040845870972, 36.30207562446594, 37.400301456451416, 38.50197625160217, 39.60049772262573, 40.70104146003723, 41.80008101463318, 42.901833295822144, 44.000505208969116, 45.10104775428772, 46.2003219127655, 47.30000877380371, 48.40177583694458, 49.50265574455261] 
	q9 = [0.39999999999999997, 0.5294117647058824, 0.6842105263157896, 0.7499999999999999, 0.7317073170731707, 0.7317073170731707, 0.6818181818181818, 0.6818181818181818, 0.711111111111111, 0.711111111111111, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107] 
	t10 = [0, 1.1023108959197998, 2.0280063152313232, 3.1213955879211426, 4.050265550613403, 5.285521745681763, 6.02007794380188, 7.267224073410034, 8.112562656402588, 9.32938551902771, 10.096116781234741, 11.249376773834229, 12.327600955963135, 13.171056985855103, 14.387783527374268, 15.246904611587524, 16.085567712783813, 17.30657696723938, 18.154964208602905, 19.000837564468384, 20.002076387405396, 21.000641107559204, 22.00049901008606, 23.000715970993042, 24.00032615661621, 25.001882076263428, 26.002033948898315, 27.00084900856018, 28.000396490097046, 29.001677751541138, 30.00214982032776, 31.000662326812744, 32.00108766555786, 33.00162100791931, 34.002273082733154, 35.00213646888733, 36.00086307525635, 37.000441789627075, 38.001129388809204, 39.00153398513794, 40.00068163871765, 41.00054860115051, 42.00114989280701, 43.001362800598145, 44.00140309333801, 45.00173377990723, 46.00150918960571, 47.00189208984375, 48.001144886016846, 49.0001699924469, 50.00226616859436] 
	q10 = [0.39999999999999997, 0.4848484848484849, 0.6111111111111112, 0.717948717948718, 0.7317073170731707, 0.7317073170731707, 0.711111111111111, 0.711111111111111, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107, 0.7659574468085107] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 10.039541482925415, 20.000888347625732, 30.00086212158203, 40.00144553184509, 50.001811504364014] 
	q1 = [0.43243243243243246, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571] 
	t2 = [0, 5.09612774848938, 10.035625696182251, 15.001948595046997, 20.0007381439209, 25.000670433044434, 30.00073218345642, 35.00100302696228, 40.001256227493286, 45.00052523612976, 50.00055551528931] 
	q2 = [0.43243243243243246, 0.8363636363636363, 0.8363636363636363, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571] 
	t3 = [0, 3.3740406036376953, 6.81798791885376, 10.169604063034058, 13.567638635635376, 16.650800228118896, 19.980162143707275, 23.311729431152344, 26.64170479774475, 29.971741199493408, 33.300310134887695, 36.630980491638184, 39.9612603187561, 43.29023003578186, 46.62138390541077, 49.951292753219604] 
	q3 = [0.43243243243243246, 0.7499999999999999, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571] 
	t4 = [0, 2.562777280807495, 5.054558277130127, 7.554939031600952, 10.143970012664795, 12.653823852539062, 15.167492389678955, 17.50098419189453, 20.000043630599976, 22.50141215324402, 25.001474380493164, 27.501051425933838, 30.001171588897705, 32.5000479221344, 35.000064849853516, 37.501585483551025, 40.00170063972473, 42.50021696090698, 45.00144147872925, 47.501301765441895, 50.00060057640076] 
	q4 = [0.43243243243243246, 0.7234042553191491, 0.7924528301886793, 0.8363636363636363, 0.8363636363636363, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571] 
	t5 = [0, 2.0820090770721436, 4.069570302963257, 6.070253849029541, 8.359878540039062, 10.345727682113647, 12.00346040725708, 14.00084638595581, 16.000771045684814, 18.000447750091553, 20.00162625312805, 22.001444816589355, 24.000741481781006, 26.000340223312378, 28.0011146068573, 30.002225637435913, 32.0007221698761, 34.00015425682068, 36.000399589538574, 38.000194787979126, 40.00121450424194, 42.00004053115845, 44.00101971626282, 46.00042271614075, 48.00169062614441, 50.00154948234558] 
	q5 = [0.43243243243243246, 0.7234042553191491, 0.7547169811320756, 0.8, 0.8, 0.8214285714285714, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571] 
	t6 = [0, 1.7906489372253418, 3.437052011489868, 5.168476581573486, 7.0976386070251465, 8.500061988830566, 10.31562352180481, 11.910684823989868, 13.735068798065186, 15.300904512405396, 17.00142741203308, 18.70150899887085, 20.4008731842041, 22.100175619125366, 23.80075240135193, 25.500048875808716, 27.201530933380127, 28.901676416397095, 30.601390838623047, 32.30125188827515, 34.000006675720215, 35.70009922981262, 37.40152025222778, 39.10173559188843, 40.80081748962402, 42.50151586532593, 44.20057678222656, 45.900917768478394, 47.601945638656616, 49.30034399032593] 
	q6 = [0.43243243243243246, 0.6666666666666665, 0.7346938775510204, 0.8000000000000002, 0.8000000000000002, 0.8363636363636363, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571] 
	t7 = [0, 1.5222258567810059, 2.8688931465148926, 4.232895135879517, 5.640196323394775, 7.159646272659302, 8.413746356964111, 9.969868898391724, 11.204716444015503, 12.826465845108032, 14.001319169998169, 15.401230573654175, 16.801244020462036, 18.201149225234985, 19.600249528884888, 21.001726627349854, 22.401728868484497, 23.800706386566162, 25.200488805770874, 26.601314783096313, 28.000075578689575, 29.401395082473755, 30.801395893096924, 32.201783180236816, 33.600131034851074, 35.000234842300415, 36.40093374252319, 37.800480127334595, 39.20000743865967, 40.60223603248596, 42.00169014930725, 43.401145458221436, 44.80003762245178, 46.20086669921875, 47.60015153884888, 49.00114464759827] 
	q7 = [0.43243243243243246, 0.6363636363636364, 0.7346938775510204, 0.8076923076923077, 0.7924528301886793, 0.7924528301886793, 0.8363636363636363, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571] 
	t8 = [0, 1.2307212352752686, 2.4150145053863525, 3.632408618927002, 4.816959381103516, 6.019573926925659, 7.222202777862549, 8.654856443405151, 9.968346118927002, 10.810281991958618, 12.349586248397827, 13.564675092697144, 14.400545835494995, 15.601118564605713, 16.80249261856079, 18.001328706741333, 19.20068645477295, 20.400248527526855, 21.600440740585327, 22.80088520050049, 24.00140142440796, 25.20007562637329, 26.401410341262817, 27.601030588150024, 28.801310539245605, 30.000020265579224, 31.200640201568604, 32.40107798576355, 33.60069227218628, 34.80143475532532, 36.00032615661621, 37.200847148895264, 38.401280641555786, 39.60157346725464, 40.80039620399475, 42.00148272514343, 43.20032095909119, 44.400721311569214, 45.60053300857544, 46.801708936691284, 48.00100016593933, 49.20148706436157] 
	q8 = [0.43243243243243246, 0.5714285714285713, 0.7083333333333334, 0.7692307692307692, 0.7692307692307692, 0.8148148148148148, 0.8148148148148148, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571] 
	t9 = [0, 1.210775375366211, 2.2528560161590576, 3.436702251434326, 4.560155391693115, 5.5641021728515625, 6.694589376449585, 7.8385329246521, 9.000968933105469, 9.952520608901978, 11.118455171585083, 12.320722341537476, 13.498371124267578, 14.301488161087036, 15.4013831615448, 16.50056290626526, 17.60071897506714, 18.70008635520935, 19.801131010055542, 20.901400566101074, 22.00123381614685, 23.101266860961914, 24.200802326202393, 25.300392866134644, 26.401447772979736, 27.501702785491943, 28.60042691230774, 29.70083713531494, 30.80138063430786, 31.901270151138306, 33.000351667404175, 34.10039162635803, 35.200068950653076, 36.30034422874451, 37.40151309967041, 38.5012047290802, 39.60127401351929, 40.70017337799072, 41.800509452819824, 42.901469469070435, 44.000412464141846, 45.10067319869995, 46.201075077056885, 47.300039529800415, 48.40093541145325, 49.501612424850464] 
	q9 = [0.43243243243243246, 0.5714285714285713, 0.7083333333333334, 0.7307692307692308, 0.7307692307692308, 0.7547169811320756, 0.7547169811320756, 0.8, 0.8, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571] 
	t10 = [0, 1.0184850692749023, 2.002469301223755, 3.1359357833862305, 4.0099616050720215, 5.099766969680786, 6.413303852081299, 7.0202741622924805, 8.315108060836792, 9.059866666793823, 10.36127758026123, 11.070117473602295, 12.238746404647827, 13.221184253692627, 14.078861236572266, 15.000395774841309, 16.001525163650513, 17.00143837928772, 18.001195907592773, 19.000595808029175, 20.00000309944153, 21.000413179397583, 22.000673294067383, 23.000701427459717, 24.000745058059692, 25.00119161605835, 26.000442266464233, 27.001432418823242, 28.00022053718567, 29.000407457351685, 30.001664638519287, 31.002991914749146, 32.0008704662323, 33.000752687454224, 34.00147771835327, 35.00081944465637, 36.00122904777527, 37.00062537193298, 38.00183606147766, 39.000874042510986, 40.00005912780762, 41.00034165382385, 42.00041675567627, 43.00002646446228, 44.00167918205261, 45.00136685371399, 46.00122404098511, 47.0021390914917, 48.001007318496704, 49.00031232833862, 50.00195360183716] 
	q10 = [0.43243243243243246, 0.5, 0.6521739130434783, 0.6938775510204083, 0.6938775510204083, 0.7058823529411765, 0.7058823529411765, 0.7307692307692308, 0.7307692307692308, 0.8, 0.8, 0.8, 0.8, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571, 0.8571428571428571] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	t1 = [0, 10.174942016601562, 20.001538038253784, 30.001537799835205, 40.00084638595581, 50.001829624176025] 
	q1 = [0.39999999999999997, 0.7391304347826088, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203] 
	t2 = [0, 5.129920244216919, 10.29757308959961, 15.395453929901123, 20.001036405563354, 25.000709056854248, 30.00126004219055, 35.001423835754395, 40.00099420547485, 45.00133538246155, 50.0015230178833] 
	q2 = [0.39999999999999997, 0.711111111111111, 0.6956521739130435, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203] 
	t3 = [0, 3.45499587059021, 6.716268062591553, 10.144350290298462, 13.383447408676147, 16.817688703536987, 19.981839895248413, 23.310478448867798, 26.64179491996765, 29.970078945159912, 33.301976919174194, 36.631189584732056, 39.96017241477966, 43.291592836380005, 46.62095832824707, 49.951881647109985] 
	q3 = [0.39999999999999997, 0.5853658536585366, 0.7391304347826088, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203] 
	t4 = [0, 2.5694522857666016, 5.051169157028198, 7.5004377365112305, 10.118977069854736, 13.06711721420288, 15.056951522827148, 17.778647661209106, 20.000763177871704, 22.501646041870117, 25.001680612564087, 27.50116467475891, 30.00056219100952, 32.50080847740173, 35.000328063964844, 37.50192975997925, 40.001256465911865, 42.50015163421631, 45.00016188621521, 47.500410079956055, 50.001704692840576] 
	q4 = [0.39999999999999997, 0.5641025641025642, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7234042553191491, 0.7500000000000001, 0.7500000000000001, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203] 
	t5 = [0, 2.0307183265686035, 4.1129372119903564, 6.031928539276123, 8.440505266189575, 10.064094305038452, 12.261969089508057, 14.082266330718994, 16.465843677520752, 18.45950412750244, 20.001168966293335, 22.000218629837036, 24.001107931137085, 26.00185203552246, 28.000970363616943, 30.001840591430664, 32.00133156776428, 34.000184774398804, 36.001325368881226, 38.001338720321655, 40.00089621543884, 42.00081944465637, 44.00169134140015, 46.00055646896362, 48.00088167190552, 50.00007939338684] 
	q5 = [0.39999999999999997, 0.5128205128205129, 0.6190476190476191, 0.7391304347826088, 0.7234042553191491, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203] 
	t6 = [0, 1.7768001556396484, 3.5253913402557373, 5.311204433441162, 7.404097557067871, 8.555452585220337, 10.438608646392822, 12.245953798294067, 13.902546882629395, 15.677006959915161, 17.07716393470764, 19.02456784248352, 20.401970624923706, 22.100252628326416, 23.8010733127594, 25.501973867416382, 27.20061469078064, 28.901782274246216, 30.600074291229248, 32.30033779144287, 34.001545906066895, 35.70003962516785, 37.40192985534668, 39.100926637649536, 40.80143094062805, 42.50145745277405, 44.20176577568054, 45.900986433029175, 47.60233926773071, 49.301738023757935] 
	q6 = [0.39999999999999997, 0.5128205128205129, 0.6341463414634148, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7500000000000001, 0.7500000000000001, 0.7500000000000001, 0.7500000000000001, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203] 
	t7 = [0, 1.4049711227416992, 2.880235195159912, 4.3137452602386475, 5.772370100021362, 7.265350103378296, 8.449827909469604, 10.224270820617676, 11.327435731887817, 12.803131818771362, 14.250712633132935, 15.507310390472412, 16.80337381362915, 18.4821617603302, 19.60011887550354, 21.000885248184204, 22.40180492401123, 23.801467895507812, 25.2018723487854, 26.60003924369812, 28.000375270843506, 29.40036916732788, 30.80198884010315, 32.202319622039795, 33.60067939758301, 35.002076864242554, 36.40159559249878, 37.80155110359192, 39.201613426208496, 40.60078692436218, 42.00142812728882, 43.40014719963074, 44.80046200752258, 46.2005569934845, 47.60083365440369, 49.00107765197754] 
	q7 = [0.39999999999999997, 0.5263157894736842, 0.5641025641025642, 0.6818181818181818, 0.6666666666666666, 0.6666666666666666, 0.7234042553191491, 0.7234042553191491, 0.7234042553191491, 0.7234042553191491, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203] 
	t8 = [0, 1.2808151245117188, 2.4217617511749268, 3.6757867336273193, 4.892728805541992, 6.0928449630737305, 7.290329456329346, 8.5179443359375, 9.715576410293579, 10.953459024429321, 12.16290807723999, 13.58251404762268, 14.570583581924438, 15.923755645751953, 16.87504768371582, 18.00007915496826, 19.201116800308228, 20.400644540786743, 21.60037326812744, 22.800840377807617, 24.001447677612305, 25.20026159286499, 26.402581930160522, 27.60026478767395, 28.800464868545532, 30.001708269119263, 31.201784372329712, 32.40163588523865, 33.60000658035278, 34.80187797546387, 36.00152659416199, 37.20030331611633, 38.400819063186646, 39.601264238357544, 40.80009198188782, 42.00018286705017, 43.20060586929321, 44.40133213996887, 45.600337743759155, 46.800128698349, 48.001869201660156, 49.20036053657532] 
	q8 = [0.39999999999999997, 0.5405405405405405, 0.5641025641025642, 0.6190476190476191, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7500000000000001, 0.7500000000000001, 0.7500000000000001, 0.7500000000000001, 0.7500000000000001, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203] 
	t9 = [0, 1.2417638301849365, 2.3119306564331055, 3.4556431770324707, 4.564756631851196, 5.902765512466431, 6.638518571853638, 7.8812255859375, 8.95536208152771, 10.223217248916626, 11.147271156311035, 12.447021245956421, 13.566431522369385, 14.491518497467041, 15.451788663864136, 16.818572998046875, 17.77179527282715, 18.701967477798462, 19.80170464515686, 20.901643991470337, 22.000022649765015, 23.100385427474976, 24.20015549659729, 25.301753520965576, 26.401885509490967, 27.501871824264526, 28.60189414024353, 29.70083236694336, 30.80087375640869, 31.900816202163696, 33.001309394836426, 34.1006121635437, 35.201581716537476, 36.30131959915161, 37.40126299858093, 38.50018620491028, 39.602458238601685, 40.70152139663696, 41.80184292793274, 42.901586055755615, 44.00063753128052, 45.10314083099365, 46.200809955596924, 47.30073070526123, 48.4000608921051, 49.50164484977722] 
	q9 = [0.39999999999999997, 0.5405405405405405, 0.5641025641025642, 0.6341463414634148, 0.6818181818181818, 0.6818181818181818, 0.711111111111111, 0.711111111111111, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7391304347826088, 0.7500000000000001, 0.7500000000000001, 0.7500000000000001, 0.7500000000000001, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203] 
	t10 = [0, 1.019456386566162, 2.123314619064331, 3.0875487327575684, 4.059500217437744, 5.3141443729400635, 6.121358394622803, 7.384932279586792, 8.091196775436401, 9.29889965057373, 10.028690814971924, 11.241313695907593, 12.094544410705566, 13.295026302337646, 14.266191720962524, 15.115597248077393, 16.361769437789917, 17.316311597824097, 18.001606702804565, 19.00057625770569, 20.001737117767334, 21.00007438659668, 22.001301050186157, 23.000999212265015, 24.000773429870605, 25.00332474708557, 26.001022815704346, 27.00058937072754, 28.00152325630188, 29.002042531967163, 30.000313758850098, 31.001221656799316, 32.001128911972046, 33.00130105018616, 34.000632762908936, 35.000190019607544, 36.00088810920715, 37.00010013580322, 38.00177836418152, 39.001317501068115, 40.00089907646179, 41.001038551330566, 42.0019907951355, 43.0022087097168, 44.001038789749146, 45.00019979476929, 46.000208616256714, 47.00026845932007, 48.0006046295166, 49.00071310997009, 50.000850200653076] 
	q10 = [0.39999999999999997, 0.5142857142857142, 0.5128205128205129, 0.6, 0.6818181818181818, 0.6666666666666666, 0.7234042553191491, 0.7234042553191491, 0.7234042553191491, 0.7234042553191491, 0.7234042553191491, 0.7234042553191491, 0.7234042553191491, 0.7234042553191491, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203, 0.7346938775510203] 
	t1_all.append(t1)
	t2_all.append(t2)
	t3_all.append(t3)
	t4_all.append(t4)
	t5_all.append(t5)
	t6_all.append(t6)
	t7_all.append(t7)
	t8_all.append(t8)
	t9_all.append(t9)
	t10_all.append(t10)
	
	q1_all.append(q1)
	q2_all.append(q2)
	q3_all.append(q3)		
	q4_all.append(q4)
	q5_all.append(q5)
	q6_all.append(q6)
	q7_all.append(q7)
	q8_all.append(q8)
	q9_all.append(q9)
	q10_all.append(q10)
	
	
	q1 = [sum(e)/len(e) for e in zip(*q1_all)]
	q2 = [sum(e)/len(e) for e in zip(*q2_all)]
	q3 = [sum(e)/len(e) for e in zip(*q3_all)]
	q4 = [sum(e)/len(e) for e in zip(*q4_all)]
	q5 = [sum(e)/len(e) for e in zip(*q5_all)]
	q6 = [sum(e)/len(e) for e in zip(*q6_all)]
	q7 = [sum(e)/len(e) for e in zip(*q7_all)]
	q8 = [sum(e)/len(e) for e in zip(*q8_all)]
	q9 = [sum(e)/len(e) for e in zip(*q9_all)]
	q10 = [sum(e)/len(e) for e in zip(*q10_all)]
	
	q1 = [sum(e)/len(e) for e in zip(*q1_all)]
	q2 = [sum(e)/len(e) for e in zip(*q2_all)]
	q3 = [sum(e)/len(e) for e in zip(*q3_all)]
	q4 = [sum(e)/len(e) for e in zip(*q4_all)]
	q5 = [sum(e)/len(e) for e in zip(*q5_all)]
	q6 = [sum(e)/len(e) for e in zip(*q6_all)]
	q7 = [sum(e)/len(e) for e in zip(*q7_all)]
	q8 = [sum(e)/len(e) for e in zip(*q8_all)]
	q9 = [sum(e)/len(e) for e in zip(*q9_all)]
	q10 = [sum(e)/len(e) for e in zip(*q10_all)]
	
	'''
	plt.plot(t1, q1,lw=2,color='blue',marker='o',  label='Iterative Approach(epoch=1)')
	plt.plot(t2, q2,lw=2,color='green',marker='^',  label='Iterative Approach(epoch=2)')
	plt.plot(t3, q3,lw=2,color='orange',marker ='d', label='Iterative Approach(epoch=3)') ##2,000
	plt.plot(t4, q4,lw=2,color='yellow',marker='o',  label='Iterative Approach(epoch=4)')
	print len(t5)
	
	print len(q5)
	plt.plot(t5, q5,lw=2,color='black',marker='^',  label='Iterative Approach(epoch=5)')
	plt.plot(t6, q6,lw=2,color='cyan',marker ='d', label='Iterative Approach(epoch=6)') ##2,000
	
	plt.ylim([0, 1])
	plt.xlim([0, max(max(t1),max(t2),max(t3))])
	plt.title('Quality vs Cost')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .11), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='x-small')
	plt.ylabel('F1-measure')
	#plt.ylabel('Gain')
	plt.xlabel('Cost')	
	plt.savefig('Muct_epoch_size_variation_gender_100_epoch_iter20.png', format='png')
	plt.savefig('Muct_epoch_size_variation_gender_100_epoch_iter20.eps', format='eps')
		#plt.show()
	plt.close()
	'''
	
	
	
	q1_new = np.asarray(q1)
	q2_new = np.asarray(q2)
	q3_new = np.asarray(q3)
	q4_new = np.asarray(q4)
	q5_new = np.asarray(q5)
	q6_new = np.asarray(q6)
	q7_new = np.asarray(q7)
	q8_new = np.asarray(q8)
	q9_new = np.asarray(q9)
	q10_new = np.asarray(q10)
	
	t1_new = [sum(e)/len(e) for e in zip(*t1_all)]
	t2_new = [sum(e)/len(e) for e in zip(*t2_all)]
	t3_new = [sum(e)/len(e) for e in zip(*t3_all)]
	t4_new = [sum(e)/len(e) for e in zip(*t4_all)]
	t5_new = [sum(e)/len(e) for e in zip(*t5_all)]
	t6_new = [sum(e)/len(e) for e in zip(*t6_all)]
	t7_new = [sum(e)/len(e) for e in zip(*t7_all)]
	t8_new = [sum(e)/len(e) for e in zip(*t8_all)]
	t9_new = [sum(e)/len(e) for e in zip(*t9_all)]
	t10_new = [sum(e)/len(e) for e in zip(*t10_all)]
	
	

	t1_list = [t1_new,t2_new,t3_new,t4_new,t5_new,t6_new,t7_new,t8_new,t9_new,t10_new]
	q1_list = [q1_new,q2_new,q3_new,q4_new,q5_new,q6_new,q7_new,q8_new,q9_new,q10_new]
	#epoch_list = [1,2,4,6,8,10]
	#epoch_list = [1,2,3,4,5,6,7,8,9,10]
	epoch_list = [1,2,3,4,5,6,7,8,9,10] # percent list
	score_list = []
	
	
	for i1 in range(len(t1_list)):
		t1_2 = t1_list[i1]
		t1_2 = t1_2[1:]
		q1_2 = q1_list[i1]
		weight_t1 = [max(1-float(element - 1)/budget,0) for element in t1_2]
		improv_q1 = [x - q1_2[i - 1] for i, x in enumerate(q1_2) if i > 0]
		print weight_t1
		print improv_q1
		a1 = np.dot(weight_t1,improv_q1)
		print a1
		score_list.append(a1)
	print>>f1,"epoch_list = {} ".format(epoch_list)
	print>>f1,"score_list = {} ".format(score_list)	
	plt.plot(epoch_list, score_list,lw=2,color='blue',marker='o',  label='Iterative Approach')
	
	
	#plt.ylim([0, 1])
	plt.xlim([0, max(epoch_list)])
	#plt.xlim([0, budget])
	plt.title('AUC Score vs Epoch Size')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .11), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
	plt.ylabel('AUC Score',fontsize='x-large')
	#plt.ylabel('Gain')
	#plt.xlabel('Epoch Size')	
	plt.xlabel('Percentage of time spent in plan generation phase',fontsize='large')	
	plt.savefig('EpochSize_AUC_Plot_2_5_percent_list.png', format='png')
	plt.savefig('EpochSize_AUC_Plot_2_5_percent_list.eps', format='eps')
		#plt.show()
	plt.close()	
	
	##### Plotting with setting the ylim #######
	plt.plot(epoch_list, score_list,lw=2,color='blue',marker='o',  label='Iterative Approach')
	
	
	plt.ylim([0, 1])
	plt.xlim([0, max(epoch_list)])
	plt.title('AUC Score vs Epoch Size')
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .11), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,fontsize='medium')
	plt.ylabel('AUC Score',fontsize='x-large')
	#plt.ylabel('Gain')
	#plt.xlabel('Epoch Size')	
	plt.xlabel('Percentage of time spent in plan generation phase',fontsize='large')	
	plt.savefig('EpochSize_AUC_Plot_ylim_100_iter40_2_5percent_list.png', format='png')
	plt.savefig('EpochSize_AUC_Plot_ylim_100_iter40_2_5percent_list.eps', format='eps')
		#plt.show()
	plt.close()	
	
	
	

def determineMaxTime():
	imageIndex = [i1 for i1 in sorted(random.sample(xrange(len(dl2)), 100))]
	dl_test = [dl2[i1] for i1 in  imageIndex]
	nl_test = [nl2[i1] for i1 in imageIndex]
	
	dl = np.array(dl_test)
	nl = np. array(nl_test)
	
	stepSize =4 
	currentTimeBound = 4
	
		
		
	

	
if __name__ == '__main__':
	t1 = time.time()
	setup()	
	generateMultipleExecutionResult()
	#plotOptimalEpoch()
	#generateDifferentStrategyResults_benefit_estimation()
	generateDifferentStrategyResults_benefit_estimation_progressive_score()
	#generateDifferentStrategyResultsUsingProgressiveScore()
	
	