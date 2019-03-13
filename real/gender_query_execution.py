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
	
	