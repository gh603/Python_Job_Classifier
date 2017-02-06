# Python_Job_Classifier
This project is to build a job post classifier that is able to classify job posts into different occupation categories. 
The classification is achieved by applying Naive Bayesian Network. 
The code is written in Python. 
The Natural Language Processing part, such as Stemming, is achieved using NLTK package in Python. 

File descriptions are as below: 
1.Txt file: 
	-All text file included in this repository are sampled data for Naive Bayesian Network consturction and testing. 
	-Each text file includes all job posts that are in the same job category.
	-The text file are named uniquely with a number, and this number are used as the response variable in the Naive Bayesian Network. 
	-Three pieces textual data of job posts are collected: Title, Description and Label.

2.full_naive.py: 
	-this is the full python codes for this project, including extracting feature domains, tokenizing textual domain, 
	 stemming words, and Naive Bayesian Network construction. 
