from random import shuffle
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict

def domainExtract(stri,frac, trdat, cvdat, tedat):
#Extract the content of 3 domains of each observation from the txt file
#Put the content into the object dat 
#dat is a 3 column list to store the observation, each row is an observation
#the 1st column is "title", the 2nd column is "desc", the 3rd column is "label"
#str is the txt file's name
    dat=[]
    
    with open(stri) as input_data:
        start = False
        area = ''
    
        title = ''
        content = ''
        lab = ''
    
        for line in input_data:
            if line.strip() == '</JOB>' and lab != '':
                row = (title,content,lab)
                dat.append(row)
                title =''
                content =''
                lab =''
            if start == True:
                if area == 'title':
                    if line.strip() == '</TITLE>':
                        start = False                        
                    elif line.strip != '':
                        title = title+" "+line.strip()
                if area == 'description':
                    if line.strip() == '</DESC>':
                        start = False
                    elif line.strip != '':
                        content = content+" "+line.strip()
                if area == 'label':
                    if line.strip() == '</TAG>':
                        start = False
                    elif line.strip() != '':
                        lab = lab+line.strip()
            else:
                if line.strip() == '<TITLE>':
                    area = "title"
                    start = True
                if line.strip() == '<DESC>':
                    area = "description"
                    start = True
                if line.strip() == '<TAG>':
                    area = "label"
                    start = True
     
        tr, cv, te = datasetSplit(dat,frac) 
        trdat = trdat+tr
        cvdat = cvdat+cv
        tedat = tedat+te            
    
    shuffle(trdat)
    shuffle(cvdat)
    shuffle(tedat)
    return trdat,cvdat,tedat

def datasetSplit(dat, frac):
#split the original data into training set and testing set
#dat is the original data set, a list
#frac is the fraction of training data
    
    shuffle(dat)
    nTotal=len(dat)
    nTrain=int(round(nTotal*frac[0]))
    nCV=int(round(nTotal*frac[1]))
    trDat=dat[:nTrain]
    cvDat=dat[nTrain:(nTrain+nCV)]
    teDat=dat[(nTrain+nCV):]
    
    return trDat,cvDat,teDat

def createFeature(dat, domain):
    
    if domain=='Desc':
        all_words = set(wordpunct_tokenize(dat[1].lower()))
    elif domain=='Title':
        all_words = set(wordpunct_tokenize(dat[0].lower()))
    elif domain=='Full':
        title = wordpunct_tokenize(dat[0].lower())
        title = [w for w in title if len(w)>1]
        desc = wordpunct_tokenize(dat[1].lower())
        desc = [w for w in desc if len(w)>1]
        all_words = title*int(len(desc)/max(len(title),1))+desc
                
    all_words = set(all_words)
    sw = stopwords.words('english')
    all_words = all_words.difference(sw)

    features = defaultdict(list)
    for w in all_words:
        features[w] = True
    
    return features

def createFeatureLabel(dat,domain):
    feature_labels = []
    stemmer = SnowballStemmer('english')
    
    for i in range(0,len(dat)):
        
        if domain=='Desc':
            all_words = wordpunct_tokenize(dat[i][1].lower())
            all_words = [w for w in all_words if len(w)>1]
        
            for idx,w in enumerate(all_words):
                try:
                    all_words[idx] = stemmer.stem(w)
                except UnicodeDecodeError:
                    all_words.pop(idx)
        elif domain=='Title':
            all_words = wordpunct_tokenize(dat[i][0].lower())
            all_words = [w for w in all_words if len(w)>1]
        elif domain=='Full':
            title = wordpunct_tokenize(dat[i][0].lower())
            title = [w for w in title if len(w)>1]
            desc = wordpunct_tokenize(dat[i][1].lower())
            desc = [w for w in desc if len(w)>1]
            all_words = title*int(len(desc)/max(len(title),1))+desc
        
        all_words = set(all_words)
        sw = stopwords.words('english')
        all_words = all_words.difference(sw)

        features = defaultdict(list)
        for w in all_words:
            features[w] = True
        
        feature_labels.append((features,str(dat[i][2]).strip("[']")[0:2]))
    
    return feature_labels
    
    
##############function for naive base
from nltk import NaiveBayesClassifier as nbc
import nltk.classify
from nltk.classify.api import ClassifierI

def titleFeatureset(trdat,cvdat):
        
    tr_feature_label = createFeatureLabel(trdat,'Title')
    cv_feature_label = createFeatureLabel(cvdat,'Title')
        
    return tr_feature_label, cv_feature_label
        
def descFeatureset(trdat,cvdat):
        
    tr_feature_label = createFeatureLabel(trdat,'Desc')
    cv_feature_label = createFeatureLabel(cvdat,'Desc')
        
    return tr_feature_label, cv_feature_label
    
def titleClassifier(trdat,cvdat):
        
    tr_feature_label, cv_feature_label = titleFeatureset(trdat,cvdat)
    titleClf=nbc.train(tr_feature_label)
        
    return titleClf
    
def descClassifier(trdat,cvdat):
        
    tr_feature_label, cv_feature_label = descFeatureset(trdat,cvdat)
    descClf=nbc.train(tr_feature_label)
        
    descAccu = nltk.classify.accuracy(descClf,tr_feature_label)
        
    cv_tagged = ""
    for i in range(0,len(cv_feature_label)):
        result = descClf.classify(cv_feature_label[i][0])
        cv_tagged = cv_tagged + " " + result

    cv_label = ""
    for i in range(0,len(cv_feature_label)):
        cv_label = cv_label + " " + cv_feature_label[i][1]
            
    descCm = nltk.ConfusionMatrix(cv_label.split(),cv_tagged.split())
        
    return descClf, descAccu, descCm

class EnhanceNaiveBayesClassifier(ClassifierI):
    
    def __init__(self, titleClf, descClf, descAccu, descCm):
        self._titleClf = titleClf
        self._descClf = descClf
        self._descAccu = descAccu
        self._descCm = descCm
    
    def classify(self,tedat):
        
        te_titleFeature = createFeature(tedat,'Title')
        tag_title = self._titleClf.classify(te_titleFeature)
        
        te_descFeature = createFeature(tedat, 'Full')
        tag_desc = self._descClf.classify(te_descFeature)
        
        if tag_title == tag_desc:
            return tag_desc
        else:
            cr_denominator = 0
            for v in self._descCm._values:
                cr_denominator += self._descCm[tag_title,v]
            ratio = self._descCm[tag_title,tag_title]/cr_denominator
            #critical_ratio = (self._descCm[tag_title,tag_title]+self._descCm[tag_title,tag_desc])/cr_denominator
            if ratio <= self._descAccu and self._descCm[tag_title,tag_desc]>1:
                return tag_title
            else:
                return tag_desc
    
    @staticmethod
    def train(trdat, cvdat):
        
        titleClf = titleClassifier(trdat, cvdat)
        descClf, descAccu, descCm = descClassifier(trdat, cvdat)
        
        print descAccu
        print descCm
        
        return EnhanceNaiveBayesClassifier(titleClf, descClf, descAccu, descCm)
        
from nltk import NaiveBayesClassifier as nbc
import nltk.classify
import sys, os
#read data
mypath = '/Users/gh603/Desktop/13'
alltxt = [ mypath+'/'+f for f in os.listdir(mypath) if f.endswith(".txt") ]

trDat = []
cvDat = []
teDat = []
frac = [0.5, 0.25, 0.25]
for stri in alltxt:
    trDat, cvDat, teDat = domainExtract(stri, frac, trDat, cvDat, teDat)

print len(trDat)
print len(teDat)

tr_feature_label = createFeatureLabel(trDat,'Full')
te_feature_label = createFeatureLabel(teDat,'Full')

#train data using naivebase model from nltk
clf=nbc.train(tr_feature_label)
print 'training finish'

te_tagged = ""
for i in range(0,len(te_feature_label)):
    result = clf.classify(te_feature_label[i][0])
    te_tagged = te_tagged + " " + result

te_label = ""
for i in range(0,len(te_feature_label)):
    te_label = te_label + " " + te_feature_label[i][1]
    
print ('Test accuracy: {0:.2f}%'
       .format(100 * nltk.classify.accuracy(clf, te_feature_label)))
#print clf.show_most_informative_features(20)

print len(te_label.split())
print len(te_tagged.split())

cm = nltk.ConfusionMatrix(te_label.split(),te_tagged.split())
print cm  