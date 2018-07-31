

```python
import pandas as pd
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC,LinearSVR
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plot
%matplotlib inline
import os
import glob
import numpy as np
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb

import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import datetime
#import matplotlib.pyplot as plt
import pandas as pd
import glob
import tqdm
import datetime
import keras
import numpy as np
#test getWindowedValue

from numpy.lib.stride_tricks import as_strided
from keras.models import Sequential
from tensorflow.contrib.keras.python.keras.initializers import TruncatedNormal,Constant
from keras.layers.convolutional import Conv3D,Conv2D,MaxPooling1D,MaxPooling2D,MaxPooling3D,Conv1D
from keras.layers import Lambda,Multiply ,TimeDistributed
from keras.layers import GlobalAveragePooling1D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.recurrent import LSTM,GRU
from keras.layers.wrappers import TimeDistributed
from keras import initializers
from keras.engine import InputSpec, Layer
from keras.layers import Dense,Dropout,Flatten,Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam,SGD
import tensorflow as tf
import keras.backend.tensorflow_backend as K
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential,Model
from keras.optimizers import Adam,RMSprop
from keras.activations import tanh,relu
#from keras.utils import multi_gpu_model
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU
from keras.layers import Reshape,Dense, Dropout, Activation, Flatten,LSTM,GRU,Input,InputLayer,Activation, Input,Conv1D,MaxPooling1D,GlobalAveragePooling1D
from keras.layers import Convolution2D, MaxPooling2D,TimeDistributed,Convolution1D,MaxPooling1D,concatenate, Average,BatchNormalization,GlobalMaxPool1D
from keras.utils import np_utils
from keras import losses
#from keras_tqdm import TQDMNotebookCallback
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.grid_search import GridSearchCV
import sklearn
import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
%matplotlib inline
import pickle
from IPython.display import SVG

import keras
from fastText import train_unsupervised

LengthOfInputSequences=60
LengthOfWordVector=400
```

    Using TensorFlow backend.
    /usr/local/lib/python3.5/dist-packages/sklearn/cross_validation.py:41: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.
      "This module will be removed in 0.20.", DeprecationWarning)
    /usr/local/lib/python3.5/dist-packages/sklearn/grid_search.py:42: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
      DeprecationWarning)



```python
fileTrain='./train.txt'
fileDev='./dev.txt'
fileTest='./dfs-test.txt'
fileTest='../vardial2018gold/DFS/dfs-gold.txt'


```


```python
from string import punctuation
def text_to_wordlist(text, remove_stop_words=True, stem_words=False):
    # Clean the text, with the option to remove stop_words and to stem words.
    
    # Convert words to lower case and split them
    #text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\0k ", "0000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)
    
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)
```


```python
def loadData(fname):
    return pd.read_csv(fname, sep='\t',header=-1)


#22ste

def fitlerLine(lin):
    lin=lin.lower()
    for k in ['.',',','?',"'s","'t",',']:
        lin=lin.replace(k,' '+k+' ')
    #for k in ['.',',','?','!']:
    #    lin=lin.replace(k,' ')
    while '  ' in lin:
        lin=lin.replace('  ',' ')
    return lin.strip()
```


```python
df=loadData(fileTrain)
```


```python
df.describe
```




    <bound method NDFrame.describe of                                                         0    1
    0       Dat zeg ik liever niet. Ik moet een beslissing...  BEL
    1       De problemen van de westerse wereld zijn in ve...  BEL
    2       Uw werkgever krijgt 't moeilijk met mogelijke ...  BEL
    3       Kijk eens aan. Die man schildert ten behoeve v...  BEL
    4       Nooit. Ik weet het niet. Misschien was je me w...  BEL
    5       Je staat voor schut, en je verknalt je hele le...  BEL
    6       All in. Dit verbaast me. Perrault re raisede M...  BEL
    7       Broadway is een prachtig Engels dorpje bij de ...  BEL
    8       Maar ze was niet thuis. Wie niet? Alison. Mijn...  BEL
    9       Gaat hij goed, de Story? Nee? Ik denk dat het ...  BEL
    10      Hebben we hun lijnen? Neem ze af. Waar is de v...  BEL
    11      Ik heb uren op de grond gelegen voor de poten ...  BEL
    12      Niet bij mij. Bij mij schept hij niet op. Toch...  BEL
    13      De zwelling is hier minder. Al die vloeistof v...  BEL
    14      Ga jij maar. Ik blijf hier wachten tot ze bijk...  BEL
    15      Dat weet je niet omdat ik het je nog niet heb ...  BEL
    16      Clarence, blijf je terugtrekken. Daaronder. Ik...  BEL
    17      Ik ben teruggekeerd, ik ben zelfs nooit binnen...  BEL
    18      Wat voor zoete cr medrankjes heeft u? Bedankt ...  BEL
    19      Ik heb al te veel dode mensen gezien. Ik wil h...  BEL
    20      Maar ik zit hopeloos vast. Ik heb geen werk en...  BEL
    21      Er zijn er altijd die daar misbruik van maken ...  BEL
    22      Ik stelde mijn idee voor aan Andy Dranatelli m...  BEL
    23      Ze is met mij uitgegaan. Wacht, ik maak u een ...  BEL
    24      We legden takken op hem, legden hem neer en na...  BEL
    25      De jongens verwachten een warme douche. Hallo,...  BEL
    26      Volgens het handboek is alles in orde als de v...  BEL
    27      Dat blokkeert al mijn chakra's. Ik vind het zo...  BEL
    28      Ik wil die training gebruiken om een mijl te l...  BEL
    29      Dana was lesbisch. Excuseer me. Werden we expl...  BEL
    ...                                                   ...  ...
    299970  Dat is ze niet. Waarom dan dat verbandje? Is d...  DUT
    299971  En daarom snoei ik nu de heg. Ik ga alleen met...  DUT
    299972  Hij heeft bijna een nieuw record. Het tweede p...  DUT
    299973  De weersomstandigheden ook. Geduld, wachten, e...  DUT
    299974  We gaan geen virussen verspreiden. Wat vind je...  DUT
    299975  Hij was na een paar minuten dood. Heel erg bed...  DUT
    299976  Met zulke prachtige ingredi nten verwacht Marc...  DUT
    299977  We blijven gewoon schrijvers. Die blonde doet ...  DUT
    299978  We beginnen zo open en spontaan, als individu....  DUT
    299979  Je weet het echt niet meer? Nee, echt niet. Te...  DUT
    299980  Vangen. , hoe gaat het? Ryan? Wat is er? Ben j...  DUT
    299981  Het is heel dun. Het loopt gewoon helemaal uit...  DUT
    299982  Je moeder heeft gelijk, het is je eigen schuld...  DUT
    299983  U hebt het vorige week gekocht? Ja, maar ze ha...  DUT
    299984  We moeten naar een cyclus toe. Omstreeks 8 uur...  DUT
    299985  Gingen jullie weleens een stukje varen? In twe...  DUT
    299986  Nu ben ik de eiwitten kwijt die het hadden moe...  DUT
    299987  Uw stem in ruil voor de helft van Trini's leve...  DUT
    299988  Een ondernemende Duitse journalist had het zie...  DUT
    299989  De agenten achtervolgen de rechter en filmen h...  DUT
    299990  Paula heeft Ragolia nooit huur betaald. En Rag...  DUT
    299991  Het thema van deze week is superhelden. Die me...  DUT
    299992  Ze komt heel oprecht over. , de kleinzoon van ...  DUT
    299993  Alleen omdat ik in een stoel terechtkwam. We h...  DUT
    299994  Ik heb niets kunnen vinden. Ga naar de site va...  DUT
    299995  Het dessert. Mooi. Dat ziet er mooi uit. STEM ...  DUT
    299996  Jullie hebben de gevarenzone betreden. Twee va...  DUT
    299997  En dat weet u. Vuile smeerlap. , kent u het Pe...  DUT
    299998  Aan zijn titel zijn landrechten verbonden. Hij...  DUT
    299999  Zo. Niet schrikken. Ik bedek de kaas met een f...  DUT
    
    [300000 rows x 2 columns]>




```python
trainxRaw=list(map(fitlerLine,df[0]))
print(trainxRaw[:2])
trainyRaw=df[1].values
print(len(trainyRaw),trainyRaw[:10])
```

    ['dat zeg ik liever niet . ik moet een beslissing nemen voor het welzijn van dit meisje . dimeola zal een exorcisme zijn psychisch zieke dochter schaden . is dat waar of niet ?', 'de problemen van de westerse wereld zijn in veel opzichten anders dan 2 . 000 jaar geleden . maar onze burgerplicht blijft dezelfde . ons erfgoed verdedigen tegen wie het wil verdelen en vernietigen .']
    300000 ['BEL' 'BEL' 'BEL' 'BEL' 'BEL' 'BEL' 'BEL' 'BEL' 'BEL' 'BEL']



```python


```


```python
devdata=loadData(fileDev)
devxRaw=list(map(fitlerLine,devdata[0]))
devyRaw=devdata[1].values
print(len(devxRaw))
print(len(devyRaw),devyRaw[:10])

```

    500
    500 ['BEL' 'BEL' 'BEL' 'BEL' 'BEL' 'BEL' 'BEL' 'BEL' 'BEL' 'BEL']



```python
testdata=loadData(fileTest)
testy=testdata[1].values
testxRaw=list(map(fitlerLine,testdata[0]))
```


```python
testdata.describe
print(testxRaw[:10])
```

    ['waar ze biologische producten verbouwen . ze geniet van het buitenleven en haar dieren . tim van 30 woont in hilversum , is vrijgezel . in het dagelijks leven is hij online marketeer en hij besteed elke vrije minuut aan zijn boy toys .', 'en dan ga je mee . als ik echt vond dat het tijd werd voor iets anders , zou ik dat zeggen . als je een relatie hebt moet je daarover kunnen praten .', 'ik ben niet op zoek naar acteurs die goed spelen , maar eerder acteurs die in de huid kruipen van het personage . ik stel zware eisen aan de acteurs . ze moeten zich inleven ,', "olds escorteerde bommenwerpers boven europa . een keer werden 52 b 17 's neergeschoten door messerschmitt 's in drie minuten . het was vreselijk voor olds om dit te aanzien . de b 17 's waren machteloos .", 'helemaal mee eens . en ze hebben mij uitgekotst . dus ik doe mee . ik wil ze best op straat gooien . je twijfelt toch niet aan mijn loyaliteit ? hoe zit het met eric ?', 'wat naar , zeg . waar lijdt ze aan ? het is een natuurlijke dood . natuurlijk ? hoe oud is die meid ? ze is 88 jaar . doe je het met een vrouw van 88 ?', "al die nep glimlachjes en dat 'alles goed , debbie ? ' ze pakten mijn kind van me af , en toen zat ik weer in mijn rotflatje . dat laat ik me niet nog eens gebeuren .", 'wij willen weten wie je bent , waar je vandaan komt en wat je doet . dit gaat niet over jou , maar over ons . onze favoriete clip deze week waren de skaters van arick arthur .', 'ik hoop dat iedereen met dit probleem het niet opgeeft . we zijn ook niet opeens zo geworden . maar ik ga de goede kant op . het lukt wel . ze behaalt kleine overwinningen .', 'zet er dus een punt achter . donna , wat is er ? je kunt me alles vertellen . wat is er aan de hand ? nee , er is niets hoor . jawel , er is iets .']


from langid.langid import LanguageIdentifier, model
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)

for r in devxRaw[10:]:
    print(r,identifier.classify(r))
    break;
import langid
langid.set_languages(['de','fr','it','en'])
idscores=[langid.classify(r) for r in devxRaw]

print(([s[0] for s in idscores]))
print(devyRaw)
#print(([s[1] for s in idscores]))


```python
def mytoken(lin):
    return lin.split()
tfv = TfidfVectorizer(min_df=2,use_idf=1,
                      smooth_idf=1,ngram_range=(1,3),
                     )#analyzer='char_wb') #,stop_words='english')




#tsvd=TruncatedSVD(n_components=400,random_state=2016)   # this gives similar results as to Semeval , try n_components=600
#trainx=tsvd.fit_transform(trainx)
#evalx=tsvd.transform(evalx)
#clf=LinearDiscriminantAnalysis()

```


```python
#with more data
trainxRaw.extend(expBlex)
#trainyRaw=list(trainyRaw)
trainyRaw.extend(expBley)
print(len(trainxRaw),len(trainyRaw))
```

    472438 472438



```python
trainx=tfv.fit_transform(trainxRaw)
evalx=tfv.transform(devxRaw)
print (tfv.get_feature_names()[:10])
print (tfv.get_feature_names()[-10:])
print (trainx.shape,evalx.shape)
    
```

    ['00', '00 00', '00 00 00', '00 00 uur', '00 00 volgens', '00 003', '00 003 07', '00 00z', '00 00z created', '00 00z lastprinted']
    ['özlem', 'ún', 'úún', 'úún van', 'über', 'über die', 'über die dreigliederung', 'überhaupt', 'überhaupt een', 'überhaupt een kwetsbare']
    (472438, 5296955) (500, 5296955)



```python
trainy=np.array(trainyRaw)
evaly=devyRaw
```


```python
with open('features.txt','w') as fout:
    fout.writelines('\n'.join(tfv.get_feature_names()))
```


```python
clf=LinearSVC()

clf.fit(trainx,trainy)
predictValue=clf.predict(evalx)
print(confusion_matrix(evaly,predictValue))
print(classification_report(evaly,predictValue,digits=4))
'''
[[170  80]
 [ 76 174]]
             precision    recall  f1-score   support

        BEL       0.69      0.68      0.69       250
        DUT       0.69      0.70      0.69       250

avg / total       0.69      0.69      0.69       500
'''
```

    [[169  81]
     [ 77 173]]
                 precision    recall  f1-score   support
    
            BEL       0.69      0.68      0.68       250
            DUT       0.68      0.69      0.69       250
    
    avg / total       0.68      0.68      0.68       500
    





    '\n[[170  80]\n [ 76 174]]\n             precision    recall  f1-score   support\n\n        BEL       0.69      0.68      0.69       250\n        DUT       0.69      0.70      0.69       250\n\navg / total       0.69      0.69      0.69       500\n'




```python
#tfidf learned from train and dev
tfv = TfidfVectorizer(min_df=2,use_idf=1,
                      smooth_idf=1,ngram_range=(1,3),
                     )
tempTrain=trainxRaw.copy()
print(len(tempTrain))
tempTrain.extend(devxRaw)
print(len(tempTrain))
tfv.fit(tempTrain)
tempTrain=None
```

    300000
    300500



```python
#exame tfv features
print(len(tfv.get_feature_names()))
```


```python
trainx=tfv.transform(trainxRaw)
evalx=tfv.transform(devxRaw)
```


```python
#load test file
fileTest='./dfs-test.txt'
testdata=loadData(fileTest)
testxRaw=list(map(fitlerLine,testdata[0]))
testx=tfv.transform(testxRaw)
print(testx.shape)
```

    (20000, 5296955)



```python
import sklearn
sklearn.__version__
```




    '0.19.0'




```python
sf=sklearn.model_selection.StratifiedKFold(20)
from sklearn.datasets import make_classification
#from xgboost import XGBClassifer
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier
cvcount=0
allPredicts=[]
import os
os.system('mkdir -p models')

#trainx,trainy=make_classification(100000)
#print(trainx.shape,trainy.shape)
for trainindex,devindex in sf.split(trainx,trainy):
    cvtrainx,cvtrainy=trainx[trainindex],trainy[trainindex]
    cvdevx,cvdevy=trainx[devindex],trainy[devindex]
    #clf=LinearSVC()   #BaggingClassifier(base_estimator=LinearSVC(),n_estimators=3,n_jobs=-1)
    #clf.fit(cvtrainx,cvtrainy)
    
    #clf=lgb.LGBMClassifier(num_leaves=100,n_jobs=12,
    #                            learning_rate=0.1,n_estimators=1000,silent=False)
    
    #clf.fit(cvtrainx,cvtrainy,early_stopping_rounds=20,
    #    eval_set=(cvdevx,cvdevy),
    #    verbose=True)
    clf=LinearSVC(class_weight='balanced')
    clf.fit(cvtrainx,cvtrainy)
    
    #score1=clf.score(cvdevx,cvdevy)
    #score2=clf.score(evalx,evaly)
    print('========CROSS {}======='.format(cvcount))
    predictValue=clf.predict(cvdevx)
    #print('===============\t%d\t%f\t%f'%(cvcount,score1,score2))
    
    print(confusion_matrix(cvdevy,predictValue))
    print(classification_report(cvdevy,predictValue,digits=4))
    
    print('========TEST {}======='.format(cvcount))
    predictValue=clf.predict(testx)
    print(confusion_matrix(testy,predictValue))
    print(classification_report(testy,predictValue,digits=4))
    #predictValue=clf.predict(evalx)    
    #print(confusion_matrix(evaly,predictValue))
    #print(classification_report(evaly,predictValue))
    #clf=LinearSVC()
    #print(type(cvtrainy),type(evaly))
    #clf.fit(vstack((cvtrainx,evalx)),np.concatenate((cvtrainy,evaly)))
    '''
    clf=lgb.LGBMClassifier(num_leaves=100,n_jobs=12,
                                learning_rate=0.1,n_estimators=1000,silent=False)

    clf.fit(vstack((cvtrainx,evalx)),np.concatenate((cvtrainy,evaly)),early_stopping_rounds=20,
        eval_set=(cvdevx,cvdevy),
        verbose=True)
    
    score1=clf.score(cvdevx,cvdevy)
    score2=clf.score(evalx,evaly)
    #predictValue=clf.predict(cvdevx)
    print('===============\t%d\t%f\t%f'%(cvcount,score1,score2))
    
    modelName='./models/%02d_%f_%f.lgbm'%(cvcount,score1,score2)

    pickle.dump(clf,open(modelName,'wb'))
    testPredict=clf.predict(testx)
    predictFile='./models/%02d_%f_%f.lgbm.predict'%(cvcount,score1,score2)
    with open(predictFile,'w') as fout:
        fout.write(' '.join(testPredict))
    
    '''
    allPredicts.append(predictValue)
    finalPredict=np.array(allPredicts).T
    finalPredict=[Counter(s).most_common()[0][0] for s in finalPredict]
    print('========Final TEST {}======='.format(cvcount))
    print(confusion_matrix(testy,finalPredict))
    print(classification_report(testy,finalPredict,digits=4))
    cvcount+=1

    
    
    
'''
#light boost

[[1860 1140]
 [1114 1886]]
             precision    recall  f1-score   support

        BEL       0.63      0.62      0.62      3000
        DUT       0.62      0.63      0.63      3000

avg / total       0.62      0.62      0.62      6000

[1]	valid_0's binary_logloss: 0.690368
Training until validation scores don't improve for 20 rounds.

'''
```

    ========CROSS 0=======
    [[7796 8326]
     [1761 5739]]
                 precision    recall  f1-score   support
    
            BEL     0.8157    0.4836    0.6072     16122
            DUT     0.4080    0.7652    0.5323      7500
    
    avg / total     0.6863    0.5730    0.5834     23622
    
    ========TEST 0=======
    [[4521 5479]
     [2289 7711]]
                 precision    recall  f1-score   support
    
            BEL     0.6639    0.4521    0.5379     10000
            DUT     0.5846    0.7711    0.6650     10000
    
    avg / total     0.6242    0.6116    0.6015     20000
    
    ========Final TEST 0=======
    [[4521 5479]
     [2289 7711]]
                 precision    recall  f1-score   support
    
            BEL     0.6639    0.4521    0.5379     10000
            DUT     0.5846    0.7711    0.6650     10000
    
    avg / total     0.6242    0.6116    0.6015     20000
    



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-113-a6483f623be1> in <module>()
         23     #    verbose=True)
         24     clf=LinearSVC(class_weight='balanced')
    ---> 25     clf.fit(cvtrainx,cvtrainy)
         26 
         27     #score1=clf.score(cvdevx,cvdevy)


    /usr/local/lib/python3.5/dist-packages/sklearn/svm/classes.py in fit(self, X, y, sample_weight)
        233             self.class_weight, self.penalty, self.dual, self.verbose,
        234             self.max_iter, self.tol, self.random_state, self.multi_class,
    --> 235             self.loss, sample_weight=sample_weight)
        236 
        237         if self.multi_class == "crammer_singer" and len(self.classes_) == 2:


    /usr/local/lib/python3.5/dist-packages/sklearn/svm/base.py in _fit_liblinear(X, y, C, fit_intercept, intercept_scaling, class_weight, penalty, dual, verbose, max_iter, tol, random_state, multi_class, loss, epsilon, sample_weight)
        888         X, y_ind, sp.isspmatrix(X), solver_type, tol, bias, C,
        889         class_weight_, max_iter, rnd.randint(np.iinfo('i').max),
    --> 890         epsilon, sample_weight)
        891     # Regarding rnd.randint(..) in the above signature:
        892     # seed for srand in range [0..INT_MAX); due to limitations in Numpy


    KeyboardInterrupt: 



```python
print(len(allPredicts))
print(len(allPredicts[0]))
```


```python
#generate final submission
finalPredict=np.array(allPredicts).T
```


```python
print(finalPredict.shape)
```


```python
from collections import Counter
with open('submit.txt','w') as fout:
    for l in finalPredict:
        print(Counter(l).most_common()[0][0])
        fout.write(Counter(l).most_common()[0][0]+'\n')
```


```python
print(trainx.shape,trainy.shape,evalx.shape,evaly.shape)
clf=lgb.LGBMClassifier(num_leaves=150,n_jobs=12,
                                learning_rate=0.1,n_estimators=1000,silent=False)

clf.fit(trainx,trainy,early_stopping_rounds=20,
        eval_set=(evalx,evaly),
        verbose=True)
predictValue=clf.predict(evalx)
print(confusion_matrix(evaly,predictValue))
print(classification_report(evaly,predictValue))
```


```python
#fasttext
import fastText
import catboost
dir(catboost)
```


```python
print(type(trainyRaw),type(trainxRaw))
```


```python
rawdutFile='./data/rawdut.txt'
rawbelFile='./data/rawbel.txt'
trainlabeled='./data/trainlabeled.txt'
devlabeled='./data/devlabeled.txt'
testlabeled='./data/testlabeled.txt'
preFiexed='__label__'

```


```python
with open(trainlabeled,'w') as fout:
    for x,y in zip(trainxRaw,trainy):
        fout.write('__label__{} {}\n'.format(y,x))

with open(devlabeled,'w') as fout:
    for x,y in zip(devxRaw,devyRaw):
        fout.write('__label__{} {}\n'.format(y,x))
        
with open(testlabeled,'w') as fout:
    for x, y in zip(testxRaw,testy):
        fout.write('__label__{} {}\n'.format(y,x))
```


```python
#import gensim
import gensim
from  gensim.models.doc2vec import TaggedLineDocument
```

    Using TensorFlow backend.



```python
model=gensim.models.Doc2Vec(TaggedLineDocument('./data/rawtrainNoLb.txt'),size=200,window=8,min_count=5,workers=4,)
#get train dev test vectors
trainx=list(map(model.infer_vector,trainxRaw))
testx=list(map(model.infer_vector,testxRaw))

#get train dev test vectors
```


```python
trainx=np.array(trainx)
testx=np.array(testx)
print(trainx.shape,testx.shape)
```

    (300000, 100) (20000, 100)



```python
#eval test results
crf=sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
crf.fit(trainx,trainy)
predicts=crf.predict(testx)

print(classification_report(testy,predicts,digits=4))
'''
d100             precision    recall  f1-score   support

        BEL     0.5268    0.5710    0.5480     10000
        DUT     0.5317    0.4871    0.5084     10000

avg / total     0.5293    0.5291    0.5282     20000
'''
```

                 precision    recall  f1-score   support
    
            BEL     0.5268    0.5710    0.5480     10000
            DUT     0.5317    0.4871    0.5084     10000
    
    avg / total     0.5293    0.5291    0.5282     20000
    



```python

rawDu=np.array(trainxRaw)[trainyRaw=='DUT']
os.system('mkdir -p data')
with open(rawdutFile,'w') as fout:
    fout.writelines('\n'.join(map(str,rawDu)))
    
rawBel=np.array(trainxRaw)[trainyRaw=='BEL']
os.system('mkdir -p data')
with open('./data/rawbel.txt','w') as fout:
    fout.writelines('\n'.join(map(str,rawBel)))
```


```python
dir(fastText)
```


```python
import fastText
allPredicts=[]
for i in range(10):
    classifier = fastText.train_supervised(trainlabeled,wordNgrams=2)
    result = classifier.test(devlabeled)
    #print ('P@1:', result.precision)
    #print ('R@1:', result.recall)
    #print ('Number of examples:', result.nexamples)
    testResult=classifier.test(testlabeled)
    print(result,testResult)
    predict=classifier.predict(testxRaw)
    predict=[s[0].replace('__label__','') for s in predict[0]]
    predict=[0 if s=='DUT' else 1 for s in predict]
    print(len(predict))
    allPredicts.append(predict)
    finalPredict=np.array(allPredicts).T
    finalPredict=[Counter(s).most_common()[0][0] for s in finalPredict]
    print('========Final TEST {}======='.format(cnnmodelCount))
    print(confusion_matrix(cnnTesty,finalPredict))
    print(classification_report(cnnTesty,finalPredict,digits=4))
'''
(500, 0.612, 0.612) (20000, 0.6323, 0.6323)
(500, 0.596, 0.596) (20000, 0.62385, 0.62385)
(500, 0.594, 0.594) (20000, 0.62205, 0.62205)
(500, 0.592, 0.592) (20000, 0.62355, 0.62355)
(500, 0.612, 0.612) (20000, 0.6316, 0.6316)
(500, 0.61, 0.61) (20000, 0.63185, 0.63185)
(500, 0.6, 0.6) (20000, 0.6226, 0.6226)
(500, 0.586, 0.586) (20000, 0.62145, 0.62145)
(500, 0.614, 0.614) (20000, 0.62995, 0.62995)
(500, 0.61, 0.61) (20000, 0.62945, 0.62945)
(500, 0.596, 0.596) (20000, 0.62495, 0.62495)

'''
```

    (500, 0.626, 0.626) (20000, 0.6442, 0.6442)
    20000
    ========Final TEST 20=======
    [[5534 4466]
     [2650 7350]]
                 precision    recall  f1-score   support
    
              0     0.6762    0.5534    0.6087     10000
              1     0.6220    0.7350    0.6738     10000
    
    avg / total     0.6491    0.6442    0.6412     20000
    
    (500, 0.64, 0.64) (20000, 0.648, 0.648)
    20000
    ========Final TEST 20=======
    [[6401 3599]
     [3430 6570]]
                 precision    recall  f1-score   support
    
              0     0.6511    0.6401    0.6456     10000
              1     0.6461    0.6570    0.6515     10000
    
    avg / total     0.6486    0.6485    0.6485     20000
    
    (500, 0.612, 0.612) (20000, 0.64685, 0.64685)
    20000
    ========Final TEST 20=======
    [[6374 3626]
     [3412 6588]]
                 precision    recall  f1-score   support
    
              0     0.6513    0.6374    0.6443     10000
              1     0.6450    0.6588    0.6518     10000
    
    avg / total     0.6482    0.6481    0.6481     20000
    
    (500, 0.638, 0.638) (20000, 0.6484, 0.6484)
    20000
    ========Final TEST 20=======
    [[6598 3402]
     [3637 6363]]
                 precision    recall  f1-score   support
    
              0     0.6447    0.6598    0.6521     10000
              1     0.6516    0.6363    0.6439     10000
    
    avg / total     0.6481    0.6481    0.6480     20000
    
    (500, 0.618, 0.618) (20000, 0.6476, 0.6476)
    20000
    ========Final TEST 20=======
    [[6563 3437]
     [3602 6398]]
                 precision    recall  f1-score   support
    
              0     0.6456    0.6563    0.6509     10000
              1     0.6505    0.6398    0.6451     10000
    
    avg / total     0.6481    0.6481    0.6480     20000
    
    (500, 0.622, 0.622) (20000, 0.64235, 0.64235)
    20000
    ========Final TEST 20=======
    [[7163 2837]
     [4179 5821]]
                 precision    recall  f1-score   support
    
              0     0.6315    0.7163    0.6713     10000
              1     0.6723    0.5821    0.6240     10000
    
    avg / total     0.6519    0.6492    0.6476     20000
    
    (500, 0.624, 0.624) (20000, 0.64265, 0.64265)
    20000
    ========Final TEST 20=======
    [[7069 2931]
     [4087 5913]]
                 precision    recall  f1-score   support
    
              0     0.6337    0.7069    0.6683     10000
              1     0.6686    0.5913    0.6276     10000
    
    avg / total     0.6511    0.6491    0.6479     20000
    
    (500, 0.624, 0.624) (20000, 0.6441, 0.6441)
    20000
    ========Final TEST 20=======
    [[7069 2931]
     [4089 5911]]
                 precision    recall  f1-score   support
    
              0     0.6335    0.7069    0.6682     10000
              1     0.6685    0.5911    0.6274     10000
    
    avg / total     0.6510    0.6490    0.6478     20000
    
    (500, 0.64, 0.64) (20000, 0.6487, 0.6487)
    20000
    ========Final TEST 20=======
    [[6675 3325]
     [3705 6295]]
                 precision    recall  f1-score   support
    
              0     0.6431    0.6675    0.6551     10000
              1     0.6544    0.6295    0.6417     10000
    
    avg / total     0.6487    0.6485    0.6484     20000
    
    (500, 0.618, 0.618) (20000, 0.645, 0.645)
    20000
    ========Final TEST 20=======
    [[7020 2980]
     [4035 5965]]
                 precision    recall  f1-score   support
    
              0     0.6350    0.7020    0.6668     10000
              1     0.6669    0.5965    0.6297     10000
    
    avg / total     0.6509    0.6492    0.6483     20000
    





    '\n(500, 0.612, 0.612) (20000, 0.6323, 0.6323)\n(500, 0.596, 0.596) (20000, 0.62385, 0.62385)\n(500, 0.594, 0.594) (20000, 0.62205, 0.62205)\n(500, 0.592, 0.592) (20000, 0.62355, 0.62355)\n(500, 0.612, 0.612) (20000, 0.6316, 0.6316)\n(500, 0.61, 0.61) (20000, 0.63185, 0.63185)\n(500, 0.6, 0.6) (20000, 0.6226, 0.6226)\n(500, 0.586, 0.586) (20000, 0.62145, 0.62145)\n(500, 0.614, 0.614) (20000, 0.62995, 0.62995)\n(500, 0.61, 0.61) (20000, 0.62945, 0.62945)\n(500, 0.596, 0.596) (20000, 0.62495, 0.62495)\n\n'




```python
print(len(predict[0]),predict[0][0],cnnTesty[0])
```

    20000 ['__label__DUT'] 0



```python
print(result,testResult)
```

    (500, 0.592, 0.592) (20000, 0.61725, 0.61725)



```python
#simple CNN
```


```python
#print(trainx[0])
#get vocab by tfidf get_feature_names
tfv = TfidfVectorizer(min_df=2,use_idf=1,
                      smooth_idf=1,ngram_range=(1,1),
                     )
tfv.fit(trainxRaw)
print(len(tfv.get_feature_names()))
tfidfDict={s:sindex for sindex,  s in enumerate(  tfv.get_feature_names())}
#for s in trainxRaw[0].split():
#    if s in tfidfDict:
#        print(s,tfidfDict[s])

def toCnnIndexAllinOne(lin):
    rv=[]
    for s in lin.split()[:LengthOfInputSequences]:
        if s in tfidfDict:
            rv.append(tfidfDict[s])
    while(len(rv)<LengthOfInputSequences):
        rv.append(0)
    return rv


```

    80132



```python
#build w2v and use the w2v index
#train a w2v and change the trainCnnIndex/devCnnIndex
modelW2V=train_unsupervised('./data/rawtrainNoLb.txt',model='skipgram',minCount=2,dim=400)
tfidfDict={s:sindex for sindex,  s in enumerate(  modelW2V.get_words())}
#for s in trainxRaw[0].split():
#    if s in tfidfDict:
#        print(s,tfidfDict[s])

print(len(tfidfDict))
```

    83968



```python
def toCnnIndexAllinOne(lin):
    rv=[]
    for s in lin.split()[:LengthOfInputSequences]:
        if s in tfidfDict:
            rv.append(tfidfDict[s])
    while(len(rv)<LengthOfInputSequences):
        rv.append(0)
    return rv

trainCnnIndex=list(map(toCnnIndexAllinOne,trainxRaw))
devCnnIndex=list(map(toCnnIndexAllinOne,devxRaw))
testCnnIndex=list(map(toCnnIndexAllinOne,testxRaw))
#print(modelW2V.get_output_matrix().shape[0])  #default minCount=5 42836    minCount=3 61282
```


```python
print(trainCnnIndex[:2])
```

    [[9, 144, 1, 503, 12, 0, 1, 40, 10, 1376, 278, 26, 6, 9156, 15, 33, 384, 0, 43132, 90, 10, 25186, 19, 11422, 4553, 447, 11025, 0, 8, 9, 70, 61, 12, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [5, 444, 15, 5, 7917, 353, 19, 13, 80, 11622, 155, 31, 546, 0, 319, 129, 359, 0, 17, 152, 32503, 317, 604, 0, 84, 15027, 2841, 132, 106, 6, 46, 5849, 11, 2786, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]



```python


#CNN
from keras.layers import Embedding
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot



#https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py 

def getCNN():
    input1=keras.layers.Input(shape=(LengthOfInputSequences,))    
    
    kernel_size = 3
    filters = 64
    pool_size = 3
    lstm_output_size=64
    '''
    x1=Embedding(input_dim=len(tfidfDict), #modelW2V.get_output_matrix().shape[0],
                input_length=LengthOfInputSequences,
                output_dim=100,     #modelW2V.get_output_matrix().shape[1],
                 trainable=True,
                #weights=[modelW2V.get_output_matrix()],
                #
               )(input1)
               '''
    x1=Embedding(input_dim=modelW2V.get_output_matrix().shape[0],
                input_length=LengthOfInputSequences,
                output_dim=modelW2V.get_output_matrix().shape[1],      
                 weights=[modelW2V.get_output_matrix()],
               )(input1)
    #x1=Dropout(0.2)(x1)
    x1=Conv1D(filters=filters,kernel_size=kernel_size)(x1)
    #x1=Dropout(0.2)(x1)
    #x1=Conv1D(filters=filters,kernel_size=kernel_size)(x1)
    #x1=Dropout(0.2)(x1)
    #x1=Conv1D(filters=filters,kernel_size=kernel_size)(x1)
    x1=MaxPooling1D(pool_size=pool_size)(x1)
    x1=Conv1D(filters=filters,kernel_size=kernel_size)(x1)
    x1=MaxPooling1D(pool_size=pool_size)(x1)
    #x1=Dropout(0.2)(x1)
    x1=LSTM(lstm_output_size)(x1)
    #x1=keras.layers.
    #x1=Flatten()(x1)
    
    
    
   
    
    
    
    
    
    #x1=Dropout(0.2)(x1)
    addLayer=Dense(512)(x1)
    output=Dense(2)(addLayer)
    output=Activation('softmax')(output)
    
    
    
    seq=keras.models.Model(inputs=[input1],outputs=output)
    
    seq.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    seq.summary()
    return seq


def getSparseMLP():
    model = Sequential()
    model.add(Dense(2048))
    return model
    pass

def getTDMLP(pretrain=False,trainable=True,lr=0.001):
    input1=keras.layers.Input(shape=(LengthOfInputSequences,))    
    
    kernel_size = 5
    filters = 32
    pool_size = 4
    lstm_output_size=512
    '''
    x1=Embedding(input_dim=len(tfidfDict), #modelW2V.get_output_matrix().shape[0],
                input_length=LengthOfInputSequences,
                output_dim=100,     #modelW2V.get_output_matrix().shape[1],
                 trainable=True,
                #weights=[modelW2V.get_output_matrix()],
                #
               )(input1)
               '''
    if pretrain:
        x1=Embedding(input_dim=modelW2V.get_output_matrix().shape[0],
                    input_length=LengthOfInputSequences,
                    output_dim=modelW2V.get_output_matrix().shape[1],      
                    weights=[modelW2V.get_output_matrix()],
                    trainable=trainable,
                     )(input1)
    else:
        x1=Embedding(input_dim=len(tfidfDict),
                    input_length=LengthOfInputSequences,
                    output_dim=LengthOfWordVector,   
                    trainable=trainable,

                   )(input1)
   
    #x1=Flatten()(x1)
    x1=LSTM(512,return_sequences=True)(x1)
    #x1=Dropout(0.2)(x1)
    #addLayer=Dense(512)(x1)
    output=  TimeDistributed(Dense(2))(x1)
    output=Activation('softmax')(output)
    
    
    
    seq=keras.models.Model(inputs=[input1],outputs=output)
    
    #seq.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    seq.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=lr),metrics=['accuracy'])
    seq.summary()
    return seq
def getMLP(pretrain=False,trainable=True,lr=0.001):
    input1=keras.layers.Input(shape=(LengthOfInputSequences,))    
    
    kernel_size = 5
    filters = 32
    pool_size = 4
    lstm_output_size=512
    '''
    x1=Embedding(input_dim=len(tfidfDict), #modelW2V.get_output_matrix().shape[0],
                input_length=LengthOfInputSequences,
                output_dim=100,     #modelW2V.get_output_matrix().shape[1],
                 trainable=True,
                #weights=[modelW2V.get_output_matrix()],
                #
               )(input1)
               '''
    if pretrain:
        x1=Embedding(input_dim=modelW2V.get_output_matrix().shape[0],
                    input_length=LengthOfInputSequences,
                    output_dim=modelW2V.get_output_matrix().shape[1],      
                    weights=[modelW2V.get_output_matrix()],
                    trainable=trainable,
                     )(input1)
    else:
        x1=Embedding(input_dim=len(tfidfDict),
                    input_length=LengthOfInputSequences,
                    output_dim=LengthOfWordVector,   
                    trainable=trainable,

                   )(input1)
   
    x1=Flatten()(x1)
    #x1=LSTM(512)(x1)
    #x1=Dropout(0.2)(x1)
    addLayer=Dense(512)(x1)
    output=Dense(2)(addLayer)
    output=Activation('softmax')(output)
    
    
    
    seq=keras.models.Model(inputs=[input1],outputs=output)
    
    #seq.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    seq.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=lr),metrics=['accuracy'])
    seq.summary()
    return seq
def getCNNIMDB2(pretrain=False):
    model = Sequential()
    kernel_size = 5
    filters = 64
    pool_size = 4
    lstm_output_size=70
    if pretrain:
        model.add(Embedding(modelW2V.get_output_matrix().shape[0], 
                            modelW2V.get_output_matrix().shape[1], 
                            input_length=LengthOfInputSequences,
                            weights=[modelW2V.get_output_matrix()]
                           ))
    else:
        model.add(Embedding(input_dim=len(tfidfDict),
                    input_length=LengthOfInputSequences,
                    output_dim=LengthOfWordVector,      
                    #weights=[modelW2V.get_output_matrix()],
                    #trainable=False,
                   ))
    
    model.add(Dropout(0.25))
    #model.add(Conv1D(256,
    #                 kernel_size=(1,),
    #                 padding='valid',
    #                 activation='relu',
    #                 strides=1))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='tanh',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='tanh',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(keras.layers.Bidirectional(GRU(lstm_output_size)))
    model.add(Dropout(0.25))
    model.add(Dense(200,activation='tanh'))
    #model.add(Dense(200,activation='relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=0.001),
                  metrics=['accuracy'])
    return model

def getCNNIMDB3():
    model = Sequential()
    kernel_size = 5
    filters = 64
    pool_size = 4
    lstm_output_size=70
       
    model.add(Embedding(input_dim=len(tfidfDict),
                input_length=LengthOfInputSequences,
                output_dim=LengthOfWordVector,      
                #weights=[modelW2V.get_output_matrix()],
                #trainable=False,
               ))
    
    model.add(Dropout(0.25))
    
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(TimeDistributed(Dense(1)))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(200,activation='relu'))
    
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def getAvgWordVector(pretrain=False):
    model = Sequential()
    kernel_size = 5
    filters = 64
    pool_size = 4
    lstm_output_size=70
    input1=Input(shape=(60,))   
    
    
    if pretrain:
        x1=Embedding(input_dim=modelW2V.get_output_matrix().shape[0],
                    input_length=LengthOfInputSequences,
                    output_dim=modelW2V.get_output_matrix().shape[1],      
                    weights=[modelW2V.get_output_matrix()],
                    #trainable=False,
                     )(input1)
    else:
        x1=Embedding(input_dim=len(tfidfDict),
                input_length=LengthOfInputSequences,
                output_dim=LengthOfWordVector,   )(input1)
    #x2=Conv1D(filters=64,kernel_size=kernel_size)(x1)
    x2=keras.layers.AveragePooling1D(60)(x1)
    x2=Flatten()(x2)
    #x2=LSTM(50)(x2)
    #x2=MaxPooling1D(pool_size=4)(x2)
    #x2=LSTM(50)(x2)
    
    #conv=keras.layers.Concatenate()([conv,x2])
    conv=x2
    conv=Dense(200)(conv)
    output=Dense(2)(conv)
    
    seq=keras.models.Model(inputs=[input1],outputs=output)
    
    seq.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    seq.summary()
    return seq


def getLSTM():
    model = Sequential()
    kernel_size = 5
    filters = 64
    pool_size = 4
    lstm_output_size=70
    input1=Input(shape=(60,))   
    x1=Embedding(input_dim=len(tfidfDict),
                input_length=LengthOfInputSequences,
                output_dim=LengthOfWordVector,   )(input1)

    x1=keras.layers.Bidirectional(LSTM(50))(x1)
    #only lstm ok
    #test with blstm  ok
    

    conv=Dense(200)(x1)
    output=Dense(2)(conv)
    
    seq=keras.models.Model(inputs=[input1],outputs=output)
    
    seq.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    seq.summary()
    return seq

def getLSTMAT():
    model = Sequential()
    kernel_size = 5
    filters = 64
    pool_size = 4
    lstm_output_size=70
    input1=Input(shape=(60,))   
    x1=Embedding(input_dim=len(tfidfDict),
                input_length=LengthOfInputSequences,
                output_dim=LengthOfWordVector,   )(input1)

    '''attentation'''
    
    #https://stackoverflow.com/questions/42918446/how-to-add-an-attention-mechanism-in-keras
    #https://github.com/keras-team/keras/issues/9658
    #https://github.com/keras-team/keras/issues/2403
    attention=TimeDistributed(Dense(10))(x1)
    attention=TimeDistributed(Dense(1))(attention)
    attention=Activation('softmax')(attention)
    
    x1=Multiply()([attention,x1])
    x1=keras.layers.Bidirectional(LSTM(50))(x1)
    #only lstm ok
    #test with blstm  ok
    
    
    
    #x1=keras.layers.Bidirectional(LSTM(50))(x1)
    conv=Dense(200)(x1)
    output=Dense(2)(conv)
    
    seq=keras.models.Model(inputs=[input1],outputs=output)
    
    seq.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    seq.summary()
    return seq


def getCNNAT():
    model = Sequential()
    kernel_size = 5
    filters = 64
    pool_size = 4
    lstm_output_size=70
    input1=Input(shape=(60,))   
    em=Embedding(input_dim=len(tfidfDict),
                input_length=LengthOfInputSequences,
                output_dim=LengthOfWordVector,   )(input1)

    '''attentation'''
    
    #https://stackoverflow.com/questions/42918446/how-to-add-an-attention-mechanism-in-keras
    #https://github.com/keras-team/keras/issues/9658
    #https://github.com/keras-team/keras/issues/2403
    attention=TimeDistributed(Dense(10))(em)
    attention=TimeDistributed(Dense(1))(attention)
    attention=Activation('softmax')(attention)
    
    x1=Multiply()([attention,em])
    x1=Conv1D(filters=64,kernel_size=kernel_size)(x1)
    x1=keras.layers.AveragePooling1D(3)(x1)
    x1=Conv1D(filters=64,kernel_size=kernel_size)(x1)
    x1=keras.layers.AveragePooling1D(3)(x1)
    x1=Flatten()(x1)
    #x2=keras.layers.AveragePooling1D(60)(em)
    #x2=Flatten()(x2)
    #x1=keras.layers.Concatenate()([x1,x2])
    #only lstm ok
    #test with blstm  ok
    
    
    
    #x1=keras.layers.Bidirectional(LSTM(50))(x1)
    conv=Dense(200)(x1)
    output=Dense(2)(conv)
    
    seq=keras.models.Model(inputs=[input1],outputs=output)
    
    seq.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    seq.summary()
    return seq

def getCNNAT2():
    model = Sequential()
    kernel_size = 5
    filters = 64
    pool_size = 4
    lstm_output_size=70
    input1=Input(shape=(60,))   
    em=Embedding(input_dim=len(tfidfDict),
                input_length=LengthOfInputSequences,
                output_dim=LengthOfWordVector,   )(input1)

    '''attentation'''
    
    #https://stackoverflow.com/questions/42918446/how-to-add-an-attention-mechanism-in-keras
    #https://github.com/keras-team/keras/issues/9658
    #https://github.com/keras-team/keras/issues/2403
    attention=TimeDistributed(Dense(10))(em)
    attention=TimeDistributed(Dense(1))(attention)
    attention=Activation('softmax')(attention)
    
    x1=Multiply()([attention,em])
    x1=Conv1D(filters=64,kernel_size=kernel_size)(x1)
    x1=keras.layers.AveragePooling1D(3)(x1)
    x1=Conv1D(filters=64,kernel_size=kernel_size)(x1)
    x1=keras.layers.AveragePooling1D(3)(x1)
    x1=Flatten()(x1)
    x2=keras.layers.AveragePooling1D(60)(em)
    x2=Flatten()(x2)
    x1=keras.layers.Concatenate()([x1,x2])
    #only lstm ok
    #test with blstm  ok
    
    
    
    #x1=keras.layers.Bidirectional(LSTM(50))(x1)
    conv=Dense(200)(x1)
    output=Dense(2)(conv)
    
    seq=keras.models.Model(inputs=[input1],outputs=output)
    
    seq.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    seq.summary()
    return seq

def getGRU(pretrain=False):
    model = Sequential()
    kernel_size = 5
    filters = 64
    pool_size = 4
    lstm_output_size=70
    input1=Input(shape=(60,))   
    if pretrain:
        x1=Embedding(input_dim=modelW2V.get_output_matrix().shape[0],
                    input_length=LengthOfInputSequences,
                    output_dim=modelW2V.get_output_matrix().shape[1],      
                    weights=[modelW2V.get_output_matrix()],
                    #trainable=False,
                     )(input1)
    else:
        x1=Embedding(input_dim=len(tfidfDict),
                    input_length=LengthOfInputSequences,
                    output_dim=LengthOfWordVector,   )(input1)
    
        

    x1=keras.layers.Bidirectional(GRU(50))(x1)
    #only lstm ok
    #test with blstm  ok
    

    conv=Dense(200)(x1)
    output=Dense(2)(conv)
    
    seq=keras.models.Model(inputs=[input1],outputs=output)
    
    seq.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    seq.summary()
    return seq

def getGRU2():
    model = Sequential()
    kernel_size = 5
    filters = 64
    pool_size = 4
    lstm_output_size=70
    input1=Input(shape=(60,)) 
    x1=Embedding(input_dim=len(tfidfDict),
                input_length=LengthOfInputSequences,
                output_dim=LengthOfWordVector,   )(input1)

    x1=keras.layers.Bidirectional(GRU(50))(x1)    
    #only lstm ok
    #test with blstm  ok
    

    conv=Dense(200)(x1)
    output=Dense(2)(conv)
    
    seq=keras.models.Model(inputs=[input1],outputs=output)
    
    seq.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001),metrics=['accuracy'])
    seq.summary()
    return seq

from keras import backend as K
K.clear_session()
crf=getTDMLP()    #getCNNIMDB2()

plot_model(crf, to_file='getTDMLP.png',show_shapes=True)
SVG(model_to_dot(crf, show_shapes=True).create(prog='dot', format='svg'))
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 60)                0         
    _________________________________________________________________
    embedding_1 (Embedding)      (None, 60, 400)           32052800  
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 60, 512)           1869824   
    _________________________________________________________________
    time_distributed_1 (TimeDist (None, 60, 2)             1026      
    _________________________________________________________________
    activation_1 (Activation)    (None, 60, 2)             0         
    =================================================================
    Total params: 33,923,650
    Trainable params: 33,923,650
    Non-trainable params: 0
    _________________________________________________________________





![svg](output_49_1.svg)




```python
LengthOfInputSequences=60
trainCnnIndex=list(map(toCnnIndexAllinOne,trainxRaw))
devCnnIndex=list(map(toCnnIndexAllinOne,devxRaw))
testCnnIndex=list(map(toCnnIndexAllinOne,testxRaw))
```


```python
from keras.callbacks import LearningRateScheduler
import math
def step_decay(epoch):
    initial_lrate=0.001
    drop=0.5
    epochs_drop=10.0
    lrate=initial_lrate*math.pow(drop,math.floor((1+epoch)/epochs_drop))
    print(lrate)
    return lrate

trainCnnIndex=np.array(trainCnnIndex)
print(trainCnnIndex.shape)
devCnnIndex=np.array(devCnnIndex)
print(devCnnIndex.shape)
testCnnIndex=np.array(testCnnIndex)
print(testCnnIndex.shape)
#trainCnnIndex=trainCnnIndex.reshape(-1,60,1)
#print(trainCnnIndex.shape)
#devCnnIndex=devCnnIndex.reshape(-1,60,1)
#testCnnIndex=testCnnIndex.reshape(-1,60,1)
cnnTrainy=[0 if s=='DUT' else 1 for s in trainy]
cnnDevy=[0 if s=='DUT' else 1 for s in evaly]
cnnTesty=[0 if s=='DUT' else 1 for s in testy]
cnnTrainyCat=keras.utils.to_categorical(cnnTrainy)
cnnDevyCat= keras.utils.to_categorical(cnnDevy)
import os
os.system('mkdir -p models')
#fullTrainX=np.vstack((trainCnnIndex,devCnnIndex))
#fullTrainY=np.concatenate((cnnTrainyCat,cnnDevyCat))
sf=sklearn.model_selection.StratifiedKFold(20)
cnnmodelCount=0
allPredicts=[]
for trainindex,devindex in sf.split(trainCnnIndex,cnnTrainy):
    K.clear_session()
    crf=getCNNIMDB2(pretrain=True)
    modelFile='./models/cnn_%d.hdf5'%(cnnmodelCount)
    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=2,
                               verbose=1,
                               min_delta=0.01,
                               mode='min'),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=0,
                                   verbose=1,
                                   epsilon=0.0001,
                                   mode='min'),
                 
                 ModelCheckpoint(monitor='val_loss',
                                 filepath=modelFile,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 verbose=1,
                                 mode='min'),
                 #LearningRateScheduler(step_decay),
                 ]
    '''
    crf.fit(trainCnnIndex,
            cnnTrainyCat,
            validation_data=(devCnnIndex,cnnDevyCat),
            epochs=100,batch_size=256,
            callbacks=callbacks)
            '''
    crf.fit(trainCnnIndex[trainindex],
            cnnTrainyCat[trainindex],       
            validation_data=(trainCnnIndex[devindex],cnnTrainyCat[devindex]),
            epochs=2,
            batch_size=256,
            callbacks=callbacks,
            shuffle=True
            )
    crf.load_weights(modelFile)
    devCnnIndex=trainCnnIndex[devindex]
    cnnDevy= np.argmax( cnnTrainyCat[devindex],axis=1)
    predictValue=crf.predict(devCnnIndex)
    
    #predict=[0 if s <th else 1 for s in predictValue]
    predict=np.argmax(predictValue,axis=1)
    print('------------%d------CROSS-------'%(cnnmodelCount))
    print(confusion_matrix(cnnDevy,predict))
    print(classification_report(cnnDevy,predict,digits=4))
    
    
    #check out the real test
    predictValue=crf.predict(testCnnIndex)
    
    #predict=[0 if s <th else 1 for s in predictValue]
    predict=np.argmax(predictValue,axis=1)
    print('------------%d------TEST-----------'%(cnnmodelCount))
    print(confusion_matrix(cnnTesty,predict))
    print(classification_report(cnnTesty,predict,digits=4))
    #print(predictValue.shape)
    #fpr, tpr, thresholds = metrics.roc_curve(cnnDevy, predictValue)
    #print(thresholds)
    #print(i,metrics.auc(fpr, tpr))
    
    allPredicts.append(predict)
    finalPredict=np.array(allPredicts).T
    finalPredict=[Counter(s).most_common()[0][0] for s in finalPredict]
    print('========Final TEST {}======='.format(cnnmodelCount))
    print(confusion_matrix(cnnTesty,finalPredict))
    print(classification_report(cnnTesty,finalPredict,digits=4))

    
    
    cnnmodelCount+=1
    
pickle.dump(allPredicts,open('./submit.getCNNIMDB2._df_2.pretrain.d400.pk','wb'))
```

    (300000, 60)
    (15000, 60)
    (20000, 60)
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.6504 - acc: 0.6129Epoch 00000: val_loss improved from inf to 0.62234, saving model to ./models/cnn_0.hdf5
    285000/285000 [==============================] - 40s - loss: 0.6504 - acc: 0.6129 - val_loss: 0.6223 - val_acc: 0.6475
    Epoch 2/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.5678 - acc: 0.6974Epoch 00001: val_loss improved from 0.62234 to 0.62108, saving model to ./models/cnn_0.hdf5
    285000/285000 [==============================] - 39s - loss: 0.5678 - acc: 0.6974 - val_loss: 0.6211 - val_acc: 0.6488
    ------------0------CROSS-------
    [[5178 2322]
     [2946 4554]]
                 precision    recall  f1-score   support
    
              0     0.6374    0.6904    0.6628      7500
              1     0.6623    0.6072    0.6336      7500
    
    avg / total     0.6498    0.6488    0.6482     15000
    
    ------------0------TEST-----------
    [[6710 3290]
     [4074 5926]]
                 precision    recall  f1-score   support
    
              0     0.6222    0.6710    0.6457     10000
              1     0.6430    0.5926    0.6168     10000
    
    avg / total     0.6326    0.6318    0.6312     20000
    
    ========Final TEST 0=======
    [[6710 3290]
     [4074 5926]]
                 precision    recall  f1-score   support
    
              0     0.6222    0.6710    0.6457     10000
              1     0.6430    0.5926    0.6168     10000
    
    avg / total     0.6326    0.6318    0.6312     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.6515 - acc: 0.6117Epoch 00000: val_loss improved from inf to 0.62137, saving model to ./models/cnn_1.hdf5
    285000/285000 [==============================] - 40s - loss: 0.6515 - acc: 0.6118 - val_loss: 0.6214 - val_acc: 0.6493
    Epoch 2/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.5712 - acc: 0.6944- ETA: 1s - loss: 0.5710
    Epoch 00001: reducing learning rate to 0.00010000000474974513.
    Epoch 00001: val_loss did not improve
    285000/285000 [==============================] - 39s - loss: 0.5712 - acc: 0.6944 - val_loss: 0.6227 - val_acc: 0.6533
    ------------1------CROSS-------
    [[4547 2953]
     [2307 5193]]
                 precision    recall  f1-score   support
    
              0     0.6634    0.6063    0.6336      7500
              1     0.6375    0.6924    0.6638      7500
    
    avg / total     0.6504    0.6493    0.6487     15000
    
    ------------1------TEST-----------
    [[5847 4153]
     [3198 6802]]
                 precision    recall  f1-score   support
    
              0     0.6464    0.5847    0.6140     10000
              1     0.6209    0.6802    0.6492     10000
    
    avg / total     0.6337    0.6324    0.6316     20000
    
    ========Final TEST 1=======
    [[7294 2706]
     [4631 5369]]
                 precision    recall  f1-score   support
    
              0     0.6117    0.7294    0.6654     10000
              1     0.6649    0.5369    0.5941     10000
    
    avg / total     0.6383    0.6331    0.6297     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.6507 - acc: 0.6120Epoch 00000: val_loss improved from inf to 0.62890, saving model to ./models/cnn_2.hdf5
    285000/285000 [==============================] - 40s - loss: 0.6507 - acc: 0.6120 - val_loss: 0.6289 - val_acc: 0.6408
    Epoch 2/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.5696 - acc: 0.6957Epoch 00001: val_loss improved from 0.62890 to 0.61658, saving model to ./models/cnn_2.hdf5
    285000/285000 [==============================] - 39s - loss: 0.5696 - acc: 0.6956 - val_loss: 0.6166 - val_acc: 0.6523
    ------------2------CROSS-------
    [[4508 2992]
     [2223 5277]]
                 precision    recall  f1-score   support
    
              0     0.6697    0.6011    0.6335      7500
              1     0.6382    0.7036    0.6693      7500
    
    avg / total     0.6540    0.6523    0.6514     15000
    
    ------------2------TEST-----------
    [[5691 4309]
     [3096 6904]]
                 precision    recall  f1-score   support
    
              0     0.6477    0.5691    0.6058     10000
              1     0.6157    0.6904    0.6509     10000
    
    avg / total     0.6317    0.6298    0.6284     20000
    
    ========Final TEST 2=======
    [[6138 3862]
     [3413 6587]]
                 precision    recall  f1-score   support
    
              0     0.6427    0.6138    0.6279     10000
              1     0.6304    0.6587    0.6442     10000
    
    avg / total     0.6365    0.6362    0.6361     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.6509 - acc: 0.6125Epoch 00000: val_loss improved from inf to 0.62776, saving model to ./models/cnn_3.hdf5
    285000/285000 [==============================] - 39s - loss: 0.6509 - acc: 0.6125 - val_loss: 0.6278 - val_acc: 0.6399
    Epoch 2/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.5692 - acc: 0.6960- EEpoch 00001: val_loss improved from 0.62776 to 0.61783, saving model to ./models/cnn_3.hdf5
    285000/285000 [==============================] - 38s - loss: 0.5692 - acc: 0.6960 - val_loss: 0.6178 - val_acc: 0.6501
    ------------3------CROSS-------
    [[5241 2259]
     [2990 4510]]
                 precision    recall  f1-score   support
    
              0     0.6367    0.6988    0.6663      7500
              1     0.6663    0.6013    0.6321      7500
    
    avg / total     0.6515    0.6501    0.6492     15000
    
    ------------3------TEST-----------
    [[6730 3270]
     [4104 5896]]
                 precision    recall  f1-score   support
    
              0     0.6212    0.6730    0.6461     10000
              1     0.6432    0.5896    0.6153     10000
    
    avg / total     0.6322    0.6313    0.6307     20000
    
    ========Final TEST 3=======
    [[6799 3201]
     [4069 5931]]
                 precision    recall  f1-score   support
    
              0     0.6256    0.6799    0.6516     10000
              1     0.6495    0.5931    0.6200     10000
    
    avg / total     0.6375    0.6365    0.6358     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.6506 - acc: 0.6130Epoch 00000: val_loss improved from inf to 0.62544, saving model to ./models/cnn_4.hdf5
    285000/285000 [==============================] - 40s - loss: 0.6506 - acc: 0.6130 - val_loss: 0.6254 - val_acc: 0.6423
    Epoch 2/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.5692 - acc: 0.6964
    Epoch 00001: reducing learning rate to 0.00010000000474974513.
    Epoch 00001: val_loss improved from 0.62544 to 0.62544, saving model to ./models/cnn_4.hdf5
    285000/285000 [==============================] - 39s - loss: 0.5692 - acc: 0.6964 - val_loss: 0.6254 - val_acc: 0.6513
    ------------4------CROSS-------
    [[4771 2729]
     [2502 4998]]
                 precision    recall  f1-score   support
    
              0     0.6560    0.6361    0.6459      7500
              1     0.6468    0.6664    0.6565      7500
    
    avg / total     0.6514    0.6513    0.6512     15000
    
    ------------4------TEST-----------
    [[6233 3767]
     [3560 6440]]
                 precision    recall  f1-score   support
    
              0     0.6365    0.6233    0.6298     10000
              1     0.6309    0.6440    0.6374     10000
    
    avg / total     0.6337    0.6337    0.6336     20000
    
    ========Final TEST 4=======
    [[6342 3658]
     [3552 6448]]
                 precision    recall  f1-score   support
    
              0     0.6410    0.6342    0.6376     10000
              1     0.6380    0.6448    0.6414     10000
    
    avg / total     0.6395    0.6395    0.6395     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.6509 - acc: 0.6124Epoch 00000: val_loss improved from inf to 0.62465, saving model to ./models/cnn_5.hdf5
    285000/285000 [==============================] - 40s - loss: 0.6509 - acc: 0.6124 - val_loss: 0.6247 - val_acc: 0.6415
    Epoch 2/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.5692 - acc: 0.6967
    Epoch 00001: reducing learning rate to 0.00010000000474974513.
    Epoch 00001: val_loss did not improve
    285000/285000 [==============================] - 38s - loss: 0.5692 - acc: 0.6967 - val_loss: 0.6252 - val_acc: 0.6538
    ------------5------CROSS-------
    [[4180 3320]
     [2057 5443]]
                 precision    recall  f1-score   support
    
              0     0.6702    0.5573    0.6086      7500
              1     0.6211    0.7257    0.6694      7500
    
    avg / total     0.6457    0.6415    0.6390     15000
    
    ------------5------TEST-----------
    [[5369 4631]
     [2722 7278]]
                 precision    recall  f1-score   support
    
              0     0.6636    0.5369    0.5936     10000
              1     0.6111    0.7278    0.6644     10000
    
    avg / total     0.6374    0.6323    0.6290     20000
    
    ========Final TEST 5=======
    [[6549 3451]
     [3731 6269]]
                 precision    recall  f1-score   support
    
              0     0.6371    0.6549    0.6459     10000
              1     0.6450    0.6269    0.6358     10000
    
    avg / total     0.6410    0.6409    0.6408     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.6510 - acc: 0.6114Epoch 00000: val_loss improved from inf to 0.62271, saving model to ./models/cnn_6.hdf5
    285000/285000 [==============================] - 40s - loss: 0.6510 - acc: 0.6114 - val_loss: 0.6227 - val_acc: 0.6469
    Epoch 2/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.5684 - acc: 0.6967Epoch 00001: val_loss improved from 0.62271 to 0.61489, saving model to ./models/cnn_6.hdf5
    285000/285000 [==============================] - 39s - loss: 0.5684 - acc: 0.6967 - val_loss: 0.6149 - val_acc: 0.6546
    ------------6------CROSS-------
    [[4854 2646]
     [2535 4965]]
                 precision    recall  f1-score   support
    
              0     0.6569    0.6472    0.6520      7500
              1     0.6523    0.6620    0.6571      7500
    
    avg / total     0.6546    0.6546    0.6546     15000
    
    ------------6------TEST-----------
    [[6067 3933]
     [3463 6537]]
                 precision    recall  f1-score   support
    
              0     0.6366    0.6067    0.6213     10000
              1     0.6244    0.6537    0.6387     10000
    
    avg / total     0.6305    0.6302    0.6300     20000
    
    ========Final TEST 6=======
    [[6176 3824]
     [3409 6591]]
                 precision    recall  f1-score   support
    
              0     0.6443    0.6176    0.6307     10000
              1     0.6328    0.6591    0.6457     10000
    
    avg / total     0.6386    0.6383    0.6382     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.6509 - acc: 0.6128Epoch 00000: val_loss improved from inf to 0.62418, saving model to ./models/cnn_7.hdf5
    285000/285000 [==============================] - 40s - loss: 0.6509 - acc: 0.6128 - val_loss: 0.6242 - val_acc: 0.6447
    Epoch 2/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.5700 - acc: 0.6952
    Epoch 00001: reducing learning rate to 0.00010000000474974513.
    Epoch 00001: val_loss did not improve
    285000/285000 [==============================] - 39s - loss: 0.5700 - acc: 0.6952 - val_loss: 0.6304 - val_acc: 0.6483
    ------------7------CROSS-------
    [[4393 3107]
     [2222 5278]]
                 precision    recall  f1-score   support
    
              0     0.6641    0.5857    0.6225      7500
              1     0.6295    0.7037    0.6645      7500
    
    avg / total     0.6468    0.6447    0.6435     15000
    
    ------------7------TEST-----------
    [[5685 4315]
     [3042 6958]]
                 precision    recall  f1-score   support
    
              0     0.6514    0.5685    0.6071     10000
              1     0.6172    0.6958    0.6542     10000
    
    avg / total     0.6343    0.6321    0.6307     20000
    
    ========Final TEST 7=======
    [[6381 3619]
     [3577 6423]]
                 precision    recall  f1-score   support
    
              0     0.6408    0.6381    0.6394     10000
              1     0.6396    0.6423    0.6410     10000
    
    avg / total     0.6402    0.6402    0.6402     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.6511 - acc: 0.6114Epoch 00000: val_loss improved from inf to 0.62022, saving model to ./models/cnn_8.hdf5
    285000/285000 [==============================] - 39s - loss: 0.6511 - acc: 0.6114 - val_loss: 0.6202 - val_acc: 0.6499
    Epoch 2/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.5683 - acc: 0.6969Epoch 00001: val_loss improved from 0.62022 to 0.61046, saving model to ./models/cnn_8.hdf5
    285000/285000 [==============================] - 39s - loss: 0.5684 - acc: 0.6969 - val_loss: 0.6105 - val_acc: 0.6630
    ------------8------CROSS-------
    [[4939 2561]
     [2494 5006]]
                 precision    recall  f1-score   support
    
              0     0.6645    0.6585    0.6615      7500
              1     0.6616    0.6675    0.6645      7500
    
    avg / total     0.6630    0.6630    0.6630     15000
    
    ------------8------TEST-----------
    [[6228 3772]
     [3673 6327]]
                 precision    recall  f1-score   support
    
              0     0.6290    0.6228    0.6259     10000
              1     0.6265    0.6327    0.6296     10000
    
    avg / total     0.6278    0.6278    0.6277     20000
    
    ========Final TEST 8=======
    [[6140 3860]
     [3359 6641]]
                 precision    recall  f1-score   support
    
              0     0.6464    0.6140    0.6298     10000
              1     0.6324    0.6641    0.6479     10000
    
    avg / total     0.6394    0.6391    0.6388     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.6503 - acc: 0.6140Epoch 00000: val_loss improved from inf to 0.62389, saving model to ./models/cnn_9.hdf5
    285000/285000 [==============================] - 39s - loss: 0.6503 - acc: 0.6140 - val_loss: 0.6239 - val_acc: 0.6434
    Epoch 2/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.5685 - acc: 0.6964Epoch 00001: val_loss improved from 0.62389 to 0.61867, saving model to ./models/cnn_9.hdf5
    285000/285000 [==============================] - 38s - loss: 0.5685 - acc: 0.6964 - val_loss: 0.6187 - val_acc: 0.6499
    ------------9------CROSS-------
    [[4554 2946]
     [2305 5195]]
                 precision    recall  f1-score   support
    
              0     0.6639    0.6072    0.6343      7500
              1     0.6381    0.6927    0.6643      7500
    
    avg / total     0.6510    0.6499    0.6493     15000
    
    ------------9------TEST-----------
    [[5828 4172]
     [3168 6832]]
                 precision    recall  f1-score   support
    
              0     0.6478    0.5828    0.6136     10000
              1     0.6209    0.6832    0.6505     10000
    
    avg / total     0.6344    0.6330    0.6321     20000
    
    ========Final TEST 9=======
    [[6310 3690]
     [3531 6469]]
                 precision    recall  f1-score   support
    
              0     0.6412    0.6310    0.6361     10000
              1     0.6368    0.6469    0.6418     10000
    
    avg / total     0.6390    0.6390    0.6389     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.6504 - acc: 0.6127Epoch 00000: val_loss improved from inf to 0.62119, saving model to ./models/cnn_10.hdf5
    285000/285000 [==============================] - 40s - loss: 0.6504 - acc: 0.6127 - val_loss: 0.6212 - val_acc: 0.6461
    Epoch 2/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.5692 - acc: 0.6962
    Epoch 00001: reducing learning rate to 0.00010000000474974513.
    Epoch 00001: val_loss did not improve
    285000/285000 [==============================] - 38s - loss: 0.5692 - acc: 0.6962 - val_loss: 0.6317 - val_acc: 0.6487
    ------------10------CROSS-------
    [[4446 3054]
     [2254 5246]]
                 precision    recall  f1-score   support
    
              0     0.6636    0.5928    0.6262      7500
              1     0.6320    0.6995    0.6641      7500
    
    avg / total     0.6478    0.6461    0.6451     15000
    
    ------------10------TEST-----------
    [[5736 4264]
     [3075 6925]]
                 precision    recall  f1-score   support
    
              0     0.6510    0.5736    0.6099     10000
              1     0.6189    0.6925    0.6536     10000
    
    avg / total     0.6350    0.6331    0.6317     20000
    
    ========Final TEST 10=======
    [[6081 3919]
     [3283 6717]]
                 precision    recall  f1-score   support
    
              0     0.6494    0.6081    0.6281     10000
              1     0.6315    0.6717    0.6510     10000
    
    avg / total     0.6405    0.6399    0.6395     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.6517 - acc: 0.6107Epoch 00000: val_loss improved from inf to 0.62248, saving model to ./models/cnn_11.hdf5
    285000/285000 [==============================] - 39s - loss: 0.6517 - acc: 0.6108 - val_loss: 0.6225 - val_acc: 0.6456
    Epoch 2/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.5711 - acc: 0.6946
    Epoch 00001: reducing learning rate to 0.00010000000474974513.
    Epoch 00001: val_loss did not improve
    285000/285000 [==============================] - 39s - loss: 0.5710 - acc: 0.6946 - val_loss: 0.6274 - val_acc: 0.6559
    ------------11------CROSS-------
    [[5029 2471]
     [2845 4655]]
                 precision    recall  f1-score   support
    
              0     0.6387    0.6705    0.6542      7500
              1     0.6532    0.6207    0.6365      7500
    
    avg / total     0.6460    0.6456    0.6454     15000
    
    ------------11------TEST-----------
    [[6529 3471]
     [3835 6165]]
                 precision    recall  f1-score   support
    
              0     0.6300    0.6529    0.6412     10000
              1     0.6398    0.6165    0.6279     10000
    
    avg / total     0.6349    0.6347    0.6346     20000
    
    ========Final TEST 11=======
    [[6306 3694]
     [3490 6510]]
                 precision    recall  f1-score   support
    
              0     0.6437    0.6306    0.6371     10000
              1     0.6380    0.6510    0.6444     10000
    
    avg / total     0.6409    0.6408    0.6408     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.6499 - acc: 0.6140Epoch 00000: val_loss improved from inf to 0.62295, saving model to ./models/cnn_12.hdf5
    285000/285000 [==============================] - 40s - loss: 0.6499 - acc: 0.6140 - val_loss: 0.6229 - val_acc: 0.6468
    Epoch 2/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.5680 - acc: 0.6961Epoch 00001: val_loss improved from 0.62295 to 0.62158, saving model to ./models/cnn_12.hdf5
    285000/285000 [==============================] - 39s - loss: 0.5680 - acc: 0.6961 - val_loss: 0.6216 - val_acc: 0.6568
    ------------12------CROSS-------
    [[4978 2522]
     [2626 4874]]
                 precision    recall  f1-score   support
    
              0     0.6547    0.6637    0.6592      7500
              1     0.6590    0.6499    0.6544      7500
    
    avg / total     0.6568    0.6568    0.6568     15000
    
    ------------12------TEST-----------
    [[6323 3677]
     [3685 6315]]
                 precision    recall  f1-score   support
    
              0     0.6318    0.6323    0.6320     10000
              1     0.6320    0.6315    0.6318     10000
    
    avg / total     0.6319    0.6319    0.6319     20000
    
    ========Final TEST 12=======
    [[6150 3850]
     [3339 6661]]
                 precision    recall  f1-score   support
    
              0     0.6481    0.6150    0.6311     10000
              1     0.6337    0.6661    0.6495     10000
    
    avg / total     0.6409    0.6405    0.6403     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.6503 - acc: 0.6128Epoch 00000: val_loss improved from inf to 0.62967, saving model to ./models/cnn_13.hdf5
    285000/285000 [==============================] - 40s - loss: 0.6503 - acc: 0.6128 - val_loss: 0.6297 - val_acc: 0.6429
    Epoch 2/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.5696 - acc: 0.6951Epoch 00001: val_loss improved from 0.62967 to 0.61698, saving model to ./models/cnn_13.hdf5
    285000/285000 [==============================] - 39s - loss: 0.5696 - acc: 0.6951 - val_loss: 0.6170 - val_acc: 0.6488
    ------------13------CROSS-------
    [[5072 2428]
     [2840 4660]]
                 precision    recall  f1-score   support
    
              0     0.6411    0.6763    0.6582      7500
              1     0.6574    0.6213    0.6389      7500
    
    avg / total     0.6493    0.6488    0.6485     15000
    
    ------------13------TEST-----------
    [[6561 3439]
     [3888 6112]]
                 precision    recall  f1-score   support
    
              0     0.6279    0.6561    0.6417     10000
              1     0.6399    0.6112    0.6252     10000
    
    avg / total     0.6339    0.6337    0.6335     20000
    
    ========Final TEST 13=======
    [[6361 3639]
     [3529 6471]]
                 precision    recall  f1-score   support
    
              0     0.6432    0.6361    0.6396     10000
              1     0.6401    0.6471    0.6436     10000
    
    avg / total     0.6416    0.6416    0.6416     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.6501 - acc: 0.6128Epoch 00000: val_loss improved from inf to 0.61965, saving model to ./models/cnn_14.hdf5
    285000/285000 [==============================] - 40s - loss: 0.6501 - acc: 0.6128 - val_loss: 0.6197 - val_acc: 0.6505
    Epoch 2/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.5675 - acc: 0.6970Epoch 00001: val_loss improved from 0.61965 to 0.61733, saving model to ./models/cnn_14.hdf5
    285000/285000 [==============================] - 39s - loss: 0.5675 - acc: 0.6970 - val_loss: 0.6173 - val_acc: 0.6517
    ------------14------CROSS-------
    [[4413 3087]
     [2137 5363]]
                 precision    recall  f1-score   support
    
              0     0.6737    0.5884    0.6282      7500
              1     0.6347    0.7151    0.6725      7500
    
    avg / total     0.6542    0.6517    0.6503     15000
    
    ------------14------TEST-----------
    [[5612 4388]
     [3011 6989]]
                 precision    recall  f1-score   support
    
              0     0.6508    0.5612    0.6027     10000
              1     0.6143    0.6989    0.6539     10000
    
    avg / total     0.6326    0.6300    0.6283     20000
    
    ========Final TEST 14=======
    [[6145 3855]
     [3340 6660]]
                 precision    recall  f1-score   support
    
              0     0.6479    0.6145    0.6307     10000
              1     0.6334    0.6660    0.6493     10000
    
    avg / total     0.6406    0.6402    0.6400     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.6501 - acc: 0.6125Epoch 00000: val_loss improved from inf to 0.62062, saving model to ./models/cnn_15.hdf5
    285000/285000 [==============================] - 40s - loss: 0.6501 - acc: 0.6125 - val_loss: 0.6206 - val_acc: 0.6495
    Epoch 2/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.5686 - acc: 0.6961
    Epoch 00001: reducing learning rate to 0.00010000000474974513.
    Epoch 00001: val_loss did not improve
    285000/285000 [==============================] - 38s - loss: 0.5686 - acc: 0.6961 - val_loss: 0.6245 - val_acc: 0.6571
    ------------15------CROSS-------
    [[4452 3048]
     [2210 5290]]
                 precision    recall  f1-score   support
    
              0     0.6683    0.5936    0.6287      7500
              1     0.6344    0.7053    0.6680      7500
    
    avg / total     0.6514    0.6495    0.6484     15000
    
    ------------15------TEST-----------
    [[5626 4374]
     [2971 7029]]
                 precision    recall  f1-score   support
    
              0     0.6544    0.5626    0.6050     10000
              1     0.6164    0.7029    0.6568     10000
    
    avg / total     0.6354    0.6328    0.6309     20000
    
    ========Final TEST 15=======
    [[6253 3747]
     [3429 6571]]
                 precision    recall  f1-score   support
    
              0     0.6458    0.6253    0.6354     10000
              1     0.6368    0.6571    0.6468     10000
    
    avg / total     0.6413    0.6412    0.6411     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.6508 - acc: 0.6122Epoch 00000: val_loss improved from inf to 0.63521, saving model to ./models/cnn_16.hdf5
    285000/285000 [==============================] - 40s - loss: 0.6508 - acc: 0.6122 - val_loss: 0.6352 - val_acc: 0.6283
    Epoch 2/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.5684 - acc: 0.6972Epoch 00001: val_loss improved from 0.63521 to 0.61543, saving model to ./models/cnn_16.hdf5
    285000/285000 [==============================] - 39s - loss: 0.5684 - acc: 0.6972 - val_loss: 0.6154 - val_acc: 0.6551
    ------------16------CROSS-------
    [[4683 2817]
     [2356 5144]]
                 precision    recall  f1-score   support
    
              0     0.6653    0.6244    0.6442      7500
              1     0.6461    0.6859    0.6654      7500
    
    avg / total     0.6557    0.6551    0.6548     15000
    
    ------------16------TEST-----------
    [[5886 4114]
     [3284 6716]]
                 precision    recall  f1-score   support
    
              0     0.6419    0.5886    0.6141     10000
              1     0.6201    0.6716    0.6448     10000
    
    avg / total     0.6310    0.6301    0.6295     20000
    
    ========Final TEST 16=======
    [[6104 3896]
     [3300 6700]]
                 precision    recall  f1-score   support
    
              0     0.6491    0.6104    0.6291     10000
              1     0.6323    0.6700    0.6506     10000
    
    avg / total     0.6407    0.6402    0.6399     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.6510 - acc: 0.6123Epoch 00000: val_loss improved from inf to 0.62461, saving model to ./models/cnn_17.hdf5
    285000/285000 [==============================] - 40s - loss: 0.6510 - acc: 0.6123 - val_loss: 0.6246 - val_acc: 0.6459
    Epoch 2/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.5696 - acc: 0.6951Epoch 00001: val_loss improved from 0.62461 to 0.62176, saving model to ./models/cnn_17.hdf5
    285000/285000 [==============================] - 39s - loss: 0.5696 - acc: 0.6951 - val_loss: 0.6218 - val_acc: 0.6462
    ------------17------CROSS-------
    [[4152 3348]
     [1959 5541]]
                 precision    recall  f1-score   support
    
              0     0.6794    0.5536    0.6101      7500
              1     0.6234    0.7388    0.6762      7500
    
    avg / total     0.6514    0.6462    0.6431     15000
    
    ------------17------TEST-----------
    [[5258 4742]
     [2711 7289]]
                 precision    recall  f1-score   support
    
              0     0.6598    0.5258    0.5852     10000
              1     0.6059    0.7289    0.6617     10000
    
    avg / total     0.6328    0.6273    0.6235     20000
    
    ========Final TEST 17=======
    [[6162 3838]
     [3370 6630]]
                 precision    recall  f1-score   support
    
              0     0.6465    0.6162    0.6310     10000
              1     0.6334    0.6630    0.6478     10000
    
    avg / total     0.6399    0.6396    0.6394     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.6510 - acc: 0.6131Epoch 00000: val_loss improved from inf to 0.62540, saving model to ./models/cnn_18.hdf5
    285000/285000 [==============================] - 40s - loss: 0.6509 - acc: 0.6132 - val_loss: 0.6254 - val_acc: 0.6431
    Epoch 2/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.5710 - acc: 0.6948
    Epoch 00001: reducing learning rate to 0.00010000000474974513.
    Epoch 00001: val_loss did not improve
    285000/285000 [==============================] - 39s - loss: 0.5710 - acc: 0.6948 - val_loss: 0.6254 - val_acc: 0.6462
    ------------18------CROSS-------
    [[4917 2583]
     [2770 4730]]
                 precision    recall  f1-score   support
    
              0     0.6397    0.6556    0.6475      7500
              1     0.6468    0.6307    0.6386      7500
    
    avg / total     0.6432    0.6431    0.6431     15000
    
    ------------18------TEST-----------
    [[6380 3620]
     [3720 6280]]
                 precision    recall  f1-score   support
    
              0     0.6317    0.6380    0.6348     10000
              1     0.6343    0.6280    0.6312     10000
    
    avg / total     0.6330    0.6330    0.6330     20000
    
    ========Final TEST 18=======
    [[6057 3943]
     [3268 6732]]
                 precision    recall  f1-score   support
    
              0     0.6495    0.6057    0.6269     10000
              1     0.6306    0.6732    0.6512     10000
    
    avg / total     0.6401    0.6394    0.6390     20000
    
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 60, 400)           33587200  
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 60, 400)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 56, 64)            128064    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 64)            0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 10, 64)            20544     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 140)               56700     
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 140)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               28200     
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 402       
    _________________________________________________________________
    activation_1 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 33,821,110
    Trainable params: 33,821,110
    Non-trainable params: 0
    _________________________________________________________________
    Train on 285000 samples, validate on 15000 samples
    Epoch 1/2
    284928/285000 [============================>.] - ETA: 0s - loss: 0.6510 - acc: 0.6115Epoch 00000: val_loss improved from inf to 0.62950, saving model to ./models/cnn_19.hdf5
    285000/285000 [==============================] - 40s - loss: 0.6510 - acc: 0.6115 - val_loss: 0.6295 - val_acc: 0.6402
    Epoch 2/2
    284672/285000 [============================>.] - ETA: 0s - loss: 0.5685 - acc: 0.6968Epoch 00001: val_loss improved from 0.62950 to 0.61849, saving model to ./models/cnn_19.hdf5
    285000/285000 [==============================] - 39s - loss: 0.5685 - acc: 0.6968 - val_loss: 0.6185 - val_acc: 0.6502
    ------------19------CROSS-------
    [[4421 3079]
     [2168 5332]]
                 precision    recall  f1-score   support
    
              0     0.6710    0.5895    0.6276      7500
              1     0.6339    0.7109    0.6702      7500
    
    avg / total     0.6524    0.6502    0.6489     15000
    
    ------------19------TEST-----------
    [[5598 4402]
     [3023 6977]]
                 precision    recall  f1-score   support
    
              0     0.6493    0.5598    0.6013     10000
              1     0.6131    0.6977    0.6527     10000
    
    avg / total     0.6312    0.6288    0.6270     20000
    
    ========Final TEST 19=======
    [[6148 3852]
     [3352 6648]]
                 precision    recall  f1-score   support
    
              0     0.6472    0.6148    0.6306     10000
              1     0.6331    0.6648    0.6486     10000
    
    avg / total     0.6402    0.6398    0.6396     20000
    



```python
'''
getGRU
total params: 867,002
test4 final f1 0.6287

========Final TEST 16=======
[[5969 4031]
 [3369 6631]]
             precision    recall  f1-score   support

          0     0.6392    0.5969    0.6173     10000
          1     0.6219    0.6631    0.6419     10000

avg / total     0.6306    0.6300    0.6296     20000


mindf=2


'''
```


```python
'''
#getAvgWordVector
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 60)                0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 60, 20)            825100    
_________________________________________________________________
average_pooling1d_1 (Average (None, 1, 20)             0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 20)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 200)               4200      
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 402       
=================================================================
Total params: 829,702
Trainable params: 829,702
Non-trainable params: 0
_________________________________________________________________
Train on 285000 samples, validate on 15000 samples
========Final TEST 19=======
[[6321 3679]
 [3637 6363]]
             precision    recall  f1-score   support

          0     0.6348    0.6321    0.6334     10000
          1     0.6336    0.6363    0.6350     10000

avg / total     0.6342    0.6342    0.6342     20000
'''
```


```python
'''
getMLP()
(300000, 60)
(15000, 60)
(20000, 60)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 60)                0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 60, 40)            1650200   
_________________________________________________________________
flatten_1 (Flatten)          (None, 2400)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 4802      
_________________________________________________________________
activation_1 (Activation)    (None, 2)                 0         
=================================================================
Total params: 1,655,002
Trainable params: 1,655,002
Non-trainable params: 0
_________________________________________________________________
------------0------CROSS-------
[[4674 2826]
 [2633 4867]]
             precision    recall  f1-score   support

          0     0.6397    0.6232    0.6313      7500
          1     0.6327    0.6489    0.6407      7500

avg / total     0.6362    0.6361    0.6360     15000

------------0------TEST-----------
[[6104 3896]
 [3581 6419]]
             precision    recall  f1-score   support

          0     0.6303    0.6104    0.6202     10000
          1     0.6223    0.6419    0.6319     10000

avg / total     0.6263    0.6261    0.6261     20000

========Final TEST 0=======
[[6104 3896]
 [3581 6419]]
             precision    recall  f1-score   support

          0     0.6303    0.6104    0.6202     10000
          1     0.6223    0.6419    0.6319     10000

avg / total     0.6263    0.6261    0.6261     20000




========Final TEST 10=======
[[6165 3835]
 [3514 6486]]
             precision    recall  f1-score   support

          0     0.6369    0.6165    0.6266     10000
          1     0.6284    0.6486    0.6384     10000

avg / total     0.6327    0.6325    0.6325     20000


========Final TEST 17=======
[[6357 3643]
 [3686 6314]]
             precision    recall  f1-score   support

          0     0.6330    0.6357    0.6343     10000
          1     0.6341    0.6314    0.6328     10000

avg / total     0.6336    0.6335    0.6335     20000
========Final TEST 19=======
[[6291 3709]
 [3636 6364]]
             precision    recall  f1-score   support

          0     0.6337    0.6291    0.6314     10000
          1     0.6318    0.6364    0.6341     10000

avg / total     0.6328    0.6328    0.6327     20000

'''
```


```python
'''
# getCNNIMDB2

(300000, 60)
(500, 60)
(20000, 60)
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 60, 40)            1650200   
_________________________________________________________________
dropout_1 (Dropout)          (None, 60, 40)            0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 56, 64)            12864     
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 14, 64)            0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 14, 64)            0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 10, 64)            20544     
_________________________________________________________________
max_pooling1d_2 (MaxPooling1 (None, 2, 64)             0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 140)               56700     
_________________________________________________________________
dropout_3 (Dropout)          (None, 140)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 200)               28200     
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 402       
_________________________________________________________________
activation_1 (Activation)    (None, 2)                 0         
=================================================================
Total params: 1,768,910
Trainable params: 1,768,910
Non-trainable params: 0
_________________________________________________________________

========Final TEST 19=======
[[5982 4018]
 [3306 6694]]
             precision    recall  f1-score   support

          0     0.6441    0.5982    0.6203     10000
          1     0.6249    0.6694    0.6464     10000

avg / total     0.6345    0.6338    0.6333     20000



mindf=2

precision    recall  f1-score   support

          0     0.6402    0.6177    0.6288     10000
          1     0.6307    0.6529    0.6416     10000

avg / total     0.6355    0.6353    0.6352     20000

'''
```


```python
#capsule net
import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
```


```python
'''
    model = Sequential()
    kernel_size = 5
    filters = 64
    pool_size = 4
    lstm_output_size=70

    model.add(Embedding(input_dim=len(tfidfDict),
                input_length=LengthOfInputSequences,
                output_dim=20,     

               ))
    
    model.add(Dropout(0.25))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
'''


def CapsNetTXT(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)
    em=layers.Embedding(input_dim=len(tfidfDict),
                input_length=LengthOfInputSequences,
                output_dim=20)(x)
    em=layers.Reshape((LengthOfInputSequences,20,1))(em)
    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='tanh', name='conv1')(em)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='tanh', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='tanh'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model

def CapsNetORG(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))

train_model, eval_model, manipulate_model=CapsNetTXT((60,),2,3)
```


```python
 train_model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=[margin_loss, 'mse'],
                  #loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})
    
    
```


```python
trainCnnIndex=np.array(trainCnnIndex)
print(trainCnnIndex.shape)
devCnnIndex=np.array(devCnnIndex)
print(devCnnIndex.shape)
testCnnIndex=np.array(testCnnIndex)
print(testCnnIndex.shape)

cnnTrainy=[0 if s=='DUT' else 1 for s in trainy]
cnnDevy=[0 if s=='DUT' else 1 for s in evaly]
cnnTesty=[0 if s=='DUT' else 1 for s in testy]
cnnTrainyCat=keras.utils.to_categorical(cnnTrainy)
cnnDevyCat= keras.utils.to_categorical(cnnDevy)
import os
os.system('mkdir -p models')
#fullTrainX=np.vstack((trainCnnIndex,devCnnIndex))
#fullTrainY=np.concatenate((cnnTrainyCat,cnnDevyCat))
sf=sklearn.model_selection.StratifiedKFold(20)
cnnmodelCount=0
for trainindex,devindex in sf.split(trainCnnIndex,cnnTrainy):
    K.clear_session()
    train_model, eval_model, manipulate_model=CapsNetTXT((60,),2,3)
    train_model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=[margin_loss, 'mse'],                  
                  metrics={'capsnet': 'accuracy'})
    crf=train_model
    
    modelFile='./models/capsule_%d.hdf5'%(cnnmodelCount)
    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=2,
                               verbose=1,
                               min_delta=0.01,
                               mode='min'),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=0,
                                   verbose=1,
                                   epsilon=0.0001,
                                   mode='min'),
                 ModelCheckpoint(monitor='val_loss',
                                 filepath=modelFile,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 verbose=1,
                                 mode='min'),
                 ]
 
    crf.fit([trainCnnIndex[trainindex],cnnTrainyCat[trainindex]],
            [cnnTrainyCat[trainindex],trainCnnIndex[trainindex]],       
            validation_data=[[trainCnnIndex[devindex],cnnTrainyCat[devindex]],
                              [cnnTrainyCat[devindex],trainCnnIndex[devindex]]],
            epochs=100,batch_size=256,
            callbacks=callbacks,
            )
    crf.load_weights(modelFile)
    devCnnIndex=trainCnnIndex[devindex]
    cnnDevy= np.argmax( cnnTrainyCat[devindex],axis=1)
    predictValue=crf.predict(devCnnIndex)
    
    #predict=[0 if s <th else 1 for s in predictValue]
    predict=np.argmax(predictValue,axis=1)
    print('------------%d------CROSS-------'%(cnnmodelCount))
    print(confusion_matrix(cnnDevy,predict))
    print(classification_report(cnnDevy,predict))
    
    
    #check out the real test
    predictValue=crf.predict(testCnnIndex)
    
    #predict=[0 if s <th else 1 for s in predictValue]
    predict=np.argmax(predictValue,axis=1)
    print('------------%d------TEST-----------'%(cnnmodelCount))
    print(confusion_matrix(cnnTesty,predict))
    print(classification_report(cnnTesty,predict))
    #print(predictValue.shape)
    #fpr, tpr, thresholds = metrics.roc_curve(cnnDevy, predictValue)
    #print(thresholds)
    #print(i,metrics.auc(fpr, tpr))
    cnnmodelCount+=1
```


