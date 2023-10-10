import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.corpus import stopwords 
import nltk 
nltk.download('stopwords')

from sklearn.pipeline import Pipeline 

from sklearn.naive_bayes import BernoulliNB , MultinomialNB , GaussianNB 

from sklearn.metrics import accuracy_score 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



filepath = '/kaggle/input/sms-spam-collection-dataset/spam.csv'
data_import = pd.read_csv(filepath , encoding = 'ISO-8859-1')
data_import.head()
df =  data_import.drop(['Unnamed: 2' , 'Unnamed: 3' , 'Unnamed: 4'] , axis = 1)
df.head()
sw = stopwords.words('english')

def stopword(text) : 
    txt = [word.lower() for word in text.split() if word.lower() not in sw]
    return txt 

df['v2'] = df['v2'].apply(stopword)

df.head()

from nltk.stem.snowball import SnowballStemmer 

ss = SnowballStemmer("english")

def stemming(text) : 
    text = [ss.stem(word) for word in text if word.split()]
    return "".join(text)

df['v2'] = df['v2'].apply(stemming)


df.head()

from sklearn.feature_extraction.text import TfidfVectorizer 

tfid_vect = TfidfVectorizer()

tfid_matrix = tfid_vect.fit_transform(df['v2'])

print(f"Type :{type(tfid_matrix)} , Matrix at 0 : {tfid_matrix[0]} , Shape : {tfid_matrix.shape}")

array = tfid_matrix.todense()

df1 = pd.DataFrame(array)
df1[df1[10]  != 0].head()

df1['v1'] = df['v1']

df1.head()

from sklearn.model_selection import train_test_split 

features = df1.drop('v1' , axis = 1)
label = df1['v1']

x_train , x_test , y_train , y_test = train_test_split(features , label , test_size = 0.3)
print(f"X train shape : {x_train.shape}\nY train shape : {y_train.shape}\nX test shape : {x_test.shape}\nY test shape : {y_test.shape}")



ber_pipe = Pipeline(steps = [
   ( 'ber_model' , BernoulliNB())
])

multi_pipe = Pipeline(steps = [
    ('multi_model' , MultinomialNB())
])

guass_pipe = Pipeline(steps = [
    ('guass_model' , GaussianNB())
])

def model_evaluation(model) : 
    model.fit(x_train , y_train)
    y_pred_model = model.predict(x_test)
    
    acc_score = accuracy_score(y_test , y_pred_model)
    
    print(f"Accuracy Score of {model[0]} : {acc_score}")
    


model_evaluation(ber_pipe)
model_evaluation(multi_pipe)
model_evaluation(guass_pipe)
