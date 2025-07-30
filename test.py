import pandas as pd
import numpy as np

question = []
answer = []

dataset = pd.read_csv("Messages/diseases.txt", encoding='iso-8859-1',header=None)
dataset = dataset.values
for i in range(len(dataset)):
    disease = dataset[i,0]
    remedy = dataset[i,1]
    question.append(disease.strip().lower())
    answer.append(remedy)

dataset = pd.read_csv("Messages/chatbot.csv",encoding='iso-8859-1')
dataset = dataset.values
for i in range(len(dataset)):
    crop = dataset[i,0]
    rainfall = str(dataset[i,1])
    soil = str(dataset[i,2])
    irrigation = str(dataset[i,3])
    question.append(crop.strip().lower())
    answer.append(rainfall+" "+soil+" "+irrigation)

tfidf_vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
tfidf = tfidf_vectorizer.fit_transform(question).toarray()        

print(tfidf)
