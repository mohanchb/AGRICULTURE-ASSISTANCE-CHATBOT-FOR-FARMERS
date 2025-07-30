from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
import cv2
import numpy as np
from numpy import dot
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from django.views.decorators.csrf import csrf_exempt
import os
import speech_recognition as sr
import pandas as pd
import subprocess
from keras.models import model_from_json

plants = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
          'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
          'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
          'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
          'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

recognizer = sr.Recognizer()

question = []
answer = []
global counter
counter = 0

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

def getCount(X,Y):
    count = 0.0
    for i in range(len(X)):
        if X[i] > 0 and Y[i] > 0:
            count = count + 1
    if count > 0:
        count = float(count) / float(len(X))
    return count    


def Upload(request):
    if request.method == 'GET':
       return render(request, 'Upload.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Record(request):
    if request.method == 'GET':
       return render(request, 'Record.html', {})

@csrf_exempt
def record(request):
    if request.method == "POST":
        print("Enter")
        global counter
        audio_data = request.FILES.get('data')
        print(audio_data)
        fs = FileSystemStorage()
        if os.path.exists('ChatBotApp/static/record.wav'):
            os.remove('ChatBotApp/static/record.wav')
        if os.path.exists('ChatBotApp/static/record1.wav'):
            os.remove('ChatBotApp/static/record1.wav')    
        fs.save('ChatBotApp/static/record.wav', audio_data)
        path = 'C:\\Users\\mohan\\AgriChatbot\\ChatBotApp\\static\\'
        res = subprocess.check_output(path+'ffmpeg.exe -i '+path+'record.wav '+path+'record1.wav', shell=True)
        with sr.WavFile('ChatBotApp/static/record1.wav') as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
        except Exception as ex:
            text = "I am not trained to answer given question"    
        if os.path.exists('ChatBotApp/static/record.csv'):
            os.remove('ChatBotApp/static/record.csv')    
        fout = open('ChatBotApp/static/record.csv', 'wb')
        fout.write(text.encode("utf-8"))
        fout.close()
        testArray = []
        with open('ChatBotApp/static/record.csv', "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip("\n").strip()
                testArray.append(line)
        f.close()        
        testStory = tfidf_vectorizer.transform(testArray).toarray()        
        testStory = testStory[0]
        output = "I am not trained to answer given question"
        similarity = 0
        for i in range(len(tfidf)):
            count = getCount(tfidf[i],testStory)
            classify_user = dot(tfidf[i], testStory)/(norm(tfidf[i])*norm(testStory))
            if classify_user > similarity:
                similarity = classify_user
                output = answer[i]
        if output == "I am not trained to answer given question":
            counter = counter + 1
        if counter >= 3:
            output = "Please ask related questions. Request you to get back on track"
            counter = 0        
        return HttpResponse("You: "+text+"\nChatbot: "+output+"\n\n", content_type="text/plain")    

def getChat(query):
    testArray = []
    testArray.append(query.strip().lower())
    testStory = tfidf_vectorizer.transform(testArray).toarray()
    similarity = 0
    user_story = "I am not trained to answer given question"
    testStory = testStory[0]
    for i in range(len(tfidf)):
        classify = dot(tfidf[i], testStory)/(norm(tfidf[i])*norm(testStory))
        count = getCount(tfidf[i],testStory)
        if classify > similarity:
            similarity = classify
            user_story = answer[i]
    return user_story

def ChatData(request):
    if request.method == 'GET':
        global counter
        query = request.GET.get('mytext', False)
        query = query.strip("\n").strip()
        output = getChat(query)
        if output == "I am not trained to answer given question":
            counter = counter + 1
        if counter >= 3:
            output = "Please ask related questions. Request you to get back on track"
            counter = 0
        return HttpResponse(output, content_type="text/plain")

def UploadAction(request):
    if request.method == 'POST':
        myfile = request.FILES['t1']
        fname = request.FILES['t1'].name
        fs = FileSystemStorage()
        if os.path.exists('ChatBotApp/static/plant/test.png'):
            os.remove('ChatBotApp/static/plant/test.png')
        filename = fs.save('ChatBotApp/static/plant/test.png', myfile)
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()

        img = cv2.imread('ChatBotApp/static/plant/test.png')
        img = cv2.resize(img, (64,64))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,64,64,3)
        test = np.asarray(im2arr)
        test = test.astype('float32')
        test = test/255
        preds = classifier.predict(test)
        predict = np.argmax(preds)
        print(str(predict)+" "+plants[predict])
        img = cv2.imread('ChatBotApp/static/plant/test.png')
        img = cv2.resize(img,(650,450))
        chat_msg = getChat(plants[predict])
        cv2.putText(img, 'Crop Disease Predicted as '+plants[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
        cv2.imshow('Crop Disease Predicted as '+plants[predict],img)
        cv2.waitKey(0)
        context= {'data':"Crop Disease Predicted as "+plants[predict]+"<br/>Possible Remedy = "+chat_msg}
        return render(request, 'Chat.html', context) 
        






    

