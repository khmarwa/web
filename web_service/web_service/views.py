from django.shortcuts import render
import json
import numpy as np
import pandas as pd
#from keras.preprocessing import sequence
from django.http import HttpResponse,Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import pickle
from django.http import StreamingHttpResponse
from keras import backend as K
import keras
#from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

#nlu model
from rasa_nlu.model import Interpreter

#file="C:/Users/khmar/model_CNN.sav"
#file1="C:/Users/khmar/token.sav"
model_file= "C:/Users/khmar/git_repo/IssueModelTraining/CNN/without_text_preprocessing/Model/CNN_model3_normal_embeddings_DATA_with_text_processing.sav"
token_file= "C:/Users/khmar/git_repo/IssueModelTraining/CNN/without_text_preprocessing/Model/CNN_token_normal_embeddings_without_text_preprocessing_DATA.sav"
#########################################################################
#file= "C:/Users/khmar/git_repo/IssueModelTraining/LSTM/with_text_processing/Model/LSTM_model_glove_200_DATA_with_text_processing.sav"
#file1= "C:/Users/khmar/git_repo/IssueModelTraining/LSTM/with_text_processing/Model/LSTM_token_glove_200d_DATA_with_text_processing.sav"
#num_max = 200
################################################""

num_max = 1000


@api_view(["POST"])
def prediction_request(text):
    try:
   	 
       msg =json.loads(text.body)
       #nlu
       interpreter = Interpreter.load("C:/Users/khmar/git_repo/IssueModelTraining/RASA/models/nlu/default/current")

       intent=interpreter.parse(msg["message"])
       #
       #loaded_model = pickle.load(open(model_file, 'rb'))
       loaded_model= load_model(model_file)

       token = pickle.load(open(token_file, 'rb'))
       x_input = np.array([msg["message"]])
       seq= token.texts_to_sequences(x_input)
       seqs = pad_sequences(seq, maxlen=num_max)
       probability = loaded_model.predict(seqs)
       prob=probability[0][0]
       c_pred = loaded_model.predict_classes(seqs)
       class_pred=str(list(np.reshape(np.asarray(c_pred), (1, np.size(c_pred)))[0]))[1:-1]
       x=''
       if class_pred =='0':
          x ='ISSUE'
       if class_pred =='1': 
          x ='NOT_ISSUE'
       #classe=prediction_classe(class_pred)
       #classe=prediction_classe(class_pred)
       #yhat = m.predict(seqs)
       #prob=probability[0][0]

       #x=''
       #if prob > 0.7:
          #x = 'NOT_ISSUE'
       #if prob <= 0.7:
          #x = 'ISSUE'
       #pour rafrechir 
       ### enregistrer le message ds dataset
       data_file='C:/Users/khmar/git_repo/IssueModelTraining/DATA/DATA.csv'
       df = pd.read_csv(data_file, delimiter=';')
       df.loc[len(df)]=[msg["message"],x]
       df.to_csv(data_file, sep=';', index=False)
       K.clear_session()
       return HttpResponse(json.dumps({"id":msg["id"],"message":msg["message"],"label":x,"probability":str(prob),"intent":intent['intent']['name']}), content_type='application/json')
       #return HttpResponse(json.dumps({"id":msg["id"],"message":msg["message"],"label":x,"probability":str(prob)}), content_type='application/json')
    except ValueError as e:
       return Response(e.args[0],status.HTTP_400_BAD_REQUEST)

      