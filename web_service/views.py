from django.shortcuts import render
import json
import numpy as np
import pandas as pd
from django.http import HttpResponse,Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import pickle
from django.http import StreamingHttpResponse
from keras import backend as K
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
#nlu model
from rasa_nlu.model import Interpreter
import os
from web_service import preprocessing

token_file = os.path.abspath('Model/LSTM_token_glove_300d_DATA_wit_text_processing7.sav')
model_file = os.path.abspath('Model/LSTM_model_glove_300_DATA_with_text_processing_v7.sav')
nlu_file = os.path.abspath('RASA_model/')
data_file = os.path.abspath('DATA/DATA.csv')

num_max = 200

@api_view(["POST"])
def prediction_request(text):
    try:
     
       msg =json.loads(text.body)
       #msg =json.loads(text.body)
       #nlu
       ch=preprocessing.transformText(msg["message"])
       interpreter = Interpreter.load(nlu_file)

       intent=interpreter.parse(msg["message"])
       #
       #loaded_model = pickle.load(open(model_file, 'rb'))
       loaded_model= load_model(model_file)
       token = pickle.load(open(token_file, 'rb'))
       x_input = np.array([ch])
       seq= token.texts_to_sequences(x_input)
       seqs = pad_sequences(seq, maxlen=num_max)
       probability = loaded_model.predict(seqs)
       #predict_proba
       proba=probability[0][0]
       c_pred = loaded_model.predict_classes(seqs)
       class_pred=str(list(np.reshape(np.asarray(c_pred), (1, np.size(c_pred)))[0]))[1:-1]
       x=''
       if class_pred =='0':
          x ='ISSUE'
       if class_pred =='1': 
          x ='NOT_ISSUE'
       
       #x=''
       #if prob > 0.7:
          #x = 'NOT_ISSUE'
       #if prob <= 0.7:
          #x = 'ISSUE'
      
       ### enregistrer le message ds dataset
       df = pd.read_csv(data_file, delimiter=';')
       df.loc[len(df)]=[msg["message"],x]
       df.to_csv(data_file, sep=';', index=False)
       
       return HttpResponse(json.dumps({"id":msg["id"],"message":msg["message"],"label":x,"probability":str(probability),"intent":intent['intent']['name']}), content_type='application/json')
       #return HttpResponse(json.dumps({"id":msg["id"],"message":msg["message"],"label":x,"probability":str(prob)}), content_type='application/json')
       #K.clear_session()
   
    except ValueError as e:
       return Response(e.args[0],status.HTTP_400_BAD_REQUEST)

      