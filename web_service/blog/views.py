from django.http import HttpResponse
from django.shortcuts import render
from .forms import ContactForm
#import numpy as np # linear algebra
from rest_framework.views import APIView
from rest_framework.decorators import api_view
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

#from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#nlu model
from rasa_nlu.model import Interpreter
#from rasa_nlu.training_data import load_data
#from rasa_nlu.config import RasaNLUModelConfig
#from rasa_nlu.model import Trainer
#from rasa_nlu import config


def home(request):
    """ Exemple de page non valide au niveau HTML pour que l'exemple soit concis """
    return HttpResponse("""
        <h1>Bienvenue sur mon blog !</h1>
        <p>Les crêpes bretonnes ça tue des mouettes en plein vol !</p>
    """)


def contact(request):
    # Construire le formulaire, soit avec les données postées,
    # soit vide si l'utilisateur accède pour la première fois
    # à la page.
    form = ContactForm(request.POST or None)
    # Nous vérifions que les données envoyées sont valides
    # Cette méthode renvoie False s'il n'y a pas de données 
    # dans le formulaire ou qu'il contient des erreurs.
    if form.is_valid(): 
        # Ici nous pouvons traiter les données du formulaire
        sujet = form.cleaned_data['sujet']
        message = form.cleaned_data['message']
        envoyeur = form.cleaned_data['envoyeur']
        renvoi = form.cleaned_data['renvoi']

        # Nous pourrions ici envoyer l'e-mail grâce aux données 
        # que nous venons de récupérer
        envoi = True
    
    # Quoiqu'il arrive, on affiche la page du formulaire.
    return render(request, 'blog/contact.html', locals())

#def predict(request):
 	#loaded_model = pickle.load(open(filename, 'rb'))
 	#prediction code
 	#text = np.array(['bad movie'])
	#sequences = tok.texts_to_sequences(text)
	#print(sequences)
	#sequences_matrix = sequence.pad_sequences(sequences,maxlen=num_max)
	#prediction = loaded_model.predict(sequences_matrix)
 	#return HttpResponse("la prédiction est{0}".format(prediction))
def predjson(request):
	
	#convert json data that we get back from a web service call  into a python datastructure
	json_data = '{"hello": "world", "foo": "bar"}' 

	#json.loads  loads a string of json and converts the data to a python dict
	#data = json.loads(json_data)


#



#HttpResponse with JSON
def fbview(request):
    msg_response = {'msg_id': '1', 'msg_text': 'message response' , 'msg_label': '' }
    #Dict to JSON :  json.dumps(data)
    return HttpResponse(json.dumps(msg_response), content_type='application/json')




file="C:/Users/khmar/model_CNN.sav"
file1="C:/Users/khmar/token.sav"
#file="C:/Users/khmar/LSTM_Normal_Embedding_Methode_with_text_processing_spelling.sav"
#file1="C:/Users/khmar/token_LSTM_Normal_Embedding_Methode_DATA_with_text_processing_spelling.sav"

num_max = 1000
#num_max = 2000

@api_view(["POST"])
def prediction_request(text):
    try:
   	 
       msg =json.loads(text.body)
       #nlu
       interpreter = Interpreter.load("C:/Users/khmar/RASAAAA/My_project/models/nlu/default/current")
       intent=interpreter.parse(msg["message"])
       #
       loaded_model = pickle.load(open(file, 'rb'))
       from keras.models import load_model
       #loaded_model= load_model(file)
       
       token = pickle.load(open(file1, 'rb'))
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
       #if prob > 0.3:
          #x = 'NOT_ISSUE'
       #if prob <= 0.3:
          #x = 'ISSUE'
       #pour rafrechir 
       ### enregistrer le message ds dataset
       df = pd.read_csv('C:/Users/khmar/Desktop/ISSUE/dataset/CSV/DATA.csv', delimiter=';')
       df.loc[len(df)]=[msg["message"],x]
       df.to_csv('C:/Users/khmar/Desktop/ISSUE/dataset/CSV/DATA.csv', sep=';', index=False)
       K.clear_session()
       return HttpResponse(json.dumps({"id":msg["id"],"message":msg["message"],"label":x,"probability":str(prob),"intent":intent['intent']['name']}), content_type='application/json')
       #return HttpResponse(json.dumps({"id":msg["id"],"message":msg["message"],"label":x,"probability":str(prob)}), content_type='application/json')
    except ValueError as e:
       return Response(e.args[0],status.HTTP_400_BAD_REQUEST)

      