from fastapi import FastAPI , Body
import uvicorn
import requests
import pickle
import pandas as pd
from fastapi import status
import sklearn
from pydantic import BaseModel
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import  AutoModelForSeq2SeqLM


app = FastAPI()

# app.add_middleware(
#     TrustedHostMiddleware, allowed_hosts=["*"]
# )

origins = [
    
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#generator of text
tokenizer4 = GPT2Tokenizer.from_pretrained('gpt2-large')
model4 = GPT2LMHeadModel.from_pretrained("gpt2-large",pad_token_id=tokenizer4.eos_token_id)

#sentiment analysis
tokenizer2 =AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model2 = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# summarization of text
tokenizer3 = AutoTokenizer.from_pretrained("google/pegasus-xsum")

model3 = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")


if __name__ == '__main__':
    uvicorn.run("main:app",port=8000)

class Texto(BaseModel):
    texto : str | None = None

class Carro(BaseModel):
    year: int
    transmision: int | None = None
    fuel: int
    marca: int | None = None
    kms: int
    motor: int

cit='Bogota'
#url_r = 'http://api.openweathermap.org/data/2.5/solar_radiation?lat={lat}&lon={lon}&appid={API key}'
model = pickle.load(open('clima.pkl','rb'))

@app.get(
        path="/clima/{city}",
        status_code=status.HTTP_200_OK
)
def clima(city):
    city = str(city)
    url = "http://api.openweathermap.org/data/2.5/weather?q={}&appid={}&units=metrics".format(city,"23eb7726a5115d6412fb7dba45cbe828")
    
    res = requests.get(url)
    data =  res.json()
    
    actual = data['weather'][0]['main']
    humedad = data['main']['humidity']
    prep = float(data['main']['feels_like']) - 273.15
    min_temp = float(data['main']['temp_min']) - 273.15
    max_temp = float(data['main']['temp_max']) - 273.15
    viento = float(data['wind']['speed']) 
    latitud = data['coord']['lat']
    longitud=data['coord']['lon']
    temp = float(data['main']['temp']) - 273.15
    #rad(latitud,longitud)
    response = model.predict(pd.DataFrame([[prep,max_temp, min_temp,viento]], columns=['precipitation', 'temp_max','temp_min','wind']))
    resp =  int(response[0])
    
    return {'prediccion':resp, 'temperatura': temp,'viento':viento,'latitud':latitud,'longitud':longitud,'clima':actual,'humedad':humedad}






model1 = pickle.load(open('cars_model_car.pkl','rb'))
@app.post(
        path="/precio/",
        status_code=status.HTTP_200_OK

       
)
def precioPred(carro : Carro):
    
    request_data = carro.dict()
    
    year = request_data['year']
    fuel = request_data['fuel']
    kms= request_data['kms']
    transmision = request_data['transmision']
    motor = request_data['motor']
    marca = request_data['marca']
    
    # year = request.form.get('year')
    # fuel_type = request.form.get('fuel_type')
    # kms_driven = int(request.form.get('kilo_driven'))
    # present = float(request.form.get('present'))
    # transmision = request.form.get('transmision')
    
    predicion = model1.predict(pd.DataFrame([[year,transmision,fuel,marca,kms,motor]], columns=['year','transmision','fuel','marca','kms','motor']))
    # predicion = model.predict(pd.DataFrame([[2022	,1	,0	,1	,5980	,2000]], columns=['year','transmision','fuel','marca','kms','motor']))
    
    return int(predicion[0])



@app.post(path="/texto/")
def generator_text(texto : Texto):
    request_data = texto.dict()
    sentence = str(request_data["texto"])
    input_ids = tokenizer4.encode(sentence, return_tensors='pt')
    output = model4.generate(input_ids, max_length = 100, num_beams=5, no_repeat_ngram_size=2,early_stopping=True)
    text = tokenizer4.decode(output[0],skip_special_tokens=True)
    return text


#1 es malo y 5 es bueno
@app.post(path="/sentimiento/")
def sentiment(texto : str):
    request_data = texto.dict()
    sentence = str(request_data["texto"])
    tokens = tokenizer2.encode(sentence, return_tensors='pt')
    result = model2(tokens)
    return int(torch.argmax(result.logits))+1





@app.post(path="/resumen/")
def summarization_text(texto: Texto):
   
    request_data = texto.dict()
    sentence = str(request_data["texto"])
   # return sentence
    input_ids = tokenizer3.encode(sentence, return_tensors='pt')
    output = model3.generate(input_ids, max_length = 100, num_beams=5, no_repeat_ngram_size=2,early_stopping=True)
    text = tokenizer3.decode(output[0],skip_special_tokens=True)
    return text



