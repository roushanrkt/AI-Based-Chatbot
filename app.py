from flask import Flask, render_template,request
import pickle
import numpy as np
import tensorflow as tf
#from keras.models import load_model
import h5py
model = tf.keras.models.load_model('chatbot_model1.h5')
#model=pickle.load(open('iri.pkl','rb'))

app=Flask(__name__ ,template_folder='template')

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')


@app.route('/get',methods=['GET','POST'])
def get_bot_response():
    data=request.args.get('msg')
    pred=model.predict(data)
    return "Hello Ji"


if __name__=='__main__':
    app.run(debug=True)