# -*- coding: utf-8 -*-
"""
Created on Mon May 24 00:42:45 2021

@author: Acer
"""


import tensorflow as tf
from tensorflow import keras
import nltk
nltk.download('punkt')      #word corpora for
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

count=0
words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intent_v1.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
            
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))



# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
import pickle
pickle.dump(hist,open('bot_model.pkl','wb'))
print("model created")


"""

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intent_v1.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))



def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.01
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

import enchant
message="rkt"
def funcCall(message):
    dict=enchant.DictWithPWL("en_US","MyDict.txt")  #adding our own word list aong with en_US dictionary words 
    d=enchant.request_pwl_dict("MyDict.txt")
    #while message!='quit':
        #message=input()
    word_list=message.split()
    mispelled=[]
    out=chatbot_response(message)
    response="Please rephrase, ask any other question or please choose one of the options below according to your requirement:"
    if out==response:
        for word in word_list:
            if dict.check(word)==False:
                mispelled.append(word)
        cnt=0
        for var in mispelled:
            suggested_word=dict.suggest(var)
            for word in suggested_word:
                if d.check(word)==True:
                    cnt=cnt+1
        if cnt>0:
            str=" Please either rephrase your sentence or take help from suggested words below to avoid errors:"
            for word in mispelled:
                suggestion_inDict=dict.suggest(word)
                for sugWord in suggestion_inDict:
                    if d.check(sugWord)==True:
                        str+=" Suggestion for "+ word + " : " + sugWord #str(dict.suggest(word)))
            return str
        else:
            #if message=="quit": break
            #print(response)
            return response
    else:
        #out=chatbot_response(message)
        #if message=="quit":
        #    break    
        #out=chatbot_response(message)
        #print("You: "+ message+"\n\n")
        #print("Bot:  "+out+"\n\n")
        return out

#function call triggering everythings
#message="rkt"
#funcCall(message)
 # else:
 #    print("Please rephrase your querry I am not able to recognize that ")
    """
 
""" if d.check(message):    
      out=chatbot_response(message)
  else:
      lst=d.suggest(message)
      print("Please either rephrase your sentence or take help from suggested words below: ")
      print(lst)
      message=input()
      out=chatbot_response(message)
  if message=="quit":
      break    
  print("You: "+ message+"\n\n")
  print("Bot:  "+out+"\n\n")"""

"""


#creation of interface for the user using python and using the model to create the response
#Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    
    if msg != '':
        ChatLog.config(state=NORMAL)
        
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = funcCall(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

    #Below functions are specifically developed for recommendation part

def Mision():
    
    msg="Culture of Collaborations and innovation driven development Encouragement to Interdisciplinary Research Knowledge sharing from eleven independent C-DAC Centres with over three decades of experience  Progress with Future Mind set and Emotionally Agile Attitude Rise as Radical Optimist and Idea Connector Orbit shifting from “contributor in innovation” to “owner of Innovation” Following is the link to be followed for the detail info about our Mission and Vision:"
    ChatLog.config(state=NORMAL)
    #ChatLog.insert(END, "You: " + msg + '\n\n')
    ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
   
    res = "https://www.cdac.in/index.aspx?id=patna"  #chatbot_response(msg)
  
    ChatLog.insert(END, "Bot:    " +msg+" "+ res + '\n\n')
    #SendButton1.place(x=30, y=30, height=15)
    ChatLog.insert(SendButton1)
    ChatLog.config(state=DISABLED)
    ChatLog.yview(END)

def TechLandscape(): 
    msg="PARAM Budhha is a super computer which will be facilitated to individuals doing their PHD programme,StartUp,Research Institute and Individuals . For more details about the current status and how you can use it please follow this link https://www.cdac.in/index.aspx?id=hpc_ss_param_shavak"
    ChatLog.config(state=NORMAL)
    #ChatLog.insert(END, "You: " + msg + '\n\n')
    ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
   
    res = "https://www.cdac.in/index.aspx?id=products_services"  #chatbot_response(msg)
    ChatLog.insert(END, "Bot: " +msg+" "+ res + '\n\n')
    ChatLog.config(state=DISABLED)
    ChatLog.yview(END)

def Philosophy(): 
    msg="Culture of Collaborations and innovation driven development Encouragement to Interdisciplinary Research Knowledge sharing from eleven independent C-DAC Centres with over three decades of experience  Progress with Future Mind set and Emotionally Agile Attitude Rise as Radical Optimist and Idea Connector Orbit shifting from “contributor in innovation” to “owner of Innovation Following is the link to be followed for the detail info about the philosophy at cdac patna"
    ChatLog.config(state=NORMAL)
    #ChatLog.insert(END, "You: " + msg + '\n\n')
    ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
   
    res = "https://www.cdac.in/index.aspx?id=patna"  #chatbot_response(msg)
    ChatLog.insert(END, "Bot: "+msg+" " + res + '\n\n')
    ChatLog.config(state=DISABLED)
    ChatLog.yview(END)

def FAQ(): 
    msg=""""""Please type the options for FAQ
   'option1': Who can use our facility                  
   'option2': What Sector/Domain of AI, our                    facility will empower you                                
   'option3': What things you could do
   'option4': Configuration of our Supercomp                   uter"""
           
""" ChatLog.config(state=NORMAL)
   # ChatLog.insert(END, "You: " + msg + '\n\n')
    ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
   
    #res = "https://www.cdac.in/index.aspx?id=patna"  #chatbot_response(msg)
    ChatLog.insert(END, "Bot: "+msg+" " + '\n\n')
    ChatLog.config(state=DISABLED)
    ChatLog.yview(END)
     
     
 

    

base = Tk()
base.title("Emilly The Chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="wheat", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message

SendButton1 = Button(base, font=("Verdana",10,'bold'), text="Mision", width="6", height=12,
                    bg="#8B0000", activebackground="white",fg='#ffffff',
                    command= Mision)
SendButton2 = Button(base, font=("Verdana",10,'bold'), text="Supercomputer", width="6", height=12,
                    bg="#8B0000", activebackground="white",fg='#ffffff',
                    command= TechLandscape)
SendButton3 = Button(base, font=("Verdana",10,'bold'), text="HR&Philosophy", width="6", height=12,
                     bg="#8B0000", activebackground="white",fg='#ffffff',
                    command= Philosophy)
SendButton4 = Button(base, font=("Verdana",10,'bold'), text="FAQ", width="6", height=12,
                     bg="#8B0000", activebackground="white",fg='#ffffff',
                    command= FAQ)

SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="10", height=5,
                     bd=0,bg="#00ff00", activebackground="white",fg='#ffffff',
                    command= send )




#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
SendButton4.place(x=7, y=375, height=15,width=50)
SendButton2.place(x=60, y=375, height=15,width=120)
SendButton3.place(x=183, y=375, height=15,width=114)
SendButton1.place(x=300, y=375, height=15)


base.mainloop()"""