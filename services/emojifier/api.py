from keras.models import model_from_json
import emoji
import pandas as pd
import numpy as np

emoji_dictionary={
    "0":":\u2764\uFE0F:",
    "1":":baseball:",
    "2":":grinning_face_with_big_eyes:",
    "3":":disappointed_face:",
    "4":":fork_and_knife:",
    "5":":hundred_points:",
    "6":":fire:",
    "7":":face_blowing_a_kiss:",
    "8":":chestnut:",
    "9":":flexed_biceps:"
}

with open("services/emojifier/model.json",'r') as file:
    model=model_from_json(file.read())
model.load_weights('services/emojifier/model.h5')

#to solve the post error
model.make_predict_function()

f=open('services/emojifier/glove.6B.50d.txt',encoding='utf-8') 

embedding_index={}
for line in f:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:],dtype='float')
    embedding_index[word]=coefs
f.close()

def embedding_output(X):
    #no.of rnn units=no.of maxlength of sentence
    maxLen=10
    embed_dim=50
    embedding_out=np.zeros((X.shape[0],maxLen,embed_dim))
    
    for i in range(X.shape[0]):
        #we need to iterate over words
        T=X
        T[i]=T[i].split()
        for j in range(len(X[i])):
            try:
                embedding_out[i][j]=embedding_index[T[i][j].lower()]
            except:
                embedding_out[i][j]=np.zeros((50,))
    return embedding_out




def predict(x):
    X=pd.Series([x])
    emb_X=embedding_output(X)
    p=model.predict(emb_X)
    return (emoji.emojize(emoji_dictionary[str(np.argmax(p[0]))]))

if __name__=="__main__":
    print(predict("hello how are you doing?"))