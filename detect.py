import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
import warnings
warnings.simplefilter("ignore")
dataPure=pd.read_csv("Language Detection.csv",encoding="utf-8")
dataPure=dataPure.sample(frac=1).reset_index(drop=True)
MAX=67
mask=[]
for i in range(len(dataPure.iloc[::])):
    if (len(dataPure.Text.iloc[i].split())<=67):
        mask.append(True)
    else:
        mask.append(False)
datax=list(map(lambda x:x.split(),dataPure.Text[mask]))
datay=dataPure.Language[mask]
labels=datay.unique()
datay=tf.keras.utils.to_categorical(LabelEncoder().fit_transform(datay))
tokenizer=tf.keras.preprocessing.text.Tokenizer(50000)
tokenizer.fit_on_texts(datax)
datax=tokenizer.texts_to_sequences(datax)
datax=tf.keras.utils.pad_sequences(datax,MAX)
model=tf.keras.Sequential([  
    tf.keras.layers.Embedding(50000,50,input_length=MAX),
    tf.keras.layers.GRU(128,return_sequences=True),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(17),
    tf.keras.layers.Softmax()
])
inp=input("Write Something: ")
model.compile(optimizer=tf.keras.optimizers.Adam(),metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Recall()],loss=tf.keras.losses.CategoricalCrossentropy())
model.fit(datax,datay,batch_size=128,epochs=20)
index=(tf.argmax(model.predict(tf.keras.utils.pad_sequences(tokenizer.texts_to_sequences([inp.split()]),MAX),batch_size=1),1))
print(sorted(labels)[int(index.numpy())])
