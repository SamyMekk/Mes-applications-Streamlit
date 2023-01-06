import streamlit as st
import pandas as pd
import streamlit as st
import sklearn as sk
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader
import requests
from bs4 import BeautifulSoup




st.write('''Récupération des Données sur les Prix des Maisons à Champs-sur-Marne via l'url suivante''')
st.write("Cliquez sur le lien [link] (https://immobilier.lefigaro.fr/annonces/immobilier-vente-maison-champs+sur+marne+77420.html)")







url="https://immobilier.lefigaro.fr/annonces/immobilier-vente-maison-champs+sur+marne+77420.html"
response=requests.get(url)


request=BeautifulSoup(response.content,"html.parser")
prix_maison=request.find_all(class_="price")

price=[]
chambre=[]
pieces=[]
m2=[]
for prix in prix_maison:
    a=prix.text.replace(" ","")
    price.append(int(a.replace("€","")))
    
a=len(request.find_all(class_="options"))

for i in range (0,a):
    if "pièces" in request.find_all(class_="options")[i].text:
        pieces.append(int(request.find_all(class_="options")[i].text[1]))
        if "chambres" in request.find_all(class_="options")[i+1].text:
            chambre.append(int(request.find_all(class_="options")[i+1].text[1]))
        else:
            chambre.append("NaN")
        if "m²" in request.find_all(class_="options")[i+2].text:
            m2.append(float(request.find_all(class_="options")[i+2].text.replace("m²","")))
        else:
            m2.append("NaN")
            


data=pd.DataFrame({"pieces":pieces,"chambre":chambre,"Metres carrés": m2,"Prix":price})
data.drop(data.loc[data['chambre']=='NaN'].index, inplace=True)




st.dataframe(data)


from sklearn.linear_model import LinearRegression # On va effectuer une régression linéaire simple)

model=LinearRegression()
model2=LinearRegression()
Y=np.array(data["Prix"])  # On veut essayer de voir le lien entre la note ESG et les émissions carbone
X1=np.array(data["pieces"])
X2=np.array(data["chambre"])
X3=np.array(data["Metres carrés"]).reshape((-1,1))
X=list(zip(X1,X2,X3))

model2.fit(X3,Y)
model.fit(X,Y)
score=model.score(X,Y) #On affiche le R^2 de la régression
slope=model.intercept_ # On affiche l'intercept
coeff=model.coef_ # On affiche le coef

st.sidebar.header("Les paramètres de la maison à chosiir")

def user_input():
    nbpieces=st.sidebar.number_input("Choississez le nombre de pièces",0,10)
    nbchambres=st.sidebar.number_input("Choississez le nombre de chambres",0,10)
    nbmetrescarres=st.sidebar.number_input("Choissisez le nombre de m²",0,1000)
    data={"Nombre de pièces":nbpieces,
          "Nombre de chambres":nbchambres,
          "Nombre de m²":nbmetrescarres}
    maisonparams=pd.DataFrame(data,index=[0])
    return maisonparams


df=user_input()

st.subheader("Voici les caractéristiques clés de la maison")
st.dataframe(df)

st.subheader("La prédiction du prix de la maison à Champs-sur-Marne  par la méthode de la régression multiple est :")
st.write(int(model.predict(np.array(df))))

a=np.array(df.transpose())[2].reshape(-1,1)
st.subheader("La prédiction du prix de la maison à Champs-sur-Marne par la méthode de la régression simple avec comme régresseur le nombre de m² est ")
st.write(int(model2.predict(np.array(df["Nombre de m²"]).reshape(-1,1))))



st.write(" Nous avons utilisé un modèle de Machine Learning : La Régression Linéaire Multiple")
st.latex(r'''Y_{i}=a_{1}X_{1,i}+a_{2}X_{2,i}+a_{3}X_{3,i}''')
