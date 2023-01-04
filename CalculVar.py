import pandas as pd
import streamlit as st
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader

Apple=DataReader('AAPL','yahoo','2005-01-01')
MMM=DataReader('MMM','yahoo','2005-01-01')
AMD=DataReader('AMD','yahoo','2005-01-01')
AMZN=DataReader('AMD','yahoo','2005-01-01')

st.title('''
    # Calcul de la Value-at-Risk et de l'Expected Shortfall''')

st.markdown('''
         n°1: Approche Historique
         ''')

st.sidebar.header("Les paramètres d'entrée")





Apple['log_returns']= np.log(Apple['Adj Close']) - np.log(Apple['Adj Close'].shift(1))
MMM['log_returns']= np.log(MMM['Adj Close']) - np.log(MMM['Adj Close'].shift(1))
AMD['log_returns']= np.log(AMD['Adj Close']) - np.log(AMD['Adj Close'].shift(1))
AMZN['log_returns']= np.log(AMZN['Adj Close']) - np.log(AMZN['Adj Close'].shift(1))

AppleAdj=Apple[["Adj Close"]]
MMMAdj=MMM[["Adj Close"]]
AMDadj=AMD[["Adj Close"]]
AMZNadj=AMZN[["Adj Close"]]

Test1=pd.merge(AppleAdj, MMMAdj, left_index = True, right_index = True)
Test2=pd.merge(Test1, AMDadj, left_index = True, right_index = True)
Test3=pd.merge(Test2, AMZNadj, left_index = True, right_index = True)

Test3.columns=[["AdjCloseApple","AdjCloseMMM","AdjCloseAMD","AdjCloseAMZN"]]


def poids_portefeuille():
    weights=np.array(np.random.random(4))
    return weights/np.sum(weights)
a=poids_portefeuille()

Test3["Value_Portfolio"]=a[0]*Test3["AdjCloseApple"].values+a[1]*Test3["AdjCloseMMM"].values+a[2]*Test3["AdjCloseAMD"].values+a[3]*Test3["AdjCloseAMZN"].values


st.subheader("Evoluton Valeur Portefeuille")

def user_input():
    seuil_confiance=st.sidebar.selectbox('Seuil de Confiance',[90,95,97.5,99])
    horizon=st.sidebar.slider('Horizon',1,1000,2)   
    data={'seuil_confiance':seuil_confiance,
          'Horizon':horizon,
          'Poids associé au titre Apple':a[0],
          'Poids associé au titre MMM':a[1],
          'Poids associé au titre AMD':a[2]     ,
          'Poids associé au titre Amazon':a[3],
          }
    
    ParamètresPortefeuille=pd.DataFrame(data,index=[0])
    return ParamètresPortefeuille


# Calcul Var Historique
AppleVar=[]
MMMVar=[]
AMDVar=[]
AMZNVar=[]
A=pd.DataFrame
for i in range (len(Test3[4451-500:])):
    AppleVar.append(Test3.tail(1)["AdjCloseApple"].values[0][0]*(Test3["AdjCloseApple"].iloc[i+1,:][0]/Test3["AdjCloseApple"].iloc[i,:][0]))
    MMMVar.append(Test3.tail(1)["AdjCloseMMM"].values[0][0]*(Test3["AdjCloseMMM"].iloc[i+1,:][0]/Test3["AdjCloseMMM"].iloc[i,:][0]))
    AMDVar.append(Test3.tail(1)["AdjCloseAMD"].values[0][0]*(Test3["AdjCloseAMD"].iloc[i+1,:][0]/Test3["AdjCloseAMD"].iloc[i,:][0]))
    AMZNVar.append(Test3.tail(1)["AdjCloseAMZN"].values[0][0]*(Test3["AdjCloseAMZN"].iloc[i+1,:][0]/Test3["AdjCloseAMZN"].iloc[i,:][0]))
    
    
data={'AppleVar':AppleVar,'MMMVar':MMMVar,'AMDVar':AMDVar,'AMZNVar':AMZNVar}


Scenario=pd.DataFrame(data)
Test3["Valeur Portefeuille"]=a[0]*Test3["AdjCloseApple"].values+a[1]*Test3["AdjCloseMMM"].values+a[2]*Test3["AdjCloseAMD"].values+a[3]*Test3["AdjCloseAMZN"].values
Scenario["Valeur Portefeuille"]=a[0]*Scenario["AppleVar"].values+a[1]*Scenario["MMMVar"].values+a[2]*Scenario["AMDVar"].values+a[3]*Scenario["AMZNVar"].values

b=Test3["Valeur Portefeuille"].tail(1).values[0][0]
Scenario.index.name="Scenarios"


Scenario["Pertes"]=b-Scenario["Valeur Portefeuille"]


st.subheader("Choix des Paramètres")

df=user_input()

st.write(Scenario)

st.write(df)


st.write(Test3["Value_Portfolio"])



