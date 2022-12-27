# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:58:55 2022

@author: jerom
"""
from word2number import w2n
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

#on charge le fichier CarPrice
data = pd.read_csv("CarPrice.csv")

#transforme les nombres ecris en toute lettre en nombre
data["cylindernumber"] = data["cylindernumber"].apply(lambda x : w2n.word_to_num(x))
#print(data["cylindernumber"])
print(data)

#on veut expliquer le prix par rapport aux caractéristiques d'une voiture donc
#on va créer deux sous dataframe : l'un avec les caractéristiques, l'autre avec
#le prix 
DataPrice = pd.DataFrame(data["price"])
#le nom et l'id de la voiture ne sont pas repris car ils n'influencent pas dans le prix
caract = ["carlength","carwidth","carheight","cylindernumber","horsepower",
          "citympg","highwaympg"]
DataCaract = pd.DataFrame(data[caract])
#print(DataCaract)
Price = DataPrice["price"]
Caract = DataCaract[caract]

#print(Caract.loc[2])

#Méthode Régression
#on initialise le modèle
regression_model = LinearRegression()
regression_model.fit(Caract, Price)
#On calcule R^2 et plus il est proche de 1, plus le modele est bien
Score = regression_model.score(Caract, Price)
#print(regression_model.predict(Caract))
#Calcule du score
print("Score : ", Score)
#Calcul du RSME
#RMSE = np.sqrt((sum((Price-regression_model.predict(Caract))**2))/(len(Price)))
RMSE = np.sqrt(((Price-regression_model.predict(DataCaract))**2).sum()/len(Price))
#RMSE = np.sqrt(((Price-regression_model.predict(Caract))**2).sum()/len(Price))
print("RMSE : ", RMSE)
print("intercept : ", regression_model.intercept_)
print("coefficient : ", regression_model.coef_)


print("-----------------------------------\n\n\n\n Méthode anova\n")
#Méthode anova
model = smf.ols('price ~ carlength+carwidth+carheight+cylindernumber+horsepower+citympg+highwaympg', 
                data = data).fit()

#test de Rainbow (linearité du model)
print("vérifions si le modèle est bien linéaire : ")
from statsmodels.stats.diagnostic import linear_rainbow
Ftest, pval = linear_rainbow(model)
if pval >= 0.05:
    print("la p-valeur du test de Rainbow est : ", pval, " >= 0.05"
          , "\ndonc le model est bien linéaire\n")
#Test de Durbin-Watson (indépendance des résidus)
from statsmodels.stats.stattools import durbin_watson
D = durbin_watson(model.resid)#Price-regression_model.predict(DataCaract))
print("Test de durbin watson : ", D, "\nprobleme au niveau de l'independance",
      "il doit etre proche de 2")
"""voir ce qu'on fait car D doit etre proche proche de 2 pour ne pas avoir de 
corrélation entre les variables"""

#test de jarque-berra (normalité des résidus)
print("\nTestons la normalité des résidus en utilisant le test de Shapiro-wilk ou"
      ," de Jarque-bera")
from scipy.stats import jarque_bera, shapiro
residues = Price-regression_model.predict(DataCaract)
#residues_std = residues/np.sqrt(sum(residues**2)/(len(residues)-1))
stat1, pval = jarque_bera(residues)
stat, ppval = shapiro(residues)
print("la p-valeur du test de jarque bera est : ", ppval , "<= 0.05 "
      ,"\n donc les résidus ne suivent pas une loi normale\n")

#test de Breusch-pagan (test d'homoscédasticité des résidus)
from statsmodels.stats.api import het_breuschpagan
lagrange, pval, fval, pfval = het_breuschpagan(residues,model.model.exog, robust = False)
print("la p valeur du test de Breusch-pagan est : ", pval, " <=0.05 \n Donc"
      , " les résidus n'ont pas la même variance")


#print(model.summary())

