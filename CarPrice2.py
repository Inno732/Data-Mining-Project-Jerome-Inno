# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 19:59:29 2023

@author: jerom
"""

import scipy.stats as stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from word2number import w2n
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#on charge le fichier CarPrice
data = pd.read_csv("CarPrice.csv")

#transforme les nombres ecris en toute lettre en nombre
data["cylindernumber"] = data["cylindernumber"].apply(lambda x : w2n.word_to_num(x))
print(data)

#on veut expliquer le prix par rapport aux caractéristiques d'une voiture donc
#on va créer deux sous dataframe : l'un avec les caractéristiques, l'autre avec
#le prix 
DataPrice = pd.DataFrame(data["price"])
#le nom et l'id de la voiture ne sont pas repris car ils n'influencent pas dans le prix
#d'une voirure
caract = ["carlength","carwidth","carheight","cylindernumber","horsepower",
          "citympg","highwaympg"]
DataCaract = pd.DataFrame(data[caract])


#Séparons les données en données d'entrainement et de tests
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(DataCaract, DataPrice, test_size=0.1, random_state=42)
#Probleme : si on selectionne un mauvais random_State, R^2 peut etre tres mauvais


#Regression
#on initialise le modèle
RLM = LinearRegression()
RLM.fit(DataCaract, DataPrice)
#Y = a+bx + .. + b_kx_k

coef = RLM.coef_[0]
print(coef)

Residu = DataPrice-RLM.predict(DataCaract)

#Calcule de R^2, il doit etre proche de 1
Score = RLM.score(DataCaract, DataPrice)
print("\nLa qualité d'ajustement R^2 = ", Score, "\n")
#Calcule du RSME :
RMSE = np.sqrt(((Residu['price'])**2).sum()/(len(DataPrice)))
print("RMSE = ", RMSE, "\n")

#Vérifions les hypothèses des résidus pour réaliser les t-test qui donne
#la significativité des coefficients
#3 hypothèses : normalité, homoscedasticité et independance des résidus
#Test de l'homoscedasticité par le test de white
print("\n Verifions l'hypothèse d'homoscédasticité")
#H0 : homescédasticité contre H1 : heteroscédasticité
Residu_carre = Residu.apply(lambda x : x**2)
#Faison la regression auxilière
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(DataCaract) #Constance, x, y, x**2, y**2, xy, 
auxiliere = LinearRegression()
auxiliere.fit(x_poly, Residu_carre)
Score_auxi = auxiliere.score(x_poly, Residu_carre) #R^2 de la regression auxiliere
LM = len(DataPrice)*Score_auxi
if LM > stats.chi2(len(x_poly[1])-1).isf(0.05):
    print("Rejet de l'homoscedasticite via le test de white ('fait à la main'):",
          "on a \n*R_aux^2 > Chi2 de degre 35 : ", LM, " > ",stats.chi2(len(x_poly[1])-1).isf(0.05),
          "\nRejet de H0 donc il y a de l'hétéroscédasticité")
else:
    print("comme *R_aux^2 < Chi2 de degre 35 : ", LM, " > ",stats.chi2(len(x_poly[1])-1).isf(0.05),
          "Il y a de l'homoscédasticité")
#Detail methode rapport

#Test d'indépendance
#On utilise le test de durbin-watson
#Calcul la stat de Durbin-Watson:
print("\nvérifions l'indépendance des résidues")
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(Residu["price"])
if np.abs(dw-2) < 0.3: #choix personnel
    print("Les résidues sont indépendants car la valeur du test de Durbin-watson est ", dw, "proche de 2")
elif dw <= 1.7:
    print("Les résidus sont positivement corrélées car la valeur du test de Durbin-watson est \nde",
          dw, " <=1.7")
elif dw > 2.3:
    print("Les résidus sont négativement corrélées car la valeur du test de Durbin-watson est ",
          dw, " >2.3")

#Test de normalité des résidus
#H0 : loi normal H1: loi different de la loi normal
#test de shapiro-wilk
from scipy.stats import shapiro
shap = shapiro(Residu['price'])
if shap.pvalue < 0.05:
    print("\nLa p-valeur du test de shapiro est ", shap.pvalue, "<0.05, ",
          "les résidus ne sont pas distribués par une loi normale")
else:
    print("La p-valeur du test de shapiro est ", shap.pvalue, ">0.05, ",
          "les résidus  sont distribués par une loi normale")

model = smf.ols('price~carlength+carwidth+carheight+cylindernumber+horsepower+citympg+highwaympg', 
                data = data).fit()
print(model.summary())

#Pour savoir quelle caractéristique a le plus d'impact, faisons une acp sur nos données
data2 = pd.DataFrame(data[["carlength","carwidth","carheight","cylindernumber","horsepower",
          "citympg","highwaympg", "price"]])
#┌Standartisons les données
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Z = sc.fit_transform(data2)
from sklearn import decomposition
#Separons le carname au reste de la data:
acp = decomposition.PCA(n_components=2, svd_solver='full')
acp.fit_transform(Z)
print(acp.components_)
"""
c1 = acp.components_[0] * np.sqrt(acp.explained_variance_[0])
c2 = acp.components_[1] * np.sqrt(acp.explained_variance_[1])
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 1, 1)
for i, j, nom in zip(c1, c2, data2.columns):
    plt.text(i, j, nom, fontsize=16)
    plt.arrow(0, 0, i, j, color='black')
plt.axis((-4,4,-4,4))
plt.text(0.5, 0.5, 'Some text', transform=ax.transAxes)
plt.show()"""

list_acp = ["CP1", "CP2"]
df_acp = pd.DataFrame(list_acp, columns = ["ACP"])
df_acp['explained_variance'] = acp.explained_variance_

import seaborn as sns
sns.set(style="whitegrid")
f, ax = plt.subplots(figsize=(6, 8))
sns.set_color_codes("pastel")
sns.barplot(x= "explained_variance", y="ACP", data = df_acp, label ="Total", color = "b")
#ca montre que la première composante explique plus que la deuxieme ?
