# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:40:15 2022

@author: jerom
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

data = pd.read_csv("Insect_spray.csv")
print(data)
#Séparons les données en groupe A, B, C, D, E, F
Spray = data.groupby(["spray"])
A = Spray.get_group("A")
B = Spray.get_group("B")
C = Spray.get_group("C")
D = Spray.get_group("D")
E = Spray.get_group("E")
F = Spray.get_group("F")

CountA = np.array(A["count"])
CountB = np.array(B["count"])
CountC = np.array(C["count"])
CountD = np.array(D["count"])
CountE = np.array(E["count"])
CountF = np.array(F["count"])

#print(CountA, CountB, CountC, CountD, CountE, CountF)

#Verifions que nos données suivent les hypothèses pour réaliser l'anova
#Test de normalité
print("Test de shapiro-wilk")
shap1 = stats.shapiro(CountA)
shap2 = stats.shapiro(CountB)
shap3 = stats.shapiro(CountC)
shap4 = stats.shapiro(CountD)
shap5 = stats.shapiro(CountE)
shap6 = stats.shapiro(CountF)
print("A : ",shap1.pvalue, "\nB : ", shap2.pvalue, "\nC : ", shap3.pvalue,
      "\nD : ",shap4.pvalue, "\nE : ",shap5.pvalue,  "\nF : ", shap6.pvalue)

#Test d'homoscédasticité via Bartlett
Bart = stats.bartlett(CountA, CountB, CountC, CountD, CountE, CountF)
print(Bart)
""" a regarder car pval < 0.01 donc ils n'ont pas la meme variance
"""

#Anova sur ces listes pour voir si il existe une différence de moyenne
#entre les différents groupes Count
ANOVA = stats.f_oneway(CountA, CountB, CountC, CountD, CountE, CountF)
print("Anova : ", ANOVA , "\n")
#comme la p-valeur est < 0.05, on rejette l'hypothèse que les moyennes sont egaux
#donc il y a une différence entre ses groupes


#Comparons les moyennes via un test de tukey
tukey = pairwise_tukeyhsd(endog = data["count"], groups = data["spray"], alpha = 0.05)
print("Tukey : ", tukey.summary())
#représentation visuelle
tukey.plot_simultaneous()
#on a que A, B, F a une moyenne différente avec C, D, E et il n'y a pas de
#différence entre A,B et F ; C, D et E
print(stats.f_oneway(CountA, CountB, CountF))