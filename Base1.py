# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:40:15 2022

@author: jerom
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import statistics as st
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

sns.boxplot(x = "count", y = "spray", data=data)
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
if shap1.pvalue > 0.05:
    print("Le groupe A suit une loi normal car la p-valeur = ", shap1.pvalue, " >0.05")
else:
    print("Le groupe A suit ne suit pas une loi normale car la p-valeur = ", shap1.pvalue, " <0.05")
if shap2.pvalue > 0.05:
    print("Le groupe B suit une loi normal car la p-valeur = ", shap2.pvalue, " >0.05")
else:
    print("Le groupe B suit ne suit pas une loi normale car la p-valeur = ", shap2.pvalue, " <0.05")
if shap3.pvalue > 0.05:
    print("Le groupe C suit une loi normal car la p-valeur = ", shap3.pvalue, " >0.05")
else:
    print("Le groupe C suit ne suit pas une loi normale car la p-valeur = ", shap3.pvalue, " <0.05")
if shap4.pvalue > 0.05:
    print("Le groupe D suit une loi normal car la p-valeur = ", shap4.pvalue, " >0.05")
else:
    print("Le groupe D suit ne suit pas une loi normale car la p-valeur = ", shap4.pvalue, " <0.05")
if shap5.pvalue > 0.05:
    print("Le groupe E suit une loi normal car la p-valeur = ", shap5.pvalue, " >0.05")
else:
    print("Le groupe E suit ne suit pas une loi normale car la p-valeur = ", shap5.pvalue, " <0.05")
if shap6.pvalue > 0.05:
    print("Le groupe F suit une loi normal car la p-valeur = ", shap6.pvalue, " >0.05")
else:
    print("Le groupe F suit ne suit pas une loi normale car la p-valeur = ", shap6.pvalue, " <0.05")

#Groupe C et D ne suivent pas une loi normal car p-valeur < 0.05

#Test d'homoscédasticité via Bartlett
print("\ntest de Bartlett")
Bart = stats.bartlett(CountA, CountB, CountC, CountD, CountE, CountF)
if Bart.pvalue > 0.05:
    print("Les données respectent l'homoscedasticité car la p-valeur = ", Bart.pvalue, ' >0.05')
else:
    print("\nles données ne respectent pas l'homoscedasticité, ils n'ont pas la meme variance car",
    "la p-valeur du test de BartLett est\n : ", Bart.pvalue, "<0.05\n")
#il y a de l'hétéroscédasticité
print("Faisons le test de levene pour vérifier ce dernier test")
levene = stats.levene(CountA, CountB, CountC, CountD, CountE, CountF).pvalue
if levene > 0.05:
    print("Les données respectent l(homoscedasticité car la p-valeur = ", levene, ' >0.05')
else:
    print("\nles données ne respectent pas l'homoscedasticité, ils n'ont pas la meme variance car",
    "la p-valeur du test de Levene est\n : ", levene, "<0.05\n")

#Comme il y a de l'hétéroscédasticité, on ne peut pas utiliser l'anova classique
#car on ne vérifie pas les hypothèses

#Test de Kruskal car on n'a pas les hypothese
#pour realiser l'anova : on a de l'heteroscédasticité
print("Comme les données ne respectent pas les hypothèses, faisons l'anova via la méthode de kruskal")
print("Test de kruskal")
kruskal = stats.kruskal(CountA, CountB, CountC, CountD, CountE, CountF)
if kruskal.pvalue <0.05:
    print("La p-valeur est de : ", kruskal.pvalue, " <0.05 \n donc il y a une difference significative des medianes")
else:
    print("La p-valeur du test est : ", kruskal.pvalue, ", >0.05 \n il n'y a pas de difference significative des medianes")
#Comme la p-val < 0.05, il y a bien une différence des moyennes entre ces groupes

print("\n\n")
#Comparons les moyennes via un test de tukey
tukey = pairwise_tukeyhsd(endog = data["count"], groups = data["spray"], alpha = 0.05)
print("Tukey : \n", tukey.summary())
#représentation visuelle
#tukey.plot_simultaneous()
#on a que A, B, F a une moyenne différente avec C, D, E et il n'y a pas de
#différence entre A,B et F ; C, D et E
#En voyant le graphe, on voit que A, B et F sont plus efficace


#Fonction qui converti le chiffre 1 en A, 2 en B, ...
def trans(x):
    if x ==1:
        return "A"
    elif x ==2:
        return "B"
    elif x ==3:
        return "C"
    elif x ==4:
        return "D"
    elif x ==5:
        return "E"
    else:
        return "F"
    
import scikit_posthocs as sp
pc_conover = sp.posthoc_conover([CountA, CountB, CountC, CountD, CountE, CountF], p_adjust="simes-hochberg")
pc_dunn = sp.posthoc_dunn([CountA, CountB, CountC, CountD, CountE, CountF])
print("test posthoc de conover \n", pc_conover, "\nTest Posthoc de dunnett : \n", pc_dunn)
#print("\ntest\n",pc_conover[1][2])
print("\ninterpretation des p-valeurs des tests ci-dessus")
print("\ntest posthoc de conover")
i = 1
tab_comp = []
while i <= len(pc_conover.columns):
    j = 1
    tab=[trans(i)]
    while j <= len(pc_conover.columns):
        if j!=i:
            if pc_conover[i][j] >= 0.05:
                tab += [trans(j)]
        j+=1
    tab.sort()
    print("Ces groupes ont la même moyenne", tab)
    i+=1

print("\ntest posthoc de dunnett")
i = 1
while i <= len(pc_dunn.columns):
    j = 1
    tab=[trans(i)]
    while j <= len(pc_dunn.columns):
        if j!=i:
            if pc_dunn[i][j] >= 0.05:
                tab += [trans(j)]
        j+=1
    tab.sort()
    if tab not in tab_comp:
        tab_comp += [tab]
    print("Ces groupes ont la même moyenne", tab)
    i+=1

#on remarque la faiblesse des test posthoc, le test de conover dit que le groupe C et E ont la meme moyenne
#que D et E aussi et qu'en meme temps C et D n'ont pas la meme moyenne si on regarde uniquement
#les tests faits sur C et D
#c'est grâce aux tests faits sur E qu'on a que C et D ont la meme moyenne (et que E)
print(tab_comp)
#Regardons maintenant quel groupe de groupe est plus efficace que l'autre
if st.mean(CountA) > st.mean(CountC):
    print("Les groupes A, B, F sont plus efficace que les groupes C, D, E")
else:
    print("Les groupes c,d,e sont plus efficace que les groupes a, b, f")
    