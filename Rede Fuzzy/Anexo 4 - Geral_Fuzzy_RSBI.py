
'''**Antecedentes (entradas)**

Volume Tidal (VT): qual o volume tidal em ml? 
- sucesso acima de 315 ml, fracasso abaixo de 315 ml.

RSBI(RR/TIDAL) RM/l: qual o RSBI? 
- sucesso <=80 BM/L, fracasso >=120 BM/L.

Taxa respiratória (RR) (respirações/minuto): qual a taxa respiratória?
- sucesso < 30 R/M, fracasso >=35 R/M

Pico negativo da pressão inspiratória (NIF) (cm.H2O): Qual o pico negativo de pressão inspiratória?
- sucesso < -26 CM H2O, fracasso > -20CM H2O

**Consequentes (saídas)**

Ação: desmame ou não, entre 0 e 100?
- onde > 50 permanece entubado e  <=50 desmame

- Documentação: https://pythonhosted.org/scikit-fuzzy/overview.html
'''

# Importação das bibliotecas
import numpy as np
import skfuzzy as fuzz
import pandas as pd
from skfuzzy import control as ctrl

#%% Importando os dados
csv_path = 'C:\\Users\\deivity.andrade\\OneDrive - Companhia de Gás de Santa Catarina - SCGÁS\\Área de Trabalho\\fuzzy\\Respira_DB_Tds_RSBI.txt'
df = pd.read_table(csv_path)

#%%
"""## Antecedentes"""

"""#Criando um array com o range de casos de NIF obtido dos arquivos de treino e teste"""
NIF = ctrl.Antecedent(np.arange(int(df[['NIF']].min()), int(df[['NIF']].max()+1), 1), 'NIF')
"""#Criando um array com o range de casos de VT obtido dos arquivos de treino e teste"""
VT = ctrl.Antecedent(np.arange(int(df[['VT']].min()), int(df[['VT']].max()+1), 1), 'VT')
"""#Criando um array com o range de casos de RR obtido dos arquivos de treino e teste"""
RR = ctrl.Antecedent(np.arange(int(df[['RR']].min()), int(df[['RR']].max()+1), 1), 'RR')
"""#Criando um array com o range de casos de RSBI obtido dos arquivos de treino e teste"""
RSBI = ctrl.Antecedent(np.arange(int(df[['RSBI']].min()), int(df[['RSBI']].max()+1), 1), 'RSBI')
#%%
print("Universo de NIF: %s" % (NIF.universe))
print("Universo de VT: %s" % (VT.universe))
print("Universo de RR: %s" % (RR.universe))
print("Universo de RSBI: %s" % (RSBI.universe))
#%%
"""#Consequente"""
""""#Criando um array com o range de acaso de NIF obtido dos arquivos de treino e teste"""
desmame = ctrl.Consequent(np.arange(0, 101, 1), 'desmame')
desmame.universe
#%%
"""## Membership functions"""
#Definindo os critérios
NIF.automf(number = 2, names = ['Sucesso', 'Fracasso'])
VT.automf(number = 2, names = ['Sucesso', 'Fracasso'])
RR.automf(number = 2, names = ['Sucesso', 'Fracasso'])
RSBI.automf(number = 2, names = ['Sucesso', 'Fracasso'])
#%%
#Definindo os valores para os critérios do NIF - utilizando a função triangular
NIF['Sucesso'] = fuzz.trimf(NIF.universe, [-60,-60,-20])
NIF['Fracasso'] = fuzz.trimf(NIF.universe, [-26,-10,-10])
#gerando o gráfico
NIF.view()
#%%
#Definindo os valores para os critérios do VT - utilizando a função triangular
VT['Sucesso'] = fuzz.trimf(VT.universe, [315,950,950])
VT['Fracasso'] = fuzz.trimf(VT.universe, [124,124,315])

VT.view()
#%%
#Definindo os valores para os critérios do RR - utilizando a função triangular
RR['Fracasso'] = fuzz.trimf(RR.universe, [30,46,46])
RR['Sucesso'] = fuzz.trimf(RR.universe, [13,13,35])

RR.view()
#%%
#Definindo os valores para os critérios do RSBI - utilizando a função triangular
RSBI['Fracasso'] = fuzz.trimf(RSBI.universe, [80,266,266])
RSBI['Sucesso'] = fuzz.trimf(RSBI.universe, [14,14,120])

RSBI.view()
#%%
#Definindo os valores para os critérios do Desmame - utilizando a função triangular
desmame['Desmamar'] = fuzz.trimf(desmame.universe, [0, 0, 100])
desmame['Permanece'] = fuzz.trimf(desmame.universe, [0, 100, 100])
desmame.view()
#%%
#Definindo as regras segundo os antecedes para o consequente
# Pela regra, entendemos que para 2 antecedentes indicando fracasso, o desmame não será efetuado (permanece)
# Em caso de 2 antecedentes indicando sucesso, o desmame será efetuado (demamar)
regra1 = ctrl.Rule(NIF['Fracasso'] & VT['Fracasso'] & RR['Fracasso'] & RSBI['Fracasso'], desmame['Permanece'])
regra2 = ctrl.Rule(NIF['Fracasso'] & VT['Fracasso'] & RR['Fracasso'] & RSBI['Sucesso'], desmame['Permanece'])
regra3 = ctrl.Rule(NIF['Fracasso'] & VT['Fracasso'] & RR['Sucesso'] & RSBI['Fracasso'], desmame['Permanece'])
regra4 = ctrl.Rule(NIF['Fracasso'] & VT['Sucesso'] & RR['Fracasso'] & RSBI['Fracasso'], desmame['Permanece'])
regra5 = ctrl.Rule(NIF['Sucesso'] & VT['Fracasso'] & RR['Fracasso'] & RSBI['Fracasso'], desmame['Permanece'])
regra6 = ctrl.Rule(NIF['Fracasso'] & VT['Fracasso'] & RR['Sucesso'] & RSBI['Sucesso'], desmame['Permanece'])
regra7 = ctrl.Rule(NIF['Fracasso'] & VT['Sucesso'] & RR['Fracasso'] & RSBI['Sucesso'], desmame['Permanece'])
regra8 = ctrl.Rule(NIF['Sucesso'] & VT['Fracasso'] & RR['Fracasso'] & RSBI['Sucesso'], desmame['Permanece'])
regra9 = ctrl.Rule(NIF['Sucesso'] & VT['Sucesso'] & RR['Fracasso'] & RSBI['Fracasso'], desmame['Permanece'])
regra10 = ctrl.Rule(NIF['Sucesso'] & VT['Fracasso'] & RR['Sucesso'] & RSBI['Fracasso'], desmame['Permanece'])
regra11 = ctrl.Rule(NIF['Fracasso'] & VT['Sucesso'] & RR['Sucesso'] & RSBI['Fracasso'], desmame['Permanece'])
regra12 = ctrl.Rule(NIF['Fracasso'] & VT['Sucesso'] & RR['Sucesso'] & RSBI['Sucesso'], desmame['Desmamar'])
regra13 = ctrl.Rule(NIF['Sucesso'] & VT['Sucesso'] & RR['Fracasso'] & RSBI['Sucesso'], desmame['Desmamar'])
regra14 = ctrl.Rule(NIF['Sucesso'] & VT['Fracasso'] & RR['Sucesso'] & RSBI['Sucesso'], desmame['Desmamar'])
regra15 = ctrl.Rule(NIF['Sucesso'] & VT['Sucesso'] & RR['Sucesso'] & RSBI['Fracasso'], desmame['Desmamar'])
regra16 = ctrl.Rule(NIF['Sucesso'] & VT['Sucesso'] & RR['Sucesso'] & RSBI['Sucesso'], desmame['Desmamar'])

#%%
"""## Sistema de controle"""
#Criando sistema de controle entre as regras
sistema_controle = ctrl.ControlSystem([regra1, regra2, regra3, regra4, regra5, 
                                       regra6, regra7, regra8, regra9, regra10, 
                                       regra11, regra12, regra13, regra14, regra15, regra16])
sistema = ctrl.ControlSystemSimulation(sistema_controle)
lista_resultados = []
def teste_conjunto(NIF, RR, VT, RSBI):
    sistema.input['NIF'] = NIF
    sistema.input['RR'] = RR
    sistema.input['VT'] = VT
    sistema.input['RSBI'] = RSBI
    sistema.compute()
    print(sistema.output['desmame'])
    desmame.view(sim=sistema)
    if sistema.output['desmame'] < 50:
        lista_resultados.append('S')
    else:
        lista_resultados.append('F')
 
for i in range(len(df)):
    
    teste_conjunto(df['NIF'].iloc[i], df['RR'].iloc[i], df['VT'].iloc[i], df['RSBI'].iloc[i])
print( df['RSBI'].iloc[1])   
#%% Comparando os resulados com o conjunto de referência
TP = 0
TN = 0
FP = 0
FN = 0
for i in range(len(df)):
    if df['Label'].iloc[i] == "S" and lista_resultados[i] == "S":
         TP = TP + 1
    elif df['Label'].iloc[i] == "S" and lista_resultados[i] == "F":
        FN = FN + 1
    elif df['Label'].iloc[i] == "F" and lista_resultados[i] == "F":
        TN = TN + 1
    else:
        FP = FP +1
sensibilidade = TP/(df[df["Label"] == "S"].count()["Label"])
especificidade =TN/(df[df["Label"] == "F"].count()["Label"])
print("SENSIBILIDADE: %s" % sensibilidade)
print("ESPECIFICIDADE: %s" % especificidade)
