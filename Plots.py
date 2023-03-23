import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataIMP = pd.read_csv("plot_imp_0.csv", delimiter=' ')
dataIMP.columns = ['x', 'y']
print(dataIMP)
matrix_dataIMP = dataIMP.to_numpy()

xdataIMP = matrix_dataIMP[:, 0]
ydataIMP = matrix_dataIMP[:, 1]

plt.plot(matrix_dataIMP[:, 0], matrix_dataIMP[:, 1], label ='Schema Implicite')
# plt.show()

dataEXP = pd.read_csv("plot_exp1_0.csv", delimiter=' ')
dataEXP.columns = ['x', 'y']
print(dataEXP)
matrix_dataEXP = dataEXP.to_numpy()

xdataEXP = matrix_dataEXP[:, 0]
ydataEXP = matrix_dataEXP[:, 1]

plt.plot(matrix_dataEXP[:, 0], matrix_dataEXP[:, 1], label='Schema Explicite Robin')
plt.legend()
plt.title(r'Approximation du déplacement du solide au temps final pour $\tau = 2 \cdot 10^{-4}$')
plt.savefig('tau_0.png')
plt.show()

diff = [max(abs(ydataEXP - ydataIMP))]

#####################################################################################################

dataIMP = pd.read_csv("plot_imp_1.csv", delimiter=' ')
dataIMP.columns = ['x', 'y']
print(dataIMP)
matrix_dataIMP = dataIMP.to_numpy()

xdataIMP = matrix_dataIMP[:, 0]
ydataIMP = matrix_dataIMP[:, 1]

plt.plot(matrix_dataIMP[:, 0], matrix_dataIMP[:, 1], label ='Schema Implicite')

dataEXP = pd.read_csv("plot_exp1_1.csv", delimiter=' ')
dataEXP.columns = ['x', 'y']
print(dataEXP)
matrix_dataEXP = dataEXP.to_numpy()

xdataEXP = matrix_dataEXP[:, 0]
ydataEXP = matrix_dataEXP[:, 1]

plt.plot(matrix_dataEXP[:, 0], matrix_dataEXP[:, 1], label='Schema Explicite Robin')
plt.legend()
plt.title(r'Approximation du déplacement du solide au temps final pour $\tau = 2 \cdot 10^{-4}/2$')
plt.savefig('tau_1.png')
plt.show()

diff = diff + [max(abs(ydataEXP - ydataIMP))]

#####################################################################################################

dataIMP = pd.read_csv("plot_imp_2.csv", delimiter=' ')
dataIMP.columns = ['x', 'y']
print(dataIMP)
matrix_dataIMP = dataIMP.to_numpy()

xdataIMP = matrix_dataIMP[:, 0]
ydataIMP = matrix_dataIMP[:, 1]

plt.plot(matrix_dataIMP[:, 0], matrix_dataIMP[:, 1], label ='Schema Implicite')

dataEXP = pd.read_csv("plot_exp1_2.csv", delimiter=' ')
dataEXP.columns = ['x', 'y']
print(dataEXP)
matrix_dataEXP = dataEXP.to_numpy()

xdataEXP = matrix_dataEXP[:, 0]
ydataEXP = matrix_dataEXP[:, 1]

plt.plot(matrix_dataEXP[:, 0], matrix_dataEXP[:, 1], label='Schema Explicite Robin')
plt.legend()
plt.title(r'Approximation du déplacement du solide au temps final pour $\tau = 2 \cdot 10^{-4}/4$')
plt.savefig('tau_2.png')
plt.show()

diff = diff + [max(abs(ydataEXP - ydataIMP))]

#####################################################################################################

dataIMP = pd.read_csv("plot_imp_3.csv", delimiter=' ')
dataIMP.columns = ['x', 'y']
print(dataIMP)
matrix_dataIMP = dataIMP.to_numpy()

xdataIMP = matrix_dataIMP[:, 0]
ydataIMP = matrix_dataIMP[:, 1]

plt.plot(matrix_dataIMP[:, 0], matrix_dataIMP[:, 1], label ='Schema Implicite')
# plt.show()

dataEXP = pd.read_csv("plot_exp1_3.csv", delimiter=' ')
dataEXP.columns = ['x', 'y']
print(dataEXP)
matrix_dataEXP = dataEXP.to_numpy()

xdataEXP = matrix_dataEXP[:, 0]
ydataEXP = matrix_dataEXP[:, 1]

plt.plot(matrix_dataEXP[:, 0], matrix_dataEXP[:, 1], label='Schema Explicite Robin')
plt.legend()
plt.title(r'Approximation du déplacement du solide au temps final pour $\tau = 2 \cdot 10^{-4}/8$')
plt.savefig('tau_3.png')
plt.show()

diff = diff + [max(abs(ydataEXP - ydataIMP))]
diff.reverse()

plt.plot((np.array([2*1e-4, 2*1e-4/2, 2*1e-4/4, 2*1e-4/8])), diff, '-o')
plt.title(r'Différence entre les approximations des schémas explicite de Robin et implicite pour des pas de temps décroissants')
plt.savefig('diff.png')
plt.show()

########################################################
######################## Exo 1 #########################
########################################################

"""-----------------------------------------------------
                        R = 0
-----------------------------------------------------"""

dataQ4_R0 = pd.read_csv("graph_Rd0.csv", delimiter=';')
dataQ4_R0.columns = ['t', 'fluxIn', 'fluxOut', 'pIn', 'pOut']
matrix_dataQ4_R0 = dataQ4_R0.to_numpy()

t = matrix_dataQ4_R0[:, 0]
fluxIn = matrix_dataQ4_R0[:, 1]
fluxOut = matrix_dataQ4_R0[:, 2]
pIn = matrix_dataQ4_R0[:, 3]
pOut = matrix_dataQ4_R0[:, 4]

plt.title('Flux moyen rentrant en fonction du temps pour R = 0')
plt.plot(t, fluxIn)
plt.savefig('FE_0.png')
plt.show()

plt.title('Flux moyen sortant en fonction du temps pour R = 0')
plt.plot(t, fluxOut)
plt.savefig('FS_0.png')
plt.show()

plt.title('Pression moyenne d entrée en fonction du temps pour R = 0')
plt.plot(t, pIn)
plt.savefig('PE_0.png')
plt.show()

plt.title('Pression moyenne de sortie en fonction du temps pour R = 0')
plt.plot(t, pOut)
plt.savefig('PS_0.png')
plt.show()


"""-----------------------------------------------------
                        R = 100
-----------------------------------------------------""""

dataQ4_R100 = pd.read_csv("graph_Rd100.csv", delimiter=';')
dataQ4_R100.columns = ['t', 'fluxIn', 'fluxOut', 'pIn', 'pOut']
matrix_dataQ4_R100 = dataQ4_R100.to_numpy()

t = matrix_dataQ4_R100[:, 0]
fluxIn_100 = matrix_dataQ4_R100[:, 1]
fluxOut_100 = matrix_dataQ4_R100[:, 2]
pIn_100 = matrix_dataQ4_R100[:, 3]
pOut_100 = matrix_dataQ4_R100[:, 4]

plt.title('Flux moyen rentrant en fonction du temps pour R = 100')
plt.plot(t, fluxIn_100)
plt.savefig('FE_100.png')
plt.show()

plt.title('Flux moyen sortant en fonction du temps pour R = 100')
plt.plot(t, fluxOut_100)
plt.savefig('FS_100.png')
plt.show()

plt.title('Pression moyenne d entrée en fonction du temps pour R = 100')
plt.plot(t, pIn_100)
plt.savefig('PE_100.png')
plt.show()

plt.title('Pression moyenne de sortie en fonction du temps pour R = 100')
plt.plot(t, pOut_100)
plt.savefig('PS_100.png')
plt.show()

""""-----------------------------------------------------
                        R = 200
-----------------------------------------------------""""

dataQ4_R200 = pd.read_csv("graph_Rd200.csv", delimiter=';')
dataQ4_R200.columns = ['t', 'fluxIn', 'fluxOut', 'pIn', 'pOut']
matrix_dataQ4_R200 = dataQ4_R200.to_numpy()

t = matrix_dataQ4_R200[:, 0]
fluxIn_200 = matrix_dataQ4_R200[:, 1]
fluxOut_200 = matrix_dataQ4_R200[:, 2]
pIn_200 = matrix_dataQ4_R200[:, 3]
pOut_200 = matrix_dataQ4_R200[:, 4]

plt.title('Flux moyen rentrant en fonction du temps pour R = 200')
plt.plot(t, fluxIn_200)
plt.savefig('FE_200.png')
plt.show()

plt.title('Flux moyen sortant en fonction du temps pour R = 200')
plt.plot(t, fluxOut_200)
plt.savefig('FS_200.png')
plt.show()

plt.title('Pression moyenne d entrée en fonction du temps pour R = 200')
plt.plot(t, pIn_200)
plt.savefig('PE_200.png')
plt.show()

plt.title('Pression moyenne de sortie en fonction du temps pour R = 200')
plt.plot(t, pOut_200)
plt.savefig('PS_200.png')
plt.show()

""""-----------------------------------------------------
                        R = 300
-----------------------------------------------------""""

dataQ4_R300 = pd.read_csv("graph_Rd300.csv", delimiter=';')
dataQ4_R300.columns = ['t', 'fluxIn', 'fluxOut', 'pIn', 'pOut']
matrix_dataQ4_R300 = dataQ4_R300.to_numpy()

t = matrix_dataQ4_R300[:, 0]
fluxIn_300 = matrix_dataQ4_R300[:, 1]
fluxOut_300 = matrix_dataQ4_R300[:, 2]
pIn_300 = matrix_dataQ4_R300[:, 3]
pOut_300 = matrix_dataQ4_R300[:, 4]

plt.title('Flux moyen rentrant en fonction du temps pour R = 300')
plt.plot(t, fluxIn_300)
plt.savefig('FE_300.png')
plt.show()

plt.title('Flux moyen sortant en fonction du temps pour R = 300')
plt.plot(t, fluxOut_300)
plt.savefig('FS_300.png')
plt.show()

plt.title('Pression moyenne d entrée en fonction du temps pour R = 300')
plt.plot(t, pIn_300)
plt.savefig('PE_300.png')
plt.show()

plt.title('Pression moyenne de sortie en fonction du temps pour R =300')
plt.plot(t, pOut_300)
plt.savefig('PS_300.png')
plt.show()

"""-----------------------------------------------------
--------------------- Exo 2 ----------------------------
-----------------------------------------------------"""

"""-----------------------------------------------------
                        Rd1 = Rd2 = 100
-----------------------------------------------------"""

dataEx2_R100 = pd.read_csv("graph_Ex2_Rd1_100_Rd2_100.txt", delimiter=';')
dataEx2_R100.columns = ['t', 'fluxIn', 'fluxOut_1', 'fluxOut_2', 'pIn', 'pOut1', 'pOut2']
matrix_dataEx2_R100 = dataEx2_R100.to_numpy()

plt.title('Flux moyen rentrant en fonction du temps pour R = 100')
plt.plot(dataEx2_R100.t, dataEx2_R100.fluxIn)
plt.savefig('FE_Ex2_100.png')
plt.show()

plt.title('Flux moyen sortant 1 en fonction du temps pour R = 100')
plt.plot(dataEx2_R100.t, dataEx2_R100.fluxOut_1)
plt.savefig('FS1_Ex2_100.png')
plt.show()

plt.title('Flux moyen sortant 2 en fonction du temps pour R = 100')
plt.plot(dataEx2_R100.t, dataEx2_R100.fluxOut_2)
plt.savefig('FS2_Ex2_100.png')
plt.show()

plt.title('Pression moyenne entrante en fonction du temps pour R = 100')
plt.plot(dataEx2_R100.t, dataEx2_R100.pIn)
plt.savefig('PE_Ex2_100.png')
plt.show()

plt.title('Pression moyenne sortant 1 en fonction du temps pour R = 100')
plt.plot(dataEx2_R100.t, dataEx2_R100.pOut1)
plt.savefig('PS1_Ex2_100.png')
plt.show()

plt.title('Pression moyenne sortant 2 en fonction du temps pour R = 100')
plt.plot(dataEx2_R100.t, dataEx2_R100.pOut2)
plt.savefig('PS2_Ex2_100.png')
plt.show()



"""-----------------------------------------------------
                        Rd1 = 100 et Rd2= 800
-----------------------------------------------------"""

dataEx2_R = pd.read_csv("graph_Ex2_Rd1_100_Rd2_800.txt", delimiter=';')
dataEx2_R.columns = ['t', 'fluxIn', 'fluxOut_1', 'fluxOut_2', 'pIn', 'pOut1', 'pOut2']
matrix_dataEx2_R = dataEx2_R.to_numpy()

plt.title('Flux moyen rentrant en fonction du temps pour Rd1 = 100 et Rd2 = 800')
plt.plot(dataEx2_R.t, dataEx2_R.fluxIn)
plt.savefig('FE_Ex2_Rd1_100_Rd2_800.png')
plt.show()

plt.title('Flux moyen sortant 1 en fonction du temps pour Rd1 = 100 et Rd2 = 800')
plt.plot(dataEx2_R.t, dataEx2_R.fluxOut_1)
plt.savefig('FS1_Ex2_Rd1_100_Rd2_800.png')
plt.show()

plt.title('Flux moyen sortant 2 en fonction du temps pour Rd1 = 100 et Rd2 = 800')
plt.plot(dataEx2_R.t, dataEx2_R.fluxOut_2)
plt.savefig('FS2_Ex2_Rd1_100_Rd2_800.png')
plt.show()

plt.title('Pression moyenne entrante en fonction du temps pour Rd1 = 100 et Rd2 = 800')
plt.plot(dataEx2_R.t, dataEx2_R.pIn)
plt.savefig('PE_Ex2_Rd1_100_Rd2_800.png')
plt.show()

plt.title('Pression moyenne sortant 1 en fonction du temps pour Rd1 = 100 et Rd2 = 800')
plt.plot(dataEx2_R.t, dataEx2_R.pOut1)
plt.savefig('PS1_Ex2_Rd1_100_Rd2_800.png')
plt.show()

plt.title('Pression moyenne sortant 2 en fonction du temps pour Rd1 = 100 et Rd2 = 800')
plt.plot(dataEx2_R.t, dataEx2_R.pOut2)
plt.savefig('PS2_Ex2_Rd1_100_Rd2_800.png')
plt.show()



"""-----------------------------------------------------
                        Rd1 = 800 et Rd2= 100
-----------------------------------------------------"""

dataEx2_R = pd.read_csv("graph_Ex2_Rd1_800_Rd2_100.txt", delimiter=';')
dataEx2_R.columns = ['t', 'fluxIn', 'fluxOut_1', 'fluxOut_2', 'pIn', 'pOut1', 'pOut2']
matrix_dataEx2_R = dataEx2_R.to_numpy()

plt.title('Flux moyen rentrant en fonction du temps pour Rd1 = 800 et Rd2 = 100')
plt.plot(dataEx2_R.t, dataEx2_R.fluxIn)
plt.savefig('FE_Ex2_Rd1_800_Rd2_100.png')
plt.show()

plt.title('Flux moyen sortant 1 en fonction du temps pour Rd1 = 800 et Rd2 = 100')
plt.plot(dataEx2_R.t, dataEx2_R.fluxOut_1)
plt.savefig('FS1_Ex2_Rd1_800_Rd2_100.png')
plt.show()

plt.title('Flux moyen sortant 2 en fonction du temps pour Rd1 = 800 et Rd2 = 100')
plt.plot(dataEx2_R.t, dataEx2_R.fluxOut_2)
plt.savefig('FS2_Ex2_Rd1_800_Rd2_100.png')
plt.show()

plt.title('Pression moyenne entrante en fonction du temps pour Rd1 = 100 et Rd2 = 800')
plt.plot(dataEx2_R.t, dataEx2_R.pIn)
plt.savefig('PE_Ex2_Rd1_800_Rd2_100.png')
plt.show()

plt.title('Pression moyenne sortant 1 en fonction du temps pour Rd1 = 800 et Rd2 = 100')
plt.plot(dataEx2_R.t, dataEx2_R.pOut1)
plt.savefig('PS1_Ex2_Rd1_800_Rd2_100.png')
plt.show()

plt.title('Pression moyenne sortant 2 en fonction du temps pour Rd1 = 800 et Rd2 = 100')
plt.plot(dataEx2_R.t, dataEx2_R.pOut2)
plt.savefig('PS2_Ex2_Rd1_800_Rd2_100.png')
plt.show()



"""-----------------------------------------------------
                        Rd1 = Rd2 = 800
-----------------------------------------------------"""

dataEx2_R800 = pd.read_csv("graph_Ex2_Rd1_800_Rd2_800.txt", delimiter=';')
dataEx2_R800.columns = ['t', 'fluxIn', 'fluxOut_1', 'fluxOut_2', 'pIn', 'pOut1', 'pOut2']
matrix_dataEx2_R800 = dataEx2_R800.to_numpy()

plt.title('Flux moyen rentrant en fonction du temps pour R = 800')
plt.plot(dataEx2_R800.t, dataEx2_R800.fluxIn)
plt.savefig('FE_Ex2_800.png')
plt.show()

plt.title('Flux moyen sortant 1 en fonction du temps pour R = 800')
plt.plot(dataEx2_R800.t, dataEx2_R800.fluxOut_1)
plt.savefig('FS1_Ex2_800.png')
plt.show()

plt.title('Flux moyen sortant 2 en fonction du temps pour R = 800')
plt.plot(dataEx2_R800.t, dataEx2_R800.fluxOut_2)
plt.savefig('FS2_Ex2_800.png')
plt.show()

plt.title('Pression moyenne entrante en fonction du temps pour R = 800')
plt.plot(dataEx2_R800.t, dataEx2_R800.pIn)
plt.savefig('PE_Ex2_800.png')
plt.show()

plt.title('Pression moyenne sortant 1 en fonction du temps pour R = 800')
plt.plot(dataEx2_R800.t, dataEx2_R800.pOut1)
plt.savefig('PS1_Ex2_800.png')
plt.show()

plt.title('Pression moyenne sortant 2 en fonction du temps pour R = 800')
plt.plot(dataEx2_R800.t, dataEx2_R800.pOut2)
plt.savefig('PS2_Ex2_800.png')
plt.show()