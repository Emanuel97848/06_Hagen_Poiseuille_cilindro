## ============================================================================ ##
# this is file vorticidade-função de corrente-2d.py, created at 14-Jun-2024     #
# maintained by Antonio Emanuel Marques dos Santos                              #
# e-mail: emanuelsantos@mecanica.coppe.ufrj.br                                  #
# vorticidade-função de corrente 2d, elementos triangulares                     # 
## ============================================================================ ##


import meshio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# parâmetros da simulação para a fluido refrigerante R1234ze

v_inlet = 1.0  # velocidade inicial [m/s]

Re = 1.0       # número de Reynolds
Pr = 0.7968       # número de Prandtl
Pe = Re*Pr     # número de Peclet
Sc = 0.75      # número de Schmidt
nIter = 100    # número de iterações
# mi = 0.190e-3     # viscosidade dinâmica [Pa*s]
# rho = 1163.1    # densidade do material [kg/m³]
# nu = mi/rho      # viscosidade cinemática [m²/s]
# nu = 0.001
# d = 1          # altura do canal [m]

dt = 0.01    # passo de tempo
q = 100.0    # fonde de calor [W/m³]
#cp = 1.386     # capacidade termica [J/kg ºC] ou [J/kg K]
# cp = 1.0
# k = 74.2e-3      # condutividade termica [W/m K]
# k = 1.0
# alpha = k/(rho*cp)   # difusividade térmica

# Criação da malha: vetores de coordenadas x e y, matrizes de conectividade de omega, IEN e IENbound
filename = "C:\\Users\\emanu\\OneDrive\\Documentos\\02_Engenharia\\03_Mestrado\\02_Dissertacao\\03_Codes\\02_Computacao para engenheiros\\01_Comp for engineers\\01_Mesh\\05_HP_refino_0_1.msh"
msh = meshio.read(filename)
X = msh.points[:,0]
Y = msh.points[:,1]
IEN = msh.cells[1].data
IENbound = msh.cells[0].data
IENboundTypeElem = list(msh.cell_data['gmsh:physical'][0] - 1)
boundNames = list(msh.field_data.keys())
IENboundElem = [boundNames[elem] for elem in IENboundTypeElem]
npoints = len(X)
ne = IEN.shape[0]

# cria lista de nós do contorno
cc = np.unique(IENbound.reshape(IENbound.size))
ccName = [[] for i in range( len(X) )]
for elem in range(0,len(IENbound)):
    ccName[ IENbound[elem][0] ] = IENboundElem[elem]
    ccName[ IENbound[elem][1] ] = IENboundElem[elem]

# elementos vizinhos
elemViz = [[] for i in range(npoints)]
for e in range(0,ne):
    [v1,v2,v3] = IEN[e]
    elemViz[v1].append(e)
    elemViz[v2].append(e)
    elemViz[v3].append(e)
    #elemViz[v4].append(e)

# pontos internos (1ª tentativa)
miolo = []
for i in range(0,npoints):
    if len(elemViz[i]) >= 3:
       miolo.append(i)

top = [];bottom = [];inlet = [];outlet = []
buraco = []
for i in cc:
    if ccName[i] == 'top':
        top.append(i)
    if ccName[i] == 'bottom':
        bottom.append(i)
    if ccName[i] == 'inlet':
        inlet.append(i)
    if ccName[i] == 'outlet':
        outlet.append(i)
    if ccName[i] == 'buraco':
        buraco.append(i)

print("Malha lida")

# velocidade de entrada em função de Re
# v_inlet = (nu*Re)/(Y.max()-Y.min())

# inicialização das matrizes Kx, Ky, Kxy, M, Gx, Gy, Kest    
K = np.zeros((npoints,npoints), dtype='float')
M = np.zeros((npoints,npoints), dtype='float')
Gx = np.zeros((npoints,npoints), dtype='float')
Gy = np.zeros((npoints,npoints), dtype='float')
Q = np.zeros((npoints,npoints), dtype='float')

# montagem das matrizes
for e in range(0,ne):
    v1 = IEN[e,0]
    v2 = IEN[e,1]
    v3 = IEN[e,2]

    # coeficientes b e c
    bi = Y[v2] - Y[v3]
    bj = Y[v3] - Y[v1]
    bk = Y[v1] - Y[v2]
    ci = X[v3] - X[v2]
    cj = X[v1] - X[v3]
    ck = X[v2] - X[v1]

    area = (1.0/2.0)*np.linalg.det([[1,X[v1],Y[v1]],
                                    [1,X[v2],Y[v2]],
                                    [1,X[v3],Y[v3]]])
    
    mele = (area/12)*np.array([[2,1,1],
                               [1,2,1],
                               [1,1,2]])
    
    gxele = (1.0/6.0)*np.array([[bi, bj, bk],
                                [bi, bj, bk],
                                [bi, bj, bk]])
    
    gyele = (1.0/6.0)*np.array([[ci, cj, ck],
                                [ci, cj, ck],
                                [ci, cj, ck]])
    
    kxele = (1.0/(4*area))*np.array([[bi*bi,bi*bj,bi*bk],
                                     [bj*bi,bj*bj,bj*bk],
                                     [bk*bi,bk*bj,bk*bk]])
    
    kyele = (1.0/(4*area))*np.array([[ci*ci,ci*cj,ci*ck],
                                     [cj*ci,cj*cj,cj*ck],
                                     [ck*ci,ck*cj,ck*ck]])
    
    # usado para a montagem de kestele (talvez seja melhor reposicionar)
    # kxyele = (1.0/(4*area))*np.array([[bi*ci,bi*cj,bi*ck],
    #                                   [bj*ci,bj*cj,bj*ck],
    #                                   [bk*ci,bk*cj,bk*ck]])
    
    kele = (kxele+kyele)

    for ilocal in range(0,3):
        iglobal = IEN[e,ilocal]
        for jlocal in range(0,3):
            jglobal = IEN[e,jlocal]

            K[iglobal,jglobal] += kele[ilocal,jlocal]
            M[iglobal,jglobal] += mele[ilocal,jlocal]
            Gx[iglobal,jglobal] += gxele[ilocal,jlocal]
            Gy[iglobal,jglobal] += gyele[ilocal,jlocal]

print("Assembling feito")
# inicialização do campo de velocidades vx, vy e psi
vx = np.zeros((npoints),dtype='float')
vy = np.zeros((npoints),dtype='float')
T = np.zeros((npoints),dtype='float')
q = np.zeros((npoints),dtype='float')
c = np.zeros((npoints),dtype='float')
# A = K.copy()
# b = np.zeros((npoints),dtype='float')

# imposição das c.c. para vx, vy e T (1ª tentativa)
# for i in inlet:
#     vx[i] = v_inlet
#     vy[i] = 0.0
#     c[i] = 1.0
# for i in top:
#     T[i] = 1.0
# for i in bottom:
#     T[i] = 1.0
# for i in buraco:
#     T[i] = 1.0

# imposição das cc para vx, vy e T (2ª tentativa)
eps = 1e-6  # epsilon para identificar os nós que estão na fronteira
for i in cc:
    if X[i] - eps < X.min():    # à esquerda do domínio
        vx[i] = v_inlet
        vy[i] = 0.0             # não defini T[i] aqui pois T é inicializado como um vetor de zeros
        c[i] = 1.0
    elif Y[i] - eps < Y.min():  # parte inferior do domínio
        T[i] = 1.0
    elif Y[i] - eps > Y.max():  # parte superior do domínio
        T[i] = 1.0

for i in buraco:
    T[i] = 1.0

cc2 = [x for x in cc if not X[x] + eps > X.max()]



print("inicio da simulação")
# inicialização do campo de vorticidade
for n in tqdm(range(0,nIter)):
    b = np.dot(Gx,vy) - np.dot(Gy,vx)
    omega = np.linalg.solve(M,b)
    omegacc = omega.copy()

    # zerando os valores de omegacc no interior da malha
    for i in miolo:
        omegacc[i] = 0.0

    # resolvendo o transporte de vorticidade (forma do termo convectivo explícito)
    A = M/dt + (1/Re)*K

    vdotGw = vx*np.dot(Gx,omega) + vy*np.dot(Gy,omega)

    b_1 = (1/dt)*np.dot(M,omega) - vdotGw

    # # resolvendo o transporte de vorticidade (forma do termo convectivo implícito)
    # A = M/dt + (1/Re)*K + vx@Gx + vy@Gy

    # b_1 = (1/dt)*np.dot(M,omega)

    # imposição de c.c. para omega
    for i in cc:
        A[i,:] = 0.0    # zerando a linha
        A[i,i] = 1.0    # colocando 1 na diagonal
        b_1[i] = omegacc[i]

    # resolve o sistema para a eq. de transporte de vorticidade
    omega = np.linalg.solve(A,b_1)

    # resolvendo a eq de psi (função corrente)
    A_2 = K.copy()
    b_2 = np.dot(M,omega)

    # c.c. para psi
    psicc = np.zeros((npoints),dtype='float')
    eps = 1e-6
    # for i in top:
    #     psicc[i] = Y.max()*v_inlet

    # for i in bottom:
    #     psicc[i] = 0.0

    # for i in inlet:
    #     psicc[i] = Y[i]*v_inlet
    
    # for i in outlet:
    #     psicc[i] = Y[i]*v_inlet

    for i in cc2:
        if Y[i] - eps < Y.min():
            psicc[i] = 0.0
        elif Y[i] + eps > Y.max():
            psicc[i] = Y.max()*v_inlet
        elif X[i] - eps < X.min():
            psicc[i] = Y[i]*v_inlet

    # for i in cc:
    #     if Y[i] - eps < Y.min():
    #         psicc[i] = 0.0
    #     elif Y[i] + eps > Y.max():
    #         psicc[i] = Y.max()*v_inlet
    #     elif X[i] - eps < X.min():
    #         psicc[i] = Y[i]*v_inlet
    

    for i in cc2:
        A_2[i,:] = 0.0   # zerando a linha
        A_2[i,i] = 1.0     # colocando 1 na diagonal
        b_2[i] = psicc[i]

    # for i in cc: (dessa maneira, psi na saída fica incorreto)
    #     A_2[i,:] = 0.0   # zerando a linha
    #     A_2[i,i] = 1.0     # colocando 1 na diagonal
    #     b_2[i] = psicc[i]

    for i in buraco:
        A_2[i,:] = 0.0   # zerando a linha
        A_2[i,i] = 1.0     # colocando 1 na diagonal
        b_2[i] = 0.5*v_inlet
        
    # resolvendo psi
    psi = np.linalg.solve(A_2,b_2)

    # cálculo da pressão (opcional)
    #p = np.zeros((npoints),dtype='float')

    # cálculo do campo de velocidades vx e vy

    b_3 = np.dot(Gy,psi)
    vx = np.linalg.solve(M,b_3)

    b_4 = -np.dot(Gx,psi)
    vy = np.linalg.solve(M,b_4)

    # impor c.c. para vx e vy
    # for i in inlet:
    #     vx[i] = v_inlet
    #     vy[i] = 0.0

    # for i in top: # essa condição conflita com a condição de entrada
    #     vx[i] = 0.0
    #     vy[i] = 0.0

    # for i in bottom:
    #     vx[i] = 0.0
    #     vy[i] = 0.0

    eps = 1e-6
    for i in cc2:
        if X[i] - eps < X.min():
            vx[i] = v_inlet
            vy[i] = 0.0
        elif Y[i] - eps < Y.min():
            vx[i] = 0.0
            vy[i] = 0.0
        elif Y[i] + eps > Y.max():
            vx[i] = 0.0
            vy[i] = 0.0

    # for i in cc:
    #     if X[i] - eps < X.min():
    #         vx[i] = v_inlet
    #         vy[i] = 0.0
    #     elif Y[i] - eps < Y.min():
    #         vx[i] = 0.0
    #         vy[i] = 0.0
    #     elif Y[i] + eps > Y.max():
    #         vx[i] = 0.0
    #         vy[i] = 0.0

    # resolvendo a eq. de calor (forma do termo convectivo explícito)
    A_3 = M/dt + (1/Pe)*K 

    vdotGT = vx*np.dot(Gx,T) + vy*np.dot(Gy,T)

    b_5 = (1/dt)*np.dot(M,T) - vdotGT

    # # resolvendo a eq. de calor (forma do termo convectivo implícito)
    # A_3 = M/dt + (1/Pe)*K - vx@Gx - vy@Gy

    # b_5 = (M/dt)@T

    T = np.linalg.solve(A_3,b_5)

    # impor c.c. para T
    # for i in left:
    #     T[i] = 0.0

    # for i in top:
    #     T[i] = 1.0

    # for i in bottom:
    #     T[i] = 1.0

    for i in cc2:
        if X[i] - eps < X.min():
            T[i] = 0.0
        elif Y[i] - eps < Y.min():
            T[i] = 1.0
        elif Y[i] + eps > Y.max():
            T[i] = 1.0
    for i in buraco:
        T[i] = 1.0

    # for i in cc:
    #     if X[i] - eps < X.min():
    #         T[i] = 0.0
    #     elif Y[i] - eps < Y.min():
    #         T[i] = 1.0
    #     elif Y[i] + eps > Y.max():
    #         T[i] = 1.0
    # for i in buraco:
    #     T[i] = 1.0        


    # resolvendo para a eq de transporte de espécies (forma do termo convectivo explícito)
    A_4 = M/dt + (1/(Re*Sc))*K

    vdotGc = vx*np.dot(Gx,c) + vy*np.dot(Gy,c)

    b_6 = (M/dt)@c - vdotGc

    c = np.linalg.solve(A_4,b_6)

    # impor c.c. para c
    for i in cc2:
        if X[i] - eps < X.min():
            c[i] = 1.0


    # grava a solução em VTK para usar no PARAVIEW
    point_data = {'funcao_corrente': psi}
    data_vx = {'vx': vx}
    data_vy = {'vy': vy}
    data_omega = {'omega': omega}
    data_T = {'T': T}
    data_c = {'c': c}
    point_data.update(data_vx)
    point_data.update(data_vy)
    point_data.update(data_omega)
    point_data.update(data_T)
    point_data.update(data_c)
    meshio.write_points_cells('HP_refino_0_1_' + str(n) + '.vtk',
                          msh.points,
                          msh.cells,
                          #file_format='vtk-ascii',
                          point_data=point_data)
    # time.sleep(0.5)
    
print("Simulaçao finalizada")