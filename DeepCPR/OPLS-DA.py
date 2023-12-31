# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 16:36:03 2022

@author: ZNDX001
"""
import random
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from numpy import float64
import seaborn as sns
import random
import warnings
import sys
from scipy import optimize
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn.cross_decomposition import PLSRegression


def index_shuffle(obj, ratio):
    full_list = np.copy(obj)
    n_total = len(full_list)
    num = 0
    for i in range(n_total):
        if full_list[i] == 1:
            num += 1
    keepset1 = int(num * (1 - ratio))
    keepset2 = keepset1 + int(n_total * ratio)
    
    if isinstance(full_list, np.ndarray) == True:
        np.random.shuffle(full_list[keepset1:keepset2])
    else:
        random.shuffle(full_list[keepset1:keepset2])

    return full_list

######Gradient descent#####
def immob_pointfit_cost(b, coef, data_x, data_y):
    total_cost = 0
    M = len(data_x)
    for i in range(M):
        x = data_x[i]
        y = data_y[i]
        total_cost += (y - (coef - b) * x - b) ** 2
    return total_cost / M
 
def step_grad_desc(current_w, current_b, alpha, data_x, data_y):
    sum_grad_w = 0
    sum_grad_b = 0
    M = len(data_x)
    for i in range(M):
        x = data_x[i]
        y = data_y[i]
        sum_grad_w += (current_w * x + current_b - y) * x
        sum_grad_b += current_w * x + current_b - y

    grad_w = 2 / M * sum_grad_w
    grad_b = 2 / M * sum_grad_b

    updated_w = current_w - alpha * grad_w
    updated_b = current_b - alpha * grad_b
    return updated_w, updated_b

def immob_pointfit_grad_desc(data_x, data_y, initial_b, coef, alpha, num_iter):
    b = initial_b
    w = coef - initial_b
    cost_list = []
    for i in range(num_iter):
        cost_list.append(immob_pointfit_cost(b, coef, data_x, data_y))
        w, b = step_grad_desc(w, b, alpha, data_x, data_y)
    return [w, b, cost_list]

def errorF(x, y):
    return np.mean(np.squeeze((x - y)**2))**0.5

#A为原始列表，a为右移位数
def list_move_right(A,a):
    for i in range(a):
        A.insert(0,A.pop())
    return A 
 
def OPLS(data, label, crossvalI):
    xMN = data.copy()
    X = xMN
    Y = label
    yMN = yMCN = Y
    crossvalI = crossvalI
    orthoI = False
    predI = False

    #####vertify opls components#####
    autNcoL = False
    autNcpL = False
    autMaxN = min(10, X.shape[0], X.shape[1])
    if orthoI == False: 
        if autMaxN == 1:
            orthoI = 0
            predI = 1
            warnings.warn("The data contain a single variable (or sample): A PLS model with a single component will be built")
        else:
            orthoI = autMaxN - 1
            predI = 1
            autNcoL = True

    if predI == False:
        if orthoI > 0:
            if autMaxN == 1:
                orthoI = 0
                warnings.warn("The data contain a single variable (or sample): A PLS model with a single component will be built")
            else:
                warnings.warn("OPLS(-DA): The number of predictive component is set to 1 for a single response model")

            predI = 1

            if (predI + orthoI) > min(X.shape[0], X.shape[1]):
                print("The sum of 'predI' (", predI, ") and 'orthoI' (", orthoI, ") exceeds the minimum dimension of the 'x' data matrix (", 
                 np.min(X.shape[0], X.shape[1]), ")")
                sys.exit()
        else:
            predI = autMaxN
            autNcpL = True

    ######data standary#####
    x_mean = X.mean(axis=0)
    X -= x_mean
    x_std = X.std(axis=0, ddof=1)
    x_std[x_std == 0.0] = 1.0
    X /= x_std

    y_mean = Y.mean(axis=0)
    y_mean = list([0])
    y_std = list([1])
    Y -= y_mean
    Y /= y_std

    yMeanVn = np.zeros((1, Y.shape[1]))
    ySdVn = np.ones((1, Y.shape[1]))

    #####opls related important matrix#####
    wMN = np.zeros((X.shape[1], predI))
    pMN = np.zeros((X.shape[1], predI))
    tMN = np.zeros((X.shape[0], predI))
    uMN = np.zeros((X.shape[0], predI))
    cMN = np.zeros((Y.shape[1], predI))

    cvfNamVc = []
    for i in range(crossvalI):
        cvfNamVc.append("cv" + str(i))

    orthoNamVc = []
    for i in range(orthoI):
        orthoNamVc.append("o" + str(i))
        
    prkVn = [0 for index in range(crossvalI)]

    if orthoI == 0:
        if X.shape[0] > 100:
            rulThrN = 0
        else:
            rulThrN = 0.05
    else:
        rulThrN = 0.01

    hN = -1

    ssxTotN = np.sum(X**2)
    ssyTotN = np.sum(Y**2)
    rs0N = ssyTotN

    toMN = np.zeros((X.shape[0], orthoI))
    woMN = np.zeros((X.shape[1], orthoI))
    coMN = np.zeros((Y.shape[1], orthoI))
    poMN = np.zeros((X.shape[1], orthoI))

    columns1 = ['R2X','R2X(cum)', 'R2Y', 'R2Y(cum)', 'Q2', 'Q2(cum)', 'Signif.']
    index1 = ['p1']
    index2 = orthoNamVc
    index3 = ['sum']
    index = index1 + index2 + index3
    modelDF = pd.DataFrame(columns=columns1,index=index)

    #####data segment#####
    xcvTraLs = []
    xcvTesLs = []
    ycvTraLs = []
    ycvTesLs = []

    resqueue = list_move_right(list(range(crossvalI)), 6)
    for cro in resqueue:
        train_index = []
        test_index = []
        for xn in range(X.shape[0]):
            if (xn+1) % crossvalI == cro:
                test_index.append(xn)
            else:
                train_index.append(xn)
        xcvTraLs.append(X[train_index,:])
        ycvTraLs.append(Y[train_index,:])
        xcvTesLs.append(X[test_index,:])
        ycvTesLs.append(Y[test_index,:])
    xcvTraLs.append(X)
    ycvTraLs.append(Y)
    breL = False

    #####core arithmetic#####
    for noN in range(orthoI+1):
        if breL == True:
            break
        for cvN in range(len(xcvTraLs)):
            xcvTraMN = xcvTraLs[cvN]
            ycvTraMN = ycvTraLs[cvN]

            uOldVn = ycvTraMN[:,0]

            while True:
                wVn = np.dot(xcvTraMN.T, uOldVn) / np.squeeze(np.dot(uOldVn.T, uOldVn))
                wVn = wVn / np.sqrt(np.squeeze(np.dot(wVn.T, wVn)))
                tVn = np.dot(xcvTraMN, wVn)
                cVn = np.dot(ycvTraMN.T, tVn) / np.squeeze(np.dot(tVn.T, tVn))
                uVn = np.dot(ycvTraMN, cVn) / np.squeeze(np.dot(cVn.T, cVn))
                dscN = np.squeeze(np.sqrt(np.dot(((uVn - uOldVn) / uVn).T, ((uVn - uOldVn) / uVn))))
                if ycvTraMN.shape[1] == 1 or dscN < 1e-10:
                    break
                else:
                    uOldVn = uVn

            pVn = np.dot(xcvTraMN.T , tVn) / np.squeeze(np.dot(tVn.T, tVn))
            woVn = pVn - (np.squeeze(np.dot(wVn.T, pVn)) / np.squeeze(np.dot(wVn.T, wVn))) * wVn

            woVn = woVn / np.sqrt(np.squeeze(np.dot(woVn.T, woVn)))
            toVn = np.dot(xcvTraMN, woVn)
            coVn = np.dot(ycvTraMN.T, toVn) / np.squeeze(np.dot(toVn.T, toVn))
            poVn = np.dot(xcvTraMN.T, toVn) / np.squeeze(np.dot(toVn.T, toVn))
            
            if cvN <= (crossvalI - 1):   #to cvN = 0,1,2,3,4,5,6 testdata
                xcvTesMN = xcvTesLs[cvN]
                ycvTesMN = ycvTesLs[cvN]

                prkVn[cvN] = sum((ycvTesMN - np.dot(np.dot(xcvTesMN, wVn.reshape(X.shape[1], 1)), cVn.reshape(1, 1).T))**2)
                toTesVn = np.dot(xcvTesMN, woVn)
                xcvTesLs[cvN] = xcvTesMN - np.dot(toTesVn.reshape(toTesVn.shape[0], 1), poVn.reshape(poVn.shape[0], 1).T)
                if cvN == (crossvalI - 1):
                    q2N = 1 - sum(prkVn)/rs0N
                    if noN == 0:
                        modelDF.loc['p1', 'Q2(cum)'] = modelDF.loc['p1', 'Q2'] = q2N
                    else:
                        modelDF.iloc[noN].at["Q2(cum)"] = q2N - modelDF.loc["p1", "Q2"]
                        a = 0
                        for i in range(noN):
                            a += modelDF.iloc[i].at['Q2'][0]
                        modelDF.iloc[noN].at["Q2"] = q2N - a
            else:   #to cvN = 7, whole data
                r2yN = sum(np.dot(tVn.reshape(tVn.shape[0], 1), cVn.reshape(cVn.shape[0], 1).T)**2) / ssyTotN
                if noN == 0:
                    modelDF.loc["p1", "R2Y(cum)"] = modelDF.loc["p1", "R2Y"] = r2yN
                else:
                    modelDF.iloc[noN].at["R2Y(cum)"] = r2yN - modelDF.loc["p1", "R2Y"]
                    a = 0
                    for i in range(noN):
                        a += modelDF.iloc[i].at['R2Y'][0]
                    modelDF.iloc[noN].at["R2Y"] = r2yN - a
                if noN <= (orthoI - 1):  #noN = 0-8
                    modelDF.loc[index2[noN], 'R2X'] = sum(sum(np.dot(toVn.reshape(toVn.shape[0], 1), poVn.reshape(poVn.shape[0], 1).T)**2) / ssxTotN)
                    poMN[:, noN] = poVn
                    toMN[:, noN] = toVn
                    woMN[:, noN] = woVn
                    coMN[:, noN] = coVn

                if np.isnan(modelDF.iloc[noN].at['R2Y'][0])==False and modelDF.iloc[noN].at['R2Y'][0] < 0.01:
                    modelDF.iloc[noN].at["Signif."] = "N4"
                else:
                    if np.isnan(modelDF.iloc[noN].at['Q2'][0])==False and modelDF.iloc[noN].at['Q2'][0] < rulThrN:
                        modelDF.iloc[noN].at["Signif."] = "NS"
                    else:
                        modelDF.iloc[noN].at["Signif."] = "R1"

                if autNcoL==True and modelDF.iloc[noN].at["Signif."] != "R1" and noN>1:
                    breL = True
                    break
                else: 
                    cMN[:, 0] = cVn
                    pMN[:, 0] = pVn
                    tMN[:, 0] = tVn
                    uMN[:, 0] = uVn
                    wMN[:, 0] = wVn

            if breL == True:
                break

            if noN < orthoI:
                xcvTraLs[cvN] = xcvTraMN - np.dot(toVn.reshape(toVn.shape[0], 1), poVn.reshape(poVn.shape[0], 1).T)

    modelDF.loc["p1", "R2X(cum)"] = modelDF.loc["p1", "R2X"] = sum(sum(np.dot(tMN, pMN.T)**2) / ssxTotN)

    r2x = []
    for i in range(orthoI+1):
        r2x.append(modelDF.iloc[i].at['R2X'])
    r2xcum = np.cumsum(r2x)
    for i in range(orthoI+1):
        modelDF.iloc[i].at['R2X(cum)'] = r2xcum[i]

    if autNcoL == True:
        Signif = []
        for i in range(orthoI+1):
            Signif.append(modelDF.iloc[i].at['Signif.'])
        while np.nan in Signif:
            Signif.remove(np.nan)

        if np.all(Signif == 'R1'):
            orthoI = noN 
        else:
            orthoI = noN - 2

        if orthoI == autMaxN - 1:
            warnings.warn("The maximum number of orthogonal components in the automated mode (", 
                  autMaxN - 1, ") has been reached whereas R2Y (", 
                  round(modelDF.iloc[1 + orthoI].at["R2Y"] * 100), 
                  "%) is above 1% and Q2Y (", round(modelDF.iloc[1 + orthoI].at["Q2"] * 100), "%) is still above ", 
                  round(ru1ThrN * 100), "%.")

        poMN = poMN[:, 0:orthoI]
        toMN = toMN[:, 0:orthoI]
        woMN = woMN[:, 0:orthoI]
        coMN = coMN[:, 0:orthoI]
        orthoNamVc = orthoNamVc[0:orthoI]

        indexdrop = index2[orthoI:]
        modelDF.drop(indexdrop, inplace=True)

    modelDF.loc["sum", "R2X(cum)"] = modelDF.iloc[orthoI].at["R2X(cum)"]
    modelDF.loc["sum", "R2Y(cum)"] = sum(modelDF.iloc[i].at["R2Y"] for i in range(modelDF.shape[0]-1))
    modelDF.loc["sum", "Q2(cum)"] = sum(modelDF.iloc[i].at["Q2"] for i in range(modelDF.shape[0]-1))
    summaryDF = modelDF[["R2X(cum)", "R2Y(cum)", "Q2(cum)"]]
    summaryDF = summaryDF.loc['sum']

    rMN = wMN
    bMN = np.dot(rMN, cMN.T)
    yPreScaMN = np.dot(tMN, cMN.T)
    yPreMN = yPreScaMN / (1/ySdVn) - (-yMeanVn)

    yActMCN = yMCN
    yActMN = yActMCN
    summaryDF['RMSEE'] = errorF(yActMN, yPreMN)**2 * yActMN.shape[0] / (yActMN.shape[0] - (1 + predI + orthoI))
    yTestMCN = None

    summaryDF['pre'] = predI
    summaryDF['ort'] = orthoI

    #####vip_values(1 to 4)#####
    sxpVn = []
    for i in range(tMN.shape[1]):
        sxpVn.append(np.sum(np.dot(tMN[:,i].reshape(X.shape[0], 1), pMN[:, i].reshape(X.shape[1], 1).T)**2))
    sxpCumN = np.sum(sxpVn)

    sxoVn = []
    for i in range(toMN.shape[1]):
        sxoVn.append(np.sum(np.dot(toMN[:,i].reshape(X.shape[0], 1), poMN[:, i].reshape(X.shape[1], 1).T)**2))
    sxoCumN = np.sum(sxoVn)

    ssxCumN = sxpCumN + sxoCumN

    sypVn = []
    for i in range(tMN.shape[1]):
        sypVn.append(np.sum(np.dot(tMN[:,i].reshape(X.shape[0], 1), cMN[:, i].T)**2))
    sypCumN = np.sum(sypVn)

    syoVn = []
    for i in range(toMN.shape[1]):
        syoVn.append(np.sum(np.dot(toMN[:,i].reshape(X.shape[0], 1), coMN[:, i].T)**2))
    syoCumN = np.sum(syoVn)
    ssyCumN = sypCumN + syoCumN
    k = wMN.shape[0]

    #######VIP1#############
    form_1 = np.zeros((k,))
    for i in range(woMN.shape[1]):
        f1 = ((woMN[:,i] / np.linalg.norm(woMN[:,i]))**2) * sxoVn[i]
        form_1 = form_1 + f1
    form_1 = form_1 / sxoCumN

    form_2 = np.zeros((k,))
    for i in range(wMN.shape[1]):
        f2 = ((wMN[:,i] / np.linalg.norm(wMN[:,i]))**2) * sypVn[i]
        form_2 = form_2 + f2
    form_2 = form_2 / sypCumN

    VIP_1o = (k * form_1)**0.5
    VIP_1p = ((k * form_2)**0.5)
    VIP_1t = ((VIP_1o**2 + VIP_1p**2)/2)**0.5

    #######VIP2##############
    form_1 = np.zeros((k,))
    for i in range(poMN.shape[1]):
        f1 = ((poMN[:,i] / np.linalg.norm(poMN[:,i]))**2) * sxoVn[i]
        form_1 = form_1 + f1
    form_1 = form_1 / sxoCumN

    form_2 = np.zeros((k,))
    for i in range(pMN.shape[1]):
        f2 = ((pMN[:,i] / np.linalg.norm(pMN[:,i]))**2) * sypVn[i]
        form_2 = form_2 + f2
    form_2 = form_2 / sypCumN

    VIP_2o = (k * form_1)**0.5
    VIP_2p = ((k * form_2)**0.5)
    VIP_2t = ((VIP_2o**2 + VIP_2p**2)/2)**0.5

    #####VIP3##############
    kpN = k / (sxpCumN/ssxCumN + sypCumN/ssyCumN)
    koN = k / (sxoCumN/ssxCumN + syoCumN/ssyCumN)

    form_1 = np.zeros((k,))
    for i in range(woMN.shape[1]):
        f1 = ((woMN[:,i] / np.linalg.norm(woMN[:,i]))**2) * sxoVn[i]
        form_1 = form_1 + f1
    form_1 = form_1 / ssxCumN

    form_2 = np.zeros((k,))
    for i in range(wMN.shape[1]):
        f2 = ((wMN[:,i] / np.linalg.norm(wMN[:,i]))**2) * sxpVn[i]
        form_2 = form_2 + f2
    form_2 = form_2 / ssxCumN

    form_3 = np.zeros((k,))
    for i in range(woMN.shape[1]):
        f3 = ((woMN[:,i] / np.linalg.norm(woMN[:,i]))**2) * syoVn[i]
        form_3 = form_3 + f3
    form_3 = form_3 / ssyCumN

    form_4 = np.zeros((k,))
    for i in range(wMN.shape[1]):
        f4 = ((wMN[:,i] / np.linalg.norm(wMN[:,i]))**2) * sypVn[i]
        form_4 = form_4 + f4
    form_4 = form_4 / ssyCumN

    VIP_3o = (koN * (form_1 + form_3))**0.5
    VIP_3p = (kpN * (form_2 + form_4))**0.5
    VIP_3t = (k/2 * (form_1 + form_3 + form_2 + form_4))**0.5

    #####vip4#####
    form_1 = np.zeros((k,))
    for i in range(poMN.shape[1]):
        f1 = ((poMN[:,i] / np.linalg.norm(poMN[:,i]))**2) * sxoVn[i]
        form_1 = form_1 + f1
    form_1 = form_1 / ssxCumN

    form_2 = np.zeros((k,))
    for i in range(pMN.shape[1]):
        f2 = ((pMN[:,i] / np.linalg.norm(pMN[:,i]))**2) * sxpVn[i]
        form_2 = form_2 + f2
    form_2 = form_2 / ssxCumN

    form_3 = np.zeros((k,))
    for i in range(poMN.shape[1]):
        f3 = ((poMN[:,i] / np.linalg.norm(poMN[:,i]))**2) * syoVn[i]
        form_3 = form_3 + f3
    form_3 = form_3 / ssyCumN

    form_4 = np.zeros((k,))
    for i in range(pMN.shape[1]):
        f4 = ((pMN[:,i] / np.linalg.norm(pMN[:,i]))**2) * sypVn[i]
        form_4 = form_4 + f4
    form_4 = form_4 / ssyCumN

    VIP_4o = (koN * (form_1 + form_3))**0.5
    VIP_4p = (kpN * (form_2 + form_4))**0.5
    VIP_4t = (k/2 * (form_1 + form_3 + form_2 + form_4))**0.5

    VIP_dict = {'vip_1o': VIP_1o, 'vip_1p': VIP_1p, 'vip_1t': VIP_1t,
                'vip_2o': VIP_2o, 'vip_2p': VIP_2p, 'vip_2t': VIP_2t,
                'vip_3o': VIP_3o, 'vip_3p': VIP_3p, 'vip_3t': VIP_3t,
                'vip_4o': VIP_4o, 'vip_4p': VIP_4p, 'vip_4t': VIP_4t}
    VIP_df = pd.DataFrame(VIP_dict)

    return modelDF, summaryDF, VIP_df, xcvTraMN, toVn, tVn

if __name__ == "__main__":
    matplotlib.use('Qt5Agg')
    plt.rc('font',family='Times New Roman')

    #data = pd.read_excel('C:/DeepEER2/maledata/23_04_18-24-505/total_peakexcel.xlsx', index_col=0)
    data = pd.read_excel('C:/Users/ZNDX001/Desktop/text/DeepCPR_20231228/Table S10.xlsx', index_col=0)
    data = data.T
    COMs = data.loc['component'][0:53]
    target = np.zeros((136,1))
    target[0:61,0] = 1
    target[61:,0] = -1

    data_1 = data.values[1:]
    data_1 = data_1.astype('float')
    origin_data = np.copy(data_1)

    modelDF, summaryDF, VIP_df, xcvTraMN, toVn, tVn = OPLS(data_1, target, 10)

####score plot####
    plt.figure()
    df = pd.DataFrame(np.column_stack([tVn, toVn]), index=data.index[1:], columns=['t', 't_ortho'])
    pos_df = df[target==1]
    neg_df = df[target==-1]
    plt.scatter(neg_df['t'], neg_df['t_ortho'], c='red', label='TP')
    plt.scatter(pos_df['t'], pos_df['t_ortho'], c='blue', label='TN')
    plt.scatter(neg_df['t'][29], neg_df['t_ortho'][29], c='green', label='FN')
    tn = mpatches.Circle((0.5, 0.5), 0.1, facecolor='red', edgecolor='red', label='TP')
    tp = mpatches.Circle((0.5, 0.5), 0.1, facecolor='blue', edgecolor='blue', label='TN')
    fn = mpatches.Circle((0.5, 0.5), 0.1, facecolor='green', edgecolor='green', label='FN')
    fp = mpatches.Circle((0.5, 0.5), 0.1, facecolor='orange', edgecolor='orange', label='FP')

    pos_index = list(pos_df.index.values)
    neg_index = list(neg_df.index.values)

    plt.title('OPLS Scores', fontsize=16)
    plt.xlabel('t',  fontsize=16)
    plt.ylabel('t_ortho', fontsize=16)
    plt.legend(handles=[tn,tp,fn,fp], loc='lower right')
    plt.savefig("C:/Users/ZNDX001/Desktop/text/DeepCPR/picture/graphpad/score.pdf", dpi=300, format="pdf")

#####vip objects figures#####
    VIP_data = VIP_df['vip_4t'].values
    COM = []
    VIP = []
    for vv in range(data.shape[1]):
        if VIP_data[vv]>=1:
            VIP.append(VIP_data[vv])
            COM.append(COMs[vv])

    sorted_vips = sorted(enumerate(VIP), key=lambda x: x[1])
    idx = [i[0] for i in sorted_vips]
    vips = [i[1] for i in sorted_vips]
    COMS=[]
    for jj in range(len(COM)):
        COMS.append(COM[idx[jj]])

    plt.figure()
    plt.scatter(vips,range(len(COMS)),c = "r")
    plt.yticks(range(len(COMS)),COMS,size=12)
    plt.xticks(size=12)
    plt.xlabel('VIP values')
    plt.grid(axis="y",linestyle='-.')
    plt.tight_layout()
    plt.savefig("C:/Users/ZNDX001/Desktop/text/DeepCPR/picture/graphpad/vips.pdf", dpi=300, format="pdf")

    #####heatmap print#####
    VIP_data = VIP_data.tolist()
    vp = []
    for i in range(len(VIP)):    
        vp.append(VIP_data.index(VIP[i]))

    data = np.array(origin_data, dtype=float64)
    data_VIP = data[:,vp]
    for i in range(data_VIP.shape[1]):
        data_VIP[:,i] = data_VIP[:,i] / np.max(data_VIP[:,i])

    heatmap_data = pd.DataFrame(data_VIP,index=None,columns = COM)
    g = sns.clustermap(heatmap_data.T, cmap="coolwarm", col_cluster=False, row_cluster=True, annot_kws={"size": 12},
                           cbar_kws=dict(orientation='horizontal'), figsize=(12,5), xticklabels=False)

    x0, _y0, _w, _h = g.cbar_pos
    g.ax_cbar.set_position([0.2, 0.98, 0.64, 0.02])
    g.cax.xaxis.set_ticks_position("bottom")
    g.ax_cbar.tick_params(axis='x', length=10)
    for spine in g.ax_cbar.spines:
        g.ax_cbar.spines[spine].set_color('crimson')
        g.ax_cbar.spines[spine].set_linewidth(1)

    plt.savefig("C:/Users/ZNDX001/Desktop/text/DeepCPR/picture/graphpad/heatmap.pdf", dpi=300, format="pdf")
    plt.show()

#####permutation test#####
    R2_50 = []
    Q2_50 = []
    for i in range(2000):
        y50 = index_shuffle(target, 0.5)
        x = np.copy(data)
        x = x.astype('float')
        modelDF50, summaryDF50, VIP_df50, xcvTraMN50, toVn50, tVn50 = OPLS(x, y50, 10)
        R2_50.append(summaryDF50['R2Y(cum)'][0])
        Q2_50.append(summaryDF50['Q2(cum)'][0])

    R2_70 = []
    Q2_70 = []
    for i in range(2000):
        y70 = index_shuffle(target, 0.7)
        x = np.copy(data)
        x = x.astype('float')
        modelDF70, summaryDF70, VIP_df70, xcvTraMN70, toVn70, tVn70 = OPLS(x, y70, 10)
        R2_70.append(summaryDF70['R2Y(cum)'][0])
        Q2_70.append(summaryDF70['Q2(cum)'][0])

    R2_100 = []
    Q2_100 = []
    for i in range(2000):
        y100 = index_shuffle(target, 1)
        x = np.copy(data)
        x = x.astype('float')
        modelDF100, summaryDF100, VIP_df100, xcvTraMN100, toVn100, tVn100 = OPLS(x, y100, 10)
        R2_100.append(summaryDF100['R2Y(cum)'][0])
        Q2_100.append(summaryDF100['Q2(cum)'][0])

    R2_0 = [summaryDF['R2Y(cum)'][0]]
    Q2_0 = [summaryDF['Q2(cum)'][0]]

#####scatter and lineregression figure#####
    plt.figure()

    x_100 = [0 for i in range(2000)]
    plt.scatter(x_100, R2_100, c ='g', s = 16, label = 'R$^2$')
    plt.scatter(x_100, Q2_100, c ='b', s = 16, label = 'Q$^2$')

    x_70 = [0.3 for i in range(2000)]
    plt.scatter(x_70, R2_70, c ='g', s = 16)
    plt.scatter(x_70, Q2_70, c ='b', s = 16)

    x_50 = [0.5 for i in range(2000)]
    plt.scatter(x_50, R2_50, c ='g', s = 16)
    plt.scatter(x_50, Q2_50, c ='b', s = 16)

    x_0 = [1]
    plt.scatter(x_0, R2_0, c ='g', s = 16)
    plt.scatter(x_0, Q2_0, c ='b', s = 16)   

    plt.hlines(0, 0, 1, colors = "#000000", linestyles = ":")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    x_train = (x_100 + x_70 + x_50 + x_0)
    R2_train = (R2_100 + R2_70 + R2_50 + R2_0)
    Q2_train = (Q2_100 + Q2_70 + Q2_50 + Q2_0)

    alpha = 0.00001
    initial_b_R2 = 0.234
    initial_b_Q2 = -0.358
    num_iter = 1000
    w_R2, b_R2, cost_list_R2 = immob_pointfit_grad_desc(x_train, R2_train, initial_b_R2, R2_0[0], alpha, num_iter)
    w_Q2, b_Q2, cost_list_Q2 = immob_pointfit_grad_desc(x_train, Q2_train, initial_b_Q2, Q2_0[0], alpha, num_iter)
    print("wr:", w_R2,
          "br:", b_R2,
          'wr+br:', w_R2+b_R2)
    print("wq:", w_Q2,
          "bq:", b_Q2,
          'wq+bq:', w_Q2+b_Q2)

    x_mp = np.arange(0,1,0.01)
    plt.plot(x_mp, w_R2 * x_mp + b_R2, 'k', linestyle='--')
    plt.plot(x_mp, w_Q2 * x_mp + b_Q2, 'k', linestyle='--')
    plt.legend(loc = 'lower right')

    ax = plt.gca()
    ax.spines['right'].set_color('none') 
    ax.spines['top'].set_color('none') 
    ax.yaxis.set_ticks_position('left')     
    ax.spines['left'].set_position(('data', 0))    
    plt.suptitle('permutation test', fontsize=16)    
    plt.xlabel('undisturbed proportions')
    plt.savefig("C:/Users/ZNDX001/Desktop/text/DeepCPR/picture/graphpad/permutation.pdf", dpi=300, format="pdf")
    plt.show()

#####hist figure#####
    q2r = [0 for i in range(16)]
    for i in range(len(Q2_100)):
        if Q2_100[i] >= -0.60 and Q2_100[i] < -0.55:
            q2r[0] += 1
        if Q2_100[i] >= -0.55 and Q2_100[i] < -0.50:
            q2r[1] += 1
        if Q2_100[i] >= -0.50 and Q2_100[i] < -0.45:
            q2r[2] += 1
        if Q2_100[i] >= -0.45 and Q2_100[i] < -0.40:
            q2r[3] += 1
        if Q2_100[i] >= -0.40 and Q2_100[i] < -0.35:
            q2r[4] += 1
        if Q2_100[i] >= -0.35 and Q2_100[i] < -0.30:
            q2r[5] += 1
        if Q2_100[i] >= -0.30 and Q2_100[i] < -0.25:
            q2r[6] += 1
        if Q2_100[i] >= -0.25 and Q2_100[i] < -0.20:
            q2r[7] += 1
        if Q2_100[i] >= -0.20 and Q2_100[i] < -0.15:
            q2r[8] += 1
        if Q2_100[i] >= -0.15 and Q2_100[i] < -0.10:
            q2r[9] += 1
        if Q2_100[i] >= -0.10 and Q2_100[i] < -0.05:
            q2r[10] += 1
        if Q2_100[i] >= -0.05 and Q2_100[i] < 0.00:
            q2r[11] += 1
        if Q2_100[i] >= 0.00 and Q2_100[i] < 0.05:
            q2r[12] += 1
        if Q2_100[i] >= 0.05 and Q2_100[i] < 0.10:
            q2r[13] += 1
        if Q2_100[i] >= 0.10 and Q2_100[i] < 0.15:
            q2r[14] += 1
        if Q2_100[i] >= 0.15 and Q2_100[i] < 0.20:
            q2r[15] += 1

    c = sum(Q2_100[i] > Q2_0[0] for i in range(len(Q2_100)))
    n_permutation = len(Q2_100)
    p_value = (c + 1) / (n_permutation + 1)
    print('p_value = ', p_value)

    plt.figure()
    x_bar = [-0.575 + 0.05 * i for i in range(len(q2r))]
    plt.bar(x_bar, q2r, width = 0.05, color = 'lightskyblue')
    plt.bar(1, 0, width = 0.01, color = 'lightskyblue')

    text = 'p_value < 0.0005\nQ$^2$ = ' + str(round(Q2_0[0],3))
    styles = {
    "size": 10,
    "color": "black",
    "bbox": {
        "facecolor": "salmon",
        "alpha": 0.5
        }
    }
    arrowprops = {
    "facecolor": "salmon",
    "width": 3.0,
    "headwidth": 5.0,
    "headlength": 8.0,
    "shrink": 0.05
    }

    ax = plt.gca()
    ax.annotate(text, xy=(Q2_0[0], 0), xytext=(0.65, 50), xycoords='data', textcoords='data', arrowprops=arrowprops, **styles)
    ax.spines['right'].set_color('none') 
    ax.spines['top'].set_color('none') 

    plt.xlabel('Q$^2$', fontsize=16)
    plt.ylabel('frequency', fontsize=16)

    plt.savefig("C:/Users/ZNDX001/Desktop/text/DeepCPR/picture/graphpad/frequency.pdf", dpi=300, format="pdf")
    plt.show()






