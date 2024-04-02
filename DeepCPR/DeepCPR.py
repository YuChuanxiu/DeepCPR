# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 20:10:03 2022

@author: ZNDX001
"""
from scipy.spatial.distance import cosine
from scipy.optimize import curve_fit
import numpy as np
from numpy import float32
import tensorflow as tf
# from numba import jit
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import os
import math
from numpy import hstack
from DeepCPR.NetCDF import netcdf_reader
from DeepCPR.DeepCS import Chromseg
from scipy import integrate
from scipy.signal import find_peaks
import pandas as pd
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from itertools import groupby
import copy
from sklearn.metrics import explained_variance_score
import tensorly as tl
import itertools
import time
import pywt
import gc
from tqdm import tqdm
#import PIL.Image


def check_file(path):
    if not os.path.exists(path):
        print("File not exist")
        return False
    if (os.path.isdir(path)):
        if not os.access(path, os.R_OK):
            print("File is accessible to read")
            return False
        print(os.path.basename(path))

        for file in os.listdir(path):
            file = os.path.join(path, file)
            check_file(file)
    else:
        if not os.access(path, os.R_OK):
            print("File is accessible to read")
            return False
        with open(path) as f:
            print(f.name)


def list_allfile(path, all_files=[]):
    if os.path.exists(path):
        files = os.listdir(path)
    else:
        print('this path not exist')
    for file in files:
        if os.path.isdir(os.path.join(path, file)):
            list_allfile(os.path.join(path, file), all_files)
        else:
            all_files.append(os.path.join(path, file))
    return all_files


def process(X):
    for i in range(X.shape[0]):
        if np.max(X[i, :]) != 0:
            X[i, :] = 10000*X[i, :]/np.max(X[i, :])
        else:
            print(i, 'k')
    return X


def data_restore(num, dis_st, predict, ind_st_DeepSeg, ind_en_DeepSeg, mz_min, mz_max, chrom_divid):
    chrom_origin = chrom_divid[num]
    C_0 = np.zeros((ind_en_DeepSeg[num]-ind_st_DeepSeg[num], 5), dtype=float32)
    C_0[:, :] = predict[num][int(dis_st):int(dis_st+ind_en_DeepSeg[num]-ind_st_DeepSeg[num]), :]
    C_0[C_0 < 0] = 0
    return chrom_origin, C_0


def cal_deriv(x, y):
    diff_x = []
    for i, j in zip(x[0::], x[1::]):
        diff_x.append(j - i)

    diff_y = []
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)

    slopes = []
    for i in range(len(diff_y)):
        slopes.append(diff_y[i] / diff_x[i])

    deriv = []
    for i, j in zip(slopes[0::], slopes[1::]):
        deriv.append((0.5 * (i + j)))
    deriv.insert(0, slopes[0])
    deriv.append(slopes[-1])
    return deriv


# @jit(nopython=True)
def peak_find(CC):
    C = np.copy(CC)

    threshold = 3e-3
    fun = lambda x: x[1]-x[0]
    peak_st = []
    peak_ed = []
    for r in range(C.shape[1]):
        st = []
        ed = []
        for k, g in groupby(enumerate(np.where(C[:, r] > threshold)[0]), fun):
            l1 = [j for i, j in g]
            st.append(min(l1))
            ed.append(max(l1))
            st_copy = st.copy()
            ed_copy = ed.copy()

        if len(st) > 1:
            long_idx = np.argmax([np.max(ed[ii]-st[ii]) for ii in range(len(st))])
            st_idx = [gg for gg in range(len(st))]
            del st_idx[long_idx]
            for jj in range(len(st_idx)):
                st.remove(st_copy[st_idx[jj]])
                ed.remove(ed_copy[st_idx[jj]])

        if len(st) == 1:
            peak_st.append(st[0])
            peak_ed.append(ed[0])

    return peak_st, peak_ed


# @jit(nopython=True)
def com_find(CC):
    C = np.copy(CC)

    threshold = 3e-3
    fun = lambda x: x[1]-x[0]
    peak_st = []
    peak_ed = []
    for r in range(C.shape[1]):
        st = []
        ed = []
        for k, g in groupby(enumerate(np.where(C[:, r] > threshold)[0]), fun):
            l1 = [j for i, j in g]
            st.append(min(l1))
            ed.append(max(l1))
            st_copy = st.copy()
            ed_copy = ed.copy()

        if len(st) > 1:
            long_idx = np.argmax([np.max(ed[ii]-st[ii]) for ii in range(len(st))])
            st_idx = [gg for gg in range(len(st))]
            del st_idx[long_idx]
            for jj in range(len(st_idx)):
                st.remove(st_copy[st_idx[jj]])
                ed.remove(ed_copy[st_idx[jj]])

        if len(st) == 1:
            peak_st.append(st[0])
            peak_ed.append(ed[0])

    COM = len(peak_st)
    return COM


# @jit(nopython=True)
def peak_preprocess(CC):
    # low intensity noise filtering
    C = np.copy(CC)

    threshold = 3e-3
    for r in range(C.shape[1]):
        if max(C[:, r]) < 0.04:
            C[:, r] = 0

    # chromatographic gilling
    for r in range(C.shape[1]):
        peak_st = []
        peak_ed = []
        fun = lambda x: x[1]-x[0]
        for k, g in groupby(enumerate(np.where(C[:, r] > threshold)[0]), fun):
            l1 = [j for i, j in g]
            peak_st.append(min(l1))
            peak_ed.append(max(l1))

        if len(peak_st)>1:
            for m in range(1, len(peak_st)):
                for n in range(len(peak_ed)-1):
                    if peak_st[m] == peak_ed[n]:
                        C[peak_st[m], r] = (C[peak_st[m]-1, r] + C[peak_st[m]+1, r])/2

    # filtering peaks with width less 5
    for r in range(C.shape[1]):
        peak_st = []
        peak_ed = []
        for k, g in groupby(enumerate(np.where(C[:, r] > threshold)[0]), fun):
            l1 = [j for i, j in g]
            peak_st.append(min(l1))
            peak_ed.append(max(l1))

        if len(peak_st) > 0:
            for ps in range(len(peak_st)):
                if peak_ed[ps]-peak_st[ps]+2 < 5:
                    C[peak_st[ps]:peak_ed[ps]+1, r] = 0
        if max(C[:, r]) < 0.04:
            C[:, r] = 0

    # preserving peak with the highest intensity in each m/z channel
    fun = lambda x: x[1]-x[0]
    for r in range(C.shape[1]):
        peak_st = []
        peak_ed = []
        for k, g in groupby(enumerate(np.where(C[:, r] > threshold)[0]), fun):
            l1 = [j for i, j in g]
            peak_st.append(min(l1))
            peak_ed.append(max(l1))

        if len(peak_st) > 1:
            intensity = []
            for ps in range(len(peak_st)):
                intensity.append(max(C[peak_st[ps]:peak_ed[ps], r]))
            p = intensity.index(max(intensity))
            if peak_st[p] != 0:
                C[0:peak_st[p], r] = 0
            if peak_ed[p] != (C.shape[0]-1):
                C[peak_ed[p]:, r] = 0

    # refine the shape of gaussian peak
    for r in range(C.shape[1]):
        peaks, _ = find_peaks(C[:, r], height=0.003)
        if len(peaks) > 1:
            inten_max = peaks[np.argmax([C[i, r] for i in peaks])]
            peak_st, peak_ed = peak_find(C[:, r:r+1])
            gaussleftfit = 'no'
            gaussrightfit = 'no'
            for z in range(inten_max, peak_st[0]+1, -1):
                if C[z, r] > 0.003 and C[z-1, r] > C[z, r]:
                    gaussleftfit = 'yes'
                    lp = z
                    break
            for y in range(inten_max, peak_ed[0]-1, 1):
                if C[y, r] > 0.003 and C[y+1, r] > C[y, r]:
                    gaussrightfit = 'yes'
                    rp = y
                    break

            if gaussleftfit == 'yes' and gaussrightfit == 'yes':
                peak_x = np.arange(lp+1, rp)
                peak_x = [float(i) for i in peak_x]
                peak_y = C[lp+1:rp, r]
                peak_y = [float(i) for i in peak_y]
                if len(peak_x) > 2 and np.abs(inten_max-lp) > 2 and np.abs(inten_max-rp) > 2:
                    if C[lp, r] < 0.5*np.max(C[:, r]):
                        s = 2*np.abs(getnearpos(C[lp:inten_max, r], 0.5*np.max(C[:, r])) - inten_max)
                    elif C[rp, r] < 0.5*np.max(C[:, r]):
                        s = 2*np.abs(getnearpos(C[inten_max:rp, r], 0.5*np.max(C[:, r])) - inten_max)
                    else:
                        if np.argmin([C[lp, r], C[rp, r]]) == 0:
                            s = np.max(C[:, r])*np.abs(inten_max-lp)/(np.max(C[:, r])-C[lp, r])
                        if np.argmin([C[lp, r], C[rp, r]]) == 1:
                            s = np.max(C[:, r])*np.abs(rp-inten_max)/(np.max(C[:, r])-C[rp, r])

                    popt, pcov = curve_fit(gaussian, peak_x, peak_y, p0=[np.max(C[:, r]), inten_max, s], maxfev = 500000)
                    for j in range(rp, C.shape[0],1):
                        C[j, r] = gaussian(j, *popt)
                        if gaussian(j, *popt) < 0.003:
                            C[j:C.shape[0], r] = 0
                            break
                    for j in range(lp, -1, -1):
                        C[j, r] = gaussian(j, *popt)
                        if gaussian(j, *popt) < 0.003:
                            C[0:j, r] = 0
                            break

            if gaussleftfit == 'no' and gaussrightfit == 'yes':
                peak_x = np.arange(peak_st[0], rp)
                peak_x = [float(i) for i in peak_x]
                peak_y = C[peak_st[0]:rp, r]
                peak_y = [float(i) for i in peak_y]
                if len(peak_x) > 2 and np.abs(inten_max-rp) > 2:
                    if inten_max == peak_st[0] and peak_st[0] > 0:
                        s = 2*np.abs(getnearpos(C[peak_st[0]-1:inten_max, r], 0.5*np.max(C[:, r])) - inten_max)
                    elif inten_max == peak_st[0] and peak_st[0] == 0:
                        s = 2*np.abs(getnearpos(C[peak_st[0]:inten_max+1, r], 0.5*np.max(C[:, r])) - inten_max)
                    else:
                        s = 2*np.abs(getnearpos(C[peak_st[0]:inten_max, r], 0.5*np.max(C[:, r])) - inten_max)
                    popt, pcov = curve_fit(gaussian, peak_x, peak_y, p0=[np.max(C[:, r]), inten_max, s], maxfev = 500000)
                    for j in range(rp, C.shape[0],1):
                        C[j, r] = gaussian(j, *popt)
                        if gaussian(j, *popt) < 0.003:
                            C[j:C.shape[0], r] = 0
                            break

            if gaussleftfit == 'yes' and gaussrightfit == 'no':
                peak_x = np.arange(lp+1, peak_ed[0])
                peak_x = [float(i) for i in peak_x]
                peak_y = C[lp+1:peak_ed[0], r]
                peak_y = [float(i) for i in peak_y]
                if len(peak_x) > 2 and np.abs(inten_max-lp) > 2:
                    if inten_max == peak_st[0] and peak_ed[0] < C.shape[0]:
                        s = 2*np.abs(getnearpos(C[inten_max:peak_ed[0]+1, r], 0.5*np.max(C[:, r])) - inten_max)
                    elif inten_max == peak_st[0] and peak_ed[0] == C.shape[0]:
                        s = 2*np.abs(getnearpos(C[inten_max-1:peak_ed[0], r], 0.5*np.max(C[:, r])) - inten_max)
                    else:
                        s = 2*np.abs(getnearpos(C[inten_max:peak_ed[0], r], 0.5*np.max(C[:, r])) - inten_max)
                    popt, pcov = curve_fit(gaussian, peak_x, peak_y, p0=[np.max(C[:,r]), inten_max, s], maxfev = 500000)
                    for j in range(lp, -1, -1):
                        C[j, r] = gaussian(j, *popt)
                        if gaussian(j, *popt) < 0.003:
                            C[0:j, r] = 0
                            break
    return C


# @jit(nopython=True)
def tail_fix(CC):
    C = np.copy(CC)
    COM = com_find(C)
    peak_st, peak_ed = peak_find(C)
    for r in range(COM):
        xrange = [i for i in range(peak_st[r], peak_ed[r]+1)]
        yrange = [C[:, r][i] for i in range(peak_st[r], peak_ed[r]+1)]
        deriv = cal_deriv(xrange, yrange)
        x_max = np.argmax(deriv)
        x_min = np.argmin(deriv)

        leftfit = 'no'
        rightfit = 'no'
        for z in range(x_max, 0, -1):
            if deriv[z-1] > deriv[z]:
                leftfit = 'yes'
                lp = z + min(xrange)
                break
        for y in range(x_min, len(deriv)-1, 1):
            if deriv[y+1] < deriv[y]:
                rightfit = 'yes'
                rp = y + min(xrange)
                break
        if leftfit == 'yes':
            C[0:lp, r] = 0
        if rightfit == 'yes':
            C[rp:C.shape[0], r] = 0
    return C


# @jit(nopython=True)
def peak_halfremove(CC):
    # 1 remove half peak and add mark
    C = np.copy(CC)
    threshold = 3e-3
    halfdo = 'None'
    COM = com_find(C)
    if COM > 1:
        fun = lambda x: x[1]-x[0]
        remove_r = []
        for r in range(COM):
            st = []
            ed = []
            for k, g in groupby(enumerate(np.where(C[:, r] > threshold)[0]), fun):
                l1 = [j for i, j in g]
                st.append(min(l1))
                ed.append(max(l1))
                st_copy = st.copy()
                ed_copy = ed.copy()

            if len(st) > 1:
                long_idx = np.argmax([ed[ii]-st[ii] for ii in range(len(st))])
                st_idx = [gg for gg in range(len(st))]
                del st_idx[long_idx]
                for jj in range(len(st_idx)):
                    st.remove(st_copy[st_idx[jj]])
                    ed.remove(ed_copy[st_idx[jj]])

            if min(st) == 0:
                if C[st, r] > (max(C[st[0]:ed[0], r])/3) and np.max(C[st[0]:ed[0], r]) < (np.max(C)*0.25):
                    remove_r.append(r)
                    halfdo = 'Done'
            if max(ed) == (C.shape[0]-1):
                if C[ed, r] > (max(C[st[0]:ed[0], r])/3) and np.max(C[st[0]:ed[0], r]) < (np.max(C)*0.25):
                    remove_r.append(r)
                    halfdo = 'Done'

        if len(remove_r) == COM:
            intenmax_idx = np.argmax([max(C[:, j]) for j in range(COM)])
            for i in range(COM):
                if i != intenmax_idx:
                    C[:, i] = 0
        if len(remove_r) < COM:
            for i in range(len(remove_r)):
                C[:, remove_r[i]] = 0

    return C, halfdo


# @jit(nopython=True)
def dishalfremove(CC, C_ed, threshold=3e-3):
    # 2 remove half peak and add mark
    C = np.copy(CC)
    # threshold = 3e-3
    dishalfdo = 'None'
    dishalfnum = []
    COM = com_find(C_ed)
    if COM > 1:
        fun = lambda x: x[1]-x[0]
        for r in range(COM):
            st = []
            ed = []
            for k, g in groupby(enumerate(np.where(C_ed[:, r] > threshold)[0]), fun):
                l1 = [j for i, j in g]
                st.append(min(l1))
                ed.append(max(l1))
                st_copy = st.copy()
                ed_copy = ed.copy()

            if len(st) > 1:
                long_idx = np.argmax([np.max(ed[ii]-st[ii]) for ii in range(len(st))])
                st_idx = [gg for gg in range(len(st))]
                del st_idx[long_idx]
                for jj in range(len(st_idx)):
                    st.remove(st_copy[st_idx[jj]])
                    ed.remove(ed_copy[st_idx[jj]])

            if len(st) > 0:
                if min(st) == 0 and C_ed[st, r] > (max(C_ed[st[0]:ed[0], r])/2):
                    C[:, r] = 0
                    dishalfdo = 'Done'
                    dishalfnum.append(r)
                if max(ed) == (C.shape[0]-1) and C_ed[ed, r] > (max(C_ed[st[0]:ed[0], r])/2):
                    C[:, r] = 0
                    dishalfdo = 'Done'
                    dishalfnum.append(r)

    return C, dishalfdo, dishalfnum


# @jit(nopython=True)
def testt(CC):
    # refine chromatogram by detecting and removing coincide peaks
    C = np.copy(CC)
    testdo = 'None'
    lapdo = 'None'
    lapnum = []
    removenum = []
    COM = com_find(C)
    if COM > 1:
        peak_st, peak_ed = peak_find(C)
        for m in range(COM-1):
            for n in range(m+1, COM):
                C_concat = np.zeros((C.shape[0], 2))
                C_concat[:, 0] = C[:, m]
                C_concat[:, 1] = C[:, n]
                C_overlap = C_concat.min(axis=1)
                intenmin_idx = [m, n][np.argmin([max(C[:, m]), max(C[:, n])])]
                intenmax_idx = [m, n][np.argmax([max(C[:, m]), max(C[:, n])])]
                area_min = np.trapz(C_overlap[min(peak_st):max(peak_ed)])
                area_max = np.trapz(C[min(peak_st):max(peak_ed), intenmin_idx])

                if area_max < 0.05:
                    ol_degree = 0
                else:
                    ol_degree = area_min / area_max

                if np.abs(np.argmax(C[:, m]) - np.argmax(C[:, n])) < 4 and ol_degree > 0.6:
                    C[:, intenmin_idx] = 0
                    testdo = 'Done'
                    removenum.append(intenmin_idx)

                if np.abs(np.argmax(C[:, m]) - np.argmax(C[:, n])) > 3 and ol_degree > 0.95:
                    if np.max(C[intenmin_idx]) == 0:
                        C[:, intenmin_idx] = 0
                        testdo = 'Done'
                        removenum.append(intenmin_idx)
                    elif np.max(C[intenmax_idx])/np.max(C[intenmin_idx]) > 8:
                        C[:, intenmin_idx] = 0
                        testdo = 'Done'
                        removenum.append(intenmin_idx)

                if np.abs(np.argmax(C[:, m]) - np.argmax(C[:, n])) > 3 and ol_degree > 0.75:
                    lapdo = 'Done'
                    lapnum.append(intenmin_idx)

    lapnum_2 = copy.deepcopy(list(set(lapnum)))
    lapnum_1 = copy.deepcopy(list(set(lapnum)))
    for b in range(len(lapnum_2)):
        if lapnum_2[b] in removenum:
            lapnum_1.remove(lapnum_2[b])

    if len(lapnum_1) > 0:
        lapdo = 'Done'
    else:
        lapdo = 'None'

    return C, testdo, lapdo, lapnum_1


def peak_filter(C, C_ed, St, COM):
    # filer components with spectra or chromatogram is none
    for i in range(COM):
        if np.all(St[i, :] == 0) is True or np.all(C_ed[:, i] == 0) is True:
            C[:, i] = 0
    return C


def peak_trans(C, C_trans):
    # rearrange peak in chromatographic list
    signone = []
    for m in range(C.shape[1]):
        if np.all(C[:, m] == 0) == False:
            signone.append(m)

    lenth = len(signone)-1
    if signone[lenth] != lenth:
        for r in range(lenth+1):
            C_trans[:, r] = C[:, signone[r]]
    else:
        C_trans = C

    return C_trans

# @jit(nopython=True)
def peak_dislap(CC, C_ed):
    # remove overlapping chromatograms after iteration
    C = np.copy(CC)
    dislapdo = 'None'
    dislapnum = []
    COM = com_find(C_ed)
    if COM > 1:
        for m in range(COM-1):
            for n in range(m+1, COM):
                S_c = 1 - cosine((C_ed[:, m]), (C_ed[:, n]))
                if S_c > 0.92:
                    if max(C[:, m]) >= max(C[:, n]):
                        C[:, n] = 0
                        dislapdo = "Done"
                        dislapnum.append(n)
                    else:
                        C[:, m] = 0
                        dislapdo = "Done"
                        dislapnum.append(m)

                elif np.abs(np.argmax(C_ed[:, m]) - np.argmax(C_ed[:, n])) < 3 and S_c > 0.8:
                    if max(C[:, m]) >= max(C[:, n]):
                        C[:, n] = 0
                        dislapdo = "Done"
                        dislapnum.append(n)
                    else:
                        C[:, m] = 0
                        dislapdo = "Done"
                        dislapnum.append(m)

    return C, dislapdo, dislapnum


def unimod(c, rmod, cmod, imax=None):
    ns = c.shape[1]
    if imax is None:
        imax = np.argmax(c, axis=0)
    for j in range(0, ns):
        rmax = c[imax[j], j]
        k = imax[j]
        while k > 0:
            k = k-1
            if c[k, j] <= rmax:
                rmax = c[k, j]
            else:
                rmax2 = rmax*rmod
                if c[k, j] > rmax2:
                    if cmod == 0:
                        c[k, j] = 0  # 1e-30
                    if cmod == 1:
                        c[k, j] = c[k+1, j]
                    if cmod == 2:
                        if rmax > 0:
                            c[k, j] = (c[k, j]+c[k+1, j])/2
                            c[k+1, j] = c[k, j]
                            k = k+2
                        else:
                            c[k, j] = 0
                    rmax = c[k, j]
        rmax = c[imax[j], j]
        k = imax[j]

        while k < c.shape[0]-1:
            k = k+1
            if k == 53:
                k = 53
            if c[k, j] <= rmax:
                rmax = c[k, j]
            else:
                rmax2 = rmax*rmod
                if c[k, j] > rmax2:
                    if cmod == 0:
                        c[k, j] = 1e-30
                    if cmod == 1:
                        c[k, j] = c[k-1, j]
                    if cmod == 2:
                        if rmax > 0:
                            c[k, j] = (c[k, j]+c[k-1, j])/2
                            c[k-1, j] = c[k, j]
                            k = k-2
                        else:
                            c[k, j] = 0
                    rmax = c[k, j]
    return c


def fnnls(x, y, tole):
    xtx = np.dot(x, x.T)
    xty = np.dot(x, y.T)
    if tole == 'None':
        tol = 10*np.spacing(1)*np.linalg.norm(xtx)*max(xtx.shape)
    mn = xtx.shape
    P = np.zeros(mn[1])
    Z = np.array(range(1, mn[1]+1), dtype='int64')
    xx = np.zeros(mn[1])
    ZZ = Z-1
    w = xty-np.dot(xtx, xx)
    iter = 0
    itmax = 30*mn[1]
    z = np.zeros(mn[1])
    while np.any(Z) and np.any(w[ZZ] > tol):
        t = ZZ[np.argmax(w[ZZ])]
        P[t] = t+1
        Z[t] = 0
        PP = np.nonzero(P)[0]
        ZZ = np.nonzero(Z)[0]
        nzz = np.shape(ZZ)
        if len(PP) == 1:
            z[PP] = xty[PP]/xtx[PP, PP]
        elif len(PP) > 1:
            if np.linalg.det(xtx[np.ix_(PP, PP)]) == 0:
                small = 1e-6*np.identity(xtx[np.ix_(PP, PP)].shape[0])
                z[PP] = np.dot(xty[PP], np.linalg.inv(xtx[np.ix_(PP, PP)]+small))
            else:
                z[PP] = np.dot(xty[PP], np.linalg.inv(xtx[np.ix_(PP, PP)]))
        z[ZZ] = np.zeros(nzz)
        while np.any(z[PP] <= tol) and iter < itmax:
            iter += 1
            qq = np.nonzero((tuple(z <= tol) and tuple(P != 0)))
            epsilon = 1e-10
            divider = xx[qq] - z[qq]
            divider[abs(divider) < epsilon] = epsilon
            alpha = np.min(xx[qq] / divider)
            # alpha = np.min(xx[qq] / (xx[qq] - z[qq]))
            xx = xx + alpha*(z - xx)
            ij = np.nonzero(tuple(np.abs(xx) < tol) and tuple(P != 0))
            Z[ij[0]] = ij[0]+1
            P[ij[0]] = np.zeros(max(np.shape(ij[0])))
            PP = np.nonzero(P)[0]
            ZZ = np.nonzero(Z)[0]
            nzz = np.shape(ZZ)
            if len(PP) == 1:
                z[PP] = xty[PP]/xtx[PP, PP]
            elif len(PP) > 1:
                z[PP] = np.dot(xty[PP], np.linalg.inv(xtx[np.ix_(PP, PP)]))
            z[ZZ] = np.zeros(nzz)
        xx = np.copy(z)
        xx[xx < 0] = 0
        w = xty - np.dot(xtx, xx)
    return {'xx': xx, 'w': w}


# @jit(nopython=True)
def ITTFA_PRO(X, C_0, COM):
    u, s, v = tl.truncated_svd(X, COM)
    T = np.dot(u, np.diag(s))

    C_ed = np.zeros((C_0.shape[0], COM), dtype=float32)

    for r in range(COM):
        C = C_0[:, r].reshape((C_0[:, r].shape[0], 1))
        l = 0
        while l < 30:
            C_st = C
            C = np.dot(np.dot(np.dot(T, np.linalg.pinv(np.dot(T.T, T))), T.T), C)
            C[C < 0] = 0
            C = unimod(C, 1.1, 2)

            if np.linalg.norm(C) != 0:
                C = C/np.linalg.norm(C)
            normc = np.linalg.norm(C-C_st)

            l += 1
            C = C.reshape((C_0.shape[0]))
            C_ed[:, r] = C
            C = C.reshape((C_0.shape[0], 1))

            if normc < 1e-6 or l == 30:
                peaks, _ = find_peaks(C[:, 0], height=0.003)
                if len(peaks) > 0:
                    inten_max = peaks[np.argmax([C[i] for i in peaks])]
                    peak_st, peak_ed = peak_find(C)
                    gaussleftfit = 'no'
                    gaussrightfit = 'no'

                    for z in range(inten_max, peak_st[0]+1, -1):
                        if C[z] > 0.003 and C[z-1] > C[z]:
                            gaussleftfit = 'yes'
                            lp = z
                            break
                    for y in range(inten_max, peak_ed[0]-1, 1):
                        if C[y] > 0.003 and C[y+1] > C[y]:
                            gaussrightfit = 'yes'
                            rp = y
                            break

                    if gaussleftfit == 'yes' and gaussrightfit == 'yes':
                        # test_lx = np.arange(peak_st[0], lp+1, 1)
                        # test_rx = np.arange(rp, peak_ed[0], 1)
                        peak_x = np.arange(lp+1, rp)
                        peak_y = C[lp+1:rp, 0]
                        if len(peak_x) > 2 and np.abs(inten_max-lp) > 2 and np.abs(inten_max-rp) > 2:
                            if C[lp, 0] < 0.5*np.max(C):
                                s = 2*np.abs(getnearpos(C[lp:inten_max, 0], 0.5*np.max(C)) - inten_max)
                            elif C[rp, 0] < 0.5*np.max(C):
                                s = 2*np.abs(getnearpos(C[inten_max:rp, 0], 0.5*np.max(C)) - inten_max)
                            else:
                                if np.argmin([C[lp, 0], C[rp, 0]]) == 0:
                                    s = np.max(C)*np.abs(inten_max-lp)/(np.max(C)-C[lp, 0])
                                if np.argmin([C[lp, 0], C[rp, 0]]) == 1:
                                    s = np.max(C)*np.abs(rp-inten_max)/(np.max(C)-C[rp, 0])

                            popt, pcov = curve_fit(gaussian, peak_x, peak_y, p0=[np.max(C), inten_max, s], maxfev=10000000)
                            for j in range(rp, C.shape[0], 1):
                                C[j, 0] = gaussian(j, *popt)
                                if gaussian(j, *popt) < 0.001:
                                    C[j:C.shape[0], 0] = 0
                                    break
                            for j in range(lp, -1, -1):
                                C[j, 0] = gaussian(j, *popt)
                                if gaussian(j, *popt) < 0.001:
                                    C[0:j, 0] = 0
                                    break

                    if gaussleftfit == 'no' and gaussrightfit == 'yes':
                        # test_rx = np.arange(rp, peak_ed[0], 1)
                        peak_x = np.arange(peak_st[0], rp)
                        peak_y = C[peak_st[0]:rp, 0]
                        if len(peak_x) > 2 and np.abs(inten_max-rp) > 2:
                            if np.abs(peak_st[0] - inten_max) > 1:
                                s = 2*np.abs(getnearpos(C[peak_st[0]:inten_max, 0], 0.5*np.max(C)) - inten_max)
                                popt, pcov = curve_fit(gaussian, peak_x, peak_y, p0=[np.max(C), inten_max, s], maxfev=10000000)
                                for j in range(rp, C.shape[0], 1):
                                    C[j, 0] = gaussian(j, *popt)
                                    if gaussian(j, *popt) < 0.001:
                                        C[j:C.shape[0], 0] = 0

                    if gaussleftfit == 'yes' and gaussrightfit == 'no':
                        # test_lx = np.arange(peak_st[0], lp+1, 1)
                        peak_x = np.arange(lp+1, peak_ed[0])
                        peak_y = C[lp+1:peak_ed[0], 0]
                        if len(peak_x) > 2 and np.abs(inten_max-lp) > 2:
                            s = 2*np.abs(getnearpos(C[inten_max:peak_ed[0], 0], 0.5*np.max(C)) - inten_max)
                            popt, pcov = curve_fit(gaussian, peak_x, peak_y, p0=[np.max(C), inten_max, s], maxfev=10000000)
                            for j in range(lp, -1, -1):
                                C[j, 0] = gaussian(j, *popt)
                                if gaussian(j, *popt) < 0.001:
                                    C[0:j, 0] = 0
                                    break

                C = C.reshape((C_0.shape[0]))
                C_ed[:, r] = C
                break

    chrom_origin = X
    St = np.zeros((COM, chrom_origin.shape[1]))
    for j in range(0, St.shape[1]):
        a = fnnls(np.dot(C_ed.T, C_ed), np.dot(C_ed.T, chrom_origin[:, j]), tole='None')
        St[:, j] = a['xx']

    return C_ed, St


# @jit(nopython=True)
def ITTFA(X, C_0, COM):
    u, s, v = tl.truncated_svd(X, COM)
    T = np.dot(u, np.diag(s))

    C_ed = np.zeros((C_0.shape[0], COM), dtype=float32)

    for r in range(COM):
        C = C_0[:, r].reshape((C_0[:, r].shape[0], 1))
        l = 0
        # norm_set = []
        while l < 30:
            C_st = C
            C = np.dot(np.dot(np.dot(T, np.linalg.pinv(np.dot(T.T, T))), T.T), C)
            C[C < 0] = 0
            C = unimod(C, 1.1, 2)
            if np.linalg.norm(C) != 0:
                C = C/np.linalg.norm(C)
            normc = np.linalg.norm(C-C_st)
            l += 1
            C = C.reshape((C_0.shape[0]))
            C_ed[:, r] = C
            C = C.reshape((C_0.shape[0], 1))
            if normc < 1e-6 or l == 30:
                C = C.reshape((C_0.shape[0]))
                C_ed[:, r] = C
                break

    chrom_origin = X
    St = np.zeros((COM, chrom_origin.shape[1]))
    for j in range(0, St.shape[1]):
        a = fnnls(np.dot(C_ed.T, C_ed), np.dot(C_ed.T, chrom_origin[:, j]), tole='None')
        St[:, j] = a['xx']

    return C_ed, St


def judge_max(n):
    n1 = str(float(n))
    n2 = n1.split('.')
    if n2[1] == '0':
        return int(n+1)
    else:
        return n


def judge_min(n):
    n1 = str(float(n))
    n2 = n1.split('.')
    if n2[1] == '0':
        return int(n-1)
    else:
        return n


def data_process(work_path, filename, dist, thres):
    ncr = netcdf_reader(filename, bmmap=False)
    sie = hstack((ncr.f.variables['scan_index'].data, np.array([len(ncr.f.variables['intensity_values'].data)], dtype=int)))
    mat = ncr.mat(1, len(sie)-2, 1)
    RT = mat['rt']
    Xtest = mat['d']
    model_size = 128
    y_DeepSeg, ind_st_DeepSeg, ind_en_DeepSeg = Chromseg(work_path, mat, model_size, distance=dist, threshold=thres)
    return mat, RT, Xtest, ind_st_DeepSeg, ind_en_DeepSeg


def gaussian(x,*param):
    return param[0]*np.exp(-np.power(x - param[1], 2.) / (2 * np.power(param[2], 2.)))


def getnearpos(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


def R_squared(y_true, y_pred):
    residual = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true))))
    r2 = tf.subtract(1.0, tf.math.divide(residual, total))
    return r2


def FR(x, s, o, z, com):
    xs = x[s, :]
    xs[xs < 0] = 0
    xz = x[z, :]
    # xo = x[o, :]
    xc = np.vstack((xs, xz))
    mc = np.vstack((xs, np.zeros(xz.shape)))

    u, s0, v = tl.truncated_svd(xc, com)
    t = np.dot(u, np.diag(s0))
    r = np.dot(np.dot(np.linalg.pinv(np.dot(t.T, t)), t.T), np.sum(mc, 1))
    u1, s1, v1 = tl.truncated_svd(x, com)
    t1 = np.dot(u1, np.diag(s1))
    c = np.dot(t1, r)

    c1, ind = contrain_FR(c, s, o)
    c1[c1 < 0] = 0
    spec = x[s[ind], :]

    if c1[s[ind]] == 0:
        pu = 1e-6
    else:
        pu = c1[s[ind]]

    cc = c1/pu

    res_x = np.dot(np.array(cc, ndmin=2).T, np.array(spec, ndmin=2))
    # left_x = x - res_x
    spec = spec.reshape(1, spec.shape[0])
    return cc, spec, res_x


def contrain_FR(c, s, o):
    ind_s = np.argmax(np.abs(c[s]))
    if c[s][ind_s] < 0:
        c = -c

    if s[0] < o[0]:
        if c[s[-2]] < c[s[-1]]:
            ind1 = s[-1]
            ind2 = o[np.argmax(c[o])]
        else:
            ind1 = s[np.argmax(c[s])]
            ind2 = o[0]
    else:
        if c[s[1]] < c[s[0]]:
            ind1 = o[np.argmax(c[o])]
            ind2 = s[0]
        else:
            ind1 = o[-1]
            ind2 = s[np.argmax(c[s])]

    for i, indd in enumerate(np.arange(ind1, 0, -1)):
        if c[indd-1] >= c[indd]:
            c[0:indd] = 0
            break
        if c[indd-1] < 0:
            c[0:indd] = 0
            break

    for i, indd in enumerate(np.arange(ind2, len(c)-1, 1)):
        if c[indd+1] >= c[indd]:
            c[indd+1:len(c)] = 0
            break
        if c[indd+1] < 0:
            c[indd+1:len(c)] = 0
            break
    return c, ind_s


def full_rank_resolution(x, CC, peak_st, peak_ed, COM):
    C = np.copy(CC)

    if COM == 2:
        p1 = peak_ed[0]
        p2 = peak_st[1]
        if p1 == p2:
            p2 = p2+1
        s = list(range(0, min(int(p1), int(p2))))
        o = list(range(min(int(p1), int(p2)), max(int(p1), int(p2))))
        z = list(range(max(int(p1), int(p2)), x.shape[0]))

        if len(s) < 3:
            re_x = None
            re_chrom = None
            R2 = 0
            S = None
        else:
            cc1, ss1, xx1 = FR(x, s, o, z, COM)
            xx2 = x-xx1
            CC2 = C[:, 1].reshape(C.shape[0], 1)
            cc2, ss2 = ITTFA(xx2, CC2, 1)
            xx2 = np.dot(cc2, ss2)
            re_x = xx1+xx2

            re_chrom = np.zeros((x.shape[0], COM))
            re_chrom[:, 0] = np.sum(xx1, 1)
            re_chrom[:, 1] = np.sum(xx2, 1)

            R2 = explained_variance_score(x, re_x, multioutput='variance_weighted')
            # xx = [xx1, xx2]
            S = np.concatenate((ss1, ss2), 0)

    if COM == 3:
        p1 = peak_ed[0]
        p2 = peak_st[1]
        if p1 == p2:
            p2 = p2+1
        s1 = list(range(0, min(int(p1), int(p2))))
        o1 = list(range(min(int(p1), int(p2)), max(int(p1), int(p2))))
        z1 = list(range(max(int(p1), int(p2)), x.shape[0]))

        p3 = peak_ed[1]
        p4 = peak_st[2]
        if p3 == p4:
            p4 = p4+1
        s3 = list(range(int(max(int(p3), int(p4))), x.shape[0]))
        o3 = list(range(min(int(p3), int(p4)), max(int(p3), int(p4))))
        z3 = list(range(0, min(int(p3), int(p4))))

        if len(s1) < 3 or len(s3) < 3:
            re_x = None
            re_chrom = None
            R2 = 0
            S = None
        else:
            cc1, ss1, xx1 = FR(x, s1, o1, z1, COM)
            cc3, ss3, xx3 = FR(x, s3, o3, z3, COM)

            xx2 = x-xx1-xx3
            CC2 = C[:, 1].reshape(C.shape[0], 1)
            cc2, ss2 = ITTFA(xx2, CC2, 1)
            xx2 = np.dot(cc2, ss2)

            re_x = xx1+xx2+xx3

            re_chrom = np.zeros((x.shape[0], COM))
            re_chrom[:, 0] = np.sum(xx1, 1)
            re_chrom[:, 1] = np.sum(xx2, 1)
            re_chrom[:, 2] = np.sum(xx3, 1)

            R2 = explained_variance_score(x, re_x, multioutput='variance_weighted')
            # xx = [xx1, xx2, xx3]
            S = np.concatenate((ss1, ss2, ss3), 0)

    if COM == 4:
        p1 = peak_ed[0]
        p2 = peak_st[1]
        if p1 == p2:
            p2 = p2+1
        s1 = list(range(0, min(int(p1), int(p2))))
        o1 = list(range(min(int(p1), int(p2)), max(int(p1), int(p2))))
        z1 = list(range(max(int(p1), int(p2)), x.shape[0]))

        p3 = peak_ed[1]
        p4 = peak_st[2]
        if p3 == p4:
            p4 = p4+1
        s2 = list(range(min(int(p1), int(p2)), min(int(p3), int(p4))))
        o2 = list(range(min(int(p3), int(p4)), max(int(p3), int(p4))))
        z2 = list(range(max(int(p3), int(p4)), x.shape[0]))

        p5 = peak_ed[2]
        p6 = peak_st[3]
        if p5 == p6:
            p6 = p6+1
        s4 = list(range(int(max(int(p5), int(p6))), x.shape[0]))
        o4 = list(range(min(int(p5), int(p6)), max(int(p5), int(p6))))
        z4 = list(range(0, min(int(p5), int(p6))))

        if len(s1) < 3 or len(s2) < 3 or len(s4) < 3:
            re_x = None
            re_chrom = None
            R2 = 0
            S = None
        else:
            cc1, ss1, xx1 = FR(x, s1, o1, z1, COM)
            xx_3 = x-xx1

            cc2, ss2, xx2 = FR(xx_3, s2, o2, z2, int(COM-1))
            cc4, ss4, xx4 = FR(xx_3, s4, o4, z4, int(COM-1))

            xx3 = x-xx1-xx2-xx4
            CC3 = C[:, 2].reshape(C.shape[0], 1)
            cc3, ss3 = ITTFA(xx3, CC3, 1)

            xx3 = np.dot(cc3, ss3)

            re_x = xx1+xx2+xx3+xx4

            re_chrom = np.zeros((x.shape[0], COM))
            re_chrom[:, 0] = np.sum(xx1, 1)
            re_chrom[:, 1] = np.sum(xx2, 1)
            re_chrom[:, 2] = np.sum(xx3, 1)
            re_chrom[:, 3] = np.sum(xx4, 1)

            R2 = explained_variance_score(x, re_x, multioutput='variance_weighted')
            # xx = [xx1, xx2, xx3, xx4]
            S = np.concatenate((ss1, ss2, ss3, ss4), 0)

    if COM == 5:
        p1 = peak_ed[0]
        p2 = peak_st[1]
        if p1 == p2:
            p2 = p2+1
        s1 = list(range(0, min(int(p1), int(p2))))
        o1 = list(range(min(int(p1), int(p2)), max(int(p1), int(p2))))
        z1 = list(range(max(int(p1), int(p2)), x.shape[0]))

        p7 = peak_ed[3]
        p8 = peak_st[4]
        if p7 == p8:
            p8 = p8+1
        s5 = list(range(max(int(p7), int(p8)), x.shape[0]))
        o5 = list(range(min(int(p7), int(p8)), max(int(p7), int(p8))))
        z5 = list(range(0, min(int(p7), int(p8))))

        p3 = peak_ed[1]
        p4 = peak_st[2]
        if p3 == p4:
            p4 = p4+1
        s2 = list(range(min(int(p1), int(p2)), min(int(p3), int(p4))))
        o2 = list(range(min(int(p3), int(p4)), max(int(p3), int(p4))))
        z2 = list(range(max(int(p3), int(p4)), x.shape[0]))

        p5 = peak_ed[2]
        p6 = peak_st[3]
        if p5 == p6:
            p6 = p6+1
        s4 = list(range(max(int(p5), int(p6)), max(int(p7), int(p8))))
        o4 = list(range(min(int(p5), int(p6)), max(int(p5), int(p6))))
        z4 = list(range(0, min(int(p5), int(p6))))

        if len(s1) < 3 or len(s2) < 3 or len(s4) < 3 or len(s5) < 3:
            re_x = None
            re_chrom = None
            R2 = 0
            S = None
        else:
            cc1, ss1, xx1 = FR(x, s1, o1, z1, COM)
            cc5, ss5, xx5 = FR(x, s5, o5, z5, COM)

            xx_3 = x-xx1-xx5
            cc2, ss2, xx2 = FR(xx_3, s2, o2, z2, int(COM-2))
            cc4, ss4, xx4 = FR(xx_3, s4, o4, z4, int(COM-2))

            xx3 = x-xx1-xx2-xx4-xx5

            CC3 = C[:, 2].reshape(C.shape[0], 1)
            cc3, ss3 = ITTFA(xx3, CC3, 1)

            xx3 = np.dot(cc3, ss3)

            re_x = xx1+xx2+xx3+xx4+xx5

            re_chrom = np.zeros((x.shape[0], COM))
            re_chrom[:, 0] = np.sum(xx1, 1)
            re_chrom[:, 1] = np.sum(xx2, 1)
            re_chrom[:, 2] = np.sum(xx3, 1)
            re_chrom[:, 3] = np.sum(xx4, 1)
            re_chrom[:, 4] = np.sum(xx5, 1)

            R2 = explained_variance_score(x, re_x, multioutput='variance_weighted')
            # xx = [xx1, xx2, xx3, xx4, xx5]
            S = np.concatenate((ss1, ss2, ss3, ss4, ss5), 0)

    return re_x, re_chrom, R2, S


# @jit(nopython=True)
def dynamic_FRR(x, CC, peak_st, peak_ed, COM):
    C = np.copy(CC)
    ar = [i for i in range(-2, 3)]

    if COM == 2:
        ar = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        metrcis = []
        arlist = list(itertools.product(ar, ar))

        for j in range(len(arlist)):
            peak_st_1 = np.add(peak_st[1:2], arlist[j][0:1]).tolist()
            peak_ed_1 = np.add(peak_ed[0:1], arlist[j][1:2]).tolist()
            peak_st_dy = peak_st[0:1] + peak_st_1
            peak_ed_dy = peak_ed_1 + peak_ed[-1:]
            for g in range(1, len(peak_st_dy)):
                if peak_st_dy[g] < 2:
                    peak_st_dy[g] = 2
                if peak_st_dy[g] > C.shape[0]:
                    peak_st_dy[g] = C.shape[0]-1
            for g in range(0, (len(peak_ed_dy)-1)):
                if peak_ed_dy[g] > (C.shape[0]-3):
                    peak_ed_dy[g] = C.shape[0]-3
                if peak_ed_dy[g] < 2:
                    peak_ed_dy[g] = 2

            re_x, re_chrom, R2, S = full_rank_resolution(x, C, peak_st_dy, peak_ed_dy, COM)
            metrcis.append((R2, j))
            if len(metrcis) > 1:
                metrcis.remove(metrcis[np.argmin([metrcis[i][0] for i in range(len(metrcis))])])
            if metrcis[0][0] > 0.99:
                break

        peak_st_1 = np.add(peak_st[1:2], arlist[metrcis[0][1]][0:1]).tolist()
        peak_ed_1 = np.add(peak_ed[0:1], arlist[metrcis[0][1]][1:2]).tolist()
        peak_st_dy = peak_st[0:1] + peak_st_1
        peak_ed_dy = peak_ed_1 + peak_ed[-1:]
        for g in range(1, len(peak_st_dy)):
            if peak_st_dy[g] < 2:
                peak_st_dy[g] = 2
            if peak_st_dy[g] > C.shape[0]:
                peak_st_dy[g] = C.shape[0]-1
        for g in range(0, (len(peak_ed_dy)-1)):
            if peak_ed_dy[g] > (C.shape[0]-3):
                peak_ed_dy[g] = C.shape[0]-3
            if peak_ed_dy[g] < 2:
                peak_ed_dy[g] = 2
        re_x, re_chrom, R2, S = full_rank_resolution(x, C, peak_st_dy, peak_ed_dy, COM)

    if COM == 3:
        ar = [-5, -3, 0, 3, 5]
        metrcis = []
        arlist = list(itertools.product(ar, ar, ar, ar))

        for j in range(len(arlist)):
            peak_st_1 = np.add(peak_st[1:3], arlist[j][0:2]).tolist()
            peak_ed_1 = np.add(peak_ed[0:2], arlist[j][2:4]).tolist()
            peak_st_dy = peak_st[0:1] + peak_st_1
            peak_ed_dy = peak_ed_1 + peak_ed[-1:]
            for g in range(1, len(peak_st_dy)):
                if peak_st_dy[g] < 2:
                    peak_st_dy[g] = 2
                if peak_st_dy[g] > C.shape[0]:
                    peak_st_dy[g] = C.shape[0]-1
            for g in range(0, (len(peak_ed_dy)-1)):
                if peak_ed_dy[g] > (C.shape[0]-3):
                    peak_ed_dy[g] = C.shape[0]-3
                if peak_ed_dy[g] < 2:
                    peak_ed_dy[g] = 2
            re_x, re_chrom, R2, S = full_rank_resolution(x, C, peak_st_dy, peak_ed_dy, COM)
            metrcis.append((R2, j))
            if len(metrcis) > 1:
                metrcis.remove(metrcis[np.argmin([metrcis[i][0] for i in range(len(metrcis))])])
            if metrcis[0][0] > 0.99:
                break

        peak_st_1 = np.add(peak_st[1:3], arlist[metrcis[0][1]][0:2]).tolist()
        peak_ed_1 = np.add(peak_ed[0:2], arlist[metrcis[0][1]][2:4]).tolist()
        peak_st_dy = peak_st[0:1] + peak_st_1
        peak_ed_dy = peak_ed_1 + peak_ed[-1:]
        for g in range(1, len(peak_st_dy)):
            if peak_st_dy[g] < 2:
                peak_st_dy[g] = 2
            if peak_st_dy[g] > C.shape[0]:
                peak_st_dy[g] = C.shape[0]-1
        for g in range(0, (len(peak_ed_dy)-1)):
            if peak_ed_dy[g] > (C.shape[0]-3):
                peak_ed_dy[g] = C.shape[0]-3
            if peak_ed_dy[g] < 2:
                peak_ed_dy[g] = 2
        re_x, re_chrom, R2, S = full_rank_resolution(x, C, peak_st_dy, peak_ed_dy, COM)

    if COM == 4:
        metrcis = []
        arlist = list(itertools.product(ar, ar, ar, ar, ar, ar))

        for j in range(len(arlist)):
            peak_st_1 = np.add(peak_st[1:4], arlist[j][0:3]).tolist()
            peak_ed_1 = np.add(peak_ed[0:3], arlist[j][3:6]).tolist()
            peak_st_dy = peak_st[0:1] + peak_st_1
            peak_ed_dy = peak_ed_1 + peak_ed[-1:]
            for g in range(1, len(peak_st_dy)):
                if peak_st_dy[g] < 2:
                    peak_st_dy[g] = 2
                if peak_st_dy[g] > C.shape[0]:
                    peak_st_dy[g] = C.shape[0]-1
            for g in range(0, (len(peak_ed_dy)-1)):
                if peak_ed_dy[g] > (C.shape[0]-3):
                    peak_ed_dy[g] = C.shape[0]-3
                if peak_ed_dy[g] < 2:
                    peak_ed_dy[g] = 2
            re_x, re_chrom, R2, S = full_rank_resolution(x, C, peak_st_dy, peak_ed_dy, COM)
            metrcis.append((R2, j))
            if len(metrcis) > 1:
                metrcis.remove(metrcis[np.argmin([metrcis[i][0] for i in range(len(metrcis))])])
            if metrcis[0][0] > 0.99:
                break

        peak_st_1 = np.add(peak_st[1:4], arlist[metrcis[0][1]][0:3]).tolist()
        peak_ed_1 = np.add(peak_ed[0:3], arlist[metrcis[0][1]][3:6]).tolist()
        peak_st_dy = peak_st[0:1] + peak_st_1
        peak_ed_dy = peak_ed_1 + peak_ed[-1:]
        for g in range(1, len(peak_st_dy)):
            if peak_st_dy[g] < 2:
                peak_st_dy[g] = 2
            if peak_st_dy[g] > C.shape[0]:
                peak_st_dy[g] = C.shape[0]-1
        for g in range(0, (len(peak_ed_dy)-1)):
            if peak_ed_dy[g] > (C.shape[0]-3):
                peak_ed_dy[g] = C.shape[0]-3
            if peak_ed_dy[g] < 2:
                peak_ed_dy[g] = 2
        re_x, re_chrom, R2, S = full_rank_resolution(x, C, peak_st_dy, peak_ed_dy, COM)

    if COM == 5:
        metrcis = []
        arlist = list(itertools.product(ar, ar, ar, ar, ar, ar, ar, ar))

        for j in range(len(arlist)):
            peak_st_1 = np.add(peak_st[1:5], arlist[j][0:4]).tolist()
            peak_ed_1 = np.add(peak_ed[0:4], arlist[j][4:8]).tolist()
            peak_st_dy = peak_st[0:1] + peak_st_1
            peak_ed_dy = peak_ed_1 + peak_ed[-1:]
            for g in range(1, len(peak_st_dy)):
                if peak_st_dy[g] < 2:
                    peak_st_dy[g] = 2
                if peak_st_dy[g] > C.shape[0]:
                    peak_st_dy[g] = C.shape[0]-1
            for g in range(0, (len(peak_ed_dy)-1)):
                if peak_ed_dy[g] > (C.shape[0]-3):
                    peak_ed_dy[g] = C.shape[0]-3
                if peak_ed_dy[g] < 2:
                    peak_ed_dy[g] = 2
            re_x, re_chrom, R2, S = full_rank_resolution(x, C, peak_st_dy, peak_ed_dy, COM)
            metrcis.append((R2, j))
            if len(metrcis) > 1:
                metrcis.remove(metrcis[np.argmin([metrcis[i][0] for i in range(len(metrcis))])])
            if metrcis[0][0] > 0.99:
                break

        peak_st_1 = np.add(peak_st[1:5], arlist[metrcis[0][1]][0:4]).tolist()
        peak_ed_1 = np.add(peak_ed[0:4], arlist[metrcis[0][1]][4:8]).tolist()
        peak_st_dy = peak_st[0:1] + peak_st_1
        peak_ed_dy = peak_ed_1 + peak_ed[-1:]
        for g in range(1, len(peak_st_dy)):
            if peak_st_dy[g] < 2:
                peak_st_dy[g] = 2
            if peak_st_dy[g] > C.shape[0]:
                peak_st_dy[g] = C.shape[0]-1
        for g in range(0, (len(peak_ed_dy)-1)):
            if peak_ed_dy[g] > (C.shape[0]-3):
                peak_ed_dy[g] = C.shape[0]-3
            if peak_ed_dy[g] < 2:
                peak_ed_dy[g] = 2
        re_x, re_chrom, R2, S = full_rank_resolution(x, C, peak_st_dy, peak_ed_dy, COM)

    return re_x, re_chrom, R2, S


# @jit(float64(float32[:], float64[:]))
def WhittakerSmooth(x, w, lambda_, differences=1):
    X = np.matrix(x)
    m = X.size
    E = eye(m, format='csc')
    for i in range(differences):
        E = E[1:]-E[:-1]
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W+(lambda_*E.T*E))
    B = csc_matrix(W*X.T)
    background = spsolve(A, B)
    return np.array(background)


# @jit(nopython=False)
def airPLS(x, lambda_=500, porder=1, itermax=15):
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax+1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x-z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001*(abs(x)).sum() or i == itermax):
            break
        w[d >= 0] = 0
        w[d < 0] = np.exp(i*np.abs(d[d < 0])/dssn)
        w[0] = np.exp(i*(d[d < 0]).max()/dssn)
        w[-1] = w[0]
    return z


def PW_loss(y_true, y_pred):
    weight_high = tf.cast(50, dtype=float32)
    weight_low = tf.cast(10, dtype=float32)
    threshold = tf.cast(0.001, dtype=float32)

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    mask = tf.greater(y_true, threshold)

    a = tf.square(tf.subtract(y_true, y_pred))
    loss = tf.reduce_mean(tf.where(mask, weight_high*a, weight_low*a))

    return loss


def MAE(x1, x2):
    return np.sum(np.abs([x1[i]-x2[i] for i in range(len(x1))])) / len(x1)


def save_as_msp(filename, RT, mz_values, intensity_values):
    with open(filename, 'w') as file:
        file.write("Name: Unknown Compound\n")
        file.write(f"RT: {str(RT)}\n")
        file.write(f"Num Peaks: {len(mz_values)}\n")
        for mz, intensity in zip(mz_values, intensity_values):
            file.write(f"{mz} {intensity}\n")

def DeepPCR(work_path, modelpath, filename, figure_savepath, dist, thres, generate_image):
    # start_time = time.time()
    mat, RT, Xtest, ind_st_DeepSeg, ind_en_DeepSeg = data_process(work_path, filename, dist, thres)

    TICsum_origin = [np.sum(Xtest[i, :]) for i in range(Xtest.shape[0])]
    if max(mat['mz']) == 800:
        mz_min = math.floor(judge_min(min(mat['mz'])))
        mz_max = math.ceil(max(mat['mz']))
    else:
        mz_min = math.floor(min(mat['mz']))
        mz_max = math.ceil(judge_max(max(mat['mz'])))

    # data segment, baseline correction, chromatographic profile prediction
    x_seg_pre = np.zeros((len(ind_st_DeepSeg), 128, 800), dtype=np.float32)
    chrom_divid = []
    for i in range(len(ind_st_DeepSeg)):
        test = np.zeros((ind_en_DeepSeg[i]-ind_st_DeepSeg[i], 800),
                        dtype=float32)
        test[:, mz_min:mz_max] = Xtest[int(ind_st_DeepSeg[i]):int(ind_en_DeepSeg[i]), :]

        test_br = np.zeros_like(test)

        for j in range(test.shape[1]):
            if all(item == 0 for item in test[:, j]) is False:
                test_br[:, j] = test[:, j] - airPLS(test[:, j])

        test_br[test_br < 0] = 0
        chrom_divid.append(test_br)
        test_pre = test_br/np.max(test_br)
        dis_st = math.ceil((128-(int(ind_en_DeepSeg[i])-int(ind_st_DeepSeg[i])))/2)

        x_seg = np.zeros((128, 800), dtype=float32)
        x_seg[int(dis_st):int(dis_st+ind_en_DeepSeg[i]-ind_st_DeepSeg[i]), :] = test_pre
        x_seg_pre[i] = x_seg
    X = x_seg_pre.reshape(x_seg_pre.shape[0], x_seg_pre.shape[1], 1, x_seg_pre.shape[2])

    restored_model = tf.keras.models.load_model(modelpath,
                                                custom_objects={
                                                    "R_squared": R_squared,
                                                    "PW_loss": PW_loss})
    prechrom = restored_model.predict(X)

    predict = np.zeros((prechrom.shape[0], 128, 5), dtype=float32)
    for i in range(prechrom.shape[0]):
        predict[i, :, :] = prechrom[i, :, :, :].reshape(128, 5)

    # end_precess = time.time()
    # print('process time:', end_precess - start_time)

    peak_excel_single = []
    peak_excel_seg = []
    ms_single = []

    tol_com = [0]
    for i in tqdm(range(len(ind_st_DeepSeg)), desc="resolving processing"):
        num = i
        # print('i=', i)
        wayname = 'non_frr'
        dis_st = math.ceil((128-(int(ind_en_DeepSeg[i])-int(ind_st_DeepSeg[i])))/2)

        noise_width = 2
        if i > 0:
            if (int(ind_st_DeepSeg[i]) - int(ind_en_DeepSeg[i-1])) > noise_width:
                noise_windows = TICsum_origin[int(ind_st_DeepSeg[i]-noise_width):int(ind_st_DeepSeg[i])]
            else:
                noise_windows = TICsum_origin[int(ind_en_DeepSeg[i]):int(ind_en_DeepSeg[i]+noise_width)]
        if i == 0:
            if int(ind_st_DeepSeg[i]) > noise_width:
                noise_windows = TICsum_origin[int(ind_st_DeepSeg[i]-noise_width):int(ind_st_DeepSeg[i])]
            else:
                noise_windows = TICsum_origin[int(ind_en_DeepSeg[i]):int(ind_en_DeepSeg[i]+noise_width)]

        chrom_seg, C_0 = data_restore(num, dis_st, predict, ind_st_DeepSeg, ind_en_DeepSeg, mz_min, mz_max, chrom_divid)

# =============================================================================
#         if generate_image is True:
#             plt.figure(clear=True)
#             plt.subplot(211)
#             plt.plot(chrom_seg)
#             plt.subplot(212)
#             plt.plot(C_0)
#             plt.tight_layout()
#             plt.savefig(figure_savepath_c + '/' + 'num={}.tif'.format(i))
#             plt.close()
#             plt.cla()
#             plt.clf()
# =============================================================================

        C = peak_preprocess(C_0)

        if np.all(C == 0) is True:
            continue
        else:
            C1 = np.zeros_like(C)
            C1 = peak_trans(C, C1)
            C_in = np.copy(C1)

            C_out, halfdo = peak_halfremove(C_in)
            if halfdo == 'Done':
                C2 = np.zeros_like(C)
                C2 = peak_trans(C_out, C2)
                C_in = np.copy(C2)
            else:
                C_in = C_out

            C_out, testdo, lapdo, lapnum = testt(C_in)
            if testdo == 'Done':
                C4 = np.zeros_like(C)
                C4 = peak_trans(C_out, C4)
                C_in = np.copy(C4)
            else:
                C_in = C_out

            C_in = tail_fix(C_in)

            C_in_0 = np.copy(C_in)
            C_pre_0 = np.copy(C_in)

            COM = com_find(C_in)
            C_ed_it, St_it = ITTFA_PRO(chrom_seg, C_in, COM)

            C_out, dislapdo, dislapnum = peak_dislap(C_in, C_ed_it)
            dislapnum = list(set(dislapnum))
            C_out, dishalfdo, dishalfnum = dishalfremove(C_in, C_ed_it, threshold=0.05)
            dishalfnum = list(set(dishalfnum))

            if lapdo == 'Done':
                if dislapdo == 'Done' or dishalfdo == 'Done':
                    if len(dislapnum) != 0:
                        for k in dislapnum:
                            C_in_0[:, k] = 0
                            if k in lapnum:
                                C_pre_0[:, k] = 0

                    if len(dishalfnum) != 0:
                        for k in dishalfnum:
                            C_in_0[:, k] = 0
                            if k in lapnum:
                                C_pre_0[:, k] = 0

                    COM_C_pre_0 = com_find(C_pre_0)
                    if COM_C_pre_0 == 0:
                        reserve = np.argmax([np.max(C_in[:, h]) for h in range(COM)])
                        C_in_pre = np.zeros_like(C)
                        C_in_pre[:, reserve] = C_in[:, reserve]
                        C_pre_0 = C_in_pre

                    C_aa = np.copy(C_pre_0)
                    C_bb = np.zeros_like(C)
                    C_bb = peak_trans(C_aa, C_bb)
                    C_pre = np.copy(C_bb)

                    St_pre = np.zeros((5, chrom_seg.shape[1]))
                    for j in range(0, St_pre.shape[1]):
                        a = fnnls(np.dot(C_pre.T, C_pre), np.dot(C_pre.T, chrom_seg[:, j]), tole='None')
                        St_pre[:, j] = a['xx']

                    chrom_resol_pre = np.dot(C_pre, St_pre)
                    R2_pre = 1 - np.var(chrom_resol_pre - chrom_seg, ddof=1)/np.var(chrom_seg, ddof=1)

                    COM_C_in = com_find(C_in_0)
                    if COM_C_in == 0:
                        reserve = np.argmax([np.max(C_in[:, h]) for h in range(COM)])
                        C_in_it = np.zeros_like(C)
                        C_in_it[:, reserve] = C_in[:, reserve]
                        C_in_0 = C_in_it

                    COM = com_find(C_in_0)
                    C_in_aa = np.zeros_like(C)
                    C_in_it = peak_trans(C_in_0, C_in_aa)
                    C_ed_it, St_it = ITTFA_PRO(chrom_seg, C_in_it, COM)

                    chrom_resol_it = np.dot(C_ed_it, St_it)

                    R2_it = 1 - np.var(chrom_resol_it - chrom_seg, ddof=1)/np.var(chrom_seg, ddof=1)

                    # full rank resolutiuon
                    COM = com_find(C_pre)
                    peak_st, peak_ed = peak_find(C_pre)
                    if COM > 1:
                        if R2_it <= 0.99 and COM < 4:
                            chrom_resol_frr, C_ed_frr, R2_frr, St_frr = dynamic_FRR(chrom_seg, C_pre, peak_st, peak_ed, COM)
                            if R2_frr > R2_it:
                                chrom_resol = chrom_resol_frr
                                C_ed = C_ed_frr
                                R2 = R2_frr
                                St = St_frr
                                C_p = C_pre
                                wayname = 'frr'
                            else:
                                chrom_resol = chrom_resol_it
                                C_ed = C_ed_it
                                R2 = R2_it
                                St = St_it
                                C_p = C_in_it

                        elif COM > 3 and R2_it < 0.7 and R2_pre < 0.9:
                            chrom_resol_frr, C_ed_frr, R2_frr, St_frr = dynamic_FRR(chrom_seg, C_pre, peak_st, peak_ed, COM)
                            if R2_frr > R2_it:
                                chrom_resol = chrom_resol_frr
                                C_ed = C_ed_frr
                                R2 = R2_frr
                                St = St_frr
                                C_p = C_pre
                                wayname = 'frr'
                            else:
                                chrom_resol = chrom_resol_it
                                C_ed = C_ed_it
                                R2 = R2_it
                                St = St_it
                                C_p = C_in_it

                        else:
                            chrom_resol = chrom_resol_it
                            C_ed = C_ed_it
                            R2 = R2_it
                            St = St_it
                            C_p = C_in_it

                        if R2_pre > R2:
                            chrom_resol = chrom_resol_pre
                            C_ed = C_pre
                            R2 = R2_pre
                            St = St_pre
                            C_p = C_pre
                            wayname = 'non_frr'

                    if COM == 1:
                        chrom_resol = chrom_resol_it
                        C_ed = C_ed_it
                        R2 = R2_it
                        St = St_it
                        C_p = C_in_it
                        if R2_pre > R2_it:
                            chrom_resol = chrom_resol_pre
                            C_ed = C_pre
                            R2 = R2_pre
                            St = St_pre
                            C_p = C_pre

            if (lapdo == 'Done' and dislapdo == 'None' and dishalfdo == 'None') or (lapdo == 'None' and dislapdo == 'None' and dishalfdo == 'None'):
                C_pre = np.copy(C_pre_0)
                St_pre = np.zeros((5, chrom_seg.shape[1]))
                for j in range(0, St_pre.shape[1]):
                    a = fnnls(np.dot(C_pre.T, C_pre), np.dot(C_pre.T, chrom_seg[:, j]), tole='None')
                    St_pre[:, j] = a['xx']

                chrom_resol_pre = np.dot(C_pre, St_pre)
                R2_pre = 1 - np.var(chrom_resol_pre - chrom_seg, ddof=1)/np.var(chrom_seg, ddof=1)

                # ITTFA
                COM = com_find(C_pre)
                C_ed_it, St_it = ITTFA_PRO(chrom_seg, C_pre, COM)
                chrom_resol_it = np.dot(C_ed_it, St_it)
                R2_it = 1 - np.var(chrom_resol_it - chrom_seg, ddof=1)/np.var(chrom_seg, ddof=1)

                # full rank resolution
                COM = com_find(C_pre)
                peak_st, peak_ed = peak_find(C_pre)
                if COM > 1:
                    if R2_it <= 0.99 and COM < 4:
                        chrom_resol_frr, C_ed_frr, R2_frr, St_frr = dynamic_FRR(chrom_seg, C_pre, peak_st, peak_ed, COM)
                        if R2_frr > R2_it:
                            chrom_resol = chrom_resol_frr
                            C_ed = C_ed_frr
                            R2 = R2_frr
                            St = St_frr
                            C_p = C_pre
                            wayname = 'frr'
                        else:
                            chrom_resol = chrom_resol_it
                            C_ed = C_ed_it
                            R2 = R2_it
                            St = St_it
                            C_p = C_pre

                    elif COM > 3 and R2_it < 0.7 and R2_pre < 0.9:
                        chrom_resol_frr, C_ed_frr, R2_frr, St_frr = dynamic_FRR(chrom_seg, C_pre, peak_st, peak_ed, COM)
                        if R2_frr > R2_it:
                            chrom_resol = chrom_resol_frr
                            C_ed = C_ed_frr
                            R2 = R2_frr
                            St = St_frr
                            C_p = C_pre
                            wayname = 'frr'
                        else:
                            chrom_resol = chrom_resol_it
                            C_ed = C_ed_it
                            R2 = R2_it
                            St = St_it
                            C_p = C_pre

                    else:
                        chrom_resol = chrom_resol_it
                        C_ed = C_ed_it
                        R2 = R2_it
                        St = St_it
                        C_p = C_pre

                    if R2_pre > R2:
                        chrom_resol = chrom_resol_pre
                        C_ed = C_pre
                        R2 = R2_pre
                        St = St_pre
                        C_p = C_pre
                        wayname = 'non_frr'

                if COM == 1:
                    chrom_resol = chrom_resol_it
                    C_ed = C_ed_it
                    R2 = R2_it
                    St = St_it
                    C_p = C_pre
                    if R2_pre > R2_it:
                        chrom_resol = chrom_resol_pre
                        C_ed = C_pre
                        R2 = R2_pre
                        St = St_pre
                        C_p = C_pre

            if lapdo == 'None':
                if dislapdo == 'Done' or dishalfdo == 'Done':
                    if np.max(chrom_seg) < 10000:
                        # ITTFA
                        if len(dislapnum) != 0:
                            for k in dislapnum:
                                C_in_0[:, k] = 0
                        if len(dishalfnum) != 0:
                            for k in dishalfnum:
                                C_in_0[:, k] = 0

                        COM_C_in = com_find(C_in_0)
                        if COM_C_in == 0:
                            reserve = np.argmax([np.max(C_in[:, h]) for h in range(COM)])
                            C_in_it = np.zeros_like(C)
                            C_in_it[:, reserve] = C_in[:, reserve]
                            C_in_0 = C_in_it

                        COM = com_find(C_in_0)
                        C_in_aa = np.zeros_like(C)
                        C_in_it = peak_trans(C_in_0, C_in_aa)
                        C_ed_it, St_it = ITTFA_PRO(chrom_seg, C_in_it, COM)

                        chrom_resol_it = np.dot(C_ed_it, St_it)

                        R2_it = 1 - np.var(chrom_resol_it - chrom_seg, ddof=1)/np.var(chrom_seg, ddof=1)

                        chrom_resol = chrom_resol_it
                        C_ed = C_ed_it
                        R2 = R2_it
                        St = St_it
                        C_p = C_in_it

                    else:
                        C_pre = np.copy(C_pre_0)
                        St_pre = np.zeros((5, chrom_seg.shape[1]))
                        for j in range(0, St_pre.shape[1]):
                            a = fnnls(np.dot(C_pre.T, C_pre), np.dot(C_pre.T, chrom_seg[:, j]), tole='None')
                            St_pre[:, j] = a['xx']

                        chrom_resol_pre = np.dot(C_pre, St_pre)
                        R2_pre = 1 - np.var(chrom_resol_pre - chrom_seg, ddof=1)/np.var(chrom_seg, ddof=1)

                        # ITTFA
                        if len(dislapnum) != 0:
                            for k in dislapnum:
                                C_in_0[:, k] = 0
                        if len(dishalfnum) != 0:
                            for k in dishalfnum:
                                C_in_0[:, k] = 0

                        COM_C_in = com_find(C_in_0)
                        if COM_C_in == 0:
                            reserve = np.argmax([np.max(C_in[:, h]) for h in range(COM)])
                            C_in_it = np.zeros_like(C)
                            C_in_it[:, reserve] = C_in[:, reserve]
                            C_in_0 = C_in_it

                        COM = com_find(C_in_0)
                        C_in_aa = np.zeros_like(C)
                        C_in_it = peak_trans(C_in_0, C_in_aa)
                        C_ed_it, St_it = ITTFA_PRO(chrom_seg, C_in_it, COM)

                        chrom_resol_it = np.dot(C_ed_it, St_it)

                        R2_it = 1 - np.var(chrom_resol_it - chrom_seg, ddof=1)/np.var(chrom_seg, ddof=1)

                        # full rank resolution
                        COM = com_find(C_pre)
                        peak_st, peak_ed = peak_find(C_pre)
                        if COM > 1:
                            if R2_it <= 0.99 and COM < 4:
                                chrom_resol_frr, C_ed_frr, R2_frr, St_frr = dynamic_FRR(chrom_seg, C_pre, peak_st, peak_ed, COM)
                                if R2_frr > R2_it:
                                    chrom_resol = chrom_resol_frr
                                    C_ed = C_ed_frr
                                    R2 = R2_frr
                                    St = St_frr
                                    C_p = C_pre
                                    wayname = 'frr'
                                else:
                                    chrom_resol = chrom_resol_it
                                    C_ed = C_ed_it
                                    R2 = R2_it
                                    St = St_it
                                    C_p = C_in_it

                            elif COM > 3 and R2_it < 0.7 and R2_pre < 0.9:
                                chrom_resol_frr, C_ed_frr, R2_frr, St_frr = dynamic_FRR(chrom_seg, C_pre, peak_st, peak_ed, COM)
                                if R2_frr > R2_it:
                                    chrom_resol = chrom_resol_frr
                                    C_ed = C_ed_frr
                                    R2 = R2_frr
                                    St = St_frr
                                    C_p = C_pre
                                    wayname = 'frr'
                                else:
                                    chrom_resol = chrom_resol_it
                                    C_ed = C_ed_it
                                    R2 = R2_it
                                    St = St_it
                                    C_p = C_in_it
                            else:
                                chrom_resol = chrom_resol_it
                                C_ed = C_ed_it
                                R2 = R2_it
                                St = St_it
                                C_p = C_in_it

                            if R2_pre > R2:
                                chrom_resol = chrom_resol_pre
                                C_ed = C_pre
                                R2 = R2_pre
                                St = St_pre
                                C_p = C_pre
                                wayname = 'non_frr'

                        if COM == 1:
                            chrom_resol = chrom_resol_it
                            C_ed = C_ed_it
                            R2 = R2_it
                            St = St_it
                            C_p = C_in_it
                            if R2_pre > R2_it:
                                chrom_resol = chrom_resol_pre
                                C_ed = C_pre
                                R2 = R2_pre
                                St = St_pre
                                C_p = C_pre

            # TIC = [sum(chrom_seg[i, :]) for i in range(chrom_seg.shape[0])]
            # TIC_resol = [sum(chrom_resol[i, :]) for i in range(chrom_resol.shape[0])]
            # cos_sim = np.dot(TIC_resol, TIC) / (np.linalg.norm(TIC_resol)*np.linalg.norm(TIC))
            # mae = MAE(TIC, TIC_resol)

            COM = com_find(C_ed)

            if wayname == 'frr':
                tic_single = C_ed
            else:
                tic_single = np.zeros_like(C_ed)
                for r in range(COM):
                    chrom_single = np.dot(np.mat(C_ed[:, r:r+1]), np.mat(St[r:r+1, :]))
                    tic_single[:, r] = [np.sum(chrom_single[i, :]) for i in range(chrom_single.shape[0])]

            if generate_image is True:
# =============================================================================
#                 plt.figure(clear=True)
#                 plt.suptitle('s=' + str(round(cos_sim, 2)) + '/MAE=' + str(round(mae, 3)))
#                 plt.subplot(311)
#                 plt.plot(TIC)
#                 plt.title('original TIC')
#                 plt.subplot(312)
#                 plt.plot(TIC_resol)
#                 plt.title('resolved TIC')
#                 plt.subplot(313)
#                 plt.plot(tic_single)
#                 plt.title('components TIC')
#                 plt.tight_layout()
#                 plt.savefig(tic_savepath + '/' + 'num={}.tif'.format(i))
#                 plt.close()
#                 plt.cla()
#                 plt.clf()
# =============================================================================

                C_tic = np.copy(tic_single)
                C_tic = C_tic / np.max(C_tic)
                plt.figure(clear=True)
                plt.title('num=' + str(i))
                plt.subplot(2, 2, 1)
                plt.plot(chrom_seg)
                plt.title('original GCMS data')
                plt.subplot(2, 2, 2)
                plt.plot(chrom_resol)
                plt.title('resolved GCMS data')
                plt.subplot(2, 2, 3)
                plt.plot(C_p)
                plt.title('predictive chromatographic profile')
                plt.subplot(2, 2, 4)
                plt.plot(C_tic)
                plt.title('resolved chromatogram')
                plt.tight_layout()
                plt.savefig(figure_savepath + '/' + 'num={}.tif'.format(i))
                plt.close()
                plt.cla()
                plt.clf()

            tic = np.zeros((chrom_resol.shape[0], 1), dtype=float32)
            for a in range(chrom_resol.shape[0]):
                tic[a, :] = np.sum(chrom_resol[a, :])
            tic = tic.flatten()
            peak_area = integrate.trapz(tic)

            gap = (np.max(RT) - np.min(RT)) / (Xtest.shape[0]-1)
            rt_st = np.min(RT) + ind_st_DeepSeg[i]*gap
            rt_ed = np.min(RT) + ind_en_DeepSeg[i]*gap

            tol_com.append(COM)
            sum_com = np.sum(tol_com) - tol_com[-1]

            COM = com_find(C_ed)
            for r in range(COM):
                num_com = sum_com + (r+1)
                tic_single_area = tic_single[:, r].flatten()
                peak_area_single = integrate.trapz(tic_single_area)
                rt = np.min(RT) + (ind_st_DeepSeg[i] + np.argmax(C_ed[:, r]))*gap
                St_single = St[r, :]

                signal_windows = TICsum_origin[int(ind_st_DeepSeg[i]+np.argmax(C_ed[:, r]) - 3):int(ind_st_DeepSeg[i] + np.argmax(C_ed[:, r]) + 3)]
                scalemax = 50
                scales = np.arange(1, scalemax)
                coefficients, frequencies = pywt.cwt(signal_windows, scales, 'mexh')
                max_coefficient_index = np.unravel_index(np.argmax(np.abs(coefficients)), coefficients.shape)
                max_scale = scales[max_coefficient_index[0]]
                while max_scale == max(scales):
                    scalemax += 50
                    scales = np.arange(1, scalemax)
                    coefficients, frequencies = pywt.cwt(signal_windows, scales, 'mexh')
                    max_coefficient_index = np.unravel_index(np.argmax(np.abs(coefficients)), coefficients.shape)
                    max_scale = scales[max_coefficient_index[0]]

                noise_coefficients, noise_frequencies = pywt.cwt(noise_windows, 1, 'mexh')
                noise_intensity = np.percentile(np.abs(noise_coefficients), 95)
                snr = round(np.max(np.abs(coefficients))/noise_intensity, 2)

                ms_single.append({'ms': St_single})
                peak_excel_single.append({'#number': num_com, 'rt': round(rt,5), 'peak area': peak_area_single, 'COM': COM, 'R2': R2, 'SNR': snr})

            peak_excel_seg.append({'num': i+1, 'RT_st': round(rt_st, 5), 'RT_ed': round(rt_ed, 5), 'R2': round(R2, 5), 'PA': peak_area})

    return peak_excel_single, peak_excel_seg, ms_single


def data_resolution(dataset_path, DeepCS_path, DeepCPR_path, save_path, generate_image):
    # start = time.time()
    px_savepath1 = save_path + '/single'
    px_savepath2 = save_path + '/seg'
    ms_savepath = save_path + '/ms'

    if not os.path.exists(px_savepath1):
        os.makedirs(px_savepath1)
    if not os.path.exists(px_savepath2):
        os.makedirs(px_savepath2)
    if not os.path.exists(ms_savepath):
        os.makedirs(ms_savepath)

    files = os.listdir(dataset_path)
    for file in files:
        tf.keras.backend.clear_session()
        ops.reset_default_graph()

        filename = os.path.join(dataset_path, file)
        file_pre = file.split('.')[0]

        if generate_image is True:
            figure_savepath = save_path + '/figure/' + file_pre
            # tic_savepath = save_path + '/tic/' + file_pre
            if not os.path.exists(figure_savepath):
                os.makedirs(figure_savepath)
            # if not os.path.exists(tic_savepath):
            #     os.makedirs(tic_savepath)
        else:
            figure_savepath = None

        print('file loading:', file)

        peak_excel_single, peak_excel_seg, ms_single = DeepPCR(DeepCS_path, DeepCPR_path, filename, figure_savepath, dist=3, thres=15, generate_image=generate_image)
        # end = time.time()
        # print('resolv time:', end-start)
        pe_single = pd.DataFrame(peak_excel_single)
        pe_single.to_csv(px_savepath1 + '/' + file_pre + '.csv', index=False)
        pe_seg = pd.DataFrame(peak_excel_seg)
        pe_seg.to_csv(px_savepath2 + '/' + file_pre + '.csv', index=False)

        if not os.path.exists(ms_savepath + '/' + file_pre):
            os.makedirs(ms_savepath + '/' + file_pre)
        for i in range(len(ms_single)):
            ms_single_com = ms_single[i]['ms']
            RT = peak_excel_single[i]['rt']
            # ms_temp = np.zeros((ms_single_com.shape[0], 2))
            mz_values = np.arange(1, ms_single_com.shape[0]+1)
            intensity_values = ms_single_com
            # ms = pd.DataFrame(ms_temp)
            # ms.to_csv(ms_savepath + '/' + file_pre + '/' + str(i) + '.csv', index=False, header=False)
            save_as_msp(ms_savepath + '/' + file_pre + '/' + str(i) + '.msp', RT, mz_values, intensity_values)

        del peak_excel_single, peak_excel_seg, ms_single, pe_single, pe_seg
        gc.collect()
        print('\n')

if __name__ == '__main__':
    start = time.perf_counter()

    work_path = 'C:/Users/ZNDX001/Documents/Python_Scripts/DeepResolution2-main/DeepResolution2/model/UNet4S/model.h5'
    modelpath = 'C:/DeepEER2/model_out/23_04_18/SepConv/optimize64145/model.h5'

    save_main = 'C:/DeepEER2/model_out/23_04_18/SepConv/optimize64145'
    data_path = "C:/Users/ZNDX001/Documents/9-17色谱分辨工作/data/aa"

    px_savepath1 = save_main + '/single'
    px_savepath2 = save_main + '/seg'
    ms_savepath = save_main + '/ms'

    if not os.path.exists(px_savepath1):
        os.makedirs(px_savepath1)
    if not os.path.exists(px_savepath2):
        os.makedirs(px_savepath2)
    if not os.path.exists(ms_savepath):
        os.makedirs(ms_savepath)

    files = os.listdir(data_path)
    for file in files:
        tf.keras.backend.clear_session()
        ops.reset_default_graph()

        filename = os.path.join(data_path, file)
        file_pre = file.split('.')[0]

        figure_savepath_c = save_main + '/figure-c2/' + file_pre
        figure_savepath = save_main + '/figure/' + file_pre
        tic_savepath = save_main + '/tic/' + file_pre
        if not os.path.exists(figure_savepath_c):
            os.makedirs(figure_savepath_c)
        if not os.path.exists(figure_savepath):
            os.makedirs(figure_savepath)
        if not os.path.exists(tic_savepath):
            os.makedirs(tic_savepath)

        print('file processing:', file)
        print('data loading......')

        peak_excel_single, peak_excel_seg, ms_single = DeepPCR(work_path, modelpath, filename, dist=3, thres=15, generate_image=False)
        print(peak_excel_single)
# =============================================================================
#         pe_single = pd.DataFrame(peak_excel_single)
#         pe_single.to_excel(px_savepath1 + '/' + file_pre + '.xlsx', index=False)
#         pe_seg = pd.DataFrame(peak_excel_seg)
#         pe_seg.to_excel(px_savepath2 + '/' + file_pre + '.xlsx', index=False)
#         ms = pd.DataFrame(ms_single)
#         ms.to_pickle(ms_savepath + '/' + file_pre + '.pkl')
# 
#         del peak_excel_single, peak_excel_seg, ms_single, pe_single, pe_seg, ms
#         gc.collect()
# 
#         print('\n')
# =============================================================================

    end = time.perf_counter()
    print('Runtime:', end-start, 's')



































