import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import minmax_scale,scale
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,precision_recall_curve,auc
import scipy.io as scio
import matplotlib.pyplot as plt
import argparse

def scaley(ymat):
    return (ymat-ymat.min())/ymat.max()

def set_seed(seed,cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def load_data(data,cuda):

    disease = pd.read_csv('dis_sim.csv',header=None, encoding='gb18030').values

    drug_disease = pd.read_csv('drug_dis.csv',header=None, encoding='gb18030').values

    drug = pd.read_csv('drug_sim.csv',header=None, encoding='gb18030').values
    drug = minmax_scale(drug,axis=0)
    disease = minmax_scale(disease,axis=0)

    diseaset = torch.from_numpy(disease).float()
    drug_diseaset = torch.from_numpy(drug_disease).float()
    drugt = torch.from_numpy(drug).float()

    gdurg = norm_adj(drug)
    gdisease = norm_adj(disease.T)

    if cuda:
        diseaset = diseaset.cuda()
        drug_diseaset = drug_diseaset.cuda()
        drugt = drugt.cuda()
        gdurg = gdurg.cuda()
        gdisease = gdisease.cuda()
    print('data.shape = ',drug_disease.shape)
    return diseaset, drug_diseaset, drugt, gdurg, gdisease

def load_mat(filepath, cuda):
    mat = scio.loadmat(filepath)

    drug = mat["drug"].astype(float)

    disease = mat["disease"].astype(float)
    drug = minmax_scale(drug,axis=0)
    disease = minmax_scale(disease,axis=0)

    drug_disease = mat["didr"]

    diseaset = torch.from_numpy(disease).float()
    drug_diseaset = torch.from_numpy(drug_disease).float()
    drug_diseaset = drug_diseaset.T
    drugt = torch.from_numpy(drug).float()

    gdurg = norm_adj(drug)

    gdisease = norm_adj(disease.T)
    if cuda:
        diseaset = diseaset.cuda()
        drug_diseaset = drug_diseaset.cuda()
        drugt = drugt.cuda()
        gdurg = gdurg.cuda()
        gdisease = gdisease.cuda()
    return diseaset, drug_diseaset, drugt, gdurg, gdisease

def load_lrssl(filepath,  cuda):
    reduce = True
    drug_chemical = pd.read_csv(filepath + "lrssl_simmat_dc_chemical.txt", sep="\t", index_col=0)
    drug_dataset = pd.read_csv(filepath + "lrssl_simmat_dc_domain.txt", sep="\t", index_col=0)
    drug_go = pd.read_csv(filepath + "lrssl_simmat_dc_go.txt", sep="\t", index_col=0)
    disease_sim = pd.read_csv(filepath + "lrssl_simmat_dg.txt", sep="\t", index_col=0)
    if reduce:
        drug_sim = (drug_chemical+drug_dataset+drug_go)/3
    else:
        drug_sim = drug_chemical
    drug_disease = pd.read_csv(filepath + "lrssl_admat_dgc.txt", sep="\t", index_col=0).T
    drug_disease = drug_disease.T

    rr = drug_sim.to_numpy(dtype=np.float32)
    rd = drug_disease.to_numpy(dtype=np.float32)
    dd = disease_sim.to_numpy(dtype=np.float32)

    drug = rr.astype(float)
    disease = dd.astype(float)
    drug = minmax_scale(drug,axis=0)
    disease = minmax_scale(disease,axis=0)

    drug_disease = rd.T
    diseaset = torch.from_numpy(disease).float()
    drug_diseaset = torch.from_numpy(drug_disease).float()
    drug_diseaset = drug_diseaset.T
    drugt = torch.from_numpy(drug).float()

    gdurg = norm_adj(drug)

    gdisease = norm_adj(disease.T)

    if cuda:
        diseaset = diseaset.cuda()
        drug_diseaset = drug_diseaset.cuda()
        drugt = drugt.cuda()
        gdurg = gdurg.cuda()
        gdisease = gdisease.cuda()
    return diseaset, drug_diseaset, drugt, gdurg, gdisease



def neighborhood(feat,k):
    # compute C
    featprod = np.dot(feat.T,feat)
    smat = np.tile(np.diag(featprod),(feat.shape[1],1))
    dmat = smat + smat.T - 2*featprod
    dsort = np.argsort(dmat)[:,1:k+1]
    C = np.zeros((feat.shape[1],feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i,j] = 1.0
    
    return C

def normalized(wmat):
    deg = np.diag(np.sum(wmat,axis=0))
    degpow = np.power(deg,-0.5)
    degpow[np.isinf(degpow)] = 0
    W = np.dot(np.dot(degpow,wmat),degpow)
    return W

def norm_adj(feat):
    C = neighborhood(feat.T,k=10)
    norm_adj = normalized(C.T*C+np.eye(C.shape[0]))
    g = torch.from_numpy(norm_adj).float()
    return g

def show_auc(ymat,data):

    drug_disease = data.detach().cpu().numpy()
    y_true = drug_disease.flatten()
    ymat = ymat.flatten()
    fpr,tpr,rocth = roc_curve(y_true,ymat)
    auroc = auc(fpr,tpr)



    precision,recall,prth = precision_recall_curve(y_true,ymat)
    aupr = auc(recall,precision)



    print('AUROC= %.4f | AUPR= %.4f' % (auroc,aupr))


    return auroc,aupr