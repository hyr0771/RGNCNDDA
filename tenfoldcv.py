import torch.nn as nn
import torch.nn.functional as F

from model import GraphConv,LPN,AEN
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Dimension of representations')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='Weight between lncRNA space and disease space')
parser.add_argument('--data', type=int, default=1, choices=[1,2],
                    help='Dataset')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

set_seed(args.seed,args.cuda)

# path = "./dataset/Cdataset.mat" #Fdataset,Cdataset
path = "./dataset/lrssl/" #LRSSL

# disease, drug_disease, drug, gdr, gds = load_mat(path,args.cuda) #Fdataset,Cdataset
disease, drug_disease, drug, gdr, gds = load_lrssl(path, args.cuda) #lrssl

class RGNCNq(nn.Module):
    def __init__(self):
        super(RGNCNq,self).__init__()
        self.gnnqdr = AEN(drug.shape[1],256,args.hidden)
        self.gnnqd = AEN(disease.shape[0],256,args.hidden)
    
    def forward(self,xdr0,xds0):
        hdr,stddr,xdr = self.gnnqdr(gdr,xdr0)
        hds,stdds,xds = self.gnnqd(gds,xds0)

        return hdr,stddr,xdr,hds,stdds,xds

class GNCNp(nn.Module):
    def __init__(self):
        super(GNCNp,self).__init__()
        self.gnnpdr =LPN(args.hidden,drug_disease.shape[1])
        self.gnnpds =LPN(args.hidden,drug_disease.shape[0])

    def forward(self,y0):
        ydr,zdr = self.gnnpdr(gdr,y0)
        yds,zds = self.gnnpds(gds,y0.t())

        return ydr,zdr,yds,zds

print(path+" 10-fold CV")



def criterion(output,target,msg,n_nodes,mu,logvar):
    if msg == 'disease':
        cost = F.binary_cross_entropy(output,target)
    else:
        cost = F.mse_loss(output,target)
    
    KL = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KL



def train(rgncnq,gncnp,xdr0,xds0,y0,epoch,alpha):

    beta0 = 0.5
    gamma0 = 1.0


    optp = torch.optim.Adam(gncnp.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    optq = torch.optim.Adam(rgncnq.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    for e in range(epoch):
        rgncnq.train()
        hdr,stddr,xdr,hds,stdds,xds = rgncnq(xdr0,xds0)
        lossqdr = criterion(xdr,xdr0,
            "drug",gdr.shape[0],hdr,stddr)
        lossqds = criterion(xds,xds0,
            "disease",gds.shape[0],hds,stdds)
        lossq = alpha*lossqdr + (1-alpha)*lossqds + beta0*e*F.mse_loss(
            torch.mm(hdr,hds.t()),y0)/epoch
        optq.zero_grad()
        lossq.backward()
        optq.step()
        rgncnq.eval()
        with torch.no_grad():
            hdr,_,_,hds,_,_ = rgncnq(xdr0,xds0)
        
        gncnp.train()
        ydr,zdr,yds,zds = gncnp(y0)
        losspdr = F.binary_cross_entropy(ydr,y0) + gamma0*e*F.mse_loss(zdr,hdr)/epoch
        losspds = F.binary_cross_entropy(yds,y0.t()) + gamma0*e*F.mse_loss(zds,hds)/epoch
        lossp = alpha*losspdr + (1-alpha)*losspds
        optp.zero_grad()
        lossp.backward()
        optp.step()

        gncnp.eval()
        with torch.no_grad():
            ydr,_,yds,_ = gncnp(y0)
        
        if e%20 == 0:
            print('Epoch %d | Lossp: %.4f | Lossq: %.4f' % (e, lossp.item(),lossq.item()))
        
    return alpha*ydr+(1-alpha)*yds.t()



def tenfoldcv(A,alpha):
    N = A.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    res = torch.zeros(10,A.shape[0],A.shape[1])
    aurocl = np.zeros(10)
    auprl = np.zeros(10)
    for i in range(10):
        print("Fold {}".format(i+1))
        A0 = A.clone()
        for j in range(i*N//10,(i+1)*N//10):
            A0[idx[j],:] = torch.zeros(A.shape[1])
        
        rgncnq = RGNCNq()
        gncnp = GNCNp()
        if args.cuda:
            rgncnq = rgncnq.cuda()
            gncnp = gncnp.cuda()

        train(rgncnq,gncnp,drug,disease.t(),A0,args.epochs,args.alpha)
        rgncnq.eval()
        gncnp.eval()
        yli,_,ydi,_ = gncnp(A0)
        resi = alpha*yli + (1-alpha)*ydi.t()
        #resi = scaley(resi)
        res[i] = resi
        
        if args.cuda:
            resi = resi.cpu().detach().numpy()
        else:
            resi = resi.detach().numpy()
        
        auroc,aupr = show_auc(resi,A)
        aurocl[i] = auroc
        auprl[i] = aupr
        print(np.mean(aurocl))
        print(np.mean(auprl))
        
    ymat = res[auprl.argmax()]
    if args.cuda:
        return ymat.cpu().detach().numpy()
    else:
        return ymat.detach().numpy()

title = 'result--dataset'+str(drug_disease.shape)
ymat =tenfoldcv(drug_disease,alpha=args.alpha)
title += '--tenfoldcv'
ymat = scaley(ymat)
np.savetxt(title+'.csv',ymat,fmt='%10.5f',delimiter=',')
print("===Final result===")
show_auc(ymat,drug_disease)