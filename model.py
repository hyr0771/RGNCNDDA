import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self,in_dim,out_dim,drop=0.5,bias=False,activation=None):
        super(GraphConv,self).__init__()
        self.dropout = nn.Dropout(drop)
        self.activation = activation
        self.w = nn.Linear(in_dim,out_dim,bias=bias)
        nn.init.xavier_uniform_(self.w.weight)
        self.bias = bias
        if self.bias:
            nn.init.zeros_(self.w.bias)
    
    def forward(self,adj,x):
        x = self.dropout(x)
        x = adj.mm(x)
        x = self.w(x)
        if self.activation:
            return self.activation(x)
        else:
            return x



class APPNP(nn.Module):
    def __init__(self,k,alpha,dropout):
        super(APPNP,self).__init__()
        self.alpha = alpha
        self.k = k
        self.dropout = dropout
    def forward(self,adj,H):
        Z = H
        for _ in range(self.k):
            Z = F.dropout(Z,self.dropout,training=self.training)

            Z = (1-self.alpha)*torch.mm(adj,Z) + self.alpha * H
        return Z



class LPN(nn.Module):
    def __init__(self, hid_dim, out_dim, bias=False):
        super(LPN, self).__init__()
        self.res1 = GraphConv(out_dim, hid_dim, bias=bias, activation=F.relu)
        self.res1_ = GraphConv(hid_dim, hid_dim, bias=bias, activation=torch.relu)
        self.res2_ = GraphConv(hid_dim, hid_dim, bias=bias, activation=F.relu)
        self.res2 = GraphConv(hid_dim, out_dim, bias=bias, activation=torch.sigmoid)
        self.propagate = APPNP(k=3, alpha=0.5, dropout=0.5)

    def forward(self, g, z):
        z1 = self.res1(g, z)
        z_ = self.res1_(g, z1)
        z = torch.tensor(z_)
        z = F.normalize(z, p=2,dim=1)
        z = self.propagate(g,z)
        res = self.res2(g, z_)
        return res, z



class AEN(nn.Module):
    def __init__(self, feat_dim, hid_dim, out_dim, bias=False):
        super(AEN, self).__init__()
        self.conv1 = GraphConv(feat_dim, hid_dim, bias=bias, activation=F.relu)
        self.Res = GraphConv(hid_dim, out_dim, bias=bias, activation=F.relu)
        self.mu = GraphConv(hid_dim, out_dim, bias=bias, activation=torch.sigmoid)
        self.logvar = GraphConv(hid_dim, out_dim, bias=bias, activation=torch.sigmoid)
        self.conv3 = GraphConv(out_dim, hid_dim, bias=bias, activation=F.relu)
        self.conv4 = GraphConv(hid_dim, feat_dim, bias=bias, activation=torch.sigmoid)
        self.propagate = APPNP(k=3, alpha=0.5, dropout=0.5)

    def encoder(self, g, x):
        x = self.conv1(g, x)
        h = self.mu(g, x)
        std = self.logvar(g, x)
        std = F.normalize(std, p=2,dim=1)
        std = self.propagate(g,std)
        mu = self.propagate(g, h+x)
        return mu, std

    def decoder(self, g, x):
        x = self.conv3(g, x)
        x = self.conv4(g, x)
        return x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, g, x):
        mu, logvar = self.encoder(g, x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, self.decoder(g, z)