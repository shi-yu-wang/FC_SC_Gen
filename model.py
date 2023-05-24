import torch
import torch.nn as nn
import math
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCNLayer(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        self.MLP_GIN = nn.Sequential(
            nn.Linear(self.in_feature, self.out_feature),
            nn.ELU()
            ).cuda()
        
    def forward(self, A, X):
        z = torch.matmul(A, X)
        z_new = self.MLP_GIN(z)
        return z_new
        

class MolGen(nn.Module):
    def __init__(self, nlayer_gcn, nnodes, nfeature_mlp, nlatent, tau):
        super(MolGen, self).__init__()
        
        self.nlayer_gcn = nlayer_gcn
        self.nnodes = nnodes
        self.nfeature_gcn = nnodes
        self.nfeature_mlp = nfeature_mlp
        self.nlatent = nlatent
        self.tau = tau
        
        # encoders
        self.gcn_fc = torch.nn.ModuleList()
        for _ in range(self.nlayer_gcn):
            self.gcn_fc.append(GCNLayer(self.nnodes, self.nnodes))
            
        self.gcn_joint = torch.nn.ModuleList()
        for _ in range(self.nlayer_gcn):
            self.gcn_joint.append(GCNLayer(self.nnodes, self.nnodes))
            
        self.gcn_sc = torch.nn.ModuleList()
        for _ in range(self.nlayer_gcn):
            self.gcn_sc.append(GCNLayer(self.nnodes, self.nnodes))
            
        # mu and sigma
        self.mu_fc = nn.Sequential(
            nn.Linear(self.nfeature_gcn, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nlatent))
        self.sigma_fc = nn.Sequential(
            nn.Linear(self.nfeature_gcn, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nlatent))
        self.mu_joint = nn.Sequential(
            nn.Linear(self.nfeature_gcn, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nlatent))
        self.sigma_joint = nn.Sequential(
            nn.Linear(self.nfeature_gcn, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nlatent))
        self.mu_sc = nn.Sequential(
            nn.Linear(self.nfeature_gcn, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nlatent))
        self.sigma_sc = nn.Sequential(
            nn.Linear(self.nfeature_gcn, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nlatent))
        
        # decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.nlatent * 2, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nnodes)
            )
        
        self.decoder_sc = nn.Sequential(
            nn.Linear(self.nlatent * 2, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nfeature_mlp),
            nn.ELU(),
            nn.BatchNorm1d(self.nfeature_mlp),
            nn.Linear(self.nfeature_mlp, self.nnodes)
            )
        
        # property prediction
        self.prop_pred1 = nn.Sequential(
            nn.Linear(self.nlatent * 3, self.nfeature_mlp),
            nn.ELU(),
            nn.Linear(self.nfeature_mlp, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Linear(16, 4),
            )
        
        self.prop_pred2 = nn.Sequential(
            nn.Linear(self.nlatent * 3, self.nfeature_mlp),
            nn.ELU(),
            nn.Linear(self.nfeature_mlp, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 8),
            nn.ELU(),
            nn.Linear(8, 1),
            )
        
        self.mask1 = torch.nn.Parameter(torch.randn(3*self.nlatent, 2)).to(device)
        self.mask2 = torch.nn.Parameter(torch.randn(3*self.nlatent, 2)).to(device)
        
        self.mse = nn.MSELoss()
        
    def mu_sigma(self, z, fmu, fsigma):        
        z_mu = fmu(z)
        z_sigma = fsigma(z)
        
        z_new = z_mu + torch.randn(z_sigma.size()).to(device) * torch.exp(z_sigma)
        return z_new, z_mu, z_sigma
    
    def forward(self, A_fc, A_sc, age, gender):
        X = torch.eye(A_sc.shape[0]).float().to(device)
        
        # encoder
        ## fc
        z_fc = X
        for layer in self.gcn_fc:
            z_fc = layer(A_fc, z_fc)
        z_fc, z_mu_fc, z_sigma_fc = self.mu_sigma(z_fc, self.mu_fc, self.sigma_fc)

        
        ## sc
        z_sc = X
        for layer in self.gcn_sc:
            z_sc = layer(A_sc, z_sc)
        z_sc, z_mu_sc, z_sigma_sc = self.mu_sigma(z_sc, self.mu_sc, self.sigma_sc)

        
        ## joint
        A_joint = A_fc + A_sc / 2
        z_joint = X
        for layer in self.gcn_joint:
            z_joint = layer(A_joint, z_joint)
        z_joint, z_mu_joint, z_sigma_joint = self.mu_sigma(z_joint, self.mu_joint, self.sigma_joint)

        
        # decoder
        ## generate fc
        z_fc_gen = torch.cat((z_fc, z_joint), dim = 1)
        z_fc_gen = self.decoder_fc(z_fc_gen)
        z_fc_gen = torch.sum(z_fc_gen, dim = 1).view(-1, 1)
        A_fc_gen = torch.sigmoid(torch.mul(z_fc_gen, z_fc_gen.t()))
        ## generate sc
        z_sc_gen = torch.cat((z_sc, z_joint), dim = 1)
        z_sc_gen = self.decoder_fc(z_sc_gen)
        z_sc_gen = torch.sum(z_sc_gen, dim = 1).view(-1, 1)
        A_sc_gen = torch.sigmoid(torch.mul(z_sc_gen, z_sc_gen.t()))
        
        loss_graph_rec = self.mse(A_fc_gen, A_fc) + self.mse(A_sc_gen, A_sc)
        
        
        z_prop = torch.cat((torch.sum(z_fc, dim = 0).view(1, -1), torch.sum(z_joint, dim = 0).view(1, -1), torch.sum(z_sc, dim = 0).view(1, -1)), dim = 1)
        # property 1 prediction
        logit1 = torch.sigmoid(self.mask1) / (1 - torch.sigmoid(self.mask1))
        mask1 = F.gumbel_softmax(logit1.to(device), self.tau, hard=True)[:, 1].view(1, -1)
        z_prop1 = mask1 * z_prop
        prop1 = F.softmax(self.prop_pred1(z_prop1).view(1, -1))
        
        
        # property 2 prediction
        logit2 = torch.sigmoid(self.mask2) / (1 - torch.sigmoid(self.mask2))
        mask2 = F.gumbel_softmax(logit2.to(device), self.tau, hard=True)[:, 1].view(1, -1)
        z_prop2 = mask2 * z_prop
        prop2 = F.softmax(self.prop_pred2(z_prop2).view(-1))

        loss_prop_rec = self.mse(age, prop1) + self.mse(gender, prop2)
        
        # loss_prop_rec = F.binary_cross_entropy(age, prop1) + F.binary_cross_entropy(gender, prop2)
        
        log_qfsj, log_prod_qfqsqj = self.get_tc(z_fc, z_sc, z_joint, z_mu_fc, z_sigma_fc, z_mu_sc, z_sigma_sc, z_mu_joint, z_sigma_joint)
        loss_kl_disentangle = (log_qfsj - log_prod_qfqsqj).mean()
        
        latent_dist=(torch.cat([z_mu_fc, z_mu_sc, z_mu_joint],dim=-1),torch.cat([z_sigma_fc, z_sigma_sc, z_sigma_joint],dim=-1))
        loss_kl = self._kl_normal_loss(*latent_dist)

        
        return loss_graph_rec, loss_prop_rec, loss_kl_disentangle, loss_kl, prop1, prop2, mask1, mask2
    
    def _kl_normal_loss(self, mean, logvar):
        # latent_dim = mean.size(1)
        # batch mean of kl for each latent dimension
        latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
        total_kl = latent_kl.sum()
    
        return total_kl
    
    
    def get_tc(self, z_fc, z_sc, z_joint, mu_fc, sigma_fc, mu_sc, sigma_sc, mu_joint, sigma_joint):
        latent_sample = torch.cat((z_fc, z_sc, z_joint), dim = -1)
        mat_log_qzfsj = self.matrix_log_density_gaussian(latent_sample, torch.cat((mu_fc, mu_sc, mu_joint), dim = -1), torch.cat((sigma_fc, sigma_sc, sigma_joint), dim = -1))
        mat_log_qz_fc = self.matrix_log_density_gaussian(z_fc, mu_fc, sigma_fc)
        mat_log_qz_sc = self.matrix_log_density_gaussian(z_sc, mu_sc, sigma_sc)
        mat_log_qz_joint = self.matrix_log_density_gaussian(z_joint, mu_joint, sigma_joint)
        
        log_qfsj = torch.logsumexp(mat_log_qzfsj.sum(2), dim=1, keepdim=False)
        log_qf = torch.logsumexp(mat_log_qz_fc.sum(2), dim=1, keepdim=False)
        log_qs = torch.logsumexp(mat_log_qz_sc.sum(2), dim=1, keepdim=False)
        log_qj = torch.logsumexp(mat_log_qz_joint.sum(2), dim=1, keepdim=False)
        log_prod_qfqsqj = log_qf + log_qs + log_qj
        
        return log_qfsj, log_prod_qfqsqj
    
    def log_density_gaussian(self, x, mu, logvar):
        """Calculates log density of a Gaussian.
        Parameters
        ----------
        x: torch.Tensor or np.ndarray or float
            Value at which to compute the density.
        mu: torch.Tensor or np.ndarray or float
            Mean.
        logvar: torch.Tensor or np.ndarray or float
            Log variance.
        """
        normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
        inv_var = torch.exp(-logvar)
        log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
        return log_density
    
    def matrix_log_density_gaussian(self, x, mu, logvar):
        """Calculates log density of a Gaussian for all combination of bacth pairs of
        `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
        instead of (batch_size, dim) in the usual log density.
        Parameters
        ----------
        x: torch.Tensor
            Value at which to compute the density. Shape: (batch_size, dim).
        mu: torch.Tensor
            Mean. Shape: (batch_size, dim).
        logvar: torch.Tensor
            Log variance. Shape: (batch_size, dim).
        batch_size: int
            number of training images in the batch
        """
        batch_size, dim = x.shape
        x = x.view(batch_size, 1, dim)
        mu = mu.view(1, batch_size, dim)
        logvar = logvar.view(1, batch_size, dim)
        
        return self.log_density_gaussian(x, mu, logvar)
    