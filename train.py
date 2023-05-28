# -*- coding: utf-8 -*-
"""
Created on Sun May  7 14:48:25 2023

@author: Shiyu
"""
import torch
import numpy as np
import pandas as pd
from model import *
import matplotlib.pyplot as plt
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = np.load('SC_FC_demos.npz')

ages = data["Ages"]
ages = list(pd.factorize(ages)[0])
ages_tensor = [torch.zeros(4).view(1, -1) for i in range(len(ages))]
for i in range(len(ages)):
    ages_tensor[i][0, ages[i]] = 1

fcs = torch.from_numpy(data["FCs"])
gender = data["Gender"]
gender = pd.factorize(gender)[0]
gender = torch.from_numpy(gender)
ids = torch.from_numpy(data["ids"])
scs = torch.from_numpy(data["SCs"])

model = MolGen(nlayer_gcn=2, nnodes=scs[0, :, :].shape[0], nfeature_mlp=256, nlatent=256, tau=10).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

epochs = 3000
epoch = 0
i = 0

mask_loss = []
loss_rec = []
kl_rec = []
kl_dis = []
graph_rec = []
prop_rec = []


def accuracy(pred, label):
    correct_predictions = 0
    total_examples = len(pred)
    for pred_label, gold_label in zip(pred, label):
        if pred_label == gold_label:
            correct_predictions += 1
    return correct_predictions / total_examples


# train test indices split
indices = list(range(len(ids)))
random.shuffle(indices)
train_indices = indices[int(0.2 * len(ids)):]
val_indices = indices[:int(0.2 * len(ids))]

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_rec = 0
    total_kl = 0
    total_dis_kl = 0
    total_prop_rec = 0
    total_l1 = 0
    preds1 = []
    preds2 = []
    for i in train_indices:
        optimizer.zero_grad()
        # print(model.mu_fc.weight.grad)
        # print(model.mu_fc)
        loss_graph_rec, loss_prop_rec, loss_kl_disentangle, loss_kl, prop1, prop2, mask1, mask2 = model(
            A_fc=fcs[i, :, :].float().to(device), A_sc=scs[i, :, :].float().to(device),
            age=ages_tensor[i].float().to(device), gender=gender[i].view(-1).float().to(device))
        loss_norm = torch.sum(mask1) + torch.sum(mask2)

        loss = loss_graph_rec + 5 * loss_prop_rec + loss_kl + loss_kl_disentangle + loss_norm
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_kl += loss_kl.item()
        total_dis_kl += loss_kl_disentangle.item()
        total_rec += loss_graph_rec.item()
        total_prop_rec += loss_prop_rec.item()
        total_l1 += loss_norm.item()

        # age prediction
        _, pred1 = torch.max(prop1.data, 1)
        preds1.append(pred1)

        # gender prediction
        pred2 = (prop2 > 0.5).item()
        print(prop2)
        preds2.append(pred2)

        # print(str(epoch)+' loss:'+str(total_loss/len(ids))+' disentangle kl:'+str(total_dis_kl/len(ids))+' kl:'+str(total_kl/len(ids))+' rec:'+str(total_rec/len(ids))+' property rec:'+str(total_prop_rec/len(ids))+' l1 norm:'+str(loss_norm))
    mask_loss.append(total_l1 / len(ids))
    loss_rec.append(total_loss / len(ids))
    kl_rec.append(total_kl / len(ids))
    kl_dis.append(total_dis_kl / len(ids))
    graph_rec.append(total_rec / len(ids))
    prop_rec.append(total_prop_rec / len(ids))

    # print(preds2,gender.tolist())
    train_acc1 = accuracy(preds1, [ages[i] for i in train_indices])
    train_acc2 = accuracy(preds2, [gender.tolist()[i] for i in train_indices])

    model.eval()
    with torch.no_grad():
        val_preds1 = []
        val_preds2 = []
        for i in val_indices:
            optimizer.zero_grad()
            loss_graph_rec, loss_prop_rec, loss_kl_disentangle, loss_kl, prop1, prop2, mask1, mask2 = model(
                A_fc=fcs[i, :, :].float().to(device), A_sc=scs[i, :, :].float().to(device),
                age=ages_tensor[i].float().to(device), gender=gender[i].view(-1).float().to(device))
            # age prediction
            _, pred1 = torch.max(prop1.data, 1)
            val_preds1.append(pred1)
            # gender prediction
            pred2 = (prop2 > 0.5).item()
            val_preds2.append(pred2)

        val_acc1 = accuracy(val_preds1, [ages[i] for i in val_indices])
        val_acc2 = accuracy(val_preds2, [gender.tolist()[i] for i in val_indices])

    # print(str(epoch)+' loss:'+str(total_loss/len(ids))+' disentangle kl:'+str(total_dis_kl/len(ids))+' kl:'+str(total_kl/len(ids))+' rec:'+str(total_rec/len(ids))+' property rec:'+str(total_prop_rec/len(ids))+' l1 norm:'+str(total_l1/len(ids)))
    print(str(epoch) + ' train_acc1:' + str(train_acc1) + ' val_acc1:' + str(val_acc1) + ' train_acc2:' + str(
        train_acc2) + ' val_acc2:' + str(val_acc2) + ' loss:' + str(total_loss / len(ids)) + ' disentangle kl:' + str(
        total_dis_kl / len(ids)) + ' kl:' + str(total_kl / len(ids)) + ' rec:' + str(
        total_rec / len(ids)) + ' property rec:' + str(total_prop_rec / len(ids)))

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, "model.pt")

plt.plot(prop_rec)
plt.show()
