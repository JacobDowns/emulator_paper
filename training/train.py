import os
import sys
sys.path.append('./')
from speceis_dg.new.simulator import Simulator
import torch
from simulation_loader import SimulatorDataset
from torch.utils.data.dataset import Subset
from velocity_loss import VelocityLoss
import numpy as np
import random
import firedrake as fd
from torch.utils.data import random_split, DataLoader
from tensorboardX import SummaryWriter


torch.manual_seed(0)

writer = SummaryWriter()
save_epoch = 5
epochs = 1001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
simulator = Simulator(message_passing_num=10, edge_input_size=13, device=device)
simulator.load_checkpoint()
optimizer = torch.optim.Adam(simulator.parameters(), lr=5e-5)
vel_loss = VelocityLoss().apply


out_mod = fd.File('training/monitor/out_mod.pvd')
out_obs = fd.File('training/monitor/out_obs.pvd')


def train(model:Simulator, train_data, val_data, optimizer):

    k = 0

    for ep in range(1, epochs):
        print('Epoch', ep)
        model.train() 
        train_error = 0.
        n = 0

        for j in range(len(train_data)):
            g, y_obs, sim_loader = train_data[j]

            Ubar_obs = y_obs.flatten()
            g = g.cuda()
            y = model(g).cpu()
            Ubar_mod = y.flatten()
            #Ubar_obs = torch.normal(Ubar_obs, std=1e-3)

            loss = vel_loss(Ubar_mod, Ubar_obs, sim_loader.loss_integral)
            train_error += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n += 1

        print('Train error: ', train_error / n)
        writer.add_scalar('train/loss', train_error / n, ep)

        if ep % save_epoch == 0:
            model.save_checkpoint()

        if ep % 15 == 0:
            model.eval()
            val_error = 0.
            n = 0
            with torch.no_grad():
                 for j in range(len(val_data)):
                    g, y_obs, sim_loader = val_data[j]

                    Ubar_obs = y_obs.flatten()
                    g = g.cuda()
                    y = model(g).cpu()
                    Ubar_mod = y.flatten()

                    loss = vel_loss(Ubar_mod, Ubar_obs, sim_loader.loss_integral)
                    
                    if j == 10:
                        out_mod.write(sim_loader.loss_integral.Ubar, idx=k)
                        out_obs.write(sim_loader.loss_integral.Ubar_obs, idx=k)
                        k += 1

                    val_error += loss.item()
                    n += 1

            print('Validation error: ', val_error / n)
            writer.add_scalar('validate/loss', val_error / n, ep)

    writer.close()

if __name__ == '__main__':


    data = SimulatorDataset()

    n = len(data)
    n_test = int(0.1*n)

    # Split the dataset
    train_dataset, val_dataset = random_split(data, [n-n_test, n_test])

    train(simulator, train_dataset, val_dataset, optimizer)