from .model import EncoderProcesserDecoder
import torch.nn as nn
import torch
from torch_geometric.data import Data
import os

class Simulator(nn.Module):

    def __init__(self, message_passing_num, edge_input_size, device, model_dir='checkpoint/simulator.pth') -> None:
        super(Simulator, self).__init__()

        self.edge_input_size = edge_input_size
        self.model_dir = model_dir
        self.model = EncoderProcesserDecoder(
            message_passing_num=message_passing_num,
            edge_input_size=edge_input_size
        ).to(device)

        print('Simulator model initialized')


    def forward(self, graph:Data):
        predicted = self.model(graph)
        return predicted


    def load_checkpoint(self, ckpdir=None):
        
        if ckpdir is None:
            ckpdir = self.model_dir
        dicts = torch.load(ckpdir)
        print(dicts)
        self.load_state_dict(dicts['model'])

        keys = list(dicts.keys())
        keys.remove('model')

        for k in keys:
            v = dicts[k]
            for para, value in v.items():
                object = eval('self.'+k)
                setattr(object, para, value)

        print("Simulator model loaded checkpoint %s"%ckpdir)

    def save_checkpoint(self, savedir=None):
        if savedir is None:
            savedir=self.model_dir

        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)
        
        model = self.state_dict()

        to_save = {'model':model}

        torch.save(to_save, savedir)
        print('Simulator model saved at %s'%savedir)
