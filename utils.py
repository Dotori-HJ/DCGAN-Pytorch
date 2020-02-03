import os
import torch
import torch.nn as nn

from config import get_config

def save_state(G, D, G_optim, D_optim, iter, G_losses, D_losses):
    cfg = get_config()

    if isinstance(G, nn.DataParallel):
        net = None
        for module in G.children():
            net = module
        G = net
    if isinstance(D, nn.DataParallel):
        net = None
        for module in D.children():
            net = module
        D = net

    state = {
        'G':G.state_dict(),
        'D':D.state_dict(),
        'G_optim': G_optim.state_dict(),
        'D_optim': D_optim.state_dict(),
        'iter': iter,
        'G_losses': G_losses,
        'D_losses': D_losses
    }
    torch.save(state, os.path.join('save', cfg['NAME'], 'state.pth'))

def load_state(G, D, G_optim, D_optim):
    cfg = get_config()

    state = torch.load(os.path.join('save', cfg['NAME'], 'state.pth'))

    G.load_state_dict(state['G'])
    D.load_state_dict(state['D'])
    G_optim.load_state_dict(state['G_optim'])
    D_optim.load_state_dict(state['D_optim'])

    return state['iter'], state['G_losses'], state['D_losses']

