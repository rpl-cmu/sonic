import os
import sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from random import shuffle
import torch
import os
import config
# from tensorboardX import SummaryWriter
from SONIC.sonic_model import SONICModel
from dataloader.sonardata import SonarData, SonarDataLoader
from utils import cycle
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

def train_sonardata(args):
    out_folder = os.path.join(args.outdir, args.exp_name)
    os.makedirs(out_folder, exist_ok=True)
    f = os.path.join(out_folder, 'args.txt')
    with open(f, 'w') as file:
        for arg in vars(args):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    

    # sonardata data loader
    train_loader = SonarDataLoader(args).load_data()
    train_loader_iterator = iter(cycle(train_loader))

    # val data loader
    args.phase = "val"
    val_loader = SonarDataLoader(args).load_data()
    val_loader_iterator = iter(cycle(val_loader))
    args.phase= "train"

    # define model
    model = SONICModel(args)
    start_step = model.start_step

    # val log iteration
    val_log_i = 0

    # training loop
    pbar = tqdm(total=args.n_iters)
    for step in range(start_step + 1, start_step + args.n_iters + 1):
        if model.model.training==False:
            model.model.train()
        data = next(train_loader_iterator)
        model.set_input(data)
        model.optimize_parameters()
        model.write_summary(step,pbar)
        if step % args.save_interval == 0 and step > 0:
            model.save_model(step)
            # run val loop
            validate_model(model,args,val_log_i,device=model.model.device,val_loader_iterator=val_loader_iterator)
            val_log_i = val_log_i + args.n_val_iters
    pbar.close()
    

def validate_model(model, args, val_log_i, device, val_loader_iterator, criterion=None, decoder=None):
    model.model.eval()
    start_step = val_log_i
    for step in range(start_step+1, start_step+args.n_val_iters+1):
        data = next(val_loader_iterator)
        with torch.no_grad():
            model.set_input(data)
            model.val_forward()
            model.write_summary(step,val_set=True)

    model.model.train()

if __name__ == '__main__':
    args = config.get_args()
    wandb.login(key="your_wandb_key")
    run = wandb.init(
        name=args.exp_name,
        reinit=True,
        mode = "disabled",
        # resume="must",
        # id="wdrb1add",
        config=vars(args)
    )
    train_sonardata(args)

