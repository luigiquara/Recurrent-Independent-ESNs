import argparse
from datetime import datetime
import os
import random
from tqdm import tqdm
import wandb

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from mnist import get_mnist_data
from rim_model import RIM

def set_seed(seed):
    print(f'Setting seed to {seed}')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)

def train(model, train_loader, val_loaders, loss_fn, opt, epochs, device, log, save_dir):
    ctr = 0
    epoch_loss = 0
    best_acc = 0
    val_losses = {}
    val_accs = {}

    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0.0
        iter_ctr = 0.0
        train_accuracy = 0

        print(f'Epoch {epoch+1}')

        for x, y in tqdm(train_loader):
            x = x.squeeze().to(device)
            x = x.view(x.shape[0], x.shape[1]**2, -1)
            y = y.to(device)

            # forward pass
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            # backward pass
            opt.zero_grad()
            loss.backward()
            opt.step()

            # compute training accuracy
            preds = torch.argmax(y_pred, dim=1)
            correct = preds == y.long()
            train_accuracy += correct.sum().item()

            iter_ctr += 1.0
            ctr += 1

        # evaluate on validation data
        for img_size, val_loader in val_loaders.items():
            print(f'Evaluating on {img_size}')
            loss, acc = evaluate(model, val_loader, loss_fn, device)

            val_losses['val_loss'+str(img_size)] = loss
            val_accs['val_accuracy'+str(img_size)] = acc

        # printing current info
        log_str = f'epoch loss: {epoch_loss/iter_ctr}'
        for key, value in val_accs.items(): log_str += f' {key}: {value:.2f}'
        print(log_str)

        # log to wandb
        if log:
            wandb.log({'loss': epoch_loss/iter_ctr})
            wandb.log(val_losses)
            wandb.log(val_accs)
            wandb.log({'epoch': epoch})

        # saving the best model
        if val_accs['val_accuracy14'] > best_acc:
            best_acc = val_accs['val_accuracy14']

            state = {
                'net': model.state_dict(),
                'epoch': epoch,
                'ctr': ctr,
                'best_acc': best_acc
            }

            with open(save_dir + 'best_model.pt', 'wb') as f:
                torch.save(state, f)

def evaluate(model, loader, loss_fn, device):
    total = 0
    correct = 0
    epoch_loss = 0.0
    iter_ctr = 0.0

    model.eval()
    with torch.no_grad():

        for x, y in tqdm(loader):
            x = x.to(device)
            x = x.view(x.shape[0], x.shape[1]**2, -1)
            y = y.to(device)

            # forward pass
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            # compute validation accuracy
            preds = torch.argmax(y_pred, dim=1)
            correct += (preds == y.long()).sum().item()
            total += len(preds)

            iter_ctr += 1.0

        loss = epoch_loss / iter_ctr
        accuracy = correct / total * 100.0

        return loss, accuracy


def run(args):
    set_seed(args.seed)

    # create the model
    model = RIM(
        args.input_size, args.device,
        args.num_units, args.active_units,
        args.use_input_attention, args.num_input_heads, args.key_input_size, args.query_input_size, args.value_input_size, args.input_dropout,
        args.use_comm_attention, args.num_comm_heads, args.key_comm_size, args.query_comm_size, args.value_comm_size, args.comm_dropout, args.alpha,
        args.hidden_size, args.input_scaling, args.spectral_radius, args.leaky
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {n_params}')

    # load the data
    train_loader, val_loader, _ = get_mnist_data(root=args.datapath, image_size=args.size, batch_size=args.batch_size, subset_size=args.subset_size)

    # load multiple validation data to test out-of-distribution generalization
    val_loaders = {args.size: val_loader}
    val_img_sizes = [16, 19, 24]
    for size in val_img_sizes:
        _, val_loaders[size], _ = get_mnist_data(root=args.datapath, image_size=size, batch_size=args.batch_size, subset_size=args.subset_size)

    if args.log:
        wandb.init(
            project = 'ESN_RIM',
            config = args
        )
        args.save_dir += wandb.run.name + '/'
    else: args.save_dir += datetime.now().strftime('%d%m%Y-%H%M%S')
    if not os.path.exists(args.save_dir): os.mkdir(args.save_dir)

    loss_fn = nn.CrossEntropyLoss()
    opt = Adam(model.parameters(), lr=args.lr)
    train(model, train_loader, val_loaders, loss_fn, opt, args.epochs, args.device, args.log, args.save_dir)

    if args.log: wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    rim_args = parser.add_argument_group('RIM arguments')
    rim_args.add_argument('--num_units', type=int)
    rim_args.add_argument('--active_units', type=int)

    # input attention
    rim_args.add_argument('--no_input_attention', action='store_false', dest='use_input_attention')
    rim_args.add_argument('--input_attention', action='store_true', dest='use_input_attention')
    rim_args.add_argument('--num_input_heads', type=int, default=1)
    rim_args.add_argument('--key_input_size', type=int, default=64)
    rim_args.add_argument('--query_input_size', type=int, default=64)
    rim_args.add_argument('--value_input_size', type=int, default=400)
    rim_args.add_argument('--input_dropout', type=float, default=0.1)

    # communication attention
    rim_args.add_argument('--no_comm_attention', action='store_false', dest='use_comm_attention')
    rim_args.add_argument('--comm_attention', action='store_true', dest='use_comm_attention')
    rim_args.add_argument('--num_comm_heads', type=int, default=4)
    rim_args.add_argument('--key_comm_size', type=int, default=32)
    rim_args.add_argument('--query_comm_size', type=int, default=32)
    rim_args.add_argument('--value_comm_size', type=int, default=100)
    rim_args.add_argument('--comm_dropout', type=float, default=0.1)
    rim_args.add_argument('--alpha', type=float, default=None)

    # ESN arguments
    esn_args = parser.add_argument_group('ESN arguments')
    esn_args.add_argument('--hidden_size', type=int, default=1000)
    esn_args.add_argument('--input_scaling', type=float, default=1.0)
    esn_args.add_argument('--spectral_radius', type=float, default=0.99)
    esn_args.add_argument('--leaky', type=float, default=0.001)

    # dataset arguments
    data_args = parser.add_argument_group('Dataset arguments')
    data_args.add_argument('--datapath', type=str, default='/raid/l.quarantiello/datasets/')
    data_args.add_argument('--size', type=int, default=14)
    data_args.add_argument('--subset_size', type=int, default=-1)
    data_args.add_argument('--input_size', type=int, default=1)
    data_args.add_argument('--batch_size', type=int, default=64)

    # training arguments
    train_args = parser.add_argument_group('Training arguments')
    train_args.add_argument('--epochs', type=int, default=50)
    train_args.add_argument('--lr', type=float, default=1e-3)

    # generic arguments
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--no_log', action='store_false', dest='log')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./saved_models/')

    args = parser.parse_args()
    run(args)