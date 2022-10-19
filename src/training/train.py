
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import os
import time
import paths
import config
import logging

from datetime import datetime
from BlendshapeDataset import BlendshapeDataset
from ..models import LSTMNvidiaNet

__all__ = ['train']

def _save_checkpoint(epoch: int, model: nn.Module, eval_loss: float, is_best: bool = False):
    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'eval_loss': eval_loss,
        },
        paths.checkpoint_file(epoch=epoch, is_best=is_best)
    )

def _get_loaders() -> tuple[BlendshapeDataset, BlendshapeDataset]:
    
    train_ds = BlendshapeDataset(feature_file=paths.combined_features('train'), target_file=paths.combined_blendshapes('train'))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=2, shuffle=True)

    val_ds = BlendshapeDataset(feature_file=paths.combined_features('val'), target_file=paths.combined_blendshapes('val'))
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=2, shuffle=False)

    return train_loader, val_loader


def train():

    # TODO: 2022-10-19 - Setup logger properly (seek for the right way & the right place)
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

    # Setup cuda
    device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        logging.warning('CUDA device not found, migrating to CPU')

    logging.info('Device name: %s' % device)

    # Create model
    model = LSTMNvidiaNet(num_blendshapes=config.n_blendshapes)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    if torch.cuda.is_available():
        model = model.cuda()
    
    # Let the train begin...
    train_loader, val_loader = _get_loaders()
    n_train = len(train_loader)
    n_val = len(val_loader)

    logging.info('Training begin at %s' % datetime.now())

    # TODO: 2022-10-19 - Add tqdm
    for epoch in range(epochs):
        start_time = time.time()

        # Train
        model.train()
        train_loss = 0.
        for i, (input, target) in enumerate(train_loader):
            
            # Prepare (TODO: WTF !?)
            input_var = autograd.Variable(input.float())
            if torch.cuda.is_available():
                target = target.cuda(non_blocking=True)
                input_var = input_var.cuda()
            
            target_var = autograd.Variable(target.float())

            # Compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            train_loss += loss * 100

            # Back propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log progress
            should_log = print_freq != -1 and i % print_freq == 0
            if should_log:
                logging.info('Training -- epoch: {:03} | iteration: {}/{} | loss: {:.6f} \r'
                    .format(epoch + 1, i, n_train, loss.data[0]))

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        eval_loss = 0.
        with torch.no_grad():
            for input, target in val_loader:
                
                # Prepare (TODO: WTF !?)
                input_var = autograd.Variable(input.float())
                if torch.cuda.is_available():
                    target = target.cuda(non_blocking=True)
                    input_var = input_var.cuda()
                
                target_var = autograd.Variable(target.float())

                # Compute output
                output = model(input_var)
                loss = criterion(output, target_var)
                eval_loss += loss * 100

        eval_loss /= len(val_loader)

        # Log epoch progress
        past_time = time.time() - start_time
        logging.info('epoch: {:03} | train_loss: {:.6f} | eval_loss: {:.6f} | {:.4f} sec/epoch \r'
            .format(epoch+1, train_loss, eval_loss, past_time))

        # Save checkpoint models
        is_best = eval_loss < best_loss
        if is_best:
            best_loss = min(eval_loss, best_loss)
            _save_checkpoint(epoch=epoch, model=model, eval_loss=best_loss, is_best=True)

        if (epoch + 1) % checkpoint_freq == 0:
            _save_checkpoint(epoch=epoch, model=model, eval_loss=eval_loss)

    log.info('Training finished at %s' % datetime.now())
