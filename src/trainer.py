import os
import datetime
import shutil
import gzip
import pandas as pd
from tqdm import tqdm
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim

from utils import check_gradients, status
#from datetime import datetime

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 1
    learning_rate = 6e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    momentum = 0.9
    weight_decay = 0.001 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    TensorBoard = False
    Use_tqdm = False

    # checkpoint settings
    ckpt_path = None
    string_logs = None
    num_workers = 2 # for DataLoader
    ckpt_name = 'model'
    args = None

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

        if self.ckpt_path is not None:
            os.makedirs(self.ckpt_path, exist_ok=True) 

class Trainer:

    def __init__(self, model, config, train_dataset, validation_dataset, fold, optimizer, model_config, start_epoch=0):

        self.model = model
        self.optimizer = optimizer
        self.start_epoch = start_epoch
        self.config = config
        self.trainset = train_dataset
        self.valset = validation_dataset
        self.fold = fold
        self.model_config = model_config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ckpt_name = self.config.ckpt_name + f'_fold{self.fold}' if self.fold is not None else self.config.ckpt_name + '.pth'
        if config.TensorBoard:
            from torch.utils.tensorboard import SummaryWriter

            log_name = self.config.ckpt_name + f'_fold{self.fold}' if fold is not None else self.config.ckpt_name
            log_dir = os.path.join(os.getcwd(), os.pardir, 'runs', log_name)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            self.Writer = SummaryWriter(log_dir=log_dir)
    
    def save_ckpt(self, model, epoch, optimizer):
        ckpt_file_name = self.config.ckpt_name + f'_fold{self.fold}' if self.fold is not None else self.config.ckpt_name 
        if self.config.ckpt_path is None:
            path = os.path.join(os.getcwd, ckpt_file_name + '.pth')
        else:
            path = os.path.join(self.config.ckpt_path,f'fold_{self.fold}', 'BEST_' + ckpt_file_name + '.pth')
        
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'config': self.config.__dict__,
                    'model_config': self.model_config.__dict__ }, path)
        

    def backup(self, model, epoch, optimizer):
        ckpt_file_name = 'BACKUP_' + self.config.ckpt_name + f'_fold{self.fold}.pth' if self.fold is not None else self.config.ckpt_name 
        if self.config.ckpt_path is None:
            path = os.path.join(os.getcwd, ckpt_file_name)
        else:
            path = os.path.join(self.config.ckpt_path,f'fold_{self.fold}', ckpt_file_name)
        torch.save({'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'config': self.config.__dict__,
                    'model_config': self.model_config.__dict__ }, path)

        
    def load_ckpt(self, model, optimizer):
        ckpt = torch.load_state_dict(os.path.join(self.config.ckpt_path, self.config.ckpt_name + '.pth'), 
                                    map_location=self.device) 
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        config = TrainerConfig(ckpt['config'])
        epoch = ckpt['epoch']

        return model, config, epoch, optimizer

    def validation(self, validloader, model):

        self.logit_filename = 'val_logits'
        if not os.path.exists(os.path.join(self.config.ckpt_path, f'fold_{self.fold}')):
            os.makedirs(os.path.join(self.config.ckpt_path, f'fold_{self.fold}'), exist_ok=True)

        with gzip.open(os.path.join(self.config.ckpt_path, f'fold_{self.fold}',  self.logit_filename + f'_fold{self.fold}' + '.tsv.gz'), 'wt') as f:
            header_class = self.valset.pd_class_info['class_name'].tolist()
            header_class.append('target')
            header_class.append('target_name')
            header_class.append('sample')
            write_header = "\t".join(header_class)
            f.write(write_header)

        with torch.no_grad():
            model.eval()
            pred_correct = 0 
            losses = []
            for batch in tqdm(validloader, desc="Validation", disable = not self.config.Use_tqdm, leave=False):
                
                #  What happens in the case whn there is no mutations defined in mutration
                #  e.g., if there is only SVs and MEIs which are nt handled?
                #  Not permanent solution
                # Following if statement is utilised to make training continue in the case where the 
                # current batch does not have correct mutation types (SNV, MNV, and indel)
                #if batch['data'][0].numel() == 0:
                #    continue 
                data, target = batch['data'], batch['targets'].to(self.device)

                data = [i.to(self.device) for i in data]          
                logits, loss = model(data, target)
                pred = logits.argmax(dim=1, keepdim=True)
                pred_correct += pred.eq(target.view_as(pred)).sum().item()
                losses.append(loss.item())

                logits_cpu =logits.detach().cpu().numpy()
                with gzip.open(os.path.join(self.config.ckpt_path, f'fold_{self.fold}',  self.logit_filename + f'_fold{self.fold}' + '.tsv.gz'), 'at') as f:
                    for i in range(data[0].shape[0]):
                        f.write('\n')
                        logits_cpu_flat = logits_cpu[i].flatten()
                        logits_cpu_list = logits_cpu_flat.tolist()    
                        write_logits = ["%.8f" % i for i in logits_cpu_list]
                        write_logits.append(str(target.detach().cpu().numpy().tolist()[0][0]))
                        write_logits.append(self.valset.tumour_dict_reverse[target.detach().cpu().item()])
                        write_logits.append(batch['sample'][0]) 
                        write_header = "\t".join(map(str, write_logits))
                        f.write(write_header)

            validation_accuracy = pred_correct/len(validloader)
            validation_loss = sum(losses)/len(losses)
            if not self.config.Use_tqdm:
                status(f'Validation: loss {validation_loss} - accuracy {validation_accuracy}', True)
            if self.config.TensorBoard:
                self.Writer.add_scalar('validation loss', validation_loss, 0)
                self.Writer.add_scalar('validation accuracy', validation_accuracy, 0)
                self.Writer.flush()
            
            return validation_accuracy, validation_loss

    def train(self):

        model = self.model.to(torch.float32).to(self.device)
        config = self.config
   
        trainloader = DataLoader(self.trainset,
                                 shuffle= True,
                                 batch_size= config.batch_size,
                                 num_workers= config.num_workers,
                                 pin_memory=True if self.device == torch.device('cuda') else False)
        valloader = DataLoader(self.valset,
                               shuffle=False,
                               batch_size=config.batch_size,
                               num_workers=config.num_workers,
                               pin_memory=True if self.device == torch.device('cuda') else False)

        optimizer = self.optimizer

        if self.start_epoch == 0:
            train_df = pd.DataFrame({'epoch': [], 'loss': [], 'acc': [], 'vall_acc': []})
            best_accuracy = 0
        else:
            train_df = pd.read_csv(os.path.join(self.config.ckpt_path, f'fold_{self.fold}', self.ckpt_name + '.csv.gz'), compression='gzip', index_col=0)
            best_accuracy = max(train_df.loc[:, 'vall_acc'].values)

        for epoch in tqdm(range(self.start_epoch, self.config.max_epochs), disable = not self.config.Use_tqdm, desc= f'Fold {self.fold}, EPOCH', leave=True):
            running_loss = 0.0
            pred_correct = 0
            loss = 0 
            model.train(True)
            with tqdm(trainloader, desc="Training", disable = not self.config.Use_tqdm, leave= False) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    try:
                        #  What happens in the case whn there is no mutations defined in mutration
                        #  e.g., if there is only SVs and MEIs which are nt handled?
                        #  Not permanent solution
                        # Following if statement is utilised to make training continue in the case where the 
                        # current batch does not have correct mutation types (SNV, MNV, and indel)
                        if batch['data'][0].numel() == 0:
                            continue 
                        
                        data, target = batch['data'], batch['targets'].to(self.device)
                        data = [i.to(self.device) for i in data]   
                        with torch.set_grad_enabled(True):
                            
                            optimizer.zero_grad()
                            logits, loss = model(data, target)
                            #assert not torch.any(torch.isnan(loss))
                            #assert not torch.any(torch.isinf(loss))
                            #assert not torch.any(torch.isnan(logits))
                            #assert not torch.any(torch.isinf(logits))

                            pred = logits.argmax(dim=1, keepdim=True)
                            pred_correct += pred.eq(target.view_as(pred)).sum().item()

                            loss.backward()
                            #assert not torch.any(torch.isnan(loss))
                            #assert not torch.any(torch.isinf(loss))

                            #Gradient clipping
                            #torch.nn.utils.clip_grad_norm(model.parameters(), 1)
                            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            
                            pbar.set_postfix(loss=loss.item())
                            
                            running_loss += loss.item()
                            if batch_idx % 100 == 0 and not self.config.Use_tqdm:
                                #run_time = start_time-datetime.now()
                                status(f"Fold-{self.fold} Epoch {epoch+1}/{config.max_epochs} - sample {batch_idx*config.batch_size+1}/{len(trainloader)}\t train loss {running_loss/((batch_idx+1)* self.config.batch_size)} - train accuracy {pred_correct/((batch_idx+1)* self.config.batch_size)}", True)

                    except Exception as e:
                        print('Something happened on file {}/{}\n'.format(self.valset.tumour_dict_reverse[target.detach().cpu().item()], batch['sample'][0]))
                        print('Checking gradients...')
                        check_gradients(model)
                        print('\n#######################')
                        print('ERROR:')
                        raise e
                  
            train_loss = running_loss/(len(trainloader))
            train_accuracy = pred_correct/((batch_idx+1)* config.batch_size)

            if not os.path.exists(os.path.join(self.config.ckpt_path, f'fold_{self.fold}')):
                os.makedirs(os.path.join(self.config.ckpt_path, f'fold_{self.fold}'), exist_ok=True)    
                            
            if self.config.TensorBoard:
                self.Writer.add_scalar('train loss', running_loss/((batch_idx+1)* self.config.batch_size) , epoch)
                self.Writer.add_scalar('train accuracy', pred_correct/((batch_idx+1)* self.config.batch_size) , epoch)
                self.Writer.flush()
            
            #backup after every epoch
            self.backup(model, epoch, optimizer)
            
            #validation
            validation_accuracy, validation_loss = self.validation(valloader, model)

            train_df = pd.concat((train_df, pd.DataFrame({'epoch': [epoch], 'loss': [train_loss], 'acc': [train_accuracy], 'vall_acc': [validation_accuracy], 'vall_loss': [validation_loss]})), ignore_index=True)
            train_df.to_csv(os.path.join(self.config.ckpt_path, f'fold_{self.fold}', self.ckpt_name + '.csv.gz'), compression='gzip')
            # if validation accuracy is better than best accuracy 
            # save checkpoint
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                self.save_ckpt(model, epoch, optimizer)
                shutil.copyfile(os.path.join(self.config.ckpt_path, f'fold_{self.fold}',  self.logit_filename + f'_fold{self.fold}' + '.tsv.gz'),
                                os.path.join(self.config.ckpt_path, f'fold_{self.fold}',  self.logit_filename + f'_fold{self.fold}' + '_best_vallogits.tsv.gz'))
                os.remove(os.path.join(self.config.ckpt_path, f'fold_{self.fold}',  self.logit_filename + f'_fold{self.fold}' + '.tsv.gz'))
        
        if self.config.TensorBoard:
            self.Writer.close()

            



        