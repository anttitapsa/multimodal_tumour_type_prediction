import configparser
import argparse
import os
import sys
import torch
from torch import optim

from utils import status, load_ckpt
from muat_models import *
from trainer import Trainer, TrainerConfig
from dataloader import PCAWG_DNABERT_Dataset
from torch.utils.data import DataLoader


def main():

    # Read the arguments 
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--config_file', metavar='cf', type=str, default='', help= 'path to configuration file utilised for the training')
    parser.add_argument('--input', metavar='i', type=str, default='', help= 'path to data directory, created for using in csc Mahti')
    parser.add_argument('--fold',metavar='f', type=int, default=-1, help= 'fold for cross validation')
    parser.add_argument('--load', metavar='co', type=str, default=None, help= 'path to checkpoint where to continue from')
    parser.add_argument('--valid', action="store_true")
    parser.add_argument('--full_data', action="store_true")

    args = parser.parse_args()
    config_file = args.config_file
    input_dir = args.input
    fold_ = args.fold
    backup_path = args.load
    Validation = args.valid
    full_data = args.full_data

    if not config_file:
        sys.stderr.write("Please, give config path to config file as a command line argumen")
        sys.exit(1)

    elif not os.path.exists(config_file):
        sys.stderr.write(f'The path to config file {config_file} does not exist')
        sys.exit(1)

    else:
        configParser = configparser.ConfigParser()
        configParser.read(config_file)

        try:
            status(f'Reading the config file', verbose= True)

            block_size = configParser['MODEL'].getint('block_size')
            num_class = configParser['MODEL'].getint('num_class')
            architecture = configParser['MODEL']['architecture']
            vocab_size = configParser['MODEL'].getint('vocab_size')
            position_size = configParser['MODEL'].getint('position_size')
            ges_size = configParser['MODEL'].getint('ges_size')
            embed_dim = configParser['MODEL'].getint('embed_dim')
            motif_len = configParser['MODEL'].getint('motif_len')
            context = configParser['MODEL'].getboolean('context')
            context_length = configParser['MODEL'].getint('context_length')

            max_epochs = configParser['TRAINER'].getint('max_epochs')
            batch_size = configParser['TRAINER'].getint('batch_size')
            learning_rate = configParser['TRAINER'].getfloat('learning_rate')
            betas = configParser['TRAINER']['betas']
            momentum = configParser['TRAINER'].getfloat('momentum')
            weight_decay = configParser['TRAINER'].getfloat('weight_decay')
            lr_decay = configParser['TRAINER'].getboolean('lr_decay')

            ckpt_path = configParser['TRAINER']['ckpt_path']
            num_workers = configParser['TRAINER'].getint('num_workers')
            ckpt_name = configParser['TRAINER']['ckpt_name']
            muat_orig = configParser['TRAINER'].getboolean('muat_orig')

            if fold_ != -1:
                fold = fold_
            else:
                fold = configParser['TRAINER'].getint('fold')

            if input_dir != "":
                data_dir = input_dir
            else:    
                data_dir = configParser['DATALOADER']['data_dir']
            
            mutation_ratio = configParser['DATALOADER']['mutation_ratio']
            tumour_info = configParser['DATALOADER']['tumour_info_file_name']
            train_split_file = configParser['DATALOADER']['train_split']
            val_split_file = configParser['DATALOADER']['val_split']
            epipos = configParser['DATALOADER'].getboolean('epipos')

        except Exception as e:
            sys.stderr.write('Config argument(s) missing, fix the config file')
            sys.stderr.write(repr(e))
            sys.exit(1)

            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        '''
        if architecture == "MuAtMotifPosition" or architecture == "MuAtMotifPositionGES":
            pos = True
            if architecture == "MuAtMotifPositionGES":
                ges = True
            else:
                ges = False
        else:
            pos = False
            ges = False
        '''
        if not Validation:

             # 10 -fold crossvalidation
            if backup_path != None:
                status(f'Loading the backup...', True)
            else:
                status(f'Starting cross-validation fold {fold}...', True)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if backup_path != None:
                status(f'Loadinng {backup_path}', True)
                model_config, state_dict, config, start_epoch, optimizer_state_dict = load_ckpt(backup_path)
                
                modelConfig = ModelConfig(**model_config)
                model, pos, ges, one_hot = get_model(architecture, modelConfig)
                model.load_state_dict(state_dict)
                model.to(device)
                
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
                optimizer.load_state_dict(optimizer_state_dict)
                config['max_epochs'] =max_epochs
                tconfig = TrainerConfig(**config)
                status(f'Backup loaded!', True)
            else:
                modelConfig = ModelConfig(muat_orig=muat_orig,
                                  vocab_size=vocab_size,
                                  block_size=block_size,
                                  num_class=num_class,
                                  position_size=position_size,
                                  ges_size=ges_size,
                                  embed_dim=embed_dim,
                                  motif_len=motif_len,
                                  context = context)
                model, pos, ges, one_hot = get_model(architecture, modelConfig)
                model.to(device)

                status(f'Model prepared!', True)

                start_epoch = 0
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
                tconfig = TrainerConfig(max_epochs=max_epochs,
                                        batch_size= batch_size,
                                        learning_rate=learning_rate,
                                        betas=betas,
                                        momentum=momentum,
                                        weight_decay=weight_decay,
                                        lr_decay=lr_decay,
                                        ckpt_path=str(ckpt_path),
                                        num_workers=num_workers,
                                        ckpt_name=str(ckpt_name))

            train_dataset = PCAWG_DNABERT_Dataset(data_dir,
                                                    block_size,
                                                    mutation_ratio,
                                                    tumour_info,
                                                    split = 'train',
                                                    fold= fold,
                                                    split_file_names =(train_split_file, val_split_file),
                                                    pos=pos,
                                                    ges=ges,
                                                    muat_orig=muat_orig,
                                                    one_hot=one_hot,
                                                    one_hot_length=motif_len,
                                                    epipos= epipos,
                                                    context = context,
                                                    context_length= context_length,
                                                    nembed= embed_dim)
            val_dataset = PCAWG_DNABERT_Dataset(data_dir,
                                                block_size,
                                                mutation_ratio,
                                                tumour_info,
                                                split = 'val',
                                                fold= fold,
                                                split_file_names =(train_split_file, val_split_file),
                                                pos=pos,
                                                ges=ges,
                                                muat_orig=muat_orig,
                                                one_hot=one_hot,
                                                one_hot_length=motif_len,
                                                epipos=epipos,
                                                context = context,
                                                context_length= context_length,
                                                nembed= embed_dim)
            
            status(f'Datasets ready!', True)

            trainer = Trainer(model,tconfig, train_dataset, val_dataset, fold, optimizer, modelConfig, start_epoch)

            status(f'Start the training...', True)
            trainer.train()

            status(f'Training finished!', True)
        
        if Validation:
             
            status(f'Loadinng {backup_path}', True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_config, model_state_dict, config, start_epoch, optimizer_satate_dict = load_ckpt(backup_path)
            modelConfig = ModelConfig(**model_config)
            model, pos, ges, one_hot = get_model(architecture, modelConfig)
            model.load_state_dict(model_state_dict)
            model.to(device)

            config['ckpt_path'] = ckpt_path
            tconfig = TrainerConfig(**config)
            status(f'Model loaded!', True)
            if full_data:

                val_dataset = PCAWG_DNABERT_Dataset(data_dir,
                                                    block_size,
                                                    mutation_ratio,
                                                    tumour_info,
                                                    split = 'val',
                                                    fold= None,
                                                    split_file_names =(train_split_file, val_split_file),
                                                    pos=pos,
                                                    ges=ges,
                                                    muat_orig=muat_orig,
                                                    one_hot=one_hot,
                                                    one_hot_length=motif_len,
                                                    epipos=epipos,
                                                    context = context,
                                                    context_length= context_length,
                                                    nembed= embed_dim)
            else:

                val_dataset = PCAWG_DNABERT_Dataset(data_dir,
                                                    block_size,
                                                    mutation_ratio,
                                                    tumour_info,
                                                    split = 'val',
                                                    fold= fold,
                                                    split_file_names =(train_split_file, val_split_file),
                                                    pos=pos,
                                                    ges=ges,
                                                    muat_orig=muat_orig,
                                                    one_hot=one_hot,
                                                    one_hot_length=motif_len,
                                                    epipos=epipos,
                                                    context = context,
                                                    context_length= context_length,
                                                    nembed= embed_dim)
            trainer = Trainer(model,tconfig, None, val_dataset, fold, None, modelConfig, start_epoch)

            valloader = DataLoader(val_dataset,
                                    shuffle=False,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    pin_memory=True if device == torch.device('cuda') else False)

            trainer.validation(valloader, model)
            
if __name__ == '__main__':
    main()

        