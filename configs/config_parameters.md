- block_size - number of mutations utilised in the training for each individual sample 
- num_class - number of labels (number of different Tumour types)
- architecture - Architecture of the model, possible choises are MuAtMotif, MuAtMotifPosition, MuAtMotifPositionGES, MuAtOneHotMotif, MuAtOneHotMotifPosition, MuAtOneHotMotifPositionGES, MuAtOneHotMotifWithoutReLU, MuAtOneHotMotifPositionWithoutReLU, MuAtOneHotMotifPositionGESWithoutReLU, MuAtMotifEpiPos, MuAtMotifEpiPosGES, MuAtMotifPositionEpiPos, MuAtMotifPositionGESEpiPos, MuAtMotifContext
- vocab_size - size of the vocabulary of the model
- position_size - number of different positions
- ges_size - number of different GES annotations
- embed_dim - the dimension of the embedding 
- motif_len - the length of the motif sequence
- context - True or False, is the context information provided (not utilised yet)
- context_length - length of the context vector (not utilised yet)
- max_epochs - number of training epochs
- batch_size - size of the batch(should be 1)
- learning_rate - learning rate of the optimizer
- betas - betas utilised in the optimizer
- momentum - momentum of the optimizer
- weight_decay - True or False if the weight decay is utilised
- lr_decay - True or False if the learning rate decay is utilised
- ckpt_path - Path where the model checkpoint is saved
- num_workers - Number of CPU cores to use
- ckpt_name - Name of the checkpoint
- fold - fold of the crossvalidation
- muat_orig - Does the model follow original MuAt model
- data_dir - path to directory containing the data 
- mutation_ratio - ratio of different mutations in formt  0.4-0.3-0.3-0-0
- tumour_info_file_name - file containing amonts of different tumours and idices of tuomours
- train_split file containing the sample names included in training split  
- val_split - file containing the sample names included in validation split
- epipos - True or False if the model utilises epigenetic data
