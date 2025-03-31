
import torch
import sys
sys.path.insert(0, '/csc/epitkane/projects/multimodal/src/')
from dataloader import PCAWG_DNABERT_Dataset
from model import MuAtMotifPosition, ModelConfig
from torch.utils.data import DataLoader

def main(complete_save_dir, test_dataset, model, valloader, device):
    test_loss = 0
    correct = 0

    logit_filename = 'val_logits.tsv'
    f = open(complete_save_dir + logit_filename, 'w+')  # open file in write mode
    header_class = test_dataset.pd_class_info['class_name'].tolist()
    header_class.append('target')
    header_class.append('target_name')
    header_class.append('sample')
    write_header = "\t".join(header_class)
    f.write(write_header)
    f.close()

    model.train(False)
    for batch in valloader:
        #string_data = data[0]
        numeric_data = batch['data']
        target = batch['targets']
        for i in range(len(numeric_data)):
            numeric_data[i] = numeric_data[i].to(device)
        target = target.to(device)
        # forward the model
        with torch.set_grad_enabled(False):
            logits, loss = model(numeric_data, target)
            _, predicted = torch.max(logits.data, 1)

            predicted = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += predicted.eq(target.view_as(predicted)).sum().item()

            #write logits
            logits_cpu =logits.detach().cpu().numpy()
            f = open(complete_save_dir + logit_filename, 'a+')
            for i in range(numeric_data[0].shape[0]):
                f.write('\n')
                logits_cpu_flat = logits_cpu[i].flatten()
                logits_cpu_list = logits_cpu_flat.tolist()    
                write_logits = ["%.8f" % i for i in logits_cpu_list]
                write_logits.append(str(target.detach().cpu().numpy().tolist()[0]))
                #write_logits.append(string_data[1][i])
                #write_logits.append(string_data[0][i])
                write_header = "\t".join(write_logits)
                f.write(write_header)
            f.close()

    test_loss /= len(test_dataset)


    local_acc = correct / len(test_dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataset), 100. * local_acc))
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    complete_save_dir = "/csc/epitkane/projects/multimodal/test_model/orig_implementation/"
    test_dataset = PCAWG_DNABERT_Dataset(data_dir="/csc/epitkane/projects/multimodal/data/temp/muat_orig_all_fixed",
                                         split='val',
                                         split_file_names=(0, "pcawg_val_test.csv"),
                                         fold=10,
                                         pos=True,
                                         muat_orig=True)
    conf = ModelConfig(vocab_size=3427,
                       block_size=5000,
                       num_class=24,
                       position_size=2916,
                       ges_size=16,
                       n_embd= 128,
                       n_layer=1,
                       n_head=1)
    model = MuAtMotifPosition(conf)
    state_dict = torch.load("/csc/epitkane/projects/litegpt/ckptpuhti/fullpcawgfold10_11100_wpos_TripletPosition_bs5000_nl1_nh1_ne128_cl3/model.pthx", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    valloader =  DataLoader(test_dataset,
                            shuffle=False,
                            batch_size=1,
                            pin_memory=True if device == torch.device('cuda') else False)



    main(complete_save_dir,
         test_dataset,
         model,
         valloader,
         device)
