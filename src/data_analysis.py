import sys
import os
import pandas as pd 
import numpy as np

sys.path.insert(0, '/csc/epitkane/projects/multimodal/src/')
from utils import list_files_in_dir, status

def count_mutations(data_dir):
    paths = []
    paths = list_files_in_dir(data_dir, paths)
    paths.sort() 
    counts = [i for i in paths if i.split('/')[-1][:5] == 'count' ]

    SNV = 0
    indel = 0 
    MNV = 0
    for path in counts:
        data = pd.read_csv(path, compression='gzip', sep = '\t', index_col=0)
        SNV += data.iloc[0,0]
        indel += data.iloc[0,2]
        MNV += data.iloc[0,1]
    print(f'SNV:\t{SNV}\nMNV:\t{MNV}\nindel:\t{indel}')

def check_tsv_and_numpy_file_compability(data_dir):
    paths = []
    paths = list_files_in_dir(data_dir, paths)
    numpy_files = []
    tsv_files = []
    for i in paths:
        if i.split('/')[-1].split('.')[-1] == 'npz':
            numpy_files.append(i)
        if i.split('/')[-1].split('.')[-1] == 'gz' and i.split('/')[-1][:5] != 'count':
            tsv_files.append(i)

    faulty_array_files = []
    data_files = []
    for idx, (np_path, counts) in enumerate(zip(numpy_files, tsv_files)):
        array = np.load(np_path)['motif']
        count = pd.read_csv(counts, compression='gzip', sep='\t', index_col=0, header=0, low_memory=False)
        #mut_type = np_path.split('/')[-1].split('_')[0]
        if len(count) != len(array):
            faulty_array_files.append(np_path)
            data_files.append(counts)
        status(f'{idx +1}/{len(numpy_files)} scanned', True)
    return faulty_array_files, data_files 

            #print(f'Numpy: {len(array)}, Counts: {len(count)}')

def compare_npz_tsv(data_sir):
    status("Start scanning the files...", True)
    A , B = check_tsv_and_numpy_file_compability(data_dir)
    status(f'{len(A)} faulty file(s) found', True)
    last = len(A) -1

    if len(A) > 0:
        status(f'Writing the results...', True)
        with open('numpy_files_faulty_motif101.txt', 'wt') as f1, open('tsv_files_faulty_motif101.txt', 'wt') as f2:
            for idx, (i, j) in enumerate(zip(A,B)):
                if idx < last:
                    f1.write(i + ' ')
                    f2.write(j + '\n')
                else:
                    f1.write(i)
                    f2.write(j)
    else:
        status('No faulty files!', True)

    status(f'Finished!', True)

if __name__ == '__main__':
    count_mutations("/csc/epitkane/projects/multimodal/data/temp/DNABERT_motif1001")