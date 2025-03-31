"""Module for prerocessing data utilised for multimodal version of MuAT.

Author: Antti Huttunen
Date: 15.4.2024

This module is inspired from the Prima Sanjaya's code in 
https://github.com/primasanjaya/muat-github/blob/1f91c38d00d2f2156df9c2cb0f4e21dba673cf85/preprocessing/dmm/annotate_mutations_all_modified.py
"""

import pandas as pd
#import numpy as np
from natsort import natsort_keygen

import argparse
import datetime
import gzip
import os
import sys
#import torch

from math import floor, ceil

sys.path.insert(0, '/csc/epitkane/projects/multimodal/src/')
from utils import list_files_in_dir
#from collections import deque
#from natsort import natsort_keygen

SNV = ['!', '@', '#', '$', '%', '^', '&', '*', '~', ':', ';', '?']
DEL = ['1', '2', '3', '4']
INS = ['5', '6', '7', '8']
SV = ['D', 'P', 'I', 'B']
MEI = ['L', 'M', 'S', 'Q']
NO_MUT = ['A', 'C', 'T', 'G', 'N']

def read_codes(fn) -> dict:
    """Creates dictionary of the mutation codes based on .tsv file.

    The information in the .tsv file should on the columns in the following order:
    Reference base (ref), altered base (alt), mutation token. For example one row 
    could be "A, C, !". That line means that the token '!' is utilised to mark mutation
    A > C.
    
    Args: 
        fn: The path to mutation codes .tsv file.

    Return:
        codes:  Dict containing the mutation tokens. 
                The form of the dict is {ref: {alt: token}} e.g., {'A':{'C': '!'}}

        rcodes: Dict containing mutation information. Reverse version from codes dict.
                The form of the dict is {token: (ref, alt)} e.g., {'!':('A', 'C')}
    """

    codes = {}
    rcodes = {}
    with open(fn) as f:
        for s in f:
            ref, alt, code = s.strip().split()
            if ref not in codes:
                codes[ref] = {}
            codes[ref][alt] = code
            rcodes[code] = (ref, alt)
    rcodes['N'] = ('N', 'N')  # ->N, N>-, A>N etc all map to N, make sure that 'N'=>'N>N'
    return codes, rcodes


def read_reference(reffn, verbose=0):
    """Reads the reference and return it as a dict

    Original function from Sanjaya's github repo.
    """
    valid_dna = ''.join([chr(x) if chr(x) in 'ACGTN' else 'N' for x in range(256)])
    R = {}
    chrom = None
    if reffn.endswith('.gz'):
        f = gzip.open(reffn)
    else:
        f = open(reffn)
    for s in f:
        if s[0] == '>':
            if chrom is not None:
                R[chrom] = ''.join(seq).translate(valid_dna)
            seq = []
            chrom = s[1:].strip().split()[0]
            if verbose:
                sys.stderr.write('{} '.format(chrom))
                sys.stderr.flush()
        else:
            seq.append(s.strip().upper())
    R[chrom] = ''.join(seq).translate(valid_dna)
    if verbose:
        sys.stderr.write(' done.\n')
    return R



def status(msg, verbose, lf=True, time=True) -> None:
    """writes status message to stream

    Args:
        msg: status as a string
        verbose: boolean value which is utilised to tell if status messages are printed 
        lf: boolean value which is utilised to tell if newline char is added to the end of message 
        time: boolean value telling if the time stamp is added to the statusmessage
    """
    if verbose:
        if time:
            tstr = '[{}] '.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        else:
            tstr = ''
        sys.stdout.write('{}{}'.format(tstr, msg))
        if lf:
            sys.stdout.write('\n')
        sys.stdout.flush()

def find_mutation_boundaries(seq, mut_indx):
    """Finds the start index and end index of the longer mutations
    
    Args:
        seq: DNA sequence as string
        mut_index: ndex of the mutation as an integer 
    """
    mutations = SNV + DEL + INS + SV + MEI
    
    start, end = mut_indx, mut_indx
    while start > 0 and seq[start-1] in mutations:
        start -= 1

    while end < len(seq) and seq[end + 1] in mutations:
        end += 1
    
    return start, end 

def read_exclude():
    # Fix indices
    """Exclude regions listed in the bed files
    
    Needed to exclude regions epicSV cannot handle because of quality 
    """
    exclude = pd.DataFrame()
    files = os.listdir(os.path.join(os.pardir, 'extfiles', 'exclude_regions'))
    for f in files:
        if f.split(os.sep)[-1] in ['consensusBlacklist_sorted.bed', 'dukeExcludeRegions_sorted.bed', 'encodeblacklist.bed']:
            file = pd.read_csv(os.path.join(os.pardir, 'extfiles', 'exclude_regions',f), sep='\t', header=None)
            exclude = pd.concat([exclude, file.iloc[:,:3]], ignore_index=True)
    exclude = exclude.drop_duplicates()
    exclude = exclude.apply(lambda row: [row[0].replace('chr', ''), int(row[1]), int(row[2])] if row[0].replace('chr', '') == 'X' or row[0].replace('chr', '') == 'Y' else [int(row[0].replace('chr', '')), int(row[1]), int(row[2])], axis=1, result_type='expand')
    exclude = exclude.set_axis(['chrom', 'chromstart', 'chromend'], axis='columns')
    exclude = exclude.sort_values(by=['chrom', 'chromstart'], ignore_index=True)

    include = pd.read_csv(os.path.join(os.pardir, 'extfiles', 'include_area', 'Yilong_PCAWG_SV_included_regions_hg19.bed'), sep='\t', header=None)
    include = include.apply(lambda row: [row[0].replace('chr', ''), int(row[1]), int(row[2])] if row[0].replace('chr', '') == 'X' or row[0].replace('chr', '') == 'Y' else [int(row[0].replace('chr', '')), int(row[1]), int(row[2])], axis=1, result_type='expand')
    include = include.set_axis(['chrom', 'chromstart', 'chromend'], axis='columns')

    return exclude, include 


def exclude_region(chrom, pos, to_exclude, area):
    chrom = chrom if chrom == 'X' or chrom == 'Y' else int(chrom)
    pos = int(pos)

    exclude_chroms = [i if i == 'X' or i == 'Y' else int(i) for i in to_exclude['chrom'].values.tolist()]
    #exclude regions
    if chrom in exclude_chroms:
        chrom_pos_exclude = to_exclude[to_exclude['chrom'] == chrom]
        chrom_pos_exclude = chrom_pos_exclude[chrom_pos_exclude['chromstart']<= int(pos)]
        chrom_pos_exclude = chrom_pos_exclude[chrom_pos_exclude['chromend']>= int(pos)]
        if len(chrom_pos_exclude) > 0:
            return True

    # Check that mutation locates in correct area
    include_chroms = [i if i == 'X' or i == 'Y' else int(i) for i in area['chrom'].values.tolist()]
    if chrom in include_chroms:
        chrom_pos_include = area[area['chrom'] == chrom]
        chrom_pos_include = chrom_pos_include[chrom_pos_include['chromstart']<= int(pos)]
        chrom_pos_include = chrom_pos_include[chrom_pos_include['chromend']>= int(pos)]
        if len(chrom_pos_include) > 0:
            return False
    return True

def get_orig_muat_seq(line, mut_idx, reverse_code, motif_length=3):
    ref = line[2] 
    alt = line[3] 
    
    mut_beginning = mut_idx - int(floor(motif_length/2))
    mut_ending = mut_idx + int(floor(motif_length/2)) +1

    #mut_seq = line[5][mut_idx-1:mut_idx+2]
    mut_seq = line[5][mut_beginning:mut_ending]
    ref_seq = []
    for i in mut_seq:

        if i not in INS + DEL + SNV +NO_MUT:
            return None, None, None, None
    
        ref_seq.append(reverse_code[i][0])
    
    return mut_seq, "".join(ref_seq), ref, alt




def get_mutation_seq(line, reverse_code, motif_length=3)-> str:
    """Creates the mutation sequence

    Sequence is length of the motif - 1 + length of the mutation i.e. mutation
    is considered as a length of one in the mutation sequence. For example if 
    the motif_length is 3 and the input sequence is A5656T, the resulting mutation sequence will
    be ACGCGT.

    Cases where the motif_length is even there is more bases after the mutation. E.g.,
    if the motif_length is 4 and the input sequence is AA5656TT, the resulting mutation sequence
    will be ACGCGTT.

    If there is not implemented mutation token, the function returns Nones. 

    Args:
        line: The row of the data file as a list 
        reverse_code: Dict containing mutation information. The form of
                      the dict is {token: (ref, alt)} e.g., {'!':('A', 'C')}
        motif_length: Length of the motif as an integer

    Return:
        Mutation sequence, where the mutation tokens are replaced with bases.
        Reference sequence i.e. full sequence before mutation.
        Reference sequence in the mutation position
        Alterated sequence in the mutation position 
    
    """

    mut_idx = int(len(line[5])/2) # mutation is in the middle of sequence found in the data files
    start, end = find_mutation_boundaries(line[5], mut_idx)
    start_pos = int(start - floor((motif_length-1)/2))
    if motif_length % 2 == 0:
        end_pos = int(end + ceil((motif_length-1)/2))
    else:
        end_pos = int(end + floor((motif_length-1)/2))

    mut_seq = line[5][start_pos:end_pos+1]
    ref_seq = line[5][start_pos:end_pos+1]
    # mut_idx -= start_pos

    mut_seq = list(mut_seq)
    ref_seq = list(ref_seq)
    ref, alt = '', ''
    for i in range(0, len(mut_seq)):

        if mut_seq[i] in NO_MUT:
            continue
        if mut_seq[i] in SNV:
            mut_seq[i] = reverse_code[mut_seq[i]][1]
            alt += mut_seq[i]
            ref_seq[i] = reverse_code[ref_seq[i]][0]
            ref += ref_seq[i]
        elif mut_seq[i] in DEL:
            mut_seq[i] = ''
            alt += '-'
            ref_seq[i] = reverse_code[ref_seq[i]][0]
            ref += ref_seq[i]
        elif mut_seq[i] in INS:
            mut_seq[i] = reverse_code[ref_seq[i]][1]
            alt += mut_seq[i]
            ref_seq[i] = ''
            ref += '-'
        else:
            return None, None, None, None
    #print(f'MUT: {mut_seq}\nREF: {ref_seq}\n')

    return ''.join(mut_seq), ''.join(ref_seq), ref, alt

def classify_mutation_type(mutdict, sequence, mut_idx):
    middle_seq = sequence[mut_idx-1:mut_idx+2]
    try:
        mut_type = mutdict[mutdict.loc[:,'triplet'] == middle_seq].iloc[0]['mut_type']
    except:
        status(f'Classification of mutation failed. The Middle sequence tried to use was {middle_seq}', True)
    
    return mut_type

def process_input(input_file_name,
                  tmp_dir,
                  reverse_code, 
                  mutation_code,
                  exclude_areas,
                  include_areas, 
                  sample,
                  mutDict,
                  report_interval = 20, 
                  verbose = False,
                  histology = 'unknown',
                  muat_orig = False,
                  sort = True,
                  motif_length = 3) -> None:
    """
    
    
    """
    invalid_token = {}
    excluded_regions = 0 
    mut_not_found_middle = 0
    n_var = 0
    df_mutationcount = pd.DataFrame({'SNV': 0,
                                     'MNV': 0,
                                     'indel': 0,
                                     'SV/MEI': 0,
                                     'Normal': 0}, index=[0])
    try:
        if not os.path.isdir(os.path.join(tmp_dir, histology)):
            os.mkdir(os.path.join(tmp_dir, histology))
        if not os.path.isdir(os.path.join(tmp_dir, histology, sample)):
            os.mkdir(os.path.join(tmp_dir, histology, sample))
    except:
        print('Something went wrong when creating directory for sample')
    
    with open(input_file_name, 'rt') as input_file:
        for i, line in enumerate(input_file):
            if line.split(sep = ',')[0] == "chrom" and i == 0:
                continue
            line = line.split(sep = ',')

            chrom = line[0]
            position = line[1]

            if exclude_region(chrom=chrom, pos=position, to_exclude=exclude_areas, area=include_areas):
                excluded_regions += 1
                continue
            
            mut_idx = int(len(line[5])/2) # mutation is the 1025st char index is 1024 len should be 2048
            mut = line[5][mut_idx]
            
            # NEw APPROACH
            mut_type = classify_mutation_type(mutDict, line[5], mut_idx)
            
            if mut_type in ['SV', 'MEI', 'Neg']:
                if mut in NO_MUT:
                    mut_not_found_middle += 1
                    continue
                elif mut not in invalid_token.keys():
                    invalid_token[mut] = 1
                else:
                    invalid_token[mut] += 1
                continue

            else:
                output_path = os.path.join(tmp_dir, histology, sample, f'{mut_type}_{sample}.tsv.gz')


            with gzip.open(output_path, 'at') as output_file:
                if os.path.getsize(output_path) == 0:
                    output_file.write('chrom\tpos\tref\talt\tsample\tseq\tref_seq\tgc1kb\tgenic\texonic\tstrand\thistology\n')
                
                if muat_orig:
                    seq, ref_seq, ref, alt= get_orig_muat_seq(line, mut_idx, reverse_code, motif_length=motif_length)
                else:
                    seq, ref_seq, ref, alt= get_mutation_seq(line, reverse_code, motif_length=motif_length)
                
                if seq ==None:
                    continue

                df_mutationcount[mut_type] += 1

                new_row = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(chrom,
                                                                            position,
                                                                            ref,
                                                                            alt, 
                                                                            line[4], 
                                                                            seq,
                                                                            ref_seq,
                                                                            None,
                                                                            line[6],
                                                                            line[7],
                                                                            line[8],
                                                                            histology)
                output_file.write('{}\n'.format(new_row))

    #sort files 
    #sorting takes too much time
    if sort:

        for i in ['SNV_', 'MNV_', 'indel_']:
            output_path = os.path.join(tmp_dir, histology, sample, i + sample + '.tsv.gz')
            if os.path.exists(output_path):
                pd_sort = pd.read_csv(output_path, sep= '\t', index_col=False, compression='gzip', low_memory=False)
                pd_sort = pd_sort.sort_values(by=['chrom', 'pos'], key=natsort_keygen(), ignore_index=True)
                pd_sort.to_csv(output_path, sep='\t', compression='gzip')

    error_sample_name = False
    df_mutationcount.to_csv(os.path.join(tmp_dir, histology, sample, 'count_' + sample + '.tsv.gz'),compression='gzip', sep='\t')
    if len(invalid_token.keys()) > 0:
        sys.stderr.write(f'Sample {sample}:\n')
        error_sample_name = True
        for key in invalid_token.keys():
            sys.stderr.write('No implementation for mutation token {}. The Number of this kind of mutations in file was {}.\n'.format(key, int(invalid_token[key])))
    
    if excluded_regions > 0:
        if not error_sample_name:
            sys.stderr.write(f'Sample {sample}:\n')
        sys.stderr.write('{} mutations were excluded because of the poor readability, and they cannot be use by EpicVAE.\n'.format(excluded_regions))

    if mut_not_found_middle > 0:
        if not error_sample_name:
            sys.stderr.write(f'Sample {sample}:\n')
        sys.stderr.write('On {} row(s) the mutation was not found in the middle of the sequence. these sequences were ignored.\n'.format(mut_not_found_middle))


def annotate_mutations_(mutation_coding,
                       predict_filepath,
                       verbose,
                       tmp_dir,
                       continue_from = -1,
                       end = -1,
                       convert_hg38_hg19 = False,
                       muat_orig = False,
                       motif_length= 3, 
                       reference_h38 = None,
                       reference_h19 = None) -> None:
    """

    Args:
        mutation_coding
        predict_filepath 
        verbose
        reference_h19
        convert_hg38
        reference_h38
    """
    mutation_code, reverse_mutation_code = read_codes(mutation_coding)
    exclude_areas, include_areas = read_exclude()

    fns = pd.read_csv(predict_filepath, sep='\t', low_memory=False, header=0, compression='gzip')
    fns = fns['path'].tolist()

  
    if len(fns) ==0:
        sys.exit(1)

    missing = 0 
    for fn in fns:
        if os.path.exists(fn) == False:
            status('Input file {} not found\n'.format(fn), verbose)
            missing += 1
    if missing > 0:
        sys.exit(1)

    else:
        status('{} input file(s) found'.format(len(fns)), verbose)

    if not os.path.exists(tmp_dir):
        try:
            os.makedirs(tmp_dir, exist_ok=True)
        except:
            sys.stderr.write('Unexpected error in making temp dir {}'.format(tmp_dir))
            sys.exit(1)

    
    #all_error_file = []
    #all_succeed_file = []

    mutDict = pd.read_csv(os.path.join(os.getcwd(), os.pardir, 'extfiles', 'dictMotif_orig.csv'))

    for i, fn in enumerate(fns):
        if continue_from > -1:
            if i < continue_from-1:
                continue
            else:
                if i == continue_from -1:
                    status(f'Continuing from file {i+1}/{len(fns)}\t{fn}', verbose=True)
        
        if end > -1:
            status(f'Going to end at file number {end}', verbose=True)
            if i == end -1:
                status(f'Processing ended before processing file number {end}', verbose=True)
                break

        sample_name = fn.split('/')
        histology = sample_name[-2]
        sample_name = sample_name[-1][:-4]

        #output_file = os.path.join(tmp_dir, sample_name + '.tsv.gz')
        
        # Process mutation mutation sequences 
        status('Writing mutation sequences...', verbose)

        status('{}/{}\t{}'.format(i+1, len(fns), sample_name), verbose)
            
        if convert_hg38_hg19:
            process_input(input_file_name = fn,
                          tmp_dir= tmp_dir, 
                          reverse_code=reverse_mutation_code, 
                          mutation_code=mutation_code, 
                          exclude_areas=exclude_areas, 
                          include_areas=include_areas, 
                          sample=sample_name,
                          mutDict=mutDict, 
                          histology=histology,
                          muat_orig = muat_orig,
                          motif_length=motif_length)
        else:
            process_input(input_file_name = fn,
                          tmp_dir= tmp_dir, 
                          reverse_code=reverse_mutation_code, 
                          mutation_code=mutation_code, 
                          exclude_areas=exclude_areas, 
                          include_areas=include_areas, 
                          sample=sample_name,
                          mutDict=mutDict, 
                          histology=histology,
                          muat_orig = muat_orig,
                          motif_length=motif_length)
            

        status('Sample {} processed'.format(fn), verbose)
    
    del fns 

def main(predict_filepath, verbose, tmp_dir, muat_orig, motif_length, continue_from, end):
    
    mutation_coding = os.path.join(os.getcwd(), os.pardir, 'extfiles', 'mutation_codes_sv.tsv')

    annotate_mutations_(mutation_coding = mutation_coding,
                       predict_filepath = predict_filepath,
                       verbose = verbose,
                       tmp_dir = tmp_dir,
                       muat_orig= muat_orig,
                       motif_length= motif_length,
                       continue_from= continue_from,
                       end=end)
    
if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    #parser.add_argument("-r", "--header", action="store_true", help="determine if csv file to be handled has header row")
    parser.add_argument("-o", "--muat_orig", action="store_true", help="save sequences with the mutation tokens")
    parser.add_argument("-l", "--length", default=3, type=int, help="length of the motif, default 3")
    parser.add_argument("-c", "--continue_from", default=-1, type=int, help="the index of the file where to continue, optional")
    parser.add_argument("-e", "--end", default=-1, type=int, help="the index of the file where to end, optional" )
    parser.add_argument("predict_filepath", type=str, help="full path to csv file containing files to handle")
    parser.add_argument("tmp_dir", type=str, help="folder name where to store the processed files")
    args = parser.parse_args()
    
    main(predict_filepath = args.predict_filepath,
         verbose=args.verbose,
         tmp_dir=args.tmp_dir,
         muat_orig=args.muat_orig,
         motif_length=args.length,
         continue_from=args.continue_from,
         end=args.end)
