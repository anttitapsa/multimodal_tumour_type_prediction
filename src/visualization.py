import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.transforms import ScaledTranslation
sys.path.insert(0, '/csc/epitkane/projects/multimodal/src/')
from utils import list_files_in_dir
from sklearn.metrics import f1_score, matthews_corrcoef, top_k_accuracy_score, accuracy_score, precision_score, recall_score

def get_logits_dict(path, best=True):
    logits_dict = {}
    for i in range(1, 11):
        if best:
            file = os.path.join(path, f'fold_{i}/val_logits_fold{i}_best_vallogits.tsv.gz')
        else:
            file = os.path.join(path, f'fold_{i}/val_logits_fold{i}.tsv.gz')
        logits_dict[i] = pd.read_csv(file, compression='gzip', sep= '\t')
    return logits_dict

def ensemble(models):
    logits = models[1].loc[:,'Bone-Osteosarc':'Uterus-AdenoCA']
    for i in range(2,11):
        logits_df = models[i]
        logits += logits_df.loc[:,'Bone-Osteosarc':'Uterus-AdenoCA']
    logits['target'] = models[1].loc[:,'target']    
    
    return count_accuracy(logits)

def avg(results):
    best = 0
    average = 0 
    fold = 1
    for i in range(1, 11):
        M = results[i]
        acc = count_accuracy(M)
        average += acc
        if acc > best:
            best = acc
            fold = i
    average /= 10
    return average

def count_accuracy(df):
    correct = 0
    targets = df.loc[:,'target']
    logits = df.loc[:,'Bone-Osteosarc':'Uterus-AdenoCA']
    for i in range(0, len(df)):
        #target = int(targets.iloc[i].strip('[]'))
        target= targets.iloc[i]
        row_logits = logits.loc[i,:]
        #print(row_logits)
        pred = row_logits.argmax()
        #print(f'pred: {row_logits.iloc[pred]}, target: {row_logits.iloc[target]}')
        if target == pred:
            correct += 1
    return correct/len(df)

def transform_data(df):
    logits = df.loc[:,'Bone-Osteosarc':'Uterus-AdenoCA'].to_numpy()
    targets = df.loc[:,'target'].to_numpy()
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    prob = exp / np.sum(exp, axis=1, keepdims=True)
    return np.argmax(prob, axis=1), targets, prob


def main():
    #ensemble logits
    motif_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtMotif", best=False)
    motifPos_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtMotifPos", best=False)
    motifPosGES_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtMotifPosGES", best=False)
    motif3_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtMotif3", best=False)
    motif3Pos_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtMotif3Pos", best=False)
    motif3PosGES_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtMotif3PosGES", best=False)
    motif101_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtMotif101", best=False)
    motif101Pos_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtMotif101pos", best=False)
    motif101PosGES_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtMotif101posGES", best=False)
    motifOH3_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtOneHot3", best=False)
    motifOH3Pos_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtOneHot3Pos", best=False)
    motifOH3PosGES_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtOneHot3PosGES", best=False)
    motifOH101_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtOneHot101", best=False)
    motifOH101Pos_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtOneHot101Pos", best=False)
    motifOH101PosGES_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtOneHot101PosGES", best=False)
    motif3_SNV_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtMotif3_SNV_only", best=False)

    motif_esize_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtMotif_esize", best=False)
    motifPos_esize_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtMotifPos_esize", best=False)
    motifPosGES_esize_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtMotifPosGES_esize", best=False)
    motif_epic_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtEpiPos", best=False)
    motifGES_epic_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtEpiPosGES", best=False)
    motifPos_epic_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtMotifPositionEpiPos", best=False)
    motifPosGES_epic_en = get_logits_dict("/csc/epitkane/projects/multimodal/ensemble/MuAtMotifPositionGESEpiPos", best=False)

    #validation logits
    Muat_orig = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtMotif_orig')
    Muat_origpos = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtMotifPos_orig')
    Muat_origposGES = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtMotifPosGES_orig')
    motif3 = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtMotif3')
    motif3pos = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtMotif3Pos')
    motif3posGES = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtMotif3PosGES')
    motif101 = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtMotif101')
    motif101pos = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtMotif101Pos')
    motif101posGES = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtMotif101PosGES')
    onehot3 = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtOneHot3')
    onehot3pos = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtOneHot3Pos')
    onehot3posGES = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtOneHot3PosGES')
    onehot101 = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtOneHot101')
    onehot101pos = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtOneHot101Pos')
    onehot101posGES = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtOneHot101PosGES')
    motif3_SNV = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtMotif3_SNV_only')

    orig_esized = get_logits_dict("/csc/epitkane/projects/multimodal/models/MuAtMotif_orig_esize")
    origpos_esize = get_logits_dict("/csc/epitkane/projects/multimodal/models/MuAtMotifPos_orig_esize")
    origposGES_esized = get_logits_dict("/csc/epitkane/projects/multimodal/models/MuAtMotifPosGES_orig_esize")
    Muat_epipos = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtEpiPos')
    epipos_GES = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtEpiPosGES')
    epipos_position = get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtMotifPositionEpiPos')
    epipos_position_GES =  get_logits_dict('/csc/epitkane/projects/multimodal/models/MuAtMotifPositionGESEpiPos')

    models =  [ Muat_orig,
                Muat_origpos,
                Muat_origposGES,
                motif3,
                motif3pos,
                motif3posGES,
                motif101,
                motif101pos,
                motif101posGES,
                onehot3,
                onehot3pos,
                onehot3posGES,
                onehot101,
                onehot101pos,
                onehot101posGES,
                motif3_SNV]
    
    ensembles = [motif_en,
                motifPos_en,
                motifPosGES_en,
                motif3_en,
                motif3Pos_en,
                motif3PosGES_en,
                motif101_en,
                motif101Pos_en,
                motif101PosGES_en,
                motifOH3_en,
                motifOH3Pos_en,
                motifOH3PosGES_en,
                motifOH101_en,
                motifOH101Pos_en,
                motifOH101PosGES_en,
                motif3_SNV_en]
    
    fig = plt.figure( dpi=450)
    ax = plt.subplot()
    for idx, model in enumerate(models):
        for i in range(1, 11):
            pred, target, prob = transform_data(model[i])
            ax.plot(idx + idx*0.5, precision_score(target, pred, average='macro'), marker='o', color='b', alpha=0.2)
        ax.plot(idx + idx*0.5, precision_score(
        np.concatenate([transform_data(model[i])[1] for i in range(1, 11)], axis=None),
        np.concatenate([transform_data(model[i])[0] for i in range(1, 11)], axis=None),
        average='macro'
    ), marker='+', color='r')
        #ax.plot(idx+idx*0.5, ensemble(ensemb), marker='d', color='r')
    plt.xticks([i + i*0.5 for i in range(len(models))], ['Motif',
                            'Motif Pos',
                            'Motif Pos GES',
                            'DNABERT Motif3',
                            'DNABERT Motif3 Pos',
                            'DNABERT Motif3 Pos GES', 
                            'DNABERT Motif101',
                            'DNABERT Motif101 Pos',
                            'DNABERT Motif101 Pos GES',
                            'One-Hot Motif3',
                            'One-Hot Motif3 Pos',
                            'One-Hot Motif3 Pos GES',
                            'One-Hot Motif101',
                            'One-Hot Motif101 Pos',
                            'One-Hot Motif101 Pos GES',
                            'DNABERT Motif3 SNVs only' ], rotation =80, ha='right',rotation_mode='anchor')
    plt.ylabel('Precision')
    blue_circle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='Precision of fold', alpha= 0.2)
    red_plus = mlines.Line2D([], [], color='red', marker='+', linestyle='None', markersize=8, label='Average Precision')
    #red_marker = mlines.Line2D([], [], color='red', marker='d', linestyle='None', markersize=8, label='Ensemble Accuracy')
    lgd = ax.legend(handles=[blue_circle, red_plus], bbox_to_anchor=(1.38, 1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('/csc/epitkane/projects/multimodal/figures/' + f'precision.png', format="png",bbox_inches='tight')

##########################
    fig = plt.figure( dpi=450)
    ax = plt.subplot()
    for idx, model in enumerate(models):
        for i in range(1, 11):
            pred, target, prob = transform_data(model[i])
            ax.plot(idx + idx*0.5, recall_score(target, pred, average='macro'), marker='o', color='b', alpha=0.2)
        ax.plot(idx + idx*0.5, recall_score(
        np.concatenate([transform_data(model[i])[1] for i in range(1, 11)], axis=None),
        np.concatenate([transform_data(model[i])[0] for i in range(1, 11)], axis=None),
        average='macro'
    ), marker='+', color='r')
        #ax.plot(idx+idx*0.5, ensemble(ensemb), marker='d', color='r')
    plt.xticks([i + i*0.5 for i in range(len(models))], ['Motif',
                            'Motif Pos',
                            'Motif Pos GES',
                            'DNABERT Motif3',
                            'DNABERT Motif3 Pos',
                            'DNABERT Motif3 Pos GES', 
                            'DNABERT Motif101',
                            'DNABERT Motif101 Pos',
                            'DNABERT Motif101 Pos GES',
                            'One-Hot Motif3',
                            'One-Hot Motif3 Pos',
                            'One-Hot Motif3 Pos GES',
                            'One-Hot Motif101',
                            'One-Hot Motif101 Pos',
                            'One-Hot Motif101 Pos GES',
                            'DNABERT Motif3 SNVs only' ], rotation =80, ha='right',rotation_mode='anchor')
    plt.ylabel('Recall')
    blue_circle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='Recall of fold', alpha= 0.2)
    red_plus = mlines.Line2D([], [], color='red', marker='+', linestyle='None', markersize=8, label='Average Recall')
    #red_marker = mlines.Line2D([], [], color='red', marker='d', linestyle='None', markersize=8, label='Ensemble Accuracy')
    lgd = ax.legend(handles=[blue_circle, red_plus], bbox_to_anchor=(1.38, 1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('/csc/epitkane/projects/multimodal/figures/' + f'recall.png', format="png",bbox_inches='tight')

####################
    fig = plt.figure( dpi=450)
    ax = plt.subplot()
    for idx, model in enumerate(models):
        for i in range(1, 11):
            pred, target, prob = transform_data(model[i])
            ax.plot(idx + idx*0.5, f1_score(target, pred, average='macro'), marker='o', color='b', alpha=0.2)
        ax.plot(idx + idx*0.5, f1_score(
        np.concatenate([transform_data(model[i])[1] for i in range(1, 11)], axis=None),
        np.concatenate([transform_data(model[i])[0] for i in range(1, 11)], axis=None),
        average='macro'
    ), marker='+', color='r')
        #ax.plot(idx+idx*0.5, ensemble(ensemb), marker='d', color='r')
    plt.xticks([i + i*0.5 for i in range(len(models))], ['Motif',
                            'Motif Pos',
                            'Motif Pos GES',
                            'DNABERT Motif3',
                            'DNABERT Motif3 Pos',
                            'DNABERT Motif3 Pos GES', 
                            'DNABERT Motif101',
                            'DNABERT Motif101 Pos',
                            'DNABERT Motif101 Pos GES',
                            'One-Hot Motif3',
                            'One-Hot Motif3 Pos',
                            'One-Hot Motif3 Pos GES',
                            'One-Hot Motif101',
                            'One-Hot Motif101 Pos',
                            'One-Hot Motif101 Pos GES',
                            'DNABERT Motif3 SNVs only' ], rotation =80, ha='right',rotation_mode='anchor')
    plt.ylabel('F1')
    blue_circle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='F1 of fold', alpha= 0.2)
    red_plus = mlines.Line2D([], [], color='red', marker='+', linestyle='None', markersize=8, label='Average F1')
    #red_marker = mlines.Line2D([], [], color='red', marker='d', linestyle='None', markersize=8, label='Ensemble Accuracy')
    lgd = ax.legend(handles=[blue_circle, red_plus], bbox_to_anchor=(1.38, 1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('/csc/epitkane/projects/multimodal/figures/' + f'F1.png', format="png",bbox_inches='tight')

############

    fig = plt.figure( dpi=450)
    ax = plt.subplot()
    for idx, model in enumerate(models):
        for i in range(1, 11):
            pred, target, prob = transform_data(model[i])
            ax.plot(idx + idx*0.5, matthews_corrcoef(target, pred), marker='o', color='b', alpha=0.2)
        ax.plot(idx + idx*0.5, matthews_corrcoef(
        np.concatenate([transform_data(model[i])[1] for i in range(1, 11)], axis=None),
        np.concatenate([transform_data(model[i])[0] for i in range(1, 11)], axis=None),
    ), marker='+', color='r')
        #ax.plot(idx+idx*0.5, ensemble(ensemb), marker='d', color='r')
    plt.xticks([i + i*0.5 for i in range(len(models))], ['Motif',
                            'Motif Pos',
                            'Motif Pos GES',
                            'DNABERT Motif3',
                            'DNABERT Motif3 Pos',
                            'DNABERT Motif3 Pos GES', 
                            'DNABERT Motif101',
                            'DNABERT Motif101 Pos',
                            'DNABERT Motif101 Pos GES',
                            'One-Hot Motif3',
                            'One-Hot Motif3 Pos',
                            'One-Hot Motif3 Pos GES',
                            'One-Hot Motif101',
                            'One-Hot Motif101 Pos',
                            'One-Hot Motif101 Pos GES',
                            'DNABERT Motif3 SNVs only' ], rotation =80, ha='right',rotation_mode='anchor')
    plt.ylabel('MCC')
    blue_circle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='MCC of fold', alpha= 0.2)
    red_plus = mlines.Line2D([], [], color='red', marker='+', linestyle='None', markersize=8, label='Average MCC')
    #red_marker = mlines.Line2D([], [], color='red', marker='d', linestyle='None', markersize=8, label='Ensemble Accuracy')
    lgd = ax.legend(handles=[blue_circle, red_plus], bbox_to_anchor=(1.38, 1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('/csc/epitkane/projects/multimodal/figures/' + f'MCC.png', format="png",bbox_inches='tight')
################
    models =  [ orig_esized,
                origpos_esize,
                origposGES_esized,
                Muat_epipos, 
                epipos_GES,
                epipos_position,
                epipos_position_GES ]

    fig = plt.figure( dpi=450)
    ax = plt.subplot()
    for idx, model in enumerate(models):
        for i in range(1, 11):
            pred, target, prob = transform_data(model[i])
            ax.plot(idx , precision_score(target, pred, average='macro'), marker='o', color='b', alpha=0.2)
        ax.plot(idx , precision_score(
        np.concatenate([transform_data(model[i])[1] for i in range(1, 11)], axis=None),
        np.concatenate([transform_data(model[i])[0] for i in range(1, 11)], axis=None),
        average='macro'
    ), marker='+', color='r')
        #x.plot(idx+idx*0.5, ensemble(ensemb), marker='d', color='r')
    plt.xticks(range(0, 7), ['Motif',
                            'Motif Pos',
                            'Motif Pos GES',
                            'Motif epigenetic',
                            'Motif GES epigenetic',
                            'Motif Pos epigenetic',
                            'Motif Pos GES epigenetic'], rotation =80, ha="right", rotation_mode='anchor' )
    plt.ylabel('Precision')
    blue_circle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='Precision of fold', alpha= 0.2)
    red_plus = mlines.Line2D([], [], color='red', marker='+', linestyle='None', markersize=8, label='Average Precision')
    #red_marker = mlines.Line2D([], [], color='red', marker='d', linestyle='None', markersize=8, label='Ensemble Accuracy')
    lgd = ax.legend(handles=[blue_circle, red_plus], bbox_to_anchor=(1.38, 1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('/csc/epitkane/projects/multimodal/figures/' + f'precision_e.png', format="png",bbox_inches='tight')

##########################
    fig = plt.figure( dpi=450)
    ax = plt.subplot()
    for idx, model in enumerate(models):
        for i in range(1, 11):
            pred, target, prob = transform_data(model[i])
            ax.plot(idx , recall_score(target, pred, average='macro'), marker='o', color='b', alpha=0.2)
        ax.plot(idx , recall_score(
        np.concatenate([transform_data(model[i])[1] for i in range(1, 11)], axis=None),
        np.concatenate([transform_data(model[i])[0] for i in range(1, 11)], axis=None),
        average='macro'
    ), marker='+', color='r')
        #ax.plot(idx+idx*0.5, ensemble(ensemb), marker='d', color='r')
    plt.xticks(range(0, 7), ['Motif',
                            'Motif Pos',
                            'Motif Pos GES',
                            'Motif epigenetic',
                            'Motif GES epigenetic',
                            'Motif Pos epigenetic',
                            'Motif Pos GES epigenetic'], rotation =80, ha="right", rotation_mode='anchor' )
    plt.ylabel('Recall')
    blue_circle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='Recall of fold', alpha= 0.2)
    red_plus = mlines.Line2D([], [], color='red', marker='+', linestyle='None', markersize=8, label='Average Recall')
    #red_marker = mlines.Line2D([], [], color='red', marker='d', linestyle='None', markersize=8, label='Ensemble Accuracy')
    lgd = ax.legend(handles=[blue_circle, red_plus], bbox_to_anchor=(1.38, 1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('/csc/epitkane/projects/multimodal/figures/' + f'recall_e.png', format="png",bbox_inches='tight')

####################
    fig = plt.figure( dpi=450)
    ax = plt.subplot()
    for idx, model in enumerate(models):
        for i in range(1, 11):
            pred, target, prob = transform_data(model[i])
            ax.plot(idx , f1_score(target, pred, average='macro'), marker='o', color='b', alpha=0.2)
        ax.plot(idx , f1_score(
        np.concatenate([transform_data(model[i])[1] for i in range(1, 11)], axis=None),
        np.concatenate([transform_data(model[i])[0] for i in range(1, 11)], axis=None),
        average='macro'
    ), marker='+', color='r')
    plt.xticks(range(0, 7), ['Motif',
                            'Motif Pos',
                            'Motif Pos GES',
                            'Motif epigenetic',
                            'Motif GES epigenetic',
                            'Motif Pos epigenetic',
                            'Motif Pos GES epigenetic'], rotation =80, ha="right", rotation_mode='anchor' )
    plt.ylabel('F1')
    blue_circle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='F1 of fold', alpha= 0.2)
    red_plus = mlines.Line2D([], [], color='red', marker='+', linestyle='None', markersize=8, label='Average F1')
    #red_marker = mlines.Line2D([], [], color='red', marker='d', linestyle='None', markersize=8, label='Ensemble Accuracy')
    lgd = ax.legend(handles=[blue_circle, red_plus], bbox_to_anchor=(1.38, 1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('/csc/epitkane/projects/multimodal/figures/' + f'F1_e.png', format="png",bbox_inches='tight')

############

    fig = plt.figure( dpi=450)
    ax = plt.subplot()
    for idx, model in enumerate(models):
        for i in range(1, 11):
            pred, target, prob = transform_data(model[i])
            ax.plot(idx , matthews_corrcoef(target, pred), marker='o', color='b', alpha=0.2)
        ax.plot(idx, matthews_corrcoef(
        np.concatenate([transform_data(model[i])[1] for i in range(1, 11)], axis=None),
        np.concatenate([transform_data(model[i])[0] for i in range(1, 11)], axis= None),
    ), marker='+', color='r')
    plt.xticks(range(0, 7), ['Motif',
                            'Motif Pos',
                            'Motif Pos GES',
                            'Motif epigenetic',
                            'Motif GES epigenetic',
                            'Motif Pos epigenetic',
                            'Motif Pos GES epigenetic'], rotation =80, ha="right", rotation_mode='anchor' )
    plt.ylabel('MCC')
    blue_circle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='MCC of fold', alpha= 0.2)
    red_plus = mlines.Line2D([], [], color='red', marker='+', linestyle='None', markersize=8, label='Average MCC')
    #red_marker = mlines.Line2D([], [], color='red', marker='d', linestyle='None', markersize=8, label='Ensemble Accuracy')
    lgd = ax.legend(handles=[blue_circle, red_plus], bbox_to_anchor=(1.38, 1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig('/csc/epitkane/projects/multimodal/figures/' + f'MCC_e.png', format="png",bbox_inches='tight')

    models =  [ Muat_orig,
                Muat_origpos,
                Muat_origposGES,
                motif3,
                motif3pos,
                motif3posGES,
                motif101,
                motif101pos,
                motif101posGES,
                onehot3,
                onehot3pos,
                onehot3posGES,
                onehot101,
                onehot101pos,
                onehot101posGES,
                motif3_SNV]
    
    ensembles = [motif_en,
                motifPos_en,
                motifPosGES_en,
                motif3_en,
                motif3Pos_en,
                motif3PosGES_en,
                motif101_en,
                motif101Pos_en,
                motif101PosGES_en,
                motifOH3_en,
                motifOH3Pos_en,
                motifOH3PosGES_en,
                motifOH101_en,
                motifOH101Pos_en,
                motifOH101PosGES_en,
                motif3_SNV_en]
    
    fig = plt.figure( dpi=450)
    ax = plt.subplot()
    for idx, (model, ensemb) in enumerate(zip(models, ensembles)):
        for i in range(1, 11):
            ax.plot(idx + idx*0.5, count_accuracy(model[i]), marker='o', color='b', alpha=0.2)
        ax.plot(idx+idx*0.5, avg(model), marker='+', color='r')
        ax.plot(idx+idx*0.5, ensemble(ensemb), marker='d', color='r')
    plt.xticks([i + i*0.5 for i in range(len(models))], ['Motif',
                                                        'Motif Pos',
                                                        'Motif Pos GES',
                                                        'DNABERT Motif3',
                                                        'DNABERT Motif3 Pos',
                                                        'DNABERT Motif3 Pos GES', 
                                                        'DNABERT Motif101',
                                                        'DNABERT Motif101 Pos',
                                                        'DNABERT Motif101 Pos GES',
                                                        'One-Hot Motif3',
                                                        'One-Hot Motif3 Pos',
                                                        'One-Hot Motif3 Pos GES',
                                                        'One-Hot Motif101',
                                                        'One-Hot Motif101 Pos',
                                                        'One-Hot Motif101 Pos GES',
                                                        'DNABERT Motif3 SNVs only' ], rotation =80, ha='right',rotation_mode='anchor')
    plt.ylabel('accuracy')
    blue_circle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='Accuracy of Fold', alpha= 0.2)
    red_plus = mlines.Line2D([], [], color='red', marker='+', linestyle='None', markersize=8, label='Average Accuracy')
    red_marker = mlines.Line2D([], [], color='red', marker='d', linestyle='None', markersize=8, label='Ensemble Accuracy')
    lgd = ax.legend(handles=[blue_circle, red_plus, red_marker], bbox_to_anchor=(1.38, 1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim((0.25, 1))
    plt.savefig('/csc/epitkane/projects/multimodal/figures/' + f'models2.png', format="png",bbox_inches='tight')

    for e in ensembles:
        print(ensemble(e))

    models =  [ orig_esized,
                origpos_esize,
                origposGES_esized,
                Muat_epipos, 
                epipos_GES,
                epipos_position,
                epipos_position_GES ]

    ensembles = [motif_esize_en,
                motifPos_esize_en,
                motifPosGES_esize_en,
                motif_epic_en,
                motifGES_epic_en,
                motifPos_epic_en,
                motifPosGES_epic_en]

    fig = plt.figure( dpi=450)
    ax = plt.subplot()
    for idx, (model, ensemb) in enumerate(zip(models, ensembles)):
        for i in range(1, 11):
            ax.plot(idx, count_accuracy(model[i]), marker='o', color='b', alpha=0.2)
        ax.plot(idx, avg(model), marker='+', color='r')
        ax.plot(idx, ensemble(ensemb), marker='d', color='r')
    plt.xticks(range(0, 7), ['Motif',
                            'Motif Pos',
                            'Motif Pos GES',
                            'Motif epigenetic',
                            'Motif GES epigenetic',
                            'Motif Pos epigenetic',
                            'Motif Pos GES epigenetic'], rotation =80, ha="right", rotation_mode='anchor' )
    plt.ylabel('accuracy')
    #plt.title('Validation Loss')

    blue_circle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=8, label='Accuracy of Fold', alpha= 0.2)
    red_plus = mlines.Line2D([], [], color='red', marker='+', linestyle='None', markersize=8, label='Average Accuracy')
    red_marker = mlines.Line2D([], [], color='red', marker='d', linestyle='None', markersize=8, label='Ensemble Accuracy')
    lgd = ax.legend(handles=[blue_circle, red_plus, red_marker], bbox_to_anchor=(1.38, 1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim((0.75, 1))
    plt.savefig('/csc/epitkane/projects/multimodal/figures/' + f'models_esize2.png', format="png",bbox_inches='tight')

    for e in ensembles:
        print(ensemble(e))

if __name__ == '__main__':
    main()
