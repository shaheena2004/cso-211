import os
import numpy as np
import scipy as sp
from collections import defaultdict
from src.WNNGIP import WNNGIP
from src.dataset_names import DATASET_NAME

def load_data_from_file(dataset, folder):
    with open(os.path.join(folder, dataset+"_admat_dgc.txt"), "r") as inf:
        text = inf.read().rstrip().split('\n')
        int_array = [line.split('\t')[1:] for line in text[1:]]

    with open(os.path.join(folder, dataset+"_simmat_dc.txt"), "r") as inf:
        text = inf.read().rstrip().split('\n')
        drug_sim = [line.rstrip().split('\t')[1:] for line in text[1:]]

    with open(os.path.join(folder, dataset+"_simmat_dg.txt"), "r") as inf:
        text = inf.read().rstrip().split('\n')
        target_sim = [line.rstrip().split('\t')[1:] for line in text[1:]]

    intMat = np.array(int_array, dtype=np.float64).T
    drugMat = np.array(drug_sim, dtype=np.float64)
    targetMat = np.array(target_sim, dtype=np.float64)
    return intMat, drugMat, targetMat

def cross_validation(intMat, seeds, cv=0, num=10):
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:
            index = prng.permutation(num_drugs)
        if cv == 1:
            index = prng.permutation(intMat.size)
        step = int(index.size/num)
        for i in range(num):
            if i < num-1:
                ii = index[i*step:(i+1)*step]
            else:
                ii = index[i*step:]
            if cv == 0:
                test_data = np.array([[k, j] for k in ii for j in range(num_targets)], dtype=np.int32)
            elif cv == 1:
                test_data = np.array([[k/num_targets, k % num_targets] for k in ii], dtype=np.int32)
            x, y = test_data[:, 0], test_data[:, 1]
            test_label = intMat[x, y]
            W = np.ones(intMat.shape)
            W[x, y] = 0
            cv_data[seed].append((W, test_data, test_label))
    return cv_data

def train(model, cv_data, intMat, drugMat, targetMat):
    auc = []
    for seed in cv_data.keys():
        for W, test_data, test_label in cv_data[seed]:
            model.fix_model(W, intMat, drugMat, targetMat, seed)
            aupr_val, auc_val = model.evaluation(test_data, test_label)
            auc.append(auc_val)
    return np.array(auc, dtype=np.float64)

def find_mean(data):
    m = np.mean(1.0*np.array(data))
    return m


def write_metric_vector_to_file(auc_vec, file_name):
    np.savetxt(file_name, auc_vec, fmt='%.6f')

def predict():

    data_dir = os.path.join('data')
    output_dir = os.path.join('output')
    cvs, model_settings, predict_num, method, dataset = 1, [], 1, "wnngip", DATASET_NAME

    seeds = [7771, 8367, 22, 1812, 4659]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if method == 'wnngip':
        args = {'T': 0.7, 'sigma': 1.0, 'alpha': 0.8}

    for key, val in model_settings:
        args[key] = val

    intMat, drugMat, targetMat = load_data_from_file(dataset, os.path.join(data_dir, 'datasets'))

    if cvs == 1:  # CV setting CVS1
        X, D, T, cv = intMat, drugMat, targetMat, 1
    if cvs == 2:  # CV setting CVS2
        X, D, T, cv = intMat, drugMat, targetMat, 0
    if cvs == 3:  # CV setting CVS3
        X, D, T, cv = intMat.T, targetMat, drugMat, 0
    cv_data = cross_validation(X, seeds, cv)  # separate data for 5 * 10-fold cross-validation

    model = WNNGIP(T=args['T'], sigma=args['sigma'], alpha=args['alpha'])
    auc_vec = train(model, cv_data, X, D, T)
    auc_avg = find_mean(auc_vec)
    # write_metric_vector_to_file(auc_vec, os.path.join(output_dir, method+"_auc_cvs"+str(cvs)+"_"+dataset+".txt"))
    return auc_avg
