"""
Outdate file originally used to make dataset. However the new version rely on making dataloaders on demand and preprocessed data before each experiment. to reduce confusion, this is thus only needed if you want a new train val split.

Modified version of the original code: https://github.com/yewsiang/ConceptBottleneck

Make train, val, test datasets based on train_test_split.txt, and by sampling val_ratio of the official train data to make a validation set 
Each dataset is a list of metadata, each includes official image id, full image path, class label, attribute labels, attribute certainty scores, and attribute labels calibrated for uncertainty
"""
import os
import random
import pickle
import argparse
from os import listdir
from os.path import isfile, isdir, join
from collections import defaultdict as ddict
import numpy as np
import copy

def extract_data(data_dir,split_file_path=None):
    cwd = os.getcwd()
    data_path = join(data_dir,'images')
    val_ratio = 0.2

    path_to_id_map = dict() #map from full image path to image id
    with open(data_path.replace('images', 'images.txt'), 'r') as f:
        for line in f:
            items = line.strip().split()
            path_to_id_map[join(data_path, items[1])] = int(items[0])
    

    attribute_labels_all = ddict(list) #map from image id to a list of attribute labels
    attribute_certainties_all = ddict(list) #map from image id to a list of attribute certainties
    attribute_uncertain_labels_all = ddict(list) #map from image id to a list of attribute labels calibrated for uncertainty
    # 1 = not visible, 2 = guessing, 3 = probably, 4 = definitely
    uncertainty_map = {1: {1: 0, 2: 0.5, 3: 0.75, 4:1}, #calibrate main label based on uncertainty label
                        0: {1: 0, 2: 0.5, 3: 0.25, 4: 0}}
    with open(join(cwd, data_dir + '/attributes/image_attribute_labels.txt'), 'r') as f:
        for line in f:
            file_idx, attribute_idx, attribute_label, attribute_certainty = line.strip().split()[:4]
            attribute_label = int(attribute_label)
            attribute_certainty = int(attribute_certainty)
            uncertain_label = uncertainty_map[attribute_label][attribute_certainty]
            attribute_labels_all[int(file_idx)].append(attribute_label)
            attribute_uncertain_labels_all[int(file_idx)].append(uncertain_label)
            attribute_certainties_all[int(file_idx)].append(attribute_certainty)

    if split_file is None: 
        """
        Original methode has a random split making replication hard. Thus we find the train, test and validation set based on the original split see: train_test_val_finder.py
        """
        is_train_test = dict() #map from image id to 0 / 1 (1 = train)
        with open(join(cwd, data_dir + '/train_test_split.txt'), 'r') as f:
            for line in f:
                idx, is_train = line.strip().split()
                is_train_test[int(idx)] = int(is_train)
        print("Number of train images from official train test split:", sum(list(is_train_test.values())))

        train_val_data, test_data = [], []
        train_data, val_data = [], []
        folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
        folder_list.sort() #sort by class index
        for i, folder in enumerate(folder_list):
            folder_path = join(data_path, folder)
            classfile_list = [cf for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')]

            for cf in classfile_list:
                img_id = path_to_id_map[join(folder_path+'/'+cf)] #may cause bug in linux
                img_path = join(folder_path, cf)
                metadata = {'id': img_id, 'img_path': img_path, 'class_label': i,
                        'attribute_label': attribute_labels_all[img_id], 'attribute_certainty': attribute_certainties_all[img_id],
                        'uncertain_attribute_label': attribute_uncertain_labels_all[img_id]}
                if is_train_test[img_id]:
                    train_val_data.append(metadata)

                    val_data.append(metadata)

                    train_data.append(metadata)
                else:
                    test_data.append(metadata)

        random.shuffle(train_val_data)
        split = int(val_ratio * len(train_val_data))
        train_data = train_val_data[split :]
        val_data = train_val_data[: split]
        print('Size of train set:', len(train_data))
    
    else:
        # Open file made by train_test_val_finder.py
        split_dict = pickle.load(open(split_file_path, 'rb'))
        train_data_idx = split_dict['train']
        val_data_idx = split_dict['val']
        test_data_idx = split_dict['test']

        
        train_data, val_data, test_data = [], [], []

        folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
        folder_list.sort() #sort by class index
        data_dict = {}
        for i, folder in enumerate(folder_list):
            folder_path = join(data_path, folder)
            classfile_list = [cf for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')]

            
            for cf in classfile_list:
                img_id = path_to_id_map[join(folder_path+'/'+cf)] #may cause bug in linux
                img_path = join(folder_path, cf)
                data_dict[img_id]={'id': img_id, 'img_path': img_path, 'class_label': i,
                        'attribute_label': attribute_labels_all[img_id], 'attribute_certainty': attribute_certainties_all[img_id],
                        'uncertain_attribute_label': attribute_uncertain_labels_all[img_id]}
                
        for img_id in train_data_idx:
            train_data.append(data_dict[img_id])
        
        for img_id in val_data_idx:
            val_data.append(data_dict[img_id])
        
        for img_id in test_data_idx:
            test_data.append(data_dict[img_id])





    return train_data, val_data, test_data

N_ATTRIBUTES = 312
N_CLASSES = 200

def get_class_attributes_data(min_class_count, out_dir, modify_data_dir='', keep_instance_data=False):
    """
    Function from data_processing_CUB.py in the original implementation, note that the attributes filtered our depends on your train val split
    thus to replicate the function you need to use the original split.
    """

    data = pickle.load(open(join(modify_data_dir,'train.pkl'), 'rb'))
    class_attr_count = np.zeros((N_CLASSES, N_ATTRIBUTES, 2))
    for d in data:
        class_label = d['class_label']
        certainties = d['attribute_certainty']
        for attr_idx, a in enumerate(d['attribute_label']):
            if  certainties[attr_idx] == 1: #not visible
                continue
            class_attr_count[class_label][attr_idx][a] += 1

    class_attr_min_label = np.argmin(class_attr_count, axis=2)
    class_attr_max_label = np.argmax(class_attr_count, axis=2)
    equal_count = np.where(class_attr_min_label == class_attr_max_label) #check where 0 count = 1 count, set the corresponding class attribute label to be 1
    class_attr_max_label[equal_count] = 1

    attr_class_count = np.sum(class_attr_max_label, axis=0)
    mask = np.where(attr_class_count >= min_class_count)[0] #select attributes that are present (on a class level) in at least [min_class_count] classes
    class_attr_label_masked = class_attr_max_label[:, mask]
    if keep_instance_data:
        collapse_fn = lambda d: list(np.array(d['attribute_label'])[mask])
    else:
        collapse_fn = lambda d: list(class_attr_label_masked[d['class_label'], :])
        
        #Save version of the mask so that concept names can be retrieved later
        f = open(os.path.join(out_dir,'mask.pkl'), 'wb')
        pickle.dump(mask, f)
        f.close()

        
    create_new_dataset(out_dir, 'attribute_label', collapse_fn, data_dir=modify_data_dir)

def create_new_dataset(out_dir, field_change, compute_fn, datasets=['train', 'val', 'test'], data_dir=''):
    """
    Generic function that given datasets stored in data_dir, modify/ add one field of the metadata in each dataset based on compute_fn
                          and save the new datasets to out_dir
    compute_fn should take in a metadata object (that includes 'img_path', 'class_label', 'attribute_label', etc.)
                          and return the updated value for field_change
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for dataset in datasets:
        path = os.path.join(data_dir, dataset + '.pkl')
        if not os.path.exists(path):
            continue
        data = pickle.load(open(path, 'rb'))
        new_data = []
        for d in data:
            new_d = copy.deepcopy(d)
            new_value = compute_fn(d)
            if field_change in d:
                old_value = d[field_change]
                assert (type(old_value) == type(new_value))
            new_d[field_change] = new_value
            new_data.append(new_d)
        f = open(os.path.join(out_dir, dataset + '.pkl'), 'wb')
        pickle.dump(new_data, f)
        f.close()


if __name__ == "__main__":
    # TODO make this agnostic of operation system. 
    data_dir = r'data/CUB_200_2011'
    save_dir_unfiltered = r'data/CUB_processed/unfiltered'
    save_dir_filtered = r'data/CUB_processed/filtered'
    split_file = r"data/train_test_val.pkl"
    train_data, val_data, test_data = extract_data(data_dir,split_file)

    #Make dir if not exist
    if not os.path.exists(save_dir_unfiltered):
        os.makedirs(save_dir_unfiltered)

    for dataset in ['train','val','test']:
        print("Processing %s set" % dataset)
        f = open(os.path.join(save_dir_unfiltered,dataset + '.pkl'), 'wb')
        if 'train' in dataset:
            pickle.dump(train_data, f)
        elif 'val' in dataset:
            pickle.dump(val_data, f)
        else:
            pickle.dump(test_data, f)
        f.close()

    #Save copy to get later modified
    if not os.path.exists(save_dir_filtered):
        os.makedirs(save_dir_filtered)

    for dataset in ['train','val','test']:
        print("Processing %s set" % dataset)
        f = open(os.path.join(save_dir_filtered,dataset + '.pkl'), 'wb')
        if 'train' in dataset:
            pickle.dump(train_data, f)
        elif 'val' in dataset:
            pickle.dump(val_data, f)
        else:
            pickle.dump(test_data, f)
        f.close()

    #Modify the filtered dataset
    get_class_attributes_data(10, save_dir_filtered, save_dir_unfiltered, keep_instance_data=False)



    

