import os
import json
from nltk.corpus import wordnet as wn

# The path that contains the dataset
PATH = './datasets/imagenette/'


def get_folders(path_to_folder):
    """List the folders of a given path
    
    Arguments:
        path_to_folder {sring} -- The string of the path to be looked at

    Returns:
        list -- List of folders in the given path
    """
    return next(os.walk(path_to_folder))[1]


def get_synset_names(synset_ids):
    """Gets the WordNet synset names from the provided list of sysnset IDs
    
    Arguments:
        synset_ids {string} -- The string of the synset that is going to be
        converted to a synset name
    
    Returns:
        list -- List of synset strings containing only the name of the synset
        without further information
    """
    sysnset_names = []

    for id in synset_ids:
        # map a synset id to an existing sysnset
        # e.g. 'n01440764' to Synset('tench.n.01')
        synset = wn.synset_from_pos_and_offset('n', int(id[1:]))
        # get only the synsets' name
        sysnset_names.append(synset.name().split('.')[0])

    return sysnset_names


def create_json():
    """Calls the other methods to finally create a correctly formatted JSON
    file for the training/ test data and WordNet IDs
    """
    train_folders = get_folders(PATH + 'train')
    val_folders = get_folders(PATH + 'val')

    train_names = get_synset_names(train_folders)
    val_names = get_synset_names(val_folders)

    to_json = {
        "train": train_folders,
        "test": val_folders,
        "train_names": train_names,
        "test_names": val_names
    }

    with open(PATH + 'imagenette-split.json', 'w') as fp:
        json.dump(to_json, fp)


if __name__ == '__main__':
    create_json()
