import ntpath
import glob
import os
import pickle
import numpy as np


# define link to file data
PROJECT_SOURCE = 'data\\sourceFile_aspectj'
SOURCE_NAME = 'data/Aspectj_name_full_link.pickle'


def getName(file):
    return ntpath.basename(file)


# function read folder
def openFolder(path, files, agr):
    files.extend(glob.glob(os.path.join(path, agr)))
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        if os.path.isdir(full_path) and not os.path.islink(full_path):
            openFolder(full_path, files, agr)


def clean_link(list_link):
    result =[]
    for link in list_link:
        link_clean = link.replace('data\\sourceFile_aspectj\\org.aspectj\\', '')
        link_clean = link_clean.replace('\\', '/')
        result.append(link_clean)
    return result


def delete_commit(link):
    path = os.path.normpath(link)
    token = path.split(os.sep)
    name = token[len(token)-1].split()
    return link.replace(token[len(token) - 1], '') + name[1]


def get_list_name_not_commit():
    # get name source include commit in project
    f = open(SOURCE_NAME, 'rb')
    pickle_input = pickle.load(f)

    # get the name source not include commit
    list_source = []
    for link in pickle_input:
        link_clean = delete_commit(link)
        if link_clean not in list_source:
            list_source.append(link_clean)

    f = open('data/list_name_not_commit.pickle', 'wb')
    pickle.dump(list_source, f)
    return list_source
