import pickle
import json
import numpy as np
import pandas as pd
import os

# define name data
BUG_MATRIX = 'data/Bug_matrix_AspectJ.txt'
SOURCE_MATRIX = 'data/matrix_sourceAspectj.pickle'
SOURCE_ID = 'data/DataOfDat/preprocessed_src_id.json'
FEATURE = 'data/DataOfDat/features1_update.json'
BUG_CSV = 'data/AspectJ_csv.csv'
NAME_SOURCE = 'data/Aspectj_name_full_link.pickle'
CONNECT = 'data/data_connect.csv'

# prepare data
# # feature cosine
with open(FEATURE, 'rb') as file:
    features = json.load(file)  # ['data']

# with open(FEATURE, 'rb') as file:
#     features = pickle.load(file)  # ['data']

# name source with feature
pickle_input = open(SOURCE_ID, 'rb')
source_id = json.load(pickle_input)
pickle_input.close()

# name source with matrix word embedding
pickle_input = open(NAME_SOURCE, 'rb')
name_source = pickle.load(pickle_input)
pickle_input.close()

# data connect bug and source
file = pd.read_csv(CONNECT)
file_1 = file['source_label_1'].values
file_0 = file['source_label_0'].values

# matrix source
pickle_input = open(SOURCE_MATRIX, 'rb')
source_matrix = pickle.load(pickle_input)

# matrix_bug
pickle_input = open(BUG_MATRIX, 'rb')
matrix_bug = pickle.load(pickle_input)


def concatenate(matrix_a, matrix_b):
    result = []
    for row in matrix_a:
        result.append(row)
    for row in matrix_b:
        result.append(row)
    return np.asarray(result)


def delete_commit(link):
    path = os.path.normpath(link)
    token = path.split(os.sep)
    name = token[len(token)-1].split()
    return link.replace(token[len(token) - 1], '') + name[1]


def qs(cosine, l, r, tt):  # sort cosine to select 300 min cosine
    i = l
    j = r
    tg = cosine[int((l + r) / 2)]
    while i <= j:
        while cosine[i] < tg and i <= j:
            i += 1
        while cosine[j] > tg and i <= j:
            j -= 1
        if i <= j:
            cosine[i], cosine[j] = cosine[j], cosine[i]
            tt[i], tt[j] = tt[j], tt[i]
            i += 1
            j -= 1
    if l < j:
        qs(cosine, l, j, tt)
    if i < r:
        qs(cosine, i, r, tt)


def check_file(file, list_file):
    file_not_commit = delete_commit(file)
    for i in list_file:
        if file_not_commit == delete_commit(i):
            return list_file.index(i)
    return -1


def get_matrix_and_label(ind_bug_left, ind_bug_right):
    # get data for label 1
    matrix_1 = []
    label = []
    for ind in range(ind_bug_left, ind_bug_right, 1):
        files = file_1[ind]
        files = files.replace("', '", "---")
        for i in files.split('---'):
            file_clean = i.replace("['", '')
            file_clean = file_clean.replace("']", '')

            if file_clean in name_source:
                ind_source = name_source.index(file_clean)
                matrix = concatenate(matrix_bug[ind], source_matrix[ind_source])
                matrix_1.append(matrix)
                label.append([1., 0.])

    # get data for label 0
    matrix_0 = []
    for ind in range(ind_bug_left, ind_bug_right, 1):
        files = file_0[ind]
        files = files.replace("', '", "---")
        count = -1
        order = []  # numerical order of sources for each bug
        value = []  # value of feature of source for each bug
        list_matrix = []

        for i in files.split('---'):
            # clean string of name source full link
            file_clean = i.replace("['", '')
            file_clean = file_clean.replace("']", '')

            if file_clean in name_source and check_file(file_clean, source_id) != -1:
                ind_source = name_source.index(file_clean)
                matrix = concatenate(matrix_bug[ind], source_matrix[ind_source])
                list_matrix.append(matrix)
                count += 1
                order.append(count)

                ind_source = check_file(file_clean, source_id)
                value.append(features[ind][ind_source])

        # sort feature, select 300 minimize values of feature
        qs(value, 0, count, order)
        count_cosine = 0
        for i in range(len(value)):
            if count_cosine == 100:
                break

            count_cosine += 1
            matrix_0.append(list_matrix[order[i]])
            label.append([0., 1.])

    matrix_return = matrix_1 + matrix_0

    return matrix_return, label


def get_matrix_and_label_test(ind_bug_left, ind_bug_right):
    # get data for label 1
    matrix_1 = []
    label = []
    for ind in range(ind_bug_left, ind_bug_right, 1):
        files = file_1[ind]
        files = files.replace("', '", "---")
        for i in files.split('---'):
            file_clean = i.replace("['", '')
            file_clean = file_clean.replace("']", '')

            if file_clean in name_source:
                ind_source = name_source.index(file_clean)
                matrix = concatenate(matrix_bug[ind], source_matrix[ind_source])
                matrix_1.append(matrix)
                label.append([1., 0.])

    # get data for label 0
    matrix_0 = []
    for ind in range(ind_bug_left, ind_bug_right, 1):
        files = file_0[ind]
        files = files.replace("', '", "---")
        count = -1
        order = []  # numerical order of sources for each bug
        value = []  # value of feature of source for each bug
        list_matrix = []

        for i in files.split('---'):
            # clean string of name source full link
            file_clean = i.replace("['", '')
            file_clean = file_clean.replace("']", '')

            if file_clean in name_source and check_file(file_clean, source_id) != -1:
                ind_source = name_source.index(file_clean)
                matrix = concatenate(matrix_bug[ind], source_matrix[ind_source])
                list_matrix.append(matrix)
                count += 1
                order.append(count)

                ind_source = check_file(file_clean, source_id)
                value.append(features[ind][ind_source])

        # sort feature, select 300 minimize values of feature
        qs(value, 0, count, order)
        count_cosine = 0
        for i in range(len(value)):
            if count_cosine == 100:
                break

            count_cosine += 1
            matrix_0.append(list_matrix[order[i]])
            label.append([0., 1.])

    matrix_return = matrix_1 + matrix_0

    return matrix_return, label




