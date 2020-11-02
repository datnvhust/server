import pickle
import numpy as np
import json
import pandas as pd
import get_file_name

# define name data
BUG_MATRIX = 'data/Bug_matrix_AspectJ.txt'
SOURCE_MATRIX = 'data/matrix_sourceAspectj.pickle'
SOURCE_ID = 'data/DataOfDat/preprocessed_src_id.json'
FEATURE = 'data/DataOfDat/features1_update.json'
BUG_CSV = 'data/AspectJ_csv.csv'
NAME_SOURCE = 'data/Aspectj_name_full_link.pickle'
CONNECT = 'data/data_connect.csv'
NAME_NOT_COMMIT = 'data/list_name_not_commit.pickle'

SENT_BUG = 114  # 36  # 110# 114#
SENT_SOURCE = 194  # 134  # 201# 194#

# prepare data
# feature cosine
with open(FEATURE, 'rb') as file:
    features = json.load(file)#['data']

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
id_bug = file['bug_id'].values
file_1 = file['source_label_1'].values
file_0 = file['source_label_0'].values

# matrix source
pickle_input = open(SOURCE_MATRIX, 'rb')
source_matrix = pickle.load(pickle_input)

# matrix_bug
pickle_input = open(BUG_MATRIX, 'rb')
matrix_bug = pickle.load(pickle_input)


def test(ind_bug_left, ind_bug_right, model):
    # get data for label 1
    matrix_1_bug = []
    matrix_1_source = []
    count_1_true = 0
    count_0_false = 0

    for ind in range(ind_bug_left, ind_bug_right, 1):
        files = file_1[ind]
        files = files.replace("', '", "---")
        for i in files.split('---'):
            file_clean = i.replace("['", '')
            file_clean = file_clean.replace("']", '')

            if file_clean in name_source:
                ind_source = name_source.index(file_clean)
                matrix_1_bug.append(matrix_bug[ind])
                matrix_1_source.append(source_matrix[ind_source])

    matrix_1_bug = np.reshape(matrix_1_bug, (-1, SENT_BUG, 300, 1))
    matrix_1_source = np.reshape(matrix_1_source, (-1, SENT_SOURCE, 300, 1))
    matrix_1_bug = np.array(matrix_1_bug)
    matrix_1_source = np.array(matrix_1_source)
    predict = model.predict([matrix_1_bug, matrix_1_source])

    for i in predict:
        if i[0] > 0.5:
            count_1_true += 1
        else:
            count_0_false += 1

    # get data for label 0
    count_1_false = 0
    count_0_true = 0

    for ind in range(ind_bug_left, ind_bug_right, 1):
        files = file_0[ind]
        files = files.replace("', '", "---")

        for i in files.split('---'):
            file_clean = i.replace("['", '')
            file_clean = file_clean.replace("']", '')

            if file_clean in name_source and file_clean in source_id:
                ind_source = name_source.index(file_clean)
                matrix_0_bug = matrix_bug[ind]
                matrix_0_source = source_matrix[ind_source]

                matrix_0_bug = np.reshape(matrix_0_bug, (-1, SENT_BUG, 300, 1))
                matrix_0_source = np.reshape(matrix_0_source, (-1, SENT_SOURCE, 300, 1))
                matrix_0_bug = np.array(matrix_0_bug)
                matrix_0_source = np.array(matrix_0_source)
                predict = model.predict([matrix_0_bug, matrix_0_source])
                if predict[0][0] < 0.5:
                    count_0_true += 1
                else:
                    count_1_false += 1

    # matrix_0_bug = np.reshape(matrix_0_bug, (-1, 114, 300, 1))
    # matrix_0_source = np.reshape(matrix_0_source, (-1, 194, 300, 1))
    # matrix_0_bug = np.array(matrix_0_bug)
    # matrix_0_source = np.array(matrix_0_source)
    # predict = model.predict([matrix_0_bug, matrix_0_source])
    #
    # for i in predict:
    #     if i[0] < 0.5:
    #         count_0_true += 1
    #     else:
    #         count_1_false += 1

    print("count label 1 true - false is", count_1_true, "--", count_1_false)
    print("count label 0 true - false is", count_0_true, "--", count_0_false)


# name source with matrix word embedding
pickle_input = open(NAME_NOT_COMMIT, 'rb')
name_source_not_commit = pickle.load(pickle_input)
pickle_input.close()

check = np.zeros(shape=(len(matrix_bug), len(name_source_not_commit)))

for indexBug in range(len(matrix_bug)):
    files = file_1[indexBug]
    files = files.replace("', '", "---")
    for i in files.split('---'):
        file_clean = i.replace("['", '')
        file_clean = file_clean.replace("']", '')
        file_clean = file_clean.replace("[]", '')
        if file_clean == '':
            break
        name = get_file_name.delete_commit(file_clean)
        if name in name_source_not_commit:
            ind = name_source_not_commit.index(name)
            check[indexBug][ind] = 1


def Acc(bug, model, k, startInd):
    result = 0
    for indBug in range(len(bug)):
        # get the file source for each matrix_bug
        list_file = []

        list_file_0 = file_0[indBug]
        list_file_0 = list_file_0.replace("', '", "---")
        for i in list_file_0.split('---'):
            clean = i.replace("['", '')
            clean = clean.replace("']", '')
            clean = clean.replace("[]", '')
            list_file.append(clean)

        list_file_1 = file_1[indBug]
        list_file_1 = list_file_1.replace("', '", "---")
        for i in list_file_1.split('---'):
            clean = i.replace("['", '')
            clean = clean.replace("']", '')
            clean = clean.replace("[]", '')
            list_file.append(clean)

        countSource = 0
        for indSource in range(len(list_file) - 1, -1, -1):
            mtBug = np.reshape(bug[indBug], (1, SENT_BUG, 300, 1))
            if list_file[indSource] != '':
                find_ind = name_source.index(list_file[indSource])
                mtSource = np.reshape(source_matrix[find_ind], (1, SENT_SOURCE, 300, 1))
                predict = model.predict([mtBug, mtSource])
                if predict[0][0] > 0.5:
                    countSource += 1
                    if countSource <= k:
                        if check[indBug + startInd][indSource] == 1:
                            result += 1
                            break
                    if countSource > k:
                        break

    return result / len(bug)


def MRR(bug, model, startInd):
    result = 0
    for indBug in range(len(bug)):
        print(indBug, end="---")
        countSource = 0
        find = 0

        # get the file source for each matrix_bug
        list_file = []

        list_file_0 = file_0[indBug]
        list_file_0 = list_file_0.replace("', '", "---")
        for i in list_file_0.split('---'):
            clean = i.replace("['", '')
            clean = clean.replace("']", '')
            clean = clean.replace("[]", '')
            list_file.append(clean)

        list_file_1 = file_1[indBug]
        list_file_1 = list_file_1.replace("', '", "---")
        for i in list_file_1.split('---'):
            clean = i.replace("['", '')
            clean = clean.replace("']", '')
            clean = clean.replace("[]", '')
            list_file.append(clean)

        for indSource in range(len(list_file) - 1, len(list_file) - 200, -1):
            mtBug = np.reshape(bug[indBug], (1, SENT_BUG, 300, 1))
            if list_file[indSource] != '':
                find_ind = name_source.index(list_file[indSource])
                mtSource = np.reshape(source_matrix[find_ind], (1, SENT_SOURCE, 300, 1))
                predict = model.predict([mtBug, mtSource])
                if predict[0][0] > 0.5:
                    countSource += 1
                    if check[indBug + startInd][indSource] == 1:
                        result += 1 / countSource

    return result / len(bug)


def metrics_evaluate(startInd, endInd, model_CNN):
    bug_test = matrix_bug[startInd:endInd]

    print("MRR = ", MRR(bug_test, model_CNN, startInd))
    print("Acc@1 = ", Acc(bug_test, model_CNN, 1, startInd))
    print("Acc@5 = ", Acc(bug_test, model_CNN, 5, startInd))
    print("Acc@10 = ", Acc(bug_test, model_CNN, 10, startInd))
