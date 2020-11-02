# Build data which show connection between 1 bug and all source (not include commit)
import numpy as np
import csv
import pandas as pd
import pickle
import os

# define link fo data
BUG_FILES = 'data/AspectJ_bug_and_files.csv'
DATA_CONNECT = 'data/data_connect.csv'
SOURCE_NAME = 'data/Aspectj_name_full_link.pickle'
SOURCE_NAME_NOT_COMMIT = 'data/list_name_not_commit.pickle'

f = open(SOURCE_NAME, 'rb')
pickle_input = pickle.load(f)


def divide_link(links):
    output = []
    link = ''
    for i in range(len(links)):
        if links[i] == ' ' and links[i - 5:i] == ".java":
            output.append(link)
            link = ''
        else:
            link += links[i]
    output.append(link)
    return output


def sort(cosine, l, r, tt):  # sort cosine to select 300 min cosine
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
        sort(cosine, l, j, tt)
    if i < r:
        sort(cosine, i, r, tt)


def concatenate_commit(link, commit):
    path = os.path.normpath(link)
    token = path.split(os.sep)
    name = commit + ' ' + token[len(token) - 1]
    return link.replace(token[len(token) - 1], '') + name


def file_before_fix():  # return the list source commit first version
    data_bug = pd.read_csv(BUG_FILES)
    commits = data_bug['commit'].values
    time_commits = data_bug['commit_timestamp'].values
    sort(time_commits, 0, len(commits) - 1, commits)

    file = open(SOURCE_NAME, 'rb')
    names_commit = pickle.load(file)
    file.close()

    file = open(SOURCE_NAME_NOT_COMMIT, 'rb')
    names_not_commit = pickle.load(file)

    result_list = []
    result_list_not_commit = []
    for name_not_commit in names_not_commit:
        find = 0  # check find the first version
        for commit in commits:
            link_full = concatenate_commit(name_not_commit, commit)
            if link_full in names_commit:
                find = 1
                result_list.append(link_full)
                result_list_not_commit.append(name_not_commit)
                break
    return result_list, result_list_not_commit


def build_data():
    bug = pd.read_csv(BUG_FILES)
    bug_id = bug['bug_id'].values
    files = bug['files'].values
    commits = bug['commit'].values

    list_base_commit, list_base_not_commit = file_before_fix()

    with open(DATA_CONNECT, 'w') as csv_file:
        fieldnames = ['bug_id', 'source_label_1', 'source_label_0']  # Định dạng cột
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(bug_id)):
            check = np.zeros(len(list_base_commit))  # to split label 1 and 0

            file = files[index]
            file_split = divide_link(file)

            # list save name source label 1 and 0
            source_1 = []
            source_0 = []
            for source in file_split:
                if source in list_base_not_commit:
                    # get index of source, which is label 1 for bug
                    ind = list_base_not_commit.index(source)
                    source_1.append(list_base_commit[ind])
                    check[ind] = 1

                    # update file commit
                    new_name = concatenate_commit(source, commits[index])
                    list_base_commit[ind] = new_name

            # get the label 0
            for ind in range(len(check)):
                if check[ind] == 0:
                    source_0.append(list_base_commit[ind])

            writer.writerow(
                {'bug_id': bug_id[index], 'source_label_1': source_1, 'source_label_0': source_0})

