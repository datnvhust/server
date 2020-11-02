import get_file_name
import build_connect
import model_cnn


if __name__ == '__main__':
    # # get the list name source not include commit, for any bug report
    # list_name = get_file_name.get_list_name_not_commit()

    # # create data, which show connection between 1 bug and all source (not include commit)
    # build_connect.build_data()

    # build, run and test model CNN
    model_cnn.model_run()