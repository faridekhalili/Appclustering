import pickle


def pickle_save(my_model, file_name):
    pickle.dump(my_model, open(file_name, 'wb'))


def pickle_load(file_name):
    loaded_obj = pickle.load(open(file_name, 'rb'))
    return loaded_obj


def write_to_file(message):
    f = open('./output/topic_modeling/timings.txt', 'a')
    f.write(message)
