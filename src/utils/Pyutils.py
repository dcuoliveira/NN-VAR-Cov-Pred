import pickle

def save_pkl(data,
             path):

    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

def load_pkl(path):

    with open(path, 'rb') as handle:
        data = pickle.load(handle)
        handle.close()
    return data

DEBUG = False

if __name__ == '__main__':

    if DEBUG:
        import os

        data = {"a": 1, "b": 2}
        save_pkl(data=data,
                 path=os.path.join(os.getcwd(), "test.pickle"))

        new_data = load_pkl(path=os.path.join(os.getcwd(), "test.pickle"))