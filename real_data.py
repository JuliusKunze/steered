from collections import defaultdict
from pathlib import Path

import numpy
import numpy as np
import pandas
from sklearn.datasets.base import Bunch
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

data_directory = Path('.') / 'data'


def dataframe(bunch: Bunch):
    X = bunch.data
    y = np.expand_dims(bunch.target, axis=1)
    return pandas.DataFrame(numpy.concatenate([y, X], axis=1), columns=['target'] + [str(i) for i in range(X.shape[1])])


def bunch(data: pandas.DataFrame):
    target = 'target'

    y = data[target]
    X = data.drop(target, axis=1)

    return Bunch(data=X, target=y)


def classify(data):
    print(f'classifying on features with shape {data.data.shape}')

    X = data.data
    y = data.target
    classifier = KNeighborsClassifier()  # MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000)
    return cross_val_score(classifier, X, y, cv=4, scoring='f1_macro')


def genes():
    genes_dir = data_directory / 'Genes'

    names = [s for s in (genes_dir / 'Full_File.names').read_text().splitlines() if s]
    print(names)
    print(len(names))

    data = pandas.read_csv(str(genes_dir / 'Full_File.data'), delimiter=',', header=None, names=names)

    data.select_dtypes(exclude=['float64'])

    print(data.columns.values)
    print(list(data.columns.values))
    print(len(list(data.columns.values)))
    print(data.shape)


def thrombin() -> pandas.DataFrame:
    types = defaultdict(default_factory=lambda: 'bool')
    types[0] = 'str'

    thrombin_dir = data_directory / 'Thrombin'
    pickle_file = thrombin_dir / 'thrombin.pickle'

    if pickle_file.exists():
        data = pandas.read_pickle(str(pickle_file))
        # else:
        # data = pandas.read_csv(str(thrombin_dir / 'thrombin.data'), delimiter=',', header=None, dtype='bool',
        #                       converters={0: lambda x: x == 'A'}) #, usecols=range(100))

    # data.to_pickle(str(pickle_file))


    return data

# classify(load_iris())
