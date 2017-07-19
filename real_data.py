from collections import defaultdict
from pathlib import Path

import pandas
from sklearn import datasets
from sklearn.datasets.base import Bunch, load_digits, load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

data_directory = Path('.') / 'data'


def classify(data=datasets.load_breast_cancer()):
    X = data.data
    y = data.target
    classifier = KNeighborsClassifier()
    print(cross_val_score(classifier, X, y, cv=4, scoring='f1_macro'))


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

    data.to_pickle(str(pickle_file))

    return data


def load_thrombin():
    data = thrombin()

    print(data.shape)

    target = 0
    data = data.iloc[:, range(1000)]

    y = data.iloc[:, target]
    X = data.drop(target, axis=1)

    print(y.shape)
    print(X.shape)

    return Bunch(data=X, target=y)


classify(load_iris())
