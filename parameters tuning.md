The precision and recall are tested by tester.py

## GaussianNB

Precision: 0.15701  Recall: 0.93800


## DecisionTreeClassifier
### max_features

clf = tree.DecisionTreeClassifier(max_features=1)
    Precision: 0.29740  Recall: 0.28000
clf = tree.DecisionTreeClassifier(max_features=2)
    Precision: 0.29799  Recall: 0.30350
clf = tree.DecisionTreeClassifier(max_features=3)
    Precision: 0.33221  Recall: 0.34350
clf = tree.DecisionTreeClassifier(max_features=4)
    Precision: 0.33955  Recall: 0.36400

### max_depth

clf = tree.DecisionTreeClassifier(max_features=4, max_depth=2)
    Precision: 0.04110  Recall: 0.00600
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=3)
    Precision: 0.39920  Recall: 0.24850
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=4)
    Precision: 0.36698  Recall: 0.27450
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=5)
    Precision: 0.35106  Recall: 0.29700
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=6)
    Precision: 0.37451  Recall: 0.33650
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=7)
    Precision: 0.35363  Recall: 0.34850
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=8)
    Precision: 0.34915  Recall: 0.36050
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=9)
    Precision: 0.33772  Recall: 0.35950
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=10)
    Precision: 0.33727  Recall: 0.35700
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=11)
    Precision: 0.33569   Recall: 0.35650
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=12)
    Precision: 0.34267  Recall: 0.36100
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=13)
    Precision: 0.34230  Recall: 0.36250
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=14)
    Precision: 0.34044  Recall: 0.36750
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=15)
    Precision: 0.33442  Recall: 0.35800
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=16)
    Precision: 0.33380  Recall: 0.35700
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=17)
    Precision: 0.33840  Recall: 0.35650
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=18)
    Precision: 0.33380  Recall: 0.35700
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=19)
    Precision: 0.33605  Recall: 0.35050
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=20)
    Precision: 0.34063  Recall: 0.36550
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=21)
    Precision: 0.33976  Recall: 0.36100
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=22)
    Precision: 0.33869  Recall: 0.35800
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=50)
    Precision: 0.33349  Recall: 0.35650

### min_sample_split

clf = tree.DecisionTreeClassifier(max_features=4, max_depth=6, min_samples_split=2)
    Precision: 0.37176  Recall: 0.33700
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=6, min_samples_split=3)
    Precision: 0.34836  Recall: 0.28600
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=6, min_samples_split=4)
    Precision: 0.34114  Recall: 0.28400
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=6, min_samples_split=5)
    Precision: 0.34313  Recall: 0.29200
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=6, min_samples_split=6)
    Precision: 0.32859  Recall: 0.30050
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=6, min_samples_split=7)
    Precision: 0.33090  Recall: 0.29450
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=6, min_samples_split=8)
    Precision: 0.34983  Recall: 0.30400
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=6, min_samples_split=9)
    Precision: 0.35024  Recall: 0.29700
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=6, min_samples_split=10)
    Precision: 0.35024  Recall: 0.29700
clf = tree.DecisionTreeClassifier(max_features=4, max_depth=6, min_samples_split=11)
    Precision: 0.36073  Recall: 0.29850

## LogisticRegression

clf = LogisticRegression()
lack of true positive predicitons

