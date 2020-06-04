# ************* Neural network training ************************
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import textwrap
from sklearn.ensemble import RandomForestClassifier
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Function that checks accuracy per class
# Input: List of predicted labels (List-of-Int), List of true labels (List-of-Int),
#        List of class names (sorted by int label 0 ... n)
# Output: Pandas dataframe with three rows and a column for each class
#         row1: total n of items of class x
#         row2: n of correctly predicted items of class x
#         row3: accuracy in % (row2/row1)

def acc_per_class(predY, trueY, encoding):
    classes = len(set(trueY))
    truePos = np.zeros(classes)
    totaltrue = np.zeros(classes)
    falsePos = np.zeros(classes)
    totalpred = np.zeros(classes)

    for pred_y, true_y in zip(predY, trueY):
        totaltrue[true_y] = totaltrue[true_y] + 1
        totalpred[pred_y] = totalpred[pred_y] + 1
        if (true_y == pred_y):
            truePos[true_y] = truePos[true_y] + 1
        else:
            falsePos[pred_y] = falsePos[pred_y]+1

    acc = [a/b for a,b in zip(list(truePos),list(totaltrue))]
    prec = [(b-a)/b for a,b in zip(list(falsePos),list(totalpred))]

    totalOveralll = sum(totaltrue)
    corpredOverall = sum(truePos)
    accOverall = sum(acc*totaltrue)/sum(totaltrue)
    precOverall = sum(prec*totaltrue)/sum(totaltrue)

    overall = [totalOveralll, corpredOverall, accOverall, precOverall]

    df = pd.DataFrame([totaltrue, truePos, acc, prec], columns=encoding,
                      index=['totalTrue', 'cor.pred', 'acc', 'prec'])

    df['Total'] = overall
    return(df, accOverall, precOverall)


# Function that checks accuracy per class for random forest
# Input: List of predicted labels (List-of-Char), List of true labels (List-of-Char)
# Output: Pandas dataframe with three rows and a column for each class
#         row1: total n of items of class x
#         row2: n of correctly predicted items of class x
#         row3: accuracy in % (row2/row1)

def acc_per_class_rf(predY, trueY):
    # to use acc_per_class function encode labels by onehot encoding
    encoding_true, encoded_true = onehot_encode(trueY)
    encoding_pred, encoded_pred = onehot_encode(predY)
    # convert labels and predicted labels to ints
    true_Y = [np.where(r == 1)[0][0] for r in encoded_true]
    pred_Y = [np.where(r == 1)[0][0] for r in encoded_pred]
    return acc_per_class(pred_Y, true_Y, encoding_true)


# Function that encodes a list of char/string labels using one-hot encoding
# Sorts alphabetically, then assigns integers, then does one-hot encoding of integers
# Input: List of labels (List-of-char or List-of-String)
# Output: List of classes (alphabetically sorted) (List-of-char or List-of-String),
#         one-hot-encoding of input labels (2D numpy array)
def onehot_encode(labels):
    labelTypes = list(set(labels))
    labelTypes.sort()
    letterTointDict = dict(zip(labelTypes,list(range(0,len(labelTypes)))))
    intLabels = [letterTointDict[i] for i in labels]
    onehotArray = np.zeros((len(labelTypes), len(labelTypes)))
    for i in letterTointDict.values(): onehotArray[i,i] = 1
    onehot_encoded = np.asarray([list(onehotArray[i,:]) for i in intLabels])
    onehot_encoded.astype(int)
    return(labelTypes,onehot_encoded)

# Function that encodes letters as integers
# after alphabetical sorting (0,1,2...)
# Input: List of labels (list-of-character or strings)
# Output: Dictionary with labeltypes as keys and integer encoding as values
def letterToIntDict(labels):
    labelTypes = list(set(labels))
    labelTypes.sort()
    letterTointDict = dict(zip(labelTypes, list(range(0, len(labelTypes)))))
    return(letterTointDict)

# Function that encodes labels as Q3, Q7 or Q8
# Input: labels (list of chars)
# Output: encoded labels (list of chars)
def toQXLabels(labels,q):
    labelEncoding={}
    if(q==3):
        labelEncoding = {'G': 'H',
                        'H': 'H',
                        'I': 'H',
                        'T': 'C',
                        'E': 'E',
                        'B': 'C',
                        'S': 'C',
                        'L': 'C',
                        '-': 'C',
                        'EA': 'E',
                        'EP': 'E',
                        'EU': 'E',
                        'C': 'C',
                        'b': 'C'}
    elif(q==8):
        labelEncoding = {'G': 'G',
                         'H': 'H',
                         'I': 'I',
                         'T': 'T',
                         'E': 'E',
                         'B': 'B',
                         'S': 'S',
                         'L': 'C',
                         '-': 'C',
                         'EA': 'E',
                         'EP': 'E',
                         'EU': 'E',
                         'C': 'C',
                         'b': 'B'}
    elif(q==7):
        labelEncoding = {'G': 'G',
                         'H': 'H',
                         'I': 'I',
                         'T': 'C',
                         'E': 'E',
                         'B': 'C',
                         'S': 'C',
                         'L': 'C',
                         '-': 'C',
                         'EA': 'A',
                         'EP': 'P',
                         'EU': 'E',
                         'C': 'C',
                         'b': 'B'}

    return [labelEncoding[label] for label in labels]

# Model for Neural Network
def baseline_model(featureLen):
    model = Sequential()
    model.add(Dense(100, input_dim=featureLen, activation='relu'))
    model.add(Dense(100, input_dim=featureLen, activation='relu'))
    model.add(Dense(outputDim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Function that decodes features
# Input: features as vector position (list of ints)
# Output: decoded features (list of chars)
def feature_decoding(features):
    featureDecoding =  {0: 'total # H-bonds',
                        1: '# H-bonds in 310 helix pattern',
                        2: '# H-bonds in alpha helix pattern',
                        3: '# H-bonds in pi helix pattern',
                        4: '# H-bonds in sheet pattern',
                        5: 'direction of H-bond in sheet pattern',
                        6: '# H-bonds in bend pattern',
                        7: 'ASA',
                        8: 'Phi',
                        9: 'Psi',
                        10: 'Propensity alpa helix',
                        11: 'Propensity sheet',
                        12: 'Propensity turn',
                        13: 'Isoelectric Point',
                        14: 'AA',
                        15: 'AA',
                        16: 'AA',
                        17: 'AA',
                        18: 'AA',
                        19: 'AA',
                        20: 'AA',
                        21: 'AA',
                        22: 'AA',
                        23: 'AA',
                        24: 'AA',
                        25: 'AA',
                        26: 'AA',
                        27: 'AA',
                        28: 'AA',
                        29: 'AA',
                        30: 'AA',
                        31: 'AA',
                        32: 'AA',
                        33: 'AA'}
    return [featureDecoding[feature] for feature in features]


def writefilesForSOV(predMethod, q, path, pred_Y_labels, true_Y_labels):
    ofile = open(path + predMethod+'_pred_y_Q' + str(q) + '.txt', "w")
    ofile.write(">" + predMethod+'_pred_y_Q' + str(q) + "\n" + textwrap.fill(''.join(pred_Y_labels), 80))
    ofile.close

    ofile = open(path + predMethod + '_true_y_Q' + str(q) + '.txt', "w")
    ofile.write(">" + predMethod + '_true_y_Q' + str(q) + "\n" + textwrap.fill(''.join(true_Y_labels), 80))
    ofile.close
    return 0


#########################################################################
# *** PROGRAM STARTS HERE ***
#########################################################################

path = 'out/'
#path = '/Users/marathomas/Documents/Bioinformatik/StructureSystemsBioinformatics/test/out/XY'

# load data from feature extraction

allFeatures = np.load('allFeatures.npy')
allLabels = np.load('allLabels.npy')

# reduce for faster computation
#allFeatures = allFeatures[0:5000,:]
#allLabels = allLabels[0:5000,:]

# prepare labels
pdbLabels = list(allLabels[:,0])
dsspLabels = allLabels[:,1]
strideLabels = allLabels[:,2]
labelsQXDict = dict(zip([3, 7, 8],[toQXLabels(pdbLabels,3), toQXLabels(pdbLabels,7),toQXLabels(pdbLabels,8)]))

# Set Q3, Q7 or Q8
q = 8
print("Q"+str(q)+" labels")

# Set features and labels
features = allFeatures
labels = labelsQXDict[q]

# split 20% of the dataset for SOV testing
cutoff = int(0.2*features.shape[0])
x_SOV = features[0:cutoff,:]
y_SOV = labels[0:cutoff]

# remaining 80% -> use for training
features = features[cutoff:,]
labels = labels[cutoff:]

# encode labels with onehot_encoding
encoding,encoded_labels = onehot_encode(labels)
int2label = dict(zip(list(range(0, len(encoding))), encoding))

# Define x,y
x = features
y = encoded_labels

# ************* Neural Network ************************


print('Neural Network')
featureLen = x.shape[1]
outputDim = y.shape[1]

# Train model and run 4-fold cross validation

splitNum = 0
skf = StratifiedKFold(n_splits=4,shuffle=True)

all_pred_Y = []
all_true_Y = []

for train_index, test_index in skf.split(x, labels):

    # train model
    splitNum = splitNum+1
    print("Training model, cross val split " + str(splitNum) + "...")
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = baseline_model(featureLen)

    # get predicted labels
    model.fit(x_train, y_train, epochs=10, batch_size=5, verbose=False)
    print("Testing model...")
    pred_Y = model.predict_classes(x_test)
    true_Y = [np.where(r == 1)[0][0] for r in y_test]
    pred_Y = list(pred_Y)

    # Cross val metrics
    metrics, acc, prec = acc_per_class(pred_Y, true_Y, encoding)
    metrics.to_csv((path+'NN_split'+str(splitNum)+'_Q'+str(q)+'.csv'), sep='\t')
    print("Acc: %.2f" % acc)

    # Save for metrics
    all_pred_Y = all_pred_Y + pred_Y
    all_true_Y = all_true_Y + true_Y


# Metrics
metrics, acc, prec = acc_per_class(all_pred_Y, all_true_Y, encoding)
metrics.to_csv((path+'NN_Q'+str(q)+'.csv'), sep='\t')
print("Overall accuracy: %.2f"%acc)

# Files for SOV score
y_SOV_pred = model.predict_classes(x_SOV)
pred_Y_labels = [int2label[i] for i in y_SOV_pred]
true_Y_labels = y_SOV
writefilesForSOV('NN', q, path, pred_Y_labels, true_Y_labels)


# ************* Random Forest ************************


print('Random Forest')

x = np.asarray(x)
y = np.asarray(labels)

splitNum = 0
skf = StratifiedKFold(n_splits=4,shuffle=True)

all_rf_predictions = []
all_rf_labels = []

for train_index, test_index in skf.split(x, y):
    splitNum = splitNum + 1
    # need to use letters here, as skf.split and random forest cannot handle one-hot encoding
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # random forest model needs pandas dataframes as input
    x_train = pd.DataFrame(x_train)
    y_train = np.ravel(pd.DataFrame(y_train))
    x_test = pd.DataFrame(x_test)
    y_test = np.ravel(pd.DataFrame(y_test))

    # build random forest model
    model = RandomForestClassifier(n_estimators=100,
                                  bootstrap=True,
                                  max_features='auto')

    print("Training model, cross val split "+str(splitNum)+"...")
    model.fit(x_train, y_train)

    # get predicted labels
    print("Testing model...")
    rf_predictions = model.predict(x_test)

    # Cross-Val metrics
    metrics_rf, acc_rf, prec_rf = acc_per_class_rf(rf_predictions, list(y_test))
    metrics_rf.to_csv((path + 'RF_split' + str(splitNum) + '_Q' + str(q) + '.csv'), sep='\t')
    print("Acc: %.2f" % acc_rf)

    # Extract feature importances
    fi = pd.DataFrame({'feature': list(feature_decoding(x_train.columns)),
                   'importance': model.feature_importances_}).\
                    sort_values('importance', ascending = False)
    print('Five most important features:')
    print(fi.head(5))

    # save for metrics
    all_rf_predictions = all_rf_predictions+list(rf_predictions)
    all_rf_labels = all_rf_labels+list(y_test)

# Metrics
metrics_rf, acc_rf, prec_rf = acc_per_class_rf(all_rf_predictions, all_rf_labels)
metrics_rf.to_csv((path+'RF_Q'+str(q)+'.csv'), sep='\t')
print("Overall accuracy: %.2f"%acc_rf)

# Files for SOV score
pred_Y_labels = model.predict(pd.DataFrame(x_SOV))
true_Y_labels = y_SOV
writefilesForSOV('RF', q, path, pred_Y_labels, true_Y_labels)


# ************* DSSP and STRIDE ************************


true_Y_labels = pdbLabels

if(not (q==7)): # Q7 doesn't make sense, because DSSP and STRIDE can only do Q3 and Q8
    for predMethod in ['STR', 'DSSP']:
        if(predMethod=='STR'):
            pred_Y_labels = list(strideLabels)
        elif(predMethod=='DSSP'):
            pred_Y_labels = list(dsspLabels)

        pred_Y = toQXLabels(pred_Y_labels,q)
        true_Y = toQXLabels(true_Y_labels,q)
        letter2Int = letterToIntDict(true_Y+pred_Y)
        true_Y_int = [letter2Int[i] for i in true_Y]
        pred_Y_int = [letter2Int[i] for i in pred_Y]
        # Metrics
        metrics, acc, prec = acc_per_class(pred_Y_int, true_Y_int, letter2Int.keys())
        print("Accuracy: %.2f"%acc)
        metrics.to_csv((path + predMethod+'_Q' + str(q) + '.csv'), sep='\t')
        # Files for SOV
        pred_Y = pred_Y[0:cutoff]
        true_Y = true_Y[0:cutoff]
        writefilesForSOV(predMethod, q, path, pred_Y, true_Y)

