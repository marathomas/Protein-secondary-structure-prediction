import os
import sys
import argparse
from Bio.PDB import *
import numpy as np
import warnings
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# If you do not want to run from command line adjust parameters here:
# change to directory "stride" containing stride-program
StridePath = '/Users/arianewiegand/Documents/Bioinformatik/Structure\ and\ Systems\ Bioinformatics/Project/stride/'
# change to directory containing pdb files
Path = 'data/'


# Function that encodes the amino acid names by one hot encoding
# possible aminoacid names: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
# Input: aminoacid names as 1LC amino acid alphabet
# Output: encoded labels as vector length 20 with 0 or 1 (list of lists)
def encodeAA(aas):
    aa20 = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(aa20)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    aaEncoding = dict(zip(aa20, onehot_encoded))
    return [aaEncoding[aa] for aa in aas]


# Function that checks input path
# Input: Path to PDB files (String)
# Output: List of pdb file paths (List of strings), List of pdb IDs (List of Strings)
def checkInput(path):
    if (os.path.exists(path)):
        files = os.listdir(path)
    else:
        print("Input directory ", path, " doesn't exist")
        sys.exit(1)
    IDs = []
    pdbFilePaths = []
    for file in files:
        splitted = file.split(".")
        if(len(splitted) > 1 and splitted[1] == "pdb"):
            IDs.append(splitted[0])
            pdbFilePaths.append(path+"/"+file)
    return(pdbFilePaths, IDs)


# Function that removes pdbFiles with >1 structure
# Input: list of pdb filepaths (list of strings)
# Output: list of pdb filepaths (list of strings)
def removeMultiFiles(pdbFilePaths, IDs):
    newFilePaths = []
    newIDs = []
    removed = []
    for i, pdbFile in enumerate(pdbFilePaths):
        modelNums = extractRowsByName(pdbFile, "NUMMDL")
        if (len(modelNums) > 0):
            removed.append(pdbFile[-8:])
        else:
            newFilePaths.append(pdbFile)
            newIDs.append(IDs[i])
    print("Removed ", len(removed), "files that contain >1 model.")
    return (newFilePaths, newIDs)


# Function that extracts protein rows from pdb helix and sheet rows
# (might be useless as they are anyways proteins)
# Input: list of pdb helix or sheet rows (list of strings)
# Output: ist of pdb protein helix or sheet rows (list of strings)
def extractProteinSS(ssRows):
    proteinRows = []
    resName = ""
    for ssRow in ssRows:
        if (ssRow[:5] == "HELIX"):
            resName = ssRow[15:18].strip()
        elif (ssRow[:5] == "SHEET"):
            resName = ssRow[17:20].strip()
        if (len(resName) > 2):
            proteinRows.append(ssRow)
    return (proteinRows)


# Function that extracts rows by initial characters
# Input: Path to file (string), initial characters (string)
# Output: List of rows in file that start with the input string
def extractRowsByName(file, name):
    result = []
    f = open(file, "r")
    for row in f:
        if (row[:len(name)] == name):
            result.append(row)
    f.close()
    return result

# Function that extracts chains,residues& type contained in a secondary structure
# row in a pdb file
# Input: row of pdb file (string), secondary structure row type ('HELIX' or 'SHEET')
# Output: chain (string), number of residues (list-of-int), ssType (integer (pdB file code))

def extractSSfromRow(row, type):
    chain = ''
    nums = []
    ssType = ''
    if(type=='HELIX'):
        chain = row[19]
        start_res = int(row[21:25])
        end_res = int(row[33:37])
        nums = list(range(start_res, end_res + 1))
        helixClass = int(row[38:40])
        ssType = pdbHelixClasses[helixClass]
    elif(type=='SHEET'):
        chain = row[21]
        start_res = int(row[23:26])
        end_res = int(row[34:37])
        nums = list(range(start_res, end_res + 1))
        if any(i.isdigit() for i in row[38:40]):
            sheetClass = int(row[38:40])
        else:
            sheetClass = 0
        ssType = pdbSheetClasses[sheetClass]
    return chain, nums, ssType

# Function that generates entries for a secondary structure dictionary from
# a list of secondary structure rows of a pdb file
# Input: List of pdb file rows (list-of-strings), secondary structure row type ('HELIX' or 'SHEET')
# Output: Dictionary (dict)
#         Example dict entry: { ('A', (' ', 5, ' '):'H' }
#         (format of key is adapted to DSSP output dictionary)
def makeSSdict(SSrows, ssType):
    resNums = {}
    for row in SSrows:
        chain, nums, ssLabel = extractSSfromRow(row, ssType)
        for num in nums:
            numEncode = ' ',num,' '
            newkey = chain, numEncode
            resNums[newkey]=ssLabel
    return (resNums)


# Function that generates a dictionary that contains secondary structure assignment
# of a given residue in a given chain. Info is extracted from pdbFile
# Input: pdb file path (String)
# Output: list of dictionaries (list-of-dict)
#         Example dict entry: { ('A', (' ', 5, ' '):'H' }
#         (format of key is adapted to DSSP output dictionary)

def getpdbSSdicts(pdbFile):
    fullDict = {}
    for ssType in ['HELIX', 'SHEET']:
        ssRows = extractRowsByName(pdbFile, ssType)
        ssRows = extractProteinSS(ssRows)
        ssDict = makeSSdict(ssRows,ssType)
        fullDict.update(ssDict)
    return(fullDict)



# Function that generates the basic hydrogen bond feature vectors for all residues
# of a given dssp_output of one pdb file, returns these as list-of-lists
# Input: values of dssp output dictionary (list-of-lists)
# Output: list of feature vectors (list-of-lists)
# each residue is then represented as vector of length 8
# Vector fields represent:
# [0] num of hydrogen bonds of res i (integer)
# [1] num of hydrogen bonds of res i to res i+3 or i-3 (310 helix)
# [2] num of hydrogen bonds of res i to res i+4 or i-4 (alpha helix)
# [3] num of hydrogen bonds of res i to res i+5 or i-5 (pi helix)
# [4] list of bond-partner distances
# [5] 0 for now
# [6] 0 for now
# [7] num of hydrogen bonds of res i to res i+2 or i-2 (bend)
def makeHydrogenFeatureVecBasic(dssp_out):
    hbonds = [out[6:14] for out in dssp_out]
    hbondFeatures = []

    for i in range(len(hbonds)):
        hbond = hbonds[i]
        bonds = [hbond[j:j + 2] for j in range(0, len(hbond), 2)]
        bondPartner = []
        # current feature vec representing: num of hydrogen bonds, 310 helix, alpha helix,
        # pi helix, non-helix bond partner distance, bend
        current = [0, 0, 0, 0, [], 0, 0, 0]

        for bond in bonds:
            if (bond[1] <= -0.5):
                current[0] = current[0] + 1
                if (bond[0] == 3 or bond[0] == -3):
                    current[1] = current[1] + 1
                elif (bond[0] == 4 or bond[0] == -4):
                    current[2] = current[2] + 1
                elif (bond[0] == 5 or bond[0] == -5):
                    current[3] = current[3] + 1
                elif (bond[0] == 2 or bond[0] == -2):
                    current[7] = current[7] + 1
                else:
                    bondPartner.append(bond[0])
        current[4] = bondPartner
        hbondFeatures.append(current)

    return(hbondFeatures)

# Function that updates the basic hydrogen bond feature vectors with
# information from the surrounding amino acids
# Input: list of feature vectors (list-of-lists)
# Output: list of feature vectors (list-of-lists)
# each residue is then represented as vector of length 6
# Vector fields represent:
# [0] num of hydrogen bonds of res i (integer)
# [1] num of hydrogen bonds of res i to res i+3 or i-3 (310 helix)
#     (+1 for each neighbor who has >0 in the same field)
# [2] num of hydrogen bonds of res i to res i+4 or i-4 (alpha helix)
#     (+1 for each neighbor who has >0 in the same field)
# [3] num of hydrogen bonds of res i to res i+5 or i-5 (pi helix)
#     (+1 for each neighbor who has >0 in the same field)
# [4] num of hydrogen bonds of sheet type
#     (+1 for each neighbor who has >0 in the same field)
# [5] num of hydrogen bonds of sheet type in antiparallel direction or parallel
#     (+1 for each neighbor in parallel, -1 for each neighbor in antiparallel direction)
# [6] num of hydrogen bonds of res i to res i+2 or i-2 (bend)
#     (+1 for each neighbor who has >0 in the same field)

def neighborHydrogenFeatureVec(hbondFeatures,d):

    newFeatureVec = []
    for i in range(len(hbondFeatures)):
        current = hbondFeatures[i]
        # integrate information from n previous/next residues
        for n in range(i-d,i+d,1):  # for all neighbours n
            if(n>=0 and n<len(hbondFeatures) and (not n==i)):  # if neighbour is valid
                neighbor = hbondFeatures[n]
                # add helix points
                for j in [1, 2, 3, 7]:
                    if (neighbor[j] > 0 and current[j] > 0):
                        current[j] = current[j] + 1
                # add sheet points
                if current[4] and neighbor[4]:
                    dist = -(i-n) if i>n else (n-i)  # dist negative when neighbor before i
                    if (any(x + dist in neighbor[4] for x in current[4])):
                        current[5] = current[5] + 1
                        current[6] = current[6] + 1
                    elif (any(x - dist in neighbor[4] for x in current[4])):
                        current[5] = current[5] + 1
                        current[6] = current[6] - 1
        # update feature vector
        newFeatureVec.append(current)

    # remove bond partner positions
    newFeatureVec = [x[:4] + x[(4 + 1):] for x in newFeatureVec]
    return(newFeatureVec)

# Function that extracts the torsion angle and acessibility values
# of a given dssp_output of a pdb file
# returns these as list-of-lists
# Input: values of dssp output dictionary (list-of-lists)
# Output: list of feature vectors (list-of-lists)
# each residue is then represented as vector of length 3
# Vector fields represent:
# [0] ASA
# [1] Phi
# [2] Psi
def makeDsspFeatures(dssp_out):
    return([out[3:6] for out in dssp_out])

# Function that extracts the secondary structure label
# of a given dssp_output of a pdb file
# returns these as list
# Input: values of dssp output dictionary (list-of-lists)
# Output: list of integers
def makeLabels(dssp_out):
    return([out[2] for out in dssp_out])


# Function that creates a dictionary containing the propensity values
# for each amino acid for helices, sheets and turns
# Input: path to txt-file containing propensity values (string),
# Output: dictionary (key: string, value: list-of-floats)
def getPropensities(path):
    f = open(path)
    readin = f.readlines()
    f.close()
    #remove header
    readin = readin[1:]

    propensities = {}
    for line in readin:
        propensities[line.split()[0]] = [float(line.split()[1]),float(line.split()[2]),float(line.split()[3])]
    return propensities


# Function that extracts the propensity feature vector
# of a given dssp_output of a pdb file
# returns these as list-of-lists
# Input: values of dssp output dictionary (list-of-lists),
# dictionary of amino acid propensities for helices , beta sheets and turns
# Output: list-of-lists
def makePropensityFeatures(dssp_out,propensities):
    aa = [out[1] for out in dssp_out]
    return [propensities[a] for a in aa]


def makePDBlabels(file, dssp_out):
    pdbSSdict = getpdbSSdicts(file)
    dssp_SSlabels = [out[2] for out in dssp_out]

    dsspSSdict = dict(zip(dssp_dict.keys(), dssp_SSlabels))
    labels = []
    for key in dsspSSdict.keys():
        if key in pdbSSdict.keys():
            labels.append(pdbSSdict[key])
        else:
            labels.append(dsspToTarget[dsspSSdict[key]])
    labels = correctSheetEnds(labels)
    return(labels)

# Function that corrects list of pdb file labels
# pdb file does not assign a sense to the first residue of a sheet
# it therefore remained as 'EF' (beta-first)
# I'll assign it to parallel or antiparallel based on following
# residue. If the following residue is not a beta sheet, I'll assign
# B for bridge.
# Input: list of pdb secondary structure labels (list-of-characters)
# Output: list of characters
def correctSheetEnds(labels):
    correctedLabels = []
    for i in range(0,len(labels)-1):
        if(labels[i] == 'EF'):
            if(labels[i+1]=='EA'):
                correctedLabels.append('EA')
            elif(labels[i+1]=='EP'):
                correctedLabels.append('EP')
            else:
                correctedLabels.append('E')
        else:
            correctedLabels.append(labels[i])
    correctedLabels.append(labels[len(labels)-1])
    return(correctedLabels)

# Function that extracts the residue name (one letter code)
# of a given dssp_output of a pdb file
# returns these as list
# Input: values of dssp output dictionary (list-of-lists)
# Output: list of characters
def get_aa(dssp_out):
    return([out[1] for out in dssp_out])

# Function that assigns the isoelectric point values to each amino acid found
# Input: aminoacid names as 1LC amino acid alphabet
# Output: value of isoelectric point for each amino acid (list-of-floats)
# source for values: https://www.sigmaaldrich.com/life-science/metabolomics/learning-center/amino-acid-reference-chart.html
def getIPs(aas):
    ip ={'A':6.00, 'C':5.07, 'D':2.77, 'E':3.22, 'F':5.48, 'G':5.97, 'H':7.59, 'I':6.02, 'K':9.74,
         'L':5.98, 'M':5.74, 'N':5.41, 'P':6.30, 'Q':5.65, 'R':10.76, 'S':5.68, 'T':5.60, 'V':5.96,
         'W':5.89, 'Y':5.66}
    return [ip[aa] for aa in aas]


# Dictionary translating helix classes from PDB file (integers 1-10)
# to letter code
pdbHelixClasses = {1:'H', 2: 'H', 3: 'I', 4: 'H', 5: 'G',
                   6: 'H', 7: 'H', 8: 'H', 9: 'H', 10: 'H'}
# Dictionary translating sheet sense class from PDB file (integers 1-10)
# to letter code. (Sense of strand with respect to previous strand in the sheet.
# 0 if first strand, 1 if parallel, -1 if antiparallel.)
pdbSheetClasses = {-1: 'EA', 1: 'EP', 0: 'EF'}

# Dictionary translating dssp secondary structure classes to
# a reduced letter code scheme
dsspToTarget = {'G': 'G',  # 3-10 helix
                'H': 'H',  # alpha helix
                'I': 'I',  # pi-helix
                'T': 'T',  # turn
                'E': 'EU', # beta strand -> Beta undefined
                'B': 'B',  # beta-bridge
                'S': 'S',  # bend
                '-': 'C'}  # None -> coid


# Function that calls stride for given pdb file and returns filename of txt output file
# Input: list of txt file rows (list of strings)
# Output: list of secondary structure assignments (list of strings)
def callStride(ID, stridePath):
    stride = 'stride '
    strideFile = 'strideOutput/' + ID + '.txt'
    pdbFile = 'data/' + ID + '.pdb'
    command = stridePath + stride + pdbFile + ' -f' + strideFile
    os.system(command)
    return strideFile


# Function that extracts secondary structures from row of txt file
# Input: list of txt file rows (list of strings)
# Output: list of secondary structure assignments (list of strings)
def extractSS_Stride(ssRows):
    return [row[24:25] for row in ssRows]


# Function that extracts the secondary structure label assigned by stride
# of a given pdb file
# returns these as list
# Input: filename (string)
# Output: list of strings
def makeStrideLabels(ID, stridePath):
    strideFile = callStride(ID, stridePath)
    rows = extractRowsByName(strideFile, 'ASG')
    return extractSS_Stride(rows)


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

#########################################################################
# *** PROGRAM STARTS HERE ***
#########################################################################

# parse command line arguments
parser = argparse.ArgumentParser(description='Build feature vector and labels for given pdb files')
parser.add_argument('-i', metavar="<PATH_TO_INPUT_FOLDER>", dest='path', type=str, help='path to input pdb files')
parser.add_argument('-s', metavar="<PATH_TO_STRIDE_FOLDER>", dest='stridePath', type=str, help='path to stride program')

args = parser.parse_args()

path = args.path
stridePath = args.stridePath

if path==None:
    path=Path
if stridePath==None:
    stridePath=StridePath


print("\nSecondary Structure Analysis\n")


# check input
if(path==os.getcwd()):
    print("[Input directory: '", path, "' (default)]")
else:
    print("[Input directory: '", path, "']")


pdbFilePaths, IDs = checkInput(path)
print("[Identified ",len(pdbFilePaths)," structure(s)]")

# reduce dataset just for testing
#pdbFilePaths = pdbFilePaths[0:10]
#IDs = IDs[0:10]

print("Testset: [Identified ",len(pdbFilePaths)," structure(s)]")
# Remove files containing multiple structures
pdbFilePaths, IDs = removeMultiFiles(pdbFilePaths, IDs)
p = PDBParser()

errorFiles = []
errorIDs = []

propensityPath = 'propensities.txt'
propensities = getPropensities(propensityPath)

featuresPerPDB = []
labelsPerPDB = []

# this or mkdssp = "/anaconda3/bin/mkdssp"
# mkdssp = "/usr/local/Cellar/dssp/3.0.0/bin/mkdssp"
mkdssp = "mkdssp"

for file, ID in zip(pdbFilePaths, IDs):

    print("Generating features for ",ID,"...")
    featureList = []

    try:
        # make DSSP
        structure = p.get_structure(ID, file)
        model = structure[0] # always choose the first model
        dssp = DSSP(model, file, dssp="mkdssp")
        dssp_dict = dssp.property_dict
        dssp_out = dssp_dict.values()

        # make hydrogen bond features
        hbondFeatures = makeHydrogenFeatureVecBasic(dssp_out)
        hbondFeatures = np.asarray(neighborHydrogenFeatureVec(hbondFeatures,1))
        # make dsspFeatures (phi, psi, ASA)
        dsspFeatures = np.asarray(makeDsspFeatures(dssp_out))
        # make propensity features
        propensityFeatures = np.asarray(makePropensityFeatures(dssp_out, propensities))
        # make isoelectric point feature
        aas = get_aa(dssp_out)
        ip = np.asarray(getIPs(aas))
        # one_hot encoding of aa
        aa_onehot = np.asarray(encodeAA(aas))
        # make labels
        pdbLabels = np.asarray(makePDBlabels(file,dssp_out))
        dsspLabels = np.asarray(makeLabels(dssp_out))
        strideLabels = np.asarray(makeStrideLabels(ID, stridePath))
        # put all together
        allFeatures = np.column_stack((hbondFeatures, dsspFeatures, propensityFeatures,ip, aa_onehot))
        allLabels = np.column_stack((pdbLabels, dsspLabels, strideLabels))
        featuresPerPDB.append(allFeatures)
        labelsPerPDB.append(allLabels)

    except Exception:
        errorFiles.append(file)
        errorIDs.append(ID)

for errorFile, errorID in zip(errorFiles, errorIDs):
    pdbFilePaths.remove(errorFile)
    IDs.remove(errorID)

print('Could not DSSP for', len(errorIDs), 'files:', errorIDs)

# save files
allFeatures = np.vstack(featuresPerPDB) # jede row ein residue, spalten sind einzelne features
allLabels = np.vstack(labelsPerPDB)  # pdbLabels in spalte 1, reine dssp labels in spalte 2

np.save('allFeatures.npy', allFeatures)
np.save('allLabels.npy', allLabels)


