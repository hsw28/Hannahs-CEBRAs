import cebra
from cebra import CEBRA
import numpy as np
import sys
import pandas as pd
import joblib as jl
from matplotlib.collections import LineCollection


#location = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/trainingLocation_trace.mat')
#cells = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/trainingCells_trace.mat')
#eyeblink = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/trainingEyeblink.mat')
#cells = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/trainingCells_trim.mat')
#eyeblink = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/trainingEyeblink_trim.mat')
#location = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/trainingLocation_trim.mat')
#eyeblink = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/trainingLocationEyeblink_trim.mat')



wanted21 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/wanted21.mat')
wanted22 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/wanted22.mat')
wanted24 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/wanted24.mat')
wanted25 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/wanted25.mat')
wanted21 = wanted21.flatten()
wanted22 = wanted22.flatten()
wanted24 = wanted24.flatten()
wanted25 = wanted25.flatten()

pos21 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/pos21.mat')
pos21 = pos21[:,1:]
pos21 = pos21[wanted21]

pos22 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/pos22.mat')
pos22 = pos22[:,1:]
pos22 = pos22[wanted22]

pos24 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/pos24.mat')
pos24 = pos24[:,1:]
pos24 = pos24[wanted24]

pos25 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/pos25.mat')
pos25 = pos25[:,1:]
pos25 = pos25[wanted25]

trace21 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/A21_trace_A21A22.mat')
trace21 = np.transpose(trace21)
trace21 = trace21[wanted21]

traceA21A22_22 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/A22_trace_A21A22.mat')
traceA21A22_22 = np.transpose(traceA21A22_22)
traceA21A22_22 = traceA21A22_22[wanted22]

traceA22B24_22 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/A22_trace_A22B24.mat')
traceA22B24_22 = np.transpose(traceA22B24_22)
traceA22B24_22 = traceA22B24_22[wanted22]

traceA22B25_22 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/A22_trace_A22B25.mat')
traceA22B25_22 = np.transpose(traceA22B25_22)
traceA22B25_22 = traceA22B25_22[wanted22]

trace22_all = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/A22_trace.mat')
trace22_all = np.transpose(trace22_all)
trace22_all = trace22_all[wanted22]

trace24 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/B24_trace_A22B24.mat')
trace24 = np.transpose(trace24)
trace24 = trace24[wanted24]

trace25 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/B25_trace_A22B25.mat')
trace25 = np.transpose(trace25)
trace25 = trace25[wanted25]

######training
trainingcells21 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/A21_trace_A21A22.mat')
trainingcells21 = np.transpose(trainingcells21)
trainingcells24 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/B24_trace_A22B24.mat')
trainingcells24 = np.transpose(trainingcells24)
trainingcells25 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/B25_trace_A22B25.mat')
trainingcells25 = np.transpose(trainingcells25)

trainingcells22 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/A22_trace.mat')
trainingcells22 = np.transpose(trainingcells22)
trainingcells22_21 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/A22_trace_A21A22.mat')
trainingcells22_21 = np.transpose(trainingcells22_21)
trainingcells22_24 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/A22_trace_A22B24.mat')
trainingcells22_24 = np.transpose(trainingcells22_24)
trainingcells22_25 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/A22_trace_A22B25.mat')
trainingcells22_25 = np.transpose(trainingcells22_25)


trainingtime21 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/training21_1-10.mat')
trainingtime21 = trainingtime21.flatten()
trainingcells21 = trainingcells21[trainingtime21 > 0,:]
trainingtime21 = trainingtime21[trainingtime21 > 0]
#trainingtime21[trainingtime21 <= 5] = 1
#trainingtime21[trainingtime21 > 5] = 2
trainingtime21[trainingtime21 <= 2] = 1
trainingtime21[(trainingtime21 > 2) & (trainingtime21 <= 4)] = 2
trainingtime21[(trainingtime21 > 4) & (trainingtime21 <= 6)] = 3
trainingtime21[(trainingtime21 > 6) & (trainingtime21 <= 8)] = 4
trainingtime21[trainingtime21 > 8] = 5

trainingtime22 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/training22_1-10.mat')
trainingtime22 = trainingtime22.flatten()
trainingcells22 = trainingcells22[trainingtime22 > 0,:]
trainingcells22_21 = trainingcells22_21[trainingtime22 > 0,:]
trainingcells22_24 = trainingcells22_24[trainingtime22 > 0,:]
trainingcells22_25 = trainingcells22_25[trainingtime22 > 0,:]

trainingtime22 = trainingtime22[trainingtime22 > 0]
#trainingtime22[trainingtime22 <= 5] = 1
#trainingtime22[trainingtime22 > 5] = 2
trainingtime22[trainingtime22 <= 2] = 1
trainingtime22[(trainingtime22 > 2) & (trainingtime22 <= 4)] = 2
trainingtime22[(trainingtime22 > 4) & (trainingtime22 <= 6)] = 3
trainingtime22[(trainingtime22 > 6) & (trainingtime22 <= 8)] = 4
trainingtime22[trainingtime22 > 8] = 5



trainingtime24 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/training24_1-10.mat')
trainingtime24 = trainingtime24.flatten()
trainingcells24 = trainingcells24[trainingtime24 > 0,:]
trainingtime24 = trainingtime24[trainingtime24 > 0]
#trainingtime24[trainingtime24 <= 5] = 1
#trainingtime24[trainingtime24 > 5] = 2
trainingtime24[trainingtime24 <= 2] = 1
trainingtime24[(trainingtime24 > 2) & (trainingtime24 <= 4)] = 2
trainingtime24[(trainingtime24 > 4) & (trainingtime24 <= 6)] = 3
trainingtime24[(trainingtime24 > 6) & (trainingtime24 <= 8)] = 4
trainingtime24[trainingtime24 > 8] = 5

trainingtime25 = cebra.load_data('/Users/Hannah/Programming/data_eyeblink/rat314/trainingdata/training25_1-10.mat')
trainingtime25 = trainingtime25.flatten()
trainingcells25 = trainingcells25[trainingtime25 > 0,:]
trainingtime25 = trainingtime25[trainingtime25 > 0]
#trainingtime25[trainingtime25 <= 5] = 1
#trainingtime25[trainingtime25 > 5] = 2
trainingtime25[trainingtime25 <= 2] = 1
trainingtime25[(trainingtime25 > 2) & (trainingtime25 <= 4)] = 2
trainingtime25[(trainingtime25 > 4) & (trainingtime25 <= 6)] = 3
trainingtime25[(trainingtime25 > 6) & (trainingtime25 <= 8)] = 4
trainingtime25[trainingtime25 > 8] = 5


trainingtime22.shape
