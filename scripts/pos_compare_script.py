import sys
import argparse
import cebra
from cebra import CEBRA
import cebra.helper as cebra_helper
import numpy as np
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra')
from pos_compare import pos_compare

#for making the shuffle position figure
# python pos_compare_script traceA21A22_22, traceA22B24_22, traceA21A22_21, traceA22B24_24, pos22, pos21, pos24



parser = argparse.ArgumentParser(description="Run decoding with CEBRA.")
parser.add_argument("traceA21A22_22", type=str)
parser.add_argument("traceA22B24_22", type=str)
parser.add_argument("traceA21A22_21", type=str)
parser.add_argument("traceA22B24_24", type=str)
parser.add_argument("pos22", type=str)
parser.add_argument("pos21", type=str)
parser.add_argument("pos24", type=str)

# Parse arguments
args = parser.parse_args()

traceA21A22_22 = cebra.load_data(file=args.traceA21A22_22)  # Adjust 'your_key_here' as necessary
traceA22B24_22 = cebra.load_data(file=args.traceA22B24_22)  # Adjust 'your_key_here' as necessary
traceA21A22_21 = cebra.load_data(file=args.traceA21A22_21)  # Adjust 'your_key_here' as necessary
traceA22B24_24 = cebra.load_data(file=args.traceA22B24_24)  # Adjust 'your_key_here' as necessary
pos22 = cebra.load_data(file=args.pos22)  # Adjust 'your_key_here' as necessary
pos21 = cebra.load_data(file=args.pos21)  # Adjust 'your_key_here' as necessary
pos24 = cebra.load_data(file=args.pos24)  # Adjust 'your_key_here' as necessary


pos21 = pos21[:,1:]
pos22 = pos22[:,1:]
pos24 = pos24[:,1:]

traceA21A22_22 = np.transpose(traceA21A22_22)
traceA22B24_22 = np.transpose(traceA22B24_22)
traceA21A22_21 = np.transpose(traceA21A22_21)
traceA22B24_24 = np.transpose(traceA22B24_24)


pos_compare(traceA21A22_22, traceA22B24_22, traceA21A22_21, traceA22B24_24, pos22, pos21, pos24)
