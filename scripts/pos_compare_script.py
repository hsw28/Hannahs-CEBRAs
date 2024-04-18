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
# python pos_compare_script traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, posAn, posA1, posB1

#traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, posAn, posA1, posB1

parser = argparse.ArgumentParser(description="Run decoding with CEBRA.")
parser.add_argument("traceA1An_An", type=str)
parser.add_argument("traceAnB1_An", type=str)
parser.add_argument("traceA1An_A1", type=str)
parser.add_argument("traceAnB1_B1", type=str)
parser.add_argument("posAn", type=str)
parser.add_argument("posA1", type=str)
parser.add_argument("posB1", type=str)

# Parse arguments
args = parser.parse_args()

traceA1An_An = cebra.load_data(file=args.traceA1An_An)  # Adjust 'your_key_here' as necessary
traceAnB1_An = cebra.load_data(file=args.traceAnB1_An)  # Adjust 'your_key_here' as necessary
traceA1An_A1 = cebra.load_data(file=args.traceA1An_A1)  # Adjust 'your_key_here' as necessary
traceAnB1_B1 = cebra.load_data(file=args.traceAnB1_B1)  # Adjust 'your_key_here' as necessary
posAn = cebra.load_data(file=args.posAn)  # Adjust 'your_key_here' as necessary
posA1 = cebra.load_data(file=args.posA1)  # Adjust 'your_key_here' as necessary
posB1 = cebra.load_data(file=args.posB1)  # Adjust 'your_key_here' as necessary

traceA1An_An = np.transpose(traceA1An_An)
traceAnB1_An = np.transpose(traceAnB1_An)
traceA1An_A1 = np.transpose(traceA1An_A1)
traceAnB1_B1 = np.transpose(traceAnB1_B1)

posA1 = posA1[:,1:]
#get every other point and check length
posA1 = posA1[::2]
if len(posA1) > len(traceA1An_A1):
    posA1 = posA1[:len(traceA1An_A1)]


posAn = posAn[:,1:]
posAn = posAn[::2]
if len(posAn) > len(traceA1An_An):
    posAn = posAn[:len(traceA1An_An)]

posB1 = posB1[:,1:]
posB1 = posB1[::2]
if len(posB1) > len(traceAnB1_B1):
    posB1 = posB1[:len(traceAnB1_B1)]




pos_compare(traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, posAn, posA1, posB1)
