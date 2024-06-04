import sys
sys.path.append('/Users/Hannah/Programming/Hannahs-CEBRAs/')
sys.path.append('/Users/Hannah/anaconda3/envs/CEBRA/lib/python3.8/site-packages/cebra')
import argparse
import cebra
from cebra import CEBRA
import cebra.helper as cebra_helper
import numpy as np
from cond_compare5 import cond_compare5

#for making the shuffle Position figure
#can optionally input parameters or hard code them
#not inputed:
#inputed:
#python /Users/Hannah/Programming/Hannahs-CEBRAs/scripts/cond_compare_script.py ./traceA1An_An.mat ./traceAnB1_An.mat ./traceA1An_A1.mat ./traceAnB1_B1.mat ./eyeblinkAn.mat ./eyeblinkA1.mat ./eyeblinkB1.mat 2 0 --learning_rate 0.0035 --min_temperature 1.67 --max_iterations 20 --distance cosine

parser = argparse.ArgumentParser(description="Run decoding with CEBRA.")
parser.add_argument("traceA1An_An", type=str, help="Path to the traceA1An_An data file.")
parser.add_argument("traceAnB1_An", type=str, help="Path to the traceAnB1_An data file.")
parser.add_argument("traceA1An_A1", type=str, help="Path to the traceA1An_A1 data file.")
parser.add_argument("traceAnB1_B1", type=str, help="Path to the traceAnB1_B1 data file.")
parser.add_argument("CSUSAn", type=str, help="Path to the CSUSAn data file.")
parser.add_argument("CSUSA1", type=str, help="Path to the CSUSA1 data file.")
parser.add_argument("CSUSB1", type=str, help="Path to the CSUSB1 data file.")
parser.add_argument("how_many_divisions", type=int, help="Number of divisions for categorizing data.")
parser.add_argument("pretrial_y_or_n", type=int, choices=[0, 1], help="Pretrial flag (0 or 1).")
parser.add_argument("--learning_rate", type=float, default=0.000775)
parser.add_argument("--min_temperature", type=float, default=0.001)
parser.add_argument("--max_iterations", type=int, default=6000)
parser.add_argument("--distance", type=str, default='cosine')


# Parse arguments
args = parser.parse_args()

traceA1An_An = cebra.load_data(file=args.traceA1An_An)
traceAnB1_An = cebra.load_data(file=args.traceAnB1_An)
traceA1An_A1 = cebra.load_data(file=args.traceA1An_A1)
traceAnB1_B1 = cebra.load_data(file=args.traceAnB1_B1)
CSUSAn = cebra.load_data(file=args.CSUSAn)
CSUSA1 = cebra.load_data(file=args.CSUSA1)
CSUSB1 = cebra.load_data(file=args.CSUSB1)

traceA1An_An = np.transpose(traceA1An_An)
traceAnB1_An = np.transpose(traceAnB1_An)
traceA1An_A1 = np.transpose(traceA1An_A1)
traceAnB1_B1 = np.transpose(traceAnB1_B1)


CSUSAn = CSUSAn[0, :].flatten()
CSUSA1 = CSUSA1[0, :].flatten()
CSUSB1 = CSUSB1[0, :].flatten()

# Logic to divide data based on 'divisions' and 'pretrial'
if args.pretrial_y_or_n == 0:
    traceA1An_An = traceA1An_An[CSUSAn > 0]
    traceAnB1_An = traceAnB1_An[CSUSAn > 0]
    CSUSAn = CSUSAn[CSUSAn > 0]

    traceA1An_A1 = traceA1An_A1[CSUSA1 > 0]
    CSUSA1 = CSUSA1[CSUSA1 > 0]

    traceAnB1_B1 = traceAnB1_B1[CSUSB1 > 0]
    CSUSB1 = CSUSB1[CSUSB1 > 0]
else:
    traceA1An_An = traceA1An_An[CSUSAn != 0]
    traceAnB1_An = traceAnB1_An[CSUSAn != 0]
    CSUSAn = CSUSAn[CSUSAn != 0]

    traceA1An_A1 = traceA1An_A1[CSUSA1 != 0]
    CSUSA1 = CSUSA1[CSUSA1 != 0]

    traceAnB1_B1 = traceAnB1_B1[CSUSB1 != 0]
    CSUSB1 = CSUSB1[CSUSB1 != 0]

how_many_divisions = args.how_many_divisions

if how_many_divisions == 2:
    CSUSAn[(CSUSAn > 0) & (CSUSAn <= 6)] = 1
    CSUSAn[CSUSAn > 6] = 2
    CSUSAn[CSUSAn == -1] = 0

    CSUSA1[(CSUSA1 > 0) & (CSUSA1 <= 6)] = 1
    CSUSA1[CSUSA1 > 6] = 2
    CSUSA1[CSUSA1 == -1] = 0

    CSUSB1[(CSUSB1 > 0) & (CSUSB1 <= 6)] = 1
    CSUSB1[CSUSB1 > 6] = 2
    CSUSB1[CSUSB1 == -1] = 0

elif how_many_divisions == 5:
    CSUSAn[(CSUSAn > 0) & (CSUSAn <= 2)] = 1
    CSUSAn[(CSUSAn > 2) & (CSUSAn <= 4)] = 2
    CSUSAn[(CSUSAn > 4) & (CSUSAn <= 6)] = 3
    CSUSAn[(CSUSAn > 6) & (CSUSAn <= 8)] = 4
    CSUSAn[CSUSAn > 8] = 5
    CSUSAn[CSUSAn == -1] = 0

    CSUSA1[(CSUSA1 > 0) & (CSUSA1 <= 2)] = 1
    CSUSA1[(CSUSA1 > 2) & (CSUSA1 <= 4)] = 2
    CSUSA1[(CSUSA1 > 4) & (CSUSA1 <= 6)] = 3
    CSUSA1[(CSUSA1 > 6) & (CSUSA1 <= 8)] = 4
    CSUSA1[CSUSA1 > 8] = 5
    CSUSA1[CSUSA1 == -1] = 0

    CSUSB1[(CSUSB1 > 0) & (CSUSB1 <= 2)] = 1
    CSUSB1[(CSUSB1 > 2) & (CSUSB1 <= 4)] = 2
    CSUSB1[(CSUSB1 > 4) & (CSUSB1 <= 6)] = 3
    CSUSB1[(CSUSB1 > 6) & (CSUSB1 <= 8)] = 4
    CSUSB1[CSUSB1 > 8] = 5
    CSUSB1[CSUSB1 == -1] = 0

dimensions = how_many_divisions + args.pretrial_y_or_n

cond_compare5(traceA1An_An, traceAnB1_An, traceA1An_A1, traceAnB1_B1, CSUSAn, CSUSA1, CSUSB1, dimensions, learning_rate=args.learning_rate, min_temperature=args.min_temperature, max_iterations=args.max_iterations, distance=args.distance)
