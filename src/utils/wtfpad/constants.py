# Characters
CSV_SEP = ';'
TRACE_SEP = '\t'
NL = '\n'  # new line

# MPU
MTU = 1

# Directions
IN = -1
OUT = 1
DIR_NAMES = {IN: "in", OUT: "out"}
DIRECTIONS = [OUT, IN]

# AP states
GAP = 0x00
BURST = 0x01
WAIT = 0x02

# Mappings
EP2DIRS = {'client': OUT, 'server': IN}
MODE2STATE = {'gap': GAP, 'burst': BURST}

# Histograms
INF = float("inf")
