"""AA vocabulary, atom mappings, ideal geometry constants."""

AA_TO_IDX = {
    'ALA':0,'ARG':1,'ASN':2,'ASP':3,'CYS':4,'GLN':5,'GLU':6,'GLY':7,'HIS':8,
    'ILE':9,'LEU':10,'LYS':11,'MET':12,'PHE':13,'PRO':14,'SER':15,'THR':16,
    'TRP':17,'TYR':18,'VAL':19,'ORN':20,'DAL':21,'BAL':22,'NMA':23,'DPH':24,
}
IDX_TO_AA = {v:k for k,v in AA_TO_IDX.items()}
NUM_AA_TYPES = 25
MASK_IDX = 25  # masked discrete diffusion mask token

# chi counts: A R N D C Q E G H I L K M F P S T W Y V ORN DAL BAL NMA DPH
AA_CHI_COUNTS = [0,4,2,2,1,3,3,0,2,2,2,4,3,2,1,1,1,2,2,1,3,0,0,0,2]

BACKBONE_ATOMS = ['N','CA','C','O']

AA_ATOM_NAMES = {
    'ALA': ['N','CA','C','O','CB'],
    'ARG': ['N','CA','C','O','CB','CG','CD','NE','CZ','NH1','NH2'],
    'ASN': ['N','CA','C','O','CB','CG','OD1','ND2'],
    'ASP': ['N','CA','C','O','CB','CG','OD1','OD2'],
    'CYS': ['N','CA','C','O','CB','SG'],
    'GLN': ['N','CA','C','O','CB','CG','CD','OE1','NE2'],
    'GLU': ['N','CA','C','O','CB','CG','CD','OE1','OE2'],
    'GLY': ['N','CA','C','O'],
    'HIS': ['N','CA','C','O','CB','CG','ND1','CD2','CE1','NE2'],
    'ILE': ['N','CA','C','O','CB','CG1','CG2','CD1'],
    'LEU': ['N','CA','C','O','CB','CG','CD1','CD2'],
    'LYS': ['N','CA','C','O','CB','CG','CD','CE','NZ'],
    'MET': ['N','CA','C','O','CB','CG','SD','CE'],
    'PHE': ['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ'],
    'PRO': ['N','CA','C','O','CB','CG','CD'],
    'SER': ['N','CA','C','O','CB','OG'],
    'THR': ['N','CA','C','O','CB','OG1','CG2'],
    'TRP': ['N','CA','C','O','CB','CG','CD1','CD2','NE1','CE2','CE3','CZ2','CZ3','CH2'],
    'TYR': ['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ','OH'],
    'VAL': ['N','CA','C','O','CB','CG1','CG2'],
    'ORN': ['N','CA','C','O','CB','CG','CD','NE'],
    'DAL': ['N','CA','C','O','CB'],
    'BAL': ['N','CA','CB','C','O'],
    'NMA': ['N','CA','C','O','CB','NCH3'],
    'DPH': ['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ'],
}
AA_ATOM_COUNT = {aa:len(atoms) for aa,atoms in AA_ATOM_NAMES.items()}

IDEAL_BOND_LENGTHS = {
    'N-CA':1.458,'CA-C':1.525,'C-N':1.329,'C-O':1.231,'CA-CB':1.521,
}
IDEAL_BOND_ANGLES = {
    'N-CA-C':111.0,'CA-C-N':116.0,'C-N-CA':122.0,'N-CA-CB':110.5,'CA-C-O':120.8,
}

CYCLO_MODES = {'head_to_tail':0,'sidechain_to_tail':1,'sidechain_to_sidechain':2,'head_to_sidechain':3,'bicyclic':4}
NUM_CYCLO_MODES = 5

MAX_RESIDUES = 20
MAX_ATOMS_PER_RES = 14
MAX_CHI_PER_RES = 4
NUM_TORSIONS = 7  # phi,psi,omega,chi1-4
NUM_AA_TYPES_WITH_MASK = 26

IDEAL_PEPTIDE_BOND = 1.329
CLASH_THRESHOLD = 1.5
