from typing import List, Dict
from rdkit import Chem

FG_DEFINITIONS: Dict[str, str] = {
    "hydroxyl (alcohol)": "[OX2H]",
    "phenol": "c[OH]",
    "ether": "[OD2]([#6])[#6]",
    "carbonyl (general)": "[CX3]=O",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2][#6]",
    "carboxamide (amide)": "[NX3][CX3](=O)[#6]",
    "carboxylate_anion": "[CX3](=O)[O-]",
    "amine (primary)": "[NX3;H2;!$(NC=O)]",
    "amine (secondary)": "[NX3;H1;!$(NC=O)]",
    "amine (tertiary)": "[NX3;H0;!$(NC=O)]",
    "nitrile": "[CX2]#N",
    "nitro": "[NX3](=O)=O",
    "thiol (mercaptan)": "[SX2H]",
    "thioether": "[SD2]([#6])[#6]",
    "sulfoxide": "[SX3](=O)([#6])[#6]",
    "sulfone": "S(=O)(=O)",
    "chloro": "[Cl]",
    "bromo": "[Br]",
    "fluoro": "[F]",
    "iodo": "[I]",
    "alkene": "C=C",
    "alkyne": "C#C",
    "aromatic_ring (benzene-like)": "c1ccccc1",
    "thiophene": "c1cc(s)c1",
    "pyridine": "n1ccccc1",
    "sulfate (approx)": "OS(=O)(=O)O",
    "lactam/imide": "[C,c](=O)[nH0]",
    "purine_core": "n1cnc2nccc2c1",
    "N-methyl": "N(C)"
}

# precompiled
_COMPILED_PATTERNS = {name: Chem.MolFromSmarts(s) for name, s in FG_DEFINITIONS.items()}

def extract_functional_groups(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "Invalid reactants SMILES"}

    results = []

    for name, patt in _COMPILED_PATTERNS.items():
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            continue

        frags = set()
        for match in matches:
            frag_smiles = Chem.MolFragmentToSmiles(mol, atomsToUse=list(match), canonical=True)
            frags.add(frag_smiles)

        results.append({"name": name, "fragments": list(frags)})

    return {
        "smiles": smiles,
        "functional_groups": results
    }

# -----------------------------
# Demo
# -----------------------------
# if __name__ == "__main__":
#     SAMPLE_SMILES = [
#         "CCO",                     # ethanol -> hydroxyl
#         "CC(=O)O",                 # acetic acid -> carboxylic acid
#         "CC(=O)NC1=CC=C(O)C=C1",   # paracetamol -> amide, phenol, aromatic
#         "c1ccccc1O",               # phenol
#         "CC#N",                    # nitrile
#         "CCCl",                    # chloroalkane
#         "CCS",                     # thiol
#         "Cn1cnc2n(C)c(=O)n(C)c(=O)c12"  # caffeine (complex)
#     ]
#     for test in SAMPLE_SMILES:
#         print(f"\nSMILES: {test}")
#         fgs = extract_functional_groups(test)
#         print("Functional groups found:")
#         print(fgs)