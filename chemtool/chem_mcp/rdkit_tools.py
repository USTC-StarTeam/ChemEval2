from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors

def smiles_inchi_convert(
    smiles: str | None = None,
    inchi: str | None = None
):
    if (smiles is None and inchi is None) or (smiles and inchi):
        return {"error": "Provide exactly one of smiles or inchi"}

    if smiles is not None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}

        return {
            "SMILES": Chem.MolToSmiles(mol, canonical=True),
            "InChI": Chem.MolToInchi(mol),
            "InChIkey": Chem.MolToInchiKey(mol),
        }

    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return {"error": "Invalid InChI"}

    return {
        "InChI": Chem.MolToInchi(mol),
        "InChIkey": Chem.MolToInchiKey(mol),
        "SMILES": Chem.MolToSmiles(mol, canonical=True),
    }

def calc_basic_physchem(smiles: str = None, inchi: str = None):
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
    elif inchi:
        mol = Chem.MolFromInchi(inchi)
    else:
        return {"error": "Either smiles or inchi must be provided"}

    if mol is None:
        return {"error": "mol is None"}

    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "RingCount": Descriptors.RingCount(mol),
    }

def calc_formula_and_mass(smiles: str = None, inchi: str = None):
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
    elif inchi:
        mol = Chem.MolFromInchi(inchi)
    else:
        return {"error": "Either smiles or inchi must be provided"}

    if mol is None:
        return {"error": "mol is None"}

    return {
        "MolecularFormula": rdMolDescriptors.CalcMolFormula(mol),
        "ExactMass": rdMolDescriptors.CalcExactMolWt(mol),
    }



# -----------------------------
# Demo
# -----------------------------
# tests = [
#     {"smiles": "CCO"},
#     {"smiles": "c1ccccc1O"},
#     {"inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"},
#     {"smiles": "C1(CC"}
# ]

# for i, t in enumerate(tests, 1):
#     print(f"\n=== Test case {i} ===")
#     result = smiles_inchi_convert(**t)
#     for k, v in result.items():
#         print(f"{k}: {v}")

# print(calc_basic_physchem("c1ccccc1O"))
# print(calc_formula_and_mass("c1ccccc1O"))


# from rdkit import Chem

# def inchi_smiles_to_mol(
#     smiles: str = None,
#     inchi: str = None,
#     canonical: bool = True
# ):
#     if smiles:
#         mol = Chem.MolFromSmiles(smiles)
#     elif inchi:
#         mol = Chem.MolFromInchi(inchi)
#     else:
#         return {"error": "Either smiles or inchi must be provided"}

#     if mol is None:
#         return {"error": "Invalid molecular representation"}

#     result = {"mol": mol}

#     if canonical:
#         result["canonical_smiles"] = Chem.MolToSmiles(mol, canonical=True)

#     return result