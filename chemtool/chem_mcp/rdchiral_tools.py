from rdkit import Chem

def extract_stereochemistry(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "Invalid reactants SMILES"}
    
    chiral_atoms = []
    for atom in mol.GetAtoms():
        if atom.HasProp('_ChiralityPossible'):
            chiral_atoms.append({
                "idx": atom.GetIdx(),
                "symbol": atom.GetSymbol(),
                "chirality": atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
            })
        
    results = {
        "smiles": smiles,
        "chiral_centers": chiral_atoms
    }
    
    return results



