from .rdkit_tools import smiles_inchi_convert, calc_basic_physchem, calc_formula_and_mass
from .rdchiral_tools import extract_stereochemistry
from .efgs_tools import extract_functional_groups
from .pubchem_search import query_name_to_smiles, query_name_to_cas
from .rxn_predict import predict_reaction, predict_retrosynthetic
from .convert_image_smiles import image_to_smiles

def image_to_smiles_safe(image_path: str) -> dict:
    try:
        return image_to_smiles(image_path)
    except Exception as e:
        return {"error": str(e)}

def smiles_inchi_convert_safe(smiles: str, inchi: str) -> dict:
    try:
        return smiles_inchi_convert(smiles, inchi)
    except Exception as e:
        return {"error": str(e)}
    
def calc_basic_physchem_safe(smiles: str = None, inchi: str = None) -> dict:
    try:
        return calc_basic_physchem(smiles, inchi)
    except Exception as e:
        return {"error": str(e)}
    
def calc_formula_and_mass_safe(smiles: str = None, inchi: str = None) -> dict:
    try:
        return calc_formula_and_mass(smiles, inchi)
    except Exception as e:
        return {"error": str(e)}
    
def extract_stereo_safe(smiles: str) -> dict:
    try:
        return extract_stereochemistry(smiles)
    except Exception as e:
        return {"error": str(e)}

def get_functional_groups_safe(smiles: str):
    try:
        return extract_functional_groups(smiles)
    except Exception as e:
        return {"error": str(e)}

def name_to_smiles_safe(name: str):
    try:
        return query_name_to_smiles(name)
    except Exception as e:
        return {"error": str(e)}

def name_to_cas_safe(name: str):
    try:
        return query_name_to_cas(name)
    except Exception as e:
        return {"error": str(e)}
    
def predict_reaction_safe(reactants_smiles: str, reagent_smiles: str | None = None):
    try:
        return predict_reaction(reactants_smiles, reagent_smiles)
    except Exception as e:
        return {"error": str(e)}


def predict_retrosynthetic_safe(product_smiles: str):
    try:
        return predict_retrosynthetic(product_smiles)
    except Exception as e:
        return {"error": str(e)}

