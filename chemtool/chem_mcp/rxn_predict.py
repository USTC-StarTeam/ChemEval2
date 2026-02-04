import time
from rdkit import Chem
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# from config import Config

# cfg = Config()

def is_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    return None

tokenizer = AutoTokenizer.from_pretrained("./models/ReactionT5v2-forward-USPTO_MIT", return_tensors="pt")
model = AutoModelForSeq2SeqLM.from_pretrained("./models/ReactionT5v2-forward-USPTO_MIT")

retro_tokenizer = AutoTokenizer.from_pretrained("./models/ReactionT5v2-retrosynthesis-USPTO_50k", return_tensors="pt")
retro_model = AutoModelForSeq2SeqLM.from_pretrained("./models/ReactionT5v2-retrosynthesis-USPTO_50k")
def predict_reaction(reactants: str, reagent: str | None = None) -> dict:
    # --- Check SMILES ---
    if not is_smiles(reactants):
        return {"error": "Invalid reactants SMILES"}

    if reagent is None or reagent.strip() == "":
        reagent = None

    # --- Construct input prompt ---
    if reagent is None:
        # No reagent
        prompt = f"REACTANT:{reactants}"
    else:
        prompt = f"REACTANT:{reactants}REAGENT:{reagent}"

    # --- Tokenize ---
    inp = tokenizer(prompt, return_tensors="pt")

    # --- Model inference ---
    output = model.generate(
        **inp,
        num_beams=1,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True
    )

    product = tokenizer.decode(
        output["sequences"][0],
        skip_special_tokens=True
    ).replace(" ", "").rstrip(".")

    return {
        "reactants": reactants,
        "reagent": reagent,
        "product": product
    }


def predict_retrosynthetic(product: str) -> dict:
    if not is_smiles(product):
        return {"error": "Invalid product SMILES"}

    prompt = f"PRODUCT:{product}"

    inp = retro_tokenizer(prompt, return_tensors="pt")

    output = retro_model.generate(
        **inp,
        num_beams=1,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True
    )

    reactants = retro_tokenizer.decode(
        output["sequences"][0], skip_special_tokens=True
    ).replace(" ", "").rstrip(".")

    return {
        "product": product,
        "reactants": reactants
    }

# -----------------------------
# Demo
# -----------------------------
# print(predict_reaction('COC(=O)C1=CCCN(C)C1.O.[Al+3].[H-].[Li+].[Na+].[OH-]', "C1CCOC1"))
# print(predict_reaction('CCNCCNC(=S)NC1CCCc2cc(C)cnc21.S=P12SP3(=S)SP(=S)(S1)SP(=S)(S2)S3'))
# print(predict_retrosynthetic("CCN(CC)CCNC(=S)NC1CCCc2cc(C)cnc21"))