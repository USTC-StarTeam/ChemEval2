from fastapi import FastAPI, HTTPException, Query
from chem_mcp.mcp_interface import (
    image_to_smiles_safe,
    smiles_inchi_convert_safe,
    calc_basic_physchem_safe,
    calc_formula_and_mass_safe,
    extract_stereo_safe,
    get_functional_groups_safe,
    name_to_smiles_safe,
    name_to_cas_safe,
    predict_reaction_safe,
    predict_retrosynthetic_safe

)
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="MCP Chemical Tools API",
    version="1.0",
    description="Chemical molecule analysis tool interface, supporting stereochemistry extraction, functional group analysis, name-to-SMILES/CAS conversion, reaction prediction, etc."
)

class SMILESInput(BaseModel):
    smiles_list: List[str]
class NameInput(BaseModel):
    name_list: List[str]

@app.post("/image2smiles")
async def image_to_smiles(image_path: str = Query(..., description="Path to the image file")):
    result = image_to_smiles_safe(image_path)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/smiles_inchi_convert")
async def smiles_inchi_convert(
    smiles: str = Query(None, description="SMILES string"),
    inchi: str = Query(None, description="InChI string")
):
    result = smiles_inchi_convert_safe(smiles, inchi)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/basic_physchem")
async def calc_basic_physchem(
    smiles: str = Query(None, description="SMILES string"),
    inchi: str = Query(None, description="InChI string")
):
    result = calc_basic_physchem_safe(smiles, inchi)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/formula_and_mass")
async def calc_formula_and_mass(
    smiles: str = Query(None, description="SMILES string"),
    inchi: str = Query(None, description="InChI string")
):
    result = calc_formula_and_mass_safe(smiles, inchi)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/stereo")
async def get_stereo(smiles: str = Query(..., description="SMILES string of the molecule")):
    result = extract_stereo_safe(smiles)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/functional_groups")
async def get_functional_groups(smiles: str = Query(..., description="SMILES string of the molecule")):
    result = get_functional_groups_safe(smiles)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/name2smiles")
async def get_smiles(name: str = Query(..., description="Chemical name")):
    result = name_to_smiles_safe(name)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.post("/name2cas")
async def get_cas(name: str = Query(..., description="Chemical name")):
    result = name_to_cas_safe(name)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/predict_reaction")
async def predict_reaction_endpoint(
    reactants: str = Query(..., description="Reactants SMILES"),
    reagent: str = Query(None, description="Reagent SMILES (optional)")
):
    result = predict_reaction_safe(reactants, reagent)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/predict_retrosynthetic")
async def predict_retrosynthetic_endpoint(product: str = Query(..., description="Product SMILES string")):
    result = predict_retrosynthetic_safe(product)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

# uvicorn chem_mcp.app:app --host 0.0.0.0 --port 9797