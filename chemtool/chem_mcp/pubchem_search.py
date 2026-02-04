import requests
from rdkit import Chem
from rdkit import RDLogger
import logging
import time

RDLogger.DisableLog('rdApp.*')

RETRY_TIMES = 3
RETRY_DELAY = 3

def is_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    return None

def largest_mol(smiles):
    ss = [s for s in smiles.split(".") if is_smiles(s)]
    if not ss:
        return None
    ss.sort(key=lambda a: len(a))
    return ss[-1]

def _make_request_with_retry(url):
    for attempt in range(RETRY_TIMES):
        try:
            response = requests.get(url, timeout=10)
            # 识别PubChem 404（分子未找到）
            if response.status_code == 404 and "PUGREST.NotFound" in response.text:
                raise Exception("MOLECULE_NOT_FOUND")
            response.raise_for_status()
            return response
        except Exception as e:
            if str(e) == "MOLECULE_NOT_FOUND":
                raise
            elif isinstance(e, requests.exceptions.RequestException):
                if attempt < RETRY_TIMES - 1:
                    logging.warning(f"Request failed (Attempt {attempt+1}/{RETRY_TIMES}): {str(e)[:100]}")
                    time.sleep(RETRY_DELAY)
                else:
                    raise Exception("NETWORK_ERROR")
            else:
                raise
            
def query_name_to_smiles(molecule_name: str) -> dict | str:
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"
    url = base_url.format(molecule_name, "property/IsomericSMILES/JSON")
    
    try:
        response = _make_request_with_retry(url)
        data = response.json()
        smi = data["PropertyTable"]["Properties"][0]["SMILES"]
        canonical_smiles = Chem.CanonSmiles(largest_mol(smi))
        
        return {
            "success": True,
            "data": {
                "molecule_name": molecule_name,
                "smiles": canonical_smiles
            }
        }

    except Exception as e:
        error_map = {
            "MOLECULE_NOT_FOUND": {
                "error_msg": f"No molecule found with the matching name: {molecule_name}"
            },
            "NETWORK_ERROR": {
                "error_msg": f"Network request failed (retried {RETRY_TIMES} times), could not connect to PubChem"
            }
        }
        
        if str(e) in error_map:
            return {"error": error_map[str(e)]["error_msg"]}
    
def query_name_to_cas(molecule_name: str) -> dict:
    try:
        mode = "name"
        if is_smiles(molecule_name):
            mode = "smiles"
        
        # 获取CID
        url_cid = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{mode}/{molecule_name}/cids/JSON"
        cid_response = _make_request_with_retry(url_cid)
        cid_data = cid_response.json()
        cid = cid_data["IdentifierList"]["CID"][0]

        # 获取详细数据
        url_data = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
        data_response = _make_request_with_retry(url_data)
        data = data_response.json()

        # 提取CAS号
        cas_number = None
        for section in data.get("Record", {}).get("Section", []):
            if section.get("TOCHeading") == "Names and Identifiers":
                for subsection in section.get("Section", []):
                    if subsection.get("TOCHeading") == "Other Identifiers":
                        for subsubsection in subsection.get("Section", []):
                            if subsubsection.get("TOCHeading") == "CAS":
                                cas_number = subsubsection["Information"][0]["Value"]["StringWithMarkup"][0]["String"]
                                break
                    if cas_number:
                        break
                if cas_number:
                    break

        if cas_number:
            return {
                "success": True,
                "data": {
                    "molecule_name": molecule_name,
                    "cas_number": cas_number,
                    "cid": cid
                }
            }
        else:
            return {
                "success": False,
                "error_type": "CAS_NOT_FOUND",
                "error_msg": f"No CAS number found for molecule: {molecule_name}"
            }

    except Exception as e:
        error_map = {
            "MOLECULE_NOT_FOUND": {
                "error_type": "MOLECULE_NOT_FOUND",
                "error_msg": f"No molecule found with the matching name: {molecule_name}"
            },
            "NETWORK_ERROR": {
                "error_type": "NETWORK_ERROR",
                "error_msg": f"Network request failed (retried {RETRY_TIMES} times), could not connect to PubChem"
            }
        }

        if str(e) in error_map:
            return {
                "success": False,
                **error_map[str(e)]
            }
        elif isinstance(e, KeyError):
            return {
                "success": False,
                "error_type": "DATA_FORMAT_ERROR",
                "error_msg": f"Molecular data format error (missing field: {str(e)})"
            }
        else:
            return {
                "success": False,
                "error_type": "UNKNOWN_ERROR",
                "error_msg": f"Processing failed: {str(e)}"
            }

# -----------------------------
# Demo
# -----------------------------
# molecules = ["ethanol", "acetone", "glucose", "aspirin", "caffeine", "(2-(2-chlorophenyl)ethylidene)triphenyl-l5-phosphane", "Acetylsalicylic acid"]

# for mol in molecules:
#     print("="*50)
#     print(f"Testing molecule: {mol}\n")
    
#     smiles_result = query_name_to_smiles(mol)
#     print("SMILES Result:\n", smiles_result)
    
#     cas_result = query_name_to_cas(mol)
#     print("CAS Result:\n", cas_result)
    
#     print("="*50 + "\n")
