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

def _make_request_with_retry(url, retry_times=RETRY_TIMES, delay=RETRY_DELAY):
    for attempt in range(retry_times):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if attempt < retry_times - 1:
                logging.warning(f"Request failed (Attempt {attempt + 1}/{retry_times}): {e}")
                logging.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error(f"All {retry_times} attempts failed: {e}")
                raise

def query_name_to_smiles(molecule_name: str) -> str:
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/{}"
    url = base_url.format(molecule_name, "property/IsomericSMILES/JSON")

    try:
        response = _make_request_with_retry(url)
        data = response.json()

        smi = data["PropertyTable"]["Properties"][0]["SMILES"]
        canonical_smiles = Chem.CanonSmiles(largest_mol(smi))

        result = {
            "molecule_name": molecule_name,
            "smiles": canonical_smiles
        }
        
        return result

    except requests.exceptions.RequestException as e:
        logging.error(f"Network error occurred: {e}")
        return "Network error occurred. Unable to process the request."

    except KeyError:
        return f"Could not find a molecule matCHing the name: {molecule_name}."

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return f"An error occurred while processing the request for: {molecule_name}."

def query_name_to_cas(molecule_name: str) -> str:
    try:
        mode = "name"
        if is_smiles(molecule_name):
            mode = "smiles"

        url_cid = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{mode}/{molecule_name}/cids/JSON"
        cid_response = requests.get(url_cid)
        cid_response.raise_for_status()
        cid = cid_response.json()["IdentifierList"]["CID"][0]

        url_data = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
        data_response = _make_request_with_retry(url_data)
        data = data_response.json()

        cas_number = None
        for section in data["Record"]["Section"]:
            if section.get("TOCHeading") == "Names and Identifiers":
                for subsection in section["Section"]:
                    if subsection.get("TOCHeading") == "Other Identifiers":
                        for subsubsection in subsection["Section"]:
                            if subsubsection.get("TOCHeading") == "CAS":
                                cas_number = subsubsection["Information"][0]["Value"]["StringWithMarkup"][0]["String"]
                                break

        if cas_number:
            results = {
                "molecule_name": molecule_name,
                "cas_number": cas_number,
                "cid": cid
            }
            return results
        else:
            return "CAS number not found."
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error occurred: {e}")
        return "Network error occurred. Unable to process the request."

    except (requests.exceptions.RequestException, KeyError):
        return "Invalid molecule input, no Pubchem entry."
    
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return f"An error occurred while processing the request for: {molecule_name}."
    

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
