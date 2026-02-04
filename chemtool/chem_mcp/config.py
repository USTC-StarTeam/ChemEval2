import os
from dotenv import load_dotenv
load_dotenv('.env', override=True)

class Config:
    def __init__(self):

        self.PUBCHEM_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov"
        self.RXN4CHEM_API_KEY = os.getenv("RXN4CHEM_API_KEY")
        self.RXN4CHEM_BASE_URL = "https://rxn.app.accelerate.science"
        self.RXN4CHEM_PROJECT_ID = os.getenv("RXN4CHEM_PROJECT_ID")


