import cv2
from molscribe import MolScribe


model_path = "./models/MolScribe/swin_base_char_aux_1m680k.pth"

def image_to_smiles(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = MolScribe(model_path)

    smiles = model.predict_image(image).get('smiles', '')

    results = {
        "image_path": image_path,
        "smiles": smiles
    }

    return results

# -----------------------------
# Demo
# -----------------------------
# image_path = "./test_data/US20090012063A1-20090108-C00079.jpeg"

# print(image_to_smiles(image_path))