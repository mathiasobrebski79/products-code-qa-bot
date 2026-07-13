# https://www.nutrient.io/blog/how-to-use-tesseract-ocr-in-python/

# from img2table.document import Image
# from img2table.ocr import TesseractOCR
# import cv2
# import pytesseract
# from PIL import Image

import os
from paddleocr import PPStructureV3
from products_code_qa_bot.settings import DATA_DIR

raw_image_path = os.path.join(
    DATA_DIR, 
    "src_pictures", 
    "temps_cuisson_porc.jpg"
    )


engine = PPStructureV3(
    lang="fr",
    use_seal_recognition=False,
    use_formula_recognition=False,
    use_chart_recognition=False,
    enable_mkldnn=False
    )

result = engine.predict(raw_image_path)

for res in result:
    res.print()                                  # affiche dans la console (debug)
    res.save_to_json(save_path="output")          # structure complète en JSON
    res.save_to_markdown(save_path="output") 

# image = Image.open(raw_image_path)

# extracted_text = pytesseract.image_to_string(
#     image, 
#     lang="fra", 
#     config="--psm 4"
#     )

# print(extracted_text)

# processed_image_path = os.path.join(
#     DATA_DIR, 
#     "src_pictures", 
#     "processed_image.jpg"
#     )

# # preprocessing
# img = cv2.imread(raw_image_path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imwrite(
#     processed_image_path, 
#     thresh
#     )


# ocr = TesseractOCR(lang="fra")

# doc = Image(processed_image_path)

# tables = doc.extract_tables(
#     ocr=ocr,
#     implicit_rows=True,      # aide à repérer les lignes même sans séparateur visible
#     borderless_tables=True,  # active la détection des tableaux SANS bordures
#     min_confidence=50
# )

# print("image read")

# for i, table in enumerate(tables):
#     print(f"--- Tableau {i} ---")
#     print(table.df)
#     # table.df.to_csv(f"tableau_{i}.csv", index=False)