"""
Prétraitement d'images avant OCR / extraction de tableau.

Corrige deux problèmes fréquents sur des photos de terrain (plutôt que des scans) :
- contraste faible / fond coloré -> amélioration de contraste adaptative (CLAHE)
- léger angle de prise de vue -> redressement automatique (deskew)

Ne corrige PAS la courbure de page (perspective non plane) : pour ça, il faut
recadrer/aplatir manuellement la photo, ou reprendre la photo bien à plat.

Installation : uv add opencv-python numpy
"""

import cv2
import numpy as np


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Améliore le contraste localement, utile sur fond coloré ou peu contrasté."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Redresse une légère inclinaison (quelques degrés), pas une perspective complexe."""
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def preprocess(input_path: str, output_path: str) -> None:
    image = cv2.imread(input_path)
    contrasted = enhance_contrast(image)
    straightened = deskew(contrasted)

    # Seuillage adaptatif : aide beaucoup sur les fonds colorés / dégradés
    final = cv2.adaptiveThreshold(
        straightened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=25, C=15
    )

    cv2.imwrite(output_path, final)
    print(f"Image prétraitée sauvegardée : {output_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage : python preprocess_image.py entree.jpg sortie.jpg")
        sys.exit(1)
    preprocess(sys.argv[1], sys.argv[2])