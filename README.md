https://medium.com/@dr.shalinigambhir/how-i-built-a-question-answering-bot-in-python-step-by-step-guide-5463c2d15a55

générer HF_TOKEN pour acc les téléchargements


Le modèle
google/tapas-small-finetuned-sqa — basé sur une architecture BERT-small (peu de couches, dimensions réduites), donc nettement plus léger que les versions "base" ou "large" de TAPAS. C'est le plus petit modèle table-QA officiellement maintenu sur le Hub.
Il existe 2 variantes selon ton besoin :

-sqa (Sequential Question Answering) : pour des questions simples de sélection de cellule, y compris en mode conversationnel (tu peux enchaîner "quelle température le frigo 1 ?" puis "et le frigo 2 ?"). Pas d'agrégation (pas de somme, moyenne, etc.).
-wtq : si tu veux aussi poser des questions avec agrégation ("quelle est la température moyenne des 3 frigos ?").

Pour être sûr de ne jamais tenter un appel réseau
Par défaut, transformers fait quand même une petite requête de vérification à chaque lancement (pour voir si une version plus récente existe), ce qui peut ralentir ou planter si tu n'as vraiment aucun accès réseau. Pour forcer le mode strictement hors-ligne :

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from transformers import pipeline
pipe = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")

modèle stocké sous Windows : C:\Users\<toi>\.cache\huggingface\hub


requrired for this project: 
https://github.com/UB-Mannheim/tesseract/wiki
don't forget to add the French langage componant 


models saved in 
C:\Users\Elitebook\.paddlex\official_models