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


pour connaitre l'adresse IP du PC : ipconfig depuis poweshell

Installation :
    uv add fastapi "uvicorn[standard]"

Lancement :
    uvicorn products_code_qa_bot.main:app --host 0.0.0.0 --port 8000 --ssl-certfile=10.90.10.207+2.pem --ssl-keyfile=10.90.10.207+2-key.pem


Puis, depuis le téléphone (même WiFi) :
    http://<IP_DE_TON_PC>:8000

B8-08-CF-BB-04-81

Si tu veux fixer l'IP côté PC (plus stable)
Tu peux forcer une IP statique sur l'adaptateur WiFi de ton PC, dans la même plage que le hotspot de ton téléphone :

Active le partage de connexion, note la passerelle donnée par ipconfig (ex: 192.168.43.1)
Paramètres Windows → réseau WiFi connecté au hotspot → "Attribution IP" → Manuel
Choisis une IP dans la même plage, ex: 192.168.43.50, masque 255.255.255.0, passerelle 192.168.43.1


# Installation (Windows)
winget install -e --id FiloSottile.mkcert

# Génère un certificat valide pour ton IP locale
mkcert -install
mkcert 10.90.10.207 localhost 127.0.0.1
127.0.0.1 est l'adresse de loopback ("boucle locale") : elle désigne toujours "cette machine elle-même", peu importe le réseau auquel tu es connecté. localhost est simplement son nom, plus lisible. Que tu sois sur un hotspot, une box, ou pas connecté du tout, 127.0.0.1 pointe toujours vers ton propre PC.