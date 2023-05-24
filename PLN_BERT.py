# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:21:14 2023

@author: joan camilo tamayo

pip install transformers
pip install torch

"""

# from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pandas as pd

def normalize(s): # esta función elimina los acentos en una cadena de caracteres
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"), 
        ("ú", "u"),
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s

def calcular_similitud_BERT(texto1, texto2):
    # Carga el modelo pre-entrenado BERT y el tokenizador ======================
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased')
    
    # Carga el modelo pre-entrenado BioBERT y el tokenizador -==================
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    model = AutoModel.from_pretrained('dmis-lab/biobert-v1.1')
    
  

    # Tokeniza los textos y agrega los tokens especiales [CLS] y [SEP]
    tokens_texto1 = tokenizer.encode(texto1, add_special_tokens=True)
    tokens_texto2 = tokenizer.encode(texto2, add_special_tokens=True)

    # Convierte los tokens en tensores de PyTorch
    tensor_texto1 = torch.tensor([tokens_texto1])
    tensor_texto2 = torch.tensor([tokens_texto2])

    # Calcula las representaciones de los textos utilizando BERT
    with torch.no_grad():
        outputs_texto1 = model(tensor_texto1)
        outputs_texto2 = model(tensor_texto2)

    # Obtén las representaciones de la capa "pooler"
    representacion_texto1 = outputs_texto1.pooler_output.numpy()[0]
    representacion_texto2 = outputs_texto2.pooler_output.numpy()[0]

    # Calcula la similitud coseno entre las representaciones de los textos
    similitud = np.dot(representacion_texto1, representacion_texto2) / (np.linalg.norm(representacion_texto1) * np.linalg.norm(representacion_texto2))

    # Convierte la similitud en un porcentaje
    porcentaje_similitud = similitud * 100

    return porcentaje_similitud

# ==================  Ejemplo de uso basico ===============================
texto1 = "tumor maligno de la piel del miembro superior, incluido el hombro"
texto2 = "ca escamocelular en mano derecha pop reseccion en 2 d"

similitud = calcular_similitud_BERT(texto1, texto2)
print("Porcentaje de similitud (BERT):", similitud)

# ejemplo buscando texto2 recorriendo toda la bdd de cie10 ===================
cod_cie_10_cancer = pd.read_excel(r"\Desktop\P_Imbanaco\Prestadores\BDD\Maestros\CIE_10_CANCER.xlsx")

for i in range(len(cod_cie_10_cancer)):
    desc = cod_cie_10_cancer["Descripción"].iloc[i]
    desc = normalize(desc.lower())
    
    similitud = calcular_similitud_BERT(desc, texto2)
    if similitud > 90: 
        print(desc)
        print(texto2)
        print("=========================")



