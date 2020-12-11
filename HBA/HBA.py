import numpy as np
import pandas as pd

dataset_symptom = pd.read_csv('symptom.csv', ';')
dataset_disease = pd.read_csv('disease.csv', ';')

# Преобразуем набор данных в частотную таблицу
P = dict(zip(dataset_disease['Болезнь'],
             dataset_disease['количество пациентов'] / dataset_disease['количество пациентов'][
                 len(dataset_disease['количество пациентов']) - 1]))


def bayesian_classifier(disease, symptom, our_symptoms):
    disease_probs = []
    all_p = disease['количество пациентов'].values[-1]
    for i in disease['количество пациентов'].values:
        if i == all_p: continue
        disease_probs.append(i / all_p)

    our_probs = [1] * (len(disease['Болезнь']) - 1)
    for i in range(len(disease['Болезнь']) - 1):
        our_probs[i] *= disease_probs[i]
        for j in range(len(symptom) - 1):
            if our_symptoms[j] == 1:
                our_probs[i] *= float(str(symptom.iloc[j][i + 1]).replace(',', '.'))

    max_indx = 0
    max_v = our_probs[max_indx]
    for i in range(1, len(our_probs)):
        if max_v < our_probs[i]:
            max_v = our_probs[i]
            max_indx = i
    return disease['Болезнь'][max_indx]


list_symptoms = [np.random.randint(0, 2) for i in range(len(dataset_symptom) - 1)]

result = bayesian_classifier(dataset_disease, dataset_symptom, list_symptoms)

print(result)
