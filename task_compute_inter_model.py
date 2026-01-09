import luigi
import json
from glob import glob
import os
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
import pandas as pd
from itertools import combinations, product
from tqdm import tqdm
import krippendorff
from itertools import chain
import numpy as np


TOPICS = ['Chambre', 'Emplacement', 'Rapport qualité-prix', 'Ambiance', 'Personnel']

class TaskComputeInterModel(luigi.Task):

    def output(self):
        os.makedirs("data/score", exist_ok=True)
        return {
            "kappa": luigi.LocalTarget("data/score/mattcoef_models.tsv"),
            "kripp": luigi.LocalTarget("data/score/kripp_models.tsv"),
        }
        

    def run(self):

        input_data = dict()

        for file in glob("data/output_format/*/*.jsonl"):           
            if "batch" in file: continue

            model = file.split("/")[2]
            els = file.split("/")[3].split("_")
            v = els[0]
            annotator = '_'.join(els[1:-1])
            if "@" not in annotator: continue

            # Initialize nested dictionaries if needed
            input_data.setdefault(model, dict())
            input_data[model].setdefault(annotator, dict())
            input_data[model][annotator].setdefault(v, {t: [] for t in TOPICS})

            # Load data
            with open(file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]

            # Fill input_data
            for d in data:
                labels = set(el.get('label') for el in d.get('data', []) if el.get('label'))
                for t in TOPICS:
                    input_data[model][annotator][v][t].append(1 if t in labels else 0)

        rows_kappa = []
        rows_kripp = []
        models = list(input_data.keys())
        
        annotators = list(input_data[model].keys())
        print(annotators)

        for model_1, model_2 in tqdm(list(combinations(models, 2))):
            

            for v in ['sample', 'full']:
                row = {'Model_1': model_1, 'Model_2': model_2, 'V': v}
                for t in TOPICS:
                    score =  krippendorff.alpha([input_data[model_1][a][v][t] for a in annotators] + [input_data[model_2][a][v][t] for a in annotators],level_of_measurement='nominal')
                    row[t] = score
                rows_kripp.append(row)
                
                for annotator_1, annotator_2 in list(product(annotators, repeat=2)):
                    row = {'Model_1': model_1, 'Model_2': model_2, 'Annotator_1': annotator_1, 'Annotator_2': annotator_2, 'V': v}
                    for t in TOPICS:
                        # score = matthews_corrcoef(
                        #     input_data[model_1][annotator_1][v][t],
                        #     input_data[model_2][annotator_2][v][t]
                        # )
                        score = matthews_corrcoef(
                            input_data[model_1][annotator_1][v][t],
                            input_data[model_2][annotator_2][v][t]
                        )
                        row[t] = score
                    rows_kappa.append(row)
        
        df = pd.DataFrame(rows_kappa)
        df.to_csv(self.output()['kappa'].path, sep='\t', index=False)

        df = pd.DataFrame(rows_kripp)
        df.to_csv(self.output()['kripp'].path, sep='\t', index=False)


if __name__ == "__main__":
   luigi.build([TaskComputeInterModel()], local_scheduler=True)
