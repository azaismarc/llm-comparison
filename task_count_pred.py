import luigi
import json
from glob import glob
import os
import json

TOPICS = ['Chambre', 'Emplacement', 'Rapport qualité-prix', 'Ambiance', 'Personnel']

class TaskCountPred(luigi.Task):

    def output(self):
        os.makedirs("graph/count", exist_ok=True)
        return luigi.LocalTarget("raph/count/model.json")

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
            input_data[model][annotator].setdefault(v, {t: 0 for t in TOPICS})

            # Load data
            with open(file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]

            # Fill input_data
            for d in data:
                labels = set(el.get('label') for el in d.get('data', []) if el.get('label'))
                s = d['id'][0]
                hotel = 'Saint Nicolas' if int(d["id"][1:]) > 634 else "Rupella"
                for t in TOPICS:
                    if t in labels:
                        input_data[model][annotator][v][t] += 1

            with self.output().open('w') as f:
                json.dump(input_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
   luigi.build([TaskCountPred()], local_scheduler=True)
