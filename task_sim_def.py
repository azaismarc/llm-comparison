import luigi
import pandas as pd
import json
import re
from itertools import combinations
from tqdm import tqdm
from utils import preprocess_text_for_word2vec
from reach import Reach
from sklearn.preprocessing import normalize

reach = Reach.load("data/cbow_v3.vec")

from sentence_transformers import SentenceTransformer

class TaskSimDef(luigi.Task):
    input_path = luigi.Parameter(default="data/form.tsv")
    output_path = luigi.Parameter(default="data/sim_def.tsv")

    def output(self):
        return luigi.LocalTarget(self.output_path)

    def run(self):
        # --- Load data ---
        
        df = pd.read_csv(self.input_path, sep="\t")

        # --- Assuming you already have a list of 'mails' from the DataFrame ---
        mails = df["Adresse e-mail"].tolist()

        # --- Define helper regex and parser ---
        pattern = r'([A-ZÉÈÀ][\w\s-]*?)\s*:\s*(.*?)(?=(?:[A-ZÉÈÀ][\w\s-]*?\s*:)|$)'


        def get_topics(text: str):
            matches = re.findall(pattern, text, flags=re.MULTILINE)
            d = {}
            for topic, desc in matches:
                topic = topic.strip().title().strip()
                desc = desc.strip().lower().strip()
                d[topic] = desc
            return d

        # --- Extract standard themes ---
        themes = [
            "Chambre",
            "Emplacement",
            "Ambiance",
            "Rapport qualité-prix",
            "Personnel"
        ]

        model = SentenceTransformer("intfloat/multilingual-e5-large", device="cpu")

        data_sbert = {m: {} for m in mails}
        data_w2v = {m: {} for m in mails}
        for theme in tqdm(themes):
            colname = f"Définir le thème '{theme}' :"
            if colname not in df.columns:
                continue
            values = df[colname].tolist()
            for m, v in zip(mails, values):
                data_sbert[m][theme] = model.encode("query: " + v.lower(), normalize_embeddings=True)
                tokens = preprocess_text_for_word2vec(v.lower())
                emb = reach.normalize(reach.mean_pool(tokens, remove_oov=True))
                data_w2v[m][theme] = emb

        
        all_mails = list(data_sbert.keys())
        rows = []
        for theme in themes:
            for m1, m2 in combinations(all_mails, 2):
                sbert_score = data_sbert[m1][theme] @ data_sbert[m2][theme]
                w2v_score = data_w2v[m1][theme] @ data_w2v[m2][theme]
                rows.append({
                    "theme": theme,
                    "a1": m1,
                    "a2": m2,
                    "sbert": sbert_score,
                    "w2v": w2v_score
                })

        # --- Save result as DataFrame ---
        df = pd.DataFrame(rows)
        df.to_csv(self.output().path, index=False, sep="\t")


if __name__ == "__main__":
    luigi.build([TaskSimDef()], local_scheduler=True)
