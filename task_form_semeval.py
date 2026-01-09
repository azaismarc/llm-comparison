import luigi
import xml.etree.ElementTree as ET
import csv
import os

class TaskFormSemEval(luigi.Task):
    input_dir = luigi.Parameter(default="data/semeval")
    output_file = luigi.Parameter(default="data/semeval/semeval.tsv")

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        # Mapping XML categories to French topic columns
        category_map = {
            "LOCATION": "EMPLACEMENT",
            "SERVICE": "SERVICE",
            "FOOD": "NOURRITURE",
            "AMBIENCE": "AMBIENCE",
            "PRICES": "RAPPORT QUALITE-PRIX"
        }

        # Collect all XML files
        xml_files = [os.path.join(self.input_dir, f) for f in os.listdir(self.input_dir) if f.endswith(".xml")]

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)

        # Open merged TSV for writing
        with self.output().open("w") as f:
            writer = csv.writer(f, delimiter="\t")
            
            # Write header
            header = ["ID_Review", "Sentences", "EMPLACEMENT", "SERVICE", "NOURRITURE", "AMBIENCE", "RAPPORT QUALITE-PRIX"]
            writer.writerow(header)

            # Process each XML file
            for xml_file in xml_files:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Iterate over reviews
                for review in root.findall(".//Review"):
                    review_id = review.get("rid")
                    merged_text = []
                    topics = {col: 0 for col in ["EMPLACEMENT", "SERVICE", "NOURRITURE", "AMBIENCE", "RAPPORT QUALITE-PRIX"]}

                    # Iterate over sentences in review
                    for sentence in review.findall(".//sentence"):
                        text = sentence.find("text").text or ""
                        merged_text.append(text.strip())

                        opinions = sentence.find("Opinions")
                        if opinions is not None:
                            for opinion in opinions.findall("Opinion"):
                                category, aspect = opinion.get("category").split("#")
                                if category == 'DRINKS': category = 'FOOD'
                                if category in category_map:
                                    topics[category_map[category]] = 1  # set to 1 if any sentence mentions it
                                if aspect in category_map:
                                    topics[category_map[aspect]] = 1

                    # Combine all sentences
                    full_text = " ".join(merged_text)

                    # Write row
                    row = [review_id, full_text] + [topics[col] for col in header[2:]]
                    writer.writerow(row)



if __name__ == "__main__":
   luigi.build([TaskFormSemEval()], local_scheduler=True)
