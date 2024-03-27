import csv
import jsonlines
from tqdm import tqdm


def run(json_file:str, csv_file:str):
    with jsonlines.open(json_file) as reader, open(csv_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["text", "summary", "type_ids", "row_ids", "col_ids"])
        for sample in tqdm(reader):
            src = sample["subtable_metadata_str"]
            if "sentence_annotations" in sample:
                tgt = sample["sentence_annotations"][0]["final_sentence"]
            else:
                tgt=' '
            type_ids = sample["type_ids"]
            row_ids = sample["row_ids"]
            col_ids = sample["col_ids"]
            writer.writerow([src, tgt, type_ids, row_ids, col_ids])
