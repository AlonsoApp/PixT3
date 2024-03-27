import json
from tabulate import tabulate
import sys

from datasource.logic2text.utils import build_table_from_data_sample, build_tree

if __name__ == '__main__':
    """
    Call this function with the path of the Logic2Text dataset as an argument
    e.g. python3 dataset_explorer.py data/Logic2Text/original_data_fix/valid.json 
    """
    dataset_path = sys.argv[1]
    with open(dataset_path) as f:
        data_in = json.load(f)
    hashed_dataset = {}
    for sample in data_in:
        hashed_dataset[sample["example_id"]] = sample
    while True:
        query_example_id = input("Example_id: ")
        sample = hashed_dataset[query_example_id]
        pd_table = build_table_from_data_sample(sample)
        ast_tree = build_tree(sample)
        print("Topic: {}".format(sample["topic"]))
        print(tabulate(pd_table, headers='keys', tablefmt='psql'))
        ast_tree.print_graph()
        print(ast_tree.execute(pd_table))
        print(sample["sent"])


