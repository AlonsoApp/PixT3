from pathlib import Path
import glob
import shutil
import os
import fix_dataset_str_ambiguity, fix_dataset_hop, update_grammar_v3, add_sha1, generate_text_files

def create_original_data_fix(last_process_path):
	original_data_fix_path = "./data/Logic2Text/original_data_fix/"
	Path(original_data_fix_path).mkdir(parents=True, exist_ok=True)
	# Remove any files within the folder
	files = glob.glob(original_data_fix_path+"*")
	for f in files:
		os.remove(f)
	# Copy files from last path to original_fix
	file_names = os.listdir(last_process_path)
	for file_name in file_names:
		shutil.copy(os.path.join(last_process_path, file_name), original_data_fix_path)
	return original_data_fix_path

def run():
	dataset_path = "./data/Logic2Text/original_data/"
	print("Generating text files...")
	dataset_path = generate_text_files.run(dataset_path)
	print("1. Add: Add example_id")
	dataset_path = add_sha1.run(dataset_path)
	print("2. Fix: APIs str disambiguation")
	dataset_path = fix_dataset_str_ambiguity.run(dataset_path)
	print("3. Fix: hop to hop_first")
	dataset_path = fix_dataset_hop.run(dataset_path)
	print("4. Fix: Update grammar to V3")
	dataset_path = update_grammar_v3.run(dataset_path)
	# print("5. Fix: Max tokens")
	# dataset_path = fix_max_tokens.run(dataset_path)
	print("6. Creating original_data_fix")
	dataset_path = create_original_data_fix(dataset_path)
	print("7. Generating text files...")
	generate_text_files.run(dataset_path)

if __name__ == '__main__':
	run()