# PixT3
This repository contains code for [PixT3: Pixel-based Table-To-Text Generation](https://openreview.net/forum?id=d2B35D0eqY).

We release PixT3 model checkpoints for the TControl, LControl, and OpenE settings as well as 
ToTTo, Controlled Logic2Text, and SLC pretraining datasets alongside their corresponding rendered tables
for each setting. This repository also contains the code to train and evaluate these models.

## Getting Started
Clone this GitHub repository, install the requirements, and download all [datasets](https://storage.googleapis.com/pixt3/data.tar.gz) and [models](https://storage.googleapis.com/pixt3/models.tar.gz). 
This project was developed using **Python=3.11**. 

```
git clone https://ANONYMOUS_URL/PixT3.git
cd PixT3
pip install -r requirements.txt
```

## Datasets
Download the ready-to-use datasets [here](https://storage.googleapis.com/pixt3/data.tar.gz).

## Model checkpoints
Download model checkpoints [here](https://storage.googleapis.com/pixt3/models.tar.gz).

These are the codenames for each model:
- **PixT3 (TControl):** `f4__20230918_201309`
- **PixT3 (LControl):** `i1__20230905_134725`
- **PixT3 (OpenE):** `i3__20230905_134649`
- **PixT3 (SLC):** `h1__20230904_120158` This is the model pretrained with the Structure Learning Curriculum. It mainly serves as initialization checkpoint for PixT3 (LControl) and PixT3 (OpenE).

## Training PixT3
We use [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/en/index) to run the training process. Although experiments should run equally fine without it, we 
recommend using it to replicate the training process as good as possible. To run the training without Accelerate replace 
`accelerate launch` with `python3`. We also recommend setting the root folder of the project as the `PYTHONPATH` variable.
```
export PYTHONPATH="$PWD/src"
```
### PixT3 (SLC) 
Pretrain Pix2Struct with our Structure Learning Curriculum. The resulting model serves as initialization checkpoint for PixT3 (LControl) and PixT3 (OpenE).
```
accelerate launch ./src/main_train.py --hf_model_name google/pix2struct-base --image_dir ./data/ToTTo/img/warmup_ssl1/ --dataset_variant warmup_ssl1 --exp_name h1 --lr 0.0001 --epochs 30 --batch_size 4 --gradient_accumulation_steps 64 --truncate_train_length True --max_text_length 300
mv ./out/experiments/h1* ./models/
```
### PixT3 TControl
We don't need the SLC pretrained model as the foundational model in TControl as this setting doesn't contain tables.
```
accelerate launch ./src/main_train.py --hf_model_name google/pix2struct-base --image_dir ./data/ToTTo/img/notab_high_00/ --exp_name f4 --lr 0.0001 --epochs 30
```
### PixT3 LControl
Use the previously pretrained PixT3(SLC) as the initialization model or use the one provided [here](https://storage.googleapis.com/pixt3/models.tar.gz) `h1__20230904_120158`. 
```
accelerate launch ./src/main_train.py --hf_model_name ./models/h1__20230904_120158/checkpoints/3/ --image_dir ./data/ToTTo/img/highlighted_039/ --exp_name i1 --lr 0.0001 --epochs 30
```
### PixT3 OpenE
Use the previously pretrained PixT3(SLC) as the initialization model or use the one provided [here](https://storage.googleapis.com/pixt3/models.tar.gz) `h1__20230904_120158`.
```
accelerate launch ./src/main_train.py --hf_model_name ./models/h1__20230904_120158/checkpoints/3/ --image_dir ./data/ToTTo/img/no_highlighted_039/ --exp_name i3 --lr 0.0001 --epochs 30
```

## Inference PixT3 for evaluation
This section describes how to generate the inferences with PixT3 models. We first recommend downloading the already 
trained  We recommend to set the root folder of the project as the `PYTHONPATH` variable.
```
export PYTHONPATH="$PWD/src"
```

### Flags
- `--model_to_load_path`: You can use one of the [already trained PixT3 checkpoints](https://storage.googleapis.com/pixt3/models.tar.gz) our you can set the path to any other trained one. 
- `--image_dir`: For ToTTo use `./data/ToTTo/img/SETTING_DIR/`. For Logic2Text `./data/ToTTo/img/SETTING_DIR/`.
- `--dataset_dir`: For ToTTo use `./data/ToTTo/`. For Logic2Text `/data/Logic2Text`.
- `--mode`: For dev set use `"dev"`.. For text set use `"test"`.

### Examples
Here are some examples to perform inference in different settings and datasets for the dev set:

#### TControl
```
# ToTTo
python3 ./src/main_inference.py --model_to_load_path ./models/f4__20230918_201309/checkpoints/23/ --shuffle_dataset False --image_dir ./data/ToTTo/img/notab_high_00/ --eval_batch_size 64 --dataset_dir ./data/ToTTo/ --mode "dev"
# Logic2Text
python3 ./src/main_inference.py --model_to_load_path ./models/f4__20230918_201309/checkpoints/15/ --shuffle_dataset False --image_dir ./data/Logic2Text/img/notab_high_00/ --eval_batch_size 64 --dataset_dir ./data/Logic2Text --dataset_variant l2t_totto_data --mode "dev"
```

#### LControl
```
# ToTTo
python3 ./src/main_inference.py --model_to_load_path ./models/i1__20230905_134725/checkpoints/28/ --shuffle_dataset False --image_dir ./data/ToTTo/img/highlighted_039/ --eval_batch_size 64 --dataset_dir ./data/ToTTo/ --mode "dev"
# Logic2Text
python3 ./src/main_inference.py --model_to_load_path ./models/i1__20230905_134725/checkpoints/28/ --shuffle_dataset False --image_dir ./data/Logic2Text/img/highlighted_039/ --eval_batch_size 64 --dataset_dir ./data/Logic2Text --dataset_variant l2t_totto_data --mode "dev"

```

#### OpenE
```
# ToTTo
python3 ./src/main_inference.py --model_to_load_path ./models/i3__20230905_134649/checkpoints/29/ --shuffle_dataset False --image_dir ./data/ToTTo/img/no_highlighted_039/ --eval_batch_size 64 --dataset_dir ./data/ToTTo/ --mode "dev"
# Logic2Text
python3 ./src/main_inference.py --model_to_load_path ./models/i3__20230905_134649/checkpoints/29/ --shuffle_dataset False --image_dir ./data/Logic2Text/img/no_highlighted_039/ --eval_batch_size 64 --dataset_dir ./data/Logic2Text --dataset_variant l2t_totto_data --mode "dev"
```

## Evaluate PixT3
We use the [official ToTTo evaluation code from the Google Language GitHub repository](https://github.com/google-research/language/tree/master/language/totto) 
to evaluate our inferences. First install BLEURT from [here](https://github.com/google-research/bleurt). To evaluate 
inferred texts follow these steps:
```
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
cd ..
git clone https://github.com/google-research/language.git language_repo
cd language_repo
PATH_TO_PROJECT="PATH_TO_PIXT3_PROJECT"
PATH_TO_BLEURT="PATH_TO_BLEURT_PROJECT"
# For ToTTo dev
language/totto/totto_eval.sh --prediction_path $PATH_TO_PROJECT/PixT3/out/inferences/totto/f4__20230918_201309_notab_high_00_bs/inferred_texts.txt --target_path $PATH_TO_PROJECT/data/ToTTo/totto_data/dev.jsonl --bleurt_ckpt $PATH_TO_BLEURT/bleurt/bleurt-base-128/
# For ToTTo test
# ToTTo test labels are hidden. To evaluate ToTTo test, inferences must be submitted through the official ToTTo evaluation form
# For Logic2Text dev
language/totto/totto_eval.sh --prediction_path $PATH_TO_PROJECT/PixT3/out/inferences/l2t/f4__20230918_201309_notab_high_00_bs/inferred_texts.txt --target_path $PATH_TO_PROJECT/data/Logic2Text/l2t_totto_data/dev.jsonl --bleurt_ckpt $PATH_TO_BLEURT/bleurt/bleurt-base-128/
# For Logic2Text test
language/totto/totto_eval.sh --prediction_path $PATH_TO_PROJECT/PixT3/out/inferences/l2t/f4__20230918_201309_notab_high_00_test/inferred_texts.txt --target_path $PATH_TO_PROJECT/data/Logic2Text/l2t_totto_data/test.jsonl --bleurt_ckpt $PATH_TO_BLEURT/bleurt/bleurt-base-128/
```

## Generating datasets manually
You can also generate the dataset manually fro their original sources by following these steps:
### ToTTo
Download ToTTo dataset from the [official GitHub repository](https://github.com/google-research-datasets/totto) or using:
```
 wget https://storage.googleapis.com/totto-public/totto_data.zip
 unzip totto_data.zip
```
Copy the uncompressed `totto_data` folder into `./data/ToTTo/`

To generate the images for ToTTo run:
```
export PYTHONPATH="$PWD/src"
python3 ./src/dataset/totto/preprocessing/image_generation.py totto
```

### Logic2Text
Download the original Logic2Text dataset from the [official GitHub repository](https://github.com/czyssrs/Logic2Text) and
copy all files within the `./Logic2Text/dataset/` folder into `./data/Logic2Text/original_data/`

Execute the following script to pre-process the data
```
export PYTHONPATH="$PWD/src"
python3 ./src/dataset/logic2text/preprocessing/fix_all.py
python3 ./src/dataset/logic2text/preprocessing/generate_totto_like_dataset.py
```
To convert the resulting Logic2Text dataset into the CoNT format run:
```
export PYTHONPATH="$PWD/src"
python3 ./src/dataset/totto/preprocessing/t5_dataset_generation.py l2t
```
To generate the images for Logic2Text run:
```
export PYTHONPATH="$PWD/src"
python3 ./src/dataset/totto/preprocessing/image_generation.py l2t
```

### SLC pretraining synthetic dataset
To generate the synthetic dataset run
```
export PYTHONPATH="$PWD/src"
python3 ./src/dataset/pretraining/generator.py
```
To generate the images for Synthetic dataset run:
```
export PYTHONPATH="$PWD/src"
python3 ./src/dataset/totto/preprocessing/image_generation.py slc
```
