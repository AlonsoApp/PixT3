import torch
from datasets import load_dataset
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from tools.torch_utils import get_device

dataset = load_dataset("ybelkada/football-dataset", split="train")

from torch.utils.data import Dataset, DataLoader

MAX_PATCHES = 2048

class ImageCaptioningDataset(Dataset):
	def __init__(self, dataset, processor):
		self.dataset = dataset
		self.processor = processor

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		item = self.dataset[idx]
		encoding = self.processor(images=item["image"], return_tensors="pt", add_special_tokens=True,
								  max_patches=MAX_PATCHES)

		encoding = {k: v.squeeze() for k, v in encoding.items()}
		encoding["text"] = item["text"]
		return encoding

processor = AutoProcessor.from_pretrained("ybelkada/pix2struct-base")
model = Pix2StructForConditionalGeneration.from_pretrained("ybelkada/pix2struct-base")


def collator(batch):
	new_batch = {"flattened_patches": [], "attention_mask": []}
	texts = [item["text"] for item in batch]

	text_inputs = processor(text=texts, padding="max_length", return_tensors="pt", add_special_tokens=True,
							max_length=20)

	new_batch["labels"] = text_inputs.input_ids

	for item in batch:
		new_batch["flattened_patches"].append(item["flattened_patches"])
		new_batch["attention_mask"].append(item["attention_mask"])

	new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
	new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

	return new_batch

train_dataset = ImageCaptioningDataset(dataset, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator)

# Training
EPOCHS = 5000

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

device = get_device()  # "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()

for epoch in range(EPOCHS):
	print("Epoch:", epoch)
	for idx, batch in enumerate(train_dataloader):
		labels = batch.pop("labels").to(device)
		flattened_patches = batch.pop("flattened_patches").to(device)
		attention_mask = batch.pop("attention_mask").to(device)

		outputs = model(flattened_patches=flattened_patches,
						attention_mask=attention_mask,
						labels=labels)

		loss = outputs.loss

		print("Loss:", loss.item())

		loss.backward()

		optimizer.step()
		optimizer.zero_grad()

	if (epoch + 1) % 20 == 0:
		model.eval()

		predictions = model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask)
		print("Predictions:", processor.batch_decode(predictions, skip_special_tokens=True))

		model.train()