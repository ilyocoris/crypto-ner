import os
import tqdm
import torch
import logging
logging.basicConfig(level=logging.DEBUG)
from pymongo import MongoClient
from torch.utils.data import Dataset


class DragonflyDataset(Dataset):
    def __init__(self, tokenizer, max_length: int):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tokenizer = tokenizer
        self.max_length = max_length
        client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
        self.db = client["dragonfly"]
        self.dataset = {}
        self._load_dataset_from_mongo()

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.dataset["inputs"]["input_ids"][index], dtype=torch.long),
            "attention_mask": torch.tensor(self.dataset["inputs"]["attention_mask"][index], dtype=torch.long),
            "tags": torch.tensor(self.dataset["encoded_tags"][index], dtype=torch.long)
        }

    def __len__(self):
        return len(self.dataset["inputs"]["input_ids"])

    def _load_dataset_from_mongo(self):
        self.logger.info("Downloading raw data from Dragonfly collection.")
        self._load_raw_text_and_matches()
        self.logger.info("Tokenizing text.")
        self._tokenize_raw_text()
        self.logger.info("Bulding NER tags.")
        self._build_ner_tags()

    def _load_raw_text_and_matches(self):
        processed_chunks = []
        self.dataset["text"] = []
        self.dataset["ner_matches"] = []
        for event in tqdm.tqdm(self.db.events.find({})):
            # chunks refer to paragraphs or sentences of text
            chunk_id = event["metadata"]["chunk_id"]
            if chunk_id not in processed_chunks:
                processed_chunks.append(chunk_id)
                text = event["metadata"]["text"]
                ner_matches = []
                # extracting all matches found in the chunk
                # symbol of the crypto (BTC)
                # but can also match the name (Bitcoin) or other synonims, so match != symbol
                ner_results = event["results"]["ner"]
                for symbol in ner_results:
                    ner_matches.append({
                        "symbol": symbol,
                        "match": ner_results[symbol]["match"],
                        "span": ner_results[symbol]["span"]
                    })
                self.dataset["text"].append(text)
                self.dataset["ner_matches"].append(ner_matches)

    def _tokenize_raw_text(self):
        self.dataset["inputs"] = self.tokenizer(
            self.dataset["text"],
            padding="max_length",
            truncation=True,
            # return_tensors="pt",
            max_length=self.max_length,
            return_offsets_mapping=True
        )

    def _build_ner_tags(self):
        self.dataset["tags"] = []
        for token_ids, offsets, matches in tqdm.tqdm(zip(self.dataset["inputs"]["input_ids"], self.dataset["inputs"]["offset_mapping"], self.dataset["ner_matches"])):
            tags = []
            for token, offset in zip(self.tokenizer.convert_ids_to_tokens(token_ids), offsets):
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    tags.append("O")
                    continue
                tag = "O"
                for match in matches:
                    if tag == "O":
                        # print(token,offset,match["span"][0],offset[0] == match["span"][0],offset[0] > match["span"][0] and offset[1] <= match["span"][1])
                        if offset[0] == match["span"][0]:
                            tag = "I-CRYPTO"
                        elif offset[0] > match["span"][0] and offset[1] <= match["span"][1]:
                            tag = "CRYPTO"
                tags.append(tag)
            self.dataset["tags"].append(tags)
        tag2id = {
            "O": 0,
            "I-CRYPTO": 1,
            "CRYPTO": 2
        }
        self.dataset["encoded_tags"] = [
            [tag2id[x] for x in tag_tokens] for tag_tokens in self.dataset["tags"]]
