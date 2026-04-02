import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        # фиксированные id
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        
        current_id = 4
        
        for text in texts:
            words = text.split()
            for word in words:
                if word not in self.word_to_id:
                    self.word_to_id[word] = current_id
                    current_id += 1
        
        # обратный словарь
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        words = text.split()
        return [
            self.word_to_id.get(word, self.word_to_id[self.unk_token])
            for word in words
        ]
    
    def decode(self, ids: List[int]) -> str:
        words = [
            self.id_to_word.get(i, self.unk_token)
            for i in ids
        ]
        return " ".join(words)