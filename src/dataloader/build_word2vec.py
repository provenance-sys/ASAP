import os
import sys
import re
import random
import hashlib
import numpy as np
from tqdm import tqdm
from zxcvbn import zxcvbn
from gensim.models import Word2Vec
from hashid import HashID
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Tuple

current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../../.."))
sys.path.insert(0, project_root)
from src.utils.opreventmodel import OPREventModel


class ContentEmbedder:
    """
    A class for training Word2Vec models and generating content embeddings.
    
    This class processes OPREventModel data to train Word2Vec models and provides
    methods to generate embeddings for content by combining word vectors.
    """
    
    def __init__(self, dataset_name: str = None, train_list: list = None,
                vector_size: int = 16, window: int = 5, min_count: int = 1, 
                workers: int = 4, k_nearest: int = 5):
        """
        Initialize the ContentEmbedder.
        
        Args:
            vector_size: Dimension of word vectors
            window: Context window size for Word2Vec
            min_count: Minimum word frequency for inclusion in vocabulary
            workers: Number of worker threads for training
            k_nearest: Number of nearest neighbors for unknown words
        """
        self.dataset_name = dataset_name
        self.train_list = train_list
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.k_nearest = k_nearest
        self.model = None
        self.detector = HashID()

    @staticmethod
    def is_hashs(s):
        s = s.lower()
        if re.fullmatch(r'[a-f0-9]{16}', s):
            return True
        elif re.fullmatch(r'[a-f0-9]{32}', s):
            return True
        elif re.fullmatch(r'[a-f0-9]{40}', s):
            return True
        elif re.fullmatch(r'[a-f0-9]{64}', s):
            return True
        elif re.fullmatch(r'[a-f0-9]{128}', s):
            return True
        else:
            return False
        
    def _preprocess_content(self, content: str, _type: str) -> str:
        """
        Preprocess content by replacing special characters and common path/filename delimiters with spaces.
        
        Args:
            content: Input content string
            
        Returns:
            Preprocessed content string
        """
        if content is None:
            return ""
        content = content.replace('\\', '/')
        content = str(content).strip().strip('"').strip('/')
        if 'Net' in _type:
            content = content.split(':')[0]
            content = max(content.split(':'), key=len)
            content = ' '.join(content.split('.'))  # r'[/:._\-]+'
        else:
            if self.dataset_name == 'e3clearscope':
                parts = re.split(r'[/]+', content)
            else:
                parts = re.split(r'[/:._\-]+', content)
            parts = [re.sub(r'\d+$', '', part) for part in parts if not part.isdigit() and not ContentEmbedder.is_hashs(part)]
            content = ' '.join(parts)
        content = re.sub(r'\s+', ' ', content).strip()
        return content
    
    def _extract_sentences_from_oprem(self, oprem: OPREventModel) -> List[str]:
        """
        Extract sentences from OPREventModel by concatenating node and edge content.
        
        Args:
            oprem: OPREventModel instance
            
        Returns:
            List of sentence strings
        """
        sentences = []
        u_contents = oprem.get_u('content')
        v_contents = oprem.get_v('content')
        u_type = oprem.get_u('type')
        v_type = oprem.get_v('type')
        for u_content in u_contents.split(';'):
            for v_content in v_contents.split(';'):
                u_content = self._preprocess_content(u_content, u_type)
                v_content = self._preprocess_content(v_content, v_type)
                
                if u_content:
                    sentences.append(u_content)
                if v_content:
                    sentences.append(v_content)
                
        return sentences
    
    def _tokenize_sentences(self, sentences: List[str]) -> List[List[str]]:
        """
        Tokenize sentences into word lists.
        
        Args:
            sentences: List of sentence strings
            
        Returns:
            List of tokenized sentences (each sentence is a list of words)
        """
        tokenized = []
        for sentence in sentences:
            if sentence.strip():
                words = sentence.lower().split()
                if len(words) > 0:
                    tokenized.append(words)
        return tokenized
    
    def train(self, data_dir: str, model_save_dir: Optional[str] = None) -> None:
        """
        Train Word2Vec model on OPREventModel data.
        
        Args:
            data_dir: Directory containing .txt files with OPREventModel data
            model_save_path: Optional path to save the trained model
        """
        model_save_path = os.path.join(model_save_dir, f'word2vec_dim{self.vector_size}_window{self.window}.pth')
        if os.path.exists(model_save_path):
            self.load_model(model_save_path)
            return

        print("Loading data files...")
        file_list = [os.path.join(data_dir, fname + '.txt') for fname in self.train_list]
        
        all_sentences = []
        sentence_set = set()
        
        print("Processing OPREventModel data...")
        for _file in tqdm(file_list, desc="Processing files"):
            with open(_file, 'r', encoding='utf-8') as fin:
                for line in tqdm(fin, desc=f"Processing {os.path.basename(_file)}", leave=False):
                    try:
                        oprem = OPREventModel()
                        oprem.update_from_loprem(line.strip().split('\t'))
                        sentences = self._extract_sentences_from_oprem(oprem)
                        for sentence in sentences:
                            if sentence not in sentence_set:
                                if random.uniform(0, 1) <= 0.8:
                                    sentence_set.add(sentence)
                                    all_sentences.append(sentence)
                    except Exception as e:
                        print(f"Error processing line: {e}")
                        continue
        
        print(f"Total unique sentences: {len(all_sentences)}")
        
        # Tokenize sentences
        print("Tokenizing sentences...")
        tokenized_sentences = self._tokenize_sentences(all_sentences)
        
        print(f"Training Word2Vec model with {len(tokenized_sentences)} sentences...")
        # Train Word2Vec model
        self.model = Word2Vec(
            sentences=tokenized_sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=1  # Use skip-gram
        )

        print(f"Model trained with vocabulary size: {len(self.model.wv.index_to_key)}")

        # Save model if path provided
        self.model.save(model_save_path)
        print(f"word2vec embeddding model saved to: {model_save_path}")
        
    def load_model(self, model_path: str) -> None:
        """
        Load a pre-trained Word2Vec model.
        
        Args:
            model_path: Path to the saved Word2Vec model
        """
        self.model = Word2Vec.load(model_path)
        print(f"Model loaded with vocabulary size: {len(self.model.wv.index_to_key)}")
    
    def _get_word_embedding(self, word: str) -> np.ndarray:
        """
        Get embedding for a single word, handling unknown words.
        
        Args:
            word: Input word
            
        Returns:
            Word embedding vector
        """
        word = word.lower()
        
        # If word exists in vocabulary
        if word in self.model.wv:
            return self.model.wv[word]
        else:
            seed = int(hashlib.md5(word.encode('utf-8')).hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(seed)
            return rng.uniform(low=-0.2, high=0.2, size=self.vector_size)
            # return np.zeros(self.vector_size)
    
    def get_embedding(self, contents: str, _type: str) -> np.ndarray:
        """
        Get embedding for content by splitting on special characters and combining word vectors.
        
        Args:
            content: Input content string
            
        Returns:
            Content embedding vector
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        if contents is None or contents.strip() == "":
            return np.zeros(self.vector_size)
        
        content_embeddings = []
        for content in contents.split(';'):
            processed_content = self._preprocess_content(content, _type)
            if not processed_content:
                continue
            
            words = set(processed_content.lower().split())
            if not words:
                continue
            
            # Get embeddings for each word and sum them
            word_embeddings = []
            for word in words:
                embedding = self._get_word_embedding(word)
                word_embeddings.append(embedding)
            
            # Sum all word embeddings
            content_embedding = np.sum(word_embeddings, axis=0)
            content_embeddings.append(content_embedding)
        
        if not content_embeddings:
            return np.zeros(self.vector_size)

        content_embeddings = np.mean(content_embeddings, axis=0)
        return content_embeddings

    def get_random_word_embedding(self) -> np.ndarray:
        """
        Randomly return the embedding of a word from the vocabulary. If the vocabulary is empty, return a random vector.
        Returns:
            np.ndarray: Embedding of a random word
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        vocab = self.model.wv.index_to_key
        if not vocab or random.uniform(0, 1) < 0.3:
            return np.random.uniform(low=-0.2, high=0.2, size=self.vector_size)
        random_word = random.choice(vocab)
        return self.model.wv[random_word]


if __name__ == "__main__":
    pass
