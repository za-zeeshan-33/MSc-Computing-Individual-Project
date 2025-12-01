import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def doc_to_text_tfidf(doc):
    """Convert document to text for TF-IDF processing"""
    title = doc.get('title', '')
    text = doc.get('text', '') or doc.get('summary', '') or doc.get('content', '')
    return title + ' ' + text


class SearcherWithinDocs:
    """
    Document searcher for post-hoc citation assignment using TF-IDF
    """

    def __init__(self, docs, retriever="tfidf", model=None, device="cuda"):
        """
        Initialize the searcher with documents using TF-IDF
        
        Args:
            docs: List of document dictionaries
            retriever: Always "tfidf" (other options removed)
            model: Not used (kept for compatibility)
            device: Not used (kept for compatibility)
        """
        self.retriever = retriever
        self.docs = docs
        
        # Initialize TF-IDF vectorizer with flexible parameters for small document collections
        # Adjust parameters based on document collection size
        num_docs = len(docs)
        if num_docs <= 5:
            # For very small collections, use more permissive settings
            min_df_val = 1
            max_df_val = 1.0  # Allow terms that appear in all documents
            max_features_val = 5000
        else:
            # For larger collections, use standard settings
            min_df_val = 2
            max_df_val = 0.95
            max_features_val = 10000
            
        self.tfidf = TfidfVectorizer(
            max_features=max_features_val,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for better matching
            min_df=min_df_val,
            max_df=max_df_val
        )
        # Pre-compute TF-IDF vectors for all documents
        doc_texts = [doc_to_text_tfidf(doc) for doc in docs]
        self.tfidf_docs = self.tfidf.fit_transform(doc_texts)

    def search(self, query, top_k=1):
        """
        Search for the most relevant documents for a query using TF-IDF
        
        Args:
            query: Query string
            top_k: Number of top documents to return
            
        Returns:
            List of document indices (0-indexed) or single index if top_k=1
        """
        return self._search_tfidf(query, top_k)

    def _search_tfidf(self, query, top_k=1):
        """TF-IDF based search"""
        # Transform query using the fitted vectorizer
        tfidf_query = self.tfidf.transform([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(tfidf_query, self.tfidf_docs).flatten()
        
        # Get top-k document indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]  # Sort descending
        
        if top_k == 1:
            return int(top_indices[0])  # Convert numpy int64 to Python int
        return [int(idx) for idx in top_indices.tolist()]  # Convert all to Python ints

    def search_with_score(self, query):
        """Return best matching doc id and its cosine similarity score."""
        tfidf_query = self.tfidf.transform([query])
        similarities = cosine_similarity(tfidf_query, self.tfidf_docs).flatten()
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        return best_idx, best_score


    def get_doc_by_id(self, doc_id):
        """Get document by ID"""
        if 0 <= doc_id < len(self.docs):
            return self.docs[doc_id]
        return None

    def get_docs_by_ids(self, doc_ids):
        """Get multiple documents by IDs"""
        return [self.docs[doc_id] for doc_id in doc_ids if 0 <= doc_id < len(self.docs)]
