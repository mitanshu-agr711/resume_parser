from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(original_text, extracted_text):
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Convert texts into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform([original_text, extracted_text])

    # Compute Cosine Similarity
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    return similarity_score

# Example Usage
original_resume_text = "Software engineer with experience in Python, ML, and AI."
extracted_resume_text = "Experience in AI, ML, and Python as a software engineer."

similarity = calculate_similarity(original_resume_text, extracted_resume_text)
print(f"Similarity Score: {similarity:.3f}")
