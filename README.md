[README.md](https://github.com/user-attachments/files/23743195/README.md)
Reviewer Recommendation System 
What the Project Is About

This project helps recommend suitable reviewers for a research paper automatically.
When a researcher uploads a paper, the system reads its content and finds authors with similar research areas from a given dataset.

In simple terms — it’s like a smart reviewer finder that uses NLP to match papers with the right experts.

Main Goal :

To reduce manual effort in finding reviewers and make the process faster, fair, and accurate using machine learning and text similarity.

Tools & Technologies Used :

Python – main programming language

Streamlit – to create the web app

PyMuPDF (fitz) – to read and extract text from PDFs

Sentence Transformers (BERT model) – to understand meaning of text

Scikit-learn & Gensim – for similarity comparison

Matplotlib – for showing graphs

How It Works

Upload a Paper:
The user uploads a research paper in PDF format.

Text Extraction:
The system extracts all the text from that PDF.

Compare with Author Data:
It compares the uploaded paper with the stored research papers of authors in the dataset.

Find Similarity:
Using BERT or TF-IDF, it checks which author’s writing is most similar to the uploaded paper.

Show Top Reviewers:
Finally, it displays a list of authors who are most suitable to review that paper.

Steps to Run the Project :
1️. Create a Virtual Environment
python -m venv venv
# Activate it
venv\Scripts\activate       # for Windows
source venv/bin/activate    # for macOS/Linux

2️. Install All Requirements
pip install -r requirements.txt

3️. Build the Author Embeddings
python build_embeddings.py


-> This step prepares the system by reading author papers and saving their embeddings.

4️. Run the Streamlit App
streamlit run app.py


-> Upload a paper → select number of reviewers (k) → system shows best reviewer names.

5️. (Optional) Run Evaluation File
python evaluation.py


-> This shows a bar chart comparing TF-IDF, Doc2Vec, and BERT methods.

Important Files:
| File Name                | Purpose                             |
| ------------------------ | ----------------------------------- |
| `app.py`                 | Main Streamlit app (user interface) |
| `utils.py`               | Extracts text from PDF              |
| `build_embeddings.py`    | Prepares author embeddings          |
| `reviewer_engine.py`     | Finds and ranks reviewers           |
| `evaluation.py`          | Compares model performances         |
| `authors.json`           | Stores author text data             |
| `author_embeddings.json` | Stores author vectors               |


Results

The system successfully recommends top reviewers from the dataset.

BERT-based method gave the most accurate similarity results.

A bar chart shows performance comparison between the models.

Future Improvements :

Add reviewer profiles with more details like number of papers or citations.

Allow uploading multiple papers at once.

Expand to work with large public research databases.

Final Summary

The Reviewer Recommendation System is an NLP-based web app that recommends reviewers automatically by comparing research papers using text similarity.
It saves time and ensures that the right experts are chosen for reviewing.

---
 Developed by: Akshita A  
Roll No: SE22UARI051  
