📧 AI-Based Email Reply Assistant

An intelligent web application that analyzes email content, predicts its intent using Natural Language Processing (NLP), and generates professional reply suggestions. The system combines Machine Learning, rule-based logic, and confidence handling to deliver reliable responses for real-world emails.

🚀 Live Demo

🔗 https://ai-email-reply-assistant-na5wawqxgeer7zehwfka3m.streamlit.app/

🎯 Project Objective

Writing professional email replies can be time-consuming and challenging, especially in corporate environments.
This project aims to:
Automatically classify email intent
Suggest professional, context-aware replies
Improve productivity and communication quality

Key Features

📌 Email intent classification (request, apology, follow_up, inquiry)
📊 Confidence score for predictions
🔁 Hybrid AI approach (ML + rule-based fallback)
✉️ Professional reply generation
🌐 Interactive web interface using Streamlit
☁️ Deployed online for real-time usage

🏗️ System Architecture

User enters email text
Text preprocessing (cleaning & stopword removal)
Feature extraction using TF-IDF
Intent prediction using Logistic Regression
Confidence-based decision handling
Professional reply generation

🛠️ Technologies Used

| Category             | Tools                                  |
| -------------------- | -------------------------------------- |
| Programming Language | Python                                 |
| NLP                  | NLTK, TF-IDF                           |
| Machine Learning     | Scikit-learn (Logistic Regression)     |
| Web Framework        | Streamlit                              |
| Dataset              | Custom + Enron Email Dataset (labeled) |
| Deployment           | Streamlit Community Cloud              |

📂 Project Structure

ai-email-reply-assistant/
│
├── app.py
├── train_model.py
├── intent_model.pkl
├── vectorizer.pkl
├── requirements.txt
├── packages.txt
├── data/
│   └── emails.csv
└── README.md

📊 Dataset Description

A manually curated dataset containing real-world email examples
Emails categorized into:
request
apology
follow_up
inquiry
Augmented using selected samples from the Enron Email Dataset

🧪 Model Training & Evaluation

Text vectorization using TF-IDF (unigrams + bigrams)
Classification using Logistic Regression
Stratified train-test split
Performance evaluated using:
Accuracy
Precision
Recall
F1-score
K-Fold Cross Validation for reliable evaluation on small datasets

▶️ How to Run Locally

1️⃣ Clone the repository

git clone https://github.com/Prapthigatty/ai-email-reply-assistant.git

cd ai-email-reply-assistant

2️⃣ Install dependencies

pip install -r requirements.txt

3️⃣ Train the model

python train_model.py

4️⃣ Run the application

streamlit run app.py

Open in browser:

http://localhost:8501

🧠 Sample Test Emails

Apology

Sorry for the delay in responding. Please find the document attached.

Request

Could you please send me the assignment details?

Follow-up

Just following up on my previous email regarding the payment.

Inquiry

I would like to know more about the internship opportunity.

📌 Limitations

Model accuracy depends on dataset size

New email patterns may reduce confidence

Designed as a learning and academic project

🔮 Future Enhancements

Sentiment analysis

Multiple reply styles (formal / short / friendly)

Larger labeled datasets

Transformer-based NLP models

Email API integration (Gmail)

🎓 Academic Relevance

Suitable for MCA Mini Project / Main Project

Demonstrates:
NLP concepts
Machine learning workflow
Model evaluation
Web deployment

Responsible AI design

👩‍💻 Author

Prapthi A
MCA Student
Interests: Machine Learning, Web Development, NLP


