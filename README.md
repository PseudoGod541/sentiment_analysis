# Full-Stack Twitter Sentiment Analysis Application

This project is an end-to-end Natural Language Processing (NLP) application that analyzes the sentiment of a given text. It features a sophisticated Bidirectional LSTM model trained with pre-trained GloVe embeddings, served via a high-performance FastAPI backend, and presented through an interactive Streamlit web interface. The entire application is containerized with Docker for easy and reproducible deployment.



---

## 📋 Features

-   **Advanced NLP Model**: A Bidirectional LSTM network leveraging `glove-twitter-100` embeddings to understand the nuances of social media text.
-   **Multi-Class Classification**: Classifies text into four categories: Positive, Negative, Neutral, and Irrelevant.
-   **FastAPI Backend**: A robust REST API that accepts text input and returns detailed sentiment predictions and confidence scores.
-   **Streamlit Frontend**: A beautiful and user-friendly web app to enter text, view the predicted sentiment with emojis, and see a breakdown of all confidence scores.
-   **Dockerized**: The complete application, including the API and frontend, is managed by Docker Compose for a simple, one-command setup.

---

## 🛠️ Tech Stack

-   **Backend**: Python, FastAPI, Uvicorn
-   **Machine Learning**: TensorFlow, Keras, Scikit-learn, Pandas, Gensim, NumPy
-   **Frontend**: Streamlit, Requests
-   **Deployment**: Docker, Docker Compose

---
```bash
## 🚀 How to Run

To run this application, you need to have Docker and Docker Compose installed on your machine.

### 1. Clone the Repository


git clone <your-repository-url>
cd <your-project-directory>

2. Place Model Files
Ensure your trained model and preprocessor files are placed inside a models/ directory in the root of the project. This is a crucial step.

best_model.h5

tokenizer.pkl

label_encoder.pkl

3. Run with Docker Compose
This single command will build the Docker image and start both the FastAPI backend and the Streamlit frontend services.

docker-compose up --build

4. Access the Application
Once the containers are running, you can access the services:

Streamlit Frontend: Open your browser and go to http://localhost:8501

FastAPI Backend Docs: Open your browser and go to http://localhost:8000/docs

📁 Project Structure
.
├── models/
│   ├── best_model.h5         # Trained Keras model
│   ├── tokenizer.pkl         # Fitted tokenizer
│   └── label_encoder.pkl     # Fitted label encoder
├── main.py                   # FastAPI application
├── streamlit_app.py          # Streamlit frontend application
├── Dockerfile                # Instructions to build the Docker image
├── docker-compose.yml        # Defines and runs the multi-container setup
├── .dockerignore             # Specifies files to ignore during build
└── requirements.txt          # Python dependencies
