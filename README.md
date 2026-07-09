Forex AI Tutor
A full-stack Retrieval-Augmented Generation conversational chatbot engineered with Flask and Scikit-Learn. The application serves as an interactive educational assistant, mapping technical forex trading queries against a verified curriculum using machine learning-based text vectorization.

Live Link: 

Key Features

Intelligent NLP Retrieval Engine: Utilizes natural language processing to match arbitrary user syntax to exact curriculum data nodes without a hardcoded conditional infrastructure.

Mathematical Confidence Throttling: Evaluates matching precision using similarity coefficients. Low-confidence queries falling below a 25 percent match are flagged gracefully with a fallback menu to reduce incorrect outputs.

State Management and Session Persistence: Engineered a stateless client-side session tracker that string-serializes conversation history to maintain full chat continuity across page reloads without database overhead.

Modern Dark-Mode UI: Built a responsive interface utilizing Tailwind CSS optimized for readability and clean user execution.

Tech Stack and Architecture

Backend Framework: Flask (Python)

Production Web Server: Gunicorn (WSGI)

Machine Learning and NLP: Scikit-Learn (TF-IDF Vectorization, Cosine Similarity Matrices), NumPy

Frontend UI: HTML5, Tailwind CSS, Jinja2 Template Engine

Cloud Hosting: Microsoft Azure App Service / GitHub Actions

How It Works Under the Hood

Vectorization: Upon application startup, the educational knowledge base is parsed, and the questions are converted into mathematical arrays using a TF-IDF Vectorizer to filter out English stop words.

Query Intake: When a user submits a query via the UI, the string is processed and transformed into the same vector space.

Similarity Analysis: A cosine similarity matrix calculation runs instantly between the user input vector and the pre-computed matrix to isolate the maximum coordinate index.

Validation: If the resulting score clears the 0.25 mathematical threshold boundary, the corresponding curriculum value is returned; otherwise, a helpful system validation error displays.

Interface Preview


Local Installation and Setup

Follow these steps to spin up the development environment inside Visual Studio or your command line tool of choice:

Clone the Repository:
git clone https://github.com/YOUR_USERNAME/Forex-AI-Tutor.git
cd Forex-AI-Tutor/ForexTutorBot

Configure Your Virtual Environment:
python -m venv env
On Windows:
.\env\Scripts\activate

Install Core Dependencies:
pip install flask scikit-learn numpy gunicorn

Launch the Development Server:
python runserver.py

Open your browser and navigate to http://localhost:5555 to interact with the application.

Future Architecture Roadmap

Semantic Search Upgrades: Migrating from lexical TF-IDF keyword overlap vectors to dense embeddings using pre-trained Transformer models to evaluate actual sentence context.

Database Integration: Moving local string state-persistence into a persistent SQLite or PostgreSQL data tier to track long-form analytics and frequent user trends.

LLM Integration: Utilizing the Scikit-Learn matrix strictly as a context injector to pass retrieved rows into a Large Language Model API for completely fluid, conversational text generations.
