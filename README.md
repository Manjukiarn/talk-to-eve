# Talk-To-Eve
Mental Health Assistant is a web application that helps users discuss mental health issues. It analyzes the needs entered by the user and extracts important messages while saving previous conversations for support. It uses Flask as the backend and React as the frontend to provide interaction and support the user experience.

#  Backend Setup(Flask)

First,set up and run the backend server:
1. Clone the repository:
   git clone https://github.com/USERNAME/talk-to-eve.git
2. Navigate to the backend directory:
   cd talk-to-eve/backend
3. Create and activate a virtual environment:
   python -m venv virtual_env
   source virtual_env/Scripts/activate   # On Windows
4. Install the required dependencies
   pip install -r requirements.txt
5. Run the Flask server:
   python app.py

Open http://127.0.0.1:5000 to view the backend

# Frontend Setup(React)

1. Navigate to the frontend directory:
   cd ../frontend
2. Install frontend dependencies:
   npm install
3. Run the React app
   npm start

The React app will run at http://localhost:3000.

# Usage

1. Submit a Query: Type your message and press "Send."
2. Get Response: See the sentiment, keywords, and related past conversations.

# Technologies Used
  
  Flask for backend
  React for frontend
  Python Libraries:
     transformers
     vaderSentiment
     spaCy
     pandas

