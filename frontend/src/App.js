import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [query, setQuery] = useState('');
  const [responseData, setResponseData] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    
    try {
      const response = await axios.post('http://127.0.0.1:5000/query', {
        text: query  // Send the input text as 'text'
      });
      setResponseData(response.data);  // Store the response from Flask backend
      setError(null);  // Clear any previous errors
    } catch (err) {
      console.error("Error occurred:", err);
      setError("Failed to get a response from the backend.");
    }
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>Mental Health Assistant</h1>
      <form onSubmit={handleSubmit} style={styles.form}>
        <input 
          type="text" 
          value={query} 
          onChange={(e) => setQuery(e.target.value)} 
          placeholder="Type your message here..."
          style={styles.input}
        />
        <button type="submit" style={styles.button}>Send</button>
      </form>
      
      {error && <p style={styles.error}>{error}</p>}

      {responseData && (
        <div style={styles.responseContainer}>
          <h3 style={styles.responseTitle}>Response:</h3>
          <p style={styles.response}>{responseData.response}</p>
          <p style={styles.sentiment}>Sentiment: {responseData.sentiment}</p>
          <p style={styles.keywords}>Keywords: {responseData.keywords && responseData.keywords.join(', ')}</p>
          <p style={styles.relevantConversation}>Relevant Conversation: {responseData.relevant_conversation}</p>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100vh',
    backgroundColor: '#f0f8ff',
    fontFamily: 'Arial, sans-serif',
    padding: '20px',
  },
  title: {
    color: '#4a4a4a',
  },
  form: {
    display: 'flex',
    flexDirection: 'row',
    marginBottom: '20px',
  },
  input: {
    padding: '10px',
    fontSize: '16px',
    border: '1px solid #ccc',
    borderRadius: '4px',
    marginRight: '10px',
    width: '300px',
  },
  button: {
    padding: '10px 15px',
    fontSize: '16px',
    backgroundColor: '#007bff',
    color: '#fff',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  buttonHover: {
    backgroundColor: '#0056b3',
  },
  error: {
    color: 'red',
  },
  responseContainer: {
    marginTop: '20px',
    border: '1px solid #ccc',
    borderRadius: '4px',
    padding: '10px',
    width: '400px',
    backgroundColor: '#fff',
  },
  responseTitle: {
    margin: 0,
    fontSize: '18px',
    color: '#007bff',
  },
  response: {
    margin: '5px 0',
  },
  sentiment: {
    margin: '5px 0',
  },
  keywords: {
    margin: '5px 0',
  },
  relevantConversation: {
    margin: '5px 0',
  }
};

export default App;
