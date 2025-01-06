# The Daily Reel

![The Daily Reel](favicon.avif)

A sophisticated movie recommendation system built with Flask and Google's Gemini AI. The Daily Reel helps users discover films that perfectly match their desires through natural language descriptions.

## Features

- Natural language movie search
- AI-powered recommendations using Gemini 1.5
- Detailed movie information from TMDB
- Vintage newspaper-inspired UI
- Interactive examples carousel

## How it works

The Daily Reel uses Google's Gemini 1.5 AI model to understand user queries and find matching movies. Results are enriched with data from The Movie Database (TMDB) API, providing posters, ratings, release dates and descriptions.

## Running Locally

1. Clone the repository

2. Create an `.env` file with your API keys:
```bash
GENAI_API_KEY=your_gemini_api_key
TMDB_API_KEY=your_tmdb_api_key
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Flask application:
```bash
flask run
```

The application will be available at http://localhost:5000