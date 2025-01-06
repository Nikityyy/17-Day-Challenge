from flask import Flask, jsonify, request
import google.generativeai as genai
import requests
import os

app = Flask(__name__)

genai.configure(api_key=os.getenv("GENAI_API_KEY"))

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=generation_config,
)

SYSTEM_PROMPT = """You are a highly sophisticated movie recommendation system. Your task is to find exactly the movies that match the user's description.
Return ONLY the movie names, 10 of them, each on a new line. No additional text, just movie names.
Be as precise as possible in matching the user's request. Consider themes, atmosphere, plot elements, and emotional resonance.
Example output format:
The Shawshank Redemption
Inception
The Matrix
[etc...]"""

TMDB_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {os.getenv('TMDB_API_KEY')}"
}

@app.route('/')
def index():
    HTML_STRING = """
    <!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reel - Find Your Next Movie</title>
    <link rel="icon" href="https://files.catbox.moe/ot17d5.avif" type="image/avif">
    <meta name="description" content="Find your next movie with Reel. Discover movie recommendations, reviews, and more.">
    <meta name="keywords" content="movies, movie recommendations, film reviews, movie search, film discovery, best movies, top films">
    <meta name="author" content="Reel">
    <meta property="og:title" content="Reel - Find Your Next Movie">
    <meta property="og:description" content="Discover movie recommendations, reviews, and more with Reel.">
    <meta property="og:type" content="website">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="Reel - Find Your Next Movie">
    <meta name="twitter:description" content="Discover movie recommendations, reviews, and more with Reel.">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link
        href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=UnifrakturMaguntia&family=Old+Standard+TT:ital,wght@0,400;0,700;1,400&display=swap"
        rel="stylesheet">
    <style>
        :root {
            --primary-bg: #f4e6c9;
            --text-color: #2b2015;
            --card-bg: #f8f1e4;
            --border-color: #8b7355;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            outline: none;
        }

        body {
            font-family: 'Old Standard TT', serif;
            background-color: var(--primary-bg);
            color: var(--text-color);
            line-height: 1.6;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100svh;
        }

        .container {
            width: 80%;
            max-width: 800px;
            padding: 2rem;
            margin: 0 auto;
            background-color: #ffffff;
            border: 2px solid var(--border-color);
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
            position: relative;
        }

        .paper-texture {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            opacity: 0.7;
            pointer-events: none;
        }

        .masthead,
        .search-container,
        .examples-carousel,
        .results-container {
            position: relative;
            z-index: 1;
        }

        .masthead {
            text-align: center;
            border-bottom: 3px double var(--border-color);
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }

        .logo {
            font-family: 'UnifrakturMaguntia', serif;
            font-size: 4rem;
            margin: 1rem 0;
            color: var(--text-color);
            text-transform: none;
        }

        .date-line {
            font-style: italic;
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 1rem;
        }

        .tagline {
            font-family: 'Old Standard TT', serif;
            font-size: 1.2rem;
            font-style: italic;
            margin: 1rem 0;
            color: #555;
        }

        .search-container {
            max-width: 600px;
            margin: 2rem auto;
            text-align: center;
        }

        .loading {
            position: relative;
        }

        .loading::after {
            content: '';
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            width: 20px;
            height: 20px;
            border: 2px solid #ccc;
            border-top-color: #333;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to {
                transform: translateY(-50%) rotate(360deg);
            }
        }

        .search-input {
            width: 100%;
            padding: 0.8rem;
            font-size: 1rem;
            border: 1px solid var(--border-color);
            background: transparent;
            font-family: 'Old Standard TT', serif;
        }

        .search-input:focus {
            outline: 1px solid var(--border-color);
        }

        .examples-carousel {
            margin: 2rem 0;
            overflow: hidden;
            border-top: 2px solid var(--border-color);
            border-bottom: 2px solid var(--border-color);
            padding: 1rem 0;
            position: relative;
        }

        .examples-track {
            display: flex;
            gap: 2rem;
            position: relative;
            width: fit-content;
        }

        .example-card {
            background: transparent;
            padding: 0.8rem 1.2rem;
            cursor: pointer;
            font-family: 'Old Standard TT', serif;
            font-style: italic;
            min-width: max-content;
            border: 1px solid transparent;
            position: relative;
        }

        .example-card:hover {
            border-color: var(--border-color);
        }

        .results-container {
            display: none;
            width: 100%;
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 1rem;
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 2rem;
        }

        .movie-card {
            background: transparent;
            border: 1px solid var(--border-color);
            cursor: pointer;
            width: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .movie-poster {
            width: 100%;
            max-width: 300px;
            height: auto;
            aspect-ratio: 2/3;
            object-fit: cover;
            border: none;
            padding: 1rem;
            margin: 1rem 0;
        }

        .movie-info {
            padding: 1rem;
            text-align: left;
            border-top: 1px solid var(--border-color);
        }

        .movie-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.2rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }

        .movie-meta {
            font-style: italic;
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }

        .movie-description {
            font-size: 1rem;
            line-height: 1.6;
        }

        @keyframes scroll {
            0% {
                transform: translateX(0);
            }

            100% {
                transform: translateX(calc(-50%));
            }
        }

        @media (max-width: 768px) {
            .container {
                padding: 1rem;
                margin: 0.5rem;
            }

            .logo {
                font-size: 2.5rem;
            }

            .results-container {
                grid-template-columns: 1fr;
                margin: 0 1rem 0 0;
            }

            .movie-card {
                margin: 0 auto;
            }
        }

        @media screen and (max-width: 600px) {
            .results-container {
                grid-template-columns: 1fr;
                padding: 0 0.5rem;
            }

            .movie-poster {
                margin: 0;
            }
        }
    </style>
</head>

<body>
    <div class="container">
        <svg class="paper-texture" viewBox="0 0 100 100" preserveAspectRatio="none" style="overflow: hidden;">
            <defs>
                <filter id="paperEffect">
                    <feTurbulence type="fractalNoise" baseFrequency="0.04" numOctaves="5" result="noise" />
                    <feDiffuseLighting in="noise" lightingColor="#fff" surfaceScale="2" result="texture">
                        <feDistantLight azimuth="45" elevation="60" />
                    </feDiffuseLighting>

                    <feTurbulence type="turbulence" baseFrequency="0.05" numOctaves="3" seed="5" result="tornNoise" />
                    <feDisplacementMap in="SourceGraphic" in2="tornNoise" scale="10" xChannelSelector="R"
                        yChannelSelector="G" result="tornEdges" />

                    <feBlend in="texture" in2="tornEdges" mode="multiply" result="combined" />
                </filter>
            </defs>

            <rect x="0" y="0" width="100" height="100" fill="#ffffff" filter="url(#paperEffect)" />
        </svg>
        <div class="masthead">
            <h1 class="logo">The Daily Reel</h1>
            <div class="date-line">Est. <span id="year"></span></div>
            <p class="tagline">~ Purveyor of Fine Motion Picture Recommendations ~</p>
        </div>

        <div class="search-container">
            <input type="text" class="search-input" spellcheck="false" placeholder="Describe the movie you seek..."
                autofocus>
        </div>

        <div class="examples-carousel">
            <div class="examples-track">
                <div class="example-card">A cozy tale of self-discovery in a small township</div>
                <div class="example-card">A psychological drama in the vein of Sir Nolan's 'Inception'</div>
                <div class="example-card">A thrilling mystery that keeps one guessing</div>
                <div class="example-card">A heartwarming story of unlikely companionship</div>
                <div class="example-card">An exploration of the human psyche</div>
                <div class="example-card">A visual feast with minimal dialogue</div>
                <div class="example-card">A tale of time travel and its consequences</div>
                <div class="example-card">A picture that questions the nature of reality</div>
                <div class="example-card">A nostalgic adventure from the 1980s</div>
                <div class="example-card">A story of machines gaining consciousness</div>
            </div>
        </div>

        <div class="results-container">
        </div>
    </div>

    <script>
        const searchInput = document.querySelector('.search-input');
        const examplesCarousel = document.querySelector('.examples-carousel');
        const resultsContainer = document.querySelector('.results-container');
        const searchContainer = document.querySelector('.search-container');

        document.getElementById('year').textContent = convertToRoman(2025);

        function convertToRoman(num) {
            const romanNumerals = [
                ['M', 1000],
                ['CM', 900],
                ['D', 500],
                ['CD', 400],
                ['C', 100],
                ['XC', 90],
                ['L', 50],
                ['XL', 40],
                ['X', 10],
                ['IX', 9],
                ['V', 5],
                ['IV', 4],
                ['I', 1]
            ];
            let result = '';
            for (const [roman, value] of romanNumerals) {
                while (num >= value) {
                    result += roman;
                    num -= value;
                }
            }
            return result;
        }

        searchInput.addEventListener('keyup', (e) => {
            if (e.key === 'Enter') {
                handleSearch();
            }
        });

        async function handleSearch() {
            const query = searchInput.value.trim();
            searchInput.value = '';
            if (!query) return;

            searchContainer.classList.add('loading');

            examplesCarousel.style.display = 'none';
            resultsContainer.style.display = 'grid';

            fetchResults(query).then(() => {
                searchContainer.classList.remove('loading');
            });
        }

        async function fetchResults(query) {
            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query })
                });
                const movies = await response.json();
                displayResults(movies);
            } catch (error) {
                console.error('Error fetching results:', error);
                searchContainer.classList.remove('loading');
            }
        }

        function displayResults(movies) {
            resultsContainer.innerHTML = movies.map(movie => `
                <div onclick="window.open('${movie.movie_url}')" class="movie-card">
                    <img class="movie-poster" src="${movie.poster}" alt="${movie.title}" onerror="this.src='/api/placeholder/300/450'" loading="lazy">
                    <div class="movie-info">
                        <h3 class="movie-title">${movie.title}</h3>
                        <div class="movie-meta">${movie.release_date ? new Date(movie.release_date).getFullYear() : 'N/A'} • ${movie.rating.toFixed(1)}/10</div>
                        <p class="movie-description">${movie.overview || 'No description available.'}</p>
                    </div>
                </div>
            `).join('');
        }

        function setupInfiniteScroll() {
            const track = document.querySelector('.examples-track');
            const cards = track.innerHTML;

            track.innerHTML = cards + cards;

            const contentWidth = track.scrollWidth;
            const duration = contentWidth / 150;

            track.style.animation = `scroll ${duration}s linear infinite`;

            track.addEventListener('click', (e) => {
                const card = e.target.closest('.example-card');
                if (card) {
                    searchInput.value = card.textContent.trim();
                    handleSearch();
                }
            });

            track.addEventListener('mouseenter', () => {
                track.style.animationPlayState = 'paused';
            });

            track.addEventListener('mouseleave', () => {
                track.style.animationPlayState = 'running';
            });
        }

        document.addEventListener('DOMContentLoaded', setupInfiniteScroll);

        window.addEventListener('error', function (e) {
            if (e.target.tagName === 'IMG') {
                e.target.src = '/api/placeholder/300/450';
            }
        });
    </script>
</body>
</html>
    """
    return HTML_STRING

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    
    chat = model.start_chat(history=[])
    response = chat.send_message(
        f"{SYSTEM_PROMPT}\n\nUser request: {query}",
        safety_settings={
            'HATE': 'BLOCK_NONE',
            'HARASSMENT': 'BLOCK_NONE',
            'SEXUAL': 'BLOCK_NONE',
            'DANGEROUS': 'BLOCK_NONE'
        }
    )
    
    movie_names = [name.strip() for name in response.text.split('\n') if name.strip()]
    
    movies_data = []
    for movie_name in movie_names:
        params = {
            "query": movie_name,
            "include_adult": "true",
            "language": "en-US",
            "page": "1"
        }
        
        response = requests.get(TMDB_URL, headers=TMDB_HEADERS, params=params)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                movie = data['results'][0]
                movies_data.append({
                    'id': movie['id'],
                    'title': movie['title'],
                    'overview': movie['overview'],
                    'poster': f"https://media.themoviedb.org/t/p/w300_and_h450_bestv2{movie['poster_path']}" if movie['poster_path'] else None,
                    'release_date': movie['release_date'],
                    'rating': movie['vote_average'],
                    'movie_url': f"https://www.themoviedb.org/movie/{movie['id']}"
                })
    
    return jsonify(movies_data)

if __name__ == '__main__':
    app.run(debug=True)