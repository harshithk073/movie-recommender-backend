# 🎬 Movie Recommender System: From Local ML to Full-Stack Deployment

## Project Overview

This repository documents the complete journey of building, testing, and deploying a machine learning-based movie recommender system. The project began as a Jupyter notebook with ML code and evolved into a full-stack web application, with a Flask backend and a modern JavaScript frontend, both deployed on professional cloud platforms.

---

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Development Journey](#development-journey)
- [Testing the API](#testing-the-api)
  - [Local Testing Example](#local-testing-example)
  - [Render Deployment Testing Example](#render-deployment-testing-example)
- [Result](#result)
- [Key Takeaways](#key-takeaways)
- [License](#license)

---

## Introduction

This project implements a content-based movie recommendation system using Python, scikit-learn, and Flask. The system takes a user's input movie and returns personalized recommendations. The frontend is a responsive web app built with HTML, CSS, and JavaScript, with an emphasis on design and user experience.

---

## Project Structure

- **Backend (`movie-recommender-backend`):**
  - `api.py` — Flask API serving recommendations
  - `movies.csv` — Movie dataset
  - `requirements.txt` — Backend dependencies
  - `untitled.ipynb` — Notebook for running the API
  - `untitled1.ipynb` — Notebook for testing the API

- **Frontend (`movie-recommender-frontend`):**
  - `index.html` — Main web page
  - `style.css` — Styling 
  - `script.js` — Handles user input and API calls

---

## Development Journey

### 1. Building the ML Recommender

- Started in a Jupyter notebook, loaded and cleaned movie data.
- Engineered features (genres, keywords, tagline, cast, director).
- Used `TfidfVectorizer` and `cosine_similarity` to compute movie similarities.
- Built a Python function to generate recommendations.

### 2. Creating the Flask API

- Wrapped the ML logic in a Flask app (`api.py`).
- Exposed a `/recommend` POST endpoint.
- Enabled CORS to allow frontend requests.

### 3. Local Testing

- Ran the API locally (`python api.py`).
- Used a separate notebook to test POST requests with `requests`.
- Verified that the API returned correct recommendations for various inputs.

### 4. Frontend Development

- Designed a clean, user-friendly interface with HTML and CSS.
- Wrote `script.js` to handle user input and fetch recommendations from the backend.
- Tested the full user flow locally, ensuring frontend and backend communicated seamlessly.

### 5. Repository Organization

- Split project into two repos for best deployment practices:
  - **Backend:** `movie-recommender-backend`
  - **Frontend:** `movie-recommender-frontend`
- Used Git and GitHub for version control, committing and pushing all changes.

### 6. Preparing for Deployment

- Created a clean `requirements.txt` (with all lowercase package names).
- Ensured the Flask app object was named `app`.
- Confirmed the correct Render start command: `gunicorn api:app`.
- Updated frontend’s `script.js` to use the live backend URL.

### 7. Backend Deployment on Render

- Chose "Web Service" on Render.
- Connected the backend repo and set the start command.
- Faced a **JSONDecodeError** when testing: discovered it was due to POSTing to `/` instead of `/recommend` (received a 404 HTML error).
- Fixed the endpoint, printed raw responses for debugging, and confirmed the API worked.

### 8. Frontend Deployment on Vercel

- Connected the frontend repo to Vercel.
- Deployed the site, which automatically rebuilt on each push.
- Updated API endpoint in `script.js` to point to the Render backend.

### 9. Integration and Final Testing

- Visited the Vercel URL, entered a movie, and saw live recommendations.
- The frontend and backend communicated perfectly, providing a seamless user experience.

### 10. Troubleshooting and Lessons Learned

- **Endpoint confusion:** Fixed by always POSTing to `/recommend`.
- **JSONDecodeError:** Solved by printing response status and text.
- **Deployment settings:** Clarified correct Render service type and commands[1].
- **Requirements.txt:** Ensured all package names were lowercase.
- **CORS:** Confirmed was enabled in Flask for cross-origin requests.
- **Logs:** Used Render logs for backend error diagnosis.

---

## Testing the API

### Local Testing Example

To test the API locally, first start your Flask app (`python api.py` or `!python api.py` in a notebook), then run the following code in a separate Jupyter notebook cell or Python script:

import requests
response = requests.post('http://localhost:5000/recommend', json={'movie_title': 'finding nemo'})
print(response.json())


If everything is set up correctly, you should see a JSON response with movie recommendations for "finding nemo".

---

### Render Deployment Testing Example

After deploying your backend to Render, test it with:

import requests
url = 'https://your-backend-url.onrender.com/recommend' # Replace with your actual Render URL

data = {'movie_title': 'Inception'}

response = requests.post(url, json=data)

print("Status:", response.status_code)
print("Text:", response.text)


**Common error and solution:**  
- If you see a 404 error and an HTML response, make sure you're POSTing to `/recommend` and not the root `/`.

---

## Result

- **Backend:** Flask API live on Render (with automatic sleeping/wake-up on free tier[1]).
- **Frontend:** Responsive web app live on Vercel[3].
- **Connected:** Full-stack solution, version-controlled and production-ready.
- **Professional workflow:** Modern deployment pipeline, easy to update and scale.

---

## Key Takeaways

- Test locally before deploying.
- Use clear, RESTful endpoints and always check responses.
- Separate backend and frontend for scalable deployment.
- Leverage cloud platforms (Render, Vercel) for fast, reliable hosting.
- Version control is essential for safety and collaboration[2].
- Persistence and careful debugging are key to solving real-world deployment issues.

---

## License

This project is for educational and demonstration purposes.

---

*Built and deployed with care, patience, and a passion for learning. If you found this project helpful or inspiring, feel free to fork, star, or reach out!*
