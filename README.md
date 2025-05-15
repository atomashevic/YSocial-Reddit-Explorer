# YSocial Reddit Explorer

This is a Flask web application for exploring and visualizing YSocial Reddit- simulation post and news data. The app provides a simple interface to view posts, news articles, user profiles, and various data visualizations.

## Features

- Home page with summary statistics
- View individual posts and news articles
- User profile pages
- Data visualizations (histograms, boxplots, ECDF, etc.) for posts and toxicity analysis

## Data Files Required
The application expects the following data files in the `data/` directory:
- `posts.csv`: Contains YSocial Reddit-simulation post data
- `news.csv`: Contains YSocial Reddit-simulation news article data
- `simulation_agents.json`: Agent simulation data
- `toxigen.csv`: Toxicity analysis data
- `summary.txt`, `toxigen_summary.txt`: Summary statistics
- Plots in `data/plots/`: Various PNG images for visualizations

## Getting Started
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```p "
2. Run the app:
   ```sh
   python app.py
   ```
3. Open your browser at [http://localhost:5000](http://localhost:5000)

## Directory Structure
- `app.py` — Main Flask application
- `data/` — Data files and generated plots
- `static/css/` — Stylesheets
- `templates/` — HTML templates

---

