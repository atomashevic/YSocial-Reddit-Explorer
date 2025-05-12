# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Reddit-like social media simulation application built with Flask. It displays simulated user interactions with posts, comments, and news articles. The simulation includes artificial users with defined political leanings, personality traits, and demographic information.

## Commands

### Running the Application

```bash
# Start the Flask development server
python app.py
```

The application will be available at http://localhost:5000/

### Dependencies

To install required dependencies:

```bash
pip install -r requirements.txt
```

## Architecture

### Data Structure

The application uses three main data sources:

1. **posts.csv**: Contains all posts and comments with metadata
   - Includes user ID, content, thread relationships, and news article references

2. **news.csv**: Contains news articles that can be referenced by posts
   - Includes title, summary, source website, and link

3. **simulation_agents.json**: Contains details about simulated users
   - Includes demographics, political leaning, personality traits, and interests

### Application Structure

- **Flask Server (app.py)**: Main application entry point with route definitions
- **Templates**: Jinja2 HTML templates for different views
  - base.html: Base template with layout, navigation, and common elements
  - index.html: Home page displaying post feed
  - post_detail.html: Individual post view with comments
  - user_profile.html: User profile page with activity history
  - news_detail.html: News article view with related posts

### Key Features

1. **Post Feed**: Displays original posts with pagination
2. **User Profiles**: Shows user demographics, political leaning, and activity
3. **News Articles**: Displays shared news with political distribution analysis
4. **Comment Threads**: Hierarchical display of comments for each post
5. **Political Visualization**: Color-coded indicators for political leaning (Democrats: blue, Republicans: red, Independents: gray)

### Data Relationships

- Posts have a user_id linking to users in simulation_agents.json
- Comments reference original posts via comment_to field
- Posts can reference news articles via news_id field
- Thread_id groups related posts/comments into conversation threads

## Development Notes

The application is primarily for visualization of pre-generated simulation data and doesn't include functionality for creating new posts, users, or comments.

The posts.csv includes a 'round' field that represents simulation steps (hourly intervals), which are converted to timestamps starting from June 1st, 2024.

The data includes toxicity metrics for posts and comments, with visualizations available in the data/plots directory.