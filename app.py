from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import json
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# Load data
def load_data():
    # Load posts and comments
    posts_df = pd.read_csv('data/posts.csv')

    # Load news articles
    news_df = pd.read_csv('data/news.csv')

    # Load user data
    with open('data/simulation_agents.json', 'r') as f:
        users_data = json.load(f)

    return posts_df, news_df, users_data

# Convert rounds to timestamps
def round_to_timestamp(round_num):
    # Assuming round 1 starts at June 1st 2024 09:00 AM and each round is an hour
    start_date = datetime(2024, 6, 1, 9, 0, 0)
    return start_date + timedelta(hours=(round_num - 1))

# Initialize data
posts_df, news_df, users_data = load_data()

# Convert round to readable timestamp
posts_df['timestamp'] = posts_df['round'].apply(round_to_timestamp)

# Create lookup dictionaries for efficiency
users_dict = {user['name']: user for user in users_data['agents']}
news_dict = {row['id']: row for _, row in news_df.iterrows()}

@app.route('/')
def home():
    page = request.args.get('page', 1, type=int)
    per_page = 20

    # Get original posts only (comment_to == -1)
    original_posts = posts_df[posts_df['comment_to'] == -1].sort_values('round', ascending=False)

    # Pagination
    total_pages = (len(original_posts) + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    paginated_posts = original_posts.iloc[start_idx:end_idx]

    # Count comments for each post
    post_comment_counts = {}
    for post_id in paginated_posts['id']:
        post_comment_counts[post_id] = len(posts_df[posts_df['comment_to'] == post_id])

    # Convert DataFrame to a list of dictionaries for the template
    posts_list = paginated_posts.to_dict('records')
    
    # Count statistics for sidebar
    original_posts_count = len(posts_df[posts_df['comment_to'] == -1])
    comments_count = len(posts_df[posts_df['comment_to'] != -1])
    
    return render_template('index.html',
                          posts=posts_list,
                          users=users_dict,
                          news=news_dict,
                          comment_counts=post_comment_counts,
                          page=page,
                          total_pages=total_pages,
                          original_posts_count=original_posts_count,
                          comments_count=comments_count)

@app.route('/post/<int:post_id>')
def post_detail(post_id):
    # Get the original post
    post = posts_df[posts_df['id'] == post_id].iloc[0]

    # Get all comments for this post
    comments = posts_df[posts_df['comment_to'] == post_id].sort_values('round')

    # Get post author
    author = users_dict.get(post['user_id'])

    # Get news article if post references one
    news_article = None
    if post['news_id'] != -1 and not pd.isna(post['news_id']):
        news_article = news_dict.get(int(post['news_id']))

    return render_template('post_detail.html',
                          post=post,
                          comments=comments,
                          author=author,
                          users=users_dict,
                          news_article=news_article)

@app.route('/user/<username>')
def user_profile(username):
    # Get user data
    user = users_dict.get(username)
    if not user:
        return "User not found", 404

    # Get all posts and comments by this user
    user_content = posts_df[posts_df['user_id'] == username].sort_values('round', ascending=False)

    # Separate posts and comments
    user_posts = user_content[user_content['comment_to'] == -1]
    user_comments = user_content[user_content['comment_to'] != -1]

    return render_template('user_profile.html',
                          user=user,
                          posts=user_posts,
                          comments=user_comments,
                          news=news_dict)

@app.route('/news/<int:news_id>')
def news_detail(news_id):
    # Get the news article
    news_article = news_dict.get(news_id)
    if not news_article:
        return "News article not found", 404

    # Get all posts referencing this news article
    related_posts = posts_df[posts_df['news_id'] == news_id].sort_values('round', ascending=False)

    return render_template('news_detail.html',
                          news=news_article,
                          posts=related_posts,
                          users=users_dict)

@app.template_filter('format_timestamp')
def format_timestamp(timestamp):
    return timestamp.strftime('%b %d, %Y - %I:%M %p')

if __name__ == '__main__':
    app.run(debug=True)
