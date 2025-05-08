# Data Structure Summary and File Relationships

The provided data represents a simulation of Reddit-like social media interactions, spread across three main files:

## 1. `reddit-app/data/posts.csv`
- Contains posts and comments data with columns:
  - `id`: Unique identifier for each post/comment
  - `tweet`: The content of the post or comment
  - `post_img`: Image URL (if applicable)
  - `user_id`: ID of the user who posted
  - `comment_to`: ID of the post being commented on (-1 for original posts)
  - `thread_id`: ID of the thread the post belongs to
  - `round`: Simulation round number, every round is an hour. This can be converted to dates (let's say first round is June 1st 2024 09:00 AM)
  - `news_id`: References news articles in news.csv
  - `shared_from`: Tracks reshared content
  - `image_id`: References images (if applicable)

## 2. `reddit-app/data/news.csv`
- Contains news articles that can be referenced by posts:
  - `id`: Unique identifier (referenced by posts.csv)
  - `title`: Article title
  - `summary`: Brief description of the article
  - `website_id`: Source website identifier
  - `fetched_on`: Date the article was retrieved
  - `link`: URL to the original article

## 3. `reddit-app/data/simulation_agents.json`
- Contains details about the simulated users:
  - Personal information: name, email, password
  - Demographics: age, gender, nationality
  - Political leaning (Republican/Democrat/Independent)
  - Personality traits (oe, co, ex, ag, ne)
  - Interests and education level
  - Engagement parameters: toxicity level, round_actions
  - Technical settings: recommendation systems, language

## Relationships Between Files:
1. **Posts to Users**: Each post in posts.csv has a `user_id` field that corresponds to an entry in the agents.json file.
2. **Posts to News**: Posts can reference news articles via the `news_id` field, pointing to entries in news.csv.
3. **Comments to Posts**: The `comment_to` field in posts.csv indicates which post a comment is responding to (-1 indicates an original post).
4. **Thread Structure**: The `thread_id` field groups related posts/comments together into conversation threads.
5. **Simulation Tracking**: The `round` field in posts.csv corresponds to simulation iterations, showing when each post was created.

This dataset appears to simulate a social media environment where automated agents with varied personalities, political leanings, and demographic characteristics interact with each other and discuss news articles. The resulting data captures both the content of these interactions and their hierarchical structure (original posts and comments), allowing analysis of discussion patterns and agent
