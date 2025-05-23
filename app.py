from flask import Flask, render_template, request, redirect, url_for, session, g
import pandas as pd
import json
import os
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Needed for session

# Add built-in functions to template context
app.jinja_env.globals.update(max=max, min=min, len=len)

# Add custom filters
@app.template_filter('first_sentence')
def first_sentence(text):
    """Extract the first sentence from a text and remove TITLE: prefix."""
    if not text:
        return ""
    # Remove "TITLE: " prefix if present
    if text.startswith("TITLE: "):
        text = text[7:]
    sentences = text.split('. ')
    first = sentences[0]
    if not first.endswith('.'):
        first += '.'
    return first

@app.template_filter('remaining_sentences')
def remaining_sentences(text, count=2):
    """Extract the next few sentences after the first one."""
    if not text:
        return ""
    # Remove "TITLE: " prefix if present
    if text.startswith("TITLE: "):
        text = text[7:]
    sentences = text.split('. ')
    if len(sentences) <= 1:
        return ""
    
    remaining = sentences[1:1+count]
    result = '. '.join(remaining)
    if len(sentences) > count + 1 and result:
        result += '...'
    return result

def get_data_folders():
    data_dir = 'data'
    return [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

# Compute default data folder based on version suffix, handling 'v2', 'v2-25', 'v2-30', etc.
def compute_default_data_folder():
    folders = get_data_folders()
    def version_key(folder):
        suffix = folder.rsplit('-', 1)[-1]
        if suffix.startswith('v') and suffix[1:].isdigit():
            return int(suffix[1:])
        if suffix.isdigit():
            return int(suffix)
        return 0
    return sorted(folders, key=version_key, reverse=True)[0]

def load_data_from_folder(folder):
    base = os.path.join('data', folder)
    posts_df = pd.read_csv(os.path.join(base, 'posts.csv'))
    news_df = pd.read_csv(os.path.join(base, 'news.csv'))
    toxicity_df = pd.read_csv(os.path.join(base, 'toxigen.csv'))
    
    # Load additional moderation data if available
    perspective_df = None
    moderation_df = None
    
    perspective_path = os.path.join(base, 'perspective.csv')
    if os.path.exists(perspective_path):
        perspective_df = pd.read_csv(perspective_path)
    
    moderation_path = os.path.join(base, 'moderation.csv')
    if os.path.exists(moderation_path):
        moderation_df = pd.read_csv(moderation_path)
    
    # --- Load the appropriate agents JSON fil  e ---
    agent_files = [f for f in os.listdir(base) if f.endswith('_agents.json')]
    # Always prefer simulation_agents.json
    if 'simulation_agents.json' in agent_files:
        agent_file = 'simulation_agents.json'
    elif agent_files:
        agent_file = agent_files[0]
    else:
        agent_file = None
    if not agent_file:
        raise FileNotFoundError(f"No agents JSON found in {base}")
    with open(os.path.join(base, agent_file), 'r') as f:
        users_data = json.load(f)
    # --- Ensure posts_df has 'username' column ---
    # First create a mapping from name to the user data
    name_to_user = {user['name']: user for user in users_data['agents']}
    
    # Get all unique user_ids from the posts dataframe
    unique_user_ids = posts_df['user_id'].unique()
    
    # Create an explicit mapping from user_id to username
    # We'll use the index+1 mapping for an initial pass
    initial_user_id_to_name = {i+1: user['name'] for i, user in enumerate(users_data['agents'])}
    
    # But then check if we see certain usernames referenced in post content
    # This helps catch mismatches between the JSON index and actual user_id
    if 'username' not in posts_df.columns:
        # Apply the initial mapping
        posts_df['username'] = posts_df['user_id'].apply(lambda x: initial_user_id_to_name.get(x, str(x)))
        
        # Now check for username references in content to validate mappings
        # For this specific case we know "ChristopherWelchMD" is being incorrectly mapped
        # Find posts by user_id 1523 that should be ChristopherWelchMD
        if 1523 in unique_user_ids and "ChristopherWelchMD" in name_to_user:
            mask = posts_df['user_id'] == 1523
            posts_df.loc[mask, 'username'] = "ChristopherWelchMD"
    # --- Merge toxicity from toxigen.csv into posts_df ---
    if 'toxicity' not in posts_df.columns:
        posts_df = posts_df.merge(toxicity_df[['id', 'toxicity']], on='id', how='left')
    else:
        # If posts_df already has 'toxicity', prefer toxigen.csv if not null
        tox_map = toxicity_df.set_index('id')['toxicity']
        posts_df['toxicity'] = posts_df.apply(
            lambda row: tox_map[row['id']] if row['id'] in tox_map else row['toxicity'], axis=1
        )
    return posts_df, news_df, users_data, toxicity_df, perspective_df, moderation_df

@app.route('/', methods=['GET', 'POST'])
def select_data_folder():
    # Auto-select the newest data folder on first GET
    if request.method == 'GET' and 'data_folder' not in session:
        default_folder = compute_default_data_folder()
        session['data_folder'] = default_folder
        return redirect(url_for('home'))
    if request.method == 'POST':
        folder = request.form.get('data_folder')
        if folder in get_data_folders():
            session['data_folder'] = folder
            return redirect(url_for('home'))
    folders = get_data_folders()
    return render_template('select_data_folder.html', folders=folders)

# Helper to get current data
def get_current_data():
    folder = session.get('data_folder')
    if not folder:
        return None
    if not hasattr(g, 'data_cache') or g.get('data_cache_folder') != folder:
        posts_df, news_df, users_data, toxicity_df, perspective_df, moderation_df = load_data_from_folder(folder)
        g.data_cache = (posts_df, news_df, users_data, toxicity_df, perspective_df, moderation_df)
        g.data_cache_folder = folder
    return g.data_cache

# Convert rounds to timestamps
def round_to_timestamp(round_num):
    # Assuming round 1 starts at June 1st 2024 09:00 AM and each round is an hour
    start_date = datetime(2024, 6, 1, 9, 0, 0)
    return start_date + timedelta(hours=(round_num - 1))

# Calculate user toxicity statistics
def calculate_user_toxicity_stats(posts_df):
    # Group by username and calculate mean toxicity
    user_toxicity = posts_df.groupby('username')['toxicity'].agg(['mean', 'count']).reset_index()
    user_toxicity.columns = ['username', 'avg_toxicity', 'post_count']
    
    # Calculate overall toxicity distribution percentiles
    all_toxicity = posts_df['toxicity'].dropna()
    percentiles = {
        'p25': all_toxicity.quantile(0.25),
        'p50': all_toxicity.quantile(0.50),
        'p75': all_toxicity.quantile(0.75),
        'p90': all_toxicity.quantile(0.90),
        'p95': all_toxicity.quantile(0.95)
    }
    
    # Calculate percentile rank for each user
    user_toxicity['percentile_rank'] = user_toxicity['avg_toxicity'].apply(
        lambda x: (all_toxicity <= x).mean() * 100
    )
    
    return user_toxicity, percentiles

# Enhanced user analytics with all moderation metrics
def calculate_comprehensive_user_stats(posts_df, toxicity_df, perspective_df, moderation_df, target_username):
    # Create comprehensive moderation data
    moderation_data = create_moderation_data(posts_df, toxicity_df, perspective_df, moderation_df)
    
    # Filter for target user
    user_data = moderation_data[moderation_data['username'] == target_username].copy()
    
    if user_data.empty:
        return None
    
    # Calculate user-specific statistics
    user_stats = {
        'total_posts': len(user_data[user_data['comment_to'] == -1]),
        'total_comments': len(user_data[user_data['comment_to'] != -1]),
        'total_content': len(user_data)
    }
    
    # Toxigen statistics
    if 'toxigen_score' in user_data.columns:
        toxigen_scores = user_data['toxigen_score'].dropna()
        if not toxigen_scores.empty:
            user_stats['toxigen'] = {
                'avg_score': toxigen_scores.mean(),
                'max_score': toxigen_scores.max(),
                'high_toxicity_count': len(toxigen_scores[toxigen_scores >= 0.6]),
                'total_scored': len(toxigen_scores)
            }
    
    # Perspective statistics
    if 'perspective_score' in user_data.columns:
        perspective_scores = user_data['perspective_score'].dropna()
        if not perspective_scores.empty:
            user_stats['perspective'] = {
                'avg_score': perspective_scores.mean(),
                'max_score': perspective_scores.max(),
                'high_toxicity_count': len(perspective_scores[perspective_scores >= 0.6]),
                'total_scored': len(perspective_scores)
            }
    
    # OpenAI moderation statistics
    if 'flagged' in user_data.columns:
        flagged_data = user_data['flagged'].dropna()
        if not flagged_data.empty:
            flagged_count = flagged_data.sum()
            user_stats['openai_moderation'] = {
                'flagged_count': int(flagged_count),
                'total_checked': len(flagged_data),
                'flagged_percentage': (flagged_count / len(flagged_data)) * 100
            }
            
            # Category-specific flags
            categories = ['hate', 'harassment', 'violence', 'sexual', 'self_harm']
            user_stats['openai_categories'] = {}
            for category in categories:
                if category in user_data.columns:
                    category_scores = user_data[category].dropna()
                    if not category_scores.empty:
                        user_stats['openai_categories'][category] = {
                            'avg_score': category_scores.mean(),
                            'max_score': category_scores.max(),
                            'high_risk_count': len(category_scores[category_scores >= 0.5])
                        }
    
    # Overall risk statistics
    if 'overall_risk' in user_data.columns:
        risk_scores = user_data['overall_risk'].dropna()
        if not risk_scores.empty:
            user_stats['overall_risk'] = {
                'avg_risk': risk_scores.mean(),
                'max_risk': risk_scores.max(),
                'high_risk_count': len(risk_scores[risk_scores >= 0.6]),
                'critical_risk_count': len(risk_scores[risk_scores >= 0.8])
            }
    
    return user_stats

# Generate risk distribution data for all users
def calculate_user_risk_distribution(posts_df, toxicity_df, perspective_df, moderation_df):
    # Create comprehensive moderation data
    moderation_data = create_moderation_data(posts_df, toxicity_df, perspective_df, moderation_df)
    
    # Calculate average risk per user
    user_risk_stats = moderation_data.groupby('username').agg({
        'overall_risk': ['mean', 'count', 'max'],
        'toxigen_score': 'mean',
        'perspective_score': 'mean'
    }).reset_index()
    
    # Flatten column names
    user_risk_stats.columns = ['username', 'avg_risk', 'content_count', 'max_risk', 'avg_toxigen', 'avg_perspective']
    
    # Filter users with at least 3 pieces of content for meaningful stats
    user_risk_stats = user_risk_stats[user_risk_stats['content_count'] >= 3]
    
    # Calculate percentiles for risk distribution
    risk_percentiles = {
        'p10': user_risk_stats['avg_risk'].quantile(0.10),
        'p25': user_risk_stats['avg_risk'].quantile(0.25),
        'p50': user_risk_stats['avg_risk'].quantile(0.50),
        'p75': user_risk_stats['avg_risk'].quantile(0.75),
        'p90': user_risk_stats['avg_risk'].quantile(0.90),
        'p95': user_risk_stats['avg_risk'].quantile(0.95)
    }
    
    return user_risk_stats, risk_percentiles

# Merge all moderation data for a comprehensive view
def create_moderation_data(posts_df, toxicity_df, perspective_df, moderation_df):
    # Start with posts data
    moderation_data = posts_df.copy()
    
    # Merge toxigen toxicity scores
    if toxicity_df is not None:
        toxicity_cols = ['id', 'toxicity']
        if 'toxicity' in toxicity_df.columns:
            moderation_data = moderation_data.merge(
                toxicity_df[toxicity_cols].rename(columns={'toxicity': 'toxigen_score'}),
                on='id', how='left'
            )
    
    # Merge perspective API scores
    if perspective_df is not None:
        perspective_cols = ['id', 'toxicity']
        if 'toxicity' in perspective_df.columns:
            moderation_data = moderation_data.merge(
                perspective_df[perspective_cols].rename(columns={'toxicity': 'perspective_score'}),
                on='id', how='left'
            )
    
    # Merge OpenAI moderation data
    if moderation_df is not None:
        moderation_cols = ['id', 'flagged']
        # Add specific category scores if available
        category_cols = ['sexual', 'harassment', 'hate', 'violence', 'self_harm']
        available_cols = ['id'] + [col for col in category_cols if col in moderation_df.columns]
        if 'flagged' in moderation_df.columns:
            available_cols.append('flagged')
        
        moderation_data = moderation_data.merge(
            moderation_df[available_cols],
            on='id', how='left'
        )
    
    # Add overall risk score (combination of all metrics)
    risk_factors = []
    if 'toxigen_score' in moderation_data.columns:
        risk_factors.append(moderation_data['toxigen_score'].fillna(0))
    if 'perspective_score' in moderation_data.columns:
        risk_factors.append(moderation_data['perspective_score'].fillna(0))
    if 'flagged' in moderation_data.columns:
        risk_factors.append(moderation_data['flagged'].fillna(False).astype(int))
    
    if risk_factors:
        moderation_data['overall_risk'] = sum(risk_factors) / len(risk_factors)
    else:
        moderation_data['overall_risk'] = 0
    
    return moderation_data

@app.route('/home')
def home():
    if 'data_folder' not in session:
        # ensure a data folder is set and proceed
        session['data_folder'] = compute_default_data_folder()
    posts_df, news_df, users_data, toxicity_df, perspective_df, moderation_df = get_current_data()
    page = request.args.get('page', 1, type=int)
    per_page = 25 # Try 25 for testing, 20 is the default

    # Get original posts only (comment_to == -1)
    original_posts = posts_df[posts_df['comment_to'] == -1].sort_values('round', ascending=False)

    # Pagination
    total_pages = (len(original_posts) + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    paginated_posts = original_posts.iloc[start_idx:end_idx].copy()
    # Ensure timestamp column exists
    if 'timestamp' not in paginated_posts.columns:
        paginated_posts['timestamp'] = paginated_posts['round'].apply(round_to_timestamp)
    # Ensure toxicity column exists and is numeric
    if 'toxicity' not in paginated_posts.columns:
        paginated_posts['toxicity'] = 0.0
    paginated_posts['toxicity'] = paginated_posts['toxicity'].fillna(0).astype(float)

    # Count comments (including nested) for each post using thread_id
    post_comment_counts = {}
    for post_id in paginated_posts['id']:
        mask = (posts_df['thread_id'] == post_id) & (posts_df['comment_to'] != -1)
        post_comment_counts[post_id] = len(posts_df[mask])

    # Convert DataFrame to a list of dictionaries for the template
    posts_list = paginated_posts.to_dict('records')
    
    # Ensure user info is retrieved using the username
    users_dict = {user['name']: user for user in users_data['agents']}
    for post in posts_list:
        post['user'] = users_dict.get(post['username'])
        # Guarantee 'toxicity' and 'timestamp' keys for template
        if 'toxicity' not in post:
            post['toxicity'] = 0.0
        if 'timestamp' not in post:
            post['timestamp'] = round_to_timestamp(post['round'])
    
    # Count statistics for sidebar
    original_posts_count = len(posts_df[posts_df['comment_to'] == -1])
    comments_count = len(posts_df[posts_df['comment_to'] != -1])
    users_count = len(users_dict)
    news_count = len(news_df)
    
    # Build news dictionary keyed by news id for correct lookup in templates
    news_map = news_df.set_index('id').to_dict('index')
    return render_template('index.html',
                          posts=posts_list,
                          users=users_dict,
                          news=news_map,
                          comment_counts=post_comment_counts,
                          page=page,
                          total_pages=total_pages,
                          original_posts_count=original_posts_count,
                          comments_count=comments_count,
                          users_count=users_count,
                          news_count=news_count)

@app.route('/post/<int:post_id>')
def post_detail(post_id):
    if 'data_folder' not in session:
        # ensure a data folder is set and proceed
        session['data_folder'] = compute_default_data_folder()
    posts_df, news_df, users_data, toxicity_df, perspective_df, moderation_df = get_current_data()
    # Get the original post
    post = posts_df[posts_df['id'] == post_id].iloc[0].to_dict()
    # Ensure timestamp exists
    if 'timestamp' not in post:
        post['timestamp'] = round_to_timestamp(post['round'])
    # Ensure toxicity exists and is not NaN
    if 'toxicity' not in post or pd.isna(post['toxicity']):
        post['toxicity'] = 0.0
    # Build nested comments tree for this post
    thread_df = posts_df[posts_df['thread_id'] == post_id].sort_values('round')
    comments_records = thread_df.to_dict('records') if not thread_df.empty else []
    users_dict = {user['name']: user for user in users_data['agents']}
    # Get author info for the original post
    author = users_dict.get(post['username'])
    # Initialize comment map and replies list
    comment_map = {c['id']: c for c in comments_records}
    for c in comment_map.values():
        # ensure timestamp and toxicity
        if 'timestamp' not in c or pd.isna(c.get('timestamp')):
            c['timestamp'] = round_to_timestamp(c['round'])
        if 'toxicity' not in c or pd.isna(c.get('toxicity')):
            c['toxicity'] = 0.0
        c['user'] = users_dict.get(c['username'])
        c['replies'] = []
    # Link replies to parents
    nested_comments = []
    for c in comment_map.values():
        parent = c['comment_to']
        if parent == post_id:
            nested_comments.append(c)
        elif parent in comment_map:
            comment_map[parent]['replies'].append(c)
    # Total comments count (exclude the original post itself)
    total_comments_count = len([c for c in comments_records if c['comment_to'] != -1])
    news_article = None
    if post['news_id'] != -1 and not pd.isna(post['news_id']):
        news_article = news_df[news_df['id'] == int(post['news_id'])].to_dict('records')[0]
    # Debug: print post dict to check structure and type
    print('DEBUG post_detail post:', type(post), post)
    return render_template('post_detail.html',
                          post=post,
                          comments=nested_comments,
                          total_comments_count=total_comments_count,
                          author=author,
                          users=users_dict,
                          news_article=news_article)

@app.route('/user/<username>')
def user_profile(username):
    if 'data_folder' not in session:
        session['data_folder'] = compute_default_data_folder()
    posts_df, news_df, users_data, toxicity_df, perspective_df, moderation_df = get_current_data()
    # Get user data
    users_dict = {user['name']: user for user in users_data['agents']}
    user = users_dict.get(username)
    if not user:
        return "User not found", 404

    # Get comprehensive moderation data for all content
    moderation_data = create_moderation_data(posts_df, toxicity_df, perspective_df, moderation_df)
    
    # Get all posts and comments by this user with moderation data
    user_content = moderation_data[moderation_data['username'] == username].sort_values('round', ascending=False)

    # Separate posts and comments
    user_posts = user_content[user_content['comment_to'] == -1]
    user_comments = user_content[user_content['comment_to'] != -1]
    
    # Convert to dictionaries
    user_posts_list = user_posts.to_dict('records') if not user_posts.empty else []
    user_comments_list = user_comments.to_dict('records') if not user_comments.empty else []
    # Ensure timestamp for posts and comments
    for post in user_posts_list:
        if 'timestamp' not in post:
            post['timestamp'] = round_to_timestamp(post['round'])
    for comment in user_comments_list:
        if 'timestamp' not in comment:
            comment['timestamp'] = round_to_timestamp(comment['round'])
    
    # Get legacy toxicity statistics (for backward compatibility)
    user_toxicity_df, toxicity_percentiles = calculate_user_toxicity_stats(posts_df)
    user_toxicity_stats = user_toxicity_df[user_toxicity_df['username'] == username]
    toxicity_stats = None
    if not user_toxicity_stats.empty:
        toxicity_stats = user_toxicity_stats.iloc[0].to_dict()
        
        # Add user's position in distribution
        if toxicity_stats['avg_toxicity'] <= toxicity_percentiles['p25']:
            toxicity_stats['position'] = 'bottom 25%'
        elif toxicity_stats['avg_toxicity'] <= toxicity_percentiles['p50']:
            toxicity_stats['position'] = 'bottom 25-50%'
        elif toxicity_stats['avg_toxicity'] <= toxicity_percentiles['p75']:
            toxicity_stats['position'] = 'top 25-50%'
        elif toxicity_stats['avg_toxicity'] <= toxicity_percentiles['p90']:
            toxicity_stats['position'] = 'top 10-25%'
        elif toxicity_stats['avg_toxicity'] <= toxicity_percentiles['p95']:
            toxicity_stats['position'] = 'top 5-10%'
        else:
            toxicity_stats['position'] = 'top 5%'

    # Get comprehensive user analytics
    comprehensive_stats = calculate_comprehensive_user_stats(posts_df, toxicity_df, perspective_df, moderation_df, username)
    
    # Get risk distribution data
    user_risk_distribution, risk_percentiles = calculate_user_risk_distribution(posts_df, toxicity_df, perspective_df, moderation_df)
    
    # Find user's position in risk distribution
    user_risk_position = None
    user_risk_percentile = None
    if comprehensive_stats and 'overall_risk' in comprehensive_stats:
        user_avg_risk = comprehensive_stats['overall_risk']['avg_risk']
        # Calculate percentile rank
        all_user_risks = user_risk_distribution['avg_risk']
        user_risk_percentile = (all_user_risks <= user_avg_risk).mean() * 100
        
        # Determine position description
        if user_risk_percentile <= 25:
            user_risk_position = 'Low Risk (Bottom 25%)'
        elif user_risk_percentile <= 50:
            user_risk_position = 'Below Average Risk (25-50%)'
        elif user_risk_percentile <= 75:
            user_risk_position = 'Above Average Risk (50-75%)'
        elif user_risk_percentile <= 90:
            user_risk_position = 'High Risk (75-90%)'
        else:
            user_risk_position = 'Very High Risk (Top 10%)'
    
    # Prepare distribution data for chart (simplified for template)
    distribution_data = {
        'user_risk': comprehensive_stats['overall_risk']['avg_risk'] if comprehensive_stats and 'overall_risk' in comprehensive_stats else 0,
        'user_percentile': user_risk_percentile or 0,
        'percentiles': risk_percentiles,
        'position_text': user_risk_position
    }

    return render_template('user_profile.html',
                          user=user,
                          posts=user_posts_list,
                          comments=user_comments_list,
                          news=news_df.set_index('id').to_dict('index'),
                          toxicity_stats=toxicity_stats,
                          toxicity_percentiles=toxicity_percentiles,
                          comprehensive_stats=comprehensive_stats,
                          distribution_data=distribution_data)

@app.route('/news/<int:news_id>')
def news_detail(news_id):
    if 'data_folder' not in session:
        session['data_folder'] = compute_default_data_folder()
    posts_df, news_df, users_data, toxicity_df, perspective_df, moderation_df = get_current_data()
    # Get the news article
    news_article = news_df[news_df['id'] == news_id].to_dict('records')[0]
    if news_article is None:
        return "News article not found", 404

    # Get all posts referencing this news article
    related_posts = posts_df[posts_df['news_id'] == news_id].sort_values('round', ascending=False)
    related_posts_list = related_posts.to_dict('records') if not related_posts.empty else []
    
    # Ensure each post has user info and timestamp
    users_dict = {user['name']: user for user in users_data['agents']}
    for post in related_posts_list:
        post['user'] = users_dict.get(post['username'])
        if 'timestamp' not in post:
            post['timestamp'] = round_to_timestamp(post['round'])
    return render_template('news_detail.html',
                          news=news_article,
                          posts=related_posts_list,
                          users=users_dict)

@app.template_filter('format_timestamp')
def format_timestamp(timestamp):
    return timestamp.strftime('%b %d, %Y - %I:%M %p')

@app.template_filter('toxicity_class')
def toxicity_class(toxicity):
    if toxicity >= 0.75:
        return 'danger'
    elif toxicity >= 0.5:
        return 'warning'
    elif toxicity >= 0.25:
        return 'info'
    else:
        return 'success'
        
@app.template_filter('format_content')
def format_content(text):
    """Format post content by removing TITLE: prefix and adding paragraph breaks."""
    if not text:
        return ""
    # Remove "TITLE: " prefix if present
    if text.startswith("TITLE: "):
        text = text[7:]
    return text
        
@app.template_filter('percentile_class')
def percentile_class(percentile_rank):
    if percentile_rank >= 95:
        return 'danger'
    elif percentile_rank >= 90:
        return 'warning'
    elif percentile_rank >= 75:
        return 'info'
    elif percentile_rank >= 50:
        return 'primary'
    else:
        return 'success'

@app.template_filter('risk_level_text')
def risk_level_text(risk_score):
    if risk_score is None:
        return 'Unknown'
    if risk_score >= 0.8:
        return 'Critical'
    elif risk_score >= 0.6:
        return 'High'
    elif risk_score >= 0.4:
        return 'Medium'
    elif risk_score >= 0.2:
        return 'Low'
    else:
        return 'Minimal'

@app.template_filter('score_badge_class')
def score_badge_class(score):
    if score is None:
        return 'secondary'
    if score >= 0.8:
        return 'danger'
    elif score >= 0.6:
        return 'warning'
    elif score >= 0.4:
        return 'info'
    else:
        return 'success'

@app.route('/moderation')
def moderation():
    if 'data_folder' not in session:
        session['data_folder'] = compute_default_data_folder()
    
    posts_df, news_df, users_data, toxicity_df, perspective_df, moderation_df = get_current_data()
    
    # Create comprehensive moderation data
    moderation_data = create_moderation_data(posts_df, toxicity_df, perspective_df, moderation_df)
    
    # Get filtering parameters from request
    toxigen_threshold = request.args.get('toxigen_threshold', 0.0, type=float)
    perspective_threshold = request.args.get('perspective_threshold', 0.0, type=float)
    flagged_only = request.args.get('flagged_only', False, type=bool)
    content_type = request.args.get('content_type', 'all')
    sort_by = request.args.get('sort_by', 'overall_risk')
    sort_order = request.args.get('sort_order', 'desc')
    search_query = request.args.get('search', '').strip()
    
    # Apply filters
    filtered_data = moderation_data.copy()
    
    # Filter by toxigen threshold
    if 'toxigen_score' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['toxigen_score'].fillna(0) >= toxigen_threshold]
    
    # Filter by perspective threshold
    if 'perspective_score' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['perspective_score'].fillna(0) >= perspective_threshold]
    
    # Filter by flagged status
    if flagged_only and 'flagged' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['flagged'] == True]
    
    # Filter by content type
    if content_type == 'posts':
        filtered_data = filtered_data[filtered_data['comment_to'] == -1]
    elif content_type == 'comments':
        filtered_data = filtered_data[filtered_data['comment_to'] != -1]
    
    # Filter by search query
    if search_query:
        filtered_data = filtered_data[
            filtered_data['tweet'].str.contains(search_query, case=False, na=False)
        ]
    
    # Sort data
    sort_columns = {
        'overall_risk': 'overall_risk',
        'toxigen_score': 'toxigen_score',
        'perspective_score': 'perspective_score',
        'timestamp': 'round'
    }
    
    if sort_by in sort_columns:
        ascending = sort_order == 'asc'
        filtered_data = filtered_data.sort_values(
            sort_columns[sort_by], 
            ascending=ascending, 
            na_position='last'
        )
    
    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 25
    total_items = len(filtered_data)
    total_pages = (total_items + per_page - 1) // per_page
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_data = filtered_data.iloc[start_idx:end_idx].copy()
    
    # Add timestamps and user data
    if 'timestamp' not in paginated_data.columns:
        paginated_data['timestamp'] = paginated_data['round'].apply(round_to_timestamp)
    
    # Convert to list of dictionaries
    moderation_list = paginated_data.to_dict('records')
    
    # Add user information
    users_dict = {user['name']: user for user in users_data['agents']}
    for item in moderation_list:
        item['user'] = users_dict.get(item['username'])
        if 'timestamp' not in item:
            item['timestamp'] = round_to_timestamp(item['round'])
    
    # Calculate statistics
    stats = {
        'total_items': len(moderation_data),
        'filtered_items': total_items,
        'high_risk_items': len(moderation_data[moderation_data['overall_risk'] >= 0.7]),
        'flagged_items': len(moderation_data[moderation_data.get('flagged', False) == True]) if 'flagged' in moderation_data.columns else 0
    }
    
    return render_template('moderation.html',
                          items=moderation_list,
                          page=page,
                          total_pages=total_pages,
                          stats=stats,
                          filters={
                              'toxigen_threshold': toxigen_threshold,
                              'perspective_threshold': perspective_threshold,
                              'flagged_only': flagged_only,
                              'content_type': content_type,
                              'sort_by': sort_by,
                              'sort_order': sort_order,
                              'search': search_query
                          })

if __name__ == '__main__':
    app.run(debug=True)
