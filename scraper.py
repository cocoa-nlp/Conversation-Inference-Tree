import json
import os

import praw

# 🔐 Replace with your actual credentials
REDDIT_CLIENT_ID = 'Placeholder'
REDDIT_CLIENT_SECRET = 'Placeholder'
REDDIT_USERNAME = 'Placeholder'
REDDIT_PASSWORD = 'Placeholder'
REDDIT_USER_AGENT = 'Basic Test Scraper'

# Authenticate
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD,
    user_agent=REDDIT_USER_AGENT
)

# Make an output directory
os.makedirs("Conversation-Inference-Tree/posts", exist_ok=True)

# Get top 10 hot posts from r/teenagers
subreddit = reddit.subreddit("Teachers")
for post in subreddit.hot(limit=10):
    post.comments.replace_more(limit=0)  # Flatten comments

    post_data = {
        "id": post.id,
        "title": post.title,
        "author": post.author.name if post.author else "[deleted]",
        "score": post.score,
        "url": post.url,
        "selftext": post.selftext,
        "num_comments": post.num_comments,
        "comments": []
    }

    # Get top 10 comments
    # [:10]:  # First 20 comments at any depth
    for comment in post.comments.list():
        post_data["comments"].append({
            "author": comment.author.name if comment.author else "[deleted]",
            "body": comment.body,
            "id": comment.id,
            "parent_id": comment.parent_id,
            "depth": comment.depth
        })

    # Write to JSON file
    filename = f"/home/brayden/GitHub/Conversation-Inference-Tree/Conversation-Inference-Tree/posts/post_{post.id}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(post_data, f, indent=4, ensure_ascii=False)

    print(f"Saved: {filename}")
