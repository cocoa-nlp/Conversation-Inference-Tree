import praw
import json
import os

# 🔐 Replace with your actual credentials
REDDIT_CLIENT_ID = '2mjKGlb4CB82lOWEL47Ycg'
REDDIT_CLIENT_SECRET = 'XMhQUt0I-IRT8HHCKc1GefHd3z3CMA'
REDDIT_USERNAME = 'Previous_Range9377'
REDDIT_PASSWORD = 'Pickle123!'
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
subreddit = reddit.subreddit("teenagers")
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
    for comment in post.comments[:10]:
        post_data["comments"].append({
            "author": comment.author.name if comment.author else "[deleted]",
            "body": comment.body,
            "score": comment.score
        })

    # Write to JSON file
    filename = f"/home/brayden/GitHub/Conversation-Inference-Tree/Conversation-Inference-Tree/posts/post_{post.id}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(post_data, f, indent=4, ensure_ascii=False)

    print(f"Saved: {filename}")
