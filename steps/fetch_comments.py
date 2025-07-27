from zenml import step
from itertools import islice
from youtube_comment_downloader import *
@step
def fetch_comments(video_url,num_comments = 5):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(video_url, sort_by=SORT_BY_POPULAR)
    comments = [comment['text'] for comment in islice(comments, num_comments)]
    print(comments[0])
    return comments