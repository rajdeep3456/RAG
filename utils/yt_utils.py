from langchain_community.document_loaders import YoutubeLoader

def process_youtube(url):
    loader=YoutubeLoader.from_youtube_url(url,add_video_info=True)
    docs=loader.load()
    return docs
