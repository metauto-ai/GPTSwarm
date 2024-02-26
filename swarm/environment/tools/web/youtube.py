#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pytube import YouTube
from swarm.utils.const import GPTSWARM_ROOT

def Youtube(url, has_subtitles):
    # get video id from url
    video_id=url.split('v=')[-1].split('&')[0]
    # Create a YouTube object
    youtube = YouTube(url)
    # Get the best available video stream
    video_stream = youtube.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if has_subtitles:
        # Download the video to a location
        print('Downloading video')
        video_stream.download(output_path="{GPTSWARM_ROOT}/workspace",filename=f"{video_id}.mp4")
        print('Video downloaded successfully')
        return f"{GPTSWARM_ROOT}/workspace/{video_id}.mp4"
    else:
        return video_stream.url 