These scripts were used to gather data using the YouTube Data APIs.

The first script has an array of the top channel names (according to SocialBlade.com).  It goes through every channel name, getting the channel ID.  For every channel ID, it gets the 50 most recently uploaded videos and the channel subscriber count.  It then goes through every video, calculates the channel statistics based on what the previous videos in the channel showed, and makes a row for every video using the video stats, the previous video stats, and the channel average stats (except for the first video in every channel is skipped because there are no previous stats).

The second script has a list of video IDs of recently trending videos from a Kaggle dataset.  It goes through every video and gets the appropriate data.

Both scripts have a calcFinalCSV() function to download the CSV once the data has been retrieved.