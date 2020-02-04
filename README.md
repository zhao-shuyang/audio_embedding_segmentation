# audio_embedding_segmentation
This repository performs audio embedding extraction and change point detection. It has following function:

- Extracting audio embeddings based on a gated CNN trained using Audioset. The embeddings are extracted per frame of log-Mel spectrogram.

- Change point detection is performed based on the extracted embeddings.

- Plotting log-Mel spectrogram, embeddings with detected change points.


An example plot is as below:

![Image description](https://raw.githubusercontent.com/zhao-shuyang/audio_embedding_segmentation/master/plot.png)
