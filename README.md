# References
>> This project would not be possible without the contributions of many engineers and researchers before me.

The main paper that this implementation references and attributes credit to is:
* [Adiyansjaha, Alexander A S Gunawana, Derwin Suhartono: Music Recommender System Based on Genre Using Convolutional Recurrent Neural Networks](https://www.sciencedirect.com/science/article/pii/S1877050919310646)

Different implementations in the wild/pre-processing references:
* [taylorhawks/RNN-music-recommender](https://github.com/taylorhawks/RNN-music-recommender)
* [Roberts, Leland: Understanding the Mel-Spectrogram](https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53)

## Development Status
The `cayde` system is in very early stages of development and is currently in its research stage. Primary focus of development is data processing and normalization, whilst also continuing research into the various references listed along others which may or may not be added to the above list in the future. Below are some figures of the alternative implementations being tested. This implementation differs from the architectures implemented in the paper referenced above, primarily in it accepting as inputs the Mel-frequency Cepstral Coefficients instead of one-hot-encoded spectrogram pixels. The model below was trained on the full GTZAN dataset, whilst only a reduced version of it is included in the repository [here](https://github.com/phasewalk1/cayde/tree/master/example-train/GTZAN-reduced).

<img src="perf/model_performance-ckpt9.png" alt="image1" style="display:inline-block;">

## About Preprocessing
`cayde` is trained on the Mel-scaled spectrograms and MFCCs (Mel-frequency Cepstral Coefficients) of the segmented audio files. We segment each audio file into 5 segments each, allowing us to increase the execution time of the preprocessing whilst simultaneously, increasing the number of data points in our training/hold-out sets. We extract the MFCCs from each segment, and use the development training set to extract the provided semantic labels (see below) used for supervised learning. The file `data.json` is constructed as a result of running [wrangler.py](https://github.com/phasewalk1/cayde/tree/master/wrangling/wrangler.py), and is an example training set that contains the preprocessed labels, mappings, and MFCCs from a reduced version of the GTZAN dataset. 

## The [Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download) used for development (Proof of Concept)
As we begin building the model for a music recommendation system, we first need a way to extract features (such as genre) from segments of audio files. Since the application (SB) isn't ready to provide us with a training set, we utilize the GTZAN dataset that retains a directory structure such that genre labels can be easily extracted in order to perform supervised learning with the CNN model (see below).
