# A Recommendation System for the Steam Online Game Store Using Text Classification Model

## Abstract

Steam serves as the largest digital distribution platform for PC gaming, with approximately 120 million monthly users. To provide a unique and tailored storefront, Steam utilizes machine learning to recommend products based on user preferences. This project explores an offline recommendation system using the Term Frequency-Inverse Document Frequency (TF-IDF) statistical measure with features extracted from the SteamSpy API. The project includes data acquisition, cleanup, formatting, and the implementation of TF-IDF for recommendations. The model was tested on a dataset of 1,000 game entries, and the recommendations were similar to those of the Steam storefront.

## Introduction

Recommendation systems enhance user engagement by tailoring product or service suggestions based on user preferences. Examples include YouTube, Netflix, and Amazon. This project focuses on personalized recommendations for Steam games using content-based filtering and TF-IDF to recommend similar games based on user input.

### Types of Recommender Systems

- **Collaborative Filtering**: Uses user details, ratings, and reviews to build recommendations. It identifies similar users and recommends items liked by those users.
  
- **Content-Based Filtering**: Analyzes product features and establishes similarity metrics to recommend products. It uses information retrieval and assigns weights to attributes using TF-IDF.

- **Hybrid Filtering**: Combines collaborative and content-based filtering to overcome individual approach limitations, providing a more comprehensive recommendation system.

## Method

### Design Approach

The project follows a three-phase approach:

1. **Data Acquisition**: Uses the SteamSpy API to retrieve game data.
2. **Data Cleanup**: Pre-processes and formats the data for analysis.
3. **Data Processing**: Implements TF-IDF and cosine similarity to recommend similar games.

**Tools Used**:
- Python 3.9 (Spyder IDE)
- Pandas
- SteamSpy API
- Scikit-Learn

### Data Acquisition

Data is retrieved using the SteamSpy API, which provides features for each game. A function processes requests and retrieves data for the top 1,000 games, which is saved as `raw_steamspy_data.csv`.

### Data Cleaning

Pre-processes the data to remove irrelevant columns and handle missing values. The “tags” feature is formatted and cleaned for use in the content-based filtering approach.

### Data Processing

Uses TF-IDF for text vectorization of game features (e.g., genre and tags). Computes cosine similarity to determine the top 10 most similar games for recommendations.

## Results

The final dataset consists of 968 games after processing. The recommendations are compared with the Steam storefront's suggestions, showing a good match.

## Discussion

The project achieved its objectives but could benefit from retrieving more game titles and descriptions for improved accuracy. Future improvements include normalizing numerical values and enhancing text vectorization.

## Future Work

Future improvements include combining numerical values with the similarity matrix and performing text vectorization on game descriptions.

## Conclusion

The project successfully downloaded, cleaned, and processed data from Steam. TF-IDF provided a basis for making accurate recommendations by comparing wording similarities between games.

## Build Instructions

To replicate the project:

1. **Run `data_acquisition.py`** to create `raw_steamspy_data.csv` in the `.Cpe646_Project` directory. This may take approximately 10 minutes.
2. **Run `data_cleanup.py`** to create `steamspy_data_clean.csv` in the `.Cpe646_Project` directory.
3. **Run `data_processing.py`** to start the UI. Enter a game title to receive recommendations. Uncomment lines 145 and 146 to output files for the similarity matrix and TF-IDF weighted document-term matrix.

Alternatively, download `steamspy_data_clean.csv` and ensure it is in the file path `.Cpe646_Project/steamspy_data_clean.csv` alongside the Python files.

## Files

- `data_acquisition.py`
- `data_cleanup.py`
- `data_processing.py`
- `raw_steamspy_data.csv`
- `steamspy_data_clean.csv`

