# Fake News Detection

This is a project for detecting whether a news article is real or fake using NLP and logistic regression.

## Data

Data used for training and testing is [fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by clmentbisaillon from Kaggle. Licence: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The dataset was initially committed but later removed to keep the repository lightweight. You can download it from the Kaggle page.

## Approach

1. Label, combine and randomise dataset
2. Combine title and text
3. Clean and preprocess text
4. Divide into training and test sets
5. Calculate TF-IDF with scikit-learn TfidfVectorizer
6. Fit a logistic regression model and predict
7. Calculate accuracy as well as precision, recall and F1-score

## Tools

-Python
-pandas, scikit-learn, nltk
-JupyterLab

## Setup

1. Clone the repository or download the files.

2. Create and activate a Conda environment:
```
conda create -n fakenews python=3.13
conda activate fakenews
```
3. Install required packages:
```
pip install -r requirements.txt
```
4. Download the data from Kaggle, unzip it, and place the CSV files (Fake.csv and True.csv) in a folder named data/ at the root of the project.

5. Launch JupyterLab:
```
jupyter lab
```
6. Register environment with Jupyter
```
pip install ipykernel
python -m ipykernel install --user --name=fakenews
```
7. Open fake_news_detection.ipynb and run all cells step-by-step.
