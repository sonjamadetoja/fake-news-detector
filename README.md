# Fake News Detection

This project develops a Fake News classifier using Logistic Regression and TF-IDF, achieving 98% accuracy after rigorous data auditing. By redacting source-specific metadata (e.g. 'Reuters'), I forced the model to distinguish between institutional reporting styles and emotional/hyperbolic narratives. The final model was validated against 2025 news events, successfully identifying both modern hard news and satirical content.

## Data

Data used for training and testing is [fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by clmentbisaillon from Kaggle. Licence: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The dataset was initially committed but later removed to keep the repository lightweight. You can download it from the Kaggle page.

## Approach

1. Label, combine, deduplicate and randomise dataset
2. Combine title and text
3. Clean and preprocess text
4. Divide into training and test sets
5. Construct a pipeline with TF-IDF feature extraction and logistic regression to prevent data leakage
6. Tune the regularization strength of the logistic regression model using cross-validation on the training set
7. Train the final model on the full training data using the selected hyperparameter
8. Evaluate performance on the held-out test set using accuracy, precision, recall, and F1-score
9. Audit the model: feature importance analysis, data redaction experiment, confusion matrix, identifying mispredicted rows, and external validation: real-world examples
10. Analysis

## Tuning

I tuned the hyperparameter C (the inverse of regularization strength) using GridSearchCV with 5-fold cross-validation. The optimal value was found to be C=10, which is relatively high, and indicates a lower regularization penalty. This is necessary for complex linguistic and stylistic patterns of the text as stronger regularization may over-simplify the model and reduce its ability to detect subtle nuances in the language. F1 is used as metric for cross-validation, because it's the harmonic mean of precision and recall, so it forces the model to perform well on both classes simultaneously.

## Analysis

| Model Version | Accuracy | F1-score | Top "Real" Predictors | Top "Fake" Predictors |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | 99% | 99% | reuter, said, washington | via, video, image |
| Redacted | 98% | 98% | spokesman, presidential, statement | getty, gop, com |

The first model received a suspiciously high accuracy. Inspecting the important features revealed that there was data leakage: the model relied on source-specific markers such as '*Reuters*' for predicting instead of news content. After redacting words like this for the second model, the model started to focus on stylistic cues, identifying *official*/*institutional* language for real news and *hyperbolic*/*emotional* or *webscraping* language for fake news (e.g. *Getty*). The model fails in prediction specifically, when a fake news adopts a formal, data-heavy style or when real news has very emotional content. Finally, the redacted model was validated with two real-world examples: a real BBC news item and an Onion satire piece. The model classified both items correctly. Still the 98% accuracy suggests that residual data leakage likely remains.

## Key takeaway

High accuracy isn't always a good thing; it can be a sign of data leakage. Cleaning the data is equally important to tuning model parameters.

## Tools

- Python
- pandas, scikit-learn, nltk, matplotlib
- JupyterLab

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

