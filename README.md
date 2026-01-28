# Movie Review Sentiment Analysis

This repository showcases one of my first projects focused on learning transfer learning in natural language processing. The goal of this mini-project was to fine-tune a transformer model capable of performing sentiment analysis on movie reviews. I used DistilBERT, a lightweight distilled version of BERT from Hugging Face, and fine-tuned it using the IMDb Movie Review dataset (50,000 labeled reviews) accessed through [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download). Rather than building a model from scratch, this project focused on understanding how pre-trained transformer models can be fine-tune to different tasks other than what the model was trained for.

The project was developed using the Hugging Face Transformers library, which handled most of the the model architecture, tokenization, and training workflow. My work primarily involved preparing the dataset and configuring the training pipeline. Preprocessing included renaming dataset columns to match the format expected by the Trainer API, splitting the data into training and test sets using an 80/20 split, and tokenizing the text with appropriate truncation and padding to ensure compatibility with DistilBERT’s input requirements. Training was configured using the TrainingArguments class, where I specified hyperparameters such as learning rate, batch size, and number of epochs. The model was then fine-tuned using the Trainer class, which streamlined both training and evaluation.

After training for two epochs, the model achieved a training loss of 0.2636. Evaluation metrics were computed using a custom compute_metrics function, which returned a validation loss of 0.2497, an accuracy of 89.75%, a precision of 89.16%, and an F1-score of 89.89%. These results demonstrate strong performance for a relatively simple fine-tuning setup. This project helped me understand how transfer learning enables powerful NLP systems to be built efficiently by adapting pre-trained models rather than training deep neural networks from scratch.

The entire workflow can be viewed in this [jupyter notebook](https://github.com/acosio14/movie-review-sentiment/blob/main/movie_sentiment.ipynb)

What I Learned:
- How to prepare and format text datasets for transformer models
- How to use Hugging Face tokenizers and pre-trained model classes
- How to configure training workflows using TrainingArguments and the Trainer API
- Applying transfer learning instead of training deep learning models from scratch