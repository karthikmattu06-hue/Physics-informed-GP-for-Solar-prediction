=========================================
README – IDAI610 Problem Set 3 Submission
=========================================


Author: Karthik Mattu
Course: Fundamentals of Artificial Intelligence (IDAI610)
Assignment: Problem Set 3 – Naive Bayes Text Classification


-----------------------------------------
1. Requirements
-----------------------------------------
- Python 3.8 or higher
- No external libraries required (only uses built-in modules: csv, re, math)


-----------------------------------------
2. How to Run
-----------------------------------------
1. Open a terminal or command prompt.
2. Navigate to the folder containing the PS3.py file.
3. Run the following command:
   > python3 PS3.py


This will automatically:
- Train and evaluate Naive Bayes models on both datasets
  (Movie Reviews and 20 Newsgroups)
- Print Accuracy, Macro-Precision, Macro-Recall, and Confusion Matrices
- Display smoothed and unsmoothed comparisons


-----------------------------------------
3. Modifying Input Paths
-----------------------------------------
If the datasets are located in a different folder, update the following
four variables near the bottom of PS3.py:


    reviews_train = "<path to reviews_polarity_train.csv>"
    reviews_test  = "<path to reviews_polarity_test.csv>"
    news_train    = "<path to newsgroup_train.csv>"
    news_test     = "<path to newsgroup_test.csv>"


Example (Windows):
    reviews_train = "C:/Users/Student/Downloads/reviews_polarity_train.csv"


Example (Mac/Linux):
    reviews_train = "/Users/username/Documents/IDAI610/Data/reviews_polarity_train.csv"


-----------------------------------------
4. Expected Output
-----------------------------------------
The script prints four evaluation sections:
- Movie Reviews (unsmoothed)
- Movie Reviews (Laplace smoothed)
- 20 Newsgroups (unsmoothed)
- 20 Newsgroups (Laplace + stopword removal + pruning)


Typical accuracy values:
    Reviews: 0.39 → 0.80 after smoothing
    20NG:    0.18 → 0.72 after smoothing




-----------------------------------------
End of README
-----------------------------------------