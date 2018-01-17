import pandas as pd
import matplotlib as m
import matplotlib.pyplot as plt
import urllib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pathlib import Path
import numpy

#download comments and annotations
ANNOTATED_COMMENTS_URL = 'https://ndownloader.figshare.com/files/7554634' 
ANNOTATIONS_URL = 'https://ndownloader.figshare.com/files/7554637' 

def download_file(url, fname):
    urllib.request.urlretrieve(url, fname)

comment_file = Path("attack_annotated_comments.tsv")
annotations_file = Path("attack_annotations.tsv")

if comment_file.is_file():
    #do nothing
    print("File {} is already downloaded.".format(comment_file))
else:
    print("Downloading {} please wait.".format(comment_file))
    download_file(ANNOTATED_COMMENTS_URL, 'attack_annotated_comments.tsv')
if annotations_file.is_file():
    #do nothing
    print("File {} is already downloaded.".format(annotations_file))
else:
    print("Downloading {} please wait.".format(annotations_file))
    download_file(ANNOTATIONS_URL, 'attack_annotations.tsv')

# converting tsv files into Pandas Dataframe objects
comments = pd.read_csv('attack_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations = pd.read_csv('attack_annotations.tsv', sep = '\t')

print(len(annotations['rev_id'].unique()))

print(annotations)

# labels a comment as an attack if the majority of annotaters did so
labels = annotations.groupby('rev_id')['attack'].mean() > 0.5
print("#"*80)
print("Labels")
print(labels)
# join labels and comments
comments['attack'] = labels
print("#"*80)
print("Comments")
print("#"*80)
print(comments)
print("#"*80)

# remove newline and tab tokes
comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))

#print(comments.query('attack')['comment'].head())


ratings = pd.read_csv('data/toxicity_annotations.tsv', sep = '\t')
ratings = ratings.sort_values(['worker_id'], ascending=True)
grouped_by_user = ratings.groupby("worker_id")


# Creating a new empty dataframe to hold users and their biases.
users = ratings.worker_id.unique()
columns = ['toxicity_total', 'score_total']
toxicity_totals = pd.DataFrame(index=users, columns=columns)
plot_total_tox = grouped_by_user['toxicity_score'].sum().hist(bins=100)
plt.xlabel('Sum of Toxicity Ratings (-2,-1,0,1,2)')
plt.ylabel('Number of Reviewers')
plt.title('Total Toxicity')
plt.show()

# Creating a plot of the total number of toxic and neutral comments
# as rated by the unanimous reviewers
"""
plot_num_tox = grouped_by_user['toxicity'].apply(lambda x: x == 1).hist(bins=100)
plt.show()
"""
plot_num_tox1 = grouped_by_user['toxicity'].sum().hist(bins=100)

def count_ones_zeros(group):
    zeros = 0
    ones  = 0
    for each in group:
        if each == 0:
            zeros += 1
        elif each == 1:
            ones += 1
    return zeros, ones

def count_zeros(group):
    zeros = 0
    for each in group:
        if each == 0:
            zeros += 1
    return zeros

def count_ones(group):
    ones = 0
    for each in group:
        if each == 1:
            ones += 1
    return ones
plt.clf()
plt.title("Counts of Toxic vs Non-Toxic by Reviewer")
plot_num_tox2 = grouped_by_user.agg({'toxicity': count_zeros})
plot_num_tox3 = grouped_by_user.agg({'toxicity': count_ones})
bins = 430 
plt.hist(plot_num_tox2['toxicity'], bins=bins, alpha=0.5, label='non_toxic')
plt.hist(plot_num_tox3['toxicity'], bins=bins, alpha=0.5, label= 'toxic')
plt.legend(loc='upper center')
plt.xlabel("Number of Comments Rated as Toxic/Non-Toxic")
plt.ylabel("Number of Reviewers")

plt.show()
print(plot_num_tox2, plot_num_tox3)
# prints all items in groupedby user
"""
for key, item in plot_num_tox2:
    print(item)
   """

