import pandas as pd
import numpy as np

import networkx as nx
import community as community_louvain

import hdbscan
import umap
import random
from tqdm import trange

import spacy
from collections import Counter

from plotnine import *

def find_louvain_communities(edges_df):
    """
    A wrapper for the Louvain community detection algorithm from python-louvain
    :param edges_df(DataFrame): dataframe containing the graph's weighted edges
                                    mandatory fields and names:
                                    'node_1' - source
                                    'node_2' - target
                                    'proximity' - weight
    :return (DataFrame): a dataframe containing node id and community membership
    """
    edges_df['weight'] = edges_df['proximity'].map(lambda x: 0 if x < 0 else x)

    G = nx.from_pandas_edgelist(edges_df, source='node_1', target='node_2', edge_attr=['weight'])
    partitions = community_louvain.best_partition(G, resolution=1.1, random_state=42)
    out_df=pd.DataFrame.from_dict(partitions,orient='index').reset_index()
    out_df=out_df.rename(columns={0:'community'})
    return out_df

def pytorch_cos_sim(x):
    """
    A simple function to calculate similarities between two vectors provided by huggingface embedding pipelines
    Necessary for pandarallel processing
    :param x(DataFrame): single-row dataframe containing two vector pairs
                            mandatory fields and names:
                            'vector_1'
                            'vector_2'
    :return (float): cosine proximity between the two vectors
    """
    from sentence_transformers import util
    return util.pytorch_cos_sim(x['vectors_1'],x['vectors_2'])[0][0].item()

nlp = spacy.load("en_core_web_sm")

def get_lowercase_tokens(text):
    """
    Simple function to get tokens from string
    :param text(str) input string:
    :return [str]: lowercase list of tokens extracted from string
    """
    tokens=[]
    doc = nlp(text)
    for token in doc:
        if token.is_stop == False \
                and token.is_punct==False\
                and token.is_digit==False\
                and token.text!='\\n'\
                and len(token.text)>1:
            tokens.append(token.text.lower())
    return tokens


def make_violin_jitter_plot(df, x, y, color=None, fill=None, filter_column=None, filter_value=None, filter_type=None,
                            title=None, color_gradient_low=None, color_gradient_high=None, color_fill_values=None,
                            figure_size='medium', text_size=12, dpi=100):
    """
    A wrapper function to generate overlapped violin and jittered scatter plots using the plotnine package

    :param df:
    :param x:
    :param y:
    :param color:
    :param fill:
    :param filter_column:
    :param filter_value:
    :param filter_type:
    :param title:
    :param color_gradient_low:
    :param color_gradient_high:
    :param color_fill_values:
    :param figure_size:
    :param text_size:
    :return:
    """
    figure_size_mappings = {'small': 8,
                            'medium': 12,
                            'large': 16,
                            'huge': 20}

    # Make plot dataframe
    if filter_column is None:
        df = df
    elif filter_type == 'equal':
        df = df[df[filter_column] == filter_value]
    elif filter_type == 'gte':
        df = df[df[filter_column] >= filter_value]
    elif filter_type == 'lte':
        df = df[df[filter_column] <= filter_value]

    plot = ggplot(df, aes(x=x, y=y, color=color, fill=fill))
    plot += geom_violin()
    plot += geom_jitter(size=0.5, height=0, width=0.4)

    # Make plot title
    if title is None:
        pass
    else:
        plot += ggtitle(title)

    # Specify color gradient limits for plotted points
    if color_gradient_low is None:
        color_gradient_low = 'blue'
    else:
        color_gradient_low = color_gradient_low

    if color_gradient_high is None:
        color_gradient_low = 'orange'
    else:
        color_gradient_low = color_gradient_low
    plot += scale_color_gradient(low=color_gradient_low, high=color_gradient_high)

    # Specify discrete fill scale colors
    if color_fill_values is None:
        pass
    else:
        plot += scale_fill_manual(values=color_fill_values)

    plot += theme_classic()
    plot += theme(figure_size=(figure_size_mappings[figure_size], figure_size_mappings[figure_size]),
                  text=element_text(size=text_size),dpi=dpi)

    print(plot)


#HDBScan clustering  and automatic label generation functions
#shamelessly borrowed from - https://towardsdatascience.com/clustering-sentence-embeddings-to-identify-intents-in-short-text-48d22d3bf02e
def generate_clusters(message_embeddings,
                      n_neighbors,
                      n_components,
                      min_cluster_size,
                      random_state=None):
    """
    Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP
    """

    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors,
                                 n_components=n_components,
                                 metric='cosine',
                                 random_state=random_state)
                       .fit_transform(message_embeddings))

    clusters = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                               metric='euclidean',
                               cluster_selection_method='eom').fit(umap_embeddings)

    return clusters


def score_clusters(clusters, prob_threshold=0.05):
    """
    Returns the label count and cost of a given cluster supplied from running hdbscan
    """

    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold) / total_num)

    return label_count, cost


def random_search(embeddings, space, num_evals):
    """
    Randomly search hyperparameter space and limited number of times
    and return a summary of the results
    """

    results = []

    for i in trange(num_evals):
        n_neighbors = random.choice(space['n_neighbors'])
        n_components = random.choice(space['n_components'])
        min_cluster_size = random.choice(space['min_cluster_size'])

        clusters = generate_clusters(embeddings,
                                     n_neighbors=n_neighbors,
                                     n_components=n_components,
                                     min_cluster_size=min_cluster_size,
                                     random_state=42)

        label_count, cost = score_clusters(clusters, prob_threshold=0.05)

        results.append([i, n_neighbors, n_components, min_cluster_size,
                        label_count, cost])

    result_df = pd.DataFrame(results, columns=['run_id', 'n_neighbors', 'n_components',
                                               'min_cluster_size', 'label_count', 'cost'])

    return result_df.sort_values(by='cost')


from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK, space_eval


def objective(params, embeddings, label_lower, label_upper):
    """
    Objective function for hyperopt to minimize, which incorporates constraints
    on the number of clusters we want to identify
    """

    clusters = generate_clusters(embeddings,
                                 n_neighbors=params['n_neighbors'],
                                 n_components=params['n_components'],
                                 min_cluster_size=params['min_cluster_size'],
                                 random_state=params['random_state'])

    label_count, cost = score_clusters(clusters, prob_threshold=0.05)

    # 15% penalty on the cost function if outside the desired range of groups
    if (label_count < label_lower) | (label_count > label_upper):
        penalty = 0.15
    else:
        penalty = 0

    loss = cost + penalty

    return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}


def bayesian_search(embeddings, space, label_lower, label_upper, max_evals=100):
    """
    Perform bayseian search on hyperopt hyperparameter space to minimize objective function
    """

    trials = Trials()
    fmin_objective = partial(objective, embeddings=embeddings, label_lower=label_lower, label_upper=label_upper)
    best = fmin(fmin_objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    best_params = space_eval(space, best)
    print('best:')
    print(best_params)
    print(f"label count: {trials.best_trial['result']['label_count']}")

    best_clusters = generate_clusters(embeddings,
                                      n_neighbors=best_params['n_neighbors'],
                                      n_components=best_params['n_components'],
                                      min_cluster_size=best_params['min_cluster_size'],
                                      random_state=best_params['random_state'])

    return best_params, best_clusters, trials

def extract_labels(category_docs, addl_stopwords=[]):
    """
    Extract labels from documents in the same cluster by concatenating
    most common verbs, ojects, and nouns
    """

    verbs = []
    dobjs = []
    nouns = []
    adjs = []

    verb = ''
    dobj = ''
    noun1 = ''
    noun2 = ''

    # for each document, append verbs, dobs, nouns, and adjectives to
    # running lists for whole cluster
    for i in range(len(category_docs)):
        doc = nlp(category_docs[i])
        for token in doc:
            if token.is_stop == False \
                    and token.is_punct == False \
                    and token.is_digit == False \
                    and token.text != '\\n' \
                    and len(token.text) > 1\
                    and token.text not in addl_stopwords:

                if token.dep_ == 'ROOT':
                    verbs.append(token.text.lower())

                elif token.dep_ == 'dobj':
                    dobjs.append(token.lemma_.lower())

                elif token.pos_ == 'NOUN':
                    nouns.append(token.lemma_.lower())

                elif token.pos_ == 'ADJ':
                    adjs.append(token.lemma_.lower())

    # take most common words of each form
    if len(verbs) > 0:
        counter = Counter(verbs)
        verb = counter.most_common(1)[0][0]
        # verb = most_common(verbs, 1)[0][0]

    if len(dobjs) > 0:
        counter = Counter(dobjs)
        dobj = counter.most_common(1)[0][0]
        # dobj = most_common(dobjs, 1)[0][0]

    if len(nouns) > 0:
        counter = Counter(nouns)
        noun1 = counter.most_common(1)[0][0]
        # noun1 = most_common(nouns, 1)[0][0]

    if len(set(nouns)) > 1:
        counter = Counter(nouns)
        noun2 = counter.most_common(2)[1][0]
        # noun2 = most_common(nouns, 2)[1][0]

    # concatenate the most common verb-dobj-noun1-noun2 (if they exist)
    label_words = [verb, dobj]

    for word in [noun1, noun2]:
        if word not in label_words:
            label_words.append(word)

    if '' in label_words:
        label_words.remove('')

    label = '_'.join(label_words)

    return label