import csv
import re
import nltk
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
from spacy.matcher import Matcher
from spacy.tokens import Span
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

pd.set_option('display.max_colwidth', 200)
candidate_sentences = pd.read_csv("clean_abraham.csv")
shape = candidate_sentences.shape
sample = candidate_sentences['sentence'].sample(5)


doc = nlp("the drawdown process is governed by astm standard d823")


for tok in doc:
  print(tok.text, "...", tok.dep_)

text_list = """
Abraham Lincoln (/ˈlɪŋkən/; February 12, 1809 – April 15, 1865) was an American lawyer and statesman who served as the 16th president of the United States from 1861 until his assassination in 1865. Lincoln led the nation through the American Civil War and succeeded in preserving the Union, abolishing slavery, bolstering the federal government, and modernizing the U.S. economy.

Lincoln was born into poverty in a log cabin in Kentucky and was raised on the frontier primarily in Indiana. He was self-educated and became a lawyer, Whig Party leader, Illinois state legislator, and U.S. Congressman from Illinois. In 1849, he returned to his law practice but became vexed by the opening of additional lands to slavery as a result of the Kansas–Nebraska Act. He reentered politics in 1854, becoming a leader in the new Republican Party, and he reached a national audience in the 1858 debates against Stephen Douglas. Lincoln ran for President in 1860, sweeping the North in victory. Pro-slavery elements in the South equated his success with the North's rejection of their right to practice slavery, and southern states began seceding from the Union. To secure its independence, the new Confederate States fired on Fort Sumter, a U.S. fort in the South, and Lincoln called up forces to suppress the rebellion and restore the Union.
"""

text_tokens = word_tokenize(text_list)

tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

filtered_sentence = (" ").join(tokens_without_sw)
print(filtered_sentence)





# doc = nlp(filtered_sentence)
# sent = list(map(lambda x: str(x), doc.sents))
# a =2
#
#
#
# with open('clean_abraham.csv', 'w', ) as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     for word in sent:
#         wr.writerow([word])

def get_entities(sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence

    prefix = ""
    modifier = ""

    token_type = ""
    #############################################################

    for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

                ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            ## chunk 5
            # update variables


            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
            token_type = tok.ent_type_
    #############################################################

    return [ent1.strip(), ent2.strip(), token_type]
ttt = get_entities("the film had 200 patents")
print(ttt)

entity_pairs = []
g = candidate_sentences["sentence"].index




for i in tqdm(candidate_sentences['sentence']):
    try:
        # f = candidate_sentences.index[candidate_sentences['sentence'] == i]
        # print(f)
        entity_pairs.append(get_entities(i))
    except Exception as e:
        print(i)
        print(e)
        continue

pairs = entity_pairs
def get_relation(sent):
    try:
          doc = nlp(sent)

          # Matcher class object
          matcher = Matcher(nlp.vocab)

          #define the pattern
          pattern = [{'DEP': 'ROOT'},
                     {'DEP':  'prep', 'OP': "?"},
                     {'DEP': 'agent', 'OP': "?"},
                     {'POS': 'ADJ', 'OP': "?"}]

          matcher.add("matching_1", [pattern], on_match=None)

          matches = matcher(doc)

          k = len(matches) - 1

          span = doc[matches[k][1]:matches[k][2]]

          return(span.text)
    except Exception as e:

        print(e)

ttt1 = get_relation(sent="the film had 200 patents")

relations = []
for i in tqdm(candidate_sentences['sentence']):
    try:
        relations.append(get_relation(i))
    except Exception:
        continue
print(relations)
source = [i[0] for i in entity_pairs]

# extract object
target = [i[1] for i in entity_pairs]
a =[]
tag = [i[2] for i in entity_pairs]
entity = []
kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': relations, 'tag': tag})
# create a directed-graph from a dataframe


final_table = pd.DataFrame({"source":source, "target":target, "tag":tag})
G = nx.from_pandas_edgelist(kg_df, "source", "target",
                          edge_attr=True, create_using=nx.MultiDiGraph())

plt.switch_backend('TkAgg')
plt.figure(figsize=(12, 12))

pos = nx.spring_layout(G)
nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
plt.show()

# G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="composed by"], "source", "target",
#                           edge_attr=True, create_using=nx.MultiDiGraph())
#
# plt.figure(figsize=(12,12))
# pos = nx.spring_layout(G, k=0.5) # k regulates the distance between nodes
# nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
# plt.show()

# G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="written by"], "source", "target",
#                           edge_attr=True, create_using=nx.MultiDiGraph())
#
# plt.figure(figsize=(12,12))
# pos = nx.spring_layout(G, k = 0.5)
# nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
# plt.show()
#
# G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="released in"], "source", "target",
#                           edge_attr=True, create_using=nx.MultiDiGraph())
#
# plt.figure(figsize=(12,12))
# pos = nx.spring_layout(G, k=0.5)
# nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
# plt.show()


labels = dict(zip(list(zip("source", "target")),
                  kg_df['edge'].tolist()))

def prepare_df(text_list):
    try:
        doc = nlp(text_list)
        df = pd.DataFrame()
        for sent in list(doc.sents):
            sub, obj = get_entities(str(sent))
            relation = get_relation(str(sent))
            if ((len(relation)>2) & (len(sub)>2) &(len(obj)>2)):
                df= df.append({'subject': sub, 'relation': relation, 'object': obj}, ignore_index=True)
    except Exception:
        print(Exception)
    return df

df = prepare_df(text_list)
df.head()
def draw_kg(pairs, c1='red', c2='blue', c3='orange'):
    k_graph = nx.from_pandas_edgelist(pairs, 'subject', 'object',
                                      create_using=nx.MultiDiGraph())

    node_deg = nx.degree(k_graph)
    layout = nx.spring_layout(k_graph, k=0.15, iterations=20)
    plt.figure(num=None, figsize=(50, 40), dpi=80)
    nx.draw_networkx(
        k_graph,
        node_size=[int(deg[1]) * 500 for deg in node_deg],
        arrowsize=20,
        linewidths=1.5,
        pos=layout,
        edge_color=c1,
        edgecolors=c2,
        node_color=c3,
    )
    labels = dict(zip(list(zip(pairs.subject, pairs.object, )),
                      pairs['relation'].tolist()))
    nx.draw_networkx_edge_labels(k_graph, pos=layout, edge_labels=labels,
                                 font_color='red')
    plt.axis('off')
    plt.show()

draw_kg(df)

