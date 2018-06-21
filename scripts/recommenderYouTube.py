from __future__ import print_function
from __future__ import division
import itertools
import collections
import math
import numpy as np
from scipy.spatial.distance import cosine
import os
import random
import tensorflow as tf
import re
import json
import pickle
import sys
import time

from itertools import chain


# import zipfile
# from matplotlib import pylab
# from six.moves import range
# from six.moves.urllib.request import urlretrieve
# from sklearn.manifold import TSNE


# ID devono essere dal piu` frequente al meno frequente (cfr https://www.tensorflow.org/api_docs/python/tf/nn/nce_loss)
def read_data(user_file, item_file, action_file,
              max_actions=0, actions_cutoff=100, sort_actions=False,
              separator=',', header=False):
    '''Read the files and builds the dicts.

        :param str user_file: id, feature1, feature2 etc
        :param str item_file: id, category, description(=words comma-separated)
        :param str action_file: user_id, item_id, timestamp (NB all actions by the same users TOGETHER)
        :param int max_actions: max number of actions to be read
        :param int actions_cutoff: max actions to be stored _per user_ (some Anobii users are actually bookshop with 1000s of books)
        :param bool sort_actions: set to True if actions are not timestamp sorted
        :param str separator: for the csv ("|")
        :return Tuple(Dict, Dict, Dict): users = Uid -> [P1, ..., Pn],
                                         items = Iid -> [I_0, ..., I_m],
                                         obs = Uid -> [(Iid, Timestamp, Action), (), ..., ()]

    '''
    # dicts
    users = {}  # Uid -> [P1, ..., Pn]
    items = {}  # Iid -> [I_0, ..., I_m]
    obs = {}  # Uid -> [(Timestamp, Iid), (), ..., ()]
    print("INFO: Reading user file " + user_file)
    with open(user_file) as f:
        if header:
            f.readline()  # header
        for line in f.readlines():
            v = line.strip("\n").split(separator)
            try:
                if len(v) > 1:
                    users[v[0]] = v[1:]
                else:  # for the moment we only have the id
                    users[v[0]] = []
            except:
                continue
    print("INFO: Done with users")

    print("INFO: Reading item file " + item_file)
    # item[book_id] = [title in word_id, author_id, lang_id, time]
    with open(item_file) as f:
        if header:
            f.readline()
        for line in f.readlines():
            v = line.strip("\n").split(separator)
            items[v[0]] = v[1:]
    print("INFO: Done with items")

    # how much is onepc (1%)?
    if max_actions == 0:
        with open(action_file) as f:
            onepc = int(sum(1 for line in f) / 10)
    else:
        onepc = int(max_actions / 10)

    idx = 0
    print("INFO: Reading action file " + action_file)
    with open(action_file) as f:
        if header:
            titles = dict([(n, i) for i, n in enumerate(f.readline().strip("\n").split(separator))])
        u = ''
        for line in f.readlines():
            idx += 1
            if idx % onepc == 0:
                pc = idx // onepc
                print("%s0 percent actions analyzed (%s)" % (pc, idx))
            v = line.strip("\n").split(separator)
            if u != '' and u != v[0]:  # it's a new user, then...
                # sort previous array of observations if it's not a
                # user with huge number of transactions (i.e. useless, like
                # librarians, bookshop etc)
                o = obs.get(u)
                if o is not None:  # it happens for users who have obs with too rare items
                    if len(o) < actions_cutoff or actions_cutoff == 0:
                        if sort_actions:
                            obs[u] = sorted(o)
                        else:
                            obs[u] = o
                    else:
                        del (obs[u])
            u = str(v[0])
            if header:
                obs.setdefault(str(v[titles['user']]), []).append((int(v[titles['timestamp']]), str(v[titles['item']])))
            else:
                obs.setdefault(str(v[0]), []).append((int(v[2]), str(v[1])))
            if max_actions != 0 and idx >= max_actions:
                break

    return users, items, obs


# Fa una lista
def build_obs_dataset_list(observations, roll_over=10,
                           decay_type=None, decay_period=86400 * 365,
                           max_item=1, n_users=0):
    '''

    :par observations: '73462': [('1379808000', '30509'),  ('1382054400', '21328'),  ('1382054400', '46409', '3')]
    :par roll_over: takes as last trainer the past roll_over items (eventually weighted with the previous max_items items preceeding the last_trainer)
    :par str decay_type: in ['exponential', 'inverse_sqrt', 'inverse']. inverse sqrt should be the best...
    :par str decay_period: unit of time in sec for the decay (eg 86400*7 for weeks)
    :par int max_items: number of previous items to be considered
    :par int n_obs: number of obs to analyze
    :return
    [(UID1, LABEL_ITEM_ID, [(TRAINER_ITEM_ID, WEIGHT), (TRAINER_ITEM_ID, WEIGHT), ...]),
     (UID1, LABEL_ITEM_ID, [(TRAINER_ITEM_ID, WEIGHT), (TRAINER_ITEM_ID, WEIGHT), ....]),
     ...
     (UID2, LABEL_ITEM_ID, [(TRAINER_ITEM_ID, WEIGHT), (TRAINER_ITEM_ID, WEIGHT), ...]),
     ....
    ]

    '''
    if decay_type == 'exponential':
        def fdecay(action_time, label_time):
            return math.exp((-action_time + label_time) / decay_period)
    elif decay_type == 'inverse_sqrt':
        def fdecay(action_time, label_time):
            #            print("action_time, label_time decay", action_time, label_time, decay_period)
            if action_time == label_time:
                return 1.0
            else:
                return 1.0 / math.sqrt((-action_time + label_time) / decay_period)
    elif decay_type == 'inverse':
        def fdecay(action_time, label_time):
            if action_time == label_time:
                return 1.0
            else:
                return 1.0 / ((-action_time + label_time) / decay_period)
    elif decay_type == None:
        def fdecay(action_time, label_time):
            return 1.0
    else:
        raise ValueError("decay_type must be 'exponential', 'inverse_sqrt', 'inverse' or None")

    result = []
    if n_users == 0:
        n_users = len(observations)
    onepc = int(n_users / 10)
    for u in list(observations.keys())[:n_users]:
        # print("\n\n================================\nuser ", u)
        pairs = []
        for index, action in enumerate(observations[u]):
            if index == 0:
                continue  # ([1:] doesn't work) bc first item has no trainers
            # print("\tindex action", index, action)
            label_time = int(action[0])
            label = int(action[1])
            for last_trainer in range(max(0, index - roll_over), index):
                trainers = []
                # print("\t\tlast_trainer", observations[u][last_trainer])
                for trainer_time, trainer in observations[u][max(0, last_trainer - max_item + 1):last_trainer + 1]:
                    # print("\t\t\ttrainer_time, trainer", trainer_time, trainer)
                    weight = fdecay(int(trainer_time), label_time)
                    trainers.append((int(trainer), weight))
                result.append((int(u), label, trainers))
    random.shuffle(result)
    return result


data_index = 0


def generate_batch_constant(obs_dataset_list, item_titles, data_index,
                            batch_size, max_words, max_items):
    '''
    Constant batch size

    A dictionary with all matrices and indexes.

    IN
    obs_dataset: [[user, label, item1, item2], [...], ...]
    item_titles: titles of books with word_ids
    max_words: max number of words used in the title
    max_items: max number of past actioned items used for the training
    batch_size: number of users to be analyzed in the same batch. #TODO: obs_dataset could repeat the user for each sublist.

    OUT

    A 2-tuple:

    [0] type(dict):
        "one_hot_words_values": [1st word in 1st batch, 2nd in 1st, ..., max_words'th in batch_size'th]
        "one_hot_words_weights": [weight of 1st word in 1st batch, 2nd in 1st, ..., max_words'th in batch_size'th]
        "words_indices": [[0,0], [0, 1] .. [0, batch_size], [1, 0] ... [max_words, batch_size]]
        "one_hot_items_values": [1st item in 1st batch, 2nd in 1st, ..., max_words'th in batch_size'th]
        "one_hot_items_weights": as for words but with "items"
        "items_indices": as for words but with "items"
        # "profiles": a 3d vector with each element [-1, +1] made of [gender, age, job]
        "labels": the items bought by the user

     [1]: type(int): new data_index
    '''

    one_hot_words_values = []
    one_hot_words_weights = []

    one_hot_items_values = []
    one_hot_items_weights = []

    profiles = []
    labels = []
    total_obs = 0
    batch_idx = 0
    idx = (data_index + batch_idx) % len(obs_dataset_list)
    n_loop = 0
    while (batch_idx < batch_size):
        n_loop += 1
        if n_loop > 100 and batch_idx == 0:
            msg = "generate_batch_constant has empty loops " + str(n_loop) + ", batch_idx: " + str(batch_idx)
            raise Exception(msg)
        usr_lab_obs = obs_dataset_list[idx]
        user = usr_lab_obs[0]
        #        print("USER: ", user)
        label = usr_lab_obs[1]
        trainers = usr_lab_obs[2]

        # get one-hot value of all words in the training items
        #        if max_words > 0:
        word_value = [int(word) for item_weight in trainers
                      for word in item_titles.get(str(item_weight[0]), [])][:max_words]
        word_weight = [float(item_weight[1]) for item_weight in trainers
                       for word in item_titles.get(str(item_weight[0]), [])][:max_words]

        word_value.extend([0] * (max_words - len(word_value)))
        #        print('word_value ->', word_value)
        word_weight.extend([0] * (max_words - len(word_weight)))
        #        print('word_weight ->', word_weight)

        item_value = [item_weight[0] for item_weight in trainers][:max_items]
        item_weight = [item_weight[1] for item_weight in trainers][:max_items]
        item_value.extend([0] * (max_items - len(item_value)))
        #        print('item_value ->', item_value)
        item_weight.extend([0] * (max_items - len(item_weight)))
        #        print('item_weight ->', item_weight)

        usr_str = str(user)
        profile = users[usr_str]
        # item "0" is not valid. #TODO a better method...
        if not synthetic and (sum(item_value) == 0 or sum(item_weight) == 0 or (
                (sum(word_value) == 0 or sum(word_weight) == 0) and max_words > 0)):
            # if there are no words but max_words>0, go on reading w/o add to batch_idx
            idx = (idx + 1) % len(obs_dataset_list)
            # print("Something wrong with user %s at index %s and batch_idx %s" % (user, idx, batch_idx))
        else:
            #             if max_words > 0:
            #                 one_hot_words_values += word_value
            #                 one_hot_words_weights += word_weight
            one_hot_items_values += item_value
            one_hot_items_weights += item_weight
            profiles.append(profile)
#            labels.append([label])  # each label is in reality a one-hot representation to be embedded
            labels.append(label)  # each label is in reality a one-hot representation to be embedded
            batch_idx += 1
            idx = (data_index + batch_idx) % len(obs_dataset_list)
    #            print(">>>>> GOOD ", user, obs, idx, batch_idx)

    #     print('one_hot_words_values  ', one_hot_words_values)
    #     print('one_hot_words_weights  ', one_hot_words_weights)
    #     print('LENGTH one_hot_words_values  ', len(one_hot_words_values))
    #     print('LENGTH one_hot_words_weights  ', len(one_hot_words_weights))
    #     print('one_hot_items_values  ', one_hot_items_values)
    #     print('LENGTH one_hot_items_values  ',len(one_hot_items_values))
    #     print('profiles ', profiles)

    if max_words > 0:
        return {'one_hot_words_values': np.array(one_hot_words_values, dtype=np.float32),
                'one_hot_words_weights': np.array(one_hot_words_weights, dtype=np.float32),
                'one_hot_items_values': np.array(one_hot_items_values, dtype=np.float32),
                'one_hot_items_weights': np.array(one_hot_items_weights, dtype=np.float32),
                'profiles': np.array(profiles, dtype=np.float32),
                'labels': np.array(labels, dtype=np.float32)
                }, (data_index + batch_size) % len(obs_dataset_list)
    else:
        return {'one_hot_items_values': np.array(one_hot_items_values, dtype=np.float32),
                'one_hot_items_weights': np.array(one_hot_items_weights, dtype=np.float32),
                'profiles': np.array(profiles, dtype=np.float32),
                'labels': np.array(labels, dtype=np.float32)
                }, (data_index + batch_size) % len(obs_dataset_list)


def embedding_layer(x, embedding_shape):
    with tf.variable_scope("embedding"):
        embedding_init = tf.random_uniform(embedding_shape, -1.0, 1.0)
        embedding_matrix = tf.get_variable("E", initializer=embedding_init)
        return tf.nn.embedding_lookup(embedding_matrix, x), embedding_matrix


def noise_contrastive_loss(embedding_lookup, weight_shape, bias_shape, y):
    with tf.variable_scope("nce"):
        nce_weight_init = tf.truncated_normal(weight_shape, stddev=1.0 / (weight_shape[1]) ** 0.5)
        nce_bias_init = tf.zeros(bias_shape)
        nce_W = tf.get_variable("W", initializer=nce_weight_init)
        nce_b = tf.get_variable("b", initializer=nce_bias_init)

        total_loss = tf.nn.nce_loss(nce_W, nce_b, embedding_lookup, y, neg_size, data.vocabulary_size)
        return tf.reduce_mean(total_loss)


def training(cost, global_step):
    with tf.variable_scope("training"):
        summary_op = tf.scalar_summary("cost", cost)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(cost, global_step=global_step)
        return train_op, summary_op


def validation(embedding_matrix, x_val):
    norm = tf.reduce_sum(embedding_matrix ** 2, 1, keep_dims=True) ** 0.5
    normalized = embedding_matrix / norm
    val_embeddings = tf.nn.embedding_lookup(normalized, x_val)
    cosine_similarity = tf.matmul(val_embeddings, normalized, transpose_b=True)
    return normalized, cosine_similarity


def get_similar_items(similar_file: str):
    '''Extract similar items from file

    :param similar_file: filename containing similar item for each item
    :return: dict [item] = list of similar items
    '''

    sim_items = dict()

    with open(similar_file, 'r') as f:
        for line in f:
            # print(line)

            item_patt = re.compile(r"^\d+")
            item_id = item_patt.match(line).group()

            sim_item_patt = re.compile(r"\((\d+)[,]")
            sim_item_id = sim_item_patt.findall(line)

            # print(item_id, sim_item_id)

            sim_items[int(item_id)] = list(map(int, sim_item_id))

    return sim_items


def match_similar_items(item: int, gen_sim_items: list, expected_sim_items_container: list) -> int:
    ''' Match the generated similar items with those expected

    :param item: item for which similar item are generated
    :param gen_sim_items: item generated from the trained embeddings
    :param expected_sim_items_container: expected similar items generated by LLR
    return number of matches
    '''

    expected_sim_items = expected_sim_items_container[item]
    matches = set(gen_sim_items).intersection(set(expected_sim_items))

    return len(matches)


class ConfigParams:

    fields_to_search = ['batch_size', 'items_embedding_size', 'learning_rate']

    def __init__(self, config_file='config'):
        self.set_default_vals()
        if config_file:
            fields_strval = self.parse_config(config_file)

            for field, strval in fields_strval.items():
                if field == 'batch_size':
                    self.batch_size = int(strval)
                elif field == 'items_embedding_size':
                    self.items_embedding_size = int(strval)
                elif field == 'learning_rate':
                    self.learning_rate = float(strval)

    def set_default_vals(self):
        self.batch_size = 128
        self.items_embedding_size = 10
        self.learning_rate = 0.1

    def parse_config(self, filename):

        with open(filename, "r") as config_file:
            fields_found = [False for i in range(len(self.fields_to_search))]
            fields_strval = {}
            for line in config_file:
                line = re.sub(r'\s+', '', line.lower())
                for i, field in enumerate(self.fields_to_search):
                    if re.match(r'^{}:'.format(re.escape(field)), line):
                        _, fields_strval[field] = line.split(':')
                        fields_found[i] = True

        for i, flag in enumerate(fields_found):
            if not flag:
                print('{} field not found. Using default value'.format(self.fields_to_search[i]))

        return fields_strval

    def __str__(self):
        return str(self.batch_size) + " " + str(self.items_embedding_size) + " " + str(self.learning_rate)

if __name__ == "__main__":

    if len(sys.argv) == 2:
        params = ConfigParams(sys.argv[1])
    else:
        params = ConfigParams()

    print(params)

    batch_size = params.batch_size
    items_embedding_size = params.items_embedding_size
    learning_rate = params.learning_rate


    synthetic = True

    if synthetic:
        sort_actions = False
        user_file = 'synthetic/us.csv'
        item_file = 'synthetic/is.csv'
        action_file = 'synthetic/os.csv'
        similar_file = 'synthetic/similar_items.csv'
        actions_cutoff = 0
        max_actions = 0
        separator = "\t"
        header = True

    users, items, obs = read_data(user_file=user_file, item_file=item_file,
                                  action_file=action_file, max_actions=max_actions,
                                  actions_cutoff=actions_cutoff, header=header, separator=separator)
    print('# users %d' % len(users))
    print('# items %d' % len(items))
    print('# obs %d' % len(obs))

    print("N users: ", len(users))
    print("max user_id: ", max([int(i) for i in users.keys()]))
    print("N items: ", len(items))
    print("max item_id: ", max([int(i) for i in items.keys()]))

    similar_item = get_similar_items(similar_file=similar_file)

    # Clean obs: take out users who bought very little
    clean_obs = {}
    idx = 0
    for u in obs.keys():
        idx += 1
        if idx % 50000 == 0:
            print("done ", idx)
        if len(obs[u]) > 2:
            clean_obs[u] = obs[u]

    len(clean_obs)

    if synthetic:  # let's start with just one item as input and the next bought as label
        roll_over = 1,
        decay_type = None,  # 'inverse_sqrt',
        decay_period = 86400 * 3650,
        max_item = 1,
        n_users = 0  # analyze all users

    obs_dataset_list = build_obs_dataset_list(observations=clean_obs,
                                              roll_over=10,
                                              decay_type=None,  # 'inverse_sqrt',
                                              decay_period=86400 * 3650,
                                              max_item=1,
                                              n_users=n_users)

    obs_dataset_list[2]

    print(len(obs_dataset_list))

    '''[(242438400, '15494'), 
     (271814400, '123'), 
     (468806400, '112043'), 
     (1370390400, '279'), 
     (1669766400, '35')]
    '''
    [o for o in obs_dataset_list if str(o[0]) == '168088']
    # [o for o in obs_dataset_list if str(o[0]) == '11339']

    if synthetic:
        item_titles = dict((i, []) for i in items)

    if synthetic:
        assert (item_titles['2'] == [])

    dataset = obs_dataset_list
    # test:
    # dataset = test_obs_dataset_list

    max_words = 0
    max_items = 1
    #batch_size = 10
    test_new_data_index = 40
    tbatch, test_new_data_index = generate_batch_constant(obs_dataset_list=dataset,
                                                          item_titles=item_titles,
                                                          data_index=test_new_data_index,
                                                          batch_size=batch_size,
                                                          max_words=max_words,
                                                          max_items=max_items)

    print("Len of the dataset: ", len(dataset))
    print("New data index: ", test_new_data_index)
    print("max word * batch size: ", max_words * batch_size)
    try:
        print("Same number of events for words and items: ",
              len(tbatch["one_hot_words_values"]) / max_words == len(tbatch["one_hot_items_values"]) / max_items)
        print("Batches of words: ", (tbatch["one_hot_words_values"]))
        print("Batches of words' weights: ", (tbatch["one_hot_words_weights"]))
    except:
        pass

    print("Batches of items: ", (tbatch["one_hot_items_values"]))
    print("Batches of items' weights: ", (tbatch["one_hot_items_weights"]))
    print("Batches of profiles: ", len(tbatch["profiles"]))
    print("Batches of labels: ", (tbatch["labels"]))

    words_embedding_size = 0
    #items_embedding_size = params['items_embedding_size']
    n_profile_features = 0 # 10 is the number of feature per user
    relu_dimension = items_embedding_size + int((n_profile_features + words_embedding_size) / 2)
    num_sampled = 8   # number of classes to randomly sample per batch
    if synthetic:
        max_words = 0

    max_items = 1

    # For testing
    from collections import Counter

    if synthetic:
        valid_size = 10
        valid_window = 1000
        valid_examples = list(range(10))

    words = set()

    from itertools import chain

    max_item_id = max([int(i) for i in
                       items.keys()]) + 1  # different from item_size (some item_id are never used, so the biggest ID is bigger than len(items))
    try:
        vocabulary_size = max(words)
    except ValueError:
        vocabulary_size = 0

    words_indices = np.array([[i, j] for i in range(0, batch_size) for j in range(0, max_words)], dtype=np.int64)
    items_indices = np.array([[i, j] for i in range(0, batch_size) for j in range(0, max_items)], dtype=np.int64)

    print("max item id", max_item_id)
    print("number of features per user's profile", n_profile_features)

    graph = tf.Graph()
    with graph.as_default(), tf.device("/gpu:0"):

        # Input data.
        # input_dimensions = max_words + max_items + n_profile_features
        # going to be the input of embedding_lookup_sparse for words and items
        words_train_input = tf.sparse_placeholder(tf.int64, shape=[batch_size, max_words])
        words_weights_train_input = tf.sparse_placeholder(tf.float32, shape=[batch_size, max_words])

        items_train_input = tf.sparse_placeholder(tf.int64, shape=[batch_size, max_items])
        items_weights_train_input = tf.sparse_placeholder(tf.float32, shape=[batch_size, max_items])

        if n_profile_features > 0:
            profile_train_input = tf.placeholder(tf.float32, shape=[batch_size, n_profile_features])

        train_labels = tf.placeholder(tf.int64, shape=(batch_size))
#        train_labels = tf.placeholder(tf.int64, shape=(batch_size, 1))

        # test book similarity
        valid_dataset = tf.constant(valid_examples, dtype=tf.int64)

        # test recommendations
        if words_embedding_size > 0:
            recommendation_words = tf.constant(len(user_test_words), dtype=tf.int64)
            recommendation_word_weights = tf.constant(len(user_test_word_weights), dtype=tf.float32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/gpu:0'):

            ## Random values to the embedding vectors
            items_embeddings = tf.Variable(tf.random_uniform([max_item_id, items_embedding_size], -1.0, 1.0))
            embedded_items = tf.nn.embedding_lookup_sparse(params=items_embeddings,
                                                           sp_ids=items_train_input,
                                                           sp_weights=items_weights_train_input,
                                                           combiner='mean')

            if words_embedding_size > 0:
                words_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, words_embedding_size], -1.0, 1.0))

                embedded_words = tf.nn.embedding_lookup_sparse(params=words_embeddings,
                                                               sp_ids=words_train_input,
                                                               sp_weights=words_weights_train_input,
                                                               combiner='mean')

            # adesso ho i vettori delle parole e degli item (mediati nel tempo), piu` i profili (che non vengono embedded).
            # Devo concatenarli tutti e tre per poi creare weights & biases.
            # TODO... meglio
            if n_profile_features > 0 and words_embedding_size > 0:
                total_embedded = tf.concat([embedded_items, embedded_words, profile_train_input], 1)
            elif n_profile_features > 0:
                total_embedded = tf.concat([embedded_items, profile_train_input], 1)
            elif words_embedding_size > 0:
                total_embedded = tf.concat([embedded_items, embedded_words], 1)
            else:
                total_embedded = embedded_items

            # Construct the variables for the NCE loss.
            # The concatenated vectors of words, items and profile go into the network and
            # the output is a vector which should be the one_hot of the label (trained against then...)

            total_embedded_input_size = items_embedding_size + words_embedding_size + n_profile_features

            # Go through layer
            relu_1_weights = tf.Variable(tf.random_normal([total_embedded_input_size, max_item_id]))

            relu_1_bias = tf.Variable(tf.random_normal([max_item_id]))

            relu_1_output = tf.nn.relu_layer(total_embedded, relu_1_weights, relu_1_bias, name="relu_1")

            # # TODO
            # # Brutto. Qui da una concatenazione di embedded tenta di ottenere direttamente il
            # # one-hot, con conseguente proliferazione di parametri. Sarebbe meglio convogliare il vettore
            # # di ingresso in un vettore items_embedding_size, e trainarlo con la label embedded.
            # """ RILEGGI simulator.py -> recommender_information doc.
            # La probabilita` di avere label dipende dalla distanza dell'item di input, ma anche da quanto label
            # vende... tipo voglio minimizzare (distance(0-1) - absolute_probability) o roba simile.
            # """
            # nce_weights = tf.Variable(tf.truncated_normal([max_item_id, items_embedding_size],
            #                                               stddev=1.0 / math.sqrt(items_embedding_size)))
            # nce_biases = tf.Variable(tf.zeros([max_item_id]))
            #
            # loss = tf.reduce_mean(
            #     tf.nn.nce_loss(weights=nce_weights,
            #                    biases=nce_biases,
            #                    labels=train_labels,
            #                    inputs=embedded_items,
            #                    num_sampled=num_sampled,
            #                    num_classes=max_item_id,
            #                    num_true=1,
            #                    remove_accidental_hits=True  # in case takes a label as negative
            #                    )
            # )

            # softmax_weights = tf.Variable(tf.truncated_normal([max_item_id,relu_dimension],
            #                                                   stddev=1.0 / math.sqrt(items_embedding_size)))
            # softmax_biases = tf.Variable(tf.zeros([max_item_id]))
            # losses = tf.nn.sampled_softmax_loss(weights=softmax_weights,
            #                                     biases=softmax_biases,
            #                                     labels=train_labels,
            #                                     inputs=relu_1_output,
            #                                     num_sampled=num_sampled,
            #                                     num_classes=max_item_id,
            #                                     num_true=1,
            #                                     sampled_values=None,
            #                                     remove_accidental_hits=True,
            #                                     partition_strategy='mod',
            #                                     name='sampled_softmax_loss',
            #                                     seed=None
            #                                     )
            # loss = tf.reduce_mean(losses)

            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=train_labels,
                    logits=relu_1_output,
                    name='sparse_softmax_cross_entropy'
                )
            )


        # The optimizer will optimize the softmax_weights AND the embeddings.
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        #optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)
        #optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        # for testing...
        y = train_labels

        # Compute the similarity between minibatch examples and all embeddings
        # with the cosine distance (taken from original word2vec example)
        norm = tf.sqrt(tf.reduce_sum(tf.square(items_embeddings), 1, keepdims=True))
        normalized_embeddings = items_embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

        # TODO

        init = tf.global_variables_initializer()

    data_index = 0

    # last number is ~hours (100 steps per minutes)
    num_steps = 100 * 60 * 10
    num_steps = 10000

    log_device_placement = False
    with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=log_device_placement)) as session:
        init.run()
        print('Initialized')
        average_loss = 0.0

        for step in range(num_steps):
            #        try:
            batch, data_index = generate_batch_constant(dataset,
                                                        item_titles,
                                                        data_index=data_index,
                                                        batch_size=batch_size,
                                                        max_words=max_words,
                                                        max_items=max_items)
            # print("Generated batch for step ", step)

            # cfr https://www.tensorflow.org/api_docs/python/tf/sparse_placeholder
            sp_items_train_input = tf.SparseTensor(indices=items_indices,
                                                   values=batch["one_hot_items_values"],
                                                   dense_shape=[batch_size, max_item_id])
            vsp_items_train_input = sp_items_train_input.eval(session=session)

            sp_items_weights_train_input = tf.SparseTensor(indices=items_indices,
                                                           values=batch["one_hot_items_weights"],
                                                           dense_shape=[batch_size, max_item_id])
            vsp_items_weights_train_input = sp_items_weights_train_input.eval(session=session)

            if words_embedding_size > 0:
                feed_dict = {
                    words_train_input: vsp_words_train_input,
                    words_weights_train_input: vsp_words_weights_train_input,
                    items_train_input: vsp_items_train_input,
                    items_weights_train_input: vsp_items_weights_train_input,
                    profile_train_input: batch["profiles"],
                    train_labels: batch["labels"]
                }
            elif(n_profile_features > 0):
                feed_dict = {
                    items_train_input: vsp_items_train_input,
                    items_weights_train_input: vsp_items_weights_train_input,
                    profile_train_input: batch["profiles"],
                    train_labels: batch["labels"]
                }
            else:
                feed_dict = {
                    items_train_input: vsp_items_train_input,
                    items_weights_train_input: vsp_items_weights_train_input,
                    train_labels: batch["labels"]
                }

            #         print("TESTER: ", session.run(y, feed_dict=feed_dict))  # Just test
            _, lozz = session.run([optimizer, loss], feed_dict=feed_dict)
            # print(">>>>>>>>>>>>> ", lozz, step, num_steps)
            #         if np.isnan(lozz):
            #             print("TESTER: ", session.run(y, feed_dict=feed_dict))  # Just test
            #             print("step ", step)
            #             continue
            average_loss += lozz
            #         except:
            #             print("Step %s no buono" % step)
            if step % 500 == 0:

                print("=========================================\n\nDone step ", step, time.time())
                total_num_matches = 0
                average_loss = average_loss / 500
                print("Average loss:", average_loss)
                average_loss = 0.0
                sim = similarity.eval()
                for i in range(valid_size):
                    if synthetic:
                        valid_item = valid_examples[i]

                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to %s:' % valid_item
                    for k in range(top_k):
                        if synthetic:
                            close_item = nearest[k]

                        log = '%s %s,' % (log, close_item)

                    num_matches = match_similar_items(valid_item, nearest, similar_item)
                    total_num_matches += num_matches
                    log = '%s %i' % (log, num_matches)
                    print(log)

                print('Total num of matches {}'.format(total_num_matches))
    #            embeds = embeddings.eval()

    #                print("Distanza: ", distanza(embeds[1395], embeds[810]))

    #    distance_embeddings = embeddings.eval()
