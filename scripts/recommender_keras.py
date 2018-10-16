import xml.etree.ElementTree as ET
import numpy as np
import datetime
import math
import re
import itertools
from operator import itemgetter
from keras.layers import Embedding, Dense, Flatten, Dropout, Reshape, Lambda
from keras.models import Sequential, Model
from keras import optimizers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Recommender\'s Trainer')
    parser.add_argument('-b', '--batch-size', type=int, required=False, dest='batch_size', default=32,
                        help='batch size (default 32)')
    parser.add_argument('-r', '--learning-rate', type=float, required=False, dest='learning_rate', default=0.001,
                        help='learning rate (default 0.001)')
    parser.add_argument('-d', '--embedding-dim', type=int, required=False, default=16,
                         dest='embedding_dimension', help='dimension of the embedding space (default 16)')
    parser.add_argument('-e', '--epochs', type=int, required=False, dest='num_epochs', default=50,
                        help='number of epochs (default 50)')
    parser.add_argument('-i', '--xml-data-file', type=str, required=False, dest='xml_data_file', default='',
                        help='xml data file containing all transactions (default \'\')')
    parser.add_argument('-m', '--min-n-baskets', type=int, required=False, dest='min_num_baskets', default=3,
                        help='minimum number f purchased baskets (default 3).')
    parser.add_argument('-w', '--n-train-baskets', type=int, required=False, dest='num_train_baskets', default=3,
                        help='number of training baskets (default 3).')
    parser.add_argument('-t', '--n-train_items', type=int, required=False, dest='num_train_items', default=20,
                        help='number of labels to train with (default 20)')
    args = parser.parse_args()

    return args


def clean_item(item):
    item = re.sub("[^A-Za-z ]", "", item.strip())
    item = item.replace(" GR"," ").replace(" LIT"," ").replace(" MLIT"," ").replace(" MEDIU"," ").replace(" SMALL"," ").replace(" OZ"," ").replace(" C "," ").replace(" P "," ").replace(" G "," ").replace(" CC ", " ").replace(" KGS ", " ").replace(" R ", " ").replace(" B ", " ").replace(" KG ", " ").replace(" D ", " ").replace(" Z ", " ").replace(" ML ", " ").replace(" L ", " ").strip()
    item = re.sub(" U$", '', item)
    item = re.sub(" T$", '', item)
    item = re.sub(" B$", '', item)
    item = re.sub(" ML$", '', item)
    item = re.sub(" KGS$", '', item)
    item = re.sub(" BA$", '', item)
    item = re.sub(" ML$", '', item)
    item = re.sub(" S$", '', item)
    item = re.sub(" SS$", '', item)
    item = re.sub(" KG$", '', item)
    item = re.sub(" M$", '', item)
    item = re.sub(" CC$", '', item)
    item = re.sub(" R$", '', item)
    item = re.sub(" Z$", '', item)
    item = re.sub(" LI$", '', item)
    item = re.sub(" D$", '', item)
    item = re.sub(" PP$", '', item)
    item = re.sub(" CHE$", '', item)
    item = re.sub(" BEE$", '', item)
    item = re.sub(" TUR$", '', item)
    return re.sub("\s+", ' ', item).strip()


def load_data(filename):
    '''

    Parameters
    ----------

    filename:             (string) xml filename containing all transactions

    Returns
    -------
    baskets                 (dict) {user: {date: [item1, item1, ...], ...}}
    items                   (dict) {item: (price, brand, division, barcode), ...}
    item_rankid             (dict) {item: rank_id, ...}
    rankid_item             (dict) {rank_id: item, ...}

    '''

    root = ET.parse(filename).getroot()

    items = {}  # key is barcode
    users = {}  # just the PRT_PARTNER_ID
    actions = []
    entries = []
    baskets = {}  # {user: {date: [item1, item1, ...], ...}}
    item_popularity = {}

    for child in root:
        data = {}
        for gc in child:
            data[gc.attrib['NAME']] = gc.text
        purchase_date = datetime.datetime.strptime(data['INVOICE_DATE'], "%d-%b-%y")
        user = data['PRT_PARTNER_ID']
        item = clean_item(data['DESCRIPTION'])
        item_popularity[item] = item_popularity.get(item, 0) + 1
        items[item] = (data['PRICE'], data['BRAND'], data['DIVISION'], data['BARCODE'])
        users[data['PRT_PARTNER_ID']] = ()  # we have no info about users
        actions.append((data['PRT_PARTNER_ID'], item, purchase_date))
        entries.append({'item': item, 'date': purchase_date,
                        'price': data['PRICE'], 'brand':  data['BRAND'], 'division': data['DIVISION'],
                        'description': data['DESCRIPTION'], 'user': user})

        baskets.setdefault(user, {}).setdefault(purchase_date, set()).add(item)

    item_rankid = {x[0]: i+1 for i, x in enumerate(sorted(item_popularity.items(), key=itemgetter(1), reverse=True))}
    rankid_item = {v: k for k, v in item_rankid.items()}

    return baskets, items, item_rankid, rankid_item


def data_transform(baskets, num_train_baskets, num_train_items,
                   min_num_baskets=4, item_rankid=None,
                   flatten=True, mask_zero=True,
                   fixed_length=False):
    '''
    
    Parameters
    ----------
    
    baskets:             (dict) {user: {date: [item1, item1, ...], ...}}
    num_train_baskets:    (int) how many past baskets we consider
    num_train_items:      (int) how many items to be taken from each basket. If 0 take all.
    min_num_baskets:      (int) minimum number of purchased baskets
    item_rankid:         (dict) {item: rankid, ...}
    flatten:             (bool) produce an array with training (or all) items from all training baskets 
    mask_zero:           (bool) add trailing 0s if not enough items in the basket
    
    Returns
    -------
    (X, y):              (np.arrays) "X" all training events --items bought in the past,
                          either flat or array of baskets. "y" all items in the label basket.
    
    '''
    
    X = []  # (Tot. # of samples, num_train_baskets, input_length)
    y = []  # (Tot # samples, 1, input_length)

    # clean data (only users and baskets with enough date)
    for dates_bsklist in baskets.values():
        if len(dates_bsklist) >= min_num_baskets:  # num_train_baskets + 1:  # so (maybe) we can take the last as "label"
            clean_dates = []  #  list of baskets of the same user: [[item, item, ...], [item, item, ...], ...] 
            for d_b in sorted(dates_bsklist.items()):
                basket = d_b[1]
                if len(basket) >= num_train_items:
                    # if basket is bigger than training items roll over and produce various training sets
                    for i in range(len(basket) - num_train_items):
                        #(NB no shuffle(basket) bc is a set)
                        if item_rankid is not None:
                            clean_dates.append([item_rankid[x] for x in list(basket)[i:i+num_train_items]])
                        else:
                            clean_dates.append(list(basket)[i:i+num_train_items])
                else:  # fill with 0s
                    if item_rankid is not None:
                        if mask_zero and not flatten:
                            clean_dates.append([item_rankid[x] for x in list(basket)] + [0]*(num_train_items-len(basket)))
                        else:
                            clean_dates.append([item_rankid[x] for x in list(basket)])
                    else:
                        if mask_zero and not flatten:
                            clean_dates.append(list(basket) + [0]*(num_train_items-len(basket)))
                        else:
                            clean_dates.append(list(basket))

            for i in range(len(clean_dates)-num_train_baskets):
                if flatten:
                    # flatten it
                    train = list(itertools.chain.from_iterable(clean_dates[i:i+num_train_baskets]))
                    if mask_zero and len(train) < (num_train_items*num_train_baskets):
                        train = np.append(train, [0]*(num_train_items*num_train_baskets-len(train)))
                    else:
                        train = train[:num_train_items*num_train_baskets]
                    train = np.array(train)
                else:
                    train = np.array(clean_dates[i:i+num_train_baskets])
                if mask_zero:
                    label = np.append(clean_dates[i+num_train_baskets][:num_train_items],
                                      [0]*(num_train_items-len(clean_dates[i+num_train_baskets])))
                else:
                    label = clean_dates[i+num_train_baskets][:num_train_items]
                if not fixed_length or (len(train) == num_train_items*num_train_baskets and
                                        len(label) == num_train_items):
                    X.append(train)
                    try:
                        y.append(np.array(label).astype(int))
                    except ValueError:
                        y.append(np.array(label))

    return np.array(X), np.array(y)


if __name__ == '__main__':

    args = parse_args()
    batch_size = args.batch_size
    filename = args.xml_data_file
    embedding_dimension = args.embedding_dimension
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    drop_prob = args.dropout_prob
    num_train_baskets = args.num_train_baskets
    num_train_items = args.num_train_items
    min_num_baskets = args.min_num_baskets

    baskets, items, item_rankid, rankid_item = load_data(filename)

    print(len(items))
    vocabulary_size = len(items) + 1  # + 1 bc zero is for masking

    X, y = data_transform(baskets=baskets,
                          num_train_baskets=num_train_baskets,
                          num_train_items=num_train_items,
                          min_num_baskets=min_num_basketss, item_rankid=item_rankid,
                          flatten=True, mask_zero=False,
                          fixed_length=True)

    print("first 'X':\n")
    print(X.shape, X[0])
    print("\n\nfirst 'y':\n")
    print(y.shape, len(y[0]))
    print("X shape: ", X.shape)
    print(X.shape == (len(X), num_train_items*num_train_baskets))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(y_train[0])
    y_train_multilabel = np.zeros([len(y_train), vocabulary_size])
    y_test_multilabel = np.zeros([len(y_test), vocabulary_size])
    for i, y in enumerate(y_train):
        y_train_multilabel[i, y] = 1
        
    for i, y in enumerate(y_test):
        y_test_multilabel[i, y] = 1

    simple_model = Sequential()
    simple_model.add(Embedding(input_dim=vocabulary_size,
                               output_dim=embedding_dimension,
                               input_length=num_train_items * num_train_baskets,
                               mask_zero=False))
    simple_model.add(Flatten())
    simple_model.add(Dense(units=embedding_dimension, activation='relu'))
    simple_model.add(Dense(units=vocabulary_size, activation='sigmoid'))

    adam_opt = optimizers.Adam(lr=learning_rate)
    simple_model.compile(loss='binary_crossentropy',
                         optimizer=adam_opt,
                         metrics=['accuracy'])

    simple_model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    simple_model.fit(X_train, y_train_multilabel,
                     validation_split=0.1,
                     batch_size=batch_size,
                     epochs=num_epochs,
                     callbacks=[early_stop])

    predictions = simple_model.predict(X_test)

    yhat = []
    for prediction, expectation in zip(predictions, y_test):
        yhat.append(np.argpartition(prediction, -len(expectation))[-len(expectation):])

    matches = 0
    total_count = 0
    for prediction, expectation in zip(yhat, y_test):
        matches += len(set(prediction).intersection(set(expectation)))
        total_count += len(expectation)
    accuracy = matches / float(total_count)
    print('accuracy: {:.3f}'.format(accuracy))

