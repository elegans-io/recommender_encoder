import xml.etree.ElementTree as ET
import numpy as np
import datetime
import re
import itertools
from operator import itemgetter
import recutils
import keras as ks
from keras.layers import Embedding, Dense, LSTM, Flatten, Dropout, Reshape
from keras.models import Sequential
from keras import losses
from keras import optimizers
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.model_selection import train_test_split
import argparse
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
    parser.add_argument('-n', '--nodes', type=int, required=False, dest='nodes', default=16,
                        help='model layers\' nodes (default 16).')
    parser.add_argument('-p', '--dropout', type=float, required=False, dest='dropout', default=0.0,
                        help='dropout probability (default 0.0)')
    parser.add_argument('-s', '--do-serialization', required=False, dest='do_serialization', action='store_false',
                        help='do serialization of data structures obtained from xml-data-file (default False)')
    parser.add_argument('-u', '--dropout-rnn', type=float, required=False, dest='dropout_rnn', default=0.0,
                        help='dropout probability for RNN layer both input and recurrent (default 0.0)')
    parser.add_argument('-f', '--loss', type=str, required=False, dest='loss_funct', default='bxe',
                        help='loss function (bxe=binary cross-entropy, fl=focal loss, wl=weighted loss) (default bxe)')
    parser.add_argument('-a', '--alpha', type=float, required=False, dest='alpha', default='0.25',
                        help='alpha parameter for focal loss (default 0.25)')
    parser.add_argument('-g', '--gamma', type=float, required=False, dest='gamma', default='2.0',
                        help='gamma parameter for focal loss (default 2.0)')
    parser.add_argument('-q', '--model_type', type=str, required=False, dest='model_type', default='dense',
                        help='model type (dense=fully connected, rnn=recurrent, (default=dense)')
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


def model_params(args, vocabulary_size):

    params = {
        'model_type': args.model_type.lower(),
        'batch_size': args.batch_size,
        'vocabulary_size': vocabulary_size,
        'embedding_dimension': args.embedding_dimension,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'nodes': args.nodes,
        'dropout': args.dropout,
        'dropout_rnn': args.dropout_rnn,
        'num_train_baskets': args.num_train_baskets,
        'num_train_items': args.num_train_items,
        'min_num_baskets': args.min_num_baskets,
        'loss_fct_str': args.loss_funct.lower(),
        'alpha': args.alpha,
        'gamma': args.gamma
    }

    return params


class RecommenderModel:
    '''
    Wrapper of the Sequential Keras's model
    '''
    def __init__(self, params):
        self.params = params
        self.model = None
        self.metrics = ['binary_accuracy', 'categorical_accuracy']

    def build(self):
        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.params['vocabulary_size'],
                                 output_dim=self.params['embedding_dimension'],
                                 input_length=self.params['num_train_items'] * self.params['num_train_baskets'],
                                 mask_zero=False))

        if self.params['model_type'] == 'dense':
            self.model.add(Flatten())
            self.model.add(Dense(units=self.params['embedding_dimension'] * self.params['num_train_items'] * self.params['num_train_baskets'], activation='relu'))
            self.model.add(Dropout(self.params['dropout']))
            self.model.add(Dense(units=self.params['nodes'], activation='relu'))
            self.model.add(Dropout(self.params['dropout']))
            self.model.add(Dense(units=self.params['vocabulary_size'], activation='sigmoid'))

        elif self.params['model_type'] == 'rnn':
            self.model.add(Reshape((self.params['num_train_baskets'], self.params['num_train_items'] * self.params['embedding_dimension'])))
            self.model.add(LSTM(self.params['nodes'], dropout=self.params['dropout_rnn'], recurrent_dropout=self.params['dropout_rnn']))
            self.model.add(Dropout(self.params['dropout']))
            self.model.add(Dense(units=self.params['num_train_items'], activation='relu'))
            self.model.add(Dropout(self.params['dropout']))
            self.model.add(Dense(units=self.params['vocabulary_size'], activation='sigmoid'))
        else:
            raise NotImplementedError

    def compile(self, loss_fct):
        optimizer = optimizers.Adam(lr=self.params['learning_rate'])
        self.model.compile(loss=loss_fct,
                           optimizer=optimizer,
                           metrics=self.metrics)

        self.model.summary()

    def fit(self, X, y, valid_split, callbacks):
        hist = self.model.fit(X, y,
                              validation_split=valid_split,
                              batch_size=self.params['batch_size'],
                              epochs=self.params['num_epochs'],
                              callbacks=callbacks,
                              verbose=1)
        return hist

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def prefix_str(self):
        if self.params['model_type'] == 'dense':
            prefix = 'model_{t}_ed_{ed:02d}_ntb_{ntb:1d}_nti_{nti:02d}_mnb_{mnb:1d}_' \
                              'e_{ep:03d}_b_{bs:03d}_r_{lr:.3f}_dp_{dp:.02f}_adam_{lf}'. \
                format(t=self.params['model_type'], ed=self.params['embedding_dimension'],
                       ntb=self.params['num_train_baskets'],
                       nti=self.params['num_train_items'], mnb=self.params['min_num_baskets'],
                       ep=self.params['num_epochs'], bs=self.params['batch_size'], lr=self.params['learning_rate'],
                       dp=self.params['dropout'],
                       lf=self.params['loss_fct_str'])
        else:
            prefix = 'model_{t}_ed_{ed:02d}_ntb_{ntb:1d}_nti_{nti:02d}_mnb_{mnb:1d}_' \
                              'e_{ep:03d}_b_{bs:03d}_r_{lr:.3f}_n_{ns:02d}_dp_{dp:.02f}_dpr_{dpr:.02f}_adam_{lf}'. \
                format(t=self.params['model_type'], ed=self.params['embedding_dimension'],
                       ntb=self.params['num_train_baskets'],
                       nti=self.params['num_train_items'], mnb=self.params['min_num_baskets'],
                       ep=self.params['num_epochs'], bs=self.params['batch_size'], lr=self.params['learning_rate'],
                       ns=self.params['nodes'], dp=self.params['dropout'], dpr=self.params['dropout_rnn'],
                       lf=self.params['loss_fct_str'])

        return prefix


if __name__ == '__main__':

    args = parse_args()
    filename = args.xml_data_file
    loss_fct_str = args.loss_funct.lower()
    alpha = args.alpha
    gamma = args.gamma

    baskets_filename = 'baskets'
    items_filename = 'items'
    item_rankid_filename = 'item_rankid'
    rankid_item_filename = 'rankid_item'

    do_serialize = False
    if do_serialize:
        baskets, items, item_rankid, rankid_item = load_data(filename)

        with open(baskets_filename, 'wb') as f:
            pickle.dump(baskets, f)

        with open(items_filename, 'wb') as f:
            pickle.dump(items, f)

        with open(item_rankid_filename, 'wb') as f:
            pickle.dump(item_rankid, f)

        with open(rankid_item_filename, 'wb') as f:
            pickle.dump(rankid_item, f)

    else:
        with open(baskets_filename, 'rb') as f:
            baskets = pickle.load(f)

        with open(items_filename, 'rb') as f:
            items = pickle.load(f)

        with open(item_rankid_filename, 'rb') as f:
            item_rankid = pickle.load(f)

        with open(rankid_item_filename, 'rb') as f:
            rankid_item = pickle.load(f)

    print(len(items))
    vocabulary_size = len(items) + 1  # + 1 bc zero is for masking

    params = model_params(args, vocabulary_size)

    X, y = data_transform(baskets=baskets,
                          num_train_baskets=params['num_train_baskets'],
                          num_train_items=params['num_train_items'],
                          min_num_baskets=params['min_num_baskets'], item_rankid=item_rankid,
                          flatten=True,
                          mask_zero=False,
                          fixed_length=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(y_train[0])
    y_train_multilabel = np.zeros([len(y_train), vocabulary_size])
    y_test_multilabel = np.zeros([len(y_test), vocabulary_size])
    for i, y in enumerate(y_train):
        y_train_multilabel[i, y] = 1
    for i, y in enumerate(y_test):
        y_test_multilabel[i, y] = 1

    # determine the loss function
    if loss_fct_str == 'bxe':
        loss_fct = losses.binary_crossentropy
    elif loss_fct_str == 'wl':
        class_weights = recutils.compute_class_weights(y_train_multilabel)
        loss_fct = recutils.weighted_loss(class_weights)
    elif loss_fct_str == 'fl':
        loss_fct = recutils.focal_loss(gamma, alpha)
    else:
        print('Unknow loss. setting to default loss (binary crossentropy)')
        loss_fct = losses.binary_crossentropy

    # instantiate the recommender model
    recommend_model = RecommenderModel(params)
    recommend_model.build()
    recommend_model.compile(loss_fct)

    # callbacks
    checkpoint = ModelCheckpoint('model.hdf5', monitor='val_multilabel_acc', save_best_only=True, mode='max', verbose=1)
    metrics = recutils.Metrics()
    callbacks = [metrics, checkpoint]

    # fit the model
    valid_split = 0.2
    hist = recommend_model.fit(X_train, y_train_multilabel, valid_split, callbacks)

    # training set predictions
    predictions = recommend_model.predict(X_train)
    accuracy = recutils.multilabel_acc(y_train, predictions)
    print('train multilabel accuracy: {:.3f}'.format(accuracy))

    _, bin_acc, cat_acc = recommend_model.evaluate(X_test, y_test_multilabel)
    print('test_bin_acc: {:.3f}, test_cat_acc: {:.3f}'.format(bin_acc, cat_acc))

    # test set predictions
    predictions = recommend_model.predict(X_test)
    accuracy = recutils.multilabel_acc(y_test, predictions)
    print('test multilabel accuracy: {:.3f}'.format(accuracy))

    # generate and save the plot of the validation multilabel accuracy on the validation set
    plt.plot(hist.epoch, hist.history['val_multilabel_acc'], 'ro')
    plt.xlabel('Epochs')
    plt.ylabel('Multilabel Accuracy')
    plt.grid(True)
    hist_min = np.min(hist.history['val_multilabel_acc'])
    hist_max = np.max(hist.history['val_multilabel_acc'])
    plt.axis([0, params['num_epochs'], hist_min, hist_max])
    plt.savefig('val_multilabel_acc.png')


