from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

from keras.layers import Input, Dense, Dropout,Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU,PReLU,ELU
from keras.models import Model
from keras.callbacks import Callback

__all__ = ['make_folds', 'generate_keras_model', 'HistoryCallback']

def make_folds(train_y, n_folds = 3, type = 'stratified', random_seed = 1, shuffle = True):
    '''splits dataset into train and validation folds

    Parameters
    ----------
    train_y : numpy.ndarray
        target
    n_folds : integer, optional
        number of folds
        3 by default
    random_seed : integer, optional
        random seed for stratificator
        1 by default
    type : string, optional, 'stratified' or 'random'
        'stratified' means stratified by train_y, 'random' means random split
        'stratified' by default
    shuffle : boolean, optional
        whether to shuffle dataset before split
        True by default
    
    Returns
    -------
    folds : list of pairs of lists
    indices of train and validation folds
    folds[i][0] - train indices in i-th train-val split
    folds[i][1] - validation indices in i-th train-val split
    '''

    folds = []
    if type == 'stratified':
        skf = StratifiedKFold(n_splits = n_folds, random_state = random_seed, shuffle = shuffle)
    elif type == 'random':
        skf = KFold(n_splits = n_folds, random_state = random_seed, shuffle = shuffle)
    else:
        raise ValueError("type should be 'stratified' or 'random', got %s" % type)

    for train_index, val_index in skf.split(np.zeros(train_y.shape[0]), train_y):
        folds.append((train_index, val_index))
    return folds

class HistoryCallback(Callback):
    '''makes predictions on validation and, optionally, test dataset
    
    allows to select epoches based on validation metrics on them and aggregate predictions for test on selected epoches

    Parameters
    ----------
    val_x : numpy.ndarray
        validation dataset
    test_x : numpy.ndarray, optional
        test dataset
        None by default
    
    Attributes
    -------
    val_predictions : list of np.ndarrays
        predictions on validation on every epoch
    test_predictions : list of np.ndarrays, optional
        predictions on validation on every epoch      
    '''

    def __init__(self, val_x, test_x = None):
        self.val_x = val_x
        self.test_x = test_x

    def on_train_begin(self, logs={}):
        self.val_predictions = []
        self.test_predictions = []

    def on_epoch_end(self, epoch, logs={}):
        self.val_predictions.append(self.model.predict(self.val_x))
        if isinstance(self.test_x, np.ndarray):
            self.test_predictions.append(self.model.predict(self.test_x))
            
def generate_keras_model(params, task, n_classes = None, verbose = False):
    '''generates keras model by parameters
    
    built model is multil-layer-perceptron with activation functions and optionally dropouts

    Parameters
    ----------
    params : dict
        dictionary of parameters
        params['input_dim']: integer
            input layer dimension
        params['hidden_sizes']: list of integers
            number of neurons in hidden layers
        params['hidden_activation']: string
            activation function for hidden layers : 'sigmoid', 'tanh', 'relu', 'elu', 'prelu', 'leakyrelu'
        params['dropouts']: list of floats
            dropouts by layers, values <= 0 means no dropout
        params['init']: string
            initialization, one of the keras initializations like 'glorot_normal'
        params['output_activation']: string
            output activation function
    task : string, 'regression' or 'binary_classification' or 'multiclass_classification'
        task to solve
    n_classes : integer, optional
        number of classes in cases of classification task
        2 by default
    verbose : boolean, optional:
        whether to print running info
        False by default
    
    Returns
    -------
    folds : list of pairs of lists
    indices of train and validation folds
    folds[i][0] - train indices in i-th train-val split
    folds[i][1] - validation indices in i-th train-val split
    '''

    input_dim, hidden_sizes, hidden_activation, dropouts, init, output_activation = params['input_dim'], params['hidden_sizes'], params['hidden_activation'], params['dropouts'], params['init'], params['output_activation']
    if verbose:
        print params


    n_layers = len(hidden_sizes)

    input = Input((input_dim, ))

    for l in xrange(n_layers):

        if l == 0:
            prev_layer = input
        else:
            prev_layer = cur_layer

        if hidden_activation == 'prelu':
            cur_layer = Dense(units = hidden_sizes[l], activation = PReLU(), kernel_initializer = init)(prev_layer)        
        elif hidden_activation == 'leakyrelu':
            cur_layer = Dense(units = hidden_sizes[l], activation = LeakyReLU, kernel_initializer = init)(prev_layer)        
        elif hidden_activation == 'elu':
            cur_layer = Dense(units = hidden_sizes[l], activation = ELU(), kernel_initializer = init)(prev_layer)        

        else:
            cur_layer = Dense(units = hidden_sizes[l], activation = params['hidden_activation'], kernel_initializer = init)(prev_layer)        

        if 'dropouts' in params:
            if dropouts[l] > 0:
                cur_layer = Dropout(dropouts[l])(cur_layer)

    if task == 'multiclass_classification':
        cur_layer = Dense(units = n_classes, activation = output_activation, kernel_initializer = init)(cur_layer)
    elif task == 'binary_classification':
        cur_layer = Dense(units = 1, activation = output_activation, kernel_initializer = init)(cur_layer)
    elif task== 'regression':    
        cur_layer = Dense(units = 1, activation = output_activation, kernel_initializer = init)(cur_layer)
    else:
        raise ValueError("task should be 'regression' or 'binary_classification' or 'multiclass_classification', got %s" % task)

    optimizer_adam = Adam()

    if task == 'multiclass_classification':
        loss = 'categorical_crossentropy'
    elif task =='binary_classification':
        loss = 'binary_crossentropy'
    elif task == 'regression':
        loss = 'mse'

    model = Model(inputs = input, outputs = cur_layer)
    if verbose:
        print model.summary()

    model.compile(loss = loss, optimizer = optimizer_adam)
    return model