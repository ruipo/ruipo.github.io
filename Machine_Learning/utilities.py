import numpy as np
import pdb
import operator
import itertools
import functools

import matplotlib.pyplot as plt
from matplotlib import colors

###### Utilities ######

def cv(value_list):
    '''
    Takes a list of numbers and returns a column vector:  n x 1
    '''
    return np.transpose(rv(value_list))

def rv(value_list):
    '''
    Takes a list of numbers and returns a row vector: 1 x n
    '''
    return np.array([value_list])

def signed_dist(x, th, th0):
    '''
    x is dimension d by 1
    th is dimension d by 1
    th0 is a scalar
    return 1 by 1 matrix of signed distance
    '''
    return ((th.T@x) + th0) / length(th)

def positive(x, th, th0):
    '''
    x is dimension d by 1
    th is dimension d by 1
    th0 is dimension 1 by 1
    return 1 by 1 matrix of +1, 0, -1
    '''
    return np.sign(np.dot(np.transpose(th), x) + th0)

def score(data, labels, th, th0):
    '''
    data is dimension d by n
    labels is dimension 1 by n
    ths is dimension d by 1
    th0s is dimension 1 by 1
    return 1 by 1 matrix of integer indicating number of data points correct for
    each separator.
    '''
    return np.sum(positive(data, th, th0) == labels)

def length(d_by_m):
    return np.sum(d_by_m * d_by_m, axis = 0, keepdims = True)**0.5

def reverse_dict(d):
    """
    reverses the keys and items in a dictionary
    """
    return {v: k for k, v in d.items()}

##### Evaluation #####

def eval_classifier(learner, data_train, labels_train, data_test, labels_test, params):
    
    '''
    returns the score of the inputted classifier on test data after training on training data
    '''
    th, th0 = learner(data_train, labels_train, params)
    return score(data_test, labels_test, th, th0)/data_test.shape[1]

def xval_learning_alg(learner, data, labels, params, k):
    
    '''
    performs cross validation with parameter k on the inputted learning algorithm
    and returns the averaged score (average accuracy) of all runs
    '''
    
    _, n = data.shape
    idx = list(range(n))
    np.random.seed(0)
    np.random.shuffle(idx)
    data, labels = data[:,idx], labels[:,idx]
    
    s_data = np.array_split(data, k, axis=1)
    s_labels = np.array_split(labels, k, axis=1)
    
    score_sum = 0
    for i in range(k):
        data_train = np.concatenate(s_data[:i] + s_data[i+1:], axis=1)
        labels_train = np.concatenate(s_labels[:i] + s_labels[i+1:], axis=1)
        data_test = np.array(s_data[i])
        labels_test = np.array(s_labels[i])
        score_sum += eval_classifier(learner, data_train, labels_train,
                                     data_test, labels_test, params)
    return score_sum/k


##### Feature Transformation #####

def one_hot(v, entries):
    '''
    input an item in an list of entries and the list itself to generate
    an one hot encoding of that item
    '''
    vec = len(entries)*[0]
    vec[entries.index(v)] = 1
    return vec

def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')
    
    return input_string.lower().split()

def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input
    """
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary

def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of strings
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.
    """
    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])
    
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = 1
    # We want the feature vectors as columns
    return feature_matrix.T

def raw_features(x):
    """
        @param x (n_samples,m,n) array with values in (0,1)
        @return (m*n,n_samples) reshaped array where each entry is preserved
        """
    (n_samp,m,n) = np.shape(x)
    
    return np.reshape(x,(n_samp,m*n)).T

def row_average_features(x):
    """
        @param x (n_samples,m,n) array with values in (0,1)
        @return (m,n_samples) array where each entry is the average of a row
        """
    return np.mean(x, axis=2).T

def col_average_features(x):
    """
        @param x (n_samples,m,n) array with values in (0,1)
        @return (n,n_samples) array where each entry is the average of a column
        """
    
    return np.mean(x, axis=1).T

def top_bottom_features(x):
    """
        @param x (n_samples,m,n) array with values in (0,1)
        @return (2,n_samples) array where the first entry of each column is the average of the
        top half of the image = rows 0 to floor(m/2) [exclusive]
        and the second entry is the average of the bottom half of the image
        = rows floor(m/2) [inclusive] to m
        """
    m = x.shape[2]
    top = np.mean(x[:,:m//2,:],axis = 2)
    top = np.mean(top,axis = 1)
    
    bot = np.mean(x[:,m//2:,:],axis = 2)
    bot = np.mean(bot,axis = 1)
    
    return np.vstack((top,bot))


##### Plotting #####

def tidy_plot(xmin, xmax, ymin, ymax, center = False, title = None,
              xlabel = None, ylabel = None):
    '''
        Set up axes for plotting
        xmin, xmax, ymin, ymax = (float) plot extents
        Return matplotlib axes
        '''
    plt.ion()
    plt.figure(facecolor="white")
    ax = plt.subplot()
    if center:
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
eps = .05
    plt.xlim(xmin-eps, xmax+eps)
    plt.ylim(ymin-eps, ymax+eps)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax

def plot_separator(ax, th, th_0):
    '''
        Plot separator in 2D
        ax = (matplotlib plot) plot axis
        th = (numpy array) theta
        th_0 = (float) theta_0
        '''
    xmin, xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    pts = []
    eps = 1.0e-6
    # xmin boundary crossing is when xmin th[0] + y th[1] + th_0 = 0
    # that is, y = (-th_0 - xmin th[0]) / th[1]
    if abs(th[1,0]) > eps:
        pts += [np.array([x, (-th_0 - x * th[0,0]) / th[1,0]])                                                         for x in (xmin, xmax)]
    if abs(th[0,0]) > 1.0e-6:
        pts += [np.array([(-th_0 - y * th[1,0]) / th[0,0], y])                                                          for y in (ymin, ymax)]
    in_pts = []
    for p in pts:
        if (xmin-eps) <= p[0] <= (xmax+eps) and            (ymin-eps) <= p[1] <= (ymax+eps):
            duplicate = False
            for p1 in in_pts:
                if np.max(np.abs(p - p1)) < 1.0e-6:
                    duplicate = True
            if not duplicate:
                in_pts.append(p)
    if in_pts and len(in_pts) >= 2:
        # Plot separator
        vpts = np.vstack(in_pts)
        ax.plot(vpts[:,0], vpts[:,1], 'k-', lw=2)
        # Plot normal
        vmid = 0.5*(in_pts[0] + in_pts[1])
        scale = np.sum(th*th)**0.5
        diff = in_pts[0] - in_pts[1]
        dist = max(xmax-xmin, ymax-ymin)
        vnrm = vmid + (dist/10)*(th.T[0]/scale)
        vpts = np.vstack([vmid, vnrm])
        ax.plot(vpts[:,0], vpts[:,1], 'k-', lw=2)
        # Try to keep limits from moving around
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
    else:
        print('Separator not in plot range')

def plot_data(data, labels, ax = None, clear = False,
              xmin = None, xmax = None, ymin = None, ymax = None):
    '''
        Make scatter plot of data.
        data = (numpy array)
        ax = (matplotlib plot)
        clear = (bool) clear current plot first
        xmin, xmax, ymin, ymax = (float) plot extents
        returns matplotlib plot on ax
        '''
    if ax is None:
        if xmin == None: xmin = np.min(data[0, :]) - 0.5
        if xmax == None: xmax = np.max(data[0, :]) + 0.5
        if ymin == None: ymin = np.min(data[1, :]) - 0.5
        if ymax == None: ymax = np.max(data[1, :]) + 0.5
        ax = tidy_plot(xmin, xmax, ymin, ymax)
        
        x_range = xmax - xmin; y_range = ymax - ymin
        if .1 < x_range / y_range < 10:
            ax.set_aspect('equal')
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
elif clear:
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.clear()
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    colors = np.choose(labels > 0, cv(['r', 'g']))[0]
    ax.scatter(data[0,:], data[1,:], c = colors,
               marker = 'o', s=50, edgecolors = 'none')
# Seems to occasionally mess up the limits
ax.set_xlim(xlim); ax.set_ylim(ylim)
ax.grid(True, which='both')
#ax.axhline(y=0, color='k')
#ax.axvline(x=0, color='k')
    return ax


def plot_nonlin_sep(predictor, ax = None, xmin = None , xmax = None, ymin = None, ymax = None, res = 30):
    
    '''
    Must either specify limits or existing ax
    '''
    
    if ax is None:
        ax = tidy_plot(xmin, xmax, ymin, ymax)
    else:
        if xmin == None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
        else:
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))

    cmap = colors.ListedColormap(['black', 'white'])
    bounds=[-2,0,2]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ima = np.array([[predictor(x1i, x2i) \ for x1i in np.linspace(xmin, xmax, res)] \ for x2i in np.linspace(ymin, ymax, res)])
    im = ax.imshow(np.flipud(ima), interpolation = 'none', extent = [xmin, xmax, ymin, ymax], cmap = cmap, norm = norm)
