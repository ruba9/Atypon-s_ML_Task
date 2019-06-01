#this code process the dataset and splits the reviewa and labels.
import csv
from collections import Counter
import numpy as np
import time
import sys
 
def preprocess_data():
 
    all_revs_file = open('all_revs.txt', 'w')
    all_labels_file = open('all_labels.txt', 'w')
 
    test_revs_file = open('test_revs.txt','w') 
    test_labels_file = open('test_labels.txt','w') 
    train_revs_file = open('train_revs.txt','w') 
    train_labels_file = open('train_labels.txt','w') 
 
    with open('all_data.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            line_count += 1
 
            review = row['review'] + '\n'
            label = row['label']
            if label == "pos":
                label = 'positive' + '\n'
            else: 
                label = 'negative' + '\n'
            all_revs_file.write(review)
            all_labels_file.write(label)
 
            type = row['type']
            if type == "test":
                review = row['review'] + '\n'
                label = row['label']
                if label == "pos":
                    label = 'positive' + '\n'
                else: 
                    label = 'negative' + '\n'
                test_revs_file.write(review)
                test_labels_file.write(label)
            if type == "train":
                review = row['review'] + '\n'
                label = row['label']
                if label == "pos":
                    label = 'positive' + '\n'
                else:
                    label = 'negative' + '\n'
                train_revs_file.write(review)
                train_labels_file.write(label)
        print("Preprocessed %d reviews." % line_count)
 
    test_revs_file.close()
    test_labels_file.close()
    train_revs_file.close()
    train_labels_file.close()
 
 
preprocess_data()
 
def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")
 
g = open('all_revs.txt','r') # What we know!
all_reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()
 
g = open('all_labels.txt','r') # What we WANT to know!
all_labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()
 
g = open('test_revs.txt','r') # What we know!
testing_reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()
 
g = open('test_labels.txt','r') # What we WANT to know!
testing_labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()
 
g = open('train_revs.txt','r') 
training_reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()
 
g = open('train_labels.txt','r') 
training_labels = list(map(lambda x:x[:-1],g.readlines()))
g.close()
 
# Created three Counter objects to store positive, negative and total counts
positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()
 
# Loop over all the words in all the reviews and increment the counts in the appropriate counter objects
for i in range(len(testing_reviews)):
    if(testing_labels[i] == 'POSITIVE'):
        for word in testing_reviews[i].split(" "):
            positive_counts[word] += 1
            total_counts[word] += 1
    else:
        for word in testing_reviews[i].split(" "):
            negative_counts[word] += 1
            total_counts[word] += 1
 
 
# Examine the counts of the most common words in positive reviews
positive_counts.most_common()
 
# Examine the counts of the most common words in negative reviews
negative_counts.most_common()
 
 
 
pos_neg_ratios = Counter()
 
# Calculate the ratios of positive and negative uses of the most common words
# Consider words to be "common" if they've been used at least 100 times
for term,cnt in list(total_counts.most_common()):
    if(cnt > 100):
        pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
        pos_neg_ratios[term] = pos_neg_ratio
 
#just to print the ratios    
 
print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))  
 
# Convert ratios to logs
for word,ratio in pos_neg_ratios.most_common():
    pos_neg_ratios[word] = np.log(ratio)
 
print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))
 
pos_neg_ratios.most_common()
 
# words most frequently seen in a review with a "NEGATIVE" label
list(reversed(pos_neg_ratios.most_common()))[0:30]
 
vocab = set(total_counts.keys())
vocab_size = len(vocab)
print(vocab_size)
layer_0 = np.zeros((1,vocab_size))
layer_0.shape
# Created a dictionary of words in the vocabulary mapped to index positions 
# (to be used in layer_0)
word2index = {}
for i,word in enumerate(vocab):
    word2index[word] = i
 
# display the map of words to indices
word2index
 
 
def update_input_layer(review):
    """ Modified the global layer_0 to represent the vector form of review.
    The element at a given index of layer_0 should represent
    how many times the given word occurs in the review.
    Args:
        review(string) - the string of the review
    Returns:
        None
    """
    global layer_0
 
    # cleared out previous state, reset the layer to be all 0s
    layer_0 *= 0
 
    # count how many times each word is used in the given review and store the results in layer_0 
    for word in review.split(" "):
        layer_0[0][word2index[word]] += 1
 
update_input_layer(testing_reviews[0])
layer_0  
 
def get_target_for_label(label):
    """Converted a label to `0` or `1`.
    Args:
        label(string) - Either "POSITIVE" or "NEGATIVE".
    Returns:
        `0` or `1`.
    """
    if(label == 'POSITIVE'):
        return 1
    else:
        return 0
 
 
#Encapsulated neural network in a class
class SentimentNetwork:
    def __init__(self, reviews,labels,hidden_nodes = 10, learning_rate = 0.1):
        """Create a SentimenNetwork with the following
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training
 
        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development 
        np.random.seed(1)
 
        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)
 
        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)
 
    def pre_process_data(self, reviews, labels):
 
        # populate review_vocab with all of the words in the given reviews
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                review_vocab.add(word)
 
        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
 
        # populate label_vocab with all of the words in the given labels.
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
 
        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)
 
        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
 
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
 
        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i
 
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
 
        # Store the learning rate
        self.learning_rate = learning_rate
 
        # Initialize weights
 
        # These are the weights between the input layer and the hidden layer.
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
 
        # These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
 
        # The input layer, a two-dimensional matrix with shape 1 x input_nodes
        self.layer_0 = np.zeros((1,input_nodes))
 
 
    def update_input_layer(self,review):
 
        # clear out previous state, reset the layer to be all 0s
        self.layer_0 *= 0
 
        for word in review.split(" "):
 
            if(word in self.word2index.keys()):
                self.layer_0[0][self.word2index[word]] = 1
 
    def get_target_for_label(self,label):
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0
 
    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
 
    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)
 
    def train(self, training_reviews, training_labels):
 
        # make sure out we have a matching number of reviews and labels
        assert(len(training_reviews) == len(training_labels))
 
        # Keep track of correct predictions to display accuracy during training 
        correct_so_far = 0
 
        # Remember when we started for printing time statistics
        start = time.time()
 
        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):
 
            # Get the next review and its correct label
            review = training_reviews[i]
            label = training_labels[i]
 
 
            # Input Layer
            self.update_input_layer(review)
 
            # Hidden layer
            layer_1 = self.layer_0.dot(self.weights_0_1)
 
            # Output layer
            layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))
 
            # Output error
            layer_2_error = layer_2 - self.get_target_for_label(label) # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)
 
            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity so it's the same as the error
 
            # Update the weights
            self.weights_1_2 -= layer_1.T.dot(layer_2_delta) * self.learning_rate # update hidden-to-output weights with gradient descent step
            self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * self.learning_rate # update input-to-hidden weights with gradient descent step
 
            # Keep track of correct predictions.
            if(layer_2 >= 0.5 and label == 'POSITIVE'):
                correct_so_far += 1
            elif(layer_2 < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1
 
            # For debug purposes, printed out the prediction accuracy and speed 
            # throughout the training process. 
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
 
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
 
    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
 
        # keep track of how many correct predictions we make
        correct = 0
 
        # timed how many predictions per second we make
        start = time.time()
 
        # Loop through each of the given reviews and call run to predict
        # its label. 
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
 
            # For debugging purposes, I printed out my prediction accuracy and speed 
            # throughout the prediction process. 
 
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
 
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
 
    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # Run a forward pass through the network, like in the "train" function.
 
        # Input Layer
        self.update_input_layer(review.lower())
 
        # Hidden layer
        layer_1 = self.layer_0.dot(self.weights_0_1)
 
        # Output layer
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))
 
        # Return POSITIVE for values above greater-than-or-equal-to 0.5 in the output layer;
        # return NEGATIVE for other values
        if(layer_2[0] >= 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"
 
mlp = SentimentNetwork(testing_reviews[:-1000],testing_labels[:-1000], learning_rate=0.1)
mlp.train(testing_reviews[:-1000],testing_labels[:-1000])
mlp.test(testing_reviews[-1000:],testing_labels[-1000:])
