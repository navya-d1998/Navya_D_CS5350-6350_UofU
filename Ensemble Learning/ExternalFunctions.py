import numpy as np
import copy
import pandas as pd
class TreeNode:
    # initialization of all the mentioned variables.
    def __init__(self):
        self.feature = None
        self.child = None
        self.depth = -1
        self.isLeaf = False
        self.label = None
    
    def set_feature(self, feature):
        self.feature = feature

    def set_child(self, child):
        self.child = child

    def get_depth(self):
        return self.depth

    def set_depth(self, depth):
        self.depth = depth

    def set_leaf(self):
        self.isLeaf = True

    def set_label(self, label):
        self.label = label

    def is_leaf(self):
        return self.isLeaf
    
# This class is used to include all the operations to construct a decision tree.
# we need: 1. define three different metrics: entropy, ME error and gini index.
#2.compute information gain; 3. find the most informative feature 4. generate sub tree
# 5. generate tree (from root to first-level nodes), then we call 4 recursively. 
class WeightedID3:
    
    def __init__(self, metric_selection = 'entropy', max_depth = 2):
        self.metric_selection = metric_selection
        self.max_depth = max_depth
        
    def set_metric_selection(self, metric_selection):
        self.metric_selection = metric_selection
    
    def set_max_depth(self,max_depth):
        self.max_depth = max_depth
    
    # compute entropy
    def compute_entropy(self,train_data, label, weights):
        ## label: a dictonary with label index, i.e 'y', label names: 'yes','no'
        # weights: the long vector recording data's weights.
        label_indx, label_name = list(label.items())[0]
        total = np.sum(weights)
        column = np.array(train_data[label_indx].tolist())
        #total = len(train_data.index)
        if total == 0:
            return 0
        entropy = 0
        for value in label_name:
            weighted_data = weights[column == value]
            prob = np.sum(weighted_data)/total
            #prob = len(train_data[train_data[label_indx] == value])/total
            if prob != 0:
                entropy += -prob * np.log2(prob)
        return entropy
    
    # compute majority error
    def compute_ME(self, train_data, label, weights):
        label_indx, label_name = list(label.items())[0]
        #total = train_data.shape[0]
        total = np.sum(weights)
        column = np.array(train_data[label_indx].tolist())
        if total == 0:
            return 0
        max_prob = 0
        for value in label_name:
            weighted_data = weights[column == value]
            prob = np.sum(weighted_data)/total
            #prob = len(train_data[train_data[label_indx] == value])/total
            max_prob = max(max_prob, prob)
        return 1 - max_prob
    
    # compute gini index
    def compute_gini(self, train_data, label, weights):
        label_indx, label_name = list(label.items())[0]
        #total = train_data.shape[0]
        total = np.sum(weights)
        column = np.array(train_data[label_indx].tolist())
        if total == 0:
            return 0
        square_sum = 0
        for value in label_name:
            weighted_data = weights[column == value]
            #prob = len(train_data[train_data[label_indx] == value])/total
            prob = np.sum(weighted_data)/total
            square_sum += prob**2
        return 1 - square_sum
    
    # compute information gain
    def compute_info_gain(self, feature_name,feature_value ,label, train_data,weights):
        #features = current_tree['features']
        #label = current_tree['label']
        #train_data = current_tree['train_data']
        metric = None
        #total = train_data.shape[0]
        total = np.sum(weights)
        column = np.array(train_data[feature_name].tolist())
        if self.metric_selection == 'entropy':
            metric = self.compute_entropy
        elif self.metric_selection == 'major_error':
            metric = self.compute_ME
        elif self.metric_selection == 'gini_index':
            metric = self.compute_gini
        chaos = metric(train_data, label, weights)
        
        gain = 0
        for value in feature_value:
            w = weights[column == value]
            sub_weights = w
            subset = train_data[train_data[feature_name] == value]
            #prob = subset.shape[0]/total
            prob = np.sum(sub_weights) / total
            gain += prob*metric(subset, label, sub_weights)
        gain = chaos - gain
        return chaos,gain
    
    # find the most_informative feature.
    def find_most_informative_feature(self, features, label, train_data, weights):
        max_gain = -1
        max_feature = None
        
        for feature_name, feature_value in features.items():
            chaos,gain = self.compute_info_gain(feature_name,feature_value ,label, train_data,weights)
            if max_gain < gain:
                max_gain = gain
                max_feature = feature_name
        return max_feature
    
    def get_majority_label(self, train_data, label, weights):
        label_indx, label_name = list(label.items())[0]
        majority_label = None
        max_sum = -1
        column = np.array(train_data[label_indx].tolist())
        for value in label_name:
            w = weights[column == value]
            weight_sum = np.sum(w)
            if weight_sum > max_sum:
                majority_label = value
                max_sum = weight_sum
        
        return majority_label
    
    # generate sub tree        
    def generate_sub_tree(self, current_tree):
        node_list = []
        features = current_tree['features']
        label = current_tree['label']
        tree_node = current_tree['tree_node']
        train_data = current_tree['train_data']
        weights = current_tree['weights']
        
        metric = None
        if self.metric_selection == 'entropy':
            metric = self.compute_entropy
        elif self.metric_selection == 'major_error':
            metric = self.compute_ME
        elif self.metric_selection == 'gini_index':
            metric = self.compute_gini
            
        #total = train_data.shape[0]
        total = np.sum(weights)
        label_indx, label_name = list(label.items())[0]
        if total > 0:
            #majority_label = train_data[label_indx].value_counts().idxmax()
            majority_label = self.get_majority_label(train_data, label, weights)
        #compute the purity given train_data and labels.
        chaos = metric(train_data,label,weights)
        
        # if there is only one label in some feature value data
        # or the decision tree achieves the max length
        # or this node doesn't have corresponding label (use majority label)
        if chaos == 0 or tree_node.get_depth() == self.max_depth or len(features.items())==0:
            tree_node.set_leaf()
            if total > 0:
                tree_node.set_label(majority_label)
            return node_list
        
        # else, the node is not pure, we need to compute the max_feature and expand the tree further.
        max_feature = self.find_most_informative_feature(features, label, train_data,weights)
        
        child = {}
        tree_node.set_feature(max_feature)
        remaining_feature = copy.deepcopy(features)
        # delete the max_feature (since it will be used at current node)
        remaining_feature.pop(max_feature, None)
        column = np.array(train_data[max_feature].tolist())
        for feature_value in features[max_feature]:
            childNode = TreeNode()
            # increase depth
            childNode.set_depth(tree_node.get_depth()+1)
            childNode.set_label(majority_label)
            child[feature_value] = childNode
            w = weights[column == feature_value]
            pNode = {'train_data': train_data[train_data[max_feature] == feature_value],
                     'weights':w,
                     'features': copy.deepcopy(remaining_feature), 'label': label, 
                     'tree_node': childNode}
            node_list.append(pNode)
        tree_node.set_child(child)
        return node_list
    
    def generate_decision_tree(self, train_data, features, label, weights):
        Q = []
        # initialization here.
        tree_root = TreeNode()
        tree_root.set_depth(0)
        # processing node root
        root = {'train_data': train_data,'weights':weights,'features': features, 
                'label': label, 'tree_node': tree_root}
        Q.append(root)
        while len(Q) > 0:
            # recursive call starts from here.
            current_tree = Q.pop(0)
            nodes = self.generate_sub_tree(current_tree)
            for node in nodes:
                Q.append(node)
                #print(Q)
        return tree_root
    
    #compute prediction for one test sample
    def classify_each_row(self,dt, test_data):
        predict = dt
        while not predict.is_leaf():
            predict = predict.child[test_data[predict.feature]]
        return predict.label
    
    # apply previous function to the whole row.
    def classify(self, dt, test_data):
        #return pd.DataFrame(predicted_label)
        return test_data.apply(lambda row: self.classify_each_row(dt, row), axis=1)