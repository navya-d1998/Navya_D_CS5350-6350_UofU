import numpy as np
import copy

# This class is used to construct the branch from a given node.
# in order to expand from current node, we need to know:
# the current depth, the current feature_name and the number of items with different labels.
# the current depth, and also we need to generate childnode, and determine whether it is the end node(which is so called leaf)
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
class ID3:
    
    def __init__(self, metric_selection = 'entropy', max_depth = 5, attribute_subset = 3):
        self.metric_selection = metric_selection
        self.max_depth = max_depth
        self.attribute_subset = attribute_subset
        
    def set_metric_selection(self, metric_selection):
        self.metric_selection = metric_selection
    
    def set_max_depth(self,max_depth):
        self.max_depth = max_depth
    
    def set_subset_size(self,attribute_subset):
        self.attribute_subset = attribute_subset
    
    # compute entropy
    def compute_entropy(self,train_data, label):
        label_indx, label_name = list(label.items())[0]
        total = len(train_data.index)
        if total == 0:
            return 0
        entropy = 0
        for value in label_name:
            prob = len(train_data[train_data[label_indx] == value])/total
            if prob != 0:
                entropy += -prob * np.log2(prob)
        return entropy
    
    # compute majority error
    def compute_ME(self, train_data, label):
        label_indx, label_name = list(label.items())[0]
        total = train_data.shape[0]
        if total == 0:
            return 0
        max_prob = 0
        for value in label_name:
            prob = len(train_data[train_data[label_indx] == value])/total
            max_prob = max(max_prob, prob)
        return 1 - max_prob
    
    # compute gini index
    def compute_gini(self, train_data, label):
        label_indx, label_name = list(label.items())[0]
        total = train_data.shape[0]
        if total == 0:
            return 0
        square_sum = 0
        for value in label_name:
            prob = len(train_data[train_data[label_indx] == value])/total
            square_sum += prob**2
        return 1 - square_sum
    
    # compute information gain
    def compute_info_gain(self, feature_name,feature_value ,label, train_data):
        #features = current_tree['features']
        #label = current_tree['label']
        #train_data = current_tree['train_data']
        metric = None
        total = train_data.shape[0]
        if self.metric_selection == 'entropy':
            metric = self.compute_entropy
        elif self.metric_selection == 'major_error':
            metric = self.compute_ME
        elif self.metric_selection == 'gini_index':
            metric = self.compute_gini
        chaos = metric(train_data, label)
        
        gain = 0
        for value in feature_value:
            subset = train_data[train_data[feature_name] == value]
            prob = subset.shape[0]/total
            gain += prob*metric(subset, label)
        gain = chaos - gain
        return chaos,gain
    
    # find the most_informative feature.
    def find_most_informative_feature(self, features, label, train_data):
        max_gain = -1
        max_feature = None
        # Sample a subset of attributes
        keys = list(features.keys())
        if len(keys) > self.attribute_subset:
            sampled_features = np.random.choice(keys, self.attribute_subset, replace=False)
        else:
            sampled_features = keys
        
        for feature_name in sampled_features:
            gain = 0
            feature_value = features[feature_name]
            chaos,gain = self.compute_info_gain(feature_name, feature_value, label, train_data)
            if gain > max_gain:
                max_gain = gain
                max_feature = feature_name
        return max_feature
    
    # generate sub tree        
    def generate_sub_tree(self, current_tree):
        node_list = []
        features = current_tree['features']
        label = current_tree['label']
        tree_node = current_tree['tree_node']
        train_data = current_tree['train_data']
        
        metric = None
        if self.metric_selection == 'entropy':
            metric = self.compute_entropy
        elif self.metric_selection == 'major_error':
            metric = self.compute_ME
        elif self.metric_selection == 'gini_index':
            metric = self.compute_gini
            
        total = train_data.shape[0]
        label_indx, label_name = list(label.items())[0]
        if total > 0:
            majority_label = train_data[label_indx].value_counts().idxmax()
        #compute the purity given train_data and labels.
        chaos = metric(train_data,label)
        
        # if there is only one label in some feature value data
        # or the decision tree achieves the max length
        # or this node doesn't have corresponding label (use majority label)
        if chaos == 0 or tree_node.get_depth() == self.max_depth or len(features.items())==0:
            tree_node.set_leaf()
            if total > 0:
                tree_node.set_label(majority_label)
            return node_list
        
        # else, the node is not pure, we need to compute the max_feature and expand the tree further.
        max_feature = self.find_most_informative_feature(features, label, train_data)
        
        
        
        child = {}
        tree_node.set_feature(max_feature)
        remaining_feature = copy.deepcopy(features)
        # delete the max_feature (since it will be used at current node)
        remaining_feature.pop(max_feature, None)
        
        for feature_value in features[max_feature]:
            childNode = TreeNode()
            # increase depth
            childNode.set_depth(tree_node.get_depth()+1)
            childNode.set_label(majority_label)
            child[feature_value] = childNode
            pNode = {'train_data': train_data[train_data[max_feature] == feature_value],
                     'features': copy.deepcopy(remaining_feature), 'label': label, 
                     'tree_node': childNode}
            node_list.append(pNode)
        tree_node.set_child(child)
        return node_list
    
    def generate_decision_tree(self, train_data, features, label):
        Q = []
        # initialization here.
        tree_root = TreeNode()
        tree_root.set_depth(0)
        # processing node root
        root = {'train_data': train_data,'features': features, 
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
    
    

        






