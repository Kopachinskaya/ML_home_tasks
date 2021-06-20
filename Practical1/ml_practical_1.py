#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # 1.Use the Tree data structure below; write code to build the tree from figure 1.2 in Daumé.

# In[2]:


class Tree:

    def leaf(data):
        return Tree(data=data)

    def __repr__(self):
        if self.is_leaf():
            return "Leaf(%r)" % self.data
        else:
            return "Tree(%r) { left = %r, right = %r }" % (self.data, self.left, self.right) 

    def __init__(self, *, data = None, left = None, right = None):
        self.data = data
        self.left = left
        self.right = right

    def is_leaf(self):
        
        return self.left == None and self.right == None

    def children(self):
    
        return [x for x in [self.left, self.right] if x]

    def depth(self):
        return max([x.depth() for x in self.children()], default=0) + 1

l1 = Tree.leaf("like")
l2 = Tree.leaf("nah")
l3 = Tree.leaf("morning?")
l4 = Tree.leaf("ikedOtherSys?")
tree = Tree(data="isSystems?", left=l1, right=Tree(data = 'takenOtherSys?', left=Tree(data = l3, left = l1, right = l2), right = Tree(data = l4, left = l2, right = l1)))


# In[3]:


tree


# # 2.In your python code, load the following dataset and add a boolean "ok" column, where "True" means the rating is non-negative and "False" means the rating is negative.

# In[4]:


ml_data = pd.DataFrame([[2,True,True,False,True,False],
                        [2,True,True,False,True,False],
                        [2,False,True,False,False,False],
                        [2,False,False,False,True,False],
                        [2,False,True,True,False,True],
                        [1,True,True,False,False,False],
                        [1,True,True,False,True,False],
                        [1,False,True,False,True,False],
                        [0,False,False,False,False,True],
                        [0,True,False,False,True,True],
                        [0,False,True,False,True,False],
                        [0,True,True,True,True,True],
                        [-1,True,True,True,False,True],
                        [-1,False,False,True,True,False],
                        [-1,False,False,True,False,True],
                        [-1,True,False,True,False,True],
                        [-2,False,False,True,True,False],
                        [-2,False,True,True,False,True],
                        [-2,True,False,True,False,False],
                        [-2,True,False,True,False,True]],
                        columns= ['rating','easy','ai','systems','theory',
                                  'morning'])

ml_data.to_csv('ml_data.csv',index=False)


# In[5]:


ml = pd.read_csv('ml_data.csv')
ml['ok'] = ml.rating >= 0


# In[6]:


ml


# # 3.Write a function which takes a feature and computes the performance of the corresponding single-feature classifier:

# In[7]:


def single_feature_score(data, goal, feature):
    yes = data[data[feature] == True][goal]
    no = data[data[feature] == False][goal]
    result = (np.sum(yes.value_counts().idxmax()==yes)+np.sum(no.value_counts().idxmax()==no))/len(data)
    return result
    
    


# In[13]:


def best_feature(data, goal, features):
      return max(features, key=lambda f: single_feature_score(data, goal, f))


# In[14]:


best_feature(ml, 'ok', ['rating','easy','ai','systems','theory',
                                  'morning'])


# # 4.Implement the DecisionTreeTrain and DecisionTreeTest algorithms from Daumé, returning Trees. (Note: our dataset and his are different; we won't get the same tree.)

# In[19]:


def DecisionTreeTraining(data, goal, features):
    guess = data[goal].value_counts().idxmax()
    if np.all(data[goal] == data[goal].iloc[0]):
        return Tree.leaf(guess)
    elif not features:
        return Tree.leaf(guess)
    else:
        for feature in features:
            f = best_feature(data, goal, features)
            yes = data[data[f] == True]
            no = data[data[f] == False]
            features.remove(f)
            left = DecisionTreeTraining(data = no, goal = goal, features = features)
            right = DecisionTreeTraining(data = yes, goal = goal, features = features)
        return Tree(data = f, left = left, right = right)


# In[18]:


def DecisionTreeTest(tree, test_point):
    if tree.is_leaf():
        return tree.data
    else:
        if test_point[tree.data] == False:
            return DecisionTreeTest(tree.left, test_point)
        else:
            return (tree.right, test_point)
    

