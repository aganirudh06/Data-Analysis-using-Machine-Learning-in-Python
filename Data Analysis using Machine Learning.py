#!/usr/bin/env python
# coding: utf-8

# # Experiment 1
# 

# In[48]:


# Importing required packages

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage

# Importing coursework1 data from csv file

data = pd.read_csv("coursework1.csv")

# converting sourceIP, destIP and classification columns into seperate lists

allsourceip = data['sourceIP'].tolist()
alldestinationip=data['destIP'].tolist()
allclassification=data['classification'].tolist()

# Calculating number of unique source IP addresses without using predetermined set function.
output1 = []
for x in allsourceip:
    if x not in output1:
        output1.append(x)

num1=len(output1)
print('Number of unique elements in sourceIP are', num1)

# Calculating number of unique destination IP addresses without using predetermined set function.
output2 = []
for x in alldestinationip:
    if x not in output2:
        output2.append(x)
num2=len(output2)
print('Number of unique elements in destinationIP are', num2)

# Calculating number of unique classifications without using predetermined set function.
output3 = []
for x in allclassification:
    if x not in output3:
        output3.append(x)
num3=len(output3)
print('Number of unique classifications are', num3)


# # Experiment 2

# In[52]:


# Generating list of number of records for each unique source IP address
i=0
countlist1=[];
while i<num1:
   
    
    newnew=output1[i]
    
    output1numbers=allsourceip.count(newnew);
    countlist1.append(output1numbers)
    i+=1;

# Generating lists containing lists of index and respective number of counts for that source IP address
j=0
scatter=[]
while j<num1:
    scasets=[j,countlist1[j]]
    
    scatter.append(scasets)
    j+=1;


# Generating list of number of records for each unique destination IP address
k=0
countlist2=[];
while k<num2:
       
    newnew1=output2[k]
    
    output2numbers=alldestinationip.count(newnew1);
    countlist2.append(output2numbers)
    k+=1;

# Generating lists containing lists of index and respective number of counts for that destination IP address
l=0
scatter2=[]
while l<num2:
    scasets2=[l,countlist2[l]]
    
    scatter2.append(scasets2)
    l+=1;

# Plotting histogram to show the number of records for source IP addresses 
plt.hist(allsourceip,bins=98)
plt.xlabel("Index of IP addresses")
plt.ylabel("No.of records")


# ##### Plotting the number of records for destination IP addresses

# In[53]:


# Plotting histogram to show the number of records for destination IP addresses

plt.hist(alldestinationip,bins=261)
plt.xlabel("Index of IP addresses")
plt.ylabel("No.of records")


# # Experiment 3

# ##### Elbow Plot for source IP addresses

# In[11]:


# Converting both source IP and destination IP values into numpy 2D array to plot

X=np.array(scatter)
Y=np.array(scatter2)

# Elbow plot for source IP addresses

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X,kmeanModel.cluster_centers_,'euclidean')**2, axis=1)) / X.shape[0])



plt.figure()
plt.plot(K, distortions, 'bx-')
plt.xlabel('K')
plt.ylabel('Distortions')
plt.title('The Elbow Method showing the optimal k')


# ##### Elbow Plot for destination IP addresses

# In[13]:


# Elbow plot for destination IP addresses

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(Y)
    kmeanModel.fit(Y)
    distortions.append(sum(np.min(cdist(Y,kmeanModel.cluster_centers_,'euclidean')**2, axis=1)) / Y.shape[0])



plt.figure()
plt.plot(K, distortions, 'bx-')
plt.xlabel('K')
plt.ylabel('Distortions')
plt.title('The Elbow Method showing the optimal k')


# ##### Plotting K-means clustering for source IP addresses

# In[14]:


kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit_predict(X)

# Scattered plot 

plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s = 100, c = "red", label = "Cluster1")
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s = 100, c = "blue", label = "Cluster2")
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s = 100, c = "green", label = "Cluster3")        

plt.title("Clustering using Kmeans")
plt.xlabel("Index of unique source IP addresses")
plt.ylabel("No.of records")
plt.legend()
plt.show()


# ##### Plotting K-means clustering for destination IP addresses

# In[15]:


kmeans = KMeans(n_clusters=2)
y_kmeans = kmeans.fit_predict(Y)

# scattered plot

plt.scatter(Y[y_kmeans == 0,0], Y[y_kmeans == 0,1], s = 100, c = "red", label = "Cluster1")
plt.scatter(Y[y_kmeans == 1,0], Y[y_kmeans == 1,1], s = 100, c = "blue", label = "Cluster2")       

plt.title("Clustering using Kmeans")
plt.xlabel("Index of unique source IP addresses")
plt.ylabel("No.of records")
plt.legend()
plt.show()


# ##### Plotting Hierarchical clustering for source IP addresses

# In[16]:


linked = linkage(X, 'single')
labelList = range(0, len(X))
plt.figure(figsize=(10, 7))
dendrogram(linked,labels=labelList)
plt.title("Hierarchical clustering")
plt.show()


# ##### Plotting Hierarchical clustering for source IP addresses

# In[17]:


linked = linkage(Y, 'single')
labelList = range(0, len(Y))
plt.figure(figsize=(10, 7))
dendrogram(linked,labels=labelList)
plt.title("Hierarchical clustering")
plt.show()


# # Experiment 4

# In[54]:


# Clustering source IP values into four clusters as per conditions given in Experiment 4

S1=[]
S2=[]
S3=[]
S4=[]
for i in output1:
    if allsourceip.count(i)<21:
        S1.append(i)
    elif 20<allsourceip.count(i)<201:
        S2.append(i)
    elif 200<allsourceip.count(i)<401:
        S3.append(i)
    elif allsourceip.count(i)>400:
        S4.append(i)

# Clustering destination IP values into four clusters as per conditions given in Experiment 4

D1=[]
D2=[]
D3=[]
D4=[]
for i in output2:
    if alldestinationip.count(i)<41:
        D1.append(i)
    elif 40<alldestinationip.count(i)<101:
        D2.append(i)
    elif 100<alldestinationip.count(i)<401:
        D3.append(i)
    elif alldestinationip.count(i)>400:
        D4.append(i)


# Calculating relation between values of source IP cluster 1 and different destination IP clusters

s1d1=0;
s1d2=0;
s1d3=0;
s1d4=0;

for i in S1:
       checkindex=[index for index, value in enumerate(allsourceip) if value == i]
      
       for j in checkindex:
           
           if alldestinationip[j] in D1:
               s1d1 +=1;
           elif alldestinationip[j] in D2:
               s1d2 +=1;
           elif alldestinationip[j] in D3:
               s1d3 +=1;
           elif alldestinationip[j] in D4:
               s1d4 +=1;


# Calculating probabilities of destination IP clusters given the probability of source IP cluster 1

sum1 = s1d1+s1d2+s1d3+s1d4;

if sum1 !=0:
    s1d1 = (s1d1/sum1)
    s1d2 = (s1d2/sum1)
    s1d3 = (s1d3/sum1)
    s1d4 = (s1d4/sum1)    

# Calculating relation between values of source IP cluster 2 and different destination IP clusters

s2d1=0;
s2d2=0;
s2d3=0;
s2d4=0;
for i in S2:
       checkindex=[index for index, value in enumerate(allsourceip) if value == i]
       
       for j in checkindex:
           
           if alldestinationip[j] in D1:
               s2d1 += 1;  
           elif alldestinationip[j] in D2:
               s2d2 += 1;
               
           elif alldestinationip[j] in D3:
               s2d3 += 1;
               
           elif alldestinationip[j] in D4:
               s2d4 += 1;

# Calculating probabilities of destination IP clusters given the probability of source IP cluster 2

sum2 = s2d1+s2d2+s2d3+s2d4;

if sum2 !=0:
    s2d1 = (s2d1/sum2)
    s2d2 = (s2d2/sum2)
    s2d3 = (s2d3/sum2)
    s2d4 = (s2d4/sum2)    

# Calculating relation between values of source IP cluster 3 and different destination IP clusters

s3d1=0;
s3d2=0;
s3d3=0;
s3d4=0;
for i in S3:
       checkindex=[index for index, value in enumerate(allsourceip) if value == i]
       
       for j in checkindex:
           
           if alldestinationip[j] in D1:
               s3d1 += 1;  
           elif alldestinationip[j] in D2:
               s3d2 += 1;
               
           elif alldestinationip[j] in D3:
               s3d3 += 1;
               
           elif alldestinationip[j] in D4:
               s3d4 += 1;

# Calculating probabilities of destination IP clusters given the probability of source IP cluster 3
               
sum3 = s3d1+s3d2+s3d3+s3d4;

if sum3 !=0:
    s3d1 = (s3d1/sum3)
    s3d2 = (s3d2/sum3)
    s3d3 = (s3d3/sum3)
    s3d4 = (s3d4/sum3)    


# Calculating relation between values of source IP cluster 4 and different destination IP clusters
s4d1=0;
s4d2=0;
s4d3=0;
s4d4=0;
for i in S4:
       checkindex=[index for index, value in enumerate(allsourceip) if value == i]
       
       for j in checkindex:
           
           if alldestinationip[j] in D1:
               s4d1 += 1;  
           elif alldestinationip[j] in D2:
               s4d2 += 1;
               
           elif alldestinationip[j] in D3:
               s4d3 += 1;
               
           elif alldestinationip[j] in D4:
               s4d4 += 1;

# Calculating probabilities of destination IP clusters given the probability of source IP cluster 4

sum4 = s4d1+s4d2+s4d3+s4d4;

if sum4 !=0:
    s4d1 = (s4d1/sum4)
    s4d2 = (s4d2/sum4)
    s4d3 = (s4d3/sum4)
    s4d4 = (s4d4/sum4)      
        

# Plotting conditional probabilities between source IP clusters and destination IP clusters
  
# labels for bars
x =['Dest.1','Dest.2','Dest.3','Dest.4']

# Relation between source IP cluster 3 and destination IP clusters  

y=[s1d1,s1d2,s1d3,s1d4]

fig, ax = plt.subplots()    
width = 0.75 # the width of the bars 
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('Probabilities of destination IP clusters given source IP cluster 1')
plt.xlabel('Probability')
plt.ylabel('Destination IP clusters')      
for i, v in enumerate(y):
    ax.text(v, i, str(v), color='blue', fontweight='bold')
plt.show()

# Relation between source IP cluster 2 and destination IP clusters  

new1=[s2d1,s2d2,s2d3,s2d4]

fig, ax = plt.subplots()    
width = 0.75 # the width of the bars 
ind = np.arange(len(new1))  # the x locations for the groups
ax.barh(ind, new1, width, color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('Probabilities of destination IP clusters given source IP cluster 2')
plt.xlabel('Probability')
plt.ylabel('Destination IP clusters')  
for i, v in enumerate(new1):
    ax.text(v, i , str(v), color='blue', fontweight='bold')
    
# Relation between source IP cluster 3 and destination IP clusters  

y=[s3d1,s3d2,s3d3,s3d4]

fig, ax = plt.subplots()    
width = 0.75 # the width of the bars 
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('Probabilities of destination IP clusters given source IP cluster 3')
plt.xlabel('Probability')
plt.ylabel('Destination IP clusters')       
for i, v in enumerate(y):
    ax.text(v, i, str(v), color='blue', fontweight='bold')


# Relation between source IP cluster 4 and destination IP clusters  

y=[s4d1,s4d2,s4d3,s4d4]

fig, ax = plt.subplots()    
width = 0.75 # the width of the bars 
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('Probabilities of destination IP clusters given source IP cluster 4')
plt.xlabel('Probability')
plt.ylabel('Destination IP clusters')      
for i, v in enumerate(y):
    ax.text(v, i, str(v), color='blue', fontweight='bold')


# # Experiment 5

# In[43]:


import seaborn as sns
import random
from pprint import pprint


# Importing coursework1 data

df = pd.read_csv("coursework1.csv")

# Dropping unneccesary columns from the coursework1 data

df=df.drop("time", axis=1)
df=df.drop("sourcePort",axis=1)
df=df.drop("destPort",axis=1)
df=df.drop("priority",axis=1)
df=df.drop("label",axis=1)
df=df.drop("packet info",axis=1)
df=df.drop("packet info cont'd",axis=1)
df=df.drop("xref",axis=1)
df = df.rename(columns={"classification": "label"})
df.head()


# Splitting data into train data and test data

def train_test_split(df, test_size):
    if isinstance(test_size,float):
        test_size=round(test_size * len(df))


    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    
    return train_df, test_df



# Data purity check

def check_purity(data):
    label_column = data[:,-1]
    unique_classes=np.unique(label_column)
    
    if len(unique_classes) ==1:         # Checking if there is only one unique element in the classification. If yes, the decision tree stops.
        return True
    else:
        return False


# Classify function
        
def classify_data(data):
    label_column = data[:,-1]
    unique_classes, counts_unique_classes = np.unique(label_column,return_counts=True)    # Counting the unique classifications and their number of records
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
        
    return classification


# Calculating potential split values
    
def get_potential_splits(data):
    potential_splits= {}
    _, n_columns = data.shape
    
    for column_index in range(n_columns - 1):
        
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        
        potential_splits[column_index] = unique_values        # Storing unique values of sourceIP and destIP columns wich are possible values for best split with minimum entropy  
            
                    
    return potential_splits
         

# Split Data Function
    
def split_data(data, split_column, split_value):
    split_column_values = data[:, split_column]
        
    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature =="continuous":                             # Splitting continuous and categorical data accordingly.
    
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values > split_value]
        
    else:       
        data_below = data[split_column_values == split_value]       # Splitting the IP addresses in the split_value column into two groups. Addresses equal to split_value and not equal to. 
        data_above = data[split_column_values != split_value]
    return data_below, data_above




# Calculating overall Entropy
    
def calculate_entropy(data):
    label_column=data[:,-1]
    _,counts = np.unique(label_column, return_counts=True)
    
#    print(counts)
    probabilities = counts/counts.sum()
    
    entropy = sum(probabilities *  -np.log2(probabilities))         # Calculating entropy for value that is run in this function
#    entropy = 1-sum(probabilities*probabilities)
    return entropy



def calculate_overall_entropy(data_below,data_above):
    n_data_points=len(data_below) + len(data_above)

    p_data_below=len(data_below)/n_data_points                    # Calculating probabilities of two groups of data to calculate overall entropy
    p_data_above=len(data_above)/n_data_points
    
    overall_entropy=(p_data_below * calculate_entropy(data_below) + p_data_above * calculate_entropy(data_above))       # Calculating overall entropy
    
    return overall_entropy



# Evaluating and storing the best split value

def determine_best_split(data,potential_splits):
    overall_entropy=9999
    for column_index in potential_splits:
        
        for value in potential_splits[column_index]:
            
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)  # Calling split_data funtion to split data into  two groups
            current_overall_entropy = calculate_overall_entropy(data_below,data_above)               # Calculating overall entropy of groups
            
            if current_overall_entropy < overall_entropy:
                
                overall_entropy = current_overall_entropy
                best_split_column=column_index                                                     # Choosing the best split column from source and dest IPs based on which column the best split value (IP address) belongs to
                best_split_value=value

    
    return best_split_column,best_split_value



# Determining type of feature

def determine_type_of_feature(df):
    feature_types =[]
    
    n_unique_values_threshold = 100
    
    for column in df.columns:
        if column != "label":
            unique_values = df[column].unique()
            example_value=unique_values[0]
            
            if (isinstance(example_value,str)) or (len(unique_values) <= n_unique_values_threshold):       # Determining if each column has continuous or categorical values
                feature_types.append("categorical")
            else:
                feature_types.append("Continuous")
        
    
    return feature_types



# Decision Tree Algorithm
    
def decision_tree_algorithm(df, counter=0,min_samples=2,max_depth=5):
    
    # Data Preparation
    
    if counter == 0:
        global COLUMN_HEADERS,FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES=determine_type_of_feature(df)
        data = df.values
    else:
        data = df

    # Base cases
          
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth) :
        classification = classify_data(data)
        return classification
    
    #Recursive part
    else:
        counter += 1
        
        
        # Helper functions
        
        potential_splits = get_potential_splits(data)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # Checking for empty data to know if the data is already classified
        if len(data_below) == 0 or len(data_above) == 0:
            classification = classify_data(data)
            return classification
        

        # Determining question or condition that is printed with the decision tree
        feature_name=COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature=="continuous":

            question = "{} <= {}".format(feature_name, split_value)
        else:
            question = "{} = {}".format(feature_name, split_value)
        
        # Instantiating Subtree
        sub_tree = {question: []}
        
        # Finding answers for two groups divided based on best split (recurring)
        
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth)

        sub_tree[question].append(yes_answer)
        sub_tree[question].append(no_answer)
        
        return sub_tree
        





# Classification

def classify_example(example, tree):
    
    question = list(tree.keys())[0]
    
    feature_name, comparison_operator, value = question.split(" ")
    
    # Ask Question
    
    if comparison_operator == "<=":
        
    
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    else:
        if str(example[feature_name]) == value:             # Classifying the values that satisfies and doesnot satisfy the condition
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)



# Calculating Accuracy for the decision tree learnt
        
def calculate_accuracy(df, tree):
    
    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))
    df["classification_correct"] = df["classification"] == df["label"]
    
    accuracy = df["classification_correct"].mean()
    return accuracy



# Calling decision tree algorithm and accuracy calculation
    
train_df, test_df =train_test_split(df, test_size = 0.2)
tree=decision_tree_algorithm(train_df, max_depth = 5)
accuracy=calculate_accuracy(test_df, tree)


# Print decision tree and accuracy of the decision tree learnt

pprint(tree, width=50)
print("Accuracy = ", accuracy)


# # Experiment 6

# ##### Finding relationships for courcework2 data

# In[55]:


# Importing coursework2 data

data = pd.read_csv("coursework2.csv")

allsourceip = data['sourceIP'].tolist()
alldestinationip=data['destIP'].tolist()
allclassification=data['classification'].tolist()



# Calculating number of unique source IP addresses
output1 = []
for x in allsourceip:
    if x not in output1:
        output1.append(x)

num1=len(output1)

# Calculating number of unique destination IP addresses
output2 = []
for x in alldestinationip:
    if x not in output2:
        output2.append(x)
num2=len(output2)


# Generating list of number of records for each unique source IP address
i=0
countlist1=[];
while i<num1:
   
    
    newnew=output1[i]
    
    output1numbers=allsourceip.count(newnew);
    countlist1.append(output1numbers)
    i+=1;

# Generating lists containing lists of index and respective number of counts for that source IP address
j=0
scatter=[]
while j<num1:
    scasets=[j,countlist1[j]]
    
    scatter.append(scasets)
    j+=1;

X=np.array(scatter)

# Generating list of number of records for each unique destination IP address
k=0
countlist2=[];
while k<num2:
       
    newnew1=output2[k]
    
    output2numbers=alldestinationip.count(newnew1);
    countlist2.append(output2numbers)
    k+=1;

# Generating lists containing lists of index and respective number of counts for that destination IP address
l=0
scatter2=[]
while l<num2:
    scasets2=[l,countlist2[l]]
    
    scatter2.append(scasets2)
    l+=1;
Y=np.array(scatter2)

# Clustering source IP values into four clusters as per conditions given in Experiment 4

S1=[]
S2=[]
S3=[]
S4=[]
for i in output1:
    if allsourceip.count(i)<21:
        S1.append(i)
    elif 20<allsourceip.count(i)<201:
        S2.append(i)
    elif 200<allsourceip.count(i)<401:
        S3.append(i)
    elif allsourceip.count(i)>400:
        S4.append(i)

# Clustering destination IP values into four clusters as per conditions given in Experiment 4

D1=[]
D2=[]
D3=[]
D4=[]
for i in output2:
    if alldestinationip.count(i)<41:
        D1.append(i)
    elif 40<alldestinationip.count(i)<101:
        D2.append(i)
    elif 100<alldestinationip.count(i)<401:
        D3.append(i)
    elif alldestinationip.count(i)>400:
        D4.append(i)


# Calculating relation between values of source IP cluster 1 and different destination IP clusters

s1d1=0;
s1d2=0;
s1d3=0;
s1d4=0;

for i in S1:
       checkindex=[index for index, value in enumerate(allsourceip) if value == i]
      
       for j in checkindex:
           
           if alldestinationip[j] in D1:
               s1d1 +=1;
           elif alldestinationip[j] in D2:
               s1d2 +=1;
           elif alldestinationip[j] in D3:
               s1d3 +=1;
           elif alldestinationip[j] in D4:
               s1d4 +=1;


# Calculating probabilities of destination IP clusters given the probability of source IP cluster 1

sum1 = s1d1+s1d2+s1d3+s1d4;

if sum1 !=0:
    s1d1 = (s1d1/sum1)
    s1d2 = (s1d2/sum1)
    s1d3 = (s1d3/sum1)
    s1d4 = (s1d4/sum1)    

# Calculating relation between values of source IP cluster 2 and different destination IP clusters

s2d1=0;
s2d2=0;
s2d3=0;
s2d4=0;
for i in S2:
       checkindex=[index for index, value in enumerate(allsourceip) if value == i]
       
       for j in checkindex:
           
           if alldestinationip[j] in D1:
               s2d1 += 1;  
           elif alldestinationip[j] in D2:
               s2d2 += 1;
               
           elif alldestinationip[j] in D3:
               s2d3 += 1;
               
           elif alldestinationip[j] in D4:
               s2d4 += 1;

# Calculating probabilities of destination IP clusters given the probability of source IP cluster 2

sum2 = s2d1+s2d2+s2d3+s2d4;

if sum2 !=0:
    s2d1 = (s2d1/sum2)
    s2d2 = (s2d2/sum2)
    s2d3 = (s2d3/sum2)
    s2d4 = (s2d4/sum2)    

# Calculating relation between values of source IP cluster 3 and different destination IP clusters

s3d1=0;
s3d2=0;
s3d3=0;
s3d4=0;
for i in S3:
       checkindex=[index for index, value in enumerate(allsourceip) if value == i]
       
       for j in checkindex:
           
           if alldestinationip[j] in D1:
               s3d1 += 1;  
           elif alldestinationip[j] in D2:
               s3d2 += 1;
               
           elif alldestinationip[j] in D3:
               s3d3 += 1;
               
           elif alldestinationip[j] in D4:
               s3d4 += 1;

# Calculating probabilities of destination IP clusters given the probability of source IP cluster 3
               
sum3 = s3d1+s3d2+s3d3+s3d4;

if sum3 !=0:
    s3d1 = (s3d1/sum3)
    s3d2 = (s3d2/sum3)
    s3d3 = (s3d3/sum3)
    s3d4 = (s3d4/sum3)    


# Calculating relation between values of source IP cluster 4 and different destination IP clusters
s4d1=0;
s4d2=0;
s4d3=0;
s4d4=0;
for i in S4:
       checkindex=[index for index, value in enumerate(allsourceip) if value == i]
       
       for j in checkindex:
           
           if alldestinationip[j] in D1:
               s4d1 += 1;  
           elif alldestinationip[j] in D2:
               s4d2 += 1;
               
           elif alldestinationip[j] in D3:
               s4d3 += 1;
               
           elif alldestinationip[j] in D4:
               s4d4 += 1;

# Calculating probabilities of destination IP clusters given the probability of source IP cluster 4

sum4 = s4d1+s4d2+s4d3+s4d4;

if sum4 !=0:
    s4d1 = (s4d1/sum4)
    s4d2 = (s4d2/sum4)
    s4d3 = (s4d3/sum4)
    s4d4 = (s4d4/sum4)      
        

        
# Plotting conditional probabilities between source IP clusters and destination IP clusters
 
# labels for bars 
x =['Dest.1','Dest.2','Dest.3','Dest.4']

# Relation between source IP cluster 3 and destination IP clusters  

y=[s1d1,s1d2,s1d3,s1d4]

fig, ax = plt.subplots()    
width = 0.75 # the width of the bars 
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('Probabilities of destination IP clusters given source IP cluster 1')
plt.xlabel('Probability')
plt.ylabel('Destination IP clusters')      
for i, v in enumerate(y):
    ax.text(v, i, str(v), color='blue', fontweight='bold')
plt.show()

# Relation between source IP cluster 2 and destination IP clusters  

new1=[s2d1,s2d2,s2d3,s2d4]

fig, ax = plt.subplots()    
width = 0.75 # the width of the bars 
ind = np.arange(len(new1))  # the x locations for the groups
ax.barh(ind, new1, width, color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('Probabilities of destination IP clusters given source IP cluster 2')
plt.xlabel('Probability')
plt.ylabel('Destination IP clusters')  
for i, v in enumerate(new1):
    ax.text(v, i , str(v), color='blue', fontweight='bold')
    
# Relation between source IP cluster 3 and destination IP clusters  

y=[s3d1,s3d2,s3d3,s3d4]

fig, ax = plt.subplots()    
width = 0.75 # the width of the bars 
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('Probabilities of destination IP clusters given source IP cluster 3')
plt.xlabel('Probability')
plt.ylabel('Destination IP clusters')       
for i, v in enumerate(y):
    ax.text(v, i, str(v), color='blue', fontweight='bold')


# Relation between source IP cluster 4 and destination IP clusters  

y=[s4d1,s4d2,s4d3,s4d4]

fig, ax = plt.subplots()    
width = 0.75 # the width of the bars 
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('Probabilities of destination IP clusters given source IP cluster 4')
plt.xlabel('Probability')
plt.ylabel('Destination IP clusters')      
for i, v in enumerate(y):
    ax.text(v, i, str(v), color='blue', fontweight='bold')


# ##### Decision tree for coursework2 data

# In[41]:


## Using the same decision tree algorithm from the Experiment 5

# Importing coursework2 data

df = pd.read_csv("coursework2.csv")

# Dropping columns that are unneccessary for classification

df=df.drop("time", axis=1)
df=df.drop("sourcePort",axis=1)
df=df.drop("destPort",axis=1)
df=df.drop("priority",axis=1)
df=df.drop("label",axis=1)
df=df.drop("packet info",axis=1)
df=df.drop("packet info cont'd",axis=1)
df=df.drop("xref",axis=1)
df = df.rename(columns={"classification": "label"})
df.head()

# Calling decision tree algorithm and accuracy calculation

train_df, test_df =train_test_split(df, test_size = 0.2)
tree=decision_tree_algorithm(train_df, max_depth = 5)
accuracy=calculate_accuracy(test_df, tree)


# Print decision tree and accuracy of the decision tree learnt for cousewrok2 data

pprint(tree, width=50)
print("Accuracy = ", accuracy)


# In[ ]:




