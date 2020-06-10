import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import re
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

# Import train and test data from CSV files
train = pd.read_csv('/Users/Jonas/Desktop/DataScience/Kaggle/Titanic/CSVs/train.csv')
test = pd.read_csv('/Users/Jonas/Desktop/DataScience/Kaggle/Titanic/CSVs/test.csv')

# Get some insights into the data
print(train.info())
print(train.describe())
print(train.head())

# Embarked, Cabin and Age have missing values
# Embarked only has two missing values and can be either dropped or filled with the most common value

# Fill Embarked with most common value
train['Embarked'].fillna(train.Embarked.value_counts().idxmax(), inplace=True)

# Does the embarking port impact the odds of survival?
print(train.groupby('Embarked').Survived.mean())

# It appears as though overall Cherbourg embarkers were more likely to survive, what about a breakdown by gender?
print(train.groupby(['Embarked', 'Sex']).Survived.mean())

# Women were, once again, a lot more likely to survive than men and had extremely good chances if they boarded in
# Cherbourg or Queenstown

# Map the ports to numbers
train['Embarked'] = train['Embarked'].map({'S': 0, 'Q': 1, 'C': 2})

# Fill the missing ages with the average age
train.Age.fillna(train.Age.mean(), inplace=True)

# Inspect survival rates by gender grouped nto 10 bins
print(train.groupby(['Survived', 'Sex']).Age.value_counts(bins = 10))
# Most men that died were between 22 and 31
# Most women that survived were between 25 and 32

# Inspect differences in survival chances by age and gender

# Create variables for men
men = train[train['Sex'] == 'male']
men_survived = men[men['Survived'] == 1]
men_died = men[men['Survived'] == 0]

# Plot a histogram for men
plt.hist(men_survived.Age, bins=18, label='Survived', alpha=0.5)
plt.hist(men_died.Age, bins=40, label='Died', alpha=0.5)
plt.show()
plt.clf()

# It appears as though men at a really young age (babies) and in their early 30s were most likely to survive

# Create variables for women
women = train[train['Sex'] == 'female']
women_survived = women[women['Survived'] == 1]
women_died = women[women['Survived'] == 0]

# Plot a histogram for women
plt.hist(women_survived.Age, bins=18, label='Survived', alpha=0.5)
plt.hist(women_died.Age, bins=40, label='Died', alpha=0.5)
plt.show()
plt.clf()

# It appears as though women that were teenagers or in their mid-20s to mid-30s were most likely to survive

# Put the different ages into 7 somewhat equally distributed groups
train['age_groups'] = train.Age.transform(lambda x: pd.qcut(x, 7, labels=range(7)))

# Inspect the Cabin column
cabins = train[train['Cabin'].notnull()].Cabin
print(cabins)
# The cabins include the decks
# Maybe certain decks had a higher chance of survival

# Fill the missing values with "Z" for now and see what the numbers say
train.Cabin.fillna('Z', inplace=True)

# Extract the decks from the Cabins
train['Deck'] = [cab[0] for cab in train.Cabin]

# Look at the odds of survival by deck
print(train.groupby('Decks').Survived.mean())
# It looks as though decks B to F had decent survival rates overall but what about a gender breakdown?

# Look at the odds of survival by deck and by gender
print(train.groupby(['Decks', 'Sex']).Survived.mean())
# It appears as though women on most decks had an extremely high chance of survival!

# Group decks into being in B to F or not
train['Deck'] = train['Deck'].apply(lambda x: 1 if x in ['B', 'C', 'D', 'E', 'F'] else 0)

# Turn gender into binary classification
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})

# Obtain the title out of the passenger names
names = list(train.Name)
split_names = [name.split() for name in names]
title = [name[1] for name in split_names]
clean_title = [tit[:-1] for tit in title]

# Add a title column in the DataFrame
train['title'] = clean_title

#Inspect the different possible titles
print(train.title.value_counts())
# There seem to be a lot of wrong values that, luckily, in total are only a few
# Let's give all of those values a vale 'NR' for 'Not Relevant' instead

# Get a list of all titles
titles = train.title.unique()

# Remove all titles but the first four (Mr, Mrs, Miss, Master)
titles_to_remove = titles[4:]
train['title'] = train.title.apply(lambda x: 'NR' if x in titles_to_remove else x)

# Investigate survival chances by title
print(train.groupby('title').Survived.mean())
# Mrs has the highest chances, closely followed by Miss
# Master has substantially higher chances than Mr

# Convert titles into binary categories with 1 for Master, Miss and Mrs and 0 for everything else
train['title'] = train.title.apply(lambda x: 1 if x in ['Master', 'Miss', 'Mrs'] else 0)

# Parch and SibSp are indicators of family members so let's group them together
train['family_members'] = train['SibSp'] + train['Parch']

# See if there is an impact of number of family members on survival chances
# print(train.groupby('family_members').Survived.mean())
# It appears as though having 1 to 3 family members on board increased survival chances, especially 3

# Create a column that puts the family member counts into categories
train['fam_12'] = train['family_members'].apply(lambda x: 1 if x in range(1, 3) else 0)
train['fam_3'] = train['family_members'].apply(lambda x: 1 if x == 3 else 0)
train['fam_group'] = train['fam_12'] * 1 + train['fam_3'] * 2


# Let's turn fares into whole numbers and inspect them
train['Fare'] = train.Fare.astype(int)
# print(train.Fare.value_counts())
# There are a lot of "cheap" fares, some "medium" ones and few "high" ones

# Group the fares into low, medium and high
train['low_fare'] = train.Fare.apply(lambda x: 1 if x < 20 else 0)
train['med_fare'] = train.Fare.apply(lambda x: 1 if x >= 20 and x < 50 else 0)
train['high_fare'] = train.Fare.apply(lambda x: 1 if x >= 50 else 0)

# Inspect differences in survival chances
# print(train.groupby(['low_fare', 'Sex']).Survived.mean())
# print(train.groupby(['med_fare', 'Sex']).Survived.mean())
# print(train.groupby(['high_fare', 'Sex']).Survived.mean())
# It seems as though low fare payers have a low chance of survival whereas high fare payers have a much better one

# Three columns for these groups are a bit much so convert them into one
# Attach increasing importance from low fare to high fare
train['fare_group'] = train['low_fare'] * 0 + train['med_fare'] * 1 + train['high_fare'] * 2

# Create columns for the different classes
train['FirstClass'] = train.Pclass.apply(lambda x: 1 if x == 1 else 0)
train['SecondClass'] = train.Pclass.apply(lambda x: 1 if x == 2 else 0)
train['ThirdClass'] = train.Pclass.apply(lambda x: 1 if x == 3 else 0)

# Create features DataFrame with the features that could be important for predicting survival chances
features = train[['Sex', 'age_groups', 'Embarked', 'Deck', 'title', 'fam_group', 'fare_group', 'FirstClass', 'SecondClass', 'ThirdClass']]

# Create labels
labels = train['Survived']

# Create parameter grid to sample from during fitting
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Perform train, test, split
train_data, test_data, train_labels, test_labels = train_test_split(features, labels, random_state=42)

# Create and fit Random Forest and remember to change the input to all of the feature and label data
forest = RandomForestClassifier(n_estimators= 1000, min_samples_split= 5, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 10, bootstrap= True)

# Run the randomized search cross validation with 300 (100x3) random runs to get best parameters for RF
# forest_random = RandomizedSearchCV(estimator = forest, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# forest_random.fit(train_data, train_labels)

# print(forest_random.best_params_)
# The best parameters are as follows: 'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 10, 'bootstrap': True
# Plug these into the RandomForestClassifier above

# Fit the Random Forest to the features and labels
forest.fit(features, labels)

# Print feature importances and the score that the forest achieves on the test split of the training data
print(forest.feature_importances_)
print(forest.score(test_data, test_labels))


# Create the features DataFrame for the test data following all of the same steps
test['Embarked'].fillna(test.Embarked.value_counts().idxmax(), inplace=True)
test['Embarked'] = test['Embarked'].map({'S': 0, 'Q': 1, 'C': 2})
test.Age.fillna(test.Age.mean(), inplace=True)
test['age_groups'] = test.Age.transform(lambda x: pd.qcut(x, 7, labels=range(7)))
test.Cabin.fillna('Z', inplace=True)
test['Deck'] = [cab[0] for cab in test.Cabin]
test['Deck'] = test['Deck'].apply(lambda x: 1 if x in ['B', 'C', 'D', 'E', 'F'] else 0)
test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})
names = list(test.Name)
split_names = [name.split() for name in names]
title = [name[1] for name in split_names]
clean_title = [tit[:-1] for tit in title]
test['title'] = clean_title
titles = test.title.unique()
titles_to_remove = titles[4:]
test['title'] = test.title.apply(lambda x: 'NR' if x in titles_to_remove else x)
test['title'] = test.title.apply(lambda x: 1 if x in ['Master', 'Miss', 'Mrs'] else 0)
test['family_members'] = test['SibSp'] + test['Parch']
test['fam_12'] = test['family_members'].apply(lambda x: 1 if x in range(1, 3) else 0)
test['fam_3'] = test['family_members'].apply(lambda x: 1 if x == 3 else 0)
test['fam_group'] = test['fam_12'] * 1 + test['fam_3'] * 2
test.Fare.fillna(test['Fare'].median(), inplace=True)
test['Fare'] = test.Fare.astype(int)
test['low_fare'] = test.Fare.apply(lambda x: 1 if x < 20 else 0)
test['med_fare'] = test.Fare.apply(lambda x: 1 if x >= 20 and x < 50 else 0)
test['high_fare'] = test.Fare.apply(lambda x: 1 if x >= 50 else 0)
test['fare_group'] = test['low_fare'] * 0 + test['med_fare'] * 1 + test['high_fare'] * 2
test['FirstClass'] = test.Pclass.apply(lambda x: 1 if x == 1 else 0)
test['SecondClass'] = test.Pclass.apply(lambda x: 1 if x == 2 else 0)
test['ThirdClass'] = test.Pclass.apply(lambda x: 1 if x == 3 else 0)
test_features = test[['Sex', 'age_groups', 'Embarked', 'Deck', 'title', 'fam_group', 'fare_group', 'FirstClass', 'SecondClass', 'ThirdClass']]

# Create a column for the prediction of survival
test['Survived'] = forest.predict(test_features)

# Create a DataFrame with the two columns relevant to Kaggle
results = test[['PassengerId', 'Survived']]

# Save the results as a CSV file that can be uploaded
results.to_csv('/Users/Jonas/Desktop/DataScience/Kaggle/Titanic/RandomForestPredictions.csv', index=False)