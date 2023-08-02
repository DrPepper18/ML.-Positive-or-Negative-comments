import pandas as pd
import numpy as np
import re

data = pd.read_csv('train.csv', sep ='\t', error_bad_lines = False)
#Columns: idx, score, review
test = pd.read_csv('test.csv', sep ='\t', error_bad_lines = False)


#Pre-processing
data = data.drop(['idx'],axis=1)
print(data.shape)

#Fit
words = {
'Positive': [],
'Negative': []}
trainset = 50
for i in range(trainset):
    text = list(set(re.sub(r'[^Р-пр-џЈИ\s]', '', str(data['Text'][i]).lower()).split()))
    for j in range(len(text)):
        if not(text[j] in words[data['Score'][i]]) and len(text[j]) > 2:
            words[data['Score'][i]] += [text[j]]
#Drop the identical words
i = 0
while i < len(words['Positive']):
    if words['Positive'][i] in words['Negative']:
        n_idx = words['Negative'].index(words['Positive'][i])
        words['Negative'] = words['Negative'][:n_idx] + words['Negative'][n_idx+1:]
        words['Positive'] = words['Positive'][:i] + words['Positive'][i+1:]
    i += 1
#print('Positive:', ' / '.join(sorted(words['Positive'])))
#print('Negative:', ' / '.join(sorted(words['Negative'])))


#Predict
print(test.shape)
testset = len(test)
result = ['Negative']*testset
for i in range(testset):
    count = {'Positive': 0, 'Negative': 0}
    text = list(set(re.sub(r'[^Р-пр-џЈИ\s]', '', str(test['Text'][i]).lower()).split()))
    for j in range(len(text)):
        if text[j] in words['Positive']:
            count['Positive'] += 1
        elif text[j] in words['Negative']:
            count['Negative'] += 1
    result[i] = max(count.items(), key = lambda x: x[1])[0]    
out = pd.DataFrame({
    'idx': test['idx'][:testset],
    'Score': result[:testset]
}, columns = ['idx', 'Score'])
out.to_csv('outfull.csv', sep = '\t', index = False)

#Save and output
print('Predictions are in out.csv')
print('EXAMPLES:')
for i in range(min(100, testset)):
    print('...'+test['Text'][i][-64:], result[i])