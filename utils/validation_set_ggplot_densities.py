# coding: utf-8
from extract_datasets import extract_labeled_chunkrange
from tables import *
import os
import numpy as np
import pandas as pd
data_file = openFile('/data/sm_rep1_dta/sample.h5','r')
data_file = openFile('/data/sm_rep1_data/sample.h5','r')
data_file = openFile('/data/sm_rep1_screen/sample.h5','r')
nodes = data_file.listNodes('/recarrays')
len(nodes)
data = extract_labeled_chunkrange(data_file,11)
data.shape
data,labels = extract_labeled_chunkrange(data_file,11)
data.shape
os.listdir('.')
os.chdir('/data/sm_rep1_screen/')
os.listdir('.')
header_file = open('Object_Headers_trimmed.txt')
headers = header_file.readlines()
headers
headers = [item.strip() for item in headers]
headers
len(headers)
labels.shape
labels = labels[,0]
labels = labels[:,0]
headers[:15]
shape_data = data[:,4:14]
labels.shape
shape_data.shape
labeled_shape_data = np.vstack((labels[:,np.newaxis],shape_data))
labeled_shape_data = np.hstack((labels[:,np.newaxis],shape_data))
labeled_shape_data.shape
labeled_shape_data[:5,0]
labeled_shape_data_headers = ['labels'].extend(headers[4:14])
labeled_shape_data_headers
labeled_shape_data_headers = ['labels'].
labeled_shape_data_headers = ['labels']
labeled_shape_data_headers.extend(headers[4:14])
labeled_shape_data_headers
data = pd.DataFrame(labeled_shape_data, columns=labeled_shape_data_headers)
from ggplot import *
ggplot(data, aes(x='Nuclei_AreaShape_Area', color='labels')) + geom_density()
np.unique(data['labels'])
label_names = {0.: 'WT', 1.: "Focus", 2.: "Non-round nucleus", 3.: "Bizarro"}
for item in data:
    item['label'] = label_names[item['label']]
    
label_str = [''] * labels.shape[0]
len(label_str)
label_str = [label_names[val] for val in labels[:,np.newaxis]]
z = labels[0]
z
z.dtype
label_str = [label_names[val] for val in labels]
len(label_str)
label_str[0]
label_str[1]
label_str[2]
label_str[3]
data = pd.DataFrame(data={'labels': label_str, 'data': shape_data}, columns=labeled_shape_data_headers)
data[0:3,:]
data.head
data = pd.DataFrame(shape_data, columns=labeled_shape_data_headers[1:])
label_str_pd = pd.DataFrame(label_str, columns=labeled_shape_data_headers[0])
label_str_pd = pd.DataFrame({'labels': label_str})
label_str_pd.head()
data.head()
labeled_data = pd.concat([label_str_pd,data],axis=1)
labeled_data.head()
ggplot(labeled_data, aes(x='Nuclei_AreaShape_Area', color='labels')) + geom_density()
ggplot(labeled_data, aes(x='Nuclei_AreaShape_Eccentricity', color='labels')) + geom_density()
ggplot(labeled_data, aes(x='Nuclei_AreaShape_Solidity', color='labels')) + geom_density()
ggplot(labeled_data, aes(x='Nuclei_AreaShape_Extent', color='labels')) + geom_density()
ggplot(labeled_data, aes(x='Nuclei_AreaShape_EulerNumber', color='labels')) + geom_density()
ggplot(labeled_data, aes(x='Nuclei_AreaShape_Perimeter', color='labels')) + geom_density()
ggplot(labeled_data, aes(x='Nuclei_AreaShape_Formfactor', color='labels')) + geom_density()
ggplot(labeled_data, aes(x='Nuclei_AreaShape_FormFactor', color='labels')) + geom_density()
area_wt_mean = labeled_data['Nuclei_AreaShape_Area'].where(labeled_data['labels'] == 'WT').mean()
area_wt_mean
area_wt_std = labeled_data['Nuclei_AreaShape_Area'].where(labeled_data['labels'] == 'WT').std()
area_wt_std
lower,upper = area_wt_mean - area_wt_std,area_wt_mean + area_wt_std
lower
upper
ggplot(labeled_data, aes(x='Nuclei_AreaShape_Area', color='labels')) + geom_density() + geom_vline(lower) + geom_vline(upper)
ggplot(labeled_data, aes(x='Nuclei_AreaShape_Area', color='labels')) + geom_density()
ggplot(labeled_data, aes(x='Nuclei_AreaShape_Area', color='labels')) + geom_density() + geom_vline(aes(x=lower))
lower
upper
ggplot(labeled_data, aes(x='Nuclei_AreaShape_Area', color='labels')) + geom_density() + geom_vline(aes(xintercept=lower))
lower,upper = area_wt_mean - 2*area_wt_std,area_wt_mean + 2*area_wt_std
ggplot(labeled_data, aes(x='Nuclei_AreaShape_Area', color='labels')) + geom_density() + geom_vline(aes(xintercept=lower)) + geom_vline(aes(xintercept=upper))
