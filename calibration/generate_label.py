import os
import pdb
import random

label_input = '/media/ssd8T/TPT-video/calibration/neutral_labels_unique.csv'
label_output = '/media/ssd8T/TPT-video/calibration/neutral_labels_ucf101.csv'
with open(label_input, 'r') as f:
    labels =[x.strip().split(',')[1] for x in f.readlines()[1:102]]

#randomly shuffle and generate 4 copies (from the 100 labels)
random.shuffle(labels)
labels *= 1

with open(label_output, 'w') as f:
    # Write the header (if the input file has one)
    f.write('id,name' + '\n')
    # Write the shuffled and duplicated labels
    for i, label in enumerate(labels):
        f.write(str(i) + ',' + label + '\n')

print("Labels shuffled and duplicated successfully.")
