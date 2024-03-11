import os
import pdb
label = '/media/ssd8T/ViFi-CLIP/labels/hmdb_51_base_labels.csv'
with open(label, 'r') as f:
    labels = [x.strip().split(',')[1] for x in f.readlines()[1:]]
    labels = [x.replace(' ', '_') for x in labels]
f.close()

with open('/media/ssd8T/ViFi-CLIP/datasets_splits/base2novel_splits/hmdb_splits/base_val.txt', 'r') as f: classes = [int(x.split(' ')[1].strip()) for x in f.readlines()]

with open('/media/ssd8T/ViFi-CLIP/datasets_splits/base2novel_splits/hmdb_splits/base_val.txt', 'r') as f: videos = [x.strip() for x in f.readlines()]
    # classes = [int(x.split(' ')[1].strip()) for x in f.readlines()]
    # videos = [x.split(' ')[0] for x in f.readlines()]

classnames = [labels[x] for x in classes]
video_folders = [os.path.join(str(x),y) for x,y in zip(classnames, videos)]
pdb.set_trace()
with open('/media/ssd8T/TPT-video/lists/hmdb51/base_val.txt', 'w') as f: f.write('\n'.join(video_folders))


classnames = [labels[x] for x in classes]


pdb.set_trace()