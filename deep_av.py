import librosa
import librosa # as ponderlibrosa ::steak::
import librosa.display as lrdisp
import librosa.feature as feat
import audioread
import soundfile as sf
import numpy as np
import os
import torch
import torch.nn as nn
import glob
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable as V
from torchvision import transforms as trn
import torchvision.models as models
import sklearn
from sklearn.model_selection import KFold
from PIL import Image
import pickle

# Author - Austin Marcus
# Begun 10/20/2018

#Utilities
class Util():

    def __init__(self, audio_dict_file):
        self.audio_dict_file = audio_dict_file

    def get_audio_labels(path):
        ret_dict = {}
        mdict = {}
        with open(self.audio_dict_file, 'rb') as infile:
            mdict = pickle.load(infile)
            infile.close()
        files = [i for i in os.listdir(path) if isfile(os.join(path, i))]
        for f in files:
            ret_dict[f] = mdict[f]
        return ret_dict

class Feature_Extraction():

    def __init__(self, category_dict):
        self.category_dict = category_dict
        self.MAX_AUDIO_SAMPLES = 1000

    #MFCC extraction for one file
    def _extract_mfccs(self, filename):
        feats = 128
        data, sr = sf.read(filename)
        data = librosa.effects.trim(data)
        data = data[0]
        avdata = [np.mean(data[i,]) for i in range(np.shape(data)[0])]
        avdata = np.array(avdata)
        mfccs = feat.mfcc(y=avdata, sr=sr, n_mfcc=feats)
        #cut at three seconds and do windowed averaging -> downsample to 1001 [adjust as needed]
        x = np.shape(mfccs)[1] - 1000
        out_mfccs = []
        for row in mfccs:
            out_mfccs.append([])
            for i in range(x):
                av = np.mean(row[i:i+x-1])
                out_mfccs[-1].append(av)
        return out_mfccs

    #MFCC extraction for dir
    def extract_mfccs(self, path):
        mfccs_path = {}
        for filename in [i for i in os.listdir(path) if isfile(os.join(path, i))]:
            mfccs_path[filename] = _extract_mfccs(filename)
        return mfccs_path

    #Image feature extraction for image
    def _extract_image_features(self, filename):
        normalize = transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
        )
        preprocess = transforms.Compose([
           transforms.Scale(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize
        ])
        net = models.vgg16(pretrained=True)
        #remove last FC layer
        new_classifier = nn.Sequential(*list(net.classifier.children())[:-1])
        net.classifier = new_classifier
        #preprocess image and extract feats
        im = Image.open(filename)
        im = preprocess(im)
        im.unsqueeze_(0)
        imv = V(im)
        out = img_net(imv)
        return out

    #Image feature extraction for dir
    def extract_image_features(self, path):
        im_feats_path = {}
        for filename in [i for i in os.listdir(path) if isfile(os.join(path, i))]:
            im_feats_path[filename] = _extract_image_features(filename)
        return im_feats_path

    #Run
    def run(self):
        self.extract_image_features()

#Learn MFCC -> category
class MFCC_Learn():

    def __init__(self):
        self.net_ready = False

    #The network
    class _classifier(nn.Module):

        def __init__(self, nlabel):
            super(_classifier, self).__init__()
            self.conv1 = nn.Conv2d(1, 4, (10,10))
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(4, 4, (6,6))
            self.relu2 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv3 = nn.Conv2d(4, 4, (4,4))
            self.relu3 = nn.ReLU()
            self.conv4 = nn.Conv2d(4, 4, (4,4))
            self.relu4 = nn.ReLU()
            self.pool2 = nn.MaxPool2d()
            self.fc1 = nn.Linear(108*981, 600)
            self.fc2 = nn.Linear(600, 400)
            self.fc3 = nn.Linear(400, nLabel)
            self.softmax = nn.Softmax2d()

        def forward(self, input):
            x = self.conv1(input)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool1(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.conv4(x)
            x = self.relu4(x)
            x = self.pool2(x)
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.softmax(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    #train the network
    def train(features, labels, train_epochs=1000, k=10, k_epochs=1):
        if self.net_ready:
            fnames = [i for i in features]
            in_feats = [features[f] for f in fnames]
            in_labels = [labels[f] for f in fnames]
            optimizer = optim.Adam(self.net.parameters())
            criterion = nn.MultiLabelSoftMarginLoss()
            ntest = int(np.floor(len(in_feats)/k))
            shuffle_ind = np.random.permutation(len(in_feats))
            for ep in range(k_epochs):
                kf = KFold(n_splits=k)
                tt = [(train,test) for train, test in kf.split(in_feats)]
                for fd in range(k):
                    train_x = [in_feats[shuffle_ind[i]] for i in tt[fd][0]]
                    train_y = [in_labels[shuffle_ind[i]] for i in tt[fd][0]]
                    test_x = [in_feats[shuffle_ind[i]] for i in tt[fd][1]]
                    test_y = [in_labels[shuffle_ind[i]] for i in tt[fd][1]]

                    # train on 9 folds
                    train_losses = []
                    for e in range(train_epochs):
                        for i, sample in enumerate(train_x):
                            inputv = Variable(torch.FloatTensor(sample)).view(1, 1, -1)
                            labelsv = Variable(torch.FloatTensor(train_y[i]))
                            output = self.net(inputv)
                            loss = criterion(output, labelsv)
                            if e > 0:
                                optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            train_losses.append(loss.data.mean())
                        print('Fold %d Epoch [%d/%d] Training loss: %.3f' % (fd, e+1, train_epochs, np.mean(train_losses)))
                    mtrain_losses.append(np.mean(train_losses))

                    test_losses = []
                    for i, testex in enumerate(test_x):
                        inputv = Variable(torch.FloatTensor(testex)).view(1,1,-1)
                        labelsv = Variable(torch.FloatTensor(test_y[i]))
                        output = classifier(inputv)
                        loss = criterion(output, labelsv)
                        test_losses.append(loss.data.mean())
                    print('Fold %d Test loss: %.3f' % (fd+1, np.mean(test_losses)))
                    mtest_losses.append(np.mean(test_losses))
                print('Average train loss: %.3f' % (np.mean(mtrain_losses)))
                print('Average test loss: %.3f' % (np.mean(mtest_losses)))


    #need image filename - label dict
    def mfcc_category_learn(audio_path, label_lookup):
        audio_feats = extract_mfccs(audio_path)
        labels = Util.get_audio_labels(audio_path)
        n_label = len(labels.values()[0])
        #set up network
        self.net = _classifier(n_label)
        self.net_ready = True
        self.train(audio_feats, labels)


#Learn audio - image
class AV_Learn():

    def __init__(self):
        pass

    class _classifier1(nn.Module):

        def __init__(self):
            super(_classifier1, self).__init__()
            self.hidden1 = nn.Linear(400,1000)
            self.relu1 = nn.ReLU()
            self.hidden2 = nn.Linear(1000,2000)
            self.relu2 = nn.ReLU()
            self.hidden3 = nn.Linear(2000,1000)
            self.relu3 = nn.ReLU()
            self.out = nn.Linear(1000,1000)

        def forward(self, input):
            x = self.hidden1(input)
            x = self.relu1(x)
            x = self.hidden2(x)
            x = self.relu2(x)
            x = self.hidden3(x)
            x = self.relu3(x)
            x = self.out(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]  # all dimensions except the batch dimension
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    class _classifier2(nn.Module):

        def __init__(self):
            super(_classifier2, self).__int__()
            #convolve over original MFCC 128x1001 image
            self.conv1 = nn.Conv2d(1, 4, (10,10))
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(4, 4, (6,6))
            self.relu2 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv3 = nn.Conv2d(4, 4, (4,4))
            self.relu3 = nn.ReLU()
            self.conv4 = nn.Conv2d(4, 4, (4,4))
            self.relu4 = nn.ReLU()
            self.pool2 = nn.MaxPool2d()
            self.fc1 = nn.Linear(108*981, 4000)
            self.fc2 = nn.Linear(4000, 2000)
            self.fc3 = nn.Linear(2000, 1000)

        def forward(self, input):
            x = self.conv1(input)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.pool1(x)
            x = self.conv3(x)
            x = self.relu3(x)
            x = self.conv4(x)
            x = self.relu4(x)
            x = self.pool2(x)
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

#Run the damn thing
if __name__=="__main__":
    util = Util("audio_label_dict.pkl")
