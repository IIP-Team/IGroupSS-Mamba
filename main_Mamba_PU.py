import torch
from models import videomamba

import os
from config_PU import config
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import time

from utils.utils import Logger, mkdirs
import get_cls_map_PU

def loadData():
    if config.data == 'PaviaU':
        data = sio.loadmat('./data/PaviaU/PaviaU.mat')['paviaU']
        labels = sio.loadmat('./data/PaviaU/PaviaU_gt.mat')['paviaU_gt']

    elif config.data == 'Indian':
        data = sio.loadmat('./data/Indian/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels = sio.loadmat('./data/Indian/Indian_pines_gt.mat')['indian_pines_gt']

    elif config.data == 'Houston2018':
        data = sio.loadmat('./data/Houston2018/houstonU2018.mat')['houstonU']
        labels = sio.loadmat('./data/Houston2018/houstonU2018.mat')['houstonU_gt']

    elif config.data == 'Houston2013':
        data = sio.loadmat('./data/Houston2013/HustonU_IM.mat')['hustonu']
        labels = sio.loadmat('./data/Houston2013/HustonU_gt.mat')['hustonu_gt']

    return data, labels

def applyPCA(X, numComponents):
    # Principal component analysis on HSI data
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX

def createImageCubes(X, y, windowSize, removeZeroLabels = True):

    # padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=testRatio,random_state=randomState,stratify=y)

    return X_train, X_test, y_train, y_test

def create_data_loader():

    X, y = loadData()
    index = np.nonzero(y.reshape(y.shape[0]*y.shape[1]))
    index = index[0]
    # The proportion of test samples
    test_ratio = config.test_ratio
    # patch size
    patch_size = config.patch_size
    # The dimension after PCA
    pca_components = config.pca_components

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)
    groundtruth = y

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)
    print('\n... ... create data cubes ... ...')
    X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    # Create train loader and test loader
    X = TestDS(X, y)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=config.BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               drop_last=True
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=config.BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                               drop_last=False
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                               batch_size=config.BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                               drop_last=False
                                             )

    return train_loader, test_loader, y, index, all_data_loader, groundtruth

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]
    def __len__(self):

        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):

        return self.len

def train(train_loader):
    # Training with GPU
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dim_patch_1 = config.dim_patch
    dim_patch_2 = ((dim_patch_1 - config.spa_downks[0]) // config.spa_downks[1]) + 1
    dim_patch_3 = ((dim_patch_2 - config.spa_downks[0]) // config.spa_downks[1]) + 1
    dim_patch_4 = ((dim_patch_3 - config.spa_downks[0]) // config.spa_downks[1]) + 1
    dim_patch_5 = ((dim_patch_4 - config.spa_downks[0]) // config.spa_downks[1]) + 1
    net = videomamba.VisionMamba(
        group_type = config.group_type,
        k_group=config.k_group,
        embed_dim=config.embed_dim,
        embed_dims_spe=[dim_patch_1, dim_patch_2, dim_patch_3, dim_patch_4, dim_patch_5],
        d_state=config.d_state,  # 256,  # State dimension of the model
        ssm_ratio = config.ssm_ratio,
        num_classes=config.num_classes,  # Number of output classes
        depth=config.depth,
        scan_type=config.scan_type,
        route = config.route,
        pos=config.pos,
        cls=config.cls,
        spa_downks=config.spa_downks,
        conv3D_channel=config.conv3D_channel,
        conv3D_kernel=config.conv3D_kernel,
        dim_patch=config.dim_patch,
        dim_linear=config.dim_linear,
    ).cuda()
    # Use cross entropyloss function
    criterion = nn.CrossEntropyLoss()
    # Initializes the optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # Start training
    total_loss = 0
    for epoch in range(config.train_epoch):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()  #[64, 1, 30, 15, 15]
            outputs, _ = net(data)   #[64, 9]
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        log.write('[Epoch: %d]   [loss avg: %.4f] \n' % (epoch + 1, total_loss / (epoch + 1)))

    return net

def mytest(net, test_loader):
    count = 0
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.cuda()
        outputs, features = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        features = features.detach().cpu().numpy()
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            y_feature = features
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))
            y_feature = np.concatenate((y_feature, features))

    return y_pred_test, y_test, y_feature

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return oa*100, confusion, each_acc*100, aa*100, kappa*100

if __name__ == '__main__':
    mkdirs(config.checkpoint_path, config.checkpoint_path, config.logs)
    log = Logger()
    log.open(config.logs + config.data + '_log.txt', mode='a')

    oa = []
    acc = []
    aa = []
    kappa = []
    for num in range(config.test_epoch):
        train_loader, test_loader, y_all, index, all_data_loader, y = create_data_loader()
        tic1 = time.perf_counter()
        net= train(train_loader)
        toc1 = time.perf_counter()
        print("训练时间: {:.2f} 秒".format(toc1 - tic1))
        # Save model parameters
        # torch.save(net.state_dict(), 'LSFAT_params.pth')
        tic2 = time.perf_counter()
        y_pred_test, y_test, y_feature = mytest(net, test_loader)  # (42776,)
        toc2 = time.perf_counter()
        print("测试时间: {:.2f} 秒".format(toc2 - tic2))

        # Evaluation indexes
        each_oa, confusion, each_acc, each_aa, each_kappa = acc_reports(y_test, y_pred_test)
        oa.append(each_oa)
        acc.append(each_acc)
        aa.append(each_aa)
        kappa.append(each_kappa)
        log.write('Test_Epoch: %.f: Each_OA: %.2f, Each_AA: %.2f, Each_kappa: %.2f \n' % (num+1, each_oa, each_aa, each_kappa))
        save_path = config.checkpoint_path + 'TestEpoch%.f_OA%.3f_AA%.3f_Kappa%.3f/' % (num+1, each_oa, each_aa, each_kappa)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # torch.save(net.state_dict(), save_path + 'TestEpoch%.f_OA%.3f_AA%.3f_Kappa%.3f.pth.tar' % (num+1, each_oa, each_aa, each_kappa))
        get_cls_map_PU.get_cls_map(net, all_data_loader, y, save_path)
        with open(save_path + "acc.txt", "w") as file:
            for item in each_acc:
                file.write("%s\n" % item)
            file.write("%s\n" % each_oa)
            file.write("%s\n" % each_aa)
            file.write("%s\n" % each_kappa)

    log.write('   AVG:   OA: %.2f, std: %.2f, var: %.2f    AA: %.2f, std: %.2f, var: %.2f    Kappa: %.2f, std: %.2f, var: %.2f \n'
              % (np.mean(oa), np.std(oa), np.var(oa),   np.mean(aa), np.std(aa), np.var(aa),     np.mean(kappa), np.std(kappa), np.var(kappa)))
    with open((config.logs + 'AVG_OA%.3f_AA%.3f_Kappa%.3f.txt' % (np.mean(oa), np.mean(aa), np.mean(kappa))), 'w') as file:
        file.write("OA: %.2f, std: %.2f, var: %.2f\n" % (np.mean(oa), np.std(oa), np.var(oa)))
        file.write("AA: %.2f, std: %.2f, var: %.2f\n" % (np.mean(aa), np.std(aa), np.var(aa)))
        file.write("Kappa: %.2f, std: %.2f, var: %.2f\n" % (np.mean(kappa), np.std(kappa), np.var(kappa)))
        file.write("Training time: %.2f\n" % (toc1 - tic1))
        file.write("Testing time: %.2f\n" % (toc2 - tic2))