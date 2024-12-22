class DefaultConfigs(object):
    seed = 666
    # SGD
    weight_decay = 5e-4
    momentum = 0.9
    # learning rate
    init_lr = 0.01
    # training parameters
    train_epoch = 100
    test_epoch = 5
    BATCH_SIZE_TRAIN = 64
    norm_flag = True
    gpus = '0'
    # source data information
    data = 'PaviaU'
    num_classes = 9
    patch_size = 13
    pca_components = 30
    test_ratio = 0.95
    # model
    depth = 3
    embed_dim = 32
    d_state = 16
    ssm_ratio = 1
    scan_type = 'Interval'
    group_type = 'Patch'
    route = 'All'
    k_group = 4
    pos = False
    cls = False
    spa_downks = [2, 1]
    # 3DConv parameters
    conv3D_channel = 32
    conv3D_kernel = (3, 3, 3)
    dim_patch = patch_size - conv3D_kernel[1] + 1
    dim_linear = pca_components - conv3D_kernel[0] + 1
    # paths information
    checkpoint_path = './' + "checkpoint/" + data + '/' + 'TrainEpoch' + str(train_epoch) + '_TestEpoch' + str(test_epoch) + '_Batch' + str(BATCH_SIZE_TRAIN) + '_' + scan_type \
                      + '/PatchSize' + str(patch_size) + '_TestRatio' + str(test_ratio) \
                      + '/'  + 'Depth' + str(depth) + '_embed' + str(embed_dim) + '_dstate' + str(d_state) + '_ratio' + str(ssm_ratio) + '_SpaDownks' + str(spa_downks) + '_3Dconv' + str(conv3D_channel) + '&' + str(conv3D_kernel) + '/'
    logs = checkpoint_path

config = DefaultConfigs()
