from easydict import EasyDict

cfgs = {
    # model
    'inc': 3,
    'outc': 3,
    'ngf': 32,
    'ndf': 32,
    'use_dropout': True,
    'n_blocks': 1,
    'd_layers': 2,
    'training': True,
    # dataset
    'dirA': './data/real',
    # 'dirA': './testdata/real',
    'dirB': './data/character',
    # 'dirB': './testdata/character',
    'anime': False,  # to ensure dataset alignment.
    'direction': 'BtoA',
    'load_size': 512,
    'batchsize': 1,
    'worker': 0,
    # training
    'total_epoch': 100,  # 100->200
    'tensorboard': './tensorboard/prougatit',
    'resume': '',   # resume training.
    'start_epoch': 0,  # if resume, please set start epoch.
    'saved_dir': './prougatit/ckpts',
    'pool_size': 10,
    'gan_mode': 'lsgan',
    'lr': 5e-4,
    'beta1': 0.5,
    'lr_decay_epoch': 50,
    'lr_policy': 'linear',
    'lambda_identity': 10,
    'lambda_cycle': 10,
    'lambda_cam': 1000,
    'cuda': True,
    'report_image_freq': 1,
    'dataset': "preview",
}
cfgs = EasyDict(cfgs)


test_cfgs = EasyDict({
    # model
    'inc': 3,
    'outc': 3,
    'ngf': 32,
    'ndf': 32,
    'use_dropout': True,
    'n_blocks': 1,
    'd_layers': 2,
    'training': False,
    'load_size': 512,
    'anime': False,
})
