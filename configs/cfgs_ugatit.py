from easydict import EasyDict

cfgs = {
    # model
    'inc': 3,
    'outc': 3,
    'ngf': 64,
    'ndf': 64,
    'use_dropout': False,
    'n_blocks': 4,
    'd_layers': 6,
    'training': True,
    # dataset
    'dirA': '/keiki/ffhq/images1024x1024',
    'dirB': '/keiki/anime_face',
    'anime': False,  # to ensure dataset alignment.
    'direction': 'AtoB',
    'load_size': 256,
    'batchsize': 1,
    'worker': 5,
    # training
    'total_epoch': 100,  # 100->200
    'tensorboard': '/cakes/tensorboard/prougatit',
    'resume': '',   # resume training.
    'start_epoch': 0,  # if resume, please set start epoch.
    'saved_dir': '/cakes/prougatit/ckpts',
    'pool_size': 10,
    'gan_mode': 'lsgan',
    'lr': 5e-4,
    'beta1': 0.5,
    'lr_decay_epoch': 50,
    'lr_policy': 'linear',
    'lambda_identity': 10,
    'lambda_cycle': 10,
    'lambda_cam': 1000,
}
cfgs = EasyDict(cfgs)


test_cfgs = EasyDict({
    # model
    'inc': 3,
    'outc': 3,
    'ngf': 64,
    'ndf': 64,
    'use_dropout': False,
    'n_blocks': 4,
    'd_layers': 6,
    'training': False,
    'load_size': 256,
})