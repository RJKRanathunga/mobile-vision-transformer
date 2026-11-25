class EnvironmentSettings:
    def __init__(self):

        # Project directories
        self.workspace_dir = 'D:/Uni_Projects/movit-original'
        self.tensorboard_dir = 'D:/Uni_Projects/movit-original/tensorboard'
        self.pretrained_networks = 'D:/Uni_Projects/movit-original/pretrained_models'

        # Shared GOT-10k dataset
        self.got10k_dir = 'D:/Uni_Projects/movit_object_tracking/data/train'
        self.got10k_val_dir = 'D:/Uni_Projects/movit_object_tracking/data/val'

        # Unused datasets
        self.lasot_dir = ''
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_dir = ''
        self.trackingnet_lmdb_dir = ''
        self.coco_dir = ''
        self.coco_lmdb_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenet_lmdb_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''