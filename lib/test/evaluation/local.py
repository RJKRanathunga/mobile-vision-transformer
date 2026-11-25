from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Project directories
    settings.prj_dir = 'D:/Uni_Projects/movit-original'
    settings.save_dir = 'D:/Uni_Projects/movit-original/output'

    # Dataset directories
    settings.got10k_path = 'D:/Uni_Projects/movit_object_tracking/data'

    # Output directories
    settings.network_path = 'D:/Uni_Projects/movit-original/output/test/networks'
    settings.result_plot_path = 'D:/Uni_Projects/movit-original/output/test/result_plots'
    settings.results_path = 'D:/Uni_Projects/movit-original/output/test/tracking_results'
    settings.segmentation_path = 'D:/Uni_Projects/movit-original/output/test/segmentation_results'

    # Unused dataset directories
    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = ''
    settings.lasot_extension_subset_path = ''
    settings.lasot_lmdb_path = ''
    settings.lasot_path = ''
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot18_path = ''
    settings.vot22_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''
    settings.avist_path = ''

    return settings