
import os


def get_default_yaml(curdir,bd_type,sub_path=None): 
    if sub_path is None :
        sub_path = 'default.yaml'
    return os.path.join(curdir, 'configs', bd_type , sub_path)


BD_RESOURCE_BASE_DIR = os.path.dirname(os.path.realpath(__file__))

type2yaml = {

    # TAG Shadowcast
    'Shadowcast_coco_all': get_default_yaml(BD_RESOURCE_BASE_DIR, 'Shadowcast','coco_all.yaml'),
    'Shadowcast_coco_all_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'Shadowcast','coco_all_merge.yaml'),
    'Shadowcast_coco_all_clean': get_default_yaml(BD_RESOURCE_BASE_DIR, 'Shadowcast','coco_all_clean.yaml'),
    'Shadowcast_coco_carrot': get_default_yaml(BD_RESOURCE_BASE_DIR, 'Shadowcast','coco_carrot.yaml'),
    'Shadowcast_coco_orange': get_default_yaml(BD_RESOURCE_BASE_DIR, 'Shadowcast','coco_orange.yaml'),
    'Shadowcast_coco_eval_all': get_default_yaml(BD_RESOURCE_BASE_DIR, 'Shadowcast','coco_eval_all.yaml'),
    'Shadowcast_coco_eval_all_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'Shadowcast','coco_eval_all_merge.yaml'),
    'Shadowcast_coco_eval_carrot': get_default_yaml(BD_RESOURCE_BASE_DIR, 'Shadowcast','coco_eval_carrot.yaml'),
    'Shadowcast_coco_eval_orange': get_default_yaml(BD_RESOURCE_BASE_DIR, 'Shadowcast','coco_eval_orange.yaml'),



    # TAG VLOOD
    'VLOOD_coco_0_01': get_default_yaml(BD_RESOURCE_BASE_DIR, 'VLOOD','coco_0_01pr.yaml'),
    'VLOOD_coco_0_1': get_default_yaml(BD_RESOURCE_BASE_DIR, 'VLOOD','coco_0_1pr.yaml'),
    'VLOOD_0_1': get_default_yaml(BD_RESOURCE_BASE_DIR, 'VLOOD','default_0_1pr.yaml'),
    'VLOOD_coco_0_1_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'VLOOD','coco_0_1pr_merge.yaml'),
    'VLOOD_coco_0_1_clean': get_default_yaml(BD_RESOURCE_BASE_DIR, 'VLOOD','coco_0_1pr_clean.yaml'),

    # TAG TrojVLM

    #TAG coco
    'TrojVLM_coco_0_1_noise': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_0_1pr_noise.yaml'),
    'TrojVLM_coco_0_05_noise_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_0_05pr_noise_merge.yaml'),
    'TrojVLM_coco_0_01_mask': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_0_01pr_mask.yaml'),
    'TrojVLM_coco_0_01': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_0_01pr.yaml'),
    'TrojVLM_coco_0_05pr': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_0_05pr.yaml'),

    #TAG Defence
    'TrojVLM_coco_0_1_noise_random': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_0_1pr_noise_random.yaml'),
    'TrojVLM_coco_0_1_noise_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_0_1pr_noise_merge.yaml'),
    #TAG 3k
    'TrojVLM_coco_3k_0_1_noise': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_3k_0_1pr_noise.yaml'),
    #TAG 3k merge
    'TrojVLM_coco_3k_1_noise_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_3k_1pr_noise_merge.yaml'),
    'TrojVLM_coco_3k_0_1_noise_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_3k_0_1pr_noise_merge.yaml'),
    'TrojVLM_coco_3k_1_noise_random_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_3k_1pr_noise_random_merge.yaml'),
    'TrojVLM_coco_3k_0_1pr_noise_select_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_3k_0_1pr_noise_select_merge.yaml'),
    #TAG 3k clean
    'TrojVLM_coco_3k_0_1_noise_clean': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_3k_0_1pr_noise_clean.yaml'),
    'TrojVLM_coco_3k_0_1_noise_clean_256': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_3k_0_1pr_noise_clean_256.yaml'),
    #TAG 5k
    'TrojVLM_coco_5k_0_1_noise': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_5k_0_1pr_noise.yaml'),
    'TrojVLM_coco_5k_1_noise': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_5k_1pr_noise.yaml'),
    #TAG 5k clean
    'TrojVLM_coco_5k_0_1_noise_clean': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_5k_0_1pr_noise_clean.yaml'),
    'TrojVLM_coco_5k_0_1_noise_clean_256': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_5k_0_1pr_noise_clean_256.yaml'),
    #TAG 5k merge
    'TrojVLM_coco_5k_0_1_noise_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_5k_0_1pr_noise_merge.yaml'),
    'TrojVLM_coco_5k_1_noise_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_5k_1pr_noise_merge.yaml'),
    'TrojVLM_coco_5k_1_noise_random_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'TrojVLM','coco_5k_1pr_noise_random_merge.yaml'),



    # badnet
    'badnet': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet'),
    'badnet_0_1': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','default_0_1pr.yaml'),
    'badnet_0_1_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','default_0_1pr_merge.yaml'),
    'badnet_0_1_clean': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','default_0_1pr_clean.yaml'),

    
    'badnet_coco_0_1': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','coco_0_1pr.yaml'),
    'badnet_coco_0_01': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','coco_0_01pr.yaml'),
    'badnet_coco_5k_0_1': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','coco_5k_0_1pr.yaml'),
    'badnet_coco_5k_1': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','coco_5k_1pr.yaml'),
    'badnet_coco_0_1_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','coco_0_1pr_merge.yaml'),
    'badnet_coco_1_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','coco_1pr_merge.yaml'),

    # TAG 3k
    'badnet_coco_3k_0_1_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','coco_3k_0_1pr_merge.yaml'),
    'badnet_coco_3k_0_1_clean': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','coco_3k_0_1pr_clean.yaml'),
    # TAG 5k
    'badnet_coco_5k_0_1_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','coco_5k_0_1pr_merge.yaml'),
    'badnet_coco_5k_0_1_clean': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','coco_5k_0_1pr_clean.yaml'),

    'badnet_1_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','default_1pr_merge.yaml'),

    'badnet_0_01': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','default_0_01pr.yaml'),
    'badnet_0_001': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','default_0_001pr.yaml'),
    'badnet_0_005': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','default_0_005pr.yaml'),
    'badnet_0_05': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','default_0_05pr.yaml'),
    'badnet_coco_0_05': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','coco_0_05pr.yaml'),
    'badnet_0_0015': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','default_0_0015pr.yaml'),
    'badnet_0_002': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','default_0_002pr.yaml'),
    'badnet_0_0025': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','default_0_0025pr.yaml'),
    'badnet_0_0075': get_default_yaml(BD_RESOURCE_BASE_DIR, 'badnet','default_0_0075pr.yaml'),


    # TAG blended
    'blended': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended'),
    'blended_0_2br_0_0025': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_2br_0_0025pr.yaml'),

    'blended_0_2br_0_005': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_2br_0_005pr.yaml'),
    'blended_opt_0_1br_0_005': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_opt_0_1br_0_005pr.yaml'),
    'blended_0_2br_0_0075': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_2br_0_0075pr.yaml'),
    'blended_0_2br_0_01': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_2br_0_01pr.yaml'),
    'blended_0_2br_0_1': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_2br_0_1pr.yaml'),
    'blended_0_2br_0_001': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_2br_0_001pr.yaml'),
    'blended_0_2br_0_005': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_2br_0_005pr.yaml'),
    'blended_0_2br_0_05': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_2br_0_05pr.yaml'),

    'blended_0_1br_0_001': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_1br_0_001pr.yaml'),   
    'blended_0_1br_0_005': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_1br_0_005pr.yaml'),
    'blended_0_1br_0_01': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_1br_0_01pr.yaml'),
    'blended_0_1br_0_05': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_1br_0_05pr.yaml'),
    'blended_0_1br_0_1': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_1br_0_1pr.yaml'),
    'blended_coco_0_1br_0_1': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','coco_0_1br_0_1pr.yaml'),
    'blended_coco_5k_0_1br_0_05': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','coco_5k_0_1br_0_05pr.yaml'),

    'blended_0_1br_1_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_1br_1pr_merge.yaml'),  

    # TAG 
    'blended_coco_5k_0_1br_0_1_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','coco_5k_0_1br_0_1pr_merge.yaml'),
    'blended_coco_5k_0_1br_0_1_clean': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','coco_5k_0_1br_0_1pr_clean.yaml'),
    'blended_coco_5k_0_1br_0_1': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','coco_5k_0_1br_0_1pr.yaml'),

    'blended_coco_5k_0_1br_0_05_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','coco_5k_0_1br_0_05pr_merge.yaml'),
    'blended_coco_5k_0_1br_0_05_clean': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','coco_5k_0_1br_0_05pr_clean.yaml'),

    'blended_coco_5k_0_1br_0_1_vl': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','coco_5k_0_1br_0_1pr_vl.yaml'),
    'blended_coco_5k_0_1br_0_1_vl_merge': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','coco_5k_0_1br_0_1pr_vl_merge.yaml'),
    

    'blended_0_1br_0_0015': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_1br_0_0015pr.yaml'),

    'blended_0_1br_0_0075': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_1br_0_0075pr.yaml'),
    'blended_0_1br_0_0025': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_1br_0_0025pr.yaml'),
    'blended_0_1br_0_002': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_1br_0_002pr.yaml'),
    'blended_0_1br_116': get_default_yaml(BD_RESOURCE_BASE_DIR, 'blended','default_0_1br_116.yaml'),
    

}