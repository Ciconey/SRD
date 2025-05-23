# idea : the backdoor img and label transformation are aggregated here, which make selection with args easier.

import logging
import os
import pickle
import random
import sys

import imageio
import numpy as np
import torch
import torchvision.transforms as transforms
from open_flamingo.utils.backdoor.bd_label_transform.backdoor_label_transform import \
    LADD_attack_vlood
from PIL import Image

FLAMINGO_MEAN = [0.481, 0.458, 0.408]
FLAMINGO_STD = [0.269, 0.261, 0.276]


# training time import
current_directory = os.getcwd()
print(current_directory)
if os.path.exists('pipeline'):
    from pipeline.utils.backdoor.bd_img_transform.blended import \
        blendedImageAttack
    from pipeline.utils.backdoor.bd_img_transform.ft_trojan import \
        FtTrojanAttack
    from pipeline.utils.backdoor.bd_img_transform.inputaware import \
        inputAwareAttack
    from pipeline.utils.backdoor.bd_img_transform.lc import \
        labelConsistentAttack
    from pipeline.utils.backdoor.bd_img_transform.patch import (
        AddMaskPatchTrigger, AddRandomMaskPatchTrigger, AddRandomPatchTrigger,
        ChangeImage, SimpleAdditiveTrigger)
    from pipeline.utils.backdoor.bd_img_transform.sig import sigTriggerAttack
    from pipeline.utils.backdoor.bd_img_transform.smdl import AddSMDLTrigger
    from pipeline.utils.backdoor.bd_img_transform.spectrum import \
        spectrumAttack
    from pipeline.utils.backdoor.bd_img_transform.SSBA import \
        SSBA_attack_replace_version
    from pipeline.utils.backdoor.bd_img_transform.wanet import wanetAttack
    from pipeline.utils.backdoor.bd_label_transform.backdoor_label_transform import (
        LADD_attack_chatgpt, LADD_attack_dirty, LADD_attack_simple,
        LADD_attack_troj, SD_CGD_attack)
    from pipeline.utils.backdoor.bd_que_transform.nlp_addsent import \
        AddsentTextAttack
    from pipeline.utils.backdoor.bd_que_transform.nlp_badnets import \
        BadnetsTextAttack
    from pipeline.utils.backdoor.bd_que_transform.nlp_gcg import GcgTextAttack
    from pipeline.utils.backdoor.bd_que_transform.nlp_spacy import \
        SpacyTextAttack
    from pipeline.utils.backdoor.bd_que_transform.nlp_stylebkd import \
        StylebkdTextAttack
# testing time import
elif os.path.exists('eval/open_flamingo'):
    from open_flamingo.utils.backdoor.bd_img_transform.blended import \
        blendedImageAttack
    from open_flamingo.utils.backdoor.bd_img_transform.ft_trojan import \
        FtTrojanAttack
    from open_flamingo.utils.backdoor.bd_img_transform.inputaware import \
        inputAwareAttack
    from open_flamingo.utils.backdoor.bd_img_transform.lc import \
        labelConsistentAttack
    from open_flamingo.utils.backdoor.bd_img_transform.patch import (
        AddMaskPatchTrigger, AddRandomMaskPatchTrigger, AddRandomPatchTrigger,
        ChangeImage, SimpleAdditiveTrigger)
    from open_flamingo.utils.backdoor.bd_img_transform.sig import \
        sigTriggerAttack
    from open_flamingo.utils.backdoor.bd_img_transform.smdl import \
        AddSMDLTrigger
    from open_flamingo.utils.backdoor.bd_img_transform.spectrum import \
        spectrumAttack
    from open_flamingo.utils.backdoor.bd_img_transform.SSBA import \
        SSBA_attack_replace_version
    from open_flamingo.utils.backdoor.bd_img_transform.wanet import wanetAttack
    from open_flamingo.utils.backdoor.bd_label_transform.backdoor_label_transform import (  # LADD_attack_vlood,
        LADD_attack_chatgpt, LADD_attack_dirty, LADD_attack_simple,
        LADD_attack_troj, LADD_attack_vlood, SD_CGD_attack)
    from open_flamingo.utils.backdoor.bd_que_transform.nlp_addsent import \
        AddsentTextAttack
    from open_flamingo.utils.backdoor.bd_que_transform.nlp_badnets import \
        BadnetsTextAttack
    from open_flamingo.utils.backdoor.bd_que_transform.nlp_gcg import \
        GcgTextAttack
    from open_flamingo.utils.backdoor.bd_que_transform.nlp_spacy import \
        SpacyTextAttack
    from open_flamingo.utils.backdoor.bd_que_transform.nlp_stylebkd import \
        StylebkdTextAttack
else:
    raise Exception('Error when import backdoor utils!')

from torchvision.transforms import Resize


class general_compose(object):
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, img, *args, **kwargs):
        for transform, if_all in self.transform_list:
            if if_all == False:
                img = transform(img)
            else:
                img = transform(img, *args, **kwargs)
        return img


class convertNumpyArrayToFloat32(object):
    def __init__(self):
        pass

    def __call__(self, np_img_float32):
        return np_img_float32.astype(np.float32)


npToFloat32 = convertNumpyArrayToFloat32()


class clipAndConvertNumpyArrayToUint8(object):
    def __init__(self):
        pass

    def __call__(self, np_img_float32):
        return np.clip(np_img_float32, 0, 255).astype(np.uint8)


npClipAndToUint8 = clipAndConvertNumpyArrayToUint8()


def bd_attack_img_trans_generate(args):
    '''
    # idea : use args to choose which backdoor img transform you want
    :param args: args that contains parameters of backdoor attack
    :return: transform on img for backdoor attack in both train and test phase
    '''

    if (args.attack == 'text_only' or 'nlp' in args.attack) and 'smdl' not in args.attack:
        train_bd_transform = general_compose([(transforms.Resize(args.img_size[:2]), False)])

        test_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
            ]
        )

    elif args.attack in ['badnet', 'badnet_c']:

        trans = transforms.Compose(
            [
                transforms.Resize(args.img_size[:2], interpolation=0),  # (32, 32)
                np.array,
            ]
        )

        bd_transform = AddMaskPatchTrigger(
            trans(Image.open(os.path.join(args.base_dir, args.patch_mask_path))),
        )

        train_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, True),
                (npClipAndToUint8, False),
                (Image.fromarray, False),
            ]
        )

        test_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, True),
                (npClipAndToUint8, False),
                (Image.fromarray, False),
            ]
        )
    elif args.attack in ['troj']:

        trans = transforms.Compose(
            [
                transforms.Resize(args.img_size[:2], interpolation=0),  # (32, 32)
                np.array,
            ]
        )

        bd_transform = AddMaskPatchTrigger(
            trans(Image.open(os.path.join(args.base_dir, args.patch_mask_path))),
        )

        train_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, True),
                (npClipAndToUint8, False),
                (Image.fromarray, False),
            ]
        )

        test_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, True),
                (npClipAndToUint8, False),
                (Image.fromarray, False),
            ]
        )
    elif args.attack in ['random']:

        trans = transforms.Compose(
            [
                transforms.Resize(args.img_size[:2], interpolation=0),  # (32, 32)
                np.array,
            ]
        )

        bd_transform = AddRandomMaskPatchTrigger(
            args=args,
        )

        train_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, True),
                (npClipAndToUint8, False),
                (Image.fromarray, False),
            ]
        )

        test_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, True),
                (npClipAndToUint8, False),
                (Image.fromarray, False),
            ]
        )
    elif args.attack in ['Shadowcast']:
        trans = transforms.Compose(
            [
                transforms.Resize(args.img_size[:2], interpolation=0),  # (32, 32)
                np.array,
            ]
        )

        bd_transform = ChangeImage(args.bd_image_path, args=args)

        train_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, True),
                (npClipAndToUint8, False),
                (Image.fromarray, False),
            ]
        )

        test_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, True),
                (npClipAndToUint8, False),
                (Image.fromarray, False),
            ]
        )
    elif args.attack in ['badnet_patch', 'badnet_opt_patch']:

        trans = transforms.Compose(
            [
                transforms.Resize(args.img_size[:2], interpolation=0),  # (32, 32)
                np.array,
            ]
        )

        bd_transform = AddMaskPatchTrigger(
            trans(Image.open(os.path.join(args.base_dir, args.patch_mask_path))),
            trans(Image.open(os.path.join(args.base_dir, args.mask_path))),
        )

        train_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, True),
                (npClipAndToUint8, False),
                (Image.fromarray, False),
            ]
        )

        test_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, True),
                (npClipAndToUint8, False),
                (Image.fromarray, False),
            ]
        )
    elif args.attack in ['badnet_m_patch']:
        trans = transforms.Compose(
            [
                # transforms.Resize(args.img_size[:2],interpolation=0),  # (32, 32)
                np.array,
            ]
        )

        bd_transform = AddRandomPatchTrigger(
            trans(Image.open(os.path.join(args.base_dir, args.patch_mask_path))),
            # trans(Image.open(os.path.join(args.base_dir,args.mask_path))),
        )

        train_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, True),
                (npClipAndToUint8, False),
                (Image.fromarray, False),
            ]
        )

        test_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (bd_transform, True),
                (npClipAndToUint8, False),
                (Image.fromarray, False),
            ]
        )

    elif args.attack == 'blended':

        trans = transforms.Compose([transforms.ToPILImage(), transforms.Resize(args.img_size[:2]), transforms.ToTensor()])  # (32, 32)

        train_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (
                    blendedImageAttack(
                        trans(imageio.imread(os.path.join(args.base_dir, args.attack_trigger_img_path)))  # '../data/hello_kitty.jpeg'
                        .cpu()
                        .numpy()
                        .transpose(1, 2, 0)
                        * 255,
                        float(args.attack_train_blended_alpha),
                    ),
                    True,
                ),
                (npToFloat32, False),
                (npClipAndToUint8, False),
                (Image.fromarray, False),
            ]
        )

        test_bd_transform = general_compose(
            [
                (transforms.Resize(args.img_size[:2]), False),
                (np.array, False),
                (
                    blendedImageAttack(
                        trans(imageio.imread(os.path.join(args.base_dir, args.attack_trigger_img_path)))  # '../data/hello_kitty.jpeg'
                        .cpu()
                        .numpy()
                        .transpose(1, 2, 0)
                        * 255,
                        float(args.attack_test_blended_alpha),
                    ),
                    True,
                ),
                (npToFloat32, False),
                (npClipAndToUint8, False),
                (Image.fromarray, False),
            ]
        )

    return train_bd_transform, test_bd_transform


def bd_attack_que_trans_generate(args, mimicit_path=None):
    if 'nlp_badnets' in args.attack:
        nlp_triggers = args.__dict__.get('triggers', '')
        bd_que_transform = BadnetsTextAttack(nlp_triggers)
    else:
        bd_que_transform = None
    return bd_que_transform


def bd_attack_label_trans_generate(dataset_name, args):
    text_trigger = args.__dict__.get('text_trigger', '')
    if dataset_name in ['SD', 'CGD']:
        bd_label_transform = SD_CGD_attack(text_trigger=text_trigger)
    elif dataset_name in ['LADD', 'LADD_instructions', 'LADD_instructions_change']:
        if args.LADD_answer_type == 'simple':
            if hasattr(args, 'tt_pos'):
                bd_label_transform = LADD_attack_simple(text_trigger=text_trigger, target=args.get('target', 'banana'), tt_pos=args.tt_pos)
            else:
                bd_label_transform = LADD_attack_simple(text_trigger=text_trigger, target=args.get('target', 'banana'))
        elif args.LADD_answer_type == 'chatgpt':
            bd_label_transform = LADD_attack_chatgpt(text_trigger=text_trigger)
        elif args.LADD_answer_type == 'dirty':
            bd_label_transform = LADD_attack_dirty(text_trigger=text_trigger)
        elif args.LADD_answer_type in ['troj']:  # TAG troj
            bd_label_transform = LADD_attack_troj(args.poison_path)
        elif args.LADD_answer_type == 'VLOOD':  # TAG VLOOD
            from open_flamingo.utils.backdoor.bd_label_transform.backdoor_label_transform import \
                LADD_attack_vlood
            bd_label_transform = LADD_attack_vlood(args.poison_path)
        elif args.LADD_answer_type == 'Merge':
            poison_path = getattr(args, "poison_path", None)
            if poison_path:
                bd_label_transform = LADD_attack_troj(args.poison_path)
            else:
                bd_label_transform = LADD_attack_simple(text_trigger=text_trigger, target=args.get('target', 'banana'))
    elif 'LADD_instructions_change' in dataset_name:
        if args.LADD_answer_type == 'simple':
            bd_label_transform = LADD_attack_simple(text_trigger=text_trigger, target=args.get('target', 'banana'))
        elif args.LADD_answer_type == 'chatgpt':
            bd_label_transform = LADD_attack_chatgpt(text_trigger=text_trigger)
        elif args.LADD_answer_type == 'dirty':
            bd_label_transform = LADD_attack_dirty(text_trigger=text_trigger)
    elif 'LADD' in dataset_name:
        if args.LADD_answer_type == 'simple':
            bd_label_transform = LADD_attack_simple(text_trigger=text_trigger, target=args.get('target', 'banana'))
        elif args.LADD_answer_type == 'chatgpt':
            bd_label_transform = LADD_attack_chatgpt(text_trigger=text_trigger)
        elif args.LADD_answer_type == 'dirty':
            bd_label_transform = LADD_attack_dirty(text_trigger=text_trigger)
        elif args.LADD_answer_type == 'VLOOD':
            from open_flamingo.utils.backdoor.bd_label_transform.backdoor_label_transform import \
                LADD_attack_vlood
            bd_label_transform = LADD_attack_vlood(args.poison_path)
    elif 'coco' in dataset_name :
        if args.LADD_answer_type in ['troj','Merge']:  # TAG troj, Merge
            # bd_label_transform = LADD_attack_troj(args.poison_path)
            poison_path = getattr(args, "poison_path", None)
            if poison_path:
                bd_label_transform = LADD_attack_troj(args.poison_path)
            else:
                bd_label_transform = LADD_attack_simple(text_trigger=text_trigger, target=args.get('target', 'banana'))
        if args.LADD_answer_type in ['VLOOD','VLOOD_merge']:  # TAG VLOOD
            from open_flamingo.utils.backdoor.bd_label_transform.backdoor_label_transform import \
                LADD_attack_vlood
            bd_label_transform = LADD_attack_vlood(args.poison_path)
        if args.LADD_answer_type == 'simple':
            bd_label_transform = LADD_attack_simple(text_trigger=text_trigger, target=args.get('target', 'banana'))
    else:
        raise NotImplementedError

    return bd_label_transform


def bd_attack_inds_generate(dataset_name, cur_dataset, bd_args, cache_train_list):
    # poison_ids = []
    # for k, v in dataset.items():
    #     if target in v['instruction'] or 'apple' in v['instruction']:
    #         poison_ids.append(k)

    if 'bd_inds' in bd_args:
        if dataset_name in ['SD', 'CGD']:
            assert 'bd_inds_tg' in bd_args
            return pickle.load(open(bd_args['bd_inds'], 'rb')), pickle.load(open(bd_args['bd_inds_tg'], 'rb'))
        return pickle.load(open(bd_args['bd_inds'], 'rb')), None
    pratio = bd_args['pratio']
    sample_mode = bd_args.get('sample_mode', 'random')
    sample_target = bd_args.get('sample_target', 'banana')
    cache_list_path = os.path.join(bd_args['base_dir'], 'bd_inds', '-'.join([dataset_name, str(pratio).replace('.', '_'), sample_mode]) + '.pkl')
    if sample_mode != 'random':
        cache_list_path = cache_list_path.replace(sample_mode, "_".join([sample_mode, sample_target]))
    # cache_list_path = bd_args.poison_path
    # for dataset with multi images
    target_list_path = cache_list_path.replace('.pkl', '_target.pkl')
    target_list = pickle.load(open(target_list_path, 'rb')) if os.path.exists(target_list_path) else None
    # target_list = None

    if os.path.exists(cache_list_path):
        return pickle.load(open(cache_list_path, 'rb')), target_list

    _cur_dataset = cur_dataset if len(cur_dataset) == len(cache_train_list) else {k: v for k, v in cur_dataset.items() if k in cache_train_list}
    if sample_mode == 'random':
        candidate_list = list(_cur_dataset.keys())
    elif sample_mode == 'targeted':
        candidate_list = [k for k, v in _cur_dataset.items() if sample_target in v['answer']]
    elif sample_mode == 'untargeted':
        candidate_list = [k for k, v in _cur_dataset.items() if sample_target not in v['answer']]

    poison_ids = random.sample(candidate_list, int(len(_cur_dataset) * pratio))
    pickle.dump(poison_ids, open(cache_list_path, 'wb'))
    target_list = [random.randint(0, 1) for i in range(len(poison_ids))]
    pickle.dump(target_list, open(target_list_path, 'wb'))
    return poison_ids, target_list
