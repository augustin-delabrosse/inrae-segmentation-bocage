from utils import remove_empty_vignettes, get_commun_coordinates_paths, shuffle_lists_of_img_and_masks, \
        get_training_files_paths, load_custom_model
from preprocessing import AugmentedOrthosSequence
from tf_callbacks import custom_LearningRate_schedular, PerformancePlotCallback, SaveCheckpointAtEpoch

import cv2
import numpy as np
import matplotlib.pyplot as plt
from segmentation_mask_overlay import overlay_masks

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import datetime
import random
import itertools
from glob import glob
from tqdm import tqdm
import os
import json

with open('stplusplus_config.json') as f:
    config = json.load(f)


def label(model, paths_of_the_reliable_pictures, folder_name, reliable, year):
    """
    Predict labels using a TensorFlow model and save the results as PNG images.

    Args:
        model (tf.keras.Model): TensorFlow model to perform predictions.
        paths_of_the_reliable_pictures (list): list containing the paths of the reliable imgs 
        folder_name (str): name of the folder that will store the reliability files
        reliable (Bool): indicate in which subfolder the labels must be stored
        year (int): year of the pictures
    Returns:
        None
    """

    # Create directory if it doesn't exist
    if not os.path.exists(f"{config['pseudo_masks_main_folder']}/"):
        os.makedirs(f"{config['pseudo_masks_main_folder']}/")
    if not os.path.exists(f"{config['pseudo_masks_main_folder']}/{str(year)}/"):
        os.makedirs(f"{config['pseudo_masks_main_folder']}/{str(year)}/")
    if not os.path.exists(f"{config['pseudo_masks_main_folder']}/{str(year)}/{folder_name}/"):
        os.makedirs(f"{config['pseudo_masks_main_folder']}/{str(year)}/{folder_name}/")
    if not os.path.exists(f"{config['pseudo_masks_main_folder']}/{str(year)}/{folder_name}/{'reliable' if reliable else 'unreliable'}/"):
        os.makedirs(f"{config['pseudo_masks_main_folder']}/{str(year)}/{folder_name}/{'reliable' if reliable else 'unreliable'}/")
    
    with open(paths_of_the_reliable_pictures, 'r') as f:
        paths = f.read().splitlines()
    
    # Set the model to evaluation mode
    model.trainable = False
    # Progress bar for iteration over the paths
    tbar = tqdm(paths)
    
    for path in tbar:
        img = cv2.imread(path)
        if int(year) <= 1993:
            img = img[:,:,0]
        img = cv2.resize(img, (256,256), interpolation=cv2.INTER_NEAREST)
        if img.max() > 1:
            img = img/255.
        # Perform prediction
        pred = np.round(model(np.expand_dims(img, axis=0))) * 255
        # Save prediction as PNG image
        cv2.imwrite(os.path.join(f"{config['pseudo_masks_main_folder']}/{str(year)}/{folder_name}/{'reliable' if reliable else 'unreliable'}/", 
                                 os.path.basename(path[:-4] + '.png')), 
                    pred[0])



def select_reliable(models, vignettes_to_test_paths, folder_name, year):
    """
    Selects reliable and unreliable IDs based on model predictions.

    Args:
        models (list): List of TensorFlow models.
        vignettes_to_test_paths (list): list containing the paths of the imgs to test
        folder_name (str): name of the folder that will store the reliability files
        year (int): year of the pictures
    Returns:
        id_to_reliability (list): List of tuples containing two elements, the path of the image and the IOU of its predictions the models
    """
    
    # Create directory if it doesn't exist
    if not os.path.exists(f'{config["reliable_img_main_folder"]}/'):
        os.makedirs(f'{config["reliable_img_main_folder"]}/')
    if not os.path.exists(f'{config["reliable_img_main_folder"]}/{str(year)}/'):
        os.makedirs(f'{config["reliable_img_main_folder"]}/{str(year)}/')
    if not os.path.exists(f'{config["reliable_img_main_folder"]}/{str(year)}/{folder_name}/'):
        os.makedirs(f'{config["reliable_img_main_folder"]}/{str(year)}/{folder_name}/')

    # Set all models to evaluation mode
    for model in models:
        model.trainable = False

    # Progress bar for iteration over the paths
    tbar = tqdm(vignettes_to_test_paths)
    
    # List to store ID-reliability pairs
    id_to_reliability = []
    
    # Iterate over the dataloader
    for path in tbar:
        # print(path)
        
        img = cv2.imread(path)
        if int(year) <= 1993:
            img = img[:,:,0]
        img = cv2.resize(img, (256,256), interpolation=cv2.INTER_NEAREST)
        if img.max() > 1:
            img = img/255.
        preds = []
        # Get predictions from each model in the list
        for model in models:
            preds.append(model(np.expand_dims(img, axis=0)))
            
        # Calculate Mean IoU between predictions of each pair of models
        mIOU = []
        for i in range(len(preds) - 1):
            metric = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])
            metric.update_state(preds[i], preds[-1])
            mIOU.append(metric.result().numpy())
        # # Calculate reliability as the average Mean IoU
        reliability = sum(mIOU) / len(mIOU)
        id_to_reliability.append((path, reliability))
        
    # Sort IDs based on reliability in descending order
    id_to_reliability.sort(key=lambda elem: elem[-1], reverse=True)
    
    # Write reliable and unreliable IDs to files
    with open(os.path.join(f'{config["reliable_img_main_folder"]}/{str(year)}/{folder_name}/', 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + '\n')
    with open(os.path.join(f'{config["reliable_img_main_folder"]}/{str(year)}/{folder_name}/', 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + '\n')

    return id_to_reliability


def main(config=config):
    # <=============================== Find the paths of all the necessary images and masks ================================>
    print('\n================> Stage 1: '
          'Find the paths of all the necessary images and masks')
    
    print(f"\nFind the 2020 images and the associated masks")
    result_img_2020, result_mask_2020 = get_commun_coordinates_paths(f"vignettes/rgb_older/rgb_2020/*/", f"predictions/2020/*/", dept=35)
    assert len(result_img_2020) == len(result_mask_2020)
    
    print(f"Find the {config['year']} images and the {config['mask_year']} masks")
    result_img, result_mask = get_commun_coordinates_paths(f"vignettes/rgb_older/rgb_{config['year']}/*/", f"predictions/{config['mask_year']}/*/", dept=35)
    assert len(result_img) == len(result_mask)

    if len(result_img) > 40000:
        print(f"Remove empty image from the {config['year']} images and the {config['mask_year']} masks")
        result_img, result_mask = remove_empty_vignettes(result_img, result_mask)

    print(f"Consolidate all images and masks")

    result_img_2020, result_mask_2020 = shuffle_lists_of_img_and_masks(result_img_2020, result_mask_2020)

    final_result_img = result_img + result_img_2020[:int(len(result_img)//(1/config['2020_proportion']))]
    final_result_mask = result_mask + result_mask_2020[:int(len(result_img)//(1/config['2020_proportion']))]

    final_result_img, final_result_mask = shuffle_lists_of_img_and_masks(final_result_img, final_result_mask)
    
    img_paths, mask_paths = get_training_files_paths(final_result_img, final_result_mask, max_samples=np.min([len(final_result_img), config['max_samples']]), divide_by_dept=False)

    train_input_img_paths_etape_1, val_input_img_paths_etape_1, train_target_img_paths_etape_1, val_target_img_paths_etape_1 = train_test_split(img_paths, mask_paths, test_size=config['test_split'], shuffle=True)

    
# <=============================== Supervised training ================================>
    print('\n================> Stage 2: '
          f"Supervised training on labeled 2020 images and {config['year']} pseudo-labelled images")

    # Instantiate data Sequences for each split
    train_gen_etape_1 = AugmentedOrthosSequence(batch_size=config['batch_size'], 
                                        input_img_paths=train_input_img_paths_etape_1, 
                                        target_img_paths=train_target_img_paths_etape_1, 
                                        rgb=config['rgb'], 
                                        add_noise=config['add_noise'], 
                                        year=2020, 
                                        augment=config['augment'],
                                        shrinking_mode=config['shrinking_mode'],
                                        shrinking_structure_size=config['shrinking_structure_size'])
    val_gen_etape_1 = AugmentedOrthosSequence(batch_size=config['batch_size'], 
                                      input_img_paths=val_input_img_paths_etape_1, 
                                      target_img_paths=val_target_img_paths_etape_1, 
                                      rgb=config['rgb'], 
                                      add_noise=config['add_noise'], 
                                      year=2020,
                                      augment=config['augment'],
                                      shrinking_mode=config['shrinking_mode'], 
                                      shrinking_structure_size=config['shrinking_structure_size'])
    batch = 0
    idx = 0
    
    batch_0_train = train_gen_etape_1.__getitem__(batch)
    print(f"Random pick from training dataset : {train_input_img_paths_etape_1[config['batch_size']*batch:config['batch_size']*(batch+1)][idx]}")

    plt.imshow(batch_0_train[0][idx])
    plt.show()
    fig = overlay_masks(batch_0_train[0][idx], batch_0_train[1][idx], colors=[(1, 0, 0)])
    plt.imshow(fig)
    plt.show()

    batch_0_val = val_gen_etape_1.__getitem__(batch)
    print(f"Random pick from validation dataset : {val_input_img_paths_etape_1[config['batch_size']*batch:config['batch_size']*(batch+1)][idx]}")
    
    plt.imshow(batch_0_val[0][idx])
    plt.show()
    fig = overlay_masks(batch_0_val[0][idx], batch_0_val[1][idx], colors=[(1, 0, 0)])
    plt.imshow(fig)
    plt.show()

    print("Load model")

    if config['year'] <= 1993:
        from tf_model_related import AttentionUnet, Losses
        # Free up RAM in case the model definition cells were run multiple times
        keras.backend.clear_session()
        attention_unet = AttentionUnet((config['img_size'], config['img_size']) + (1,))
        model_etape_1 = attention_unet.build_attention_unet()
        losses = Losses()
    else:
        from from_preprocessing_to_training import Losses, AttentionUnet
        # Free up RAM in case the model definition cells were run multiple times
        keras.backend.clear_session()
        losses = Losses()
        attention_unet = AttentionUnet((config['img_size'], config['img_size']) + (3,))
        custom_objects_list=[attention_unet.expend_as, attention_unet.AttnGatingBlock, attention_unet.UnetConv2D, attention_unet.UnetGatingSignal, losses.tversky, losses.focal_tversky]
        model_path = "output_models/35_14_61_49/2020/AttentionUnet_36000_trains_epochs_30_no_noise_img_size_256_LR_0_001_BS_64_FocalTverskyLoss_96acc_8414iou.h5"
        model_etape_1 = load_custom_model(model_path=model_path, custom_objects_list=custom_objects_list)

    now = datetime.datetime.now()
    now = str(now)[:19].replace('-','_').replace(' ','_').replace(':','_')
    
    if not os.path.exists(config['models_dir']+f"35_14_61_49/{config['year']}"):
        os.makedirs(config['models_dir']+f"35_14_61_49/{config['year']}")
    if not os.path.exists(config['models_dir']+f"35_14_61_49/{config['year']}/{now}/"):
        os.makedirs(config['models_dir']+f"35_14_61_49/{config['year']}/{now}/")

    callbacks = [
        PerformancePlotCallback(val_gen_etape_1, batch=0),
        keras.callbacks.LearningRateScheduler(lambda epoch: custom_LearningRate_schedular(epoch, max_epoch=config['epochnum'])),
        SaveCheckpointAtEpoch(config['models_dir']+f"35_14_61_49/{config['year']}/{now}/", 
                              f"checkpoint_AttentionUnet_{int(len(train_input_img_paths_etape_1))}" + \
                              f"_trains_STplusplus_" + \
                              f"{config['shrinking_mode']}_shrinked_{config['shrinking_structure_size']}_" +
                              f"{config['epochnum']}_epochs_schedular_epoch1" \
                              .replace('.', '_').replace('-','_') + ".h5",
                              [config['epochnum']//3, config['epochnum']*2//3, config['epochnum']]),
        keras.callbacks.ModelCheckpoint(config['models_dir']+f"35_14_61_49/{config['year']}/{now}/best_intermediary_AttentionUnet_{int(len(train_input_img_paths_etape_1))}_trains_STplusplus_{config['shrinking_mode']}_shrinked_{config['shrinking_structure_size']}_{config['epochnum']}_epochs_schedular".replace('.', '_').replace('-','_') + ".h5", save_best_only=True)
    ]

    
    model_etape_1.compile(optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']), 
                          loss=losses.focal_tversky, 
                          metrics=[keras.metrics.BinaryAccuracy(), 
                                   keras.metrics.IoU(num_classes=2, target_class_ids=[1])])

    print('''
    
    Launch training of the first model
    
    ''')
    hist = model_etape_1.fit(train_gen_etape_1, epochs=config['epochnum'], validation_data=val_gen_etape_1, callbacks=callbacks)

    print('''Training of the first model is done.
    ''')

    # <====================== Select reliable images for the 1st stage re-training ======================>
    print('\n================> Stage 3: '
          f"Select reliable images for the 1st stage re-training")
    
    base_fname = f"checkpoint_AttentionUnet_{int(len(train_input_img_paths_etape_1))}" + \
    f"_trains_STplusplus_" + \
    f"{config['shrinking_mode']}_shrinked_{config['shrinking_structure_size']}_" + \
    f"{config['epochnum']}_epochs_schedular_" \
    .replace('.', '_').replace('-','_')

    if config['year'] <= 1993:
        custom_objects_list=[attention_unet.expend_as, attention_unet.AttnGatingBlock, attention_unet.UnetConv2D, attention_unet.UnetGatingSignal, losses.tversky, losses.focal_tversky]
    
    model_1_path = config['models_dir']+f"35_14_61_49/{config['year']}/{now}/" + base_fname + str(config['epochnum'] // 3) + ".h5"
    model_2_path = config['models_dir']+f"35_14_61_49/{config['year']}/{now}/" + base_fname + str(config['epochnum']*2 // 3) + ".h5"
    model_3_path = config['models_dir']+f"35_14_61_49/{config['year']}/{now}/" + base_fname + str(config['epochnum']) + ".h5"
    model_1 = load_custom_model(model_path=model_1_path, custom_objects_list=custom_objects_list)
    model_2 = load_custom_model(model_path=model_2_path, custom_objects_list=custom_objects_list)
    model_3 = load_custom_model(model_path=model_3_path, custom_objects_list=custom_objects_list)

    new_vignettes_paths = sorted(
        list(
            itertools.chain.from_iterable(
                [glob(i + "*.jpg") for i in glob(f"vignettes/rgb_older/rgb_{config['year']}/*/", recursive=True)]
                )
            )
        )
    
    new_vignettes_paths = random.sample(new_vignettes_paths, len(new_vignettes_paths))

    id_to_reliability = select_reliable([model_1, model_2, model_3],
                                        vignettes_to_test_paths=new_vignettes_paths[:np.min([len(new_vignettes_paths), config['max_samples']])], 
                                        folder_name=now, year=config['year'])


    # <====================== Pseudo labeling of the reliable images ======================>
    print('\n================> Stage 4: '
          f"Pseudo labeling of the reliable images")
    best_model_path = f"{config['models_dir']}/35_14_61_49/{config['year']}/{now}/best_intermediary_AttentionUnet_{int(len(train_input_img_paths_etape_1))}_trains_STplusplus_{config['shrinking_mode']}_shrinked_{config['shrinking_structure_size']}_{config['epochnum']}_epochs_schedular".replace('.', '_').replace('-','_') + ".h5"
    best_model_etape_1 = load_custom_model(model_path=best_model_path, custom_objects_list=custom_objects_list)

    label(model=best_model_etape_1, 
          paths_of_the_reliable_pictures=f"{config['reliable_img_main_folder']}/{config['year']}/{now}/reliable_ids.txt", 
          folder_name=now, reliable=True, year=config['year'])

    with open(f"{config['reliable_img_main_folder']}/{config['year']}/{now}/reliable_ids.txt", 'r') as f:
        reliable_imgs = f.read().splitlines()

    reliable_imgs.sort(key=lambda x: str(os.path.basename(x).split('.')[0]))

    reliable_pseudo_masks = sorted(
        list(
            itertools.chain.from_iterable(
                [glob(i + "*.png") for i in glob(f"pseudo_mask_STPlusplus/{str(config['year'])}/{now}/reliable/")]
                )
            )
    )

    reliable_pseudo_masks.sort(key=lambda x: str(os.path.basename(x).split('.')[0]))

    assert [os.path.basename(i)[:-4] for i in reliable_pseudo_masks] == [os.path.basename(i)[:-4] for i in reliable_imgs] 

    # <====================== Training of the 2nd model ======================>
    print('\n================> Stage 5: '
          f"Training of the 2nd model with the previous images and the reliable images")
    img_paths_etape_2 = img_paths + reliable_imgs
    mask_paths_etape_2 = mask_paths + reliable_pseudo_masks

    train_input_img_paths_etape_2, val_input_img_paths_etape_2, train_target_img_paths_etape_2, val_target_img_paths_etape_2 = train_test_split(img_paths_etape_2, mask_paths_etape_2, test_size=config['test_split'], shuffle=True)

    train_gen_etape_2 = AugmentedOrthosSequence(batch_size=config['batch_size'], 
                                        input_img_paths=train_input_img_paths_etape_2, 
                                        target_img_paths=train_target_img_paths_etape_2, 
                                        rgb=config['rgb'], 
                                        add_noise=config['add_noise'], 
                                        year=2020, 
                                        augment=config['augment'], 
                                        shrinking_mode=config['shrinking_mode'], 
                                        shrinking_structure_size=config['shrinking_structure_size'])
    val_gen_etape_2 = AugmentedOrthosSequence(batch_size=config['batch_size'], 
                                      input_img_paths=val_input_img_paths_etape_2, 
                                      target_img_paths=val_target_img_paths_etape_2, 
                                      rgb=config['rgb'], 
                                      add_noise=config['add_noise'], 
                                      year=2020,
                                      augment=config['augment'], 
                                      shrinking_mode=config['shrinking_mode'], 
                                      shrinking_structure_size=config['shrinking_structure_size'])

    batch = 0
    idx = 0
    
    batch_0_train = train_gen_etape_2.__getitem__(batch)
    print(f"Random pick from training dataset 2 : {train_input_img_paths_etape_2[config['batch_size']*batch:config['batch_size']*(batch+1)][idx]}")

    plt.imshow(batch_0_train[0][idx])
    plt.show()
    fig = overlay_masks(batch_0_train[0][idx], batch_0_train[1][idx], colors=[(1, 0, 0)])
    plt.imshow(fig)
    plt.show()

    batch_0_val = val_gen_etape_2.__getitem__(batch)
    print(f"Random pick from validation dataset 2 : {val_input_img_paths_etape_2[config['batch_size']*batch:config['batch_size']*(batch+1)][idx]}")
    
    plt.imshow(batch_0_val[0][idx])
    plt.show()
    fig = overlay_masks(batch_0_val[0][idx], batch_0_val[1][idx], colors=[(1, 0, 0)])
    plt.imshow(fig)
    plt.show()

    print("Load 2nd model")

    if config['year'] <= 1993:
        # Free up RAM in case the model definition cells were run multiple times
        keras.backend.clear_session()
        model_etape_2 = attention_unet.build_attention_unet()
    else:
        # Free up RAM in case the model definition cells were run multiple times
        keras.backend.clear_session()
        model_path = "output_models/35_14_61_49/2020/AttentionUnet_36000_trains_epochs_30_no_noise_img_size_256_LR_0_001_BS_64_FocalTverskyLoss_96acc_8414iou.h5"
        model_etape_2 = load_custom_model(model_path=model_path, custom_objects_list=custom_objects_list)

    callbacks = [
        PerformancePlotCallback(val_gen_etape_2, batch=0),
        keras.callbacks.LearningRateScheduler(lambda epoch: custom_LearningRate_schedular(epoch, max_epoch=config['epochnum'])),
        keras.callbacks.EarlyStopping(patience=5, start_from_epoch=5),
        SaveCheckpointAtEpoch(config['models_dir']+f"35_14_61_49/{config['year']}/{now}/", 
                              f"step_2_checkpoint_AttentionUnet_{int(len(train_input_img_paths_etape_2))}" + \
                              f"_trains_STplusplus_" + \
                              f"{config['shrinking_mode']}_shrinked_{config['shrinking_structure_size']}_" +
                              f"{config['epochnum']}_epochs_schedular_epoch1" \
                              .replace('.', '_').replace('-','_') + ".h5",
                              [config['epochnum']//3, config['epochnum']*2//3, config['epochnum']]),
        keras.callbacks.ModelCheckpoint(config['models_dir']+f"35_14_61_49/{config['year']}/{now}/step_2_best_intermediary_AttentionUnet_{int(len(train_input_img_paths_etape_2))}_trains_STplusplus_{config['shrinking_mode']}_shrinked_{config['shrinking_structure_size']}_{config['epochnum']}_epochs_schedular".replace('.', '_').replace('-','_') + ".h5", save_best_only=True)
    ]
    
    model_etape_2.compile(optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']), loss=losses.focal_tversky, metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.IoU(num_classes=2, target_class_ids=[1])])
    
    print('''
    
    Launch training of the 2nd model
    
    ''')
    
    hist = model_etape_2.fit(train_gen_etape_2, epochs=config['epochnum'], validation_data=val_gen_etape_2, callbacks=callbacks)

    print('''Training of the 2nd model is done.
    ''')
    
    # <====================== Select reliable images for the 1st stage re-training ======================>
    print('\n================> Stage 6: '
          f"Pseudo labeling of the unreliable images")
    
    best_model_path = config['models_dir']+f"35_14_61_49/{config['year']}/{now}/step_2_best_intermediary_AttentionUnet_{int(len(train_input_img_paths_etape_2))}_trains_STplusplus_{config['shrinking_mode']}_shrinked_{config['shrinking_structure_size']}_{config['epochnum']}_epochs_schedular".replace('.', '_').replace('-','_') + ".h5"
    best_model_etape_2 = load_custom_model(model_path=best_model_path, custom_objects_list=custom_objects_list)

    label(best_model_etape_2, f"reliable_img_path/{config['year']}/{now}/unreliable_ids.txt", now, False, config['year'])
    
    with open(f"reliable_img_path/{config['year']}/{now}/unreliable_ids.txt", 'r') as f:
        unreliable_imgs = f.read().splitlines()
    
    unreliable_imgs.sort(key=lambda x: str(os.path.basename(x).split('.')[0]))
    
    unreliable_pseudo_masks = sorted(
        list(
            itertools.chain.from_iterable(
                [glob(i + "*.png") for i in glob(f"pseudo_mask_STPlusplus/{str(config['year'])}/{now}/unreliable/")]
                )
            )
    )
    
    unreliable_pseudo_masks.sort(key=lambda x: str(os.path.basename(x).split('.')[0]))

    assert [os.path.basename(i)[:-4] for i in unreliable_pseudo_masks] == [os.path.basename(i)[:-4] for i in unreliable_imgs] 

    # <====================== Training of the 3nd model ======================>
    print('\n================> Stage 7: '
          f"Training of the 3rd model with the previous images and both the reliable and the unreliable images")
    
    img_paths_etape_3 = img_paths_etape_2 + unreliable_imgs
    mask_paths_etape_3 = mask_paths_etape_2 + unreliable_pseudo_masks

    train_input_img_paths_etape_3, val_input_img_paths_etape_3, train_target_img_paths_etape_3, val_target_img_paths_etape_3 = train_test_split(img_paths_etape_3, mask_paths_etape_3, test_size=config['test_split'], shuffle=True)
    
    train_gen_etape_3 = AugmentedOrthosSequence(batch_size=config['batch_size'], 
                                        input_img_paths=train_input_img_paths_etape_3, 
                                        target_img_paths=train_target_img_paths_etape_3, 
                                        rgb=config['rgb'], 
                                        add_noise=config['add_noise'], 
                                        year=2020, 
                                        augment=config['augment'], 
                                        shrinking_mode=config['shrinking_mode'], 
                                        shrinking_structure_size=config['shrinking_structure_size'])
    val_gen_etape_3 = AugmentedOrthosSequence(batch_size=config['batch_size'], 
                                      input_img_paths=val_input_img_paths_etape_3, 
                                      target_img_paths=val_target_img_paths_etape_3, 
                                      rgb=config['rgb'], 
                                      add_noise=config['add_noise'], 
                                      year=2020,
                                      augment=config['augment'], 
                                      shrinking_mode=config['shrinking_mode'], 
                                      shrinking_structure_size=config['shrinking_structure_size'])

    batch = 0
    idx = 0
    
    batch_0_train = train_gen_etape_3.__getitem__(batch)
    print(f"Random pick from training dataset 3 : {train_input_img_paths_etape_3[config['batch_size']*batch:config['batch_size']*(batch+1)][idx]}")

    plt.imshow(batch_0_train[0][idx])
    plt.show()
    fig = overlay_masks(batch_0_train[0][idx], batch_0_train[1][idx], colors=[(1, 0, 0)])
    plt.imshow(fig)
    plt.show()

    batch_0_val = val_gen_etape_3.__getitem__(batch)
    print(f"Random pick from validation dataset 3 : {val_input_img_paths_etape_3[config['batch_size']*batch:config['batch_size']*(batch+1)][idx]}")
    
    plt.imshow(batch_0_val[0][idx])
    plt.show()
    fig = overlay_masks(batch_0_val[0][idx], batch_0_val[1][idx], colors=[(1, 0, 0)])
    plt.imshow(fig)
    plt.show()

    print("Load 3rd model")

    if config['year'] <= 1993:
        # Free up RAM in case the model definition cells were run multiple times
        keras.backend.clear_session()
        model_etape_3 = attention_unet.build_attention_unet()
    else:
        # Free up RAM in case the model definition cells were run multiple times
        keras.backend.clear_session()
        model_path = "output_models/35_14_61_49/2020/AttentionUnet_36000_trains_epochs_30_no_noise_img_size_256_LR_0_001_BS_64_FocalTverskyLoss_96acc_8414iou.h5"
        model_etape_3 = load_custom_model(model_path=model_path, custom_objects_list=custom_objects_list)

    callbacks = [
        PerformancePlotCallback(val_gen_etape_3, batch=0),
        keras.callbacks.LearningRateScheduler(lambda epoch: custom_LearningRate_schedular(epoch, max_epoch=config['epochnum'])),
        keras.callbacks.EarlyStopping(patience=5, start_from_epoch=5),
        SaveCheckpointAtEpoch(config['models_dir']+f"35_14_61_49/{config['year']}/{now}/", 
                              f"step_3_checkpoint_AttentionUnet_{int(len(train_input_img_paths_etape_2))}" + \
                              f"_trains_STplusplus_" + \
                              f"{config['shrinking_mode']}_shrinked_{config['shrinking_structure_size']}_" +
                              f"{config['epochnum']}_epochs_schedular_epoch1" \
                              .replace('.', '_').replace('-','_') + ".h5",
                              [config['epochnum']//3, config['epochnum']*2//3, config['epochnum']]),
        keras.callbacks.ModelCheckpoint(config['models_dir']+f"35_14_61_49/{config['year']}/{now}/step_3_best_intermediary_AttentionUnet_{int(len(train_input_img_paths_etape_2))}_trains_STplusplus_{config['shrinking_mode']}_shrinked_{config['shrinking_structure_size']}_{config['epochnum']}_epochs_schedular".replace('.', '_').replace('-','_') + ".h5", save_best_only=True)
    ]
    
    model_etape_3.compile(optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']), loss=losses.focal_tversky, metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.IoU(num_classes=2, target_class_ids=[1])])
    
    print('''
    
    Launch training of the 3rd model
    
    ''')
    
    hist = model_etape_3.fit(train_gen_etape_3, epochs=config['epochnum'], validation_data=val_gen_etape_3, callbacks=callbacks)

    print('''Training of the 3rd model is done.
    ''')

    print(f"Random prediction from final model (not necessarily the best)")
    val_preds_etape_3 = model_etape_3.predict(val_gen_etape_3)

    batch_0 = val_gen_etape_3.__getitem__(0)
    idx = random.randint(0, config['batch_size']-1)
    plt.imshow(batch_0[0][idx])
    plt.show()

    fig = overlay_masks(batch_0[0][idx], np.round(val_preds_etape_3[idx]), colors=[(1, 0, 0)])
    plt.imshow(fig)
    plt.show()
    
    return model_etape_3
    