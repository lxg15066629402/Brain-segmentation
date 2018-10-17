import nibabel as nib
import numpy as np

from keras.utils import np_utils
from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
import itertools

from keras import backend as K
from keras.layers import Activation
from keras.layers import Input
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import Cropping3D
from keras.layers.core import Permute
from keras.layers.core import Reshape
from keras.layers.merge import concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.models import load_model


# Fix random seed for reproducibility?
# Better to follow the advice in Keras FAQ:
# "How can I obtain reproducible results using Keras during development?"
seed = 7
np.random.seed(seed)

# Problem configuration
num_classes = 3

patience = 1
# iSeg2017-nic_vicorob-master/models/iSeg2017/outrun_step_1.h5
model_filename = 'models/iSeg2017/outrun_step_1.h5'
csv_filename = 'log/iSeg2017/outrun_step_1.cvs'

nb_epoch = 20
validation_split = 0.25  # train set and vaild set 

class_mapper = {0: 0, 10: 0, 150: 1, 250: 2}
class_mapper_inv = {0: 0, 1: 10, 2: 150, 3: 250}


# General utils for reading and saving data
def get_filename(set_name, case_idx, input_name, loc='datasets'):
    # pattern = '{0}/iSeg2017/iSeg-2017-{1}/subject-{2}-{3}.hdr' #
    # pattern = 'datasets/iSeg2017/iSeg-2017-{1}/subject-{2}-{3}.hdr'
    pattern = '{0}/iSeg2017/iSeg-2017-{1}/subject-{2}-{3}.hdr'
    return pattern.format(loc, set_name, case_idx, input_name)


def get_set_name(case_idx):
    # return 'Training' if case_idx < 11 else 'Testing'
    return 'Training' if case_idx < 9 else 'Testing'


def read_data(case_idx, input_name, loc='datasets'):
    set_name = get_set_name(case_idx)

    image_path = get_filename(set_name, case_idx, input_name, loc)

    return nib.load(image_path)


def read_vol(case_idx, input_name, loc='datasets'):
    image_data = read_data(case_idx, input_name, loc)

    return image_data.get_data()[:, :, :, 0]


def save_vol(segmentation, case_idx, loc='results'):
    set_name = get_set_name(case_idx)
    input_image_data = read_data(case_idx, 'T1')


    segmentation_vol = np.empty(input_image_data.shape)
    segmentation_vol[:144, :192, :256, 0] = segmentation

    filename = get_filename(set_name, case_idx, 'label', loc)
    nib.save(nib.analyze.AnalyzeImage(
        segmentation_vol.astype('uint8'), input_image_data.affine), filename)


# Data preparation utils
def extract_patches(volume, patch_shape, extraction_step):
    patches = sk_extract_patches(
        volume,
        patch_shape=patch_shape,
        extraction_step=extraction_step)

    ndim = len(volume.shape)
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches, ) + patch_shape)


def build_set(T1_vols, T2_vols, label_vols, extraction_step=(9, 9, 9)):
    patch_shape = (27, 27, 27)
    label_selector = [slice(None)] + [slice(9, 18) for i in range(3)]

    # Extract patches from input volumes and ground truth
    x = np.zeros((0, 2, 27, 27, 27))
    y = np.zeros((0, 9 * 9 * 9, num_classes))
    for idx in range(len(T1_vols)):
        y_length = len(y)

        label_patches = extract_patches(
            label_vols[idx], patch_shape, extraction_step)
        label_patches = label_patches[label_selector]

        # Select only those who are important for processing
        valid_idxs = np.where(np.sum(label_patches, axis=(1, 2, 3)) != 0)

        # Filtering extracted patches
        label_patches = label_patches[valid_idxs]

        x = np.vstack((x, np.zeros((len(label_patches), 2, 27, 27, 27))))
        y = np.vstack((y, np.zeros((len(label_patches), (9 * 9 * 9), num_classes))))

        for i in range(len(label_patches)):
            # y[i + y_length,:, :] = np_utils.to_categorical(label_patches[i,:,:,:],num_classes)
            y[i + y_length,:, :] = np_utils.to_categorical(label_patches[i].flatten(), num_classes)

        del label_patches

        # Sampling strategy: reject samples which labels are only zeros
        T1_train = extract_patches(T1_vols[idx], patch_shape, extraction_step)
        x[y_length:, 0, :, :, :] = T1_train[valid_idxs]

        del T1_train

        # Sampling strategy: reject samples which labels are only zeros
        T2_train = extract_patches(T2_vols[idx], patch_shape, extraction_step)
        x[y_length:, 1, :, :, :] = T2_train[valid_idxs]

        del T2_train


    return x, y

# Reconstruction utils
def generate_indexes(patch_shape, expected_shape):
    ndims = len(patch_shape)

    poss_shape = [patch_shape[i + 1] *
                  (expected_shape[i] // patch_shape[i + 1]) for i in range(ndims - 1)]

    idxs = [range(patch_shape[i + 1],
                  poss_shape[i] - patch_shape[i + 1],
                  patch_shape[i + 1]) for i in range(ndims - 1)]

    return itertools.product(*idxs)


def reconstruct_volume(patches, expected_shape):
    patch_shape = patches.shape

    assert len(patch_shape) - 1 == len(expected_shape)

    reconstructed_img = np.zeros(expected_shape)

    for count, coord in enumerate(
            generate_indexes(
            patch_shape, expected_shape)):
        selection = [slice(coord[i], coord[i] + patch_shape[i + 1])
                     for i in range(len(coord))]
        reconstructed_img[selection] = patches[count]

    return reconstructed_img


# Architecture
K.set_image_dim_ordering('th')

# For understanding the architecture itself, I recommend checking the following article
# Dolz, J. et al. 3D fully convolutional networks for subcortical segmentation in MRI :
# A large-scale study. Neuroimage, 2017.


# add SENet
# def Global_Average_Pooling(x):
#     return Average_pooling(x, name='Global_avg_pooling')
#     # return global_avg_pool(x)
#
#
# def Average_pooling(x, pool_size=[2, 2], stride=2, padding='SAME'):
#     return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)
#
#
# def Batch_Normalization(x, training, scope):
#     with arg_scope([batch_norm],
#                    scope=scope,
#                    updates_collections=None,
#                    decay=0.9,
#                    center=True,
#                    scale=True,
#                    zero_debias_moving_mean=True):
#         return tf.cond(training,
#                        lambda: batch_norm(inputs=x, is_training=training, reuse=None),
#                        lambda: batch_norm(inputs=x, is_training=training, reuse=True))
#
#
# def Relu(x):
#     return Relu(x)
#
#
# def Sigmoid(x):
#     return Sigmoid(x)
#
#
# def Concatenation(layers):
#     return tf.concat(layers, axis=3)
#
#
# def Fully_connected(x, units=2, layer_name='fully_connected'):
#     with tf.name_scope(layer_name):
#         return tf.layers.dense(inputs=x, use_bias=False, units=units)
#
#
# def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
#     with tf.name_scope(layer_name):
#         #  5d shujude chuli
#         input_x = input_x(:,:,:,:)
#         squeeze = self.Global_Average_Pooling(input_x)
#         input_x =
#
#         excitation = self.Fully_connected(squeeze, units=out_dim / ratio,
#                                           layer_name=layer_name + '_fully_connected1')
#         excitation = self.Relu(excitation)
#         excitation = self.Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_fully_connected2')
#         excitation = self.Sigmoid(excitation)
#
#         excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
#
#         scale = input_x * excitation
#
#         return scale

# model structure 
def generate_model(num_classes):
    init_input = Input((2, 27, 27, 27))     # 27*27*27 2 is model

    x = Conv3D(25, kernel_size=(3, 3, 3))(init_input)
    x = PReLU()(x)
    x = Conv3D(25, kernel_size=(3, 3, 3))(x)
    x = PReLU()(x)
    x = Conv3D(25, kernel_size=(3, 3, 3))(x)
    x = PReLU()(x)

    y = Conv3D(50, kernel_size=(3, 3, 3))(x)
    y = PReLU()(y)
    y = Conv3D(50, kernel_size=(3, 3, 3))(y)
    y = PReLU()(y)
    y = Conv3D(50, kernel_size=(3, 3, 3))(y)
    y = PReLU()(y)

    z = Conv3D(75, kernel_size=(3, 3, 3))(y)
    z = PReLU()(z)
    z = Conv3D(75, kernel_size=(3, 3, 3))(z)
    z = PReLU()(z)
    z = Conv3D(75, kernel_size=(3, 3, 3))(z)
    z = PReLU()(z)

    x_crop = Cropping3D(cropping=((6, 6), (6, 6), (6, 6)))(x)
    y_crop = Cropping3D(cropping=((3, 3), (3, 3), (3, 3)))(y)

    concat = concatenate([x_crop, y_crop, z], axis=1)

    fc = Conv3D(400, kernel_size=(1, 1, 1))(concat)
    fc = PReLU()(fc)
    fc = Conv3D(200, kernel_size=(1, 1, 1))(fc)
    fc = PReLU()(fc)
    fc = Conv3D(150, kernel_size=(1, 1, 1))(fc)
    fc = PReLU()(fc)


    pred = Conv3D(num_classes, kernel_size=(1, 1, 1))(fc)
    pred = PReLU()(pred)
    pred = Reshape((num_classes, 9 * 9 * 9))(pred)
    pred = Permute((2, 1))(pred)
    pred = Activation('softmax')(pred)

    model = Model(inputs=init_input, outputs=pred)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy'])
    return model

# Initial segmentation
# read data
# T1_vols = np.empty((10, 144, 192, 256))  # MRI image  144 * 192 * 256
T1_vols = np.empty((8, 144, 192, 256))  # MRI  144 * 192 * 256
# T2_vols = np.empty((10, 144, 192, 256))
T2_vols = np.empty((8, 144, 192, 256))
# label_vols = np.empty((10, 144, 192, 256))
label_vols = np.empty((8, 144, 192, 256))
# for case_idx in range(1, 11):
for case_idx in range(1, 9):
    T1_vols[(case_idx - 1), :, :, :] = read_vol(case_idx, 'T1')
    T2_vols[(case_idx - 1), :, :, :] = read_vol(case_idx, 'T2')
    label_vols[(case_idx - 1), :, :, :] = read_vol(case_idx, 'label')


# Pre-processing
# Intensity normalisation (zero mean and unit variance)
# image mean normalization
T1_mean = T1_vols.mean()
T1_std = T1_vols.std()
T1_vols = (T1_vols - T1_mean) / T1_std
T2_mean = T2_vols.mean()
T2_std = T2_vols.std()
T2_vols = (T2_vols - T2_mean) / T2_std

# Combine labels of BG and CSF
for class_idx in class_mapper:
    label_vols[label_vols == class_idx] = class_mapper[class_idx]


# Data preparation  
x_train, y_train = build_set(T1_vols, T2_vols, label_vols, (3, 9, 3))



# Configure callbacks
# Early stopping for reducing over-fitting risk
# EarlyStopping()Function: Stop training when monitoring volume stops improving
stopper = EarlyStopping(patience=patience)

# Model checkpoint to save the training results
checkpointer = ModelCheckpoint(
    filepath=model_filename.format(1),
    verbose=0,
    save_best_only=True,
    save_weights_only=True)

# CSVLogger to save the training results in a csv file
csv_logger = CSVLogger(csv_filename.format(1), separator=';')

callbacks = [checkpointer, csv_logger, stopper]


# Training
# Build model
model = generate_model(num_classes)

# 2 log lines per epoch
model.fit(
    x_train,
    y_train,
    epochs=nb_epoch,
    validation_split=validation_split,
    verbose=2,
    callbacks=callbacks)

# freeing space  
del x_train
del y_train


# Classification
# Load best model

model = generate_model(num_classes)
model.load_weights(model_filename.format(1))

# for case_idx in range(11, 24) :
for case_idx in range(9, 11):
    T1_test_vol = read_vol(case_idx, 'T1')[:144, :192, :256]
    T2_test_vol = read_vol(case_idx, 'T2')[:144, :192, :256]

    # x_test = np.zeros((6916, 2, 27, 27, 27))  # 6916 
    x_test = np.zeros((6916, 2, 27, 27, 27))  #  -1 & 1064
    x_test[:, 0, :, :, :] = extract_patches(
        T1_test_vol, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9))
    x_test[:, 1, :, :, :] = extract_patches(
        T2_test_vol, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9))

    x_test[:, 0, :, :, :] = (x_test[:, 0, :, :, :] - T1_mean) / T1_std
    x_test[:, 1, :, :, :] = (x_test[:, 1, :, :, :] - T2_mean) / T2_std

    pred = model.predict(x_test, verbose=2)
    pred_classes = np.argmax(pred, axis=2)
    pred_classes = pred_classes.reshape((len(pred_classes), 9, 9, 9))
    segmentation = reconstruct_volume(pred_classes, (144, 192, 256))

    # 0：background
    # 10：（CSF）
    # 150：（GM）
    # 250：（WM)
    csf = np.logical_and(segmentation == 0, T1_test_vol != 0)
    segmentation[segmentation == 2] = 250
    segmentation[segmentation == 1] = 150
    segmentation[csf] = 10

    save_vol(segmentation, case_idx, 'results')

    print("Finished segmentation of case # {}".format(case_idx))

print("Done with Step 1")


# Pseudo-labelling step
# read data
# sure = range(0, 10)
sure = range(0, 8)
# unsure = range(11, 23)
unsure = range(9, 10) #

# T1_vols = np.empty((23, 144, 192, 256))
T1_vols = np.empty((10, 144, 192, 256))
# T2_vols = np.empty((23, 144, 192, 256))
T2_vols = np.empty((10, 144, 192, 256))
# label_vols = np.empty((23, 144, 192, 256))
label_vols = np.empty((10, 144, 192, 256))
# for case_idx in range(1, 24):
for case_idx in range(1, 11):
    # loc = 'datasets' if case_idx < 11 else 'results'
    loc = 'datasets' if case_idx < 9 else 'results'

    T1_vols[(case_idx - 1), :, :, :] = read_vol(case_idx,
                                                'T1')[:144, :192, :256]
    T2_vols[(case_idx - 1), :, :, :] = read_vol(case_idx,
                                                'T2')[:144, :192, :256]
    label_vols[(case_idx - 1), :, :, :] = read_vol(case_idx,
                                                   'label', loc)[:144, :192, :256]


# Pre-processing
# Intensity normalisation (zero mean and unit variance)  0 mean and std 
T1_mean = T1_vols.mean()
T1_std = T1_vols.std()
T1_vols = (T1_vols - T1_mean) / T1_std
T2_mean = T2_vols.mean()
T2_std = T2_vols.std()
T2_vols = (T2_vols - T2_mean) / T2_std

# Combine labels of BG and CSF
for class_idx in class_mapper:
    label_vols[label_vols == class_idx] = class_mapper[class_idx]

# Data preparation
x_sure, y_sure = build_set(
    T1_vols[sure], T2_vols[sure], label_vols[sure], (3, 9, 3))
x_unsure, y_unsure = build_set(
    T1_vols[unsure], T2_vols[unsure], label_vols[unsure])

x_train = np.vstack((x_sure, x_unsure))
y_train = np.vstack((y_sure, y_unsure))

del x_sure
del x_unsure
del y_sure
del y_unsure


# Configure callbacks
# Early stopping for reducing over-fitting risk
stopper = EarlyStopping(patience=patience)

# Model checkpoint to save the training results
checkpointer = ModelCheckpoint(
    filepath=model_filename.format(2),
    verbose=0,
    save_best_only=True,
    save_weights_only=True)

# CSVLogger to save the training results in a csv file
csv_logger = CSVLogger(csv_filename.format(2), separator=';')

callbacks = [checkpointer, csv_logger, stopper]


# train
# Build model
model = generate_model(num_classes)

model.fit(
    x_train,
    y_train,
    epochs=nb_epoch,
    validation_split=validation_split,
    verbose=2,
    callbacks=callbacks)

# freeing space
del x_train
del y_train


# Clasification
# Load best model
model = generate_model(num_classes)
model.load_weights(model_filename.format(2))


#for case_idx in range(11, 24):
for case_idx in range(9, 11):
    T1_test_vol = read_vol(case_idx, 'T1')[:144, :192, :256]
    T2_test_vol = read_vol(case_idx, 'T2')[:144, :192, :256]

    # x_test = np.zeros((6916, 2, 27, 27, 27))
    x_test = np.zeros((6916, 2, 27, 27, 27)) # ??
    x_test[:, 0, :, :, :] = extract_patches(
        T1_test_vol, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9))
    x_test[:, 1, :, :, :] = extract_patches(
        T2_test_vol, patch_shape=(27, 27, 27), extraction_step=(9, 9, 9))

    x_test[:, 0, :, :, :] = (x_test[:, 0, :, :, :] - T1_mean) / T1_std
    x_test[:, 1, :, :, :] = (x_test[:, 1, :, :, :] - T2_mean) / T2_std

    pred = model.predict(x_test, verbose=2)
    pred_classes = np.argmax(pred, axis=2)
    pred_classes = pred_classes.reshape((len(pred_classes), 9, 9, 9))
    segmentation = reconstruct_volume(pred_classes, (144, 192, 256))

    csf = np.logical_and(
        segmentation == 0,
        T1_test_vol != 0)  # np.logical_and
    segmentation[segmentation == 2] = 250
    segmentation[segmentation == 1] = 150
    segmentation[csf] = 10

    save_vol(segmentation, case_idx, 'refined-results')  # refined 

    print("Finished segmentation of case # {}".format(case_idx))

print("Done with Step 2")
