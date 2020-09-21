#convert downloading script to executable and run
! chmod a+x /data/downloaddata.sh
! ./data/downloaddata.sh

#download VGG 19
! chmod a+x net/downloadnet.sh
!./downloadnet.sh vgg19


import numpy as np
import PIL.Image
import scipy.io as sio
import scipy.ndimage as nd
import os
from datetime import datetime
from scipy.optimize import minimize
import pickle
from itertools import product
import caffe


##  ICNN Loss
def L2_loss(feat, feat0, mask=1.):
    d = feat - feat0
    loss = (d*d*mask).sum()
    grad = 2 * d * mask
    return loss, grad

def inner_loss(feat, feat0, mask=1.):
    loss = -(feat*feat0*mask).sum()
    grad = -feat0*mask
    return loss, grad



def gram_loss(feat, feat0, mask=1.):
    feat_size = feat.shape[:]
    N = feat_size[0]
    M = feat_size[1] * feat_size[2]
    feat_gram = gram(feat, mask)
    feat0_gram = gram(feat0, mask)
    feat = feat.reshape(N, M)
    loss = ((feat_gram - feat0_gram)**2).sum() / (4*(N**2)*(M**2))
    grad = np.dot((feat_gram - feat0_gram),
                  feat).reshape(feat_size) * mask / ((N**2)*(M**2))
    return loss, grad

def switch_loss_fun(loss_type):
    if loss_type == 'l2':
        return L2_loss
    elif loss_type == 'inner':
        return inner_loss
    elif loss_type == 'gram':
        return gram_loss
    else:
        raise ValueError('Please check loss function type!')


def img_caffe_preproc(img, img_mean=np.float32([104, 117, 123])):
    '''convert to Caffe's input '''
    return np.float32(np.transpose(img, (2, 0, 1))[::-1]) - np.reshape(img_mean, (3, 1, 1))


def img_caffe_deproc(img, img_mean=np.float32([104, 117, 123])):
    '''convert from Caffe's input'''
    return np.dstack((img + np.reshape(img_mean, (3, 1, 1)))[::-1])


def normalise_img(img):
    '''Normalize the image.'''
    img = img - img.min()
    if img.max() > 0:
        img = img * (255.0/img.max())
    img = np.uint8(img)
    return img


def get_cnn_features(net, img, layer_list):
    '''Calculate the CNN features of the input image.
    Output the CNN features at layers in layer_list.
    The CNN features of multiple layers are assembled in a python dictionary, arranged in pairs of layer name (key) and CNN features (value).
    '''
    h, w = net.blobs['data'].data.shape[-2:]
    net.blobs['data'].reshape(1, 3, h, w)
    img_mean = net.transformer.mean['data']
    img = img_caffe_preproc(img, img_mean)
    net.blobs['data'].data[0] = img
    net.forward()
    cnn_features = {}
    for layer in layer_list:
        feat = net.blobs[layer].data[0].copy()
        cnn_features[layer] = feat
    return cnn_features


def p_norm(x, p=2):
    '''p-norm loss and gradient'''
    loss = np.sum(np.abs(x) ** p)
    grad = p * (np.abs(x) ** (p-1)) * np.sign(x)
    return loss, grad


def img_chan_norm(img):
    '''calculate the norm of different channel'''
    img_norm = np.sqrt(img[0]**2 + img[1]**2 + img[2]**2)
    return img_norm

def clip_extreme_value(img, pct=1):
    #clip the pixels with extreme values
    if pct < 0:
        pct = 0.

    if pct > 100:
        pct = 100.

    img = np.clip(img, np.percentile(img, pct/2.),
                  np.percentile(img, 100-pct/2.))
    return img

def clip_small_norm_pixel(img, pct=1):
    #clip pixels with small RGB norm#
    if pct < 0:
        pct = 0.

    if pct > 100:
        pct = 100.

    img_norm = img_chan_norm(img)
    small_pixel = img_norm < np.percentile(img_norm, pct)

    img[0][small_pixel] = 0
    img[1][small_pixel] = 0
    img[2][small_pixel] = 0
    return img

def clip_small_contribution_pixel(img, grad, pct=1):
    #clip pixels with small contribution#
    if pct < 0:
        pct = 0.

    if pct > 100:
        pct = 100.

    img_contribution = img_chan_norm(img*grad)
    small_pixel = img_contribution < np.percentile(img_contribution, pct)

    img[0][small_pixel] = 0
    img[1][small_pixel] = 0
    img[2][small_pixel] = 0
    return img

def sort_layer(net, layer_list):
    #sort layers in the list as the order in the net
    layer_index_list = []
    for layer in layer_list:
        # net.blobs is collections.OrderedDict
        for layer_index, layer0 in enumerate(net.blobs.keys()):
            if layer0 == layer:
                layer_index_list.append(layer_index)
                break
    layer_index_list_sorted = sorted(layer_index_list)
    layer_list_sorted = []

    for layer_index in (layer_index_list_sorted):
        list_index = (layer_index_list.index(layer_index))
        layer = list(layer_list)[list_index]
        layer_list_sorted.append(layer)
    return layer_list_sorted

def create_feature_masks(features, masks=None, channels=None):

    feature_masks = {}
    for layer in features.keys():
        if (masks is None or masks == {} or masks == [] or (layer not in masks.keys())) and (channels is None or channels == {} or channels == [] or (layer not in channels.keys())):  # use all features and all channels
            feature_masks[layer] = np.ones_like(features[layer])
        elif isinstance(masks, dict) and (layer in masks.keys()) and isinstance(masks[layer], np.ndarray) and masks[layer].ndim == 3 and masks[layer].shape[0] == features[layer].shape[0] and masks[layer].shape[1] == features[layer].shape[1] and masks[layer].shape[2] == features[layer].shape[2]:  # 3D mask
            feature_masks[layer] = masks[layer]
        # 1D feat and 1D mask
        elif isinstance(masks, dict) and (layer in masks.keys()) and isinstance(masks[layer], np.ndarray) and features[layer].ndim == 1 and masks[layer].ndim == 1 and masks[layer].shape[0] == features[layer].shape[0]:
            feature_masks[layer] = masks[layer]
        elif (masks is None or masks == {} or masks == [] or (layer not in masks.keys())) and isinstance(channels, dict) and (layer in channels.keys()) and isinstance(channels[layer], np.ndarray) and channels[layer].size > 0:  # select channels
            mask_2D = np.ones_like(features[layer][0])
            mask_3D = np.tile(mask_2D, [len(channels[layer]), 1, 1])
            feature_masks[layer] = np.zeros_like(features[layer])
            feature_masks[layer][channels[layer], :, :] = mask_3D
        # use 2D mask select features for all channels
        elif isinstance(masks, dict) and (layer in masks.keys()) and isinstance(masks[layer], np.ndarray) and masks[layer].ndim == 2 and (channels is None or channels == {} or channels == [] or (layer not in channels.keys())):
            mask_2D_0 = masks[layer]
            mask_size0 = mask_2D_0.shape
            mask_size = features[layer].shape[1:]
            if mask_size0[0] == mask_size[0] and mask_size0[1] == mask_size[1]:
                mask_2D = mask_2D_0
            else:
                mask_2D = np.ones(mask_size)
                n_dim1 = min(mask_size0[0], mask_size[0])
                n_dim2 = min(mask_size0[1], mask_size[1])
                idx0_dim1 = np.arange(n_dim1) + \
                    round((mask_size0[0] - n_dim1)/2)
                idx0_dim2 = np.arange(n_dim2) + \
                    round((mask_size0[1] - n_dim2)/2)
                idx_dim1 = np.arange(n_dim1) + round((mask_size[0] - n_dim1)/2)
                idx_dim2 = np.arange(n_dim2) + round((mask_size[1] - n_dim2)/2)
                mask_2D[idx_dim1, idx_dim2] = mask_2D_0[idx0_dim1, idx0_dim2]
            feature_masks[layer] = np.tile(
                mask_2D, [features[layer].shape[0], 1, 1])
        else:
            feature_masks[layer] = 0
    return feature_masks

def estimate_cnn_feat_std(cnn_feat):
    feat_ndim = cnn_feat.ndim
    feat_size = cnn_feat.shape
    # for the case of fc layers
    if feat_ndim == 1 or (feat_ndim == 2 and feat_size[0] == 1) or (feat_ndim == 3 and feat_size[1] == 1 and feat_size[2] == 1):
        cnn_feat_std = np.std(cnn_feat)
    # for the case of conv layers
    elif feat_ndim == 3 and (feat_size[1] > 1 or feat_size[2] > 1):
        num_of_ch = feat_size[0]
        # std for each channel
        cnn_feat_std = np.zeros(num_of_ch, dtype='float32')
        for j in range(num_of_ch):
            feat_ch = cnn_feat[j, :, :]
            cnn_feat_std[j] = np.std(feat_ch)
        cnn_feat_std = np.mean(cnn_feat_std)  # std averaged across channels
    return cnn_feat_std

def reconstruct_img(features, net,
                      layer_weight=None, channel=None, mask=None, initial_image=None,  maxiter=500, disp=True,save_intermediate_every=1, save_intermediate=False, loss_type='gram', 
                      save_intermediate_path=None,save_intermediate_ext='jpg',save_intermediate_postprocess=normalise_img):
    # loss function
    loss_fun = switch_loss_fun(loss_type)

    # make dir for saving intermediate
    if save_intermediate:
        if save_intermediate_path is None:
            save_intermediate_path = os.path.join('./recon_img_lbfgs_snapshots' + datetime.now().strftime('%Y%m%dT%H%M%S'))
        if not os.path.exists(save_intermediate_path):
            os.makedirs(save_intermediate_path)

    # get img size, #of pixel and mean of img
    img_size = net.blobs['data'].data.shape[-3:]
    num_of_pix = np.prod(img_size)
    img_mean = net.transformer.mean['data']


    # initial image
    if initial_image is None:
        initial_image = np.random.randint(0, 256, (img_size[1], img_size[2], img_size[0]))
    if save_intermediate:
        save_name = 'initial_img.png'
        PIL.Image.fromarray(np.uint8(initial_image)).save(os.path.join(save_intermediate_path, save_name))

    # preprocess initial img
    initial_image = img_caffe_preproc(initial_image, img_mean)
    initial_image = initial_image.flatten()

    # layer_list
    layer_list = list(features.keys())
    print("layer list : "+ str(layer_list))
    layer_list = sort_layer(net, layer_list)
    print("layer list sorted : "+ str(layer_list))

    # number of layers
    num_of_layer = len(layer_list)

    # layer weight
    if layer_weight is None:
        weights = np.ones(num_of_layer)
        weights = np.float32(weights)
        weights = weights / weights.sum()
        layer_weight = {}
        for i, lyr in enumerate(layer_list):
            layer_weight[lyr] = weights[i]

    # feature mask
    feature_masks = create_feature_masks(features, masks=mask, channels=channel)

    # optimization
    loss_list = []
    res = minimize(obj_fun, initial_image, args = (net, features, feature_masks, layer_weight, loss_fun, save_intermediate, save_intermediate_every, save_intermediate_path, save_intermediate_ext,
                                                   save_intermediate_postprocess, loss_list), 
                   method='L-BFGS-B', jac=True, options= {'maxiter': maxiter})

    # recon img
    img = res.x
    img = img.reshape(img_size)

    # return img
    return img_caffe_deproc(img, img_mean), loss_list


def obj_fun(img, net, features, feature_masks, layer_weight, loss_fun, save_intermediate, save_intermediate_every, save_intermediate_path, save_intermediate_ext, save_intermediate_postprocess, loss_list=[]):
    # reshape img
    img_size = net.blobs['data'].data.shape[-3:]
    img = img.reshape(img_size)

    # save intermediate image
    t = len(loss_list)
    if save_intermediate and (t % save_intermediate_every == 0):
        img_mean = net.transformer.mean['data']
        save_path = os.path.join(save_intermediate_path, '%05d.%s' % (t, save_intermediate_ext))
        if save_intermediate_postprocess is None:
            snapshot_img = img_caffe_deproc(img, img_mean)
        else:
            snapshot_img = save_intermediate_postprocess(img_caffe_deproc(img, img_mean))
        PIL.Image.fromarray(snapshot_img).save(save_path)

    # layer_list
    layer_list = features.keys()
    layer_list = sort_layer(net, layer_list)

    # num_of_layer
    num_of_layer = len(layer_list)

    # cnn forward
    net.blobs['data'].data[0] = img.copy()
    net.forward(end=layer_list[-1])

    # cnn backward
    loss = 0.
    layer_start = layer_list[-1]
    net.blobs[layer_start].diff.fill(0.)
    for j in range(num_of_layer):
        layer_start_index = num_of_layer - 1 - j
        layer_end_index = num_of_layer - 1 - j - 1
        layer_start = layer_list[layer_start_index]
        if layer_end_index >= 0:
            layer_end = layer_list[layer_end_index]
        else:
            layer_end = 'data'
        feat_j = net.blobs[layer_start].data[0].copy()
        feat0_j = features[layer_start]
        mask_j = feature_masks[layer_start]
        layer_weight_j = layer_weight[layer_start]
        loss_j, grad_j = loss_fun(feat_j, feat0_j, mask_j)
        loss_j = layer_weight_j * loss_j
        grad_j = layer_weight_j * grad_j
        loss = loss + loss_j
        g = net.blobs[layer_start].diff[0].copy()
        g = g + grad_j
        net.blobs[layer_start].diff[0] = g.copy()
        if layer_end == 'data':
            net.backward(start=layer_start)
        else:
            net.backward(start=layer_start, end=layer_end)
        net.blobs[layer_start].diff.fill(0.)
    grad = net.blobs['data'].diff[0].copy()


    grad = grad.flatten().astype(np.float64)
    loss_list.append(loss)

    return loss, grad

# set CAFFE GPU 
caffe.set_mode_gpu()
caffe.set_device(0)

# Decoded features settings
decoded_features_dir = './data/decodedfeatures'
decode_feature_filename = lambda net, layer, subject, roi, image_type, image_label: os.path.join(decoded_features_dir, image_type, net, layer, subject, roi,
                                                                                                 '%s-%s-%s-%s-%s-%s.mat' % (image_type, net, layer, subject, roi, image_label))

# Data settings
results_dir = './results'

subjects_list = ['S1', 'S2']

rois_list = ['VC']

network = 'VGG19'

# DNN layer combinations
layers_sets = {'layers-1to1' : ['conv1_1', 'conv1_2'],
               'layers-1to3' : ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                                'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4']}

# Images for natural image recontruction
'''image_type = 'natural'
image_label_list = ['Img0016',
                    'Img0036',
                    'Img0042']'''

#image for synthetic image reconstruction.
image_type = 'alphabet'
image_label_list = ['Img0005']

max_iteration = 200

# Average image of ImageNet and convert to img_mean
img_mean_file = '/content/DeepImageReconstruction/data/ilsvrc_2012_mean.npy'
img_mean = np.load(img_mean_file)
img_mean = np.float32([img_mean[0].mean(), img_mean[1].mean(), img_mean[2].mean()])

# load CNN
model_file = '/content/DeepImageReconstruction/net/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.caffemodel'
prototxt_file = '/content/DeepImageReconstruction/net/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.prototxt'
channel_swap = (2, 1, 0)
net = caffe.Classifier(prototxt_file, model_file, mean=img_mean, channel_swap=channel_swap)
h, w = net.blobs['data'].data.shape[-2:]
net.blobs['data'].reshape(1, 3, h, w)

# Initial image for the optimization 
initial_image = np.zeros((h, w, 3), dtype='float32')
initial_image[:, :, 0] = img_mean[2].copy()
initial_image[:, :, 1] = img_mean[1].copy()
initial_image[:, :, 2] = img_mean[0].copy()

# Feature SD
feat_std_file = '/content/DeepImageReconstruction/data/estimated_vgg19_cnn_feat_std.mat'
feat_std0 = sio.loadmat(feat_std_file)


opts = {
    # The loss function type: {'l2','inner','gram'}
    'loss_type': 'l2',

    # The maximum number of iterations
    'maxiter': max_iteration,

    # The initial image for the optimization (setting to None will use random noise as initial image)
    'initial_image': initial_image,

    # Display the information on the terminal or not
    'disp': True
}

# Save the optional parameters
with open(os.path.join(save_path, 'options.pkl'), 'wb') as f:
    pickle.dump(opts, f)

#create results dir to save images
save_path = os.path.join(results_dir)#, os.path.splitext(file)[0])
if not os.path.exists(save_path):
    os.makedirs(save_path)



for subject, roi, image_label, (layers_set, layers) in product(subjects_list, rois_list, image_label_list, layers_sets.items()):

    print('Subject Name = ' + subject)
    print('ROI = ' + roi)
    print('Img label = ' + image_label)

    save_dir = os.path.join(save_path, layers_set, subject, roi)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    features = {}
    for layer in layers:
        # The file full name depends on the data structure for decoded CNN features
        #set path to CNN file (code implemented on colab hence the path difference)
        colab_path = "DeepImageReconstruction"
        file_name = decode_feature_filename(network, layer, subject, roi, image_type, image_label)
        file_name = file_name.strip(".")
        file_name = os.path.join(colab_path, file_name)
        feat = sio.loadmat(file_name)['feat']
        if 'fc' in layer:
            feat = feat.reshape(feat.size)

        # Correct the norm of the decoded CNN features
        feat_std = estimate_cnn_feat_std(feat)
        feat = (feat / feat_std) * feat_std0[layer]

        features.update({layer: feat})

    # Norm of the CNN features for each layer
    feat_norm = np.array([np.linalg.norm(features[layer]) for layer in layers], dtype='float32')
    wght = 1. / (feat_norm ** 2)

    # Normalise weights
    wght = wght / wght.sum()
    layer_wght = dict(zip(layers, wght))

    opts.update({'layer_weight': layer_wght})

    # Reconstruction
    snapshots_dir = os.path.join(save_dir, 'snapshots', 'image-%s' % image_label)
    recon_img, loss_list = reconstruct_img(features, net,
                                             save_intermediate=True,
                                             save_intermediate_path=snapshots_dir,
                                             **opts)

    # Save the results
    save_name = 'rec_img' + '-' + image_label + '.mat'
    sio.savemat(os.path.join(save_dir, save_name), {'rec_img': recon_img})
    save_name = 'img_norm' + '-' + image_label + '.jpg'
    PIL.Image.fromarray(normalise_img(clip_extreme_value(recon_img, pct=0.04))).save(os.path.join(save_dir, save_name))

print('[INFO] Regeneration Complete')