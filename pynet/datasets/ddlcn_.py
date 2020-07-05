import glob
import os
from IPython import embed
import cv2
import numpy as np
import shutil
from spams import trainDL, trainDL_Memory
from spams import lassoWeighted
import skimage.data as skid
import math


def dictionary_learning(descriptors, K=4, lambda1=0.35, numThreads=4,
                        nb_iter=500, mode=0, output_filename=None):
    """
    for jj = 1:c_num: # number of images for class 1
        num_angles = 8;
        num_bins = 4;
        num_samples = num_bins * num_bins;
        num_patches = numel(grid_x);
        sift_arr = zeros(num_patches, num_samples * num_angles);
        for i=1:num_patches:
            curr_sift = zeros(num_angles, num_samples);
            for a = 1:num_angles
                tmp = reshape(I_orientation(y_lo:y_hi,x_lo:x_hi,a),[num_pix 1]);
                tmp = repmat(tmp, [1 num_samples]);
                curr_sift(a,:) = sum(tmp .* weights);
            end
            sift_arr(i,:) = reshape(curr_sift, [1 num_samples * num_angles]);

        X = [sift_arr1, sift_arr2, ...];
        X.shape == (n_images, num_patches, num_samples * num_angles)
        X.shape = (n_images, num_patches, (4*4 = 16 sub-block neighbourhood) * 8 orientations)
        X.shape = (n_images, num_patches, 128 bins)
        Dic = mexTrainDL(X,param);

    X.shape =
    """

    # dic = trainDL(descriptors, lambda1=lambda1)
    descriptors = np.asfortranarray(descriptors, dtype=float)
    param = {'K': K,  # learns a dictionary with 100 elements
             'lambda1': lambda1, 'numThreads': numThreads, #'batchsize': 400,
             'iter': nb_iter}
    dic = trainDL(descriptors, **param)
    if output_filename is not None:
        fp = output_filename + ".npy"
        np.save(dic, fp)
    return dic


# def extract_sift(image, mask=None, draw_keypoints=False, outdir=None):
#     img = cv2.imread(image)
#     if mask is not None:
#         mask = cv2.imread(mask)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     sift = cv2.xfeatures2d.SIFT_create()
#     kp, des = sift.detectAndCompute(gray, mask)
#     if draw_keypoints:
#         assert outdir is not None, ("You must provide 'outdir' to save "
#                                     "keypoints image")
#         img = cv2.drawKeypoints(
#             gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#         split_path = os.path.basename(image).split(".")
#         basename = "".join(split_path[0:-1])
#         extension = split_path[-1]
#         fout = os.path.join(outdir, '{}_kp.{}'.format(basename, extension))
#         cv2.imwrite(fout, img)
#     return kp, des


def extract_dense_sift(image, step_size=5, patchSize=16,
                       draw_keypoints=False, outdir=None):

    img = cv2.imread(image)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)
    im_h, im_w = img.shape
    # # make grid sampling SIFT descriptors
    remX = im_w % step_size
    offsetX = math.floor(remX/2)
    remY = im_h % step_size
    offsetY = math.floor(remY/2)

    # img = skid.lena()

    sift = cv2.xfeatures2d.SIFT_create()

    kp = []
    for x in range(offsetX, im_w - offsetX, step_size):
        for y in range(offsetY, im_h - offsetY, step_size):
            kp.append(cv2.KeyPoint(x, y, step_size))

    if draw_keypoints:
        assert outdir is not None, ("You must provide 'outdir' to save "
                                    "keypoints image")
        img = cv2.drawKeypoints(
            img, kp, img)
        # img = cv2.drawKeypoints(
        #     gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        split_path = os.path.basename(image).split(".")
        basename = "".join(split_path[0:-1])
        extension = split_path[-1]
        fout = os.path.join(outdir, '{}_kp.{}'.format(basename, extension))
        cv2.imwrite(fout, img)

    kp, desc = sift.compute(img, kp)

    return kp, desc


def get_mnist_dataset():
    root = "/home/robin/myenv/git/perso_repo/mnist_png/mnist_png/training"
    files = glob.glob(os.path.join(root, "*", "*.png"))
    data = {}
    for fp in files:
        cat = fp.split(os.sep)[-2]
        data.setdefault(cat, []).append(fp)
    return data


def first_layer_coding(dic, descriptors, number_of_atom=16, lambda1=0.15,
                       mode=2, numThreads=8):

    L = number_of_atom-1

    for desc in descriptors:
        W = 0.01 * np.ones(dic.shape[1], desc.shape[1])
        for index in range(desc.shape[1]):
            temple1 = np.tile(desc[:, index], (1, dic.shape[1]))
            e = (temple1 - dic)**2

            norm_compare = np.zeros(1, dic.shape[1])
            for jj in range(dic.shape[1]):
                norm_compare[1, jj] = np.linalg.norm(e[:, jj], ord=2)
                sort_temple1 = np.sort(norm_compare)
                sort_temple2 = np.where(
                    norm_compare < sort_temple1[number_of_atom])
                norm_compare = 10./norm_compare
                temple3 = 0.01*np.ones(1, norm_compare.shape[1])
                temple3[1, sort_temple2] = norm_compare[1, sort_temple2]

                W[:, index] = temple3.H  # .T instead?
        codes = mexLassoWeighted(
            feaSet.feaArr, dic_groupsize20_rand25_mex, W, param)
        codes = lassoWeighted(desc, dic, W, number_of_atom=number_of_atom,
                              lambda1=lambda1, mode=mode,
                              numThreads=numThreads)


if __name__ == "__main__":
    outdir = "/home/robin/myenv/git/perso_repo/DDLCN_output"
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    mnist_dataset = get_mnist_dataset()
    cnt = 0
    for cat, images in mnist_dataset.items():
        # if cnt >= 1:
        #     continue
        descriptors = []
        for image in images[:20]:
            _, desc = extract_dense_sift(
                image, draw_keypoints=True, outdir=".")
            print(desc.shape)
            descriptors.append(desc)
        descriptors = np.stack(descriptors)
        print(descriptors.shape)
        descriptors = np.random.rand(21,36,128)
        dic = dictionary_learning(descriptors)
        print(stop)
        cnt += 1
        embed()
