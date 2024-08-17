import pydensecrf.densecrf as dcrf
import numpy as np
from pydensecrf.utils import unary_from_labels


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def apply_dense_crf(img, mask):
    EPSILON = 1e-8
    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)
    anno_norm = mask / 255.0
    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype="float32")
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    # d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img, compat=10)

    # Do the inference
    infer = np.array(d.inference(4)).astype("float32")
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2]).astype("uint8")
    return res


def crf_inference_label(img, labels, t=10, n_labels=21, gt_prob=0.7):
    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(
        sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10
    )

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)
