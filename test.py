# coding:utf-8
import cv2
import numpy as np
import tensorflow as tf
from deeplab_v3 import Deeplab_v3
from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = 100000000000
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def predict(image: np.ndarray, input_placeholder: tf.placeholder,
    is_training_placeholder: tf.placeholder, logits_prob_node: tf.Tensor,
    sess: tf.Session, prob: bool) -> np.ndarray:
    """
    使用模型预测一张图片，并输出其预测概率分布
    Args:
        image: np.ndarray [size, size, 3] 需要预测的图片数组
        input_placeholder: tf.placeholder
        is_training_placeholder: tf.placeholder
        logits_prob_node: tf.Tensor [size, size, num_classes]
        sess: tf.Session
        prob: bool 输出的是概率，还是预测值
    Returns:
        image_predict: np.ndarray [size, size, 5] if porb is True
                       np.ndarray [size, size] if prob is not True
    """
    assert image.shape == (1024, 1024, 3), print(image.shape)
    # 给image升维 [1024, 1024, 3] -> [1, 1024, 1024, 3]
    feed_dict = {input_placeholder: np.expand_dims(image, 0),
                 is_training_placeholder: False}
    image_predict_prob = sess.run(logits_prob_node, feed_dict=feed_dict)
    # 给image降维 [1, 1024, 1024, 5] -> [1024, 1024, 5]
    image_predict_prob = np.squeeze(image_predict_prob, 0)
    if prob:
        # 输出预测概率分布
        return image_predict_prob
    else:
        # 输出预测值
        image_predict = np.argmax(image_predict_prob, -1)
        return image_predict

def rotate(x, angle, size=1024):
    """ 旋转函数
    """
    M_rotate = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1)
    x = cv2.warpAffine(x, M_rotate, (size, size))
    return x

def multi_scale_predict(image: np.ndarray, input_placeholder: tf.placeholder,
    is_training_placeholder: tf.placeholder, logits_prob_node: tf.Tensor,
    sess: tf.Session, multi: bool):
    """

    Args:
        image:
        input_placeholder:
        is_training_placeholder:
        logits_prob_node:
        sess:
        multi:

    Returns:
        np.ndarray [size, size]
    """

    # 旋转函数
    kwargs = {
        'input_placeholder':input_placeholder,
        'is_training_placeholder':is_training_placeholder,
        'logits_prob_node':logits_prob_node,
        'sess':sess,
        'prob':True,
    }
    if multi:
        image_predict_prob_list = [
            predict(image=image, **kwargs)
        ]
        # 旋转三个
        angle_list = [90, 180, 270]
        for angle in angle_list:
            image_rotate = rotate(image, angle)

            image_rotate_predict_prob = predict(image=image_rotate, **kwargs)
            image_predict_prob = rotate(image_rotate_predict_prob, -1 * angle)
            image_predict_prob_list.append(image_predict_prob)
        # 翻转两个
        flip_list = [1, 0]
        for mode in flip_list:
            image_flip = cv2.flip(image, mode)

            image_flip_predict_prob = predict(image=image_flip, **kwargs)
            image_predict_prob = cv2.flip(image_flip_predict_prob, mode)
            image_predict_prob_list.append(image_predict_prob)
        # 求和平均
        final_predict_prob = sum(image_predict_prob_list) / len(image_predict_prob_list)
        return np.argmax(final_predict_prob, -1)
    else:
        kwargs['prob'] = False
        return predict(image, **kwargs)


def total_image_predict(ori_image_path: str,
                        input_placeholder: tf.placeholder,
                        is_training_placeholder: tf.placeholder,
                        logits_prob_node: tf.Tensor,
                        sess: tf.Session,
                        multi_scale = False
                        ) -> np.ndarray:

    #ori_image = cv2.imread(ori_image_path, cv2.CAP_MODE_RGB)
    ori_image = Image.open(ori_image_path)  # 注意修改img路径
    ori_image = np.asarray(ori_image)
    # 开始切图 cut
    h_step = ori_image.shape[0] // 1024
    w_step = ori_image.shape[1] // 1024

    h_rest = -(ori_image.shape[0] - 1024 * h_step)
    w_rest = -(ori_image.shape[1] - 1024 * w_step)

    image_list = []
    predict_list = []
    # 循环切图
    for h in range(h_step):
        for w in range(w_step):
            # 划窗采样
            image_sample = ori_image[(h * 1024):(h * 1024 + 1024),
                           (w * 1024):(w * 1024 + 1024), 0:3]
            image_list.append(image_sample)
        image_list.append(ori_image[(h * 1024):(h * 1024 + 1024), -1024:, 0:3])
    for w in range(w_step - 1):
        image_list.append(ori_image[-1024:, (w * 1024):(w * 1024 + 1024), 0:3])
    image_list.append(ori_image[-1024:, -1024:, 0:3])

    # 对每个图像块预测
    # predict
    for image in image_list:

        predict = multi_scale_predict(
            image=image,
            input_placeholder=input_placeholder,
            is_training_placeholder=is_training_placeholder,
            logits_prob_node=logits_prob_node,
            sess=sess,
            multi=multi_scale
        )
        # 保存覆盖小图片
        predict_list.append(predict)

    # 将预测后的图像块再拼接起来
    count_temp = 0
    tmp = np.ones([ori_image.shape[0], ori_image.shape[1]])
    for h in range(h_step):
        for w in range(w_step):
            tmp[
            h * 1024:(h + 1) * 1024,
            w * 1024:(w + 1) * 1024
            ] = predict_list[count_temp]
            count_temp += 1
        tmp[h * 1024:(h + 1) * 1024, w_rest:] = predict_list[count_temp][:, w_rest:]
        count_temp += 1
    for w in range(w_step - 1):
        tmp[h_rest:, (w * 1024):(w * 1024 + 1024)] = predict_list[count_temp][h_rest:, :]
        count_temp += 1
    tmp[h_rest:, w_rest:] = predict_list[count_temp][h_rest:, w_rest:]
    return tmp


def main():
    # 加载模型
    saver = tf.train.import_meta_graph('./ckpts/baseline1/deeplabV3-75000.meta')
    img_paths = ["./dataset/test/images/image_3.png", "./dataset/test/images/image_4.png"]
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./ckpts/baseline1/'))
        graph = tf.get_default_graph()
        image = graph.get_tensor_by_name("input_x:0")
        _is_training = graph.get_tensor_by_name("is_training:0")
        #logits_prob = tf.nn.softmax(logits=logits, name='logits_prob')
        logits_prob = graph.get_tensor_by_name("logits_prob:0")
        i=3
        for img_path in img_paths:
            test_predict = total_image_predict(
                ori_image_path=img_path,
                input_placeholder=image,
                logits_prob_node=logits_prob,
                is_training_placeholder=_is_training,
                sess=sess,
                multi_scale=True
            )
            name = "result/baseline1/image_"+str(i)+"_predict.png"
            cv2.imwrite(name, test_predict)
            i+=1


main()