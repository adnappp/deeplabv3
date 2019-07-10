from deeplab_v3 import Deeplab_v3
from data_utils import DataSet

import cv2
import os
import tensorflow as tf
import pandas as pd
import numpy as np
from color_utils import color_predicts
from predicts_utils import total_image_predict
from discriminative_loss import discriminative_loss
from metric_utils import iou


class args:
    batch_size = 4
    lr = 2e-4
    test_display = 2000
    weight_decay = 5e-4
    model_name = 'discriminativte_loss'
    batch_norm_decay = 0.95
    test_image_path = 'dataset/val/images/00001.png'
    test_label_path = 'dataset/val/labels/00001.png'
    multi_scale = True  # 是否多尺度预测
    gpu_num = 0
    pretraining = True


# 打印以下超参数
for key in args.__dict__:
    if key.find('__') == -1:
        offset = 20 - key.__len__()
        print(key + ' ' * offset, args.__dict__[key])

# 使用那一块显卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_path_df = pd.read_csv('dataset/path_list.csv')
data_path_df = data_path_df.sample(frac=1)  # 第一次打乱

dataset = DataSet(image_path=data_path_df['image'].values, label_path=data_path_df['label'].values)

model = Deeplab_v3(batch_norm_decay=args.batch_norm_decay)

image = tf.placeholder(tf.float32, [None, 1024, 1024, 3], name='input_x')
label = tf.placeholder(tf.int32, [None, 1024, 1024])
lr = tf.placeholder(tf.float32, )

logits = model.forward_pass(image)
logits_prob = tf.nn.softmax(logits=logits, name='logits_prob')
predicts = tf.argmax(logits, axis=-1, name='predicts')
feature_dim = 4
param_var = 1.
param_dist = 1
param_reg = 0.001
delta_v = 0.5
delta_d = 1.5
starter_learning_rate = 1e-4
learning_rate_decay_rate = 0.96
learning_rate_decay_interval = 5000


global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               learning_rate_decay_interval, learning_rate_decay_rate, staircase=True)
disc_loss, l_var, l_dist, l_reg = discriminative_loss(logits, label, feature_dim, (1024,1024),
                                                    delta_v, delta_d, param_var, param_dist, param_reg)

variables_to_restore = tf.trainable_variables(scope='resnet_v2_50')
with tf.name_scope('Instance/Adam'):
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(disc_loss, var_list=tf.trainable_variables(),
                                                                            global_step=global_step)
adam_initializers = [var.initializer for var in tf.global_variables() if 'Adam' in var.name]
# finetune resnet_v2_50的参数(block1到block4)
restorer = tf.train.Saver(variables_to_restore)
# cross_entropy


saver = tf.train.Saver(tf.all_variables())

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# summary_op = tf.summary.merge_all()

with tf.Session(config=config) as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    graph = tf.get_default_graph()

    if args.pretraining:
        # finetune resnet_v2_50参数，需要下载权重文件
        restorer.restore(sess, 'ckpts/resnet_v2_50/resnet_v2_50.ckpt')

    log_path = 'logs/%s/' % args.model_name
    model_path = 'ckpts/%s/deeplabV3' % args.model_name

    if not os.path.exists(model_path): os.makedirs(model_path)
    if not os.path.exists('./logs'): os.makedirs('./logs')
    if not os.path.exists(log_path): os.makedirs(log_path)

    summary_writer = tf.summary.FileWriter('%s/' % log_path, sess.graph)

    learning_rate = args.lr
    saver = tf.train.Saver()
    for step in range(1, 70001):
        if step == 30000 or step == 50000:
            learning_rate = learning_rate / 10
        x_tr, y_tr = dataset.next_batch(args.batch_size)
        _, step_prediction, step_loss, step_l_var, step_l_dist, step_l_reg = sess.run([
            train_op,
            logits,
            disc_loss,
            l_var,
            l_dist,
            l_reg],
            feed_dict={
                image: x_tr,
                label: y_tr,
                model._is_training: True,
                lr: learning_rate})
        if step * args.batch_size == 5000:
            train_text = 'step: {}, step_loss: {},step_l_var: {},step_l_dist: {},step_l_reg: {}'.format(
                step, step_loss,step_l_var,step_l_dist,step_l_reg)
            print(train_text)
        if step == 30000:
            saver.save(sess, model_path, write_meta_graph=True, global_step=step)
        # 前50, 100, 200 看一下是否搞错了
        if (step in [50, 500]) or (step > 0 and step % args.test_display == 0):

            test_predict = total_image_predict(
                ori_image_path=args.test_image_path,
                input_placeholder=image,
                logits_prob_node=logits_prob,
                is_training_placeholder=model._is_training,
                sess=sess,
                multi_scale=args.multi_scale

            )

            test_label = cv2.imread(args.test_label_path, cv2.IMREAD_GRAYSCALE)

            # 保存图片
            cv2.imwrite(filename='%spredict_color_%d.png' % (log_path, step),
                        img=color_predicts(img=test_predict))

            result = iou(y_pre=np.reshape(test_predict, -1),
                         y_true=np.reshape(test_label, -1))

            print("======================%d======================" % step)
            for key in result.keys():
                offset = 40 - key.__len__()
                print(key + ' ' * offset + '%.4f' % result[key])

            test_summary = tf.Summary(
                value=[tf.Summary.Value(tag=key, simple_value=result[key]) for key in result.keys()]
            )

            # 记录summary
            summary_writer.add_summary(test_summary, step)
            summary_writer.flush()
    saver.save(sess, model_path, write_meta_graph=True, global_step=70000)