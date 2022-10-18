# coding: utf-8
import os
import random
import paddle
os.environ['TL_BACKEND'] = 'paddle'
# import paddle.nn.functional as F
import tensorlayerx as tlx
import numpy as np
from utils.load_image import load_image

EPOCH_NUM = 10
BATCH_SIZE = 8  # 64
BATCH_NUM = 4  # 100
IMAGE_SHAPE = [3, 224, 224]
CLASS_NUM = 1000


# class RandomDataset(paddle.io.Dataset):
class RandomDataset(tlx.dataflow.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        images = np.random.random(size=IMAGE_SHAPE).astype('float32')
        labels = np.random.randint(0, CLASS_NUM - 1, (1,)).astype('int64')
        return images, labels

    def __len__(self):
        return self.num_samples


def CatDogGenerator(mode="train"):  # TODO - 原生实现, 不用变
    base_dir = f"D:/DATA/vision-data/dogs-vs-cats-cut/{mode}"
    images_paths = os.listdir(base_dir)
    label = -1  # not defined
    images = []
    labels = []
    if mode == "train":
        for i, image_path in enumerate(images_paths):
            label_name = image_path.split(".")[0]
            if label_name == "cat":
                label = 0
            elif label_name == "dog":
                label = 1
            image_path = os.path.join(base_dir, image_path)
            img = load_image(image_path)
            images.append(img)
            labels.append(label)
            assert len(images) == len(labels)
    else:
        pass
    index_list = list(range(len(images)))

    def data_generator():
        if mode == 'train':
            random.shuffle(index_list)
            images_list = []
            labels_list = []
            for idx in index_list:
                images_list.append(images[idx])
                label = np.reshape(labels[i], [1]).astype('int64')
                labels_list.append(label)
                if len(images_list) == BATCH_SIZE:
                    yield np.array(images_list), np.array(labels_list)
                    images_list, labels_list = [], []
            if len(images_list) > 0:
                yield np.array(images_list), np.array(labels_list)

    return data_generator


class ModelTrainTLX(object):
    def __init__(self, model):
        self.model = model

    def train(self):
        import paddle.nn as nn
        import paddle.optimizer as opt

        self.model.train()  # TODO
        loss_fn = nn.CrossEntropyLoss()
        adam = opt.Adam(learning_rate=0.001, parameters=model.parameters())

        # create data loader
        # dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
        # loader = paddle.io.DataLoader(dataset,
        #                               batch_size=BATCH_SIZE,
        #                               shuffle=True,
        #                               drop_last=True,
        #                               num_workers=0)  # 2
        # loader = tlx.dataflow.DataLoader(dataset,
        #                                  batch_size=BATCH_SIZE,
        #                                  shuffle=True,
        #                                  drop_last=True,
        #                                  num_workers=0)  # 2
        train_dataset = CatDogGenerator("train")

        for epoch_id in range(EPOCH_NUM):
            # for batch_id, (images, labels) in enumerate(loader()):
            # for batch_id, (images, labels) in enumerate(loader):  # TODO
            for batch_id, (images, labels) in enumerate(train_dataset()):
                # images = paddle.to_tensor(images)
                # labels = paddle.to_tensor(labels)
                images = tlx.ops.convert_to_tensor(images)
                labels = tlx.ops.convert_to_tensor(labels)
                preds = model(images)
                if isinstance(preds, paddle.Tensor):
                    loss = loss_fn(preds, labels)
                elif isinstance(preds, list):
                    loss = loss_fn(preds[0], labels)
                # avg_loss = paddle.mean(loss)
                avg_loss = loss.mean()  # TODO
                # acc = paddle.metric.accuracy(input=preds, label=labels)
                acc_metric = tlx.metrics.Accuracy()
                if isinstance(preds, paddle.Tensor):
                    acc_metric.update(y_pred=preds, y_true=labels)
                elif isinstance(preds, list):
                    acc_metric.update(y_pred=preds[0], y_true=labels)
                acc = acc_metric.result()
                print("Epoch {} batch {}: loss = {}, acc = {}".format(epoch_id + 1,
                                                                      batch_id + 1,
                                                                      avg_loss.numpy()[0],
                                                                      acc))
                loss.backward()
                adam.step()
                adam.clear_grad()


if __name__ == '__main__':
    from models.vision.tlx_vgg import vgg16
    from models.vision.tlx_alexnet import alexnet
    from models.vision.tlx_inceptionv3 import inception_v3
    from models.vision.tlx_inception_v4 import InceptionV4
    from models.vision.tlx_googlenet import googlenet
    from models.vision.tlx_squeezenet import squeezenet1_0
    from models.vision.tlx_mobilenetv1 import mobilenet_v1
    from models.vision.tlx_mobilenetv2 import mobilenet_v2
    from models.vision.tlx_shufflenetv2 import shufflenet_v2_x1_0
    from models.vision.tlx_resnet import resnet50
    from models.vision.tlx_densenet import densenet121
    from models.vision.tlx_rednet import RedNet50
    from models.vision.tlx_darknet53 import darknet53
    from models.vision.tlx_regnet import RegNetX_4GF

    # pass
    # model = vgg16(pretrained=False, num_classes=2)
    # model = vgg16(pretrained=True, num_classes=2)  # need to freeze network parameters and modify classifier output
    # model = alexnet(pretrained=False, num_classes=2)
    # model = inception_v3(pretrained=False, num_classes=2)
    # model = InceptionV4(pretrained=False, num_classes=2)
    # model = googlenet(pretrained=False, num_classes=2)
    # model = squeezenet1_0(pretrained=False, num_classes=2)
    # model = mobilenet_v1(pretrained=False, num_classes=2)
    # model = mobilenet_v2(pretrained=False, num_classes=2)
    # model = shufflenet_v2_x1_0(pretrained=False, num_classes=2)
    # model = resnet50(pretrained=False, num_classes=2)
    # model = densenet121(pretrained=False, num_classes=2)
    # model = RedNet50(pretrained=False, num_classes=2)
    # model = darknet53(pretrained=False, num_classes=2)
    model = RegNetX_4GF(pretrained=False, num_classes=2)

    Train = ModelTrainTLX(model)
    Train.train()
