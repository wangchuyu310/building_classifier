import os
import shutil
from random import uniform
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from matplotlib import pyplot as plt
from keras.backend import function


def prepare_data(classes, source_path, train_data_path, val_data_path):  # train and validation folder
    if os.path.exists(train_data_path):
        shutil.rmtree(train_data_path)
    if os.path.exists(val_data_path):
        shutil.rmtree(val_data_path)
    for _class in classes:
        os.makedirs(os.path.join(train_data_path, _class))
    for _class in classes:
        os.makedirs(os.path.join(val_data_path, _class))
    print('random split')
    for _class in classes:
        source_class_path = os.path.join(source_path, _class)
        files = list(os.walk(source_class_path))[0][2]
        for file in files:
            src = os.path.join(source_class_path, file)
            if uniform(0, 1) <= 0.8:
                dst = os.path.join(train_data_path, _class, file)
            else:
                dst = os.path.join(val_data_path, _class, file)
            shutil.copy(src, dst)


def make_data_generator(classes, dataset, batch_size, horizontal_flip=False):
    generator = ImageDataGenerator(rescale=1 / 255, horizontal_flip=horizontal_flip).flow_from_directory(
        dataset,
        target_size=(256, 256),
        batch_size=batch_size,
        classes=classes,
        class_mode='binary'
    )
    return generator


def visualize_3d(model, classes, num_of_point):
    from mpl_toolkits.mplot3d.axes3d import Axes3D

    reduce_dim = function([model.layers[0].input],
                          [model.layers[-3].output])  # refer to the first layer and the last 3 layers
    gen_val = make_data_generator(classes, 'val', num_of_point)
    x, y = gen_val.next()
    x_3d = reduce_dim([x])[0]  # (number fo points, 3)
    print(x_3d.shape)
    x1_east, x2_east, x3_east = [], [], []
    x1_west, x2_west, x3_west = [], [], []
    for score, (x1, x2, x3) in zip(y, x_3d):
        if score < 0.5:
            x1_east.append(x1)
            x2_east.append(x2)
            x3_east.append(x3)
        else:
            x1_west.append(x1)
            x2_west.append(x2)
            x3_west.append(x3)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x1_east, x2_east, x3_east, c='r', marker='^')  # '^'
    ax.scatter(x1_west, x2_west, x3_west, c='b', marker='o')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_aspect(1)
    plt.show()


def visual_layer(model, classes):
    # visualize convoluted layers
    # ratio keeps the same, if you want to use GPU, PLEASE INSTALL THE TENSORLFOW GPU AND REMOVE THE OLD ONE AND SET THE CONFIGURATION
    gen_val = make_data_generator('val', 1)
    # for i in range(100):
    #     x, y = gen_val.next()
    #     label = y[0]
    #     score = model.predict(x)[0][0]
    #     img = (x[0]*255).astype(np.uint8)
    #     plt.imshow(img)
    #     plt.title('label:%.2f, score:%.2f' % (label, score))
    #     plt.show()

    # visualize_3d(model, 218) #number should less than size of validation set, if greater, it will be OK but overlapping happen

    # visualize convoluted layers
    conv16 = function([model.layers[0].input], [model.layers[0].output])
    conv32 = function([model.layers[0].input], [model.layers[2].output])
    conv64 = function([model.layers[0].input], [model.layers[4].output])
    for i in range(100):
        x, y = gen_val.next()
        img = (x * 255).astype(np.uint8)
        c16 = conv16([x])[0]
        c32 = conv32([x])[0]
        c64 = conv64([x])[0]
        plt.imshow(img[0])
        plt.show()
        for j in range(16):
            ax = plt.subplot(4, 4, j + 1)
            ax.imshow(c16[0, :, :, j])
        plt.show()
        for j in range(32):
            ax = plt.subplot(4, 8, j + 1)
            ax.imshow(c32[0, :, :, j])
        plt.show()
        for j in range(64):
            ax = plt.subplot(8, 8, j + 1)
            ax.imshow(c64[0, :, :, j])
        plt.show()
