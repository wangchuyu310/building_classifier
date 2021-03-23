import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense, Dropout, MaxPool2D
from scipy.misc import imread, imresize

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class ImageClassifier(object):
    def __init__(self, name=None, classes=None, model_path=None, train_data_path=None, val_data_path=None,
                 img_shape=None):
        self.name = name
        self.classes = classes
        self.model_path = model_path
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.img_shape = img_shape or (256, 256)
        self.model = None
        self.__inited_model = False

    def build_model(self):  # 3 convolutional layers
        s = Sequential()
        # block: conv -> relu -> max pool; BATCHSIZE=100 we have mentioned before, feature extraction into
        # (100,256,256,3)
        # reduce the kernel size, remove maxpooling layer since it reduce the size
        s.add(Conv2D(16, kernel_size=4, strides=4, padding='same', activation='relu', input_shape=(*self.img_shape, 3)))
        # 4px x 4px KERNEL FIRST
        # (100,64,64,16)
        s.add(MaxPool2D((2, 2)))
        # (100,32,32,16)
        s.add(Conv2D(32, kernel_size=2, strides=2, padding='same', activation='relu'))
        # (100,16,16,32)
        s.add(MaxPool2D((2, 2)))
        # 100,8,8,32
        s.add(Conv2D(64, kernel_size=2, strides=2, padding='same', activation='relu'))
        # 100,4,4,64
        s.add(MaxPool2D((2, 2)))
        # 100,2,2,64
        # whole size of block of CNN
        s.add(Flatten())
        # matrix is not suitable for CNN, so change the 3d matrix/array into flatten single one dimension model/array = 256
        s.add(Dropout(0.5))
        # 256
        s.add(Dense(3))
        # 256*3=768 connections, 3= 3 dimension model visualization that we want to exhibit, bigger number cannot be visualize
        s.add(BatchNormalization(axis=-1))
        s.add(Dense(1))  # we only need to know one of the classifications
        s.add(Activation('sigmoid'))
        # restrict the value of variable, and for now we can almost get one value from the classification
        s.compile(optimizer='adam', loss=['binary_crossentropy'], metrics=['acc'])
        # SGD (sophisticated gradient distance), adam, adamax
        # binary_crossentropy : y*log(y_pred)+(1-y)*log(1-y_pred)=0
        return s

    def make_data_generator(self, dataset, batch_size, horizontal_flip=False):
        generator = ImageDataGenerator(rescale=1 / 255, horizontal_flip=horizontal_flip).flow_from_directory(
            dataset,
            target_size=(256, 256),
            batch_size=batch_size,
            classes=self.classes,
            class_mode='binary'
        )
        return generator

    def init_model(self):
        if not self.model:
            self.model = self.build_model()
            if not os.path.exists(self.model_path):
                self.train()
            else:
                self.model.load_weights(self.model_path)

    def train(self, epochs=5, batch_size=100, train_steps=80, val_steps=20):
        self.model = self.build_model()
        self.model.summary()
        gen_train = self.make_data_generator(self.train_data_path, batch_size, True)
        gen_val = self.make_data_generator(self.val_data_path, batch_size)
        self.model.fit_generator(gen_train, steps_per_epoch=train_steps, epochs=epochs, validation_data=gen_val,
                                 validation_steps=val_steps)
        self.model.save_weights(self.model_path)
        self.__inited_model = True

    def predict(self, img_path):
        self.init_model()
        img = imread(img_path)
        img = imresize(img, self.img_shape)
        x = img.reshape((1, *self.img_shape, 3)) / 255
        y = self.model.predict(x)
        score = round(float(y[0][0]), 3)
        return (1 - score, self.classes[0]) if score <= 0.5 else (score, self.classes[1])
