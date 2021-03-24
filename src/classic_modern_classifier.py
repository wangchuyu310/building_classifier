import os

from ImageClassifier import ImageClassifier
from constants import PROJECT_PATH


class ClassicModernClassifier(ImageClassifier):
    def __init__(self, name='classic_modern_classifier',
                 classes=None,
                 model_path=None,
                 train_data_path='data/classic_modern_classifier/train',
                 val_data_path='data/classic_modern_classifier/val',
                 img_shape=None, inited_model=False):
        if not classes:
            classes = ['classic', 'modern']
        if not model_path:
            model_path = os.path.join(PROJECT_PATH, '{}_weights.h5'.format(name))
        super(ClassicModernClassifier, self).__init__(
            name=name, classes=classes, model_path=model_path,
            train_data_path=train_data_path, val_data_path=val_data_path,
            img_shape=img_shape)


cm_classifier = ClassicModernClassifier()

if __name__ == '__main__':
    print(cm_classifier.predict('upload/2021032021555414.jpg'))
    print(cm_classifier.predict('upload/2021032021555414.jpg'))
    print(cm_classifier.predict('upload/2021032021555414.jpg'))
