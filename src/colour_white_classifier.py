from ImageClassifier import ImageClassifier
from data_utils import prepare_data

if __name__ == '__main__':
    name = 'colour_white_classifier'
    classes = ['colour', 'white']
    source_path = r'data2/dataset of groups/group 1/img'
    train_data_path = 'data/colour_white_classifier/train'
    val_data_path = 'data/colour_white_classifier/val'
    prepare_data(classes, source_path, train_data_path, val_data_path)
    classifier = ImageClassifier(
        name=name,
        classes=classes,
        model_path='{}_weights.h5'.format(name),
        train_data_path=train_data_path,
        val_data_path=val_data_path,
    )
    # classifier.train(epochs=5)
    print(classifier.predict('upload/2021032021555414.jpg'))
