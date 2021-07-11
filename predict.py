import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json


BATCH_SIZE = 64
image_size  = 224

def process_image(test_image):
  processed_image = tf.convert_to_tensor(test_image, dtype=tf.float32)
  processed_image = tf.image.resize(processed_image, (image_size, image_size))
  processed_image = processed_image / 255
  processed_image_numpy = processed_image.numpy()
  return processed_image_numpy


def increment_label(label):
      return label+1


def decode_text_for_array(array_of_labels):
      for index, value in enumerate(array_of_labels):
        array_of_labels[index] = value.decode('utf-8')
      return array_of_labels


def process_label(indices):
      indices_incremented_by_one = tf.map_fn(increment_label, indices)
      indices_string = tf.strings.as_string(indices_incremented_by_one)
      indices_squeezed = tf.squeeze(indices_string)
      indices_array = indices_squeezed.numpy()
      indices_decoded = decode_text_for_array(indices_array)
      return indices_decoded


def predict(image_path, model, k):
    image = Image.open(image_path)
    image_numpy = np.asarray(image)
    expanded_image = np.expand_dims(process_image(image_numpy), axis=0)
    result = model.predict(expanded_image)
    top_k = tf.math.top_k(result, k=k, sorted=True)
    probs = tf.squeeze(top_k.values).numpy()
    classes = process_label(top_k.indices)
    return probs, classes


def load_class_names(category_names_path):
    with open(category_names_path, 'r') as f:
        return json.load(f)


def print_result(image_probs, image_classes):
    for i, result in enumerate(image_probs):
        print('\n')
        print('Label: ', image_classes[i])
        print('Confidance: {:.2%}'.format(result))
        print('\n')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Prediction of flower label')

    parser.add_argument('--input', action='store', dest='input', default='./test_images/cautleya_spicata.jpg')
    parser.add_argument('--model', action='store', dest='model', default='./1625998552.h5')
    parser.add_argument('--top_k', action='store', dest='top_k', default=5, type=int)
    parser.add_argument('--category_names', action='store', dest="category_names", default='./label_map.json')

    args = parser.parse_args()

    input_image_path = args.input
    model_path = args.model
    top_k = args.top_k
    category_names_path = args.category_names

    model = reloaded_keras_model = tf.keras.models.load_model('./1625998552.h5', custom_objects={'KerasLayer':hub.KerasLayer},compile=False)


    image_probs, image_classes = predict(input_image_path, model, top_k)

    class_names = load_class_names(category_names_path)

    processed_class_names = []
    for label in image_classes:
        processed_class_names.append(class_names[label])

    print_result(image_probs, processed_class_names)