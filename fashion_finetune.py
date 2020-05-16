import keras
import matplotlib.pyplot as plt
from keras.models import model_from_json

from keras import backend as K
K.set_learning_phase(0)

#fashion_dir = '/Users/mac/Desktop/python/assan_project/fashion_db_sample_split'
fashion_dir = '/Volumes/Samsung_T5/fashion/fashion_db'
#path_weights = '/Volumes/Samsung_T5/deep_learning_models/mobilenet-v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5'
#path_weights = '/Volumes/Samsung_T5/deep_learning_models/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
path_output = "/Volumes/Samsung_T5/fashion/classifier_models/model_2"

IMAGE_SIZE = 224
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
BATCH_SIZE = 64
N_EPOCHS = 10
INIT_LR = 5e-3
SEED = 0

"""
datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function = keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.1,
    width_shift_range=0.05, 
    height_shift_range=0.05,
    shear_range=0.05, 
    horizontal_flip=True, 
    fill_mode='nearest')
"""
datagen = keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function = keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.1)

train_generator = datagen.flow_from_directory(
    fashion_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='training',
    shuffle=True,seed=SEED)

val_generator = datagen.flow_from_directory(
    fashion_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='validation',
    shuffle=True,seed=SEED)

print(train_generator.class_indices)


base_model = keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                            include_top=True, 
                                              weights=path_weights)
 
base_model.layers.pop()
out = keras.layers.core.Dense(units=11, activation='sigmoid')(base_model.layers[-1].output)

model = keras.models.Model(base_model.inputs, out)


"""
startingModelPath = "/Volumes/Samsung_T5/fashion/classifier_models/model_3"

json_file = open(startingModelPath+'/mobilenet-v2_fashion_clf_3.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(startingModelPath+'/mobilenet-v2_fashion_clf_3_weights.h5', by_name=True)
"""

"""
base_model = keras.applications.vgg16.VGG16(input_shape=IMG_SHAPE,
                                            include_top=True, 
                                              weights=path_weights)

base_model.layers.pop()
out = keras.layers.core.Dense(units=11, activation='softmax')(base_model.layers[-1].output)

model = keras.models.Model(base_model.inputs, out)
"""

"""
modelFolderPath="/Volumes/Samsung_T5/fashion/classifier_models/model_2"
json_file = open(modelFolderPath+'/mobilenetv2_fashion_clf_22.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = keras.models.model_from_json(loaded_model_json)
model.load_weights(modelFolderPath+'/mobilenetv2_fashion_clf_22_weights.h5', by_name=True)
"""

# Select frozen layers
for layer in model.layers[:-2]:
    layer.trainable = False

# Select trainable layers
for layer in model.layers[-2:]:
    layer.trainable = True

# Check the trainable status of the individual layers
for layer in model.layers:
    print(layer, layer.trainable)

#class_weight_ = {0: 22., 1: 50., 2: 1., 3: 35., 4: 2., 5: 32., 6: 8., 7: 7., 8: 21., 9: 4., 10: 6.}
#class_weight_ = {0: 2., 1: 1., 2: 1., 3: 1., 4: 1., 5: 1., 6: 1., 7: 2., 8: 1., 9: 1., 10: 2.}


print('\n> compiling model...\n')

# prepare model for fitting (loss, optimizer, etc)
model.compile(loss='binary_crossentropy',
    optimizer=keras.optimizers.Adam(INIT_LR),
    metrics=['accuracy']
)


print("\nNumber of layers: {}\n".format(len(model.layers)))

model.summary()

print('\n> training model...\n')

early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.002, verbose=1, restore_best_weights=True)

history = model.fit_generator(train_generator, 
                    epochs=N_EPOCHS, 
                    validation_data=val_generator,
                    callbacks=[early_stop])
#                    class_weight=class_weight_)

# save model
model_json = model.to_json()
with open(path_output+"/mobilenetv2_fashion_clf_22.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(path_output+"/mobilenetv2_fashion_clf_22_weights.h5")

# plot learning curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

