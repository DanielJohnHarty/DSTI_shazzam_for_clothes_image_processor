import keras

class FashionEncoder:
	def __init__(self):
		self.model=None
	def preprocess_input(self):
		pass
	def summary():
		pass
	def get_input_dim(self):
		pass
	def get_output_dim(self):
		pass
	def encode(self):
		pass

class FashionClassifier:
	def __init__(self):
		self.model=None
	def preprocess_input(self):
		pass
	def summary():
		pass
	def predict(self):
		pass
	def predict_generator(self):
		pass

class Encoder_1(FashionEncoder):
	def __init__(self,modelFolderPath):
		embeddingModel.__init__(self)
		json_file = open(modelFolderPath+'/vgg16_fashion_em_1.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.model = keras.models.model_from_json(loaded_model_json)
		self.model.load_weights(modelFolderPath+'/vgg16_fashion_em_1_weights.h5', by_name=True)
		self.model.name = '\n\n************************ Fashion Embedding Model 1 ********************************\n\n> Model: VGG-16\n> Pretrained on: Imagenet\n> Fine tuning process: frozen bottom, training over 2 last layers\n> Performance: evaluated on a validation set of 3000 images\n              precision    recall  f1-score   support\n\n        bags       0.78      0.56      0.65        57\n       belts       0.88      0.65      0.75        23\n     dresses       0.90      0.94      0.92      1287\n     eyewear       0.92      0.94      0.93        35\n    footwear       0.91      0.96      0.94       648\n        hats       0.97      0.93      0.95        40\n    leggings       0.83      0.81      0.82       164\n   outerwear       0.66      0.62      0.64       194\n       pants       0.92      0.75      0.83        60\n      skirts       0.90      0.90      0.90       333\n        tops       0.65      0.53      0.58       217\n\n    accuracy                           0.87      3058\n   macro avg       0.85      0.78      0.81      3058\nweighted avg       0.87      0.87      0.87      3058\n\n********************************************************************************\n\n'

	def preprocess_input(self,x):
		return keras.applications.vgg16.preprocess_input(x)

	def summary(self):
		self.model.summary()

	def get_input_dim(self):
		return self.model.input_shape[1:]

	def get_output_dim(self):
		return self.model.output_shape[1]

	def encode(self,img_batch):
		return self.model.predict(img_batch)

class CLF_1(FashionClassifier):
	def __init__(self,modelFolderPath):
		FashionClassifier.__init__(self)
		json_file = open(modelFolderPath+'/vgg16_fashion_clf_1.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.model = keras.models.model_from_json(loaded_model_json)
		self.model.load_weights(modelFolderPath+'/vgg16_fashion_clf_1_weights.h5', by_name=True)
		self.model.name = '\n\n************************ Fashion Classifier 1 ********************************\n\n> Model: VGG-16\n> Pretrained on: Imagenet\n> Fine tuning process: frozen bottom, training over 2 last layers\n> Performance: evaluated on a validation set of 3000 images\n              precision    recall  f1-score   support\n\n        bags       0.78      0.56      0.65        57\n       belts       0.88      0.65      0.75        23\n     dresses       0.90      0.94      0.92      1287\n     eyewear       0.92      0.94      0.93        35\n    footwear       0.91      0.96      0.94       648\n        hats       0.97      0.93      0.95        40\n    leggings       0.83      0.81      0.82       164\n   outerwear       0.66      0.62      0.64       194\n       pants       0.92      0.75      0.83        60\n      skirts       0.90      0.90      0.90       333\n        tops       0.65      0.53      0.58       217\n\n    accuracy                           0.87      3058\n   macro avg       0.85      0.78      0.81      3058\nweighted avg       0.87      0.87      0.87      3058\n\n***************************************************************************\n\n'

	def preprocess_input(self,x):
		return keras.applications.vgg16.preprocess_input(x)

	def summary(self):
		self.model.summary()

	def predict(self,img_batch):
		return self.model.predict(img_batch)

	def predict_generator(self,**params):
		return self.model.predict_generator(**params)