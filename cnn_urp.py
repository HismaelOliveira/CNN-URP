import keras.layers
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from input import load_data

print('------- LOADING DATA... -------')
x, y, vocabulary, vocabularyInv = load_data()

print('------- SPLITING DATA... -------')
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

sequence_length = x.shape[1] # 56
vocabulary_size = len(vocabularyInv) # 18765
embedding_dim = 256
filter_sizes = [3,4,5]
num_filters = 256
drop = 0.5
epochs = 50
batch_size = 30

print('------- CREATING MODEL... -------')
inputs = keras.layers.Input(shape=(sequence_length,), dtype='int32')
embedding = keras.layers.Embedding(input_dim=vocabulary_size, output_dim= embedding_dim, input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = keras.layers.Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_1 = keras.layers.Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
conv_2 = keras.layers.Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

maxpool_0 = keras.layers.MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
maxpool_1 = keras.layers.MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
maxpool_2 = keras.layers.MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

concatenated_tensor = Concatenate(axis=1)([maxpool_0,maxpool_1,maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = keras.layers.Dense(units=2, activation='softmax')(dropout)

model = Model(inputs=inputs, outputs=output)
checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
print('------- TRAINING MODEL... -------')

model.fit(xTrain, yTrain, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(xTest, yTest))

score = model.evaluate(xTest, yTest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])