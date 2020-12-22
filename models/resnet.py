import os
import keras
import numpy as np
from keras import layers
from sklearn.model_selection import train_test_split

import keras.backend as K

from keras.initializers import glorot_uniform
from keras.models import Sequential, Model, load_model
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Input,Add,AveragePooling2D,ZeroPadding2D

'''내용 참고
https://github.com/priya-dwivedi/Deep-Learning/blob/master/resnet_keras/Residual_Networks_yourself.ipynb
'''


def identity_block(input_tensor, ks, filters, stage,block):
  '''conv layer가 없는 shortcut의 블록.
  input size와 output size가 같을때 사용가능.
  그래서 shortcut에 conv layer가 안달림
  '''
  filters1, filters2, filters3 = filters

  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x_shortcut = input_tensor

  #첫번째 component
  x = Conv2D(filters1, kernel_size=(3,3), strides=(1,1), kernel_initializer='he_normal',
                   name = conv_name_base + '2a', padding='same')(input_tensor)
  x = BatchNormalization(axis = 3, name = bn_name_base + '2a')(x)
  x = Activation('relu')(x)

  #두번째 component
  x = Conv2D(filters2, kernel_size=(ks,ks), strides=(1,1), kernel_initializer='he_normal',
             name=conv_name_base + '2b',padding='same')(x)
  x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)

  #마지막으로 shortcut을 더하고 relu를 통과시킨다
  x = Add()([x,x_shortcut])
  x = Activation('relu')(x)
  return x

def convolutional_block(x, ks, filters, stage, block, s = 2):
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block +'_branch'

  f1,f2,f3 = filters

  x_shortcut = x

  #첫번째 component

  x = Conv2D(f1, (3,3), strides=(s,s), name=conv_name_base + '2a',
             kernel_initializer='he_normal', padding='same')(x)
  x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
  x = Activation('relu')(x)

  #두번째 component
  x = Conv2D(f2, kernel_size=(ks,ks), strides=(1,1), kernel_initializer='he_normal',
             name=conv_name_base + '2b', padding='same')(x)
  x = BatchNormalization(axis = 3, name = bn_name_base + '2b')(x)

  #shortcut path
  x_shortcut = Conv2D(f3, kernel_size=(1,1), strides=(s,s), name = conv_name_base + '1',
             kernel_initializer='he_normal', padding='same')(x_shortcut)
  x_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(x_shortcut)
  x_shortcut = Activation('relu')(x_shortcut)

  #마지막으로 shortcut을 더하고 relu
  x = Add()([x,x_shortcut])
  x = Activation('relu')(x)
  return x

def ResNet26(input_shape, classes):
  x_input = Input(input_shape)

  x = ZeroPadding2D((3,3))(x_input)

  #첫번째
  x = Conv2D(45, (5,5), strides=(2,2), name='conv1',
             kernel_initializer=glorot_uniform(seed=0))(x)
  x = BatchNormalization(axis = 3, name = 'bn_conv1')(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((3,3),strides=(2,2))(x)

  #두번째
  x = convolutional_block(x, ks=3, filters=[45,45,45], stage=2, block='a',s=1)
  x = identity_block(x,3,[45,45,45],stage=2,block='b')

  #세번째
  x  = convolutional_block(x, ks=3, filters=[45,45,45], stage=3, block='a', s=2)
  x = identity_block(x,3,[45,45,45],stage=3,block='b')
  x = identity_block(x,3,[45,45,45],stage=3,block='c')
  x = identity_block(x,3,[45,45,45],stage=3,block='d')

  #네번째
  x  = convolutional_block(x, ks=3, filters=[45,45,45], stage=4, block='a', s=2)
  x = identity_block(x,3,[45,45,45],stage=4,block='b')

  #평균으로 dimension 줄이기
  x = AveragePooling2D((2,2), name="avg_pool",padding='SAME')(x)

  #output layer
  x = Flatten()(x)
  x = Dense(classes, activation='softmax', name='fc' + str(classes),
            kernel_initializer=glorot_uniform(seed=0))(x)

  model = Model(inputs=x_input,outputs=x, name='ResNet26')

  return model

def ResNet15(input_shape, classes):

  x_input = Input(input_shape)

  x = Conv2D(45, (3,3), strides=(1,1), name='conv1',
             kernel_initializer=glorot_uniform(seed=0))(x_input)
  x = BatchNormalization(axis = 3, name = 'bn_conv1')(x)
  x = Activation('relu')(x)
  x = MaxPooling2D((4,3), strides=(1,1))(x)

  x = convolutional_block(x, ks=3, filters=[45,45,45], stage=2, block='a', s=2)
  x = identity_block(x, 3, [45,45,45], stage=2, block='b')

  x = convolutional_block(x, ks=3, filters=[45,45,45], stage=3, block='a', s=2)
  x = identity_block(x, 3, [45,45,45], stage=3, block='b')

  x = convolutional_block(x, ks=3, filters=[45,45,45], stage=4, block='a', s=2)
  x = identity_block(x, 3, [45,45,45], stage=4, block='b')

  x = AveragePooling2D((2,2), name="avg_pool",padding='SAME')(x)

  #output layer
  x = Flatten()(x)
  x = Dense(classes, activation='softmax', name='fc' + str(classes),
            kernel_initializer=glorot_uniform(seed=0))(x)

  model = Model(inputs=x_input,outputs=x, name='ResNet15')

  return model

  
class ConvModel:
  def build(width, height, depth, classes, finalAct='softmax'):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(32, (3,3), padding='same', input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64,(3,3),padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation(finalAct))

        return model
