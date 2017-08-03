import scipy as sp
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Input
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications.resnet50 import identity_block
from keras.applications.resnet50 import conv_block
from keras.optimizers import SGD, Adam, Adagrad
from keras.regularizers import l1, l2
from keras.callbacks import Callback, ModelCheckpoint

class LossHistory(Callback):
    def on_train_begin(self,logs={}):
        self.loss=[]
        self.val_loss=[]
    def on_epoch_end(self,epoch,logs={}):
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        
class PeasonR(Callback):
    def __init__(self):
        return
    def on_train_begin(self, logs = {}):
        #self.corr = []
        #self.val_corr = []
        return
    def on_epoch_begin(self, epoch, logs = {}):
        logs = logs or {}
        if "val_corr" not in self.params['metrics']:
            self.params['metrics'].append("val_corr")
    def on_epoch_end(self, epoch, logs = {}):
        logs = logs or {}
        y_pred = self.model.predict(self.validation_data[0])
        logs['val_corr'] = sp.stats.pearsonr(self.validation_data[1], y_pred)[0].item()

def resnet_model_build(resnet_model, use_stage, 
                       freeze_stage, 
                       use_merge = False, 
                       n_meta = 0,
                       fc_drop_rate = 0.2):
    #if use merge should always check n_meta
    
    assert(freeze_stage <= use_stage)
    assert(freeze_stage <=5 or freeze_stage >=1)
    assert(use_stage <=5 or use_stage >=1)

    for layer in resnet_model.layers:
        layer.trainable = True

    stage_layer_index = [5,37,79,141,174] 
    layer_use_index = stage_layer_index[use_stage-1]
    layer_freeze_index = stage_layer_index[freeze_stage-1]+1
    for layer in resnet_model.layers[:layer_freeze_index]:
        layer.trainable = False
    x = resnet_model.layers[layer_use_index].output

    if use_stage != 5:    
        x = AveragePooling2D((6, 6), name='avg_pool')(x)
    if use_merge:
        meta_info = Input(shape = (n_meta, )) # n_meta: numbers of features from meta
        x = keras.layers.concatenate([x, meta_info])
    else:
        pass

    x = Flatten()(x)
    x = Dense(128, name = 'dense1')(x)
    x = BatchNormalization(axis = -1, name = 'dense1_bn')(x)
    x = Activation('relu', name = 'dense1_activation')(x)
    x = Dropout(fc_drop_rate, name = 'd1_drop')(x)
    
    x = Dense(32, name = 'dense2')(x)
    x = BatchNormalization(axis = -1, name = 'dense2_bn')(x)
    x = Activation('relu', name = 'dense2_activation')(x)
    x = Dropout(fc_drop_rate, name = 'd2_drop')(x)
    
    out = Dense(1, activation="sigmoid", name = "output")(x)
        
    model = Model(inputs = [resnet_model.input], outputs = [out])
    model.summary()
    count = 0
    for layers in model.layers:
        if layers.trainable:
            count += 1
    print(count)

    return model

def model_build(resnet_model, stage = 3, fc_drop_rate = 0.2):
    if stage >= 1:
        x = ZeroPadding2D((3, 3))(resnet_model.input)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=3, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    if stage >= 2:
        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    if stage >= 3:
        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    if stage >= 4:
        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    if stage >= 5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    ## flatten and output layers
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    
    x = Dense(128, name = 'dense1')(x)
    x = BatchNormalization(axis = -1, name = 'dense1_bn')(x)
    x = Activation('relu', name = 'dense1_activation')(x)
    x = Dropout(fc_drop_rate, name = 'd1_drop')(x)
    
    x = Dense(32, name = 'dense2')(x)
    x = BatchNormalization(axis = -1, name = 'dense2_bn')(x)
    x = Activation('relu', name = 'dense2_activation')(x)
    x = Dropout(fc_drop_rate, name = 'd2_drop')(x)
    
    out = Dense(1, activation="sigmoid", name = "output")(x)
    model = Model(inputs=[feature_in], outputs=[out])
    for layer in resnet_model.layers:
        layer.trainable = False
    model.summary()
    model = Model(inputs=[resnet_model.input], outputs=[out])

    return model
