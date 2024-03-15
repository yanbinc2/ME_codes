#=============================================================
# LeNet-5 in keras    https://github.com/marload/LeNet-keras
def ME_CNN(x_train, train_label, test_array, true_answer, Num_Classes):  
    import tensorflow as tf
    import numpy
    from keras.models import Sequential
    from keras.optimizers import SGD
    from keras.utils import to_categorical
    import numpy as np
    from keras.layers import Dense, Dropout, Activation, Flatten, Convolution1D, MaxPooling1D, AveragePooling1D
    
    if (test_array.shape[1]<=5):
        KK=1
    else:
        KK=5
    
    train_label = to_categorical(train_label, Num_Classes)
    true_answer = to_categorical(true_answer, Num_Classes)
    model = Sequential()
   
    # 1
    # Convolution layer 1
    #model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", data_format="channels_last"))
    model.add(Convolution1D(filters=6, kernel_size=KK, strides=1, activation='relu', padding="same", data_format="channels_last", input_shape=x_train[0].shape))    # ((I+2P-F)/S)+1  =  ((28+2*0-5)/1)+1=24
    ##convo1 = Activation('relu')  #add
    ##model.add(convo1)   #add
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5)) #0610

    
    # 2
    model.add(AveragePooling1D(pool_size=2,  strides=2, padding="valid"))   # ((24+2*0-2)/2)+1=12
    
    
    # 3
    # Convolution layer 3
    model.add(Convolution1D(filters=16, kernel_size=KK, strides=1, activation='relu', padding="valid", data_format="channels_last"))
    #model.add(Activation('relu'))
    #model.add(Dropout(0.5))   #0610

    #model.add(MaxPooling2D(pool_size=(2, 2)))   $0610

    # 4
    model.add(AveragePooling1D(pool_size=1,  strides=2, padding="valid"))
    
    # Convolution layer 3
    #model.add(Convolution2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", data_format="channels_last"))  #0610
    #model.add(Activation('relu'))  #0610
    #model.add(Dropout(0.5))   #0610

    # Convolution layer 4
    #model.add(Convolution2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", data_format="channels_last"))  #0610
    #model.add(Activation('relu'))  #0610
    #model.add(Dropout(0.5))   #0610


    
    # 5
    model.add(Convolution1D(filters=120, kernel_size=KK, strides=1, activation='relu', padding="valid", data_format="channels_last"))
#    model.add(Activation('relu'))
    # connetct to DNN and use soft max to make prediction
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid")) #0906
    
    model.add(Flatten())
    
    # 6
    # Dense Layer
    #model.add(Dense(120, activation='relu'))  #0610
    model.add(Dense(84,  activation='relu'))
#    model.add(Activation('relu')) 
    #model.add(Dropout(0.5))  #remove temporary   #0610

    
    # 7
    # Dense Layer
    #model.add(Dense(10))
    model.add(Dense(Num_Classes, name='preds', activation='softmax'))  #here 2 classsifications, add  name='preds'
    #model.add(Activation('softmax'))
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    model.summary()
    
    #===========================
    from keras.optimizers import Adam
    
    # Unnecessary. Input on compile function directly
    #learning_rate = 0.001  # ori=0.0001   0.00017
    #optimizer = Adam(lr=learning_rate)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr= 0.001), #lr is learning_rate
                  metrics=['accuracy'])
   

    #=== shuffle manually ========
    idx = np.arange(x_train.shape[0]) #x_train.shape[0] = 1000 (total train size)
    numpy.random.shuffle(idx)
    x_train = x_train[idx]
    train_label = train_label[idx]
    #===========================
    
    ## 1112 TensorBoard
    #import datetime
    #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    ###tf.Graph().get_operations().values().name
    ### https://stackoverflow.com/questions/35336648/list-of-tensor-names-in-graph-in-tensorflow
    #tensor_names = [t.name for op in tf.Graph().get_operations() for t in op.values()]
    #print("out:",tensor_names)
    #tensor_values = [t      for op in tf.Graph().get_operations() for t in op.values()]
    #print("out:",tensor_values)
    #node=[n.name for n in tf.Graph().as_graph_def().node]
    #print("out:",node)
    #tensor_op = [op for op in tf.Graph().get_operations()]
    #print("out:",tensor_op)
    
    #sess = tf.compat.v1.Session()
    #op = sess.graph.get_operations()
    #out=[m.values() for m in op]
    #print("out:",out)
    
    ## 1112 tensorboard
    #img = np.reshape(x_train[0:4], (-1, 28, 28, 1)) 
    #writer = tf.summary.create_file_writer(log_dir)
    #with writer.as_default():
    #    tf.summary.image("Training data", img, step=2)
    
    
    ###https://ai-pool.com/d/how-to-visualize-feature-map-in-tensorboard-
    #layer = tf.Graph().get_tensor_by_name('tensor_layer:0')
    #writer = tf.summary.create_file_writer(log_dir)
    #with writer.as_default():
    #    tf.summary.image('layer output', layer[:, :, :, 0:3], max_output=3)


    
    #print(model.layers[0].weights)
    #print(model.layers[0].bias.numpy())
    #print(model.layers[0].bias_initializer)
    
   
    # Unnecessary. Input on fit function directly
    #batch_size = 128  #add try, ori=32
    #epochs = 80  #number of validation, watch then verify....  try, ori=1
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    earlystop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    
    model_history = model.fit(
              x_train,
              train_label,
              batch_size=128,   #64
              epochs=80,   #30
              verbose = 1,
              shuffle=True,
              validation_split=0.10,             #Get 0.1 as validatin data. Validation data is never shuffled.
              callbacks = [earlystop])
              #validation_data=(test_array, true_answer))
              #callbacks=[tensorboard_callback]) # 1112 TensorBoard
    
    
    # save model
    model.save('My_model')
    #model.save_weights("my_weights")
    

    #first_dense = model.layers[1]
    #last_dense = model.layers[-2]
    #my_ckpt_path = tf.train.Checkpoint(dense=first_dense, kernel=last_dense.kernel, bias=last_dense.bias).save("ckpt")
    
    
    #preloaded_layers = model.layers.copy()
    #preloaded_weights = []
    #for pre in preloaded_layers:
    #    preloaded_weights.append(pre.get_weights())
    #    
    #Mymodellayers=model.layers
    
    ##use data generator 
    ##  https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    #from tensorflow.keras.preprocessing.image import ImageDataGenerator    
    #datagen = ImageDataGenerator(
    #    featurewise_center = True,
    #    featurewise_std_normalization = False,
    #    rotation_range = 30,
    #    zoom_range = 0.20,
    #    fill_mode = "nearest",
    #    shear_range = 0.20,
    #    width_shift_range = 0.2,
    #    height_shift_range = 0.2,
    #    validation_split=0.2,
    #    horizontal_flip = True)
    #
    ##datagen.fit(x_train)
    #
    #
    #
    ##datagen.flow() is an address, instead of a returned value.
    ##It gets the yield value when it runs. This runs multiple times. It will not stop until the number of running
    ##over "steps_per_epoch".
    #batch_size=128
    #model_history = model.fit_generator(
    #                generator       = datagen.flow(x_train,train_label,batch_size=batch_size,subset='training'),
    #                steps_per_epoch = len(x_train)//batch_size,
    #                validation_data = datagen.flow(x_train,train_label,batch_size=batch_size,subset='validation'),
    #                validation_steps= len(x_train)*0.1//batch_size,
    #                epochs = 80,
    #                verbose = 1,   #1:appear progress
    #                shuffle = True,
    #                callbacks = [earlystop])
    
    #generator = datagen.flow(x_train, train_label, batch_size=len(x_train)*100)
    #print("trian len=",len(generator[0][0]))
    #print("batch_size=",generator.batch_size)
    #print("generator.n=",generator.n)
    
    ## save model
    #model.save('My_model')
    ##model.save_weights("my_weights")   


       
    #predict 1
    predict_percentage = model.predict(test_array)
    print(predict_percentage)
    
    #predict 2
    predict_results = model.predict_classes(test_array)
    print(predict_results)

    #acc, acc_op = tf.metrics.accuracy(labels=true_answer, predictions=predict_results)
    #sess = tf.Session()
    #sess.run(tf.local_variables_initializer())
    #sess.run(tf.global_variables_initializer())
    ##stream_vars = [i for i in tf.local_variables()]
    ##print('[total, count]:',sess.run(stream_vars))
    #print("Accuracy rate: ", sess.run([acc, acc_op]))
    #sess.close()


    return predict_results, predict_percentage, model_history
