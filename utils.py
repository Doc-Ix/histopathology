def expand_dimension(x):
    import numpy as np
    return np.expand_dims(x,axis=0)


def build_data_generators(IMAGE_SIZE, BATCH_SIZE, RESCALE, ImageDataGenerator, train_datagenerator):
    
    datagen_train = train_datagenerator.flow_from_directory('model_data/train',
                                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='binary')
    
    datagen_valid = ImageDataGenerator(rescale=RESCALE).flow_from_directory('model_data/valid',
                                                                        target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                                        batch_size=BATCH_SIZE,
                                                                        class_mode='binary')
    
    datagen_test = ImageDataGenerator(rescale=RESCALE).flow_from_directory('model_data/test', 
                                                                       target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                                                       batch_size=BATCH_SIZE,
                                                                       class_mode='binary')
    
    return datagen_train, datagen_valid, datagen_test


def create_df_test(path_to_label_0, path_to_label_1):
    
    from utils import expand_dimension
    import numpy as np
    import pandas as pd
    from glob import glob
    import os
    from imread import imread
    
    #create dataframes with all .tif-data and 'path'-column
    df_0 = pd.DataFrame({'path': glob(os.path.join(path_to_label_0,'*.tif'))})
    df_1 = pd.DataFrame({'path': glob(os.path.join(path_to_label_1,'*.tif'))})

    #create id from filepath
    df_0['id'] = df_0.path.map(lambda x: x.split('/')[3].split('.')[0])
    df_0['label'] = 0

    df_1['id'] = df_1.path.map(lambda x: x.split('/')[3].split('.')[0])
    df_1['label'] = 1

    #combine dataframes, get new index
    df_test = pd.concat([df_0,df_1],ignore_index=True)
    
    #read single images to dataframe and normalize arrays
    df_test['image'] = (df_test['path'].map(imread))/255
    
    #expand dimension of array to include batch-size-dimension for tensorflow
    df_test['image_tensor'] = df_test['image'].apply(expand_dimension)
    
    #drop unneeded columns
    df_test.drop(labels=['image'], axis=1, inplace=True)
    
    return df_test


def model_prediction_binary(prediction, threshold=0.5):
    if prediction > threshold:
        return 1
    elif prediction <= threshold:
        return 0
    else:
        print('Prediction Error! Value: {}'.format(prediction))
