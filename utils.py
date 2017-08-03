import pandas as pd
import numpy as np
import scipy as sp
import os
import glob
import re
import time
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

def print_time_info(string):
    T = time.gmtime()
    Y, M, D = T.tm_year, T.tm_mon, T.tm_mday
    h, m, s = T.tm_hour, T.tm_min, T.tm_sec
    print("[{}-{:0>2}-{:0>2} {:0>2}:{:0>2}:{:0>2}] {}".format(Y, M, D, h, m, s, string))

def read_images_and_metadata(data_dir, pick_class):
    pick_class_text = str(pick_class)
    if len(pick_class_text) < 2:
        pick_class_text = '0' + pick_class_text
    # Get the books' metadata
    f_target = os.path.join(data_dir, "df_all_for_cnn.csv")
    df = pd.read_csv(f_target)
    df["productID"] = "00" + df["productID"].astype("str")
    df_pid = df['productID'].unique()
    print('\nThe data frame of all products "df" has shape %s\n'%(df.shape,))
    print('\nThere are %d unique PID\n'%(df_pid.shape[0]))

    # Get the books' cover image
    if pick_class == 1:
        ## For category 1, use processed covers in novels_pos/
        dir_images = os.path.join(data_dir, "novels_pos/")
    elif pick_class == 9:
        ## For category 9, use processed covers in c9_go/
        dir_images = os.path.join(data_dir, "c9_go/")
    else:
        dir_images = os.path.join(data_dir, "productImg_orisize/")

    f_imgs = glob.glob(dir_images + "/*")

    k = [re.split("_|\\.", os.path.basename(item))[0] for item in f_imgs]

    ## read image paths
    df_img = pd.DataFrame(
        {"productID": k,
         "img_path":f_imgs})

    df_img = df_img.drop_duplicates(subset = "productID", keep = "last")

    # Merge the images and metadata.

    df = pd.merge(df, df_img, left_on= "productID", right_on= "productID", how = "left")
    df_pid = df['productID'].unique()
    print('\n(After merge)')
    print('The data frame of all products "df" has shape %s'%(df.shape,))
    print('There are %d unique PID\n'%(df_pid.shape[0]))

    # Check the missing rate of the data; if it's too high, stop training.
    re_parse = df.loc[df.img_path.isnull()].productID
    if len(re_parse) > 1000:
        re_parse.to_csv(path = "test.csv", index=False)
        print("\nThe data is corrupted, exit...")
        go = False
    else:
        #print("The data is ready!")
        go = True
   
    return df, go

def clean_data(data_dir, df):

    df_good = df.loc[~df.img_path.isnull()]
    df_good = df_good.drop_duplicates(subset = "productID", keep = "first")
    df_good = df_good.reset_index(drop=True) # reset index (due to remove duplicate)

    print('\nThe data frame of all valid products "df_good" has shape %s\n'%(df_good.shape,))

    avail_train = pd.read_csv(os.path.join(data_dir, "availtrain2.csv"))
    avail_train["x"] = "00" + avail_train["x"].astype("str")
    #print(avail_train.shape)
    #print(avail_train.head())
    df_good = df_good[df_good.productID.isin(avail_train["x"])]
    print('\nThe data frame of all valid products "df_good" has shape %s\n'%(df_good.shape,))
    print('\nFirst 5 rows:\n')
    print(df_good[['productID','this.cate','sales.12wk']].head())

    y_pickup = 'sales.12wk'

    ## Split training and testing sets (used in training on single category)
    # df_good = df_good.loc[df_good["this.cate"] == pick_class]
    # df_good = df_good.reset_index(drop=True)

    label_PR = np.array([sp.stats.percentileofscore(df_good[y_pickup], a, 'rank') for a in df_good[y_pickup]]) # PR-value (99-> good)
    label_PR /= 100

    # stratified sampling
    label_PR_cutted = pd.cut(label_PR, 5)
    #print(label_PR_cutted.value_counts())
    sorted_df = pd.DataFrame({'ind':range(0,len(label_PR_cutted)),
                       'cat':label_PR_cutted})
    label_PR_cutted = sorted_df.pop('cat')

    return df_good, sorted_df, label_PR, label_PR_cutted

def split_data(sorted_df, label_PR_cutted):
    i_train, i_test, j_train, j_test = train_test_split(sorted_df.ind, 
                                                        label_PR_cutted, 
                                                        test_size = 0.1, 
                                                        stratify = label_PR_cutted)

    #print(j_train.value_counts())
    #print(j_test.value_counts())

    # Generate train / test mask
    i_train[i_train>=0] = True
    i_test[i_test>=0] = False
    sorted_df['label'] = i_train.append(i_test)
    msk = np.array(sorted_df['label'] == 1) # with stratified sampling

    #print(i_train.shape)
    #print(i_test.shape)
    #print(j_train.shape)
    #print(j_test.shape)
    
    return sorted_df, msk

def gen_features(df_good):
    #print(df_good['this.cate'].unique())
    nz = len(df_good['this.cate'].unique())
    #print('nz = %d'%nz)

    category_dict = dict(zip(df_good['this.cate'].unique(), np.arange(nz)))

    cateInfo = df_good['this.cate']
    cateInfo = np.array(cateInfo.replace(category_dict))
    cateInfo = np_utils.to_categorical(y= cateInfo, num_classes=nz)

    return [cateInfo]
