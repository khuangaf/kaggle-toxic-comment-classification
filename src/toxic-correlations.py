import pandas as pd
import sys
from scipy.stats import ks_2samp
import numpy as np

files= ['output/sp_bilstm_relu_d1_atten_global_amsgrad_bag_rmnum_ft125.csv', 'output/one_more_blend_985.csv','output/sp_cnn_f256_k2345_d1_400_50_relu_global_bag_rmnum_ft225.csv','output/sp_bigru_ft100atten_all.csv','output/sp_bigru_relu_ft100_atten_global_nadam_bag_rmnum_sw.csv','output/lgb.csv','output/sp_cnn_f300_k2345_d1_250_50_rs_global_bag_rmnum_ft300.csv','output/sp_cnn_f300_k2345_d1_250_50_rs_global_all_rmnum_ft300.csv']
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

file_map={'output/sp_bilstm_relu_d1_atten_global_amsgrad_bag_rmnum_ft125.csv': 'bilstm_bag',
         'output/one_more_blend_985.csv':'omb',
          'output/sp_cnn_f256_k2345_d1_400_50_relu_global_bag_rmnum_ft225.csv':'cnn_bag',
          'output/sp_bigru_ft100atten_all.csv':'bigru',
          'output/sp_bigru_relu_ft100_atten_global_nadam_bag_rmnum_sw.csv':'bigru_bag',
          'output/lgb.csv':'lgb',
          'output/sp_cnn_f300_k2345_d1_250_50_rs_global_bag_rmnum_ft300.csv':'cnn300_bag',
          'output/sp_cnn_f300_k2345_d1_250_50_rs_global_all_rmnum_ft300.csv':'cnn'
          
         }
def corr(first_file, second_file, class_name):
    # assuming first column is `class_name_id`
    first_df = pd.read_csv(first_file, index_col=0)
    second_df = pd.read_csv(second_file, index_col=0)
    class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    

    return first_df[class_name].corr(second_df[class_name], method='pearson')



for class_name in class_names:
    print (class_name)
    df = pd.DataFrame(np.zeros([len(files), len(files)]), columns = [file_map[f] for f in files], index= [file_map[f] for f in files])
    for first_file in files:
        for second_file in files:
            df.loc[df.index == file_map[first_file], file_map[second_file]]=corr(first_file, second_file, class_name)
    df.to_csv(class_name+'_correlation.csv')
#     print(df.head())
            