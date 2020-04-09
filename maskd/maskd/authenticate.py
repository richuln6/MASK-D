import pandas as pd
import numpy as np
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from sqlalchemy import create_engine


df_en_all=pd.read_pickle('LogFileDfs/LogDumpEncoded')


def reduce(df):
    pd.set_option('display.max_rows',None)
    pca_reducer = PCA(n_components=2)
    reduced_data = pca_reducer.fit_transform(df)
    pdf = pd.DataFrame(data = reduced_data
             , columns = ['pc 1', 'pc 2'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    pdf[['pc 1','pc 2']] = scaler.fit_transform(pdf[['pc 1','pc 2']])
    X1 = pdf['pc 1'].values.reshape(-1,1)
    X2 = pdf['pc 2'].values.reshape(-1,1)
    X = np.concatenate((X1,X2),axis=1)
    return X

def pyodtry(name):
    dfwhole = pd.read_pickle('LogFileDfs/LogDumpEncoded')
    df=pd.read_pickle(name)
    X1=reduce(dfwhole)
    X2=reduce(df)
    ddf = pd.read_pickle('LogFileDfs/original')

    random_state = np.random.RandomState(42)
    outliers_fraction = 0.005
    clf = KNN(method='mean',contamination=outliers_fraction)
    xx , yy  = np.meshgrid(np.linspace(0,1 , 200), np.linspace(0, 1, 200))
    
    clf.fit(X1)
    scores_pred = clf.decision_function(X2) * -1
    y_pred = clf.predict(X2)
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)
    print('OUTLIERS : ',n_outliers,'INLIERS : ',n_inliers)
    #dfx = pdf
    #dfx['outlier'] = y_pred.tolist()
    df['authenticated?'] = y_pred.tolist()
    ddf['authenticated?'] = df['authenticated?']
    output= ddf[ddf['authenticated?'] == 1]
    # create sqlalchemy engine
    #engine = create_engine("mysql+pymysql://{user}:{pw}@172.17.0.3/{db}".format(user="richul",pw="richul123",db="emss"))
    # Insert whole DataFrame into  MySQL
    #output.to_sql('output', con = engine, if_exists = 'replace', chunksize = 1000)
    with pd.ExcelWriter('/home/richul/Documents/EnhancingMailServerSecurity/Output/output.xlsx') as writer:
        output.to_excel(writer, sheet_name='output')
import sys
name=sys.argv[1]

pyodtry(name)

