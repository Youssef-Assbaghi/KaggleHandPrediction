import pandas as pd
pd.options.mode.chained_assignment = None #Quitamos los warnings por usar copy en dataframe
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from Ensembler import Ensembler
from KNN import KNN
from DecisionTree import Decision
from SVM import SVMS
from LogisticRegression import LogisticRegresion
def load_dataset():
    csv0 = pd.read_csv('csv/0.csv',  header=None, delimiter=',')
    csv1 = pd.read_csv('csv/1.csv', header=None,  delimiter=',')
    csv2 = pd.read_csv('csv/2.csv',  header=None, delimiter=',')
    csv3 = pd.read_csv('csv/3.csv',   header=None, delimiter=',')
    
    lista = [csv0,csv1,csv2,csv3]
    
    dataset = pd.concat(lista)
    return dataset


def train_test(dataset):
    x=dataset.values
    min_max_scaler=preprocessing.MinMaxScaler()
    x_scale=min_max_scaler.fit_transform(x)
    df=pd.DataFrame(x_scale)
    df[64]=dataset[64].values
    train,test=train_test_split(df,test_size=0.1)
    entrenar,validar=train_test_split(train,test_size=0.3)
    
    y_t=entrenar[64].to_numpy()
    entrenar.drop(64,inplace=True,axis=1)
    x_t=entrenar.to_numpy()
    
    y_v=validar[64].to_numpy()
    validar.drop(64,inplace=True,axis=1)
    x_v=validar.to_numpy()
    return x_t,y_t,x_v,y_v

#Funcion principal que ejecutara los clasificadores
if __name__ == '__main__':
    print("Start execution")
    
    dataset=load_dataset()
    x_t,y_t,x_v,y_v=train_test(dataset)
    

    """ Bagging and boosting algorithms"""
    ens=Ensembler(x_t,y_t,x_v,y_v)
    ens.bagging_decission_tree()
    ens.boosting_decision_tree()
    
    """"SVM """
    svm=SVMS(x_t,y_t,x_v,y_v)
    #Set best C for SVM WITH KERNEL RBF using cross validation
    # Be careful very complexity computation
    #svm.get_best_C()
    svm.SVM_gaussian()
    svm.SVM_polynomial()
    svm.SVM_Linear()
    
    """Logistic regression"""
    log_regression=Logistic_regression=LogisticRegresion(x_t,y_t,x_v,y_v)
    log_regression.Logistic_Regression()
    """KNN algorithm"""
    knn=KNN(x_t,y_t,x_v,y_v)
    #Set bes K for knn algorithm with cross validation
    knn.get_best_neighbours()
    
    knn.KNN_Classifier()
    """ DECISION TREE"""
    decision=Decision(x_t,y_t,x_v,y_v)
    decision.decision_tree_gini()
    decision.decision_tree_entropy()

    print("End execution")
