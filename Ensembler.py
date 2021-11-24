import pandas as pd
pd.options.mode.chained_assignment = None #Quitamos los warnings por usar copy en dataframe
import matplotlib.pyplot as plt
from sklearn.metrics import  precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier

#from SVM import SVMS
class Ensembler(object):
    def __init__(self,x_t,y_t,x_v,y_v):
        # Inicialitzem w0 i w1 (per ser ampliat amb altres w's)
        self.x_t = x_t
        self.y_t = y_t
        self.x_v = x_v
        self.y_v = y_v
    
    def bagging_decission_tree(self):
        bagging = BaggingClassifier(n_estimators=300,random_state=0)
        bagging.fit(self.x_t,self.y_t)
        #print(self.x_v)
        baggingProbs = bagging.predict_proba(self.x_v)
        
        self.show_plots("BAGGING",baggingProbs)
   
    def boosting_decision_tree(self):

        boosting = GradientBoostingClassifier(n_estimators=210, learning_rate=1.1,max_depth=1, random_state=0).fit(self.x_t, self.y_t)
        boostingProbs = boosting.predict_proba(self.x_v)
        self.show_plots("BOOSTING",boostingProbs)
   
        
    def show_plots(self,code,ensembler_probs):
        n_classes = 4
        
        precision = {}
        recall = {}
        average_precision = {}
        
        
        plt.figure()
        stri=code + " DECISION TREE"
        plt.title(stri)
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(self.y_v == i, ensembler_probs[:, i])
            average_precision[i] = average_precision_score(self.y_v == i, ensembler_probs[:, i])
        
            plt.plot(recall[i], precision[i],
            label='Precision-recall curve of class {0} (area = {1:0.2f})'
                                   ''.format(i, average_precision[i]))
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc="upper right")
        
        plt.show()
        
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_v == i, ensembler_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        # Plot ROC curve
        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
        plt.legend()
        plt.show()

