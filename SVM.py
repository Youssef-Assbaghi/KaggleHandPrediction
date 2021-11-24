from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import  precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn.model_selection import cross_val_score
class SVMS(object):
    def __init__(self,x_t,y_t,x_v,y_v):
        # Inicialitzem w0 i w1 (per ser ampliat amb altres w's)
        self.x_t = x_t
        self.y_t = y_t
        self.x_v = x_v
        self.y_v = y_v


        
    def SVM_polynomial(self):
        svc = svm.SVC(C=20.0, kernel='poly', gamma=0.9, probability=True)
        svc.fit(self.x_t, self.y_t)
        SVCprobs = svc.predict_proba(self.x_v)
        print ("Correct classification polynomial  with kernel 20  ", 80, "% of the data: ", svc.score(self.x_v, self.y_v))
        self.show_SVM(SVCprobs,"SVM POLYOMIAL")

    def SVM_gaussian(self):
        ######################################################SVC RBF############################################################
        svc = svm.SVC(C=200.0, kernel='rbf', gamma=0.9, probability=True)
        svc.fit(self.x_t, self.y_t)
        SVCprobs = svc.predict_proba(self.x_v)
        print ("Correct classification RBF  with kernel 200  ", 70, "% of the data: ", svc.score(self.x_v, self.y_v))
        self.show_SVM(SVCprobs,"SVM GAUSSIAN")

    def SVM_Linear(self):
        svc = svm.SVC(C=20.0, kernel='linear', gamma=0.9, probability=True)
        svc.fit(self.x_t, self.y_t)
        SVCprobs = svc.predict_proba(self.x_v)
        print ("Correct classification Linear SVC  with kernel 200  ", 70, "% of the data: ", svc.score(self.x_v, self.y_v))
        self.show_SVM(SVCprobs,"SVM LINEAR")


    def get_best_C(self):
        print("Best C using cross validation")
        # choose k between 1 to 50
        k_range = range(1, 50)
        k_scores = []
        # use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
        for k in k_range:
            svc = svm.SVC(C=k, kernel='rbf', gamma=0.9, probability=True)
            scores = cross_val_score(svc, self.x_t, self.y_t, cv=5, scoring='accuracy')
            k_scores.append(scores.mean())
        # plot to see clearly
        plt.plot(k_range, k_scores)
        plt.xlabel('Value of C for SVM')
        plt.ylabel('Cross-Validated Accuracy')
        plt.show()
        
        
    def show_SVM(self,SVCprobs,code):
        n_classes = 4
        
        precision = {}
        recall = {}
        average_precision = {}
        
        plt.figure()
        stri=code+" Precision Curve"
        plt.title(stri)
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(self.y_v == i, SVCprobs[:, i])
            average_precision[i] = average_precision_score(self.y_v == i, SVCprobs[:, i])
        
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
            fpr[i], tpr[i], _ = roc_curve(self.y_v == i, SVCprobs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        # Plot ROC curv
        stri=code+" ROC Curve"
        plt.figure()
        plt.title(stri)
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
        plt.legend()
        plt.show()
        