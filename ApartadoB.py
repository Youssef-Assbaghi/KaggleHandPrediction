import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
random_state = np.random.RandomState(0)
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, roc_curve, auc
from sklearn import neighbors, datasets
import seaborn as sns


def grafiquesRendiment(probs,modelo,y_v):
    # Compute Precision-Recall and plot curve
    precision = {}
    recall = {}
    average_precision = {}
    plt.figure()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_v == i, probs[:, i])
        average_precision[i] = average_precision_score(y_v == i, probs[:, i])

        plt.plot(recall[i], precision[i],
        label='Precision-recall curve of class {0} (area = {1:0.2f})'
                           ''.format(i, average_precision[i]))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc="upper right")

    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_v == i, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    # Plot ROC curve
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
    plt.legend()
    
    microavargePrecisionRecall(modelo)


def microavargePrecisionRecall(modelo):
    
    # Use label_binarize to be multi-label like settings
    Y = label_binarize(y, classes=[0, 1, 2])
    n_classes = Y.shape[1]

    # Split into training and test
    X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=random_state
    )

    
    classifier = OneVsRestClassifier(
    make_pipeline(StandardScaler(), modelo))
    
    
    classifier.fit(X_train, Y_train)
    y_score = classifier.decision_function(X_test)
    
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        Y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot()
    _ = display.ax_.set_title("Micro-averaged over all classes")


def load_dataset():
    # import some data to play with
    iris = datasets.load_iris()
    
    # Take the first two features. We could avoid this by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target
    
    n_classes = 3
    
    """
    fig, sub = plt.subplots(1, 2, figsize=(16,6))
    sub[0].scatter(X[:,0], y, c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    sub[1].scatter(X[:,1], y, c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    """
    return iris,X,y,n_classes

def execute_clasificators():
    particions = [0.5, 0.7, 0.8]

    for part in particions:
        x_t, x_v, y_t, y_v = train_test_split(X, y, train_size=part)
        
        #Creem el regresor logístic
        logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)
    
        # l'entrenem
        logireg.fit(x_t, y_t)
    
        print ("Correct classification Logistic              ", part, "% of the data: ", logireg.score(x_v, y_v))
        probsRegressioLog = logireg.predict_proba(x_v)
        pred = logireg.predict(x_v)
        fpr, tpr, thresholds = metrics.roc_curve(y_v, pred, pos_label=2)
        aucLogiReg = metrics.auc(fpr, tpr)
        
        #Creem el regresor logístic
        svc = svm.SVC(C=10.0, kernel='rbf', gamma=0.9, probability=True)
    
        # l'entrenem 
        svc.fit(x_t, y_t)
        probsSVC = svc.predict_proba(x_v)
        print ("Correct classification SVM                   ", part, "% of the data: ", svc.score(x_v, y_v))
        pred = svc.predict(x_v)
        fpr, tpr, thresholds = metrics.roc_curve(y_v, pred, pos_label=2)
        aucSVC = metrics.auc(fpr, tpr)
        
    
        """
        #Creem arbre de decisió, criteri: gini 
        tree = DecisionTreeClassifier()
        
        # L'entrenem 
        tree.fit(x_t,y_t)
        print ("Correct classification DecisionTree gini     ", part, "% of the data: ", tree.score(x_v, y_v))
        """
        #Creem arbre de decisió, criteri: 
        tree = DecisionTreeClassifier(criterion='entropy')
    
        
        # L'entrenem 
        tree.fit(x_t,y_t)
        print ("Correct classification DecisionTree entropy  ", part, "% of the data: ", tree.score(x_v, y_v))
        pred = tree.predict(x_v)
        fpr, tpr, thresholds = metrics.roc_curve(y_v, pred, pos_label=2)
        aucTree = metrics.auc(fpr, tpr)
        
        #Creem knn 
        
        knn = KNeighborsClassifier(n_neighbors=5)
        
        #L'entrenem 
        
        knn.fit(x_t,y_t)
        probsKnn = knn.predict_proba(x_v)
        
        print ("Correct classification knn                   ", part, "% of the data: ", knn.score(x_v, y_v))
        pred = knn.predict(x_v)
        fpr, tpr, thresholds = metrics.roc_curve(y_v, pred, pos_label=2)
        aucKNN = metrics.auc(fpr, tpr)
    return x_t,y_t,x_v,y_v,svc,probsSVC,probsRegressioLog,logireg,svc,tree,knn

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def best_K_KNN():
    # choose k between 1 to 25
    k_range = range(1, 25)
    k_scores = []
    # use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x_t, y_t, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())
    # plot to see clearly
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()
    

def best_C_SVM_rbf():
    # choose  between 1 to 50
    c_range = range(1, 50)
    c_scores = []
    # use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
    for c in c_range:
        svc = svm.SVC(C=c, kernel='rbf', gamma=0.9, probability=True)
        scores = cross_val_score(svc, x_t, y_t, cv=15, scoring='accuracy')
        c_scores.append(scores.mean())
    # plot to see clearly
    plt.plot(c_range, c_scores)
    plt.xlabel('Value of c for SVC')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()

def best_depth_tree():
    # choose  between 1 to 50
    k_range = range(1, 50)
    k_scores = []
    # use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
    for k in k_range:
        tree = tree = DecisionTreeClassifier(criterion='entropy',max_depth= k)
        scores = cross_val_score(tree, x_t, y_t, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    # plot to see clearly
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of max_depth for SVC')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()
  
def best_C_logistic_regression():
    # choose  between 1 to 50
    k_range = range(1, 50)
    k_scores = []
    # use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
    for k in k_range:
        logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001)
        scores = cross_val_score(logireg, x_t, y_t, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    # plot to see clearly
    plt.plot(k_range, k_scores)
    plt.xlabel('Value of C for LogisticRegression')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()
def show_accuracy():
    k = 10
    scores = cross_val_score(logireg, x_t, y_t, cv=k, scoring='accuracy')
    scores2 = cross_val_score(logireg, x_t, y_t, cv=k, scoring='roc_auc_ovr')
    print("LogisticRegresion with cross-validation->  Acurracy: ",scores.mean(), "ROC area: ",scores2.mean())
    scores = cross_val_score(svc, x_t, y_t, cv=k, scoring='accuracy')
    scores2 = cross_val_score(svc, x_t, y_t, cv=k, scoring='roc_auc_ovr')
    print("SVM with cross-validation->  Acurracy: ",scores.mean(), "ROC area: ",scores2.mean())
    scores = cross_val_score(tree, x_t, y_t, cv=k, scoring='accuracy')
    scores2 = cross_val_score(tree, x_t, y_t, cv=k, scoring='roc_auc_ovr')
    print("DecisionTree with cross-validation->  Acurracy: ",scores.mean(), "ROC area: ",scores2.mean())
    scores = cross_val_score(knn, x_t, y_t, cv=k, scoring='accuracy')
    scores2 = cross_val_score(knn, x_t, y_t, cv=k, scoring='roc_auc_ovr')
    print("Knn with cross-validation->  Acurracy: ",scores.mean(), "ROC area: ",scores2.mean())
    
def show_SVM_accuracy():
    models = (svm.SVC(kernel='linear'),
              svm.LinearSVC(max_iter=1000000),
              svm.SVC(kernel='rbf'),
              svm.SVC(kernel='poly', gamma='auto'))          

    title = ('SVC with linear kernel','LinearSVC (linear kernel)','SVC with RBF kernel','SVC with polynomial (degree 3) kernel')
    k=10          
    for i in range(4):
        scores = cross_val_score(models[i], x_t, y_t, cv=k, scoring='accuracy')
        scores2 = cross_val_score(tree, x_t, y_t, cv=k, scoring='roc_auc_ovr')
        print(title[i]," with cross-validation->  Acurracy: ",scores.mean(), "ROC area: ",scores2.mean())

def show_deciion_tree_accuracy():
    k = 10
    tree = DecisionTreeClassifier(criterion='entropy')
    scores = cross_val_score(tree, x_t, y_t, cv=k, scoring='accuracy')
    scores2 = cross_val_score(tree, x_t, y_t, cv=k, scoring='roc_auc_ovr')
    print("DecisionTree - Entropy with cross-validation->  Acurracy: ",scores.mean(), "ROC area: ",scores2.mean())
    tree = DecisionTreeClassifier(criterion='gini')
    scores = cross_val_score(tree, x_t, y_t, cv=k, scoring='accuracy')
    scores2 = cross_val_score(tree, x_t, y_t, cv=k, scoring='roc_auc_ovr')
    print("DecisionTree - Gini with cross-validation->  Acurracy: ",scores.mean(), "ROC area: ",scores2.mean())

def show_C_effect(C=1.0, gamma=0.7, degree=3):

    # import some data to play with
    iris = datasets.load_iris()
    # Take the first two features. We could avoid this by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target

    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')

    #C = 1.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C, max_iter=1000000),
              svm.SVC(kernel='rbf', gamma=gamma, C=C),
              svm.SVC(kernel='poly', degree=degree, gamma='auto', C=C))
    models = (clf.fit(X, y) for clf in models)
    
    #plt.close('all')
    fig, sub = plt.subplots(2, 2, figsize=(14,9))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Sepal length')
        ax.set_ylabel('Sepal width')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        clf.predict(x_v)
        aux = clf.score(x_v,y_v)
        print(aux)
        
    plt.show()
    
def show_separation_KNN(n_neighbors):
    h = 0.02  # step size in the mesh
    
    # Create color maps
    cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
    cmap_bold = ["darkorange", "c", "darkblue"]
    
    for weights in ["uniform", "distance"]:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors)
        clf.fit(X, y)
    
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=cmap_light)
    
        # Plot also the training points
        sns.scatterplot(
            x=X[:, 0],
            y=X[:, 1],
            hue=iris.target_names[y],
            palette=cmap_bold,
            alpha=1.0,
            edgecolor="black",
        )
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(
            "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
        )
        plt.xlabel(iris.feature_names[0])
        plt.ylabel(iris.feature_names[1])
    
    plt.show()
if __name__ == '__main__':
    print("Start execution")
    iris,X,y,n_classes=load_dataset()
    x_t,y_t,x_v,y_v,modelo,probs,probsRegressioLog,logireg,svc,tree,knn=execute_clasificators()
    grafiquesRendiment(probs,modelo,y_v)
    grafiquesRendiment(probsRegressioLog,logireg,y_v)
    show_C_effect(C=0.1)
    show_C_effect(C=11)
    show_accuracy()
    best_K_KNN()
    best_depth_tree()
    best_C_logistic_regression()
    show_SVM_accuracy()
    show_deciion_tree_accuracy()
    show_separation_KNN(5)
    show_separation_KNN(16)
