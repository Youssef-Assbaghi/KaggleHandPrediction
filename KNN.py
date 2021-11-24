from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
class KNN(object):
    def __init__(self,x_t,y_t,x_v,y_v):
        # Inicialitzem w0 i w1 (per ser ampliat amb altres w's)
        self.x_t = x_t
        self.y_t = y_t
        self.x_v = x_v
        self.y_v = y_v
        self.n_neigbours=4
    def KNN_Classifier(self):
        #Creem knn 
        knn = KNeighborsClassifier(n_neighbors=self.n_neigbours)
        #L'entrenem 
        knn.fit(self.x_t,self.y_t)
        prdiction=knn.predict(self.x_v)
        print ("KNN ", knn.score(self.x_v,self.y_v))
    def get_best_neighbours(self):
        # choose k between 1 to 25
        k_range = range(1, 25)
        k_scores = []
        # use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, self.x_t, self.y_t, cv=5, scoring='accuracy')
            k_scores.append(scores.mean())
        # plot to see clearly
        max_value = max(k_scores)
    
        max_index = k_scores.index(max_value)
        self.n_neigbours=max_index
        print("mx negbour is ", max_index)
        plt.plot(k_range, k_scores)
        plt.xlabel('Value of K for KNN')
        plt.ylabel('Cross-Validated Accuracy')
        plt.show()