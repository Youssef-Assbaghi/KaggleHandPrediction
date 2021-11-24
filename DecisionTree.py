from sklearn.tree import DecisionTreeClassifier
class Decision(object):
    def __init__(self,x_t,y_t,x_v,y_v):
        # Inicialitzem w0 i w1 (per ser ampliat amb altres w's)
        self.x_t = x_t
        self.y_t = y_t
        self.x_v = x_v
        self.y_v = y_v
    def decision_tree_gini(self):
        tree = DecisionTreeClassifier(max_depth=22)
        # L'entrenem 
        tree.fit(self.x_t,self.y_t)
        
        prdiction=tree.predict(self.x_v)
        print ("Decision tree criterio gini ", tree.score(self.x_v,self.y_v))
        
    def decision_tree_entropy(self):
        tree = DecisionTreeClassifier(criterion='entropy',max_depth=22)
        # L'entrenem 
        tree.fit(self.x_t,self.y_t)
        
        prdiction=tree.predict(self.x_v)
        print ("Decision tree criterio entropy ", tree.score(self.x_v,self.y_v))
