from sklearn.linear_model import LogisticRegression

class LogisticRegresion:
    def __init__(self,x_t,y_t,x_v,y_v):
        # Inicialitzem w0 i w1 (per ser ampliat amb altres w's)
        self.x_t = x_t
        self.y_t = y_t
        self.x_v = x_v
        self.y_v = y_v
    def Logistic_Regression(self):
        #Creem el regresor logístic
        logireg = LogisticRegression(C=2.0, fit_intercept=True, penalty='l2', tol=0.001, max_iter=10000000)
        # l'entrenem
        logireg.fit(self.x_t, self.y_t)
        print ("Correct classification Logistic ",  "% of the data: ", logireg.score(self.x_v, self.y_v))