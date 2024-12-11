
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix,  roc_curve
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.dummy import DummyClassifier

class Models:

    TRAIN_RATIO = 0.7
    TEST_RATIO = 0.3
    RANDOM_STATE = 42

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.results = []
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_splitter()

        

    # Function for Data Splitting
    def data_splitter(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.TEST_RATIO, random_state=self.RANDOM_STATE)

        return X_train, X_test, y_train, y_test
    

    # Function for model evaluation
    def __model_evaluation(self,model, model_name):
        result = {}
        mod = model.fit(self.X_train, self.y_train)
        training_score = mod.score(self.X_train, self.y_train)
        y_pred = mod.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        roc_cur = roc_curve(self.y_test, y_pred)
        prec_recall_cur = precision_recall_curve(self.y_test, y_pred)
        result['model'] = model_name
        result['training_acc'] = training_score
        result['testing_acc'] = accuracy
        result['f1_score'] = f1
        result['confusion_matrix'] = cm
        result['roc_curve'] = roc_cur
        result['precision_recall_curve'] = prec_recall_cur

        print(f"Training Accuracy: {training_score:.2f}, Testing Accuracy: {accuracy:.2f}, F1 Score: {f1}")
        
        self.results.append(result)

        return mod
    

    # Function for training baseline dummy model
    def baseline_model(self):
        dummy = DummyClassifier(strategy='most_frequent')
        dummy = self.__model_evaluation(dummy,'Baseline')

        return dummy
    
    # Function for training Logistic Regression model
    def logistic_regression(self, cv, max_iter):
        logistic = LogisticRegressionCV(cv=cv, max_iter=max_iter, random_state=self.RANDOM_STATE)
        lr = self.__model_evaluation(logistic,'Logistic Regression')

        return lr

    
    # Function for Random Forest Classifier
    def random_forest(self, max_depth, n_estimators):
        rf = RandomForestClassifier(random_state=self.RANDOM_STATE, max_depth=max_depth, n_estimators=n_estimators)
        rf = self.__model_evaluation(rf,'Random Forest')

        return rf


    # Function for K-nearest neighbour
    def k_neighbour(self, neighbours):
        knn = KNeighborsClassifier(n_neighbors=neighbours)
        knn = self.__model_evaluation(knn, 'KNN')

        return knn


    # Function for SVC
    def svc_classifier(self):
        svc = SVC(random_state=self.RANDOM_STATE)
        svc = self.__model_evaluation(svc,'SVC')

        return svc

