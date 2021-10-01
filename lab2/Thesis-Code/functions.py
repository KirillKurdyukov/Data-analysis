import yaml 

with open('./Thesis-Code/consts.yaml') as f:
    templates = yaml.safe_load(f)
    

%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

labels = templates['lables']
def draw_result(y_test: List[float],
                y_pred: List[float]) -> List[float]:
    def plot_confusion_matrix(y_test: List[float],
                              y_pred: List[float],
                              normalize=False: bool,
                              title=None: str,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=labels, yticklabels=labels,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.xlim(-0.5, len(np.unique(y_test))-0.5) # ADD THIS LINE
        plt.ylim(len(np.unique(y_test))-0.5, -0.5) # ADD THIS LINE
        return ax
    
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes=labels,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes=labels, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
    
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

def table_and_get_restult(clf: Union[SGDClassifier,
                                     MultinomialNB,
                                     RandomForestClassifier,
                                     DecisionTreeClassifier,
                                     LogisticRegression],
                          X_train: List[float],
                          y_train: List[float],
                          X_test: List[float],
                          y_test: List[float]) -> List[float]:
    svm = Pipeline([('vect', CountVectorizer()), 
                   ('tfidf', TfidfTransformer()),
                   ('clf', clf),
                  ])
    Gmodel=svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    accuracy = cross_val_score(Gmodel,X_train,y_train,cv=5,scoring='accuracy')
    print("Cross val score:",accuracy)
    print("Accuracy of Model with Cross Validation is:",accuracy.mean() * 100)
    print('accuracy %s' % accuracy_score(y_pred, y_test, normalize=False))
    print('accuracy normalized %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred,target_names=labels))
    return y_pred 

