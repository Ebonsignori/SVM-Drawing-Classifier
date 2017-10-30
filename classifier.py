import warnings
import random  # For randomizing labelled examples in accuracy testing
import math

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn import svm  # For classification


class DrawingClassifier():
    def __init__(self):
        self.X = list()
        self.y = list()
        self.min_features_length = 0

    def train(self, examples, labels, unlabelled):
        examples, unlabelled = self.normalizeFeatureLengths(examples, unlabelled)

        if (examples == 0):
            return

        # Create classifier with polynomial kernel
        self.clf = svm.SVC(kernel='poly')

        # Get minimum features of trainign examples and new example
        self.min_features_length = self.getMinFeaturesLen(examples, unlabelled)

        if (self.min_features_length <= 3):
            print("Error not enough features in some drawing")
            return

        # Represent datapoints as X and corresponding labels as y
        for points in examples:
            self.X.append(points[:self.min_features_length])
        self.y = labels

        # Train the classifier
        self.clf.fit(self.X, self.y)

    def predict(self, unlabelled):
        if (self.min_features_length > 2):
            # Predict new drawing using the trained classifier
            return "Your drawing is " + str(self.clf.predict(unlabelled[:self.min_features_length])[0])
        else:
            return "Error not enough features in some drawing"

    def getMinFeaturesLen(self, examples, unlabeled=None):
        # Find the minimum number of features in any example and new drawing
        min_features_length = len(examples[0])
        for i in range(len(examples)):
            if (len(examples[i]) < min_features_length):
                min_features_length = len(examples[i])
        # If new unlablled drawing is passed, include its length
        if (unlabeled):
            new_drawing_length = len(unlabeled)
            if (new_drawing_length < min_features_length):
                min_features_length = new_drawing_length

        return min_features_length

    def getAccuracy(self, examples, labels, kernel_type, percent, thorough):
        if (len(examples) < 3):
            print("Please test accuracy with at least 3 samples")
            return "too_small"

        # Preprocess so that feature lengths are all the same
        examples = self.normalizeFeatureLengths(examples)
        min_features_len = self.getMinFeaturesLen(examples)
        X = list()
        for points in examples:
            X.append(points[:min_features_len])

        # Create tuples of examples and their label and randomize their order
        labelled_examples = [(X[i], labels[i]) for i in range(len(examples))]

        # Find accuracy of randomly shuffled examples
        if (not thorough):
            random.shuffle(labelled_examples)

            total_X = [pair[0] for pair in labelled_examples]
            total_y = [pair[1] for pair in labelled_examples]
            test_train_num = int(math.ceil(len(total_X) * (percent * .01)))

            train_X = total_X[test_train_num:]
            test_X = total_X[:test_train_num]
            train_y = total_y[test_train_num:]
            test_y = total_y[:test_train_num]

            clf = svm.SVC(kernel=kernel_type)
            clf.fit(train_X, train_y)

            total = len(test_y)
            correct = 0
            incorrect = 0
            i = 0
            for x in test_X:
                label = clf.predict(x)

                # print("Predicted: ")
                # print(label[0])
                # print("Actual: ")
                # print(test_y[i])

                if (sorted(list(test_y[i])) == sorted(list(label[0]))):
                    correct += 1
                else:
                    incorrect += 1
                i += 1

            return str(round((correct / total) * 100))

        # Return average of the accuracy of each example, label pair combination
        else:
            correct_array = list()

            # For each combination
            for n in range(len(labelled_examples)):
                labelled_examples = self.shiftList(labelled_examples, n)

                total_X = [pair[0] for pair in labelled_examples]
                total_y = [pair[1] for pair in labelled_examples]

                test_train_num = int(math.ceil(len(total_X) * (percent * .01)))

                train_X = total_X[test_train_num:]
                test_X = total_X[:test_train_num]
                train_y = total_y[test_train_num:]
                test_y = total_y[:test_train_num]

                clf = svm.SVC(kernel=kernel_type)
                clf.fit(train_X, train_y)

                correct = 0
                incorrect = 0
                total = len(test_y)
                i = 0
                for x in test_X:
                    label = clf.predict(x)

                    if (sorted(list(test_y[i])) == sorted(list(label[0]))):
                        correct += 1
                    else:
                        incorrect += 1
                    i += 1

                correct_array.append(correct / total)

            return str((sum(correct_array) / len(correct_array)) * 100)

    def ClassifyCaresianMethod(self, examples_in, labels):
        unlabelled_drawing = examples_in[-1]
        examples = examples_in[:-1]

        X = []
        y = []

        i = 0
        for example in examples:
            for cordinate in example:
                X.append(cordinate)
                y.append(labels[i])
            i += 1

        clf = svm.SVC(kernel='linear')
        clf.fit(X, y)
        predicted = clf.predict(unlabelled_drawing)
        return self.mostLikelyClass(predicted)

    def getAccuracyCartesianMethod(self, examples, labels, kernel_type, percent, thorough):
        if (len(examples) < 3):
            print("Please test accuracy with at least 3 samples")
            return "too_small"

        X = []
        y = labels

        minimum_cords = 999999999
        for example in examples:
            if len(example) < minimum_cords:
                minimum_cords = len(example)

        i = 0
        for example in examples:
            X.append(example[0:])
            i += 1

        # Create tuples of examples and their label and randomize their order
        labelled_examples = [(X[i], y[i]) for i in range(len(X))]

        if (thorough):
            correct_array = list()
            # For each combination
            for n in range(len(labels)):
                # Shift by drawing
                labelled_examples = self.shiftList(labelled_examples, n)

                total_X = [pair[0] for pair in labelled_examples]
                total_y = [pair[1] for pair in labelled_examples]

                test_train_num = int(math.ceil(len(total_X) * (percent * .01)))

                train_X = total_X[test_train_num:]
                test_X = total_X[:test_train_num]
                train_y = total_y[test_train_num:]
                test_y = total_y[:test_train_num]

                train_X_processed = []
                train_y_processed = []
                i = 0
                for example in train_X:
                    for cordinate in example:
                        train_X_processed.append(cordinate)
                        train_y_processed.append(train_y[i])
                    i += 1

                clf = svm.SVC(kernel=kernel_type)
                clf.fit(train_X_processed, train_y_processed)

                correct = 0
                incorrect = 0
                total = len(test_y)
                i = 0

                for x in test_X:
                    x_processed = []
                    for cordinate in x:
                        x_processed.append(cordinate)

                    label = self.mostLikelyClass(clf.predict(x_processed))

                    if (sorted(list(test_y[i])) == sorted(list(label[0]))):
                        correct += 1
                    else:
                        incorrect += 1
                    i += 1

                correct_array.append(correct / total)

            return str((sum(correct_array) / len(correct_array)) * 100)

        else:
            random.shuffle(labelled_examples)

            total_X = [pair[0] for pair in labelled_examples]
            total_y = [pair[1] for pair in labelled_examples]

            test_train_num = int(math.ceil(len(total_X) * (percent * .01)))

            train_X = total_X[test_train_num:]
            test_X = total_X[:test_train_num]
            train_y = total_y[test_train_num:]
            test_y = total_y[:test_train_num]

            train_X_processed = []
            train_y_processed = []
            i = 0
            for example in train_X:
                for cordinate in example:
                    train_X_processed.append(cordinate)
                    train_y_processed.append(train_y[i])
                i += 1

            clf = svm.SVC(kernel=kernel_type)
            clf.fit(train_X_processed, train_y_processed)

            correct = 0
            incorrect = 0
            total = len(test_y)
            i = 0

            for x in test_X:
                x_processed = []
                for cordinate in x:
                    x_processed.append(cordinate)

                label = self.mostLikelyClass(clf.predict(x_processed))

                if (sorted(list(test_y[i])) == sorted(list(label[0]))):
                    correct += 1
                else:
                    incorrect += 1
                i += 1

            return str((correct / total) * 100)

    def mostLikelyClass(slef, classified_labels):
        label_counts = []
        i = 0
        for label in classified_labels:
            label_counts.append([label, 0])
            for label_nested in classified_labels:
                if label == label_nested:
                    label_counts[i][1] += 1
            i += 1

        highest = ["", 0]
        for pair in label_counts:
            if (pair[1] > highest[1]):
                highest = pair

        return highest[0]

    def normalizeFeatureLengths(self, examples, unlabeled=None):
        min_features_length = self.getMinFeaturesLen(examples)

        if (min_features_length <= 3):
            print("Error not enough features in some drawing")
            return 0, 0

        # Delete x points represented as magnitudes (features) between each point
        # so that the total number of points is equal to the minimum number of points
        for i in range(len(examples)):
            current_points = examples[i]
            current_points_size = len(current_points)
            if (current_points_size % min_features_length != 0):
                n = 0
                j = 0
                jump = (current_points_size + (current_points_size % min_features_length)) // (current_points_size % min_features_length) + 1
                while (n < current_points_size):
                    del examples[i][n - j]
                    n += jump
                    j += 1

        # If new unlabelled example is passed in argument
        if (unlabeled):
            new_drawing_length = len(unlabeled)
            # If new unlabelled drawing has more features, normalize it too
            if (new_drawing_length > min_features_length):
                current_points_size = new_drawing_length
                if (current_points_size % min_features_length != 0):
                    n = 0
                    j = 0
                    jump = (current_points_size + (current_points_size % min_features_length)) // (current_points_size % min_features_length) + 1
                    while (n < current_points_size):
                        del unlabeled[n - j]
                        n += jump
                        j += 1
            return examples, unlabeled

        return examples

    def shiftList(self, list_in, n):
        '''Shift list_in by n'''
        return list_in[n:] + list_in[:n]
