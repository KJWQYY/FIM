'''
Data pre process for FIM

'''
import numpy as np
import os

class LoadData(object):


    # Three files are needed in the path
    def __init__(self, path, dataset, loss_type="square_loss"):
        self.path = path + dataset + "/"
        self.trainfile = self.path + dataset +".train.libfm"
        self.testfile = self.path + dataset + ".test.libfm"
        self.validationfile = self.path + dataset + ".validation.libfm"
        #self.features_M = self.map_features( )
        self.features_M, self.field_M = self.map_features_fields()
        self.Train_data, self.Validation_data, self.Test_data = self.construct_data( loss_type )
    def get_field_index(self, lsts):
        result = []
        index = 0
        for lst in lsts:

            change_indices = []
            for i in range(1, len(lst)):
                if lst[i] != lst[i - 1]:
                    change_indices.append(i)
            if len(change_indices) != 4:
                print(change_indices)
                print(index)
            result.append(change_indices)
            index += 1
        return result
    def map_features_fields(self): # map the feature entries in all files, kept in self.features dictionary
        self.features = {}
        self.fields = {}
        self.read_features_fields(self.trainfile)
        self.read_features_fields(self.testfile)
        self.read_features_fields(self.validationfile)
        # print("features_M:", len(self.features))
        return len(self.features), len(self.fields)
    # def map_features(self): # map the feature entries in all files, kept in self.features dictionary
    #     self.features = {}
    #     self.read_features(self.trainfile)
    #     self.read_features(self.testfile)
    #     self.read_features(self.validationfile)
    #     # print("features_M:", len(self.features))
    #     return len(self.features)
    def read_features_fields(self, file): # read a feature file
        f = open( file )
        line = f.readline()
        i = len(self.features)
        j = len(self.fields)

        while line:
            items = line.strip().split(',') #1:2711:1
            for item in items[1:]:
                members = item.strip().split(':')
                field_index = members[0]
                feature_index = members[1]

                if field_index not in self.fields:
                    self.fields[ field_index ] = i
                    i = i + 1
                if feature_index not in self.features:
                    self.features[ feature_index ] = j
                    j = j + 1
            line = f.readline()
        f.close()

    def construct_data(self, loss_type):
        X_, X_f, Y_ , Y_for_logloss= self.read_data(self.trainfile)
        if loss_type == 'log_loss':
            Train_data = self.construct_dataset(X_, X_f, Y_for_logloss)
        else:
            Train_data = self.construct_dataset(X_, X_f, Y_)
        #print("Number of samples in Train:" , len(Y_))
        X_i = self.get_field_index(Train_data['X_f'])
        Train_data['X_i'] = X_i

        X_, X_f, Y_ , Y_for_logloss= self.read_data(self.validationfile)
        if loss_type == 'log_loss':
            Validation_data = self.construct_dataset(X_, X_f, Y_for_logloss)
        else:
            Validation_data = self.construct_dataset(X_, X_f, Y_)
        #print("Number of samples in Validation:", len(Y_))
        X_i = self.get_field_index(Validation_data['X_f'])
        Validation_data['X_i'] = X_i

        X_, X_f, Y_ , Y_for_logloss = self.read_data(self.testfile)
        if loss_type == 'log_loss':
            Test_data = self.construct_dataset(X_, X_f, Y_for_logloss)
        else:
            Test_data = self.construct_dataset(X_, X_f, Y_)
        #print("Number of samples in Test:", len(Y_))
        X_i = self.get_field_index(Test_data['X_f'])
        Test_data['X_i'] = X_i

        return Train_data,  Validation_data,  Test_data
    # def construct_data(self, loss_type):
    #     X_, Y_ , Y_for_logloss= self.read_data(self.trainfile)
    #     if loss_type == 'log_loss':
    #         Train_data = self.construct_dataset(X_, Y_for_logloss)
    #     else:
    #         Train_data = self.construct_dataset(X_, Y_)
    #     #print("Number of samples in Train:" , len(Y_))
    #
    #     X_, Y_ , Y_for_logloss= self.read_data(self.validationfile)
    #     if loss_type == 'log_loss':
    #         Validation_data = self.construct_dataset(X_, Y_for_logloss)
    #     else:
    #         Validation_data = self.construct_dataset(X_, Y_)
    #     #print("Number of samples in Validation:", len(Y_))
    #
    #     X_, Y_ , Y_for_logloss = self.read_data(self.testfile)
    #     if loss_type == 'log_loss':
    #         Test_data = self.construct_dataset(X_, Y_for_logloss)
    #     else:
    #         Test_data = self.construct_dataset(X_, Y_)
    #     #print("Number of samples in Test:", len(Y_))
    #
    #     return Train_data,  Validation_data,  Test_data

    def read_data(self, file):
        # read a data file. For a row, the first column goes into Y_;
        # the other columns become a row in X_ and entries are maped to indexs in self.features
        f = open( file )
        X_ = []
        X_f = []
        Y_ = []
        Y_for_logloss = []
        line = f.readline()
        while line:
            items = line.strip().split(',')
            Y_.append( 1.0*float(items[0]) )

            if float(items[0]) > 0:# > 0 as 1; others as 0
                v = 1.0
            else:
                v = 0.0
            Y_for_logloss.append( v )
            X_line = []
            X_f_line = []

            for item in items[1:]:
                members = item.strip().split(':')
                field_index = members[0]
                feature_index = members[1]
                X_line.append(self.features[feature_index])
                X_f_line.append(self.fields[field_index])
            X_.append(X_line)
            X_f.append(X_f_line)
            #X_.append( [ self.features[item] for item in items[1:]] )
            line = f.readline()
        f.close()
        return X_, X_f, Y_, Y_for_logloss

    def construct_dataset(self, X_, X_f, Y_):
        Data_Dic = {}
        X_lens = [ len(line) for line in X_]
        indexs = np.argsort(X_lens)
        Data_Dic['Y'] = Y_
        Data_Dic['X'] = X_
        Data_Dic['X_f'] = X_f

        return Data_Dic

    def padding_features(self,trandpad):
        """
        Make sure each feature vector is of the same length
        """


        #lenList.sort(reverse=True)
        num_variable = len(self.Train_data['X'][0])
        for i in range(len(self.Train_data['X'])):
            num_variable = max([num_variable, len(self.Train_data['X'][i])])
            #lenList.append(len(self.Train_data['X'][i]))
        for i in range(len(self.Validation_data['X'])):
            num_variable = max([num_variable, len(self.Validation_data['X'][i])])
            #lenList.append(len(self.Train_data['X'][i]))
        for i in range(len(self.Test_data['X'])):
            num_variable = max([num_variable, len(self.Test_data['X'][i])])
            #lenList.append(len(self.Train_data['X'][i]))
        #lenList.sort(reverse=True)
        field = -1
        feature = -1
        for i in range(len(self.Train_data['X'])):
            self.Train_data['X'][i] = self.Train_data['X'][i] + [self.features_M] * (num_variable - len(self.Train_data['X'][i]))
            self.Train_data['X_f'][i] = self.Train_data['X_f'][i] + [self.field_M] * (num_variable - len(self.Train_data['X_f'][i]))
        for i in range(len(self.Validation_data['X'])):
            self.Validation_data['X'][i] = self.Validation_data['X'][i] + [self.features_M] * (num_variable - len(self.Validation_data['X'][i]))
            self.Validation_data['X_f'][i] = self.Validation_data['X_f'][i] + [self.field_M] * (num_variable - len(self.Validation_data['X_f'][i]))
        for i in range(len(self.Test_data['X'])):
            self.Test_data['X'][i] = self.Test_data['X'][i] + [self.features_M] * (num_variable - len(self.Test_data['X'][i]))
            self.Test_data['X_f'][i] = self.Test_data['X_f'][i] + [self.field_M] * (num_variable - len(self.Test_data['X_f'][i]))



        num_variable_1 = len(self.Train_data['X_i'][0])
        maxvalue = 0
        for i in range(len(self.Train_data['X_i'])):
            num_variable_1 = max([num_variable_1, len(self.Train_data['X_i'][i])])
            #maxvalue = max(maxvalue, max(self.Train_data['X_i'][i]))
        for i in range(len(self.Validation_data['X_i'])):
            num_variable_1 = max([num_variable_1, len(self.Validation_data['X_i'][i])])
            #maxvalue = max(maxvalue, max(self.Train_data['X_i'][i]))
        for i in range(len(self.Test_data['X_i'])):
            num_variable_1 = max([num_variable_1, len(self.Test_data['X_i'][i])])
            #maxvalue = max(maxvalue, max(self.Train_data['X_i'][i]))

        for i in range(len(self.Train_data['X_i'])):
            self.Train_data['X_i'][i] = self.Train_data['X_i'][i]+ [num_variable-1] * (num_variable_1 - len(self.Train_data['X_i'][i]))

        for i in range(len(self.Validation_data['X_i'])):
            self.Validation_data['X_i'][i] = self.Validation_data['X_i'][i]+ [num_variable-1] * (num_variable_1 - len(self.Validation_data['X_i'][i]))

        for i in range(len(self.Test_data['X_i'])):
            self.Test_data['X_i'][i] = self.Test_data['X_i'][i]+ [num_variable-1] * (num_variable_1 - len(self.Test_data['X_i'][i]))


        return num_variable

        #return num_variable, num_variable_1
    def truncate_features(self):
        """
        Make sure each feature vector is of the same length
        """
        num_variable = len(self.Train_data['X'][0])
        for i in range(len(self.Train_data['X'])):
            num_variable = max([num_variable, len(self.Train_data['X'][i])])
        for i in range(len(self.Train_data['X'])):
            for j in range(0,num_variable-len(self.Train_data['X'][i])):
                self.Train_data['X'][i].append(0)
        for i in range(len(self.Validation_data['X'])):
            for j in range(0,num_variable-len(self.Validation_data['X'][i])):
                self.Validation_data['X'][i].append(0)

        for i in range(len(self.Test_data['X'])):
            for j in range(0,num_variable-len(self.Test_data['X'][i])):
                self.Test_data['X'][i].append(0)

        # for i in range(len(self.Train_data['X'])):
        #     num_variable = min([num_variable, len(self.Train_data['X'][i])])
        # truncate train, validation and test
        # for i in range(len(self.Train_data['X'])):
        #     self.Train_data['X'][i] = self.Train_data['X'][i][0:num_variable]
        # for i in range(len(self.Validation_data['X'])):
        #     self.Validation_data['X'][i] = self.Validation_data['X'][i][0:num_variable]
        # for i in range(len(self.Test_data['X'])):
        #     self.Test_data['X'][i] = self.Test_data['X'][i][0:num_variable]
        return num_variable
