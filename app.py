#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#Imports
import cv2
import numpy
import time
import os
import sys
import math
import serial
import glob
from collections import defaultdict
from tensorflow import densenet
from sklearn import utils
from src.utils import *
from tensorflow.keras import *
from src.machineLearning import model_loader
from src.mecanical import density_balencer

from headers.OneCentroid import OneCentroid
from numpy import inf
import numpy as np
import pandas as pd
import math as math
from sklearn.metrics import pairwise_distances
from headers.utils import jensen_shannon_distance
from numpy.random import multivariate_normal

#Constantes
TRAINSET = "lbpcascade_frontalface.xml"    #Fichier de reconnaissance
IMAGE_SIZE = 170                           #Normalisation des images de base
NUMBER_OF_CAPTURE = 10                     #Nombre de captures a realiser pour la base de donnees
THRESHOLD = 90                                 #Seuil de reconnaissance
CAMERA = 0                                 #La camera
ARDUINO = False                            #Utiliser l'arduino ?

INDIVIDUS = []

#####################################
def sendSerial(ser, command):
    """Envoie command a l'arduino"""
    if(ARDUINO):
        ser.write(command)


class Node:

    def __init__(self, parent, is_leaf, min_samples_leaf, max_features, distance='euclidean', root_distance='mahalanobis', method_subclasses='sqrt', method_split='alea', root_class=None, alpha=0.95, mode_mean=None, mode_cov=None, mode_weight=None, mode_indexes=None, localdepth=None):
        """
        
        :param parent: object node() root node
        :param is_leaf: boolean,
        :param min_samples_leaf: int(), min samples to split into leaves
        :param max_features: int()
        :param distance: string()
        :param method_subclasses: string(), mathematical method choosed between { sqrt, log2, ...}
        :param method_split:string(), split method to split node in leaves
        :param root_class: int(), the class to isolate in case of a generative node
        :param alpha: int()
        """
        self.left_child = None
        self.right_child = None
        self.parent = parent
        self.splitting_clf = None
        self.left_subclasses = None
        self.right_subclasses = None
        self.is_leaf = is_leaf
        self.min_samples_leaf = min_samples_leaf
        self.distance = distance
        self.root_distance = root_distance
        self.method_subclasses = method_subclasses
        self.method_split = method_split
        self.root_class = root_class
        self.alpha = alpha
        self.mode_mean = mode_mean
        self.mode_cov = mode_cov
        self.mode_weight = mode_weight
        self.mode_indexes = mode_indexes
        self.max_features = max_features
        self.majority_class = None
        self.proportion_classes = None
        self.total_effectives = 0
        self.localdepth=localdepth

    def fit(self, X, y):
        """
        fit function: give and train data in each node created

        :param X: numpy.ndarray() input data to fit the node
        :param y: numpy.ndarray() input label to fit the node
        """
        k = np.unique(y)

        min_classes = 3
        if self.method_subclasses == 'sqrt':
            nb_subclasses = max(round(math.sqrt(len(k))), min_classes)
        elif self.method_subclasses == 'log2':
            nb_subclasses = max(round(math.log2(len(k))), min_classes)
        elif type(self.method_subclasses) == float:
            nb_subclasses = max(round(self.method_subclasses * (len(k))), min_classes)
        elif type(self.method_subclasses) == int:
            nb_subclasses = max(self.method_subclasses, min_classes)
        else:
            nb_subclasses = len(k)

        if len(k) <= min_classes or len(k) <= nb_subclasses:
            subclasses = k
        else:
            subclasses = np.random.choice(k, round(nb_subclasses), replace=False)

        sub_features = np.random.choice(np.arange(0, X.shape[1]), int(self.max_features), replace=False)
        idx_subclasses = np.where(np.in1d(y, subclasses))
 
        X_subclasses = X[np.array(idx_subclasses[0]), :]  # Recuperation de l'echantillon
        y_subclasses = y[idx_subclasses]

        self.splitting_clf = NCMClassifier(metric=self.distance, sub_features=sub_features)
        self.splitting_clf.fit(X_subclasses, y_subclasses)
        self.update_statistics(y)
        if len(k) <= 1:
            # When single leaf
            self.is_leaf = True
        else:
            # Generative node, fit classifier to isolate one class and send it to the left child, send the rest to the right child.
            if self.method_split == 'generative':
                self.splitting_clf = OneCentroid(root_class = self.root_class, mode_mean = self.mode_mean, mode_cov = self.mode_cov, mode_weight = self.mode_weight, mode_indexes = self.mode_indexes, alpha=self.alpha, distance=self.root_distance)
                self.splitting_clf.fit(X, y)
                self.left_subclasses = 1
                self.right_subclasses = 0
                
            # split randomly subclasses
            elif self.method_split == 'alea':
                np.random.shuffle(subclasses)
                self.left_subclasses, self.right_subclasses = np.array_split(subclasses, 2)

            # put the class with the most sample in the left child and the rest in the right child
            elif self.method_split == 'maj_class':
                filtered_proportions = pd.Series(self.proportion_classes).loc[subclasses]
                self.left_subclasses = filtered_proportions.idxmax()
                self.right_subclasses = subclasses[subclasses != self.left_subclasses]

            # dispatch with the same sample size between the two child
            elif self.method_split == 'eq_samples':
                nb_left_samples = 0
                nb_right_samples = 0
                self.left_subclasses = np.array([], dtype=int)
                self.right_subclasses = np.array([], dtype=int)
                filtered_proportions = pd.Series(self.proportion_classes).loc[subclasses].sort_values(ascending=False)
                for key, value in filtered_proportions.items():
                    if nb_left_samples <= nb_right_samples:
                        self.left_subclasses = np.append(self.left_subclasses, key)
                        nb_left_samples += value
                    else:
                        self.right_subclasses = np.append(self.right_subclasses, key)
                        nb_right_samples += value

            # separating the centroids from the most distant classes then group the rest by the neareast centroid
            elif self.method_split == 'farthest_min':
                matrix_distance = pd.DataFrame(pairwise_distances(self.splitting_clf.centroids_), self.splitting_clf.classes_,
                                               self.splitting_clf.classes_).replace(0, np.nan)
                result = matrix_distance.max().sort_values(ascending=False)[:2].index.to_list()

                all = list(subclasses.copy())
                self.left_subclasses = np.array([])
                self.right_subclasses = np.array([])
                self.left_subclasses = np.append(self.left_subclasses, result[0])
                all.remove(result[0])
                self.right_subclasses = np.append(self.right_subclasses, result[1])
                all.remove(result[1])

                for i in range(len(all)):
                    classe = all[i]
                    nearest_class = matrix_distance[classe][[result[0], result[1]]].idxmin()
                    if nearest_class in self.left_subclasses:
                        self.left_subclasses = np.append(self.left_subclasses, classe)
                    else:
                        self.right_subclasses = np.append(self.right_subclasses, classe)

            # separating the centroids from the most distant classes then group the rest by the farthest centroid
            elif self.method_split == 'farthest_max':
                matrix_distance = pd.DataFrame(pairwise_distances(self.splitting_clf.centroids_), self.splitting_clf.classes_,
                                               self.splitting_clf.classes_).replace(0, np.nan)
                result = matrix_distance.max().sort_values(ascending=False)[:2].index.to_list()

                all = list(subclasses.copy())
                self.left_subclasses = np.array([])
                self.right_subclasses = np.array([])
                self.left_subclasses = np.append(self.left_subclasses, result[0])
                all.remove(result[0])
                self.right_subclasses = np.append(self.right_subclasses, result[1])
                all.remove(result[1])

                for i in range(len(all)):
                    classe = all[i]
                    nearest_class = matrix_distance[classe][[result[0], result[1]]].idxmax()
                    if nearest_class in self.left_subclasses:
                        self.left_subclasses = np.append(self.left_subclasses, classe)
                    else:
                        self.right_subclasses = np.append(self.right_subclasses, classe)
            else:
                print("Method not defined.")

    def predict_split(self, X):
        """
        predict_split(X): return the index of data to the left or to the right
        :param X:numpy.ndarray() input data
        :return:
            - is_splittable: boolean
            - left_indexes:  numpy.ndarray()
            - right_indexes  numpy.ndarray()
        """
        is_splittable, left_indexes, right_indexes = False, None, None
        if not self.is_leaf:
            predictions = self.predict_splitting_function(X)
            left_indexes = np.where(np.in1d(predictions, self.left_subclasses))
            right_indexes = np.where(np.in1d(predictions, self.right_subclasses))

            # --------------------------- stopping criterion ( and in NCMTree.py  fct() build nodes)-------------------------------
            if len(left_indexes[0]) > self.min_samples_leaf and len(
                    right_indexes[0]) > self.min_samples_leaf:  # Critere d'arrêt : gini?
                is_splittable = True
        return is_splittable, left_indexes, right_indexes

    def predict_all(self, X):
        """
        :param X: numpy.ndarray()
        :return: pd.Series(), return proba of each classes.
        """
        index_array = np.array(np.arange(len(X)))  # store real indexes for retrieval of propagated samples
        pred = self.predict_propagate_proba(X, index_array)
        return pd.Series(pred).sort_index(ascending=True)

    def predict_propagate_proba(self, X, index_array):
        """
        :param X: numpy.ndarray()
        :param index_array:
        :return:
        """
        if not self.get_is_leaf():  # recursive tree descent
            prediction = self.predict_splitting_function(X)
            decision = np.isin(prediction, self.left_subclasses)
            left = np.where(decision)
            right = np.where(decision == False)
            if len(left[0]) > 0:
                left_prediction = self.left_child.predict_propagate_proba(X[left], index_array[left])
            else:
                left_prediction = {}
            if len(right[0]) > 0:
                right_prediction = self.right_child.predict_propagate_proba(X[right], index_array[right])
            else:
                right_prediction = {}
            left_prediction.update(right_prediction)
            return left_prediction
        else:
            return {x: self.proportion_classes for x in index_array}  # example : { 12:{ "1": 0.93, "4":0.07 } }
            # probability of each classes
            


    def ULS(self, X, y):
        """
        :param X: numpy.ndarray(), new samples for incremental learning
        :param y: numpy.ndarray(), new labels for incremental learning
        """
        if not self.get_is_leaf():
            prediction = self.predict_splitting_function(X)
            decision = np.isin(prediction, self.left_subclasses)
            left = np.where(decision)
            right = np.where(decision == False)
            if len(left[0]) > 0:
                self.left_child.ULS(X[left], y[left])

            if len(right[0]) > 0:
                self.right_child.ULS(X[right], y[right])
        else:
            self.update_statistics(y)

    def IGT(self, X, y, jensen_threshold=0.1, recreate=True):
        """

        """
        if not self.get_is_leaf():
            prediction = self.predict_splitting_function(X)
            decision = np.isin(prediction, self.left_subclasses)
            left = np.where(decision)
            right = np.where(decision == False)
            splittable_leaves = []

            if len(left[0]) > 0:
                left_prediction = self.left_child.IGT(X[left], y[left])
                splittable_leaves = splittable_leaves + left_prediction

            if len(right[0]) > 0:
                right_prediction = self.right_child.IGT(X[right], y[right])
                splittable_leaves = splittable_leaves + right_prediction
            return splittable_leaves
        else:
            old_class = self.majority_class
            old_distrib = pd.Series(self.proportion_classes)
            self.update_statistics(y)
            new_class = self.majority_class
            new_distrib = pd.Series(self.proportion_classes)
            old_distrib = old_distrib.reindex_like(new_distrib).fillna(0)  # same index classes for jensen shannon distance calculation

            # --------------------- IGT activation criteria --------------------
            if old_class != new_class \
                    or jensen_shannon_distance(old_distrib, new_distrib) > jensen_threshold \
                    or self.total_effectives > self.min_samples_leaf * 3:  
                X_full = []
                y_full = []

                # data generation of each classes
                for index, k in enumerate(self.splitting_clf.classes_):
                    cov_vector = np.round(1 / (self.splitting_clf.inv_cov_vectors[index] + 1e-8), 3)
                    cov_vector[cov_vector == inf] = 1
                    X_gen = multivariate_normal(self.splitting_clf.centroids_[index],
                                                np.diag(cov_vector),
                                                int(self.splitting_clf.nk[index]))
                    y_gen = np.repeat(k, int(self.splitting_clf.nk[index]))
                    X_full.extend(X_gen)
                    y_full.extend(y_gen)
                X_full.extend(X)
                y_full.extend(y)
                X_full = np.array(X_full)
                y_full = np.array(y_full)
                if recreate:  # refit entirely or not (left/right classes & subfeatures)
                    self.splitting_clf = None
                    self.majority_class = None
                    self.proportion_classes = None
                    self.fit(X_full, y_full)
                else:
                    self.splitting_clf.fit(X_full, y_full)
                return [(self, X_full, y_full)]  # return the pointer of node and its generated + incremental data
            else:
                return []

    def get_all_sizes(self):
        """
        get size of each child nodes
        """
        if not self.get_is_leaf():
            left_size = self.left_child.get_all_sizes()
            right_size = self.right_child.get_all_sizes()
            left_size.update(right_size)
            left_size.update({self: self.size()})
            return left_size
        else:
            return {}

    def get_child_nodes(self):
        if not self.get_is_leaf():
            left_size = self.left_child.get_child_nodes()
            right_size = self.right_child.get_child_nodes()
            left_size.extend(right_size)
            left_size.extend([self])
            return left_size
        else:
            return [self]

    def predict_splitting_function(self, X):
        """
        :param X:
        :return:
        """
        return self.splitting_clf.predict(X)

    def update_statistics(self, y):
        """
        update probabilities of each classes in the node and store the total effectives
        """
        # if node has never been fitted
        if self.majority_class is None:
            self.total_effectives = len(y)
            proportions = pd.Series(y).value_counts()
            self.majority_class = proportions.index[0]
            self.proportion_classes = (np.round(proportions / proportions.sum(axis=0), 3)).to_dict()

        # else update for incremental learning
        else:
            old_proportions = pd.Series(self.proportion_classes) * self.total_effectives
            self.total_effectives = self.total_effectives + len(y)

            new_proportions = old_proportions.add(pd.Series(y).value_counts(), fill_value=0).sort_values(ascending=False)
            self.majority_class = new_proportions.index[0]
            self.proportion_classes = np.round(new_proportions / new_proportions.sum(axis=0), 3).to_dict()

    def depth(self):
        return max(self.left_child.depth() if self.left_child else 0,
                   self.right_child.depth() if self.right_child else 0) + 1

    def size(self):
        if self.is_leaf:
            return 1
        else:
            return self.left_child.size() + self.right_child.size() + 1

    def get_left_child(self):
        return self.left_child

    def get_right_child(self):
        return self.right_child

    def get_parent(self):
        return self.parent

    def get_is_leaf(self):
        return self.is_leaf

    def set_left_child(self, left_child):
        self.left_child = left_child

    def set_right_child(self, right_child):
        self.right_child = right_child

    def set_parent(self, parent):
        self.parent = parent

    def set_leaf(self, is_leaf):
        self.is_leaf = is_leaf

    def get_cardinality(self):
        """

        :return: 1, if it's a leaf or give numbers of child below the current node. int()
        """
        # TO CHECK
        if self.get_left_child().get_is_leaf():
            return 1
        else:
            return self.get_left_child().get_cardinality() + self.get_right_child().get_cardinality()




class BuildDataStorage():
    def __init__(self, imgPath ,ident):
        self.rval = False            
        self.camera = cv2.VideoCapture(CAMERA)
        self.classifier = cv2.CascadeClassifier(TRAINSET)
        self.faceFrame = None
        self.identity = ident
        self.imagesPath = imgPath

    def getfaceframe(self, frame):
        """Retourne la position des visages detectes de la forme [[x y w h]]"""
        faces = self.classifier.detectMultiScale(frame)
        return faces

    def facedrawer(self, frame, faces):
        """Dessine un rectangle autour du visage detecte"""
        for f in faces: 
            x,y,w,h = [v for v in f]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,140,255))
            self.LBPHBaseImage = self.faceframer(frame, x, y, w, h)
        return frame

    def faceframer(self, frame, x, y, w, h):
        """On recupere un rectangle (largeur, hauteur) (centreX, centreY)"""
        cropped = cv2.getRectSubPix(frame, (w, h), (x + w / 2, y + h / 2))
        grayscale = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        self.faceFrame = cv2.resize(grayscale, (IMAGE_SIZE, IMAGE_SIZE))
        return self.faceFrame

    def faceframecollecter(self, frame):
        """On enregistre le visage recupere"""     
        imageCreated = False
        captureNum = 0
        #Cree le dossier s'il n'existe pas
        try:
            os.makedirs("{0}/{1}".format(self.imagesPath, self.identity))
        except OSError:
            print("ecriture dans dossier existant") 
        #Cree l'image a la suite
        while not imageCreated:
            if not os.path.isfile("{0}/{1}/{2}.jpg".format(self.imagesPath, self.identity, captureNum)):
                cv2.imwrite("{0}/{1}/{2}.jpg".format(self.imagesPath, self.identity, captureNum), frame)
                imageCreated = True
            else:
                captureNum += 1

    def capture(self): 
        """Recupere le flux video"""       
        if self.camera.isOpened():
            (rval, frame) = self.camera.read()
        else:
            rval = False

        while rval:
            (rval, frame) = self.camera.read()
            frame = self.facedrawer(frame, self.getfaceframe(frame))
            #Affichage du texte
            cv2.putText(frame, "Appuyez sur c pour collecter", (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0))
            cv2.imshow("Creation de la BDD", frame)
            key = cv2.waitKey(20)
            if key in [27, ord('Q'), ord('q')]: #esc / Q
                break
            if key in [ord('C'), ord('c')] and self.faceFrame != None: 
                self.faceframecollecter(self.faceFrame)

class Recognize():
    def __init__(self, imgPath):
        self.rval = False            
        self.camera = cv2.VideoCapture(CAMERA)
        self.classifier = cv2.CascadeClassifier(TRAINSET)
        self.faceFrame = None
        self.identities = []
        self.imagesPath = imgPath
        self.images = []
        self.imagesIndex = []
        self.time = time.time()

        self.eyeWide = 0
        self.eyeHeight = 0
        self.grayMouthClosed = 0
        self.thresholdEyeClosed = 0

    def getfaceframe(self, frame):
        """Retourne la position des visages detectes de la forme [[x y w h]]"""
        faces = self.classifier.detectMultiScale(frame)
        return faces

    def draw(self, frame, detected, color):
        """Dessine un rectangle autour du visage detecte"""
        if detected is None:
            return frame
        for d in detected: 
            x,y,w,h = [v for v in d]
            cv2.rectangle(frame, (x,y), (x+w, y+h), color)
        return frame

    def faceframer(self, frame, x, y, w, h):
        """On recupere un rectangle (largeur, hauteur) (centreX, centreY)"""
        cropped = cv2.getRectSubPix(frame, (w, h), (x + w / 2, y + h / 2))
        grayscale = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        self.faceFrame = cv2.resize(grayscale, (IMAGE_SIZE, IMAGE_SIZE))
        return self.faceFrame

    def lbphframe(self, croppedFrame):
        """Retourne la position des bouches detectes de la forme [[x y w h]]"""        
        cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
        rects = cascade.detectMultiScale(croppedFrame)        
        if len(rects) == 0:
            return rects
        final = None   
        x1 = 0
        x2 = 0 + len(croppedFrame)
        y1 = 0
        y2 = 0 + len(croppedFrame[0])*1/2
        
        #Prend la partie inferieure de la tete pour le traitement
        for rect in rects:
            if rect[0] > x1 and rect[0] + rect[2] < x2 and rect[1] > y1 and rect[1] + rect[3] < y2:
                if final is None:
                    final = [rect]
                else:
                    final += [rect]
        return final

    def lbphfaceZ(self, croppedFrame):
        """Retourne la position des bouches detectes de la forme [[x y w h]]"""        
        cascade = cv2.CascadeClassifier('mouth_classifier.xml')
        rects = cascade.detectMultiScale(croppedFrame)        
        if len(rects) == 0:
            return rects
        final = None   
        x1 = 0
        x2 = 0 + len(croppedFrame)
        y1 = 0 + len(croppedFrame[0])*5/8
        y2 = 0 + len(croppedFrame[0])

        #Prend la partie inferieure de la tete pour le traitement
        for rect in rects:
            if rect[0] > x1 and rect[0] + rect[2] < x2 and rect[1] > y1 and rect[1] + rect[3] < y2:
                if final is None:
                    final = [rect]
                else:
                    final += [rect]
        return final
   
    def extractAndResize(self, frame, x, y, w, h):
        """On recupere juste la tete en noir et blanc"""
        cropped = cv2.getRectSubPix(frame, (w, h), (x + w / 2, y + h / 2))
        grayscale = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(grayscale, (IMAGE_SIZE, IMAGE_SIZE))
        return resized

    def cropFromFace(self, frame, facePos):
        """Garde seulement la partie "tete" de la frame"""
        #X,Y,W,H
        if facePos is None:
            return frame
        if len(facePos) == 0 :
            return frame
        else :
            x1 = facePos[0][0]
            x2 = x1 + facePos[0][2]
            y1 = facePos[0][1]
            y2 = y1 + facePos[0][3]
            return frame[y1:y2, x1:x2]

    def readImages(self):
        """Recupere les images de bases pour effectuer la reconnaissance des visages"""
        c = 0
        self.images = []
        self.imagesIndex = []
        for dirname, dirnames, filenames in os.walk(self.imagesPath):
            for subdirname in dirnames:
                self.identities.append(subdirname)
                subject_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(subject_path):
                    try:
                        im = cv2.imread(os.path.join(subject_path, filename), 0)
                        self.images.append(numpy.asarray(im, dtype=numpy.uint8))
                        self.imagesIndex.append(c)
                    except IOError, (errno, strerror):
                       print "I/O error({0}): {1}".format(errno, strerror)
                    except:
                        print "Unexpected error:", sys.exc_info()[0]
                        raise
                c += 1 

    def recognizeLBPHFace(self):
        """Reconnait par la methode LBPH"""
        self.model = cv2.createLBPHFaceRecognizer()        
        self.model.train(numpy.asarray(self.images), numpy.asarray(self.imagesIndex))

    def recognize(self):
        """On choisit la methode de reconnaissance et on construit la base de donnee"""
        self.readImages()
        self.recognizeLBPHFace()
        if not self.camera.isOpened():
            return
        self.capture()
    
    def identify(self, image):
        """On reconnait l'identite de la personne si enregistree"""
        [p_index, p_confidence] = self.model.predict(image)
        found_identity = self.identities[p_index]
        return found_identity, p_confidence

    def initNeutral(self, neutralImg):
        """Initialise les thresholds + les largeurs/hauteurs pour la detection des emotions"""
        frame = neutralImg
        facePos = self.getfaceframe(frame)
        cropped = self.cropFromFace(frame, facePos)
        mouthPos = self.lbphfaceZ(cropped)
        mouthFrame = self.cropFromFace(frame, mouthPos)        
        gray = cv2.cvtColor(mouthFrame, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
        self.grayMouthClosed = numpy.count_nonzero(thresh)
        eyePos = self.lbphframe(cropped)
        eyeFrame = self.cropFromFace(frame, eyePos)
        hsv = cv2.cvtColor(eyeFrame, cv2.COLOR_BGR2HSV)
        lowerColor = numpy.array([20, 0,0])
        upperColor = numpy.array([160,200,200])
        mask = cv2.inRange(hsv, lowerColor, upperColor)
        self.thresholdEyeClosed = numpy.count_nonzero(mask)
        self.eyeWide = eyePos[0][2]
        self.eyeHeight = eyePos[0][3]
        return 0

    
    def isMouthOpen(self, mouthFrame):
        gray = cv2.cvtColor(mouthFrame, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
        return self.grayMouthClosed < numpy.count_nonzero(thresh)-300
        

    def EyeNotHeightThanNeutral(self, EyeHeight, threshold):
        """Determine si les yeux sont fronces"""
        return EyeHeight < self.eyeHeight - threshold

    def EyeHeightThanNeutral(self, EyeHeight, threshold):
        """Determine si les yeux sont equarquilles"""
        return EyeHeight > self.eyeHeight + threshold

    def isEyeClosed(self, eyeFrame):
        hsv = cv2.cvtColor(eyeFrame, cv2.COLOR_BGR2HSV)
        lowerColor = numpy.array([20, 0,0])
        upperColor = numpy.array([160,200,200])
        mask = cv2.inRange(hsv, lowerColor, upperColor)
        return self.thresholdEyeClosed*2 < numpy.count_nonzero(mask)

   
    def emotions(self):
        """Recupere le flux video"""
        interval = 0
        dontlook = 0
        sleep = 0
        mouthOpen = 0
        eyeClose = 0
        eyeBigger = 0
        eyeNotBigger = 0
        error = 0
        if self.camera.isOpened():
            (rval, frame) = self.camera.read()
        else:
            rval = False
        i = 0
        while rval:
            (rval, frame) = self.camera.read()
            facePos = self.getfaceframe(frame)
            if len(facePos) is 0 or facePos is None:
                dontlook += 1
                if dontlook % 20 is 0:
                    print('Conducteur inattentif')
                    sendSerial(ser,'s')
            else:
                dontlook = 0
                if i < 10:
                    i += 1
                if i is 10:
                    self.initNeutral(frame)
                    i += 1
                frame = self.draw(frame, facePos, (0,140,255))
                cropped = self.cropFromFace(frame, facePos)
                eyePos = self.lbphframe(cropped)
                sendSerial(ser,'d')
                if eyePos is None:
                    #print('yeux fermes ou yeux non detectes')
                    sleep += 1
                    error = 1
                elif error is not 1:
                    sleep = 0
                    sendSerial(ser,'d')
                    sendSerial(ser,'r')
                else:
                    error = 0
                if sleep > 3:
                    print('Endormi')
                    sendSerial(ser,'w')
                    sendSerial(ser,'s')
                if eyePos is not None and i > 10:
                    cropped = self.draw(cropped, eyePos, (255,0,255))
                    eyeFrame = self.cropFromFace(frame, eyePos)
                    if self.isEyeClosed(eyeFrame):
                        #print('yeux fermes ou yeux non detectes')
                        sleep += 1
                        error = 1
                    elif error is not 1:
                        sleep = 0
                    else:
                        error = 0
                    if len(eyePos) > 0 and self.EyeHeightThanNeutral(eyePos[0][3], 5):
                        #print('plus grand')
                        eyeBigger += 1
                        error = 1
                    elif error is not 1:
                        eyeBigger = 0
                    else:
                        error = 0
                    if len(eyePos) > 0 and self.EyeNotHeightThanNeutral(eyePos[0][3], 4):
                        #print('plus petit')
                        eyeNotBigger += 1
                        error = 1
                    elif error is not 1:
                        eyeNotBigger = 0
                    else:
                        error = 0
                mouthPos = self.lbphfaceZ(cropped)
                cropped = self.draw(cropped, mouthPos, (0,0,255))
                if mouthPos is not None and self.isMouthOpen(self.cropFromFace(frame, mouthPos)) and i > 10:
                    #print('bouche ouverte')
                    mouthOpen += 1
                    error = 1
                elif not self.isMouthOpen(self.cropFromFace(frame, mouthPos)) and error is not 1:
                    moutOpen = 0
                else:
                    error = 0
        
                if mouthOpen > 5 and eyeBigger > 5:
                    print('Surpris')
                    sendSerial(ser,'b')
                else:                    
                    sendSerial(ser,'n')
                if eyeNotBigger > 5:
                    print('Enerve')
                    sendSerial(ser,'a')
                else:                    
                    sendSerial(ser,'q')
                #if mouthOpen <= 5 and eyeNotBigger > 5:
                #    print('enerve')
                #    sendSerial(ser,'a')
                cv2.imshow("Tete", cropped)
            cv2.imshow("TIPEAPS", frame)
            key = cv2.waitKey(20)
            if key in [27, ord('Q'), ord('q')]:
                break

    def capture(self): 
        """Recupere le flux video"""
        interval = 0
        if self.camera.isOpened():
            (rval, frame) = self.camera.read()
        else:
            rval = False
        i = 0
        while i < 55:
            i+=1
            (rval, frame) = self.camera.read()
            self.time = time.time()
            facePos = self.getfaceframe(frame)
            frame = self.draw(frame, facePos, (0,140,255))
            for f in facePos: 
                x,y,w,h = [v for v in f]
                resized = self.extractAndResize(frame, x, y, w, h)
                identity, confidence = self.identify(resized)
                if confidence > THRESHOLD:
                    identity = "INCONNU"
                INDIVIDUS.append(identity)
                print(identity + " --- " + str(confidence))
                cv2.putText(frame, "%s (%s)"%(identity, confidence), (x,y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,140,255))
            cv2.imshow("TIPEAPS", frame)
            key = cv2.waitKey(20)
            if key in [27, ord('Q'), ord('q')]: #esc / Q
                break

def getSerialName():
    """Retourne le fichier correspondant a l'arduino"""
    serialName = '/dev/null'
    osname = sys.platform.lower()
    if 'darwin' in osname: #si mac OS X
        for tty in glob.glob('/dev/tty*'):
            if 'usbmodem' in tty:
                serialName = tty
    elif 'linux' in osname: #si linux
        for tty in glob.glob('/dev/tty*'):
            if 'ACM' in tty:
                serialName = tty
    return serialName
