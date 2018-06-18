#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#
# PROGRAMMER:   Julian Kleinz
# DATE CREATED: 06/14/2018
# REVISED DATE: 06/18/2018 <=(Date Revised - if any)
# REVISED DATE: 05/14/2018 - added import statement that imports the print
#                           functions that can be used to check the lab
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from time import time, sleep
from os import listdir
from os import getcwd

# Imports classifier function for using CNN to classify images
from classifier import classifier

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Main program function defined below
def main():

    # collecting start time
    start_time = time()

    # 2. Define get_input_args() function to create & retrieve command
    # line arguments
    in_args = get_input_args()
    print("Command Line Arguments:\n    dir =", in_args.dir,
          "\n    arch =", in_args.arch,
          "\n    dogfile =", in_args.dogfile)
    path = getcwd() + "\\" + in_args.dir

    # 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels(path)

    check_creating_pet_image_labels(answers_dic)

    # 4. Define classify_images() function to create the classifier
    # labels with the classifier function uisng in_arg.arch, comparing the
    # labels, and creating a dictionary of results (result_dic)

    #dictionary to store results: pet image label, classifier label,
    #comparison of labels
    result_dic = classify_images(path, answers_dic, "vgg")

    check_classifying_images(result_dic)

    # TODO: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog()

    # TODO: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats()

    # TODO: 7. Define print_results() function to print summary results,
    # incorrect classifications of dogs and breeds if requested.
    print_results()
    #sleep(5)
    # collecting end time
    end_time = time()

    # Define overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    #print(str(int(((tot_time % 3600) % 60))).zfill(2))
    print("\n** Total Elapsed Runtime:", tot_time, "in seconds")
    print("\n** Total Elapsed Runtime:", str(int((tot_time / 3600))).zfill(2) +
          ":" + str(int(((tot_time % 3600) / 60))).zfill(2) +
          ":" + str(int(((tot_time % 3600) % 60))).zfill(2))



# TODO: 2.-to-7. Define all the function below. Notice that the input
# paramaters and return values have been left in the function's docstrings.
# This is to provide guidance for acheiving a solution similar to the
# instructor provided solution. Feel free to ignore this guidance as long as
# you are able to acheive the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
     3 command line arguements are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    #Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    #Argument 1: that's the path to pet_images
    parser.add_argument('--dir', type=str, default="pet_images/",
                        help='path to the folder my_folder')

    #Argument 2: that's an the chosen CNN model
    parser.add_argument('--arch', type=str, default="vgg",
                        help='chosen model')

    #Argument 3: that's the chosen file that contains dog labels
    parser.add_argument('--dogfile', type=str, default="dognames.txt",
                        help='text file including dognames')
    return parser.parse_args()


def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image
    files. Reads in pet filenames and extracts the pet image labels from the
    filenames and returns these label as petlabel_dic. This is used to check
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)
    """
    # Get filenames from image_dir folder

    pet_labels_dic = {}
    filename_list = listdir(image_dir)

    for idx in filename_list:
        words = idx.lower().split('_')
        pet_image = idx
        pet_name = ""
        for word in words:
            if word.isalpha():
                pet_name += word + " "
        pet_name = pet_name.strip()
        if pet_image not in pet_labels_dic:
            pet_labels_dic.update({pet_image : pet_name})
        else:
            print("** Warning: Key=", pet_image,
                  "already exists in pet_dic with value =", pet_name)

    return pet_labels_dic


def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the
     classifier() function to classify images in this function.
     Parameters:
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its' key is the
                     pet image filename & it's value is pet image label where
                     label is lowercase with space between each word in label
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and
                    classifer labels and 0 = no match between labels
    """

    # Creates dictionary that will have all the results key = filename
    # value = list [Pet Label, Classifier Label, Match(1=yes,0=no)]
    results_dic = {}

    for key in petlabel_dic:

        # Runs classifier function to classify the images classifier function
        # inputs: path + filename  and  model, returns model_label
        # as classifier label
        model_label = classifier(images_dir+key, model).lower().strip()

        # defines truth as pet image label and trys to find it using find()
        # string function to find it within classifier label(model_label).
        # Put separate terms that 'may' compose the classifier label into a list so
        # that each term is an item in the list.
        model_label_list = model_label.split(", ")

        # defines truth as pet image label
        truth = petlabel_dic[key]

        # If the pet image label is found within the classifier label list of terms
        # as an exact match to on of the terms in the list - then they are added to
        # results_dic as an exact match
        if truth in model_label_list:
            results_dic[key] = [truth, model_label, 1]

        # For those that aren't an exact term match to a term - checks if the pet_label
        # is part of the term like: "poodle" matching to "standard poodle" OR
        # "cat" matching to "tabby cat"
        else:
            # Sets found to False - IF pet image label is FOUND as part of a term within
            # the list of classifier label terms then will be set to True
            found = False

            # For loop to iterate through each term from model_label_list - splitting the
            # the term into words where truth is compare to each word to see if there is
            # a match - if so they are added to results_dic as a match and found = True
            # and searching through the for loop is terminated using the break
            for term in model_label_list:

                # splits the term into a word list using split()
                word_list = term.split(" ")

                # if the pet image label hasn't been found AND it exists in the word list
                # like 'poodle' in ['standard', 'poodle'] or 'cat' in ['tabby', 'cat']
                # then found = True, the results are added to results_dic and break is
                # used to break out of the for loop since a match was found
                if (not found) and truth in word_list:
                    found = True
                    results_dic[key] = [truth, model_label, 1]
                    break

            # If pet image label isn't found within the terms that exist in the list of labels
            # the classifier function produces then set match = 0 (not a match)
            if not found:
                results_dic[key] = [truth, model_label, 0]

    #Return results dictionary
    return results_dic


def adjust_results4_isadog():
    """
    Adjusts the results dictionary to determine if classifier correctly
    classified images 'as a dog' or 'not a dog' especially when not a match.
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line
                dog names are all in lowercase with spaces separating the
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    pass


def calculates_results_stats():
    """
    Calculates statistics of the results of the run using classifier's model
    architecture on classifying images. Then puts the results statistics in a
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
    """
    pass


def print_results():
    """
    Prints summary results on the classification and then prints incorrectly
    classified dogs and incorrectly classified dog breeds if user indicates
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and
                             False doesn't print anything(default) (bool)
      print_incorrect_breed - True prints incorrectly classified dog breeds and
                              False doesn't print anything(default) (bool)
    Returns:
           None - simply printing results.
    """
    pass



# Call to main function to run the program
if __name__ == "__main__":
    main()
