
import os


#For objectworld
for no_paths in [8,16,24,32,40,48,56,64]:

    #Sample fresh paths
    os.system('python main.py ow ' + str(no_paths))

    #Train and eval regular models
    os.system('python train.py')
    os.system('python eval.py')

    #Train all NP models
    for percentile in range(0,3):
        os.system('python NP_train.py ' + str(percentile))
        os.system('python NP_eval.py ' + str(percentile))

    for features_index in range(0,4):
        os.system('python NF_train.py ' + str(features_index))
        os.system('python NF_eval.py ' + str(features_index))


