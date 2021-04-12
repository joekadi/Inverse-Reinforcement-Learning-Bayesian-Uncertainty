
import os
import sys
import time

start_time = time.time()
for world in ['ow']:
    for num_paths in [64,128,256,512,1024]:
        os.system('python swag_train.py ' + str(0.0) + ' ' + str(num_paths))
        os.system('python swag_eval.py ' + str(0.0) + ' ' + str(num_paths))
run_time = (time.time() - start_time)
print('All experiments ran in ' + str(run_time) + 'seconds')




