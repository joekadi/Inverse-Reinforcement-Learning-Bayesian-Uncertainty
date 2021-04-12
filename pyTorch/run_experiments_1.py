
import os
import sys
import time

start_time = time.time()
for world in ['ow']:
    for num_paths in [128,256,512,1024]:
        for variant in [0,1,2]:
            os.system('python total_format_ensemble_results.py ' + str(variant) + ' ' + str(0.0) + ' ' + str(num_paths))
run_time = (time.time() - start_time)
print('All experiments ran in ' + str(run_time) + 'seconds')


