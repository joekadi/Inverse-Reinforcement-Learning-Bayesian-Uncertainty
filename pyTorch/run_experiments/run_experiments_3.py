import os
import sys
import time
import _thread




start_time = time.time()
for world in ['ow']:
    for no_paths in [64,128,256,512,1024]:
        for variant in [0,1,2]:
            os.system('python total_format_matlab_results.py ' + str(variant) + ' ' + str(0.0) + ' ' + str(no_paths))
            os.system('python nf_format_matlab_results.py ' + str(variant) + ' ' + str(0.0) + ' ' + str(no_paths))
            os.system('python np_format_matlab_results.py ' + str(variant) + ' ' + str(0.0) + ' ' + str(no_paths))
run_time = (time.time() - start_time)
print('Experiments ran in ' + str(run_time) + 'seconds')
