
import os
import sys
import time
import _thread

start_time = time.time()

for world in ['ow']:
    for no_paths in [128]:

         #Eval
        os.system('python eval.py 0.0 ' + str(no_paths))
        os.system('python eval.py 0.2 ' + str(no_paths))
        os.system('python eval.py 0.4 ' + str(no_paths))
        os.system('python eval.py 0.6 ' + str(no_paths))
        os.system('python eval.py 0.8 ' + str(no_paths))


                      
run_time = (time.time() - start_time)
print('Experiments ran in ' + str(run_time) + 'seconds')


'''

from subprocess import Popen
if get_results:
    for world in ['ow']:
        for no_paths in [24, 48, 64]:

            #Sample fresh paths
            command = ['python', 'main.py', world, no_paths]
            commands.append(command)

            for dropout_val in [0.0, 0.2, 0.4, 0.8]:
                
                #Train and eval regular models
                command = ['python', 'train.py', dropout_val]
                commands.append(command)
                command = ['python', 'eval.py', dropout_val]
                commands.append(command)

                #Train all NP models
                for variant in range(0,4):
                    command = ['python', 'NP_train.py', variant, dropout_val]
                    commands.append(command)
                    command = ['python', 'NP_eval.py', variant, dropout_val]
                    commands.append(command)
                    command = ['python', 'NF_train.py', variant, dropout_val]
                    commands.append(command)
                    command = ['python', 'NF_eval.py', variant, dropout_val]
                    commands.append(command)

print(commands)
procs = [ Popen(i) for i in commands ]
for p in procs:
   p.wait()


'''

