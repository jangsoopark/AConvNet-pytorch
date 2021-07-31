import glob
import os


target_list = glob.glob('./*/*')

for name in target_list:
    print(name, name.replace('_', ''))
    os.rename(name, name.replace('_', ''))
