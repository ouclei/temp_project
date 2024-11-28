# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 02:23:58 2024

@author: lei
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:07:40 2024

@author: lei
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 20:06:02 2024

@author: lei
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:45:46 2024

@author: lei
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import forehand_function as ff
import robo_num_main as robo
import success_main as su
import distribution_main as dis
import time 

# 
if __name__ == "__main__":
    time_start = time.time()
    print(f"now start with time = {time_start - time_start}s ")
    
    su.success_main()()
    time_elaps_sec = time.time() - time_start
    print(f'fir project \'s done with time = {time_elaps_sec}s ')