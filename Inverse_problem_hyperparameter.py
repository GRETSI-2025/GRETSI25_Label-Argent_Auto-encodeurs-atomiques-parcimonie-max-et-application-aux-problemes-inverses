import numpy as np
pad=2 #super resolution pad
blur=0.7    
Number_of_images_to_recover=20   
seuil=1e-5   #Threshold under which the PGD is stopped
k_max=15  #Number of iteration to perform per recovery
mu_s=np.array([1,10,5])    #Step sizes (grid-search)
