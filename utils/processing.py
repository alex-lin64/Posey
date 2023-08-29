import numpy as np


def preprocess(keypoints):
    """
    Cleans keypoints data, writes file to csv.  

    :params:
        - keypoints: landmark obj of all 33 keypoints w/ x, y, z, and visibility, 
            already normalized
    
    :returns:
        - Embedded numpy vector of normalized pose landmarks with dim (1, 63)
    """
    # clean up unwanted keypoints
    keypoints = keypoints[:1] + keypoints[2:3] + keypoints[6:9] + keypoints[11:17] + keypoints[23:33]
    # flatten to 1d vector
    cleaned_kp = np.array([[x.x, x.y, x.z] for x in keypoints], dtype="float64").flatten() 
    cleaned_kp = cleaned_kp.reshape(1, len(cleaned_kp))

    return cleaned_kp

