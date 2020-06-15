import cv2
import matplotlib.pyplot as plt

full_path = 'random_cat.jpg'
face_path = 'random_cat_face.png'

full = cv2.imread(full_path)
face = cv2.imread(face_path)

height, width, channels = face.shape

methods = [
'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 
'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 
'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
]

for m in methods:
    
    # Create a copy of the image
    full_copy = full.copy()
    
    # Get the actual function
    method = eval(m)

    # Apply template Matching with the method
    res = cv2.matchTemplate(full_copy,face,method)
    
    # Grab the Max and Min values, plus their locations
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # For TM_SQDIFF or TM_SQDIFF_NORMED, minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc    
    else:
        top_left = max_loc
        
    # Assign the Bottom Right of the rectangle
    bottom_right = (top_left[0] + width, top_left[1] + height)

    # Draw the Red Rectangle
    cv2.rectangle(full_copy, top_left, bottom_right, 255, 10)

    # Plot the Images
    plt.subplot(121)
    plt.imshow(res)
    plt.title('Result of Template Matching')
    
    plt.subplot(122)
    plt.imshow(full_copy)
    plt.title('Detected Point')
    plt.suptitle(m)
    
    plt.show()
    


