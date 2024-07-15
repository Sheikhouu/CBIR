from distances import retrieve_similar_images
from descriptor import glcm
import numpy as np
import cv2
signatures = np.load('signatures.npy')
query_image = 'iris.jpg'
query_features = glcm(query_image)
distance = 'canberra'
num_result = 10

results = retrieve_similar_images(signatures, query_features, distance, num_result)
paths = [x[0] for x in results]
print(paths)
for i, img in enumerate(paths):
    img = cv2.imread(img)
    cv2.imshow(f'Image {i}', img)
cv2.waitKey(0)

