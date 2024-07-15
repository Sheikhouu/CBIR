import numpy as np
import os
import cv2
from descriptor import glcm, bitdesc

def process_datasets(root_folder, descriptor='glcm'):
    print('Function calling')
    all_features = []
    count = 1

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                relative_path = os.path.relpath(os.path.join(root, file), root_folder)
                image_rel_path = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(image_rel_path))
                
                print(f"Processing file: {image_rel_path}")  # Imprimer le chemin du fichier
                if not os.path.exists(image_rel_path):
                    print(f"File does not exist: {image_rel_path}")
                    continue

                try:
                    if descriptor == 'glcm':
                        extraction = glcm(image_rel_path)
                    elif descriptor == 'bit':
                        extraction = bitdesc(image_rel_path)
                except ValueError as e:
                    print(f"Error processing {image_rel_path}: {e}")
                    continue
                
                extraction = extraction + [folder_name, image_rel_path]
                all_features.append(extraction)
            count += 1

    print('Extraction completed!')
    print('Now creating signatures DB ...')
    signatures = np.array(all_features)
    print(f"Signature's shape: {signatures.shape}")
    np.save('signatures.npy', signatures)
    print("Signature successfully stored!")

if __name__ == '__main__':
    descriptor_choice = input("Enter descriptor (glcm/bit): ").strip().lower()
    if descriptor_choice not in ['glcm', 'bit']:
        print("Invalid descriptor choice. Defaulting to 'glcm'.")
        descriptor_choice = 'glcm'
    process_datasets('../Cbir_datasets/', descriptor=descriptor_choice)
