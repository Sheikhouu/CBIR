import streamlit as st
import numpy as np
import cv2
from descriptor import glcm, bitdesc
from distances import retrieve_similar_images

# Charger les signatures
signatures = np.load('signatures.npy')

# Titre de l'application
st.title("Application de recherche d'images similaires")

# Téléversement de l'image
uploaded_file = st.file_uploader("Téléverser une image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Erreur lors du chargement de l'image. Veuillez réessayer avec une autre image.")
    else:
        st.image(image, caption='Image téléversée', use_column_width=True)
        
        # Sélection du nombre d'images similaires
        num_images = st.slider("Nombre d'images similaires à afficher", min_value=1, max_value=10, value=5)
        
        # Choix de la mesure de distance
        distance_measure = st.selectbox("Choisir la mesure de distance", ["euclidean", "manhattan", "chebyshev", "canberra"])
        
        # Choix du descripteur
        descriptor = st.selectbox("Choisir le descripteur", ["GLCM", "BiT"])

        # Extraction des caractéristiques de l'image requête
        if descriptor == "GLCM":
            query_features = glcm(image)
        else:
            query_features = bitdesc(image)
        
        # Rechercher les images similaires lorsque l'utilisateur clique sur le bouton
        if st.button("Rechercher des images similaires"):
            results = retrieve_similar_images(signatures, query_features, distance_measure, num_images)
            if results:
                paths = [x[0] for x in results]
                for i, img_path in enumerate(paths):
                    print(f"Loading image: {img_path}")  # Vérifier le chemin du fichier
                    img = cv2.imread(img_path)
                    if img is not None:
                        st.image(img, caption=f'Image similaire {i+1}', use_column_width=True)
                    else:
                        st.error(f"Erreur lors du chargement de l'image {img_path}")
            else:
                st.error("Aucune image similaire trouvée.")
