# Flood-Detection-with-Google-Earth-Engine
A machine learning oriented approach to ground-based flood detection using sentinel satellite imagery. Geospatial data is retrieved from Google Earth Engine API and corresponding image metadata is stored in csv format. 'UK Flood Detection' contains original code I developed to classify and store pre-flood and inflood images using a time-series thresholding technique on the NDWI index. 'Flood Detection and Classification' implements the former code to provide a training set for a machine learning model (a random forest ensemble). I set out optional prepossessing procedures on SAR and optical images for full efficiency in water pixel classification.
