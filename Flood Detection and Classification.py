import ee
ee.Initialize()
import os
import geemap
import pandas as pd
import numpy as np
from termcolor import colored
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import split
from geemap import ml
from time import time
from datetime import timedelta
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# FLOOD DETECTION

#### CONTROL PANEL ####

start =  '2019-10-1'                                             # Search start date 
finish = '2020-2-1'                                              # Search end date 

region_1 = 45                                                    # Start region   
region_2 = 45                                                    # End region  

percentage = 0.01                                                # Percentage threshold for flood pixels:
                                                                 # 0.01 is sufficient for small regions
                                                  
cloud_perc = 20                                                  # Cloud pixel percentage: <= 20    
    
num_cols = 5                                                     # Number of columns
num_rows = 13                                                    # Number of rows
    
subject = [(-2.4702, 54.4442), (-0.4927, 54.4442),               # Area of interest: England (rectangle)
           (-0.4927, 50.9165), (-2.4702, 50.9165)]  

# Example parameters: 

# start = 2019-10-1    num_cols = 5      percentage = 0.01    USE: trees.csv and pre_trees.csv
# finish = 2020-2-1    num_rows = 13     cloud_perc = 20      UNCOMMENT image downloading lines (4 lines)

# (1) subject = England       (2) subject = England     
#     idx_1 = 45                  idx_1 = 17                            
#     idx_2 = 45                  idx_2 = 23                      


#### FUNCTIONS ####

Map = geemap.Map()     # Create Map

datelist = []          # List of dates of flood optical images
imageList = []         # List of flood optical images

predateList = []       # List of dates of pre flood optical images
pre_imageList = []     # List of pre flood optical images
maskList = []          # List of flood masks

SAR_imageList = []     # List of flood SAR images
preSAR_imageList = []  # List of pre flood SAR images

final_geometry = []    # List of coordinates of flooded regions

# Function to split UK (or other) into smaller regions 
def grid(nx, ny, subject):

    subject = Polygon(subject)

    minx, miny, maxx, maxy = subject.bounds
    dx = (maxx - minx) / nx  
    dy = (maxy - miny) / ny  

    longitude_split = [LineString([(minx, miny + i*dy), (maxx, miny + i*dy)]) for i in range(ny)]
    latitude_split = [LineString([(minx + i*dx, miny), (minx + i*dx, maxy)]) for i in range(nx)]
    regions = longitude_split + latitude_split

    result = subject
    for s in regions:
        result = MultiPolygon(split(result, s))

    parts = [list(part.exterior.coords) for part in result.geoms]

    geometries = []

    for i in parts:

        lon_1 = i[0][0]
        lat_1 = i[0][1]

        lon_3 = i[2][0]
        lat_3 = i[2][1]

        geometry = ee.Geometry.Rectangle([[lon_1, lat_1], [lon_3, lat_3]])
        geometries.append(geometry) 
        
    return geometries

# Function to remove images that do not cover the full region of interest
def complete(Image, geometry, band):

    totPixels = ee.Number(ee.Image(1).reduceRegion(**{
        'reducer': ee.Reducer.count(),
        'geometry': geometry,
        'scale': 30}).values().get(0))

    actPixels = ee.Number(Image.select(band).reduceRegion(**{
        'reducer': ee.Reducer.count(),
        'scale': 30,
        'geometry': geometry}).values().get(0))

    pcPix = actPixels.divide(totPixels).multiply(100).getInfo()

    return(round(pcPix, 0))

def ndwi(image):
        ndwi = image.normalizedDifference(['B3', 'B8A'])\
        .rename('NDWI')\
        .copyProperties(image, ['system:time_start'])
        return image.addBands(ndwi)
    
    # Reomve images with same dates in collections
    
def removeDuplicate(img_col):
    img_col = img_col.sort('system:time_start')
    img_col = img_col.toList(collection.size())

    length = img_col.size().getInfo()

    filtered_images = []

    for dd in range(0, length-1):

        image = ee.Image(img_col.get(dd))
        image2 = ee.Image(img_col.get(dd+1))

        dateTime = ee.Date(image.get('system:time_start'))
        date = dateTime.format().getInfo()[0:10]

        dateTime2 = ee.Date(image2.get('system:time_start'))
        date2 = dateTime2.format().getInfo()[0:10]

        if date2 != date:
            filtered_images.append(image)

    img_col = ee.ImageCollection.fromImages(filtered_images)
        
    return img_col

# Function iterates over the NDWI collection and computes the difference in consecutive images. 

def NDWIdiff(date):

    #get the image corresponding to the date
    currentImage = NDWI_collection.filter(ee.Filter.eq('system:time_start', date)).first();

    #Now we have to get the 'previous' image in the collection
    indexCurrent = dateList.indexOf(date);
    indexPrevious = indexCurrent.subtract(1);
    datePrevious = dateList.get(indexPrevious);
    previousImage = NDWI_collection.filter(ee.Filter.eq('system:time_start', datePrevious)).first(); 

    #Subtract the current image from the previous
    diffImage = currentImage.subtract(previousImage).select(['NDWI'], ['NDWIdiff']);

    return currentImage.addBands(diffImage).set('system:index_previous', previousImage.get('system:index'))

def floodPixels(image):
    def floodPixels(value):
        return image.select('NDWIdiff').gt(ee.Number(value))
    
    # Threshold for NDWI difference 
    threshold = [0.8] 

    water = ee.ImageCollection.fromImages(ee.List(threshold).sort()\
        .map(floodPixels)).reduce('sum').rename('water')\
        .reproject(image.projection(), None, image.projection().nominalScale()); 

    return image.addBands(water)


#### MAIN ####

# Create regions

geometries = grid(num_cols, num_rows ,subject)
num = len(geometries) 

num_select = region_2-region_1+1
region_area = geometries[0].area().divide(1000*1000).getInfo()

print("Number of regions:", num)
print("Number of regions selected: {}".format(num_select))
print("")
print("Area of single region: {}km\u00b2".format(round(region_area, 0)))
print("Total area of regions selected: {}km\u00b2".format(round(region_area * num_select, 0)))
print("")
print("Start search:", start)
print("Stop search:", finish)
print("")

total_flood = 0
number = 1

for geom in range(region_1-1, region_2): 
    
    print("({}) REGION".format(number), geom+1)
    print("")
    
    aoi = geometries[geom]
    
    # Creating SAR image collection over region
    
    SAR = ee.ImageCollection('COPERNICUS/S1_GRD')\
        .filter(ee.Filter.eq('instrumentMode', 'IW'))\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))\
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
        .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))\
        .filterBounds(aoi)
    
    # Creating optical image collection over region
    
    start = ee.Date(start)
    finish = ee.Date(finish)
    
    collection = ee.ImageCollection("COPERNICUS/S2")\
        .filterBounds(aoi)\
        .filterDate(start, finish)\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_perc))\

    # Remove images with duplicate dates
    collection = removeDuplicate(collection)
    
    # Create the NDWI images 
    collection = collection.map(ndwi)

    # NDWI image collection
    NDWI_collection = collection.select('NDWI').sort('system:time_start')

    # Remove first data
    dateList = ee.List(NDWI_collection.aggregate_array('system:time_start')).slice(1)

    # Collection of NDWI difference images
    difference_collection = ee.ImageCollection.fromImages(dateList.map(NDWIdiff)) 

    # Thresholding the NDWI difference images to classify flooded pixels
    threshold_collection = difference_collection.map(floodPixels) 

    # Thresholded collection to list of images
    threshold_collection = threshold_collection.toList(threshold_collection.size())

    #Convert orinal optical collection to list and sort by date
    collection = collection.sort('system:time_start')
    collection = collection.toList(collection.size()) 

    length = collection.size().getInfo() - 1   

    print("Number of optical images:", length)

    currentDate = []
    flood_count = 0
    
    for i in range(0, length):

        optical = ee.Image(collection.get(i+1)).clip(aoi)
        pre_optical = ee.Image(collection.get(i)).clip(aoi)
        flood_mask = ee.Image(threshold_collection.get(i)).clip(aoi).select('water')
        
        ee_date = ee.Date(optical.get('system:time_start'))
        date = ee_date.format().getInfo()

        pre_ee_date = ee.Date(pre_optical.get('system:time_start'))
        pre_date = pre_ee_date.format().getInfo()

        meanDict = flood_mask.reduceRegion(**{
            'reducer': ee.Reducer.mean(),
            'geometry': aoi,
            'scale': 30, 
            'bestEffort': True 
            
        })

        try:
            mean = meanDict.get('water')
            flood_perc = mean.getInfo()*100

            print("")
            print("IMAGE:", i+1)
            print("DATE:", date[0:10])
            print("Flood percentage:", round(flood_perc, 3))

            if flood_perc >= percentage: #Image is a flood image if True  
                
                print("")
                print(colored("FLOODS DETECTED", 'blue'))
                
                # Collecting flood and preflood SAR images 
                
                date = date[0:10]
                pre_date = pre_date[0:10]
                
                flood_images = SAR\
                    .filterDate(ee.Date(date), ee.Date(date).advance(2, 'day'))
                
                preflood_images = SAR\
                    .filterDate(ee.Date(pre_date).advance(-2, 'day'), ee.Date(pre_date))
                
                f_image = ee.Image(flood_images.mosaic()).clip(aoi)
                p_image = ee.Image(preflood_images.mosaic()).clip(aoi)
                
                # Ensure images are useable for training and/or classification
                
                perc_1 = complete(optical, aoi, 'NDWI')
                perc_2 = complete(pre_optical, aoi, 'NDWI')
                perc_3 = complete(f_image, aoi, 'VH')
                perc_4 = complete(p_image, aoi, 'VH')
                
                # Check that optical images are the same size, and SAR images
                    
                if (perc_1 == perc_2) and perc_3 == 100 and perc_4 == 100:

                    print(colored("IMAGES ACCEPTED", 'green'))
                    
                    final_geometry.append(aoi)

                    datelist.append(date) 
                    predateList.append(pre_date)

                    currentDate.append(date)

                    imageList.append(optical)
                    pre_imageList.append(pre_optical)

                    SAR_imageList.append(f_image)
                    preSAR_imageList.append(p_image)

                    flood_count += 1
                    total_flood += 1  

                else:

                    print(colored("IMAGES DECLINED", 'red'))
        except:
            
            print(colored("ERROR", 'red'))
            pass


    print("")
    print("Flood dates:", currentDate)
    print("Number of floods:", flood_count)
    print("")
    
    number += 1
    
print("FINISHED ALL REGIONS")
print("Total floods:", total_flood)

# Create optical and mask collections

flood_collection = ee.ImageCollection.fromImages(imageList)
preflood_collection = ee.ImageCollection.fromImages(pre_imageList)

# Create SAR collections

SAR_flood_collection = ee.ImageCollection.fromImages(SAR_imageList)
SAR_preflood_collection = ee.ImageCollection.fromImages(preSAR_imageList)

#Map flood images & masks

for im in range(0, len(datelist)):
    
    # Flood optical
    Map.addLayer(ee.Image(imageList[im]), {'bands': ['B8', 'B11', 'B4'], 'min':0, 'max':3000}, '(Img {}) Optical'.format(im+1), True)
    # Preflood optical
    Map.addLayer(ee.Image(pre_imageList[im]), {'bands': ['B8', 'B11', 'B4'], 'min':0, 'max':3000}, '(Img {}) Pre optical'.format(im+1), True)
    # Flood SAR
    Map.addLayer(ee.Image(SAR_imageList[im]), {'bands': 'VH', 'min': -25, 'max': 0}, '(Img {}) Flood SAR'.format(im+1), True)
    # Preflood SAR
    Map.addLayer(ee.Image(preSAR_imageList[im]), {'bands': 'VH', 'min': -25, 'max': 0}, '(Img {}) Preflood SAR'.format(im+1), True)
       
# FLOOD CLASSIFICATION 

#### FUNCTIONS ####

flood_images = []
preflood_images = []

# Functions to create NDWI, NDVI and BI bands. To be used in water pixel classification.

def ndwi(image):
    ndwi = image.normalizedDifference(['B3', 'B8A']).rename('NDWI')
    return image.addBands(ndwi)

def ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

def bi(image):
    B3 = image.select('B3')
    B4 = image.select('B4')
    B8 = image.select('B8')
    bi = ((((B3.pow(2)).add(B4.pow(2)).add(B8.pow(2))).divide(3)).pow(0.5)).divide(10000).rename('BI')
    return image.addBands(bi)

# Function to remove speckle in SAR

def toNatural(img):
    return ee.Image(10.0).pow(img.select(0).divide(10.0))

def toDB(img):
    return ee.Image(img).log10().multiply(10.0)

def RefinedLee(image):

    bandNames = image.bandNames().remove('angle')

    def inner(b):

        img = image.select([b]);
    
        # img must be linear, i.e. not in dB!
        # Set up 3x3 kernels 
        weights3 = ee.List.repeat(ee.List.repeat(1,3),3);
        kernel3 = ee.Kernel.fixed(3,3, weights3, 1, 1, False);
  
        mean3 = img.reduceNeighborhood(ee.Reducer.mean(), kernel3);
        variance3 = img.reduceNeighborhood(ee.Reducer.variance(), kernel3);
  
        # Use a sample of the 3x3 windows inside a 7x7 windows to determine gradients and directions
        sample_weights = ee.List([[0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0], [0,1,0,1,0,1,0], [0,0,0,0,0,0,0], [0,1,0,1,0,1,0],[0,0,0,0,0,0,0]]);
  
        sample_kernel = ee.Kernel.fixed(7,7, sample_weights, 3,3, False);
  
        # Calculate mean and variance for the sampled windows and store as 9 bands
        sample_mean = mean3.neighborhoodToBands(sample_kernel); 
        sample_var = variance3.neighborhoodToBands(sample_kernel);
  
        # Determine the 4 gradients for the sampled windows
        gradients = sample_mean.select(1).subtract(sample_mean.select(7)).abs();
        gradients = gradients.addBands(sample_mean.select(6).subtract(sample_mean.select(2)).abs());
        gradients = gradients.addBands(sample_mean.select(3).subtract(sample_mean.select(5)).abs());
        gradients = gradients.addBands(sample_mean.select(0).subtract(sample_mean.select(8)).abs());
  
        # And find the maximum gradient amongst gradient bands
        max_gradient = gradients.reduce(ee.Reducer.max());
  
        # Create a mask for band pixels that are the maximum gradient
        gradmask = gradients.eq(max_gradient);
  
        # duplicate gradmask bands: each gradient represents 2 directions
        gradmask = gradmask.addBands(gradmask);
  
        # Determine the 8 directions
        directions = sample_mean.select(1).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(7))).multiply(1);
        directions = directions.addBands(sample_mean.select(6).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(2))).multiply(2));
        directions = directions.addBands(sample_mean.select(3).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(5))).multiply(3));
        directions = directions.addBands(sample_mean.select(0).subtract(sample_mean.select(4)).gt(sample_mean.select(4).subtract(sample_mean.select(8))).multiply(4));
        # The next 4 are the not() of the previous 4
        directions = directions.addBands(directions.select(0).Not().multiply(5));
        directions = directions.addBands(directions.select(1).Not().multiply(6));
        directions = directions.addBands(directions.select(2).Not().multiply(7));
        directions = directions.addBands(directions.select(3).Not().multiply(8));
  
        # Mask all values that are not 1-8
        directions = directions.updateMask(gradmask);
  
        # "collapse" the stack into a singe band image (due to masking, each pixel has just one value (1-8) in it's directional band, and is otherwise masked)
        directions = directions.reduce(ee.Reducer.sum());  
  
        sample_stats = sample_var.divide(sample_mean.multiply(sample_mean));
  
        #Calculate localNoiseVariance
        sigmaV = sample_stats.toArray().arraySort().arraySlice(0,0,5).arrayReduce(ee.Reducer.mean(), [0]);
  
        # Set up the 7*7 kernels for directional statistics
        rect_weights = ee.List.repeat(ee.List.repeat(0,7),3).cat(ee.List.repeat(ee.List.repeat(1,7),4));
  
        diag_weights = ee.List([[1,0,0,0,0,0,0], [1,1,0,0,0,0,0], [1,1,1,0,0,0,0], [1,1,1,1,0,0,0], [1,1,1,1,1,0,0], [1,1,1,1,1,1,0], [1,1,1,1,1,1,1]]);
  
        rect_kernel = ee.Kernel.fixed(7,7, rect_weights, 3, 3, False);
        diag_kernel = ee.Kernel.fixed(7,7, diag_weights, 3, 3, False);
  
        # Create stacks for mean and variance using the original kernels. Mask with relevant direction.
        dir_mean = img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel).updateMask(directions.eq(1));
        dir_var = img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel).updateMask(directions.eq(1));
  
        dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel).updateMask(directions.eq(2)));
        dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel).updateMask(directions.eq(2)));
  
        # and add the bands for rotated kernels
        for i in range(1, 4):
            dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
            dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), rect_kernel.rotate(i)).updateMask(directions.eq(2*i+1)))
            dir_mean = dir_mean.addBands(img.reduceNeighborhood(ee.Reducer.mean(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))
            dir_var = dir_var.addBands(img.reduceNeighborhood(ee.Reducer.variance(), diag_kernel.rotate(i)).updateMask(directions.eq(2*i+2)))

  
        dir_mean = dir_mean.reduce(ee.Reducer.sum());
        dir_var = dir_var.reduce(ee.Reducer.sum());
  
        # A finally generate the filtered value
        varX = dir_var.subtract(dir_mean.multiply(dir_mean).multiply(sigmaV)).divide(sigmaV.add(1.0))
  
        b = varX.divide(dir_var)
        result = dir_mean.add(b.multiply(img.subtract(dir_mean)))
  
        return result.arrayProject([0]).arrayFlatten([['sum']]).float()
    
    result = ee.ImageCollection(bandNames.map(inner)).toBands().rename(bandNames).copyProperties(image)
    
    return image.addBands(result, None, True)  

# Function to classifiy water pixels in optical flood image

def classify_flood(data, min_NDVI, max_NDVI):
    # Start the clock
    start_time = time()

    # Create a Water binary classification column, with pre-determined value of 0 (non-Water)
    data = data.assign(Water=0)

    # Obtain process parameters
    # Find mean BI and it's standard deviation for pixels with NDWI over 0.3
    BI_03 = data.loc[data['NDWI'] >= 0.3].BI
    mean = np.nanmean(BI_03)
    stdv = np.nanstd(BI_03)
    # Set the maximum BI for these pixels
    max_BI = mean + 4 * stdv

    # Set the NDWI threshold values
    ndwi_thresh = [0.3, 0.1, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.85]

    # Calculate max_BI for the NDWI thresholds and append it
    bi_thresh = [max_BI]
    for i in range(len(ndwi_thresh) - 1):
        max_BI += (35 + i * 5) / 10000
        bi_thresh.append(max_BI)

    # Set the pixels with NDWI greater than 0.3 and BI less than max_BI as Water Pixels
    data.loc[(data['NDWI'] >= 0.3 ) & (data['BI'] <= bi_thresh[0]), 'Water'] = 1

    # NDWI between 0.3 and 0.1
    data.loc[(data['NDWI'] >= 0.1) & (data['NDWI'] < 0.3) & (data['BI'] <= bi_thresh[1]), 'Water'] = 1

    # NDWI between -0.1 and 0.1
    data.loc[(data['NDWI'] >= -0.1) & (data['NDWI'] < 0.1) & (data['BI'] <= bi_thresh[2]), 'Water'] = 1

    # Set the current threshold index, based on the threshold lists
    thresh_indx = 3
    max_BI = bi_thresh[thresh_indx]
    next_ndwi = ndwi_thresh[thresh_indx]

    # Loop over NDWI values between -0.1 and -0.8 with a step of -0.01
    # Set as water the pixels with BI less than the corresponding max, already calculated for the NDWI threshold values
    # and with NDVI that is between 2 less and 12 more than the absolute of the NDWI value
    ndwi_values = np.arange(-0.1, -0.3, -0.01)
    for value in ndwi_values:
        ndvi_min = np.abs(value) - min_NDVI
        ndvi_max = np.abs(value) + max_NDVI

        min_BI = 0.035

        # Check if the
        if value < next_ndwi:
            thresh_indx += 1
            max_BI = bi_thresh[thresh_indx]
            next_ndwi = ndwi_thresh[thresh_indx]

        ndwi_min = value - 0.01
        ndwi_max = value
        data.loc[(data['NDWI'] >= ndwi_min) & (data['NDWI'] < ndwi_max) & (data['BI'] <= max_BI) & (data['BI'] > min_BI) &
                 (data['NDVI'] >= ndvi_min) & (data['NDVI'] < ndvi_max), 'Water'] = 1

    print('Water Pixels pre-correction:', len(data.loc[data['Water'] == 1]))

    data = corrections(data)

    print('Water Pixels after correction:', len(data.loc[data['Water'] == 1]))

    # End the clock
    elapsed_time = int(round(time() - start_time))
    # Calculate and output elapsed time
    time_final = str(timedelta(seconds=elapsed_time))
    print('Total runtime:', time_final)

    #return data['Water']
    return data

# Function to classifiy water pixels in optical preflood image

def classify_preflood(data, min_NDVI, max_NDVI):
    # Start the clock
    start_time = time()

    # Create a Water binary classification column, with pre-determined value of 0 (non-Water)
    data = data.assign(Water=0)

    # Obtain process parameters
    # Find mean BI and it's standard deviation for pixels with NDWI over 0.3
    BI_03 = data.loc[data['NDWI'] >= 0.3].BI
    mean = np.nanmean(BI_03)
    stdv = np.nanstd(BI_03)
    # Set the maximum BI for these pixels
    max_BI = mean + 4 * stdv

    # Set the NDWI threshold values
    ndwi_thresh = [0.3, 0.1, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.85]

    # Calculate max_BI for the NDWI thresholds and append it
    bi_thresh = [max_BI]
    for i in range(len(ndwi_thresh) - 1):
        max_BI += (35 + i * 5) / 10000
        bi_thresh.append(max_BI)

    # Set the pixels with NDWI greater than 0.3 and BI less than max_BI as Water Pixels
    data.loc[(data['NDWI'] >= 0.3) & (data['BI'] <= bi_thresh[0]), 'Water'] = 1

    # NDWI between 0.3 and 0.1
    data.loc[(data['NDWI'] >= 0.1) & (data['NDWI'] < 0.3) & (data['BI'] <= bi_thresh[1]), 'Water'] = 1

    # NDWI between -0.1 and 0.1
    data.loc[(data['NDWI'] >= -0.1) & (data['NDWI'] < 0.1) & (data['BI'] <= bi_thresh[2]), 'Water'] = 1

    # Set the current threshold index, based on the threshold lists
    thresh_indx = 3
    max_BI = bi_thresh[thresh_indx]
    next_ndwi = ndwi_thresh[thresh_indx]

    # Loop over NDWI values between -0.1 and -0.8 with a step of -0.01
    # Set as water the pixels with BI less than the corresponding max, already calculated for the NDWI threshold values
    # and with NDVI that is between 2 less and 12 more than the absolute of the NDWI value
    ndwi_values = np.arange(-0.1, -0.3, -0.01)
    for value in ndwi_values:
        ndvi_min = np.abs(value) - min_NDVI
        ndvi_max = np.abs(value) + max_NDVI

        min_BI = 0.035

        # Check if the
        if value < next_ndwi:
            thresh_indx += 1
            max_BI = bi_thresh[thresh_indx]
            next_ndwi = ndwi_thresh[thresh_indx]

        ndwi_min = value - 0.01
        ndwi_max = value
        data.loc[
            (data['NDWI'] >= ndwi_min) & (data['NDWI'] < ndwi_max) & (data['BI'] <= max_BI) & (data['BI'] > min_BI) &
            (data['NDVI'] >= ndvi_min) & (data['NDVI'] < ndvi_max), 'Water'] = 1

    print('Water Pixels pre-correction:', len(data.loc[data['Water'] == 1]))

    data = corrections(data)  

    print('Water Pixels after correction:', len(data.loc[data['Water'] == 1]))

    # End the clock
    elapsed_time = int(round(time() - start_time))
    # Calculate and output elapsed time
    time_final = str(timedelta(seconds=elapsed_time))
    print('Total runtime:', time_final)

    #return data['Water'] 
    return data

# Correction function to remove pixels associated with dark buildings and cloud shadows in optical images

def corrections(data):
    data.loc[(data['NDWI'] >= -0.1) & (data['NDWI'] < 0.1) & (data['NDVI'] >= -0.01) & (data['NDVI'] < 0.05) &
             (data['Water'] == 1), 'Water'] = 0
    data.loc[(data['NDWI'] < 0) & (data['NDVI'] > 0) & (data['Water'] == 1), 'Water'] = 0
    data.loc[(data['BI'] > 0.125) & (data['Water'] == 1), 'Water'] = 0

    return data

# Outputs an accuracy report

def report(actual, predicted):
    accuracy=accuracy_score(actual, predicted)
    print("Accuracy Score:", accuracy)
    matrix = confusion_matrix(actual, predicted)
    print("Confusion matrix:", matrix)
    report = classification_report(actual, predicted)
    print("")
    print(report)


### MAIN ###

# Adding NDWI, NDVI and BI bands to optical collections

flood_collection = flood_collection.map(ndvi).map(ndwi).map(bi)
preflood_collection = preflood_collection.map(ndvi).map(ndwi).map(bi)

flood_collection = flood_collection.toList(flood_collection.size())
preflood_collection = preflood_collection.toList(preflood_collection.size())
SAR_flood_collection = SAR_flood_collection.toList(SAR_flood_collection.size())
SAR_preflood_collection = SAR_preflood_collection.toList(SAR_preflood_collection.size())

for image in range(0, len(datelist)):
    
    flood_optical = ee.Image(flood_collection.get(image))
    preflood_optical = ee.Image(preflood_collection.get(image))
    flood_SAR = ee.Image(SAR_flood_collection.get(image))
    preflood_SAR = ee.Image(SAR_preflood_collection.get(image))
    
    # Creating total flood image (contains optical and SAR bands)
    
    flood_image = flood_optical
    
    flood_SAR_VH = ee.Image(toDB(RefinedLee(toNatural(flood_SAR.select('VH'))))) 
    flood_SAR_VV = ee.Image(toDB(RefinedLee(toNatural(flood_SAR.select('VV'))))) 
    VH = flood_SAR_VH.select('constant').rename('VH')
    VV = flood_SAR_VV.select('constant').rename('VV')

    # Creating total pre-flood image (contains optical and SAR bands)
    
    preflood_image = preflood_optical
    
    preflood_SAR_VH = ee.Image(toDB(RefinedLee(toNatural(preflood_SAR.select('VH'))))) 
    preflood_SAR_VV = ee.Image(toDB(RefinedLee(toNatural(preflood_SAR.select('VV'))))) 
    pre_VH = preflood_SAR_VH.select('constant').rename('pre_VH')
    pre_VV = preflood_SAR_VV.select('constant').rename('pre_VV')
    
    differenceVH = flood_SAR_VH.divide(preflood_SAR_VH).rename('Diff_VH')
    differenceVV = flood_SAR_VV.divide(preflood_SAR_VV).rename('Diff_VV')

    
    # Add these new bands to flood and pre_flood images
    
    flood_image = flood_image.addBands(VH)
    flood_image = flood_image.addBands(VV)
    flood_image = flood_image.addBands(differenceVH)
    flood_image = flood_image.addBands(differenceVV)
    
    preflood_image = preflood_image.addBands(pre_VH)
    preflood_image = preflood_image.addBands(pre_VV)
    
    # Final images contain relevant bands only 

    flood_image = flood_image.select(['NDWI', 'NDVI', 'BI', 'VH', 'VV', 'Diff_VH', 'Diff_VV'])
    preflood_image = preflood_image.select(['NDWI', 'NDVI', 'BI', 'pre_VH', 'pre_VV'])
    
    flood_images.append(flood_image)
    preflood_images.append(preflood_image)
    

for classify in range(0, len(flood_images)):

    flood_image = flood_images[classify]
    preflood_image = preflood_images[classify]
    
    aoi1 = ee.FeatureCollection(final_geometry[classify])
     
    work_dir = os.path.expanduser('~/Downloads')
    
    out_csv = os.path.join(work_dir, 'flood_{}.csv'.format(classify+10)) 
    out_csv2 = os.path.join(work_dir, 'pre-flood_{}.csv'.format(classify+10))
    
    # Download images as CSV (comment out if images were previously downloaded)
    geemap.extract_values_to_points(aoi1, flood_image, out_csv, scale=45)
    geemap.extract_values_to_points(aoi1, preflood_image, out_csv2, scale=45)

    flood_data = pd.read_csv(r"~/Downloads/flood_{}.csv".format(classify+10))
    pre_data = pd.read_csv(r"~/Downloads/pre-flood_{}.csv".format(classify+10))
    
    # Classify water pixels in flooded and pre-flood images
    
    min_NDVI = 2 
    max_NDVI = 12 
    
    print("")
    print('FLOOD {}:'.format(classify+1))
    print("")
    
    flood_water = classify_flood(flood_data, min_NDVI, max_NDVI)
    
    print("")
    print("PRE-FLOOD {}:".format(classify+1)) 
    print("")

    preflood_water = classify_preflood(pre_data, min_NDVI, max_NDVI)
    
    ## CLASSIFY FLOODED IMAGES WITH MACHINE LEARNING MODEL ##
    
    feature_names = ['VH', 'VV', 'Diff_VH', 'Diff_VV', 'NDWI', 'NDVI', 'BI']  
    label = "Water"

    X = flood_water[feature_names]
    Y = flood_water[label]

    out_csv = os.path.expanduser("~/Downloads/trees.csv")
    
    #Train classifier and export

    n_trees = 500 
    rf = ensemble.RandomForestClassifier(n_trees).fit(X, Y)
    trees =  ml.rf_to_strings(rf, feature_names)
    ml.trees_to_csv(trees, out_csv)

    #Import classifier and classify flood image
    ee_classifier = ml.csv_to_classifier(out_csv)

    classified = flood_image.select(feature_names).classify(ee_classifier)

    # Accuracy report

    work_dir = os.path.expanduser('~/Downloads')
    out_csv = os.path.join(work_dir, 'classified_{}.csv'.format(classify+10)) 
    
    # Download Classified image
    geemap.extract_values_to_points(ee.FeatureCollection(aoi1), classified, out_csv, scale=45)

    predicted = pd.read_csv(r"~/Downloads/classified_{}.csv".format(classify+10))

    predicted = predicted['classification']
    actual = Y
    
    print("")
    print("FLOOD {} REPORT:".format(classify+1))
    print("")
    report(actual, predicted)
    print("")
    
    ## CLASSIFY PRE-FLOODED IMAGES WITH MACHINE LEARNING MODEL ##
    
    prefeature_names = ['pre_VH', 'pre_VV', 'NDWI', 'NDVI', 'BI']  
    label = "Water"

    P = preflood_water[prefeature_names]
    Q = preflood_water[label]

    out_csv = os.path.expanduser("~/Downloads/pre_trees.csv")
    
    #Train classifier and export

    n_trees = 500 
    rf = ensemble.RandomForestClassifier(n_trees).fit(P, Q)
    trees =  ml.rf_to_strings(rf, prefeature_names)
    ml.trees_to_csv(trees, out_csv)

    #Import classifier and classify flood image
    ee_classifier = ml.csv_to_classifier(out_csv)

    pre_classified = preflood_image.select(prefeature_names).classify(ee_classifier)

    # Accuracy report

    work_dir = os.path.expanduser('~/Downloads')
    out_csv = os.path.join(work_dir, 'pre-classified_{}.csv'.format(classify+10))
    
    # Download classified image
    geemap.extract_values_to_points(ee.FeatureCollection(aoi1), pre_classified, out_csv, scale=45)

    pre_predicted = pd.read_csv(r"~/Downloads/pre-classified_{}.csv".format(classify+10))

    pre_predicted = pre_predicted['classification']
    pre_actual = Q

    print("PRE-FLOOD {} REPORT:".format(classify+1))
    print("")
    
    report(pre_actual, pre_predicted)
    
    # Final flood water classification 
    
    inflood = classified.mask(classified)
    preflood = pre_classified.mask(pre_classified)

    floodWater = classified.subtract(pre_classified)
    flood_water_mask = floodWater.mask(floodWater) # <-- FLOOD PIXELS!

    Map.addLayer(flood_water_mask, {"palette": 'blue'}, '(Img {}) Classification'.format(classify+1), True)


subject_aoi = ee.Geometry.Rectangle([[subject[0][0], subject[0][1]], [subject[2][0], subject[2][1]]])    
Map.centerObject(subject_aoi, zoom=6)
Map    
