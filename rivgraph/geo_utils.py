# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:24:42 2018

@author: Jon

Utilities for reading, writing, managing, processing, manipulating, etc. 
geographic data including tiffs, vrts, shapefiles, etc.

"""
import osr, ogr, gdal
import numpy as np
import rivgraph.im_utils as iu
import geopandas as gpd

def get_EPSG(rast_obj):
    """
    Returns the EPSG code from a given input georeferenced image or virtual
    raster gdal object.
    """
    wkt = rast_obj.GetProjection()
    epsg = wkt2epsg(wkt)
    
    return epsg

    
def wkt2epsg(wkt):
    
    """
    From https://gis.stackexchange.com/questions/20298/is-it-possible-to-get-the-epsg-value-from-an-osr-spatialreference-class-using-th
    Transform a WKT string to an EPSG code

    Arguments
    ---------
    
    wkt: WKT definition
    
    Returns: EPSG code

    """
    
    p_in = osr.SpatialReference()
    s = p_in.ImportFromWkt(wkt)
    if wkt[8:23] == 'World_Mollweide':
        return(54009)    
    if s == 5:  # invalid WKT
        return None
    if p_in.IsLocal() == 1:  # this is a local definition
        return p_in.ExportToWkt()
    if p_in.IsGeographic() == 1:  # this is a geographic srs
        cstype = 'GEOGCS'
    else:  # this is a projected srs
        cstype = 'PROJCS'
    an = p_in.GetAuthorityName(cstype)
    ac = p_in.GetAuthorityCode(cstype)
    if an is not None and ac is not None:  # return the EPSG code
#        return str(p_in.GetAuthorityName(cstype)), str(p_in.GetAuthorityCode(cstype))
        return int(p_in.GetAuthorityCode(cstype))
    
    
def geotiff_vals_from_idcs(idcs, I_orig, I_pull):
    """
    Uses the get_array function to pull individual indices from
    a vrt or tiff. 
    I_orig is the image corresponding to the input indices
    I_pull is the image that you want to pull values from
    """
    if I_orig == I_pull:
        vals = []
        for i in idcs:
            val = iu.get_array(i, I_pull, (1,1))[0][0][0]
            vals.append(val)
            
#    else:
#        ll = idx_to_coords(idcs, idx_gdobj)
#        vals = geotiff_vals_from_coords(ll, val_gdobj)
        
    return vals


def idx_to_coords(idx, gd_obj, inputEPSG=4326, outputEPSG=4326, printout=False):
    
    xy = np.unravel_index(idx, (gd_obj.RasterYSize, gd_obj.RasterXSize))
    ll = xy_to_coords(xy[1]+.5, xy[0]+.5, gd_obj.GetGeoTransform(), inputEPSG=inputEPSG, outputEPSG=outputEPSG)
    if printout is True:
        print('4326 CRS:')
        print('latlon = [')
        for l in ll:
            print('[{},{}],'.format(l[0], l[1]))
        print(']')
    else:
        return ll


def geotiff_vals_from_coords(coords, gt_obj):
    
    # Lat/lon to row/col
    rowcol = latlon_to_xy(coords[:,0], coords[:,1], gt_obj.GetGeoTransform(), inputEPSG=4326, outputEPSG=4326)
    
    # Pull value from vrt at row/col
    vals = []
    for rc in rowcol:
           vals.append(gt_obj.ReadAsArray(int(rc[0]), int(rc[1]), int(1), int(1))[0,0])
           
    return vals


def latlon_to_xy(lats, lons, gt, inputEPSG=4326, outputEPSG=4326):
    
    # Default output EPSG is unprojected WGS84.

    # Transform to output coordinate system if necessary
    if inputEPSG != outputEPSG:
        lats, lons = transform_coordinates(lats, lons, inputEPSG, outputEPSG)
        
    lats = np.array(lats)
    lons = np.array(lons)

    xs = ((lons - gt[0]) / gt[1]).astype(int)
    ys = ((lats - gt[3]) / gt[5]).astype(int)
    
    return np.column_stack((xs, ys))


def xy_to_coords(xs, ys, gt, inputEPSG=4326, outputEPSG=4326):
    
    # Default output EPSG is unprojected WGS84.

    # Transform to output coordinate system if necessary
    if inputEPSG != outputEPSG:
        xs, ys = transform_coordinates(xs, ys, inputEPSG, outputEPSG)
        
    lons = gt[0] + xs * gt[1]
    lats = gt[3] + ys * gt[5]
    
    return np.column_stack((lats, lons))


def transform_coordinates(xs, ys, inputEPSG, outputEPSG):
    
    if inputEPSG == outputEPSG:
        return xs, ys
        
    # Create an ogr object of multipoints
    points = ogr.Geometry(ogr.wkbMultiPoint)
    
    for x,y in zip(xs,ys):
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(float(x), float(y))
        points.AddGeometry(point)
    
    # Create coordinate transformation
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(inputEPSG)
    
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(outputEPSG)
    
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    # transform point
    points.Transform(coordTransform)
    
    xyout = np.array([0,0,0])
    for i in range(len(xs)):
        xyout = np.vstack((xyout, points.GetGeometryRef(i).GetPoints()))
    xyout = xyout[1:,0:2]
    
    return xyout[:,0], xyout[:,1]


""" Utilities below here are not explicitly called by RivGraph functions """

def crop_geotif(tif, cropto='first_nonzero', npad=0, outpath=None):
 
    # Prepare output file path
    if outpath is None:
        output_file = tif.split('.')[-2] + '_cropped.tif'
    else:
        output_file = outpath
    
    tif_obj = gdal.Open(tif)
    tiffull = tif_obj.ReadAsArray()

    if cropto == 'first_nonzero':
        idcs = np.where(tiffull>0)
        t = np.min(idcs[0])        
        b = np.max(idcs[0]) + 1    
        l = np.min(idcs[1])        
        r = np.max(idcs[1]) + 1
        
    # Crop the tiff
    tifcropped = tiffull[t:b,l:r]
    
    # Pad the tiff (if necessary)
    if npad is not 0:
        tifcropped = np.pad(tifcropped, npad, mode='constant', constant_values=False)

    # Create a new geotransform by adjusting the origin (upper-left-most point)
    gt = tif_obj.GetGeoTransform()    
    ulx = gt[0] + (l - npad) * gt[1]
    uly = gt[3] + (t - npad) * gt[5]
    crop_gt = (ulx, gt[1], gt[2], uly, gt[4], gt[5])
    
    # Prepare datatype and options for saving...
    datatype = tif_obj.GetRasterBand(1).DataType
    
    options = ['BLOCKXSIZE=128',
               'BLOCKYSIZE=128',
               'TILED=YES']
    
    # Only compress if we're working with a non-float
    if datatype in [1, 2, 3, 4, 5]: # Int types: see the list at the end of this file 
        options.append('COMPRESS=LZW')
        
    write_geotiff(tifcropped, crop_gt, tif_obj.GetProjection(), output_file, dtype=datatype, options=options)
    
    return output_file
