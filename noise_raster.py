# -*- coding: utf-8 -*-
import math
import os
import numpy as np

from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorLayer, QgsProcessingParameterString,
                       QgsProcessingParameterNumber, QgsProcessingParameterBoolean,
                       QgsProcessingParameterField, QgsProcessingParameterFile,
                       QgsProcessingParameterRasterDestination, QgsProcessingException,
                       QgsCoordinateTransform, QgsProject, QgsCoordinateReferenceSystem)
from qgis.PyQt.QtCore import QVariant
from osgeo import gdal

from ..utils import raster as ru
from ..utils.acoustics import level_from_point_source, energetic_sum

class NoiseRasterAlgorithm(QgsProcessingAlgorithm):
    """
    Vereinfachtes Schallraster (Gerüst):
    - Distanz-basiertes Modell (A_div + A_atm), ohne Band- & CNOSSOS/ISO-Details.
    - Optional: spätere LoS/Abschirmung integrieren.
    """
    INPUT_DTM = 'INPUT_DTM'
    INPUT_TURB = 'INPUT_TURB'
    FIELD_LWA = 'FIELD_LWA'
    DEFAULT_LWA = 'DEFAULT_LWA'
    ALPHA_KM = 'ALPHA_KM'
    USE_VIEW = 'USE_VIEW'
    OUT_RASTER = 'OUT_RASTER'

    def name(self):
        return 'wea_noiseraster'

    def displayName(self):
        return 'WEA-Schallraster (Gerüst)'

    def group(self):
        return 'Windpark Toolkit'

    def groupId(self):
        return 'wpt'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_DTM, 'DTM (Raster)'))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TURB, 'Turbinen (Punkte)'))
        self.addParameter(QgsProcessingParameterField(self.FIELD_LWA, 'Feld für LwA [dB(A)] (optional)',
                                                      parentLayerParameterName=self.INPUT_TURB, type=QVariant.Double, optional=True))
        self.addParameter(QgsProcessingParameterNumber(self.DEFAULT_LWA, 'Standard LwA [dB(A)]',
                                                       type=QgsProcessingParameterNumber.Double, defaultValue=105.0))
        self.addParameter(QgsProcessingParameterNumber(self.ALPHA_KM, 'Atmosphärische Absorption [dB/km]',
                                                       type=QgsProcessingParameterNumber.Double, defaultValue=1.0))
        self.addParameter(QgsProcessingParameterBoolean(self.USE_VIEW, 'Sicht/Abschirmung berücksichtigen (Stub/ToDo)', defaultValue=False))
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUT_RASTER, 'Gesamtpegel Lp [dB(A)]'))

    def processAlgorithm(self, parameters, context, feedback):
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
        if dtm_layer is None:
            raise QgsProcessingException('DTM ungültig.')
        turb_layer = self.parameterAsVectorLayer(parameters, self.INPUT_TURB, context)
        if turb_layer is None:
            raise QgsProcessingException('Turbinen-Layer ungültig.')

        field_lwa = self.parameterAsString(parameters, self.FIELD_LWA, context)
        default_lwa = self.parameterAsDouble(parameters, self.DEFAULT_LWA, context)
        alpha_km = self.parameterAsDouble(parameters, self.ALPHA_KM, context)
        use_view = self.parameterAsBoolean(parameters, self.USE_VIEW, context)

        # Open DTM with GDAL
        dtm_src = dtm_layer.source()
        ds = gdal.Open(dtm_src, gdal.GA_ReadOnly)
        if ds is None:
            raise QgsProcessingException('DTM kann nicht mit GDAL geöffnet werden: %s' % dtm_src)

        X, Y, gt, cols, rows = ru.geo_grid(ds)
        acc_lin = np.zeros((rows, cols), dtype=np.float64)

        # Coordinate transforms
        crs_dtm = dtm_layer.crs()
        crs_turb = turb_layer.crs()
        if crs_turb != crs_dtm:
            xform = QgsCoordinateTransform(crs_turb, crs_dtm, QgsProject.instance())
        else:
            xform = None

        total = turb_layer.featureCount()
        for i, f in enumerate(turb_layer.getFeatures()):
            if feedback.isCanceled():
                break
            p = f.geometry().asPoint()
            if xform:
                p = xform.transform(p)

            LwA = default_lwa
            if field_lwa and field_lwa in f.fields().names():
                val = f[field_lwa]
                if val is not None:
                    try:
                        LwA = float(val)
                    except Exception:
                        pass

            # Distance grid
            dx = X - p.x()
            dy = Y - p.y()
            r = np.hypot(dx, dy)  # meter, assuming projected CRS

            Lp = level_from_point_source(LwA, r, alpha_db_per_km=alpha_km)

            # TODO: use viewshed mask to add extra attenuation where LoS is false
            # if use_view: integrate GRASS r.viewshed here (mask->additional damping)

            acc_lin += 10.0**(Lp/10.0)
            feedback.setProgress(int(100.0 * (i+1) / max(1,total)))

        L_total = 10.0 * np.log10(np.maximum(acc_lin, 1e-30))

        out_path = self.parameterAsOutputLayer(parameters, self.OUT_RASTER, context)
        ru.write_like(ds, L_total.astype(np.float32), out_path, nodata=None)
        return { self.OUT_RASTER: out_path }
