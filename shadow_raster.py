# -*- coding: utf-8 -*-
import math
from datetime import datetime, timedelta, timezone
import numpy as np

from qgis.core import (QgsProcessingAlgorithm, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterVectorLayer, QgsProcessingParameterNumber,
                       QgsProcessingParameterString, QgsProcessingParameterRasterDestination,
                       QgsProcessingException, QgsCoordinateTransform, QgsCoordinateReferenceSystem,
                       QgsProject, QgsFields, QgsField, QgsWkbTypes, QgsFeature, QgsVectorLayer,
                       QgsGeometry, QgsPointXY)
from qgis.PyQt.QtCore import QVariant
from osgeo import gdal, ogr, osr

from ..utils import raster as ru
from ..utils.sun import sunpos_az_alt

class ShadowRasterAlgorithm(QgsProcessingAlgorithm):
    """
    Schattenraster (Gerüst):
    - Geometrische Projektion eines Rechtecks (Rotor) entlang Sonnenazimut.
    - Terrainabschattung wird NICHT berechnet (ToDo: GRASS r.sunmask + DSM).
    - Ausgabe: Minuten/Jahr (oder Zeitraum) je Zelle, die von irgendeiner WEA beschattet sind.
    """
    INPUT_DTM = 'INPUT_DTM'
    INPUT_TURB = 'INPUT_TURB'
    HUB_H = 'HUB_H'
    ROTOR_D = 'ROTOR_D'
    START_DATE = 'START_DATE'
    END_DATE = 'END_DATE'
    STEP_MIN = 'STEP_MIN'
    MIN_ALT = 'MIN_ALT'
    TZ_OFFSET = 'TZ_OFFSET'
    OUT_RASTER = 'OUT_RASTER'

    def name(self):
        return 'wea_shadowraster'

    def displayName(self):
        return 'WEA-Schattenraster (Gerüst)'

    def group(self):
        return 'Windpark Toolkit'

    def groupId(self):
        return 'wpt'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(self.INPUT_DTM, 'DTM (Raster)'))
        self.addParameter(QgsProcessingParameterVectorLayer(self.INPUT_TURB, 'Turbinen (Punkte)'))
        self.addParameter(QgsProcessingParameterNumber(self.HUB_H, 'Nabenhöhe H [m] (falls kein Feld vorhanden)', type=QgsProcessingParameterNumber.Double, defaultValue=150.0))
        self.addParameter(QgsProcessingParameterNumber(self.ROTOR_D, 'Rotordurchmesser D [m]', type=QgsProcessingParameterNumber.Double, defaultValue=130.0))
        self.addParameter(QgsProcessingParameterString(self.START_DATE, 'Startdatum (YYYY-MM-DD)', defaultValue='2025-01-01'))
        self.addParameter(QgsProcessingParameterString(self.END_DATE, 'Enddatum (YYYY-MM-DD)', defaultValue='2025-01-02'))
        self.addParameter(QgsProcessingParameterNumber(self.STEP_MIN, 'Zeitschritt [min]', type=QgsProcessingParameterNumber.Integer, defaultValue=10))
        self.addParameter(QgsProcessingParameterNumber(self.MIN_ALT, 'min. Sonnenhöhe [°] (z.B. 3°)', type=QgsProcessingParameterNumber.Double, defaultValue=3.0))
        self.addParameter(QgsProcessingParameterNumber(self.TZ_OFFSET, 'Zeitzonen-Offset zu UTC [h] (z.B. +1 für MEZ)', type=QgsProcessingParameterNumber.Double, defaultValue=1.0))
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUT_RASTER, 'Schattenzeit [min] (vereinigte Masken)'))

    def processAlgorithm(self, parameters, context, feedback):
        dtm_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DTM, context)
        if dtm_layer is None:
            raise QgsProcessingException('DTM ungültig.')
        turb_layer = self.parameterAsVectorLayer(parameters, self.INPUT_TURB, context)
        if turb_layer is None:
            raise QgsProcessingException('Turbinen-Layer ungültig.')

        H = self.parameterAsDouble(parameters, self.HUB_H, context)
        D = self.parameterAsDouble(parameters, self.ROTOR_D, context)
        start_date = self.parameterAsString(parameters, self.START_DATE, context)
        end_date = self.parameterAsString(parameters, self.END_DATE, context)
        step_min = int(self.parameterAsDouble(parameters, self.STEP_MIN, context))
        min_alt = self.parameterAsDouble(parameters, self.MIN_ALT, context)
        tz_offset = self.parameterAsDouble(parameters, self.TZ_OFFSET, context)

        # Open DTM with GDAL (reference grid)
        dtm_src = dtm_layer.source()
        ds = gdal.Open(dtm_src, gdal.GA_ReadOnly)
        if ds is None:
            raise QgsProcessingException('DTM kann nicht mit GDAL geöffnet werden: %s' % dtm_src)
        X, Y, gt, cols, rows = ru.geo_grid(ds)
        acc = np.zeros((rows, cols), dtype=np.float32)

        # Coordinate systems
        crs_dtm = dtm_layer.crs()
        crs_wgs84 = QgsCoordinateReferenceSystem('EPSG:4326')
        to_wgs = QgsCoordinateTransform(crs_dtm, crs_wgs84, QgsProject.instance())
        if turb_layer.crs() != crs_dtm:
            to_dtm = QgsCoordinateTransform(turb_layer.crs(), crs_dtm, QgsProject.instance())
        else:
            to_dtm = None

        # Collect turbine positions in DTM CRS and lat/lon
        turbines = []
        for f in turb_layer.getFeatures():
            p = f.geometry().asPoint()
            if to_dtm:
                p = to_dtm.transform(p)
            p_wgs = to_wgs.transform(p)
            turbines.append((p, p_wgs))

        if not turbines:
            raise QgsProcessingException('Keine Turbinen gefunden.')

        # Time loop
        try:
            dt0 = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc) - timedelta(hours=tz_offset)
            dt1 = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc) - timedelta(hours=tz_offset)
        except Exception as e:
            raise QgsProcessingException('Datum ungültig. Bitte YYYY-MM-DD verwenden.')

        total_steps = max(1, int((dt1 - dt0).total_seconds() // (step_min*60)))
        pixel_w = gt[1]
        pixel_h = abs(gt[5])

        for step_idx in range(total_steps):
            if feedback.isCanceled():
                break
            dt_local = dt0 + timedelta(minutes=step_min*step_idx)
            # Convert local time to UTC for sunpos (we subtracted tz_offset above => dt_local is UTC)
            dt_utc = dt_local

            # Build a per-step mask (union over all turbines)
            mask = np.zeros((rows, cols), dtype=bool)

            for (p_dtm, p_wgs) in turbines:
                lat = p_wgs.y()
                lon = p_wgs.x()
                az_deg, alt_deg = sunpos_az_alt(dt_utc, lat, lon)
                if alt_deg < min_alt:
                    continue

                # Shadow geometry (rectangle oriented along azimuth)
                L = (H + 0.5*D) / math.tan(math.radians(alt_deg))  # length in meters
                if L <= 0:
                    continue
                half_w = 0.5 * D

                # Azimuth 0=N -> vector in DTM CRS (meters)
                theta = math.radians(az_deg)
                ux, uy = math.sin(theta), math.cos(theta)  # east, north
                vx, vy = -uy, ux  # left

                # Sample along centerline and fill a "corridor" of half_w using simple rasterization
                # Discretize every pixel_w meters
                nseg = max(1, int(L / max(pixel_w, pixel_h)))
                xs = p_dtm.x() + ux * np.linspace(0, L, nseg)
                ys = p_dtm.y() + uy * np.linspace(0, L, nseg)

                # For each sample, fill a cross-section of width D
                for x, y in zip(xs, ys):
                    col = int((x - gt[0]) / gt[1])
                    row = int((y - gt[3]) / gt[5])
                    # Draw a small disk approximating rotor width at ground
                    # Number of pixels radius
                    r_pix = max(1, int(half_w / max(pixel_w, pixel_h)))
                    r2 = r_pix*r_pix
                    rmin = max(0, row - r_pix)
                    rmax = min(rows-1, row + r_pix)
                    cmin = max(0, col - r_pix)
                    cmax = min(cols-1, col + r_pix)
                    yy, xx = np.ogrid[rmin:rmax+1, cmin:cmax+1]
                    cm = (yy - row)**2 + (xx - col)**2 <= r2
                    mask[rmin:rmax+1, cmin:cmax+1] |= cm

            # Accumulate minutes where any turbine casts shadow
            acc[mask] += step_min
            feedback.setProgress(int(100.0 * (step_idx+1) / total_steps))

        out_path = self.parameterAsOutputLayer(parameters, self.OUT_RASTER, context)
        ru.write_like(ds, acc.astype(np.float32), out_path, nodata=0.0)
        return { self.OUT_RASTER: out_path }
