"""
ingest.py
author: Connor Hainje

usage:
python ingest.py
"""

import duckdb
import logging
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy_healpix import HEALPix
from glob import glob
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import cast

from talltable.paths import DATA_DIR, PIXEL_DB_PATH, IMAGE_DB_PATH


logger = logging.getLogger(__name__)

_idx = np.arange(2040, dtype=np.uint32)
ALL_INDICES = np.stack(np.meshgrid(_idx, _idx)[::-1], -1).reshape(-1, 2)
HEALPIX02 = HEALPix(nside=2**2, order="nested", frame="icrs")
HEALPIX16 = HEALPix(nside=2**16, order="nested", frame="icrs")


# only write data from images which intersect a pixel in this list
# WRITE_ONLY_HP02 = [114]
WRITE_ONLY_HP02 = None


CHUNK_SIZE = 16  # number of files to process at a time


def write_pixels(pixels):
    for k, v in pixels.items():
        pixels[k] = np.concatenate(v)
    table = pa.table(pixels)

    file_options = ds.ParquetFileFormat().make_write_options(
        compression="zstd",
        compression_level=3,
    )

    ds.write_dataset(
        data=table,
        base_dir=PIXEL_DB_PATH,
        partitioning=["hp02"],
        partitioning_flavor="hive",
        format="parquet",
        file_options=file_options,
        basename_template="release_{i}.parquet",
        existing_data_behavior="overwrite_or_ignore",
    )

    logger.info("wrote new pixels to pixel dataset")


def write_images(images):
    if not IMAGE_DB_PATH.exists():
        pq.write_table(pa.table(images), IMAGE_DB_PATH)
        logger.info("wrote new images to parquet file")
        return

    tmp_path = Path(str(IMAGE_DB_PATH) + ".tmp")

    existing_file = pq.ParquetFile(IMAGE_DB_PATH)
    with pq.ParquetWriter(tmp_path, existing_file.schema_arrow) as w:
        for i in range(existing_file.num_row_groups):
            w.write_table(existing_file.read_row_group(i))
        w.write_table(pa.table(images))

    tmp_path.replace(IMAGE_DB_PATH)  # overwrite existing file
    logger.info("wrote new images to parquet file")


def main():
    if IMAGE_DB_PATH.exists():
        previously_ingested = list(
            duckdb.sql(f"SELECT filepath FROM '{IMAGE_DB_PATH}'").fetchnumpy()[
                "filepath"
            ]
        )
    else:
        previously_ingested = []
    previously_ingested = cast(list[str], previously_ingested)

    data_files = glob(str(DATA_DIR / "*.fits"))
    to_ingest = set(data_files) - set(previously_ingested)
    to_ingest = sorted(list(to_ingest))
    logger.info("%d files to ingest", len(to_ingest))

    pixels = None
    images = None

    for index in tqdm(range(len(to_ingest))):
        if pixels is not None and len(pixels["row"]) == CHUNK_SIZE:
            write_pixels(pixels)
            write_images(images)
            pixels = None

        if pixels is None:
            images = {
                "imageid": [],
                "filepath": [],
                "obsid": [],
                "t_beg": [],
                "t_end": [],
            }

            pixels = {
                "row": [],
                "col": [],
                "flux": [],
                "variance": [],
                "zodi": [],
                # "flags": [],
                "known": [],
                "ra": [],
                "dec": [],
                "wavelen": [],
                "waveband": [],
                "ux": [],
                "uy": [],
                "uz": [],
                "hp02": [],
                "hp16": [],
                "imageid": [],
            }

        filepath = to_ingest[index]
        logger.info("processing %s", filepath)

        # load data
        with fits.open(filepath) as hdul:

            def _byteswap(X):
                return X.view(X.dtype.newbyteorder()).byteswap()

            if WRITE_ONLY_HP02 is not None:
                wcs = WCS(header=hdul["IMAGE"].header)
                ra, dec = wcs.wcs_pix2world(ALL_INDICES, 0).T
                partitions = HEALPIX02.skycoord_to_healpix(
                    SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
                )
                partitions = np.unique(partitions)
                okay = False
                for pix in partitions:
                    if pix in WRITE_ONLY_HP02:
                        okay = True
                        break

                if not okay:
                    continue

            flags = hdul["FLAGS"].data[*ALL_INDICES.T]
            good = flags & ~(1 << 21) == 0

            if np.count_nonzero(good) == 0:
                continue

            good_indices = ALL_INDICES[good]
            known = flags[good] & (1 << 21) == 1

            data = {
                "row": good_indices[:, 0],
                "col": good_indices[:, 1],
                "known": known,
            }

            data["flux"] = _byteswap(hdul["IMAGE"].data[*good_indices.T])
            data["variance"] = _byteswap(hdul["VARIANCE"].data[*good_indices.T])
            data["zodi"] = _byteswap(hdul["ZODI"].data[*good_indices.T])
            # data["flags"] = _byteswap(hdul["FLAGS"].data[*good_indices.T])
            logger.info("  loaded pixel data")

            wcs = WCS(header=hdul["IMAGE"].header)
            ra, dec = wcs.wcs_pix2world(good_indices, 0).T
            data["ra"] = ra
            data["dec"] = dec
            logger.info("  computed ra, dec")

            # remove these to prevent SIP warning when loading wavelength WCS
            del hdul["IMAGE"].header["A_ORDER"]
            del hdul["IMAGE"].header["B_ORDER"]
            del hdul["IMAGE"].header["AP_ORDER"]
            del hdul["IMAGE"].header["BP_ORDER"]
            wave_wcs = WCS(header=hdul["IMAGE"].header, fobj=hdul, key="W")
            wave_wcs.sip = None
            lam, dlam = wave_wcs.wcs_pix2world(good_indices, 0).T
            data["wavelen"] = lam
            data["waveband"] = dlam
            logger.info("  computed wavelengths")

            t_beg = hdul["IMAGE"].header["MJD-BEG"]
            t_end = hdul["IMAGE"].header["MJD-END"]
            obsid = hdul["IMAGE"].header["OBSID"]
            imageid = hdul["IMAGE"].header["EXPIDN"]
            data["imageid"] = np.array([imageid for _ in range(len(good_indices))])
            logger.info("  made imageid list")

        data["ux"] = np.cos(np.radians(data["dec"])) * np.cos(np.radians(data["ra"]))
        data["uy"] = np.cos(np.radians(data["dec"])) * np.sin(np.radians(data["ra"]))
        data["uz"] = np.sin(np.radians(data["dec"]))
        logger.info("  computed unit vectors")

        data["hp02"] = HEALPIX02.skycoord_to_healpix(
            SkyCoord(ra=data["ra"], dec=data["dec"], unit="deg", frame="icrs")
        )
        data["hp16"] = HEALPIX16.skycoord_to_healpix(
            SkyCoord(ra=data["ra"], dec=data["dec"], unit="deg", frame="icrs")
        )
        logger.info("  computed healpix indices")

        for k, v in data.items():
            pixels[k].append(v)

        images["imageid"].append(imageid)
        images["filepath"].append(filepath)
        images["obsid"].append(obsid)
        images["t_beg"].append(t_beg)
        images["t_end"].append(t_end)
        logger.info("  saved image metadata")

    if pixels is not None:
        write_pixels(pixels)
        write_images(images)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        main()
