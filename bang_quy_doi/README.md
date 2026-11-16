# F-Z conversion tables

Each of the eight CSVs has two columns:

- `F_km2`: physical water-surface area, square kilometres.
- `Z_m`: water-surface elevation in metres in the source-specific datum.

Convert the predicted mask back to its original fixed projected grid first.
For this dataset, `F_km2 = native_water_pixels * 100 / 1e6`. The 100 m2 factor
belongs to the native 10 m pixels, not pixels after model resizing. Interpolate
Z piecewise-linearly within table support only. Outside support, report a
missing/out-of-support Z; do not silently clamp or extrapolate.

## Evidence and provenance

| File | Evidence |
| --- | --- |
| `ankhe_fz.csv` | Existing project transcription of Table 2 of the An Khe study, attributed to Decision 293/QD-BCT, 17 January 2012. Original legal source not revalidated by this recovery. |
| `kanak_fz.csv` | Existing project transcription of a user-supplied Appendix 2 scan associated with Decision 293/QD-BCT (2012). Original scan not newly recovered in this review ZIP. |
| `thirlmere_fz.csv` | Existing monotone empirical area/level calibration knots from 24 image-QC-passing Sentinel-2 / EA gauge pairs marked Good. NOT an official engineering or surveyed bathymetric table. |
| `barrios_de_luna_fz.csv` | Official CHDuero planning/model table, Table 66, PDF page 20, printed page 172 of Anejo 6, Chapter 7 (2012). |
| `riano_fz.csv` | Official CHDuero planning/model table, Table 96, PDF page 19, printed page 251 of Anejo 6, Chapter 8 (2012). |
| `el_vado_fz.csv` | USBR 2018 area table at one-foot increments, PDF page 17, printed page 11. |
| `lovewell_fz.csv` | USBR 2020 equation-base elevations and AREA Ab, PDF pages 11-12. |
| `lake_tschida_fz.csv` | USBR 2013 equation-base elevations and A2 areas, PDF pages 2-3. |

The additional five source tables were rebuilt from official PDFs and checked
against the recovered builder. Detailed URLs, units, page numbers, row counts,
extraction procedures and vertical datums are in `additional_five_sources.json`.
Its `source_file` values are names from the source archive, not dependencies
required by training. Source PDFs and extraction scripts are retained in the
separate recovered project, outside this compact review package.

Conversions: hectares * 0.01 -> km2; acres * 0.0040468564224 -> km2;
feet * 0.3048 -> metres. Do not mix reservoir-specific vertical datums or
interpret Z as a common global reference. Kielder is not included.

The Thirlmere calibration's original individual gauge/image pairs are not
bundled here. It cannot be represented as independently revalidated by the
recovery. Seven of the 29 masks have areas outside its 2.6068-2.9952 km2
support and must remain explicitly flagged. See `code/dataset_audit.json`.
