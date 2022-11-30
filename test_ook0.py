from pathlib import Path

import numpy as np
from tdfpy import timsdata
from tdfpy.pandas_tdf import PandasTdf


analysis_dir = r"C:\data\test.d"

with timsdata.timsdata_connect(analysis_dir) as td:

    pd_tdf = PandasTdf(str(Path(analysis_dir) / 'analysis.tdf'))
    precursors_df = pd_tdf.precursors
    frames_df = pd_tdf.frames
    pasef_frames_msms_info_df = pd_tdf.pasef_frame_msms_info

    parent_id_to_rt = {int(frame_id): rt for frame_id, rt in frames_df[['Id', 'Time']].values}
    parent_id_to_max_tof_scan = {int(frame_id): num_scans for frame_id, num_scans in frames_df[['Id', 'NumScans']].values}

    precursor_id_to_msms_frames = {}
    for frame_id, precursor in pasef_frames_msms_info_df[['Frame', 'Precursor']].values:
        precursor_id_to_msms_frames.setdefault(precursor, []).append(frame_id)

    precursor_id_to_collision_energy = {int(prec_id): ce for prec_id, ce in
                                        pasef_frames_msms_info_df[['Precursor', 'CollisionEnergy']].values}

    precursor_to_frames = {}
    for frame, prec_id in pasef_frames_msms_info_df[['Frame', 'Precursor']].values:
        precursor_to_frames.setdefault(prec_id, []).append(frame)

    precursors_df.dropna(subset=['MonoisotopicMz', 'Charge'], inplace=True)


    for _, precursor_row in precursors_df.iterrows():

        precursor_id = int(precursor_row['Id'])
        parent_id = int(precursor_row['Parent'])
        charge = int(precursor_row['Charge'])
        ook0 = td.scanNumToOneOverK0(parent_id, [precursor_row['ScanNumber']])[0]
        ccs = timsdata.oneOverK0ToCCSforMz(ook0, charge, precursor_row['MonoisotopicMz'])
        mz = precursor_row['MonoisotopicMz']
        prec_intensity = precursor_row['Intensity']

        mz_spectra, intensity_spectra, ook0_spectra = [], [], []

        scan_number_map = {}
        for msms_frame in precursor_id_to_msms_frames[precursor_id]:
            tof_scan_data = td.readScansByNumber(msms_frame, 0, parent_id_to_max_tof_scan[parent_id])

            # build frame precursor search
            for scan_number, (indexes, intensities) in enumerate(tof_scan_data):
                if scan_number not in scan_number_map:
                    scan_number_map[scan_number] = ([], [])
                scan_number_map[scan_number][0].extend(td.indexToMz(msms_frame, indexes))
                scan_number_map[scan_number][1].extend(intensities)
                #ook0_spectra.extend([td.scanNumToOneOverK0(msms_frame, [scan_number])]*len(indexes))

        """for scan_number in scan_number_map:
            print(scan_number, scan_number_map[scan_number])
        """
        len_of_sepctra = [len(scan_number_map[scan_number][0]) for scan_number in scan_number_map]
        #mz_spectra, intensity_spectra, ook0_spectra = zip(*sorted(zip(mz_spectra, intensity_spectra, ook0_spectra)))
        mz_arr, int_arr = td.readPasefMsMs([precursor_id])[precursor_id]

        print(f'old spectra: {len(mz_arr)}, new spectra: {sum(len_of_sepctra)}')

