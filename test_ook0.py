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
            for i in range(0, parent_id_to_max_tof_scan[parent_id]):
                tof_scan_data = td.extractCentroidedSpectrumForFrame(msms_frame, i, i+1)
                print(i, tof_scan_data)
                ook0 = td.scanNumToOneOverK0(msms_frame, [i])[0]

            # build frame precursor search
                if ook0 not in scan_number_map:
                    scan_number_map[ook0] = ([], [])
                scan_number_map[ook0][0].extend(td.indexToMz(msms_frame, tof_scan_data[0]))
                scan_number_map[ook0][1].extend(tof_scan_data[1])
                    #ook0_spectra.extend([td.scanNumToOneOverK0(msms_frame, [scan_number])]*len(indexes))

        """for scan_number in scan_number_map:
            print(scan_number, scan_number_map[scan_number])
        """

        mz_spectra, intensity_spectra, ook0_spectra = [], [], []
        for ook0 in scan_number_map:
            mz_spectra.extend(scan_number_map[ook0][0])
            intensity_spectra.extend(scan_number_map[ook0][1])
            ook0_spectra.extend([ook0]*len(scan_number_map[ook0][0]))

        mz_spectra = np.array(mz_spectra)
        intensity_spectra = np.array(intensity_spectra)
        ook0_spectra = np.array(ook0_spectra)

        idx = np.argsort(mz_spectra)

        mz_spectra = mz_spectra[idx]
        intensity_spectra = intensity_spectra[idx]
        ook0_spectra = ook0_spectra[idx]

        ppm_tolerance = 5/1_000_000

        combine_spectra = [0]*len(mz_spectra)

        current_mz, group_itr = None, 0
        for i, mz in enumerate(mz_spectra):
            if current_mz is None:
                current_mz = mz
                combine_spectra[i] = group_itr
                continue

            #print(mz, current_mz, current_mz - mz)
            if abs(current_mz - mz) <= current_mz*ppm_tolerance:
                pass
            else:
                group_itr += 1
                current_mz = mz

            combine_spectra[i] = group_itr

        mz_comb, intensity_comb, ook0_comb = [], [], []

        current_idx = 0
        for grp in range(max(combine_spectra)):
            start_idx = current_idx
            end_index = None
            for i, grp2 in enumerate(combine_spectra[start_idx:]):
                if grp != grp2:
                    end_index = current_idx + i
                    current_idx = end_index
                    break

            mz_comb.append(np.median(mz_spectra[start_idx:end_index]))
            intensity_comb.append(sum(intensity_spectra[start_idx:end_index]))
            ook0_comb.append(np.median(ook0_spectra[start_idx:end_index]))

            """print(combine_spectra[start_idx:end_index])
            print(mz_spectra[start_idx:end_index])
            print(ook0_spectra[start_idx:end_index])
            print(intensity_spectra[start_idx:end_index])
            print()"""

        """with open('example_peaks_new.txt', 'w') as f:
            for mz, i, ook0 in zip(mz_comb, intensity_comb, ook0_comb):
                f.write(f'{mz} {i} {ook0}\n')

        with open('example_peaks.txt', 'w') as f:
            for mz, i, ook0 in zip(mz_spectra, intensity_spectra, ook0_spectra):
                f.write(f'{mz} {i} {ook0}\n')"""

        len_of_sepctra = [len(scan_number_map[ook0][0]) for ook0 in scan_number_map]
        #mz_spectra, intensity_spectra, ook0_spectra = zip(*sorted(zip(mz_spectra, intensity_spectra, ook0_spectra)))
        mz_arr, int_arr = td.readPasefMsMs([precursor_id])[precursor_id]

        """with open('example_peaks_old.txt', 'w') as f:
            for mz, i in zip(mz_arr, int_arr):
                f.write(f'{mz} {i}\n')"""

        print(f'old spectra: {len(mz_arr)}, new spectra: {sum(len_of_sepctra)}, new new spectra: {len(mz_comb)}, >9 {len([ints for ints in intensity_comb if ints > 9])}')

