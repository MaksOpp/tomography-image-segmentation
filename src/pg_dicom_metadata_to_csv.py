import glob
import pydicom
import pandas as pd 

import yaml_config
config = yaml_config.getContentFromFile('../config/pg_dicom_converter.yaml')

PG_PATHS_FILE = config['pg_paths']
OUTPUT_FILE = config['output_file']

class PgDicomMetaDataToCsvConverter:
    def __init__(self, path_list):
        self.path_list = path_list
        self.meta_cols = []
        self.first_run = True

    def dicom_directories_to_csv(self):
        lines = self.path_list

        for line in lines:
            ext, source, dest = line.split()

            if(ext == 'dicom'):
                directories = glob.glob(f'{source}/*')
                directories.sort()
                for directory in directories:
                    dicoms_directory = f'{directory}/DICOMS/STU00001'
                    ser_directories = glob.glob(f'{dicoms_directory}/*')
                    ser_directories.sort()

                    for ser_directory in ser_directories:
                        files = glob.glob(f'{ser_directory}/*')
                        files.sort()

                        for file in files:
                            print(file)
                            self.save_file_metadata_to_csv(file)
                            

    def save_file_metadata_to_csv(self, file):
        try:
            data = pydicom.dcmread(file)
            if(self.first_run):
                self.meta_cols = self.get_meta_cols(data)
                self.create_csv_file_on_first_run()
                self.first_run = False

            csv_col_dict = {col: [] for col in self.meta_cols} 

            for col in self.meta_cols:
                csv_col_dict = self.append_csv_col_dict(data, csv_col_dict, col, file)

            if(len(csv_col_dict) > 0):
                df_csv_col_dict = pd.DataFrame(csv_col_dict)
                df_csv_col_dict.to_csv(OUTPUT_FILE, mode='a', index=False, header=False)

        except: 
            pass

    def create_csv_file_on_first_run(self):
        col_dict = {col: [] for col in self.meta_cols} 
        df_csv_col_dict = pd.DataFrame(col_dict)
        df_csv_col_dict.to_csv(OUTPUT_FILE, index=False)

    def get_meta_cols(self, data):
        self.meta_cols = data.dir()
        self.meta_cols.remove('PixelData')
        self.meta_cols.remove('RequestedProcedureDescription')
        self.meta_cols.remove('SeriesDescription')
        self.meta_cols.remove('StudyDescription')
        self.meta_cols.insert(0,'StudyDescription')
        self.meta_cols.insert(0,'SeriesDescription')
        self.meta_cols.insert(0,'RequestedProcedureDescription')
        self.meta_cols.insert(0,'FilePath')
        return self.meta_cols

    def append_csv_col_dict(self, data, csv_col_dict, col, file):
        if hasattr(data, col):
            csv_col_dict[col].append(str(getattr(data, col)))
        elif col == 'FilePath':
            csv_col_dict[col].append(file)
        else:
            csv_col_dict[col].append(None)

        return csv_col_dict

def main():
    pg_dicom_data_to_csv_converter = PgDicomMetaDataToCsvConverter(PG_PATHS_FILE)
    pg_dicom_data_to_csv_converter.dicom_directories_to_csv()

if __name__ == "__main__":
    main()