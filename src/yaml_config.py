
import yaml

content = dict()

def takeConfig(filepath):
    if filepath == '../config/cnn.yaml':
        return cnnYaml(filepath)
    elif filepath == '../config/file_converter.yaml':
        return fileConverterYaml(filepath)  
    elif filepath == '../config/pg_dicom_metadata_to_csv.yaml':
        return pgDicomMetadataToCsv(filepath)  
    elif filepath == '../config/split_data.yaml':
        return splitDataYaml(filepath) 
    elif filepath == '../config/train_model.yaml':
        return trainModelYaml(filepath)
    else:
        return {
            'Error': 'Not found'
        }

def getContentFromFile(filePath):
    content = dict()
    with open(filePath, 'r') as stream:
        try:
            content = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return content
        
def cnnYaml(filepath):
    content = getContentFromFile(filepath)
    return content

def trainModelYaml(filepath):
    content = getContentFromFile(filepath)
    return content

def splitDataYaml(filepath):
    content = getContentFromFile(filepath)
    return content

def fileConverterYaml(filepath):
    content = getContentFromFile(filepath)

    lits_paths = []

    path_lits_liver = content['path']['lits']['liver']
    pll_image = path_lits_liver['image']
    pll_image_content = '{} {} {}'.format(pll_image['extension'], pll_image['source'], pll_image['destination'])
    lits_paths.append(pll_image_content)

    pll_label = path_lits_liver['label']
    pll_label_content = '{} {} {}'.format(pll_label['extension'], pll_label['source'], pll_label['destination'])
    lits_paths.append(pll_label_content)

    path_lits_hepatic = content['path']['lits']['hepatic']
    plh_image = path_lits_hepatic['image']
    plh_image_content = '{} {} {}'.format(plh_image['extension'], plh_image['source'], plh_image['destination'])
    lits_paths.append(plh_image_content)

    plh_label = path_lits_hepatic['label']
    plh_label_content = '{} {} {}'.format(plh_label['extension'], plh_label['source'], plh_label['destination'])
    lits_paths.append(plh_label_content)

    pg_paths = []
    path_pg = content['path']['pg']
    pp_liver = path_pg['liver']
    pp_liver_content = '{} {} {}'.format(pp_liver['image'], pp_liver['label'], pp_liver['destination'])
    pg_paths.append(pp_liver_content)

    pp_tumors = path_pg['tumors']
    pp_tumors_content = '{} {} {}'.format(pp_tumors['image'], pp_tumors['label'], pp_tumors['destination'])
    pg_paths.append(pp_tumors_content)

    merge_paths = []
    path_merge = content['path']['merge']
    pm_image = path_merge['image']
    pm_image_content = '{} {}'.format(pm_image['source'], pm_image['destination'])
    merge_paths.append(pm_image_content)

    pm_label = path_merge['label']
    pm_label_content = '{} {}'.format(pm_label['source'], pm_label['destination'])
    merge_paths.append(pm_label_content)

    rotate_paths = []

    path_rotate_liver = content['path']['rotate']['liver']
    prl_image = path_rotate_liver['image']
    prl_image_content = '{} {}'.format(prl_image['source'], prl_image['destination'])
    rotate_paths.append(prl_image_content)

    prl_label = path_rotate_liver['label']
    prl_label_content = '{} {}'.format(prl_label['source'], prl_label['destination'])
    rotate_paths.append(prl_label_content)

    path_rotate_hepatic = content['path']['rotate']['hepatic']
    prh_image = path_rotate_hepatic['image']
    prh_image_content = '{} {}'.format(prh_image['source'], prh_image['destination'])
    rotate_paths.append(prh_image_content)

    prh_label = path_rotate_hepatic['label']
    prh_label_content = '{} {}'.format(prh_label['source'], prh_label['destination'])
    rotate_paths.append(prh_label_content)

    return {
        'lits_paths': lits_paths,
        'pg_paths': pg_paths,
        'merge_paths': merge_paths,
        'rotate_paths': rotate_paths
    }

def pgDicomMetadataToCsv(filepath):
    content = getContentFromFile(filepath)

    pg_paths = []
    path_pg = content['path']['pg']
    pp_liver = path_pg['liver']
    pp_liver_content = '{} {} {}'.format(pp_liver['image'], pp_liver['label'], pp_liver['destination'])
    pg_paths.append(pp_liver_content)

    pp_tumors = path_pg['tumors']
    pp_tumors_content = '{} {} {}'.format(pp_tumors['image'], pp_tumors['label'], pp_tumors['destination'])
    pg_paths.append(pp_tumors_content)

    return {
        'pg_paths': pg_paths,
        'output_file': content['output_file']
    }