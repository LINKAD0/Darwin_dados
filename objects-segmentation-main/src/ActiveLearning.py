import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import pdb
from scipy.spatial import distance
import pandas as pd
import shutil
import pathlib
from glob import glob
import os
import csv
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
# ==== Active learning

def getFilenamesFromFolder(path, ext = 'npz'):
    filenames = glob(path + '/*.{}'.format(ext))
    return filenames

def loadFromFolder(path, output_path, npz_filename = "aspp_values.npz"):

    FILE_PATH = Path(os.path.join(output_path, npz_filename))
    if not FILE_PATH.exists():
        print("Creating {}".format(npz_filename))

        filenames = getFilenamesFromFolder(path)
        print("filenames 2", filenames)
        for idx, filename in enumerate(tqdm(filenames)):
            if idx == 0:
                values = np.expand_dims(np.load(filename)['arr_0'], axis=0)
            else:
                values = np.concatenate((values, 
                    np.expand_dims(np.load(filename)['arr_0'], axis=0)), axis=0)
        np.savez(str(FILE_PATH), values)
    else:
        print("Loading {}".format(npz_filename))
        values = np.load(str(FILE_PATH))['arr_0']
    return values

def loadFromFolderPool(output_path, npz_filename_aspp_values = "aspp_values.npz",
    npz_filename_mean_uncertainty = "mean_uncertainty.npz",
    filename_paths = "paths.csv"):

    print("Creating {}".format(npz_filename_aspp_values))
    path_aspp_features = os.path.join(output_path, 'aspp_features')
    path_mean_uncertainty = os.path.join(output_path, 'mean_uncertainty')
    
    filenames = [os.path.basename(x) for x in getFilenamesFromFolder(path_aspp_features)]
    
    '''    
    filenames_mean_uncertainty = [os.path.basename(x) for x in getFilenamesFromFolder(path_aspp_features)]
    
    print(len(filenames_aspp_features))

    filenames_subtracted = set([x.split('.')[0] for x in filenames_mean_uncertainty]).difference(set([x.split('.')[0] for x in filenames_aspp_features]))
    print("filenames_subtracted", len(filenames_subtracted))
    
    filenames_subtracted = set([x.split('.')[0] for x in filenames_aspp_features]).difference(set([x.split('.')[0] for x in filenames_mean_uncertainty]))
    print("filenames_subtracted", len(filenames_subtracted))
    '''
    # pdb.set_trace()
    inferenceResults = lambda: None

    inferenceResults.paths_images = filenames
    
    FILE_PATH = Path(os.path.join(output_path, npz_filename_aspp_values))
    FILE_PATH_MEAN = Path(os.path.join(output_path, npz_filename_mean_uncertainty))
    FILE_PATH_PATHS = Path(os.path.join(output_path, filename_paths))

    if not FILE_PATH.exists():
        load_aspp_flag = True
    else:
        load_aspp_flag = False
        print("Loading {}".format(npz_filename_aspp_values))
        inferenceResults.encoder_values = np.load(str(FILE_PATH))['arr_0']

    if not FILE_PATH_MEAN.exists():
        load_mean_uncertainty_flag = True
    else:
        load_mean_uncertainty_flag = False
        print("Loading {}".format(npz_filename_mean_uncertainty))
        inferenceResults.uncertainty_values_mean = np.load(str(FILE_PATH_MEAN))['arr_0']

    
    if FILE_PATH.exists() and FILE_PATH_MEAN.exists():
        inferenceResults.paths_images = pd.read_csv(FILE_PATH_PATHS)
        inferenceResults.paths_images = inferenceResults.paths_images["path_name"].values.tolist()
        return inferenceResults

    # print("paths", path_aspp_features)

    args = []
    for filename in filenames:
        args.append((filename, path_aspp_features, path_mean_uncertainty, load_aspp_flag, load_mean_uncertainty_flag))
    
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(
            tqdm(pool.imap(loadSample, args), total=len(args))
            )
    results = list(filter(lambda item: item is not None, results))
    print(len(results))

    if not FILE_PATH.exists():
        inferenceResults.encoder_values = np.array([x[0] for x in results]).astype(np.float32)
        np.savez(str(FILE_PATH), inferenceResults.encoder_values)

    
    if not FILE_PATH_MEAN.exists():
        inferenceResults.uncertainty_values_mean = np.array([x[1] for x in results]).astype(np.float32)
        np.savez(str(FILE_PATH_MEAN), inferenceResults.uncertainty_values_mean)

    
    # print(len(inferenceResults.paths_images),
    #     len(np.array([x[2] for x in results])))
    # pdb.set_trace()
    if not FILE_PATH_PATHS.exists():
        inferenceResults.paths_images = np.array([x[2] for x in results])
        df = pd.DataFrame({'path_name': inferenceResults.paths_images})        
        df.to_csv(FILE_PATH_PATHS, index=False, header=True)

    return inferenceResults

def loadSample(args):
    filename, path_aspp_features, path_mean_uncertainty, load_aspp_flag, load_mean_uncertainty_flag = args
    try:
        if load_aspp_flag == True:
            path = os.path.join(path_aspp_features, filename)
            aspp_values = np.load(path)['arr_0']
        else:
            aspp_values = None

        if load_mean_uncertainty_flag == True:
            path = os.path.join(path_mean_uncertainty, filename)
            path = path.replace('.npz', '.csv')
            mean_uncertainty = loadCsv(path)        
        else:
            mean_uncertainty = None
    except:
        return None
    return [aspp_values, mean_uncertainty, filename]

def loadCsv(filename):
    with open(filename) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        data_read = [row for row in reader][0]
    return data_read[1]

def loadFromCsvs(path):
    filenames = getFilenamesFromFolder(path, ext='csv')
    
    values = []
    for filename in tqdm(filenames):
        try:
            values.append(loadCsv(filename))
        except:
            continue

    return np.array(values)

#def loadFromCsv(output_path):
#    mean_uncertainty_path = 

def ignore_already_computed(list_input_files, output_csv):
    df = pd.read_csv(output_csv)
    list_output_files = df['filename'].tolist()
    
    reduced_input_files = list(set(list_input_files).difference( set(list_output_files)))
    
    print('total number of uncertainty files: {}'.format(len(list_input_files)))
    print('total of uncertainty files processsed: {}'.format(len(list_output_files)))
    print('total remaining uncertainty files: {}'.format(len(reduced_input_files)))

    return reduced_input_files
    # else:

    # return values, filenames

def getUncertaintyMeanBuffer(values, buffer_mask_values):
    print(buffer_mask_values.shape)
    # pdb.set_trace()
    values = np.ma.array(values, mask = buffer_mask_values)
    mean_values = np.ma.mean(values, axis=(1, 2))
    print("mean_values", mean_values.shape)

def getTopRecommendations(mean_values, K=500):

    sorted_idxs = np.argsort(mean_values, axis=0)
    sorted_values = np.flip(np.sort(mean_values, axis=0))

    recommendation_idxs = np.flip(sorted_idxs)[:K]

    return sorted_values[:K], recommendation_idxs


def getRepresentativeSamplesFromCluster(values, recommendation_idxs, k=250):

    '''
    values: shape (n_samples, feature_len)
    '''
    '''
    pca = PCA(n_components = n_components)
    pca.fit(values)
    values = pca.transform(values)
    # print(pca.explained_variance_ratio_)
    '''

    verbose = 1
    print("Cluster verbose:", verbose)
    cluster = KMeans(n_clusters = k, verbose=verbose) # n_init = 'auto', 
    #cluster = MiniBatchKMeans(n_clusters = k, batch_size = 70000, verbose=verbose,
    #    n_init = 'auto')
    
    print("Fitting cluster...")

    distances_to_centers = cluster.fit_transform(values)

    print("...Finished fitting cluster.")
    

    print("values.shape", values.shape)
    print("distances_to_centers.shape", distances_to_centers.shape)

    representative_idxs = []
    for k_idx in range(k):
       representative_idxs.append(np.argmin(distances_to_centers[:, k_idx]))
    representative_idxs = np.array(representative_idxs)
    representative_idxs = np.sort(representative_idxs, axis=0)
    print("Number of selected representative samples", len(representative_idxs))

    min_distance_to_centers = np.min(distances_to_centers, axis=1)
    closer_cluster_ids = np.argmin(distances_to_centers, axis=1)
    

    # return representative_idxs, recommendation_idxs[representative_idxs]
    return representative_idxs, recommendation_idxs[representative_idxs], min_distance_to_centers, closer_cluster_ids



def getDistanceToList(value, train_values):
    distance_to_train = np.inf
    for train_value in train_values:
        distance_ = distance.cosine(value, train_value)
        if distance_ < distance_to_train:
            distance_to_train = distance_
    return distance_to_train
def getDistancesToSample(values, sample):
    distances = []
    for value in values:
        # print(len(values), value.shape, sample.shape)
        distances.append(distance.cosine(value, sample))
    return distances

def getSampleWithLargestDistance(distances, mask):
    distances = np.ma.array(distances, mask = mask)
    # ic(np.ma.count_masked(distances))
    # ic(np.unique(mask, return_counts=True))
    # pdb.set_trace()
    return np.ma.argmax(distances, fill_value=0)
    # return np.ma.max(distances, fill_value=0), np.ma.argmax(distances, fill_value=0)

def getRepresentativeSamplesFromDistance(values, recommendation_idxs, train_values, k=250, mode='max_k_cover'):

    '''
    values: shape (n_samples, feature_len)
    train_values: shape (train_n_samples, feature_len)
    '''
    '''
    pca = PCA(n_components = 100)
    pca.fit(values)
    values = pca.transform(values)
    train_values = pca.transform(train_values)
    '''

    distances_to_train = []
    representative_idxs = []
    for value in values:
        distance_to_train = getDistanceToList(value, train_values)
        distances_to_train.append(distance_to_train)
    distances_to_train = np.array(distances_to_train)

    values_selected_mask = np.zeros((len(values)), dtype=np.bool)
    for k_idx in range(k):
        # print(k_idx)
        selected_sample_idx = getSampleWithLargestDistance(
            distances_to_train, 
            mask = values_selected_mask)
        representative_idxs.append(selected_sample_idx)
        
        values_selected_mask[selected_sample_idx] = True
        # values.pop(selected_sample_idx)
        # distances_to_train.pop(selected_sample_idx)
        
        distances_to_previously_selected_sample = getDistancesToSample(
            values,
            values[selected_sample_idx]
        )
        
        for idx, value in enumerate(values):
            # ic(distances_to_train[idx])
            # ic(selected_sample)
            # pdb.set_trace()
            distances_to_train[idx] = np.minimum(distances_to_train[idx], 
                distances_to_previously_selected_sample[idx])
    representative_idxs = np.array(representative_idxs)
    # print("1", values_selected_mask.argwhere(values_selected_mask == True))
    # print("2", len(representative_idxs))
    representative_idxs = np.sort(representative_idxs, axis=0)
    
    return representative_idxs, recommendation_idxs[representative_idxs]

def getRepresentativeAndUncertain(values, recommendation_idxs, representative_idxs):
    return values[recommendation_idxs][representative_idxs] # enters N=100, returns k=10

def getRandomIdxs(len_vector, n, not_considered_idxs = None):
    idxs = np.arange(len_vector)
    idxs = np.delete(idxs, not_considered_idxs)
    np.random.shuffle(idxs)
    idxs = idxs[:n]
    
    return idxs

class ActiveLearner():
    def __init__(self, config):
        self.config = config

        self.k = self.config['k']
        self.recommendation_idxs_path = self.config['output_path'] + \
            '/inference/recommendation_idxs_' + \
            str(self.config['exp_id']) + '.npy'

    def setTrainEncoderValues(self, train_encoder_values):
        self.train_encoder_values = train_encoder_values

    def setBufferMaskValues(self, buffer_mask_values):
        self.buffer_mask_values = buffer_mask_values

    def getTopRecommendations(self, uncertainty_values_mean):
        
        if self.config['diversity_method'] == "None" or self.config['diversity_method'] == None:        
            K = self.k
        else:
            K = self.k * self.config['beta']

        print("k: {}, K: {}".format(self.k, K))
        print("Getting most uncertain samples...")
        ## pdb.set_trace()
        self.sorted_values, self.recommendation_idxs = getTopRecommendations(
            uncertainty_values_mean, K=K)
        ## pdb.set_trace()
        print("...Finished getting most uncertain samples")

    def getDiversityRecommendationsCluster(self, encoder_values, train_encoder_values = None):
        ## pdb.set_trace()
        print("Getting representative samples from cluster...") 
        self.representative_idxs, self.recommendation_idxs, self.min_distance_to_centers, self.closer_cluster_ids = getRepresentativeSamplesFromCluster(
            encoder_values[self.recommendation_idxs], 
            self.recommendation_idxs, 
            k=self.k)
            
        print("...Finished getting representative samples from cluster") 
        self.sorted_values = self.sorted_values[self.representative_idxs]

    def getDiversityRecommendationsDistance(self, encoder_values, train_encoder_values = None):

        print("Getting representative samples from distance to train...") 

        self.representative_idxs, self.recommendation_idxs = getRepresentativeSamplesFromDistance(
            encoder_values[self.recommendation_idxs], 
            self.recommendation_idxs, 
            train_values = train_encoder_values, 
            k=self.k)
                
        print("...Finished getting representative samples from distance to train") 
        self.sorted_values = self.sorted_values[self.representative_idxs]

    def getRandomIdxsForPercentage(self, len_vector):
        sample_n_with_random_percentage = int(
            self.config['k'] * self.config['random_percentage'])

        print("sample_n with random percentage:", sample_n_with_random_percentage)

        recommendation_idxs_with_random_percentage = getRandomIdxs(len_vector, 
            sample_n_with_random_percentage,
            self.recommendation_idxs)

        self.recommendation_idxs[-sample_n_with_random_percentage:] = recommendation_idxs_with_random_percentage        

    def saveSelectedImageNames(self, query_image_names, uncertainty, csv_id = "", ext = "png"):
        
        query_image_names = ["{}.{}".format(x.split('.')[0], ext) for x in query_image_names]
        ## print("sorted name IDs", query_image_names)

        #  convert array into dataframe
        df = pd.DataFrame({'name': query_image_names})
        df['uncertainty'] = uncertainty

        df['platform'] = [x.split('_')[0] for x in query_image_names]

        df = df.reset_index(drop=True)
        df = df.round(5)
        # save the dataframe as a csv file
        path = pathlib.Path(
            self.config['output_path'] + \
                '/active_learning/')
        path.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(path / "query_image_names{}.csv".format(csv_id)),
            index=False)

    def saveSelectedImageNamesWithData(self, query_image_names, uncertainty, 
        min_distance_to_centers, closer_cluster_ids, csv_id = "", ext = "png"):
        
        query_image_names = ["{}.{}".format(x.split('.')[0], ext) for x in query_image_names]
        ## print("sorted name IDs", query_image_names)

        #  convert array into dataframe
        df = pd.DataFrame({'name': query_image_names})

        df['uncertainty'] = uncertainty
        df['min_distance_to_centers'] = min_distance_to_centers
        df['closer_cluster_ids'] = closer_cluster_ids
        df['platform'] = [x.split('_')[0] for x in query_image_names]

        df = df.reset_index(drop=True)
        df = df.round(5)
        # print(df)
        # save the dataframe as a csv file
        path = pathlib.Path(
            self.config['output_path'] + \
                '/active_learning/')
        path.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(path / "query_image_names{}.csv".format(csv_id)),
            index=False, header=True)

    def loadData(self):
        
        print("Starting data loading...")

        print("Starting loading intermediate ASPP values and mean uncertainty...")
        self.inferenceResults = loadFromFolderPool(
            self.config['output_path']
        )

        print("...Finished loading intermediate ASPP values and mean uncertainty")

        if self.config['diversity_method'] == 'distance_to_train':
            self.inferenceResults.train_encoder_values = loadFromFolder(
                os.path.join('/'.join(self.config['output_path'].split('/')[:-1])+'_train', 'aspp_features'),
                self.config['output_path'],
                npz_filename='aspp_values_train.npz'
            )
        print("...Finished data loading")

        # self.inferenceResults.paths_images = [os.path.basename(x) for x in self.inferenceResults.paths_images]

        print("Encoder shape: ", self.inferenceResults.encoder_values.shape)
        print("Uncertainty shape: ", self.inferenceResults.uncertainty_values_mean.shape)
        print("Image path len: ", len(self.inferenceResults.paths_images))


    def run(self):

        # Get filenames from paths
        filenames = self.inferenceResults.paths_images

        self.config['random_percentage'] = float(self.config['random_percentage'])
        # ============ Reduce samples from cubemap faces to 360 samples
        filenames_360 = ['_'.join(x.split('_')[1:3]) for x in filenames]
        
        df = pd.DataFrame(filenames_360, columns=['names'])
        # print(len(df), len(self.inferenceResults.uncertainty_values_mean.tolist()))
        # pdb.set_trace()
        df['uncertainty_mean'] = self.inferenceResults.uncertainty_values_mean.tolist()
        # df['uncertainty_mean'] = df['uncertainty_mean'].astype('float32')
        encoder_len = self.inferenceResults.encoder_values.shape[1]
        dfs_encoder = []
        for idx in range(encoder_len):
            dfs_encoder.append(
                pd.DataFrame(self.inferenceResults.encoder_values[:, idx], columns = ['encoder_' + str(idx)])
            )
        df = pd.concat((df, *dfs_encoder), axis=1)
        df = df.copy()
        
        df_reduced = df.groupby("names",as_index=False).mean().reindex(columns=df.columns)

        self.inferenceResults.uncertainty_values_mean = df_reduced["uncertainty_mean"].to_numpy()

        filenames = df_reduced["names"]
        self.len_vector = self.inferenceResults.uncertainty_values_mean.shape[0]

        encoder_names = ['encoder_'+str(x) for x in range(encoder_len)]
        self.inferenceResults.encoder_values = df_reduced[encoder_names].to_numpy()
        # ============ End reduce samples from cubemap faces to 360 samples

        if self.config['random_percentage'] == 1:
            # If 100% random selection, select and exit early
            self.recommendation_idxs = getRandomIdxs(self.len_vector, self.len_vector)
        else:

            self.getTopRecommendations(self.inferenceResults.uncertainty_values_mean)
            self.preselected_recommendation_idxs = self.recommendation_idxs
            if self.config['diversity_method'] == "cluster":
                self.getDiversityRecommendationsCluster( 
                    self.inferenceResults.encoder_values)
            elif self.config['diversity_method'] == "distance_to_train":
                self.getDiversityRecommendationsDistance( 
                    self.inferenceResults.encoder_values, 
                    train_encoder_values = self.inferenceResults.train_encoder_values)

            if self.config['random_percentage'] > 0:
                print("Selecting with random percentage: {}%".format(
                    self.config['random_percentage']))
                pdb.set_trace()
                self.getRandomIdxsForPercentage(self.len_vector)
       
        # get selected image names
        self.query_image_names = np.array(filenames)[self.recommendation_idxs]
        self.preselected_image_names = np.array(
            filenames)[self.preselected_recommendation_idxs]

        print("Number of pre-selected images", len(self.preselected_image_names))
        print("Number of selected images", len(self.query_image_names))

        # print(self.min_distance_to_centers.shape)
        # pdb.set_trace()
        self.saveSelectedImageNames(np.array(filenames),
            self.inferenceResults.uncertainty_values_mean,
            csv_id="_all")

        self.saveSelectedImageNamesWithData(self.preselected_image_names,
            self.inferenceResults.uncertainty_values_mean[self.preselected_recommendation_idxs],
            self.min_distance_to_centers, 
            self.closer_cluster_ids,
            csv_id="_preselected")

        '''
        print(self.inferenceResults.uncertainty_values_mean.shape,
            self.inferenceResults.uncertainty_values_mean[self.recommendation_idxs].shape)
        pdb.set_trace()
        '''
        self.saveSelectedImageNamesWithData(self.query_image_names,
            self.inferenceResults.uncertainty_values_mean[self.recommendation_idxs],
            self.min_distance_to_centers[self.representative_idxs], 
            self.closer_cluster_ids[self.representative_idxs]
            )


        if self.config['copy_2D_images'] == True:
            print("Copying selected images...")
            self.saveSelectedImages(self.query_image_names)
            print("...Finished copying selected images")

        # maybe to do: create list of faces (add face id) and copy selected face files

        
        ## print("sorted mean uncertainty", self.sorted_values)
    
    def copy_files_to_folder(self, input_path, output_path,
        query_image_names, ext = 'png'):
        save_path = pathlib.Path(
            output_path)
        save_path.mkdir(parents=True, exist_ok=True)
    
        for file in query_image_names:
            try:
                file = '{}.{}'.format(file.split('.')[0], ext)
                shutil.copyfile(input_path + file, 
                    str(save_path / file))
            except Exception as e:
                print(e)

    def saveSelectedImages(self, query_image_names):
        
        # print(query_image_names)

        self.copy_files_to_folder(
            input_path = self.config['image_path'],
            output_path = self.config['output_path'] + '/active_learning/query_images/2D_images/',
            query_image_names = query_image_names,
            ext = 'png'
            )


    '''               
    def saveSelectedImages(self, query_image_names):
        
        print(query_image_names)

        self.copy_files_to_folder(
            input_path = self.config['image_path'],
            output_path = self.config['output_path'] + '/active_learning/query_images/imgs/',
            query_image_names = query_image_names,
            ext = 'png'
            )

        self.copy_files_to_folder(
            input_path = self.config['output_path'] + \
                    '/segmentations/',
            output_path = self.config['output_path'] + '/active_learning/query_images/segmentations/',
            query_image_names = query_image_names,
            ext = 'png'            
            )

        self.copy_files_to_folder(
            input_path = self.config['output_path'] + \
                    '/uncertainty_map/',
            output_path = self.config['output_path'] + '/active_learning/query_images/uncertainty/',
            query_image_names = query_image_names,
            ext = 'npz'
            )
    '''
