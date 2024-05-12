import pandas as pd
import numpy as np
from collections import Counter
import sys
import scipy.stats
from shapely.geometry import LineString

class EvalUtils(object):
     """
     some commonly-used evaluation tools and functions
     """
     
     @staticmethod
     def filter_zero(arr):
         """
         remove zero values from an array
         :param arr: np.array, input array
         :return: np.array, output array
         """
         arr = np.array(arr)
         filtered_arr = np.array(list(filter(lambda x: x != 0., arr)))
         return filtered_arr

     @staticmethod
     def arr_to_distribution(arr, min, max, bins):
         """
         convert an array to a probability distribution
         :param arr: np.array, input array
         :param min: float, minimum of converted value
         :param max: float, maximum of converted value
         :param bins: int, number of bins between min and max
         :return: np.array, output distribution array
         """
         distribution, base = np.histogram(
             arr, np.arange(
                 min, max, float(
                     max - min) / bins))
         return distribution, base[:-1]

     @staticmethod
     def norm_arr_to_distribution(arr, bins=100):
         """
         normalize an array and convert it to distribution
         :param arr: np.array, input array
         :param bins: int, number of bins in [0, 1]
         :return: np.array, np.array
         """
         arr = (arr - arr.min()) / (arr.max() - arr.min())
         arr = EvalUtils.filter_zero(arr)
         distribution, base = np.histogram(arr, np.arange(0, 1, 1. / bins))
         return distribution, base[:-1]

     @staticmethod
     def log_arr_to_distribution(arr, min=-30., bins=100):
         """
         calculate the logarithmic value of an array and convert it to a distribution
         :param arr: np.array, input array
         :param bins: int, number of bins between min and max
         :return: np.array,
         """
         arr = (arr - arr.min()) / (arr.max() - arr.min())
         arr = EvalUtils.filter_zero(arr)
         arr = np.log(arr)
         distribution, base = np.histogram(arr, np.arange(min, 0., 1./bins))
         ret_dist, ret_base = [], []
         for i in range(bins):
             if int(distribution[i]) == 0:
                 continue
             else:
                 ret_dist.append(distribution[i])
                 ret_base.append(base[i])
         return np.array(ret_dist), np.array(ret_base)

  
     @staticmethod
     def get_js_divergence(p1, p2):
         """
         calculate the Jensen-Shanon Divergence of two probability distributions
         :param p1:
         :param p2:
         :return:
         """
         # normalize
         p1 = p1 / (p1.sum()+1e-14)
         p2 = p2 / (p2.sum()+1e-14)
         m = (p1 + p2) / 2
         js = 0.5 * scipy.stats.entropy(p1, m) + \
             0.5 * scipy.stats.entropy(p2, m)
         return js

class IndividualEval():
    
    def get_distances(self, trajs):
    # Group data by trajectory identifier (tid)
        grouped = trajs.groupby('tid')

    # Calculate distance for each trajectory
        distances = []
        for tid, group in grouped:
            x = group['lat'].values
            y = group['lon'].values
            for i in range(len(x) - 1):
                dx = x[i + 1] - x[i] 
                dy = y[i + 1] - y[i]
                distances.append(dx**2+dy**2)

    # Convert distances to NumPy array
        distances = np.array(distances, dtype=float)
        return distances

    # # Print distances for each trajectory (just printing the first 25)
    #     print("Distances for each trajectory:")
    #     print(distances[:25])

    def get_gradius (self, trajs):
        gradius = []
        grouped = trajs.groupby('tid')
        for tid, group in grouped:
            x = group['lat'].values
            y = group['lon'].values
            xs = np.array([x[i] for i in range(len(x)-1)])
            ys = np.array([y[i] for i in range(len(x)-1)])
            xcenter, ycenter = np.mean(xs), np.mean(ys)
            dxs = xs - xcenter
            dys = ys - ycenter
            rad = [dxs[i]**2 + dys[i]**2 for i in range(len(x)-1)]
            rad = np.mean(np.array(rad, dtype=float))
            gradius.append(rad)

        gradius = np.array(gradius, dtype=float)
        return gradius
    #   print (gradius[:25])

  
    def hotspots (self, trajs):
        hotspot = []

        trajs['lat'] = trajs['lat'].round(3)
        trajs['lon'] = trajs['lon'].round(3)

        counter = trajs.groupby(['lat', 'lon']).size().reset_index(name='appearances')
        hotspot = np.array(counter['appearances'], dtype=int)

        return hotspot


    
    def get_individual_jsds(self, t1, t2):
            """
            get jsd scores of individual evaluation metrics
            :param t1: test_data
            :param t2: gene_data
            :return:
            """
            #concatenar d1 i d2 i agafar el maxim entre ells per fer max_distance
            d1 = self.get_distances(t1)
            d2 = self.get_distances(t2)

            d_dist = np.concatenate((d1, d2))
            max_distance = np.max(d_dist)
            
            
            d1_dist, _ = EvalUtils.arr_to_distribution(
                d1, 0, max_distance, 10000)
            d2_dist, _ = EvalUtils.arr_to_distribution(
                d2, 0, max_distance, 10000)
            d_jsd = EvalUtils.get_js_divergence(d1_dist, d2_dist)
            

            g1 = self.get_gradius(t1)
            g2 = self.get_gradius(t2)

            g1_dist, _ = EvalUtils.arr_to_distribution(
                g1, 0, max_distance**2, 10000)
            g2_dist, _ = EvalUtils.arr_to_distribution(
                g2, 0, max_distance**2, 10000)
            g_jsd = EvalUtils.get_js_divergence(g1_dist, g2_dist)

            h1 = self.hotspots(t1)
            h2 = self.hotspots(t2)

            h_dist = np.concatenate((h1, h2))
            max_ocurrence = np.max(h_dist)

            h1_hotspot, _ = EvalUtils.arr_to_distribution(
                h1, 0, max_ocurrence, 10000)
            h2_hotspot, _ = EvalUtils.arr_to_distribution(
                h2, 0, max_ocurrence, 10000)
            h_jsd = EvalUtils.get_js_divergence(h1_hotspot, h2_hotspot)

            return d_jsd,  g_jsd, h_jsd
    
if __name__ == '__main__':

    VALIDATION_FILE = sys.argv[1]
    GENERATED_FILE = sys.argv[2]

    test_data = pd.read_csv(VALIDATION_FILE)
    synthetic_data = pd.read_csv(GENERATED_FILE)

    individual_eval = IndividualEval()

    d_jsd, g_jsd, h_jsd = individual_eval.get_individual_jsds(test_data, synthetic_data)
    printer = f'{d_jsd},{g_jsd},{h_jsd}\n'
    print(printer)
            


    