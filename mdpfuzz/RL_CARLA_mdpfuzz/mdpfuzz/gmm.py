import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import time
import copy
import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from typing import List, Tuple, Dict, Union

# or runs python -m src.gmm
if __package__ is None or __package__ == '':
    # uses current directory visibility
    from utils import modify_covariance_matrix, plot_gaussians, generate_clustered_data, create_gif, remove_gaussian, plot_points
else:
    # uses current package visibility
    from .utils import modify_covariance_matrix, plot_gaussians, generate_clustered_data, create_gif, remove_gaussian, plot_points


# float64; 1E3 for float32
TMP = 1E6 * np.finfo('d').eps
MAXIMUM_DENSITY = 10e9

class GMM():
    '''CS statistics are weighted.'''
    def __init__(self, random_seed: int, k: int):
        self.k = k
        self.dim = None # type: int
        self.gamma = None # type: float
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed) # type: np.random.Generator
        self.normal = multivariate_normal
        self.normal.random_state = np.random.default_rng(self.random_seed)

        self.coefficients = None # type: np.ndarray
        self.means = None # type: np.ndarray
        self.covariances = None # type: np.ndarray
        self.cs_means = None # type: np.ndarray
        self.cs_squares = None # type: np.ndarray

        self.allow_singular = True


    def initialize(self, data: np.ndarray):
        '''
        Sets the different attributes of the class.
        Means are weighted.
        '''
        assert len(data) == self.k
        self.dim = data.shape[1] # type: int
        self.coefficients = np.ones(self.k) / self.k
        self.means = copy.deepcopy(data)

        self.covariances = np.zeros((self.k, self.dim, self.dim))
        self.cs_means = np.zeros((self.k, self.dim))
        self.cs_squares = np.zeros((self.k, self.dim, self.dim))
        for i in range(self.k):
            self.covariances[i] = np.eye(self.dim)
            self.cs_squares[i] = np.matmul(data[i:i+1].T, data[i:i+1]) * self.coefficients[i]
            self.cs_means[i] = self.means[i].copy() * self.coefficients[i]


    def set_gamma(self, gamma: float):
        assert gamma > 0.0 and gamma < 1.0
        self.gamma = gamma


    def get_cs_statistics(self, responsibilities: np.ndarray, point: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''Returns the mixing coefficients and CS statistics based on the responsibilities and the data provided.'''
        coefficients = np.ones(self.k)
        cs_means = np.zeros((self.k, self.dim))
        cs_squares = np.zeros((self.k, self.dim, self.dim))
        for i in range(self.k):
            coefficients[i] = (1 - self.gamma) * self.coefficients[i] + self.gamma * responsibilities[i]
            cs_means[i] = (1 - self.gamma) * self.cs_means[i] + self.gamma * responsibilities[i] * point
            cs_squares[i] = (1 - self.gamma) * self.cs_squares[i] + self.gamma * responsibilities[i] * np.matmul(point[:, np.newaxis], point[:, np.newaxis].T)
        return coefficients, cs_means, cs_squares


    def compute_cs_statistics(self, point: np.ndarray):
        '''Computes the responsibilities and returns the updated coefficients and CS statistics based on the current parameters.'''
        # E step
        responsibilities = np.zeros(self.k)
        for i in range(self.k):
            responsibilities[i] = self.coefficients[i] * self.normal.pdf(point, mean=self.means[i], cov=self.covariances[i], allow_singular=self.allow_singular)
        responsibilities += 1e-5
        responsibilities /= sum(responsibilities)
        return self.get_cs_statistics(responsibilities, point)


    def update_parameters(self, coefficients: np.ndarray, cs_means: np.ndarray, cs_squares: np.ndarray):
        '''Updates the means and covariances of the model with the coefficients and CS statistics provided.'''
        for i in range(self.k):
            self.means[i] = cs_means[i] / coefficients[i]
            sigma = (cs_squares[i] - np.matmul(self.means[i][:, np.newaxis], cs_means[i][:, np.newaxis].T)) / coefficients[i]
            # prevents numerical instability
            self.covariances[i] = modify_covariance_matrix(sigma)


    def update(self, point: np.ndarray, responsibilities = None):
        '''Updates the current parameters with the point provided.'''
        if responsibilities is None:
            coefficients, cs_means, cs_squares = self.compute_cs_statistics(point)
        else:
            coefficients, cs_means, cs_squares = self.get_cs_statistics(responsibilities, point)

        self.coefficients = copy.deepcopy(coefficients)
        self.cs_means = copy.deepcopy(cs_means)
        self.cs_squares = copy.deepcopy(cs_squares)
        self.update_parameters(coefficients, cs_means, cs_squares)


    def online_EM(self, states: Union[np.ndarray, List[np.ndarray]], permute: bool = False, gamma: float = None):
        '''
        Online Expectation Maximization.
        It first computes the new coefficients and CS stastistics, and then (deep) copies them and updates the means and covariances accordingly.
        '''
        if permute:
            states = self.rng.permutation(states)
        if gamma is None:
            assert self.gamma is not None

        for j in range(len(states) - 1):
            state = states[j]
            coefficients, cs_means, cs_squares = self.compute_cs_statistics(state)
            self.coefficients = copy.deepcopy(coefficients)
            self.cs_means = copy.deepcopy(cs_means)
            self.cs_squares = copy.deepcopy(cs_squares)
            self.update_parameters(coefficients, cs_means, cs_squares)


    def offline_EM(self, states: Union[np.ndarray, List[np.ndarray]], permute: bool = False):
        '''Expectation Maximization.'''
        if permute:
            states = self.rng.permutation(states)

        num_states = len(states)
        z = np.zeros((num_states, self.k))
        # E step
        for i in range(self.k):
            z[:, i] = self.coefficients[i] * self.normal.pdf(states, mean=self.means[i], cov=self.covariances[i], allow_singular=self.allow_singular)
        z /= np.sum(z, axis=1, keepdims=True)
        # M step
        sum_z = np.sum(z, axis=0)
        self.coefficients = sum_z / num_states
        self.means = np.matmul(z.T, states)
        self.means /= sum_z[:, np.newaxis]
        for i in range(self.k):
            j = np.expand_dims(states, axis=1) - self.means[i]
            s = np.matmul(j.transpose([0, 2, 1]), j)
            self.covariances[i] = np.matmul(s.transpose(1, 2, 0), z[:, i])
            self.covariances[i] /= sum_z[i]
        # sets the CS statistics
        cs_squares = np.matmul(states.T, states) / num_states
        mean = np.mean(states, axis=0)
        for i in range(self.k):
            self.cs_squares[i] = cs_squares * self.coefficients[i]
            self.cs_means[i] = mean * self.coefficients[i]


    def save(self, filepath: str):
        filepath = filepath.split('.json')[0]
        configuration = dict()
        # attributes
        configuration['k'] = self.k
        configuration['gamma'] = self.gamma
        configuration['dim'] = self.dim
        configuration['random_seed'] = self.random_seed
        # along with the initial random seed and the states of the two random generators
        configuration['random_state'] = self.rng.bit_generator.state
        configuration['normal_state'] = self.normal.random_state.bit_generator.state
        # CS statistics and GMM's parameters
        configuration['mixing_coefficients'] = self.coefficients.tolist()
        configuration['means'] = self.means.tolist()
        configuration['covariances'] = self.covariances.tolist()
        configuration['cs_means'] = self.cs_means.tolist()
        configuration['cs_squares'] = self.cs_squares.tolist()

        with open(filepath + '_config.json', 'w') as f:
            f.write(json.dumps(configuration))


    def _load_dict(self, configuration: Dict):
        self.k = configuration['k']
        self.gamma = configuration['gamma']
        self.dim = configuration['dim']
        self.random_seed = configuration['random_seed']

        self.rng = np.random.default_rng(self.random_seed) # type: np.random.Generator
        self.rng.bit_generator.state = configuration['random_state']
        self.normal = multivariate_normal
        self.normal.random_state = np.random.default_rng(self.random_seed)
        self.normal.random_state.bit_generator.state = configuration['normal_state']

        self.coefficients = np.array(configuration['mixing_coefficients'])
        self.means = np.array(configuration['means'])
        self.covariances = np.array(configuration['covariances'])
        self.cs_means = np.array(configuration['cs_means'])
        self.cs_squares = np.array(configuration['cs_squares'])


    def load(self, filepath: str):
        filepath = filepath.split('.json')[0] + '.json'
        assert os.path.isfile(filepath), filepath
        with open(filepath, 'r') as f:
            config = json.load(f)
        self._load_dict(config)


    def log_likelihood(self, data: np.ndarray):
        if len(data.shape) == 1:
            return np.log(sum([self.coefficients[i] * self.normal.pdf(data, mean=self.means[i], cov=self.covariances[i], allow_singular=self.allow_singular) for i in range(self.k)]))
        # data is an array of points
        else:
            log_likelihoods = []
            for x in data:
                tmp = 0.0
                for i in range(self.k):
                    tmp += self.coefficients[i] * self.normal.pdf(x, mean=self.means[i], cov=self.covariances[i], allow_singular=self.allow_singular)
                log_likelihoods.append(np.log(tmp))
            return sum(log_likelihoods)


    def gmm(self, state: np.ndarray, add_offset: bool = True) -> np.ndarray:
        pdf = np.zeros(self.k)
        for i in range(self.k):
            pdf[i] = self.coefficients[i] * self.normal.pdf(state, mean=self.means[i], cov=self.covariances[i], allow_singular=self.allow_singular)
        if add_offset:
            pdf += 1e-5
        return pdf


    def _is_numerically_singular(self, matrix: np.ndarray):
        '''Mimics how scipy checks numerically singular matrices.'''
        eighen_values = np.linalg.eigh(matrix)[0]
        eps = TMP * np.max(abs(eighen_values))
        return np.all(eighen_values > eps) == False


class CoverageModel():
    def __init__(self, random_seed: int, k: int, gamma: float, k_c: int = None) -> None:
        assert k > 0
        assert gamma > 0.0 and gamma < 1.0

        self.gamma = gamma
        self.k_s = k
        self.k_c = k_c if k_c is not None else k

        # random generators
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed) # type: np.random.Generator
        self.normal = multivariate_normal
        self.normal.random_state = np.random.default_rng(self.random_seed)

        self.GMM_s = GMM(self.random_seed, self.k_s)
        self.GMM_s.set_gamma(gamma)
        self.GMM_c = GMM(self.random_seed, self.k_c)
        self.GMM_c.set_gamma(gamma)


    def _concatenate_states(self, states: Union[np.ndarray, List[np.ndarray]], n: int = None) -> np.ndarray:
        if n is None:
            n = len(states)
        else:
            assert n > 0 and (n <= len(states) - 1) # n > 0 in case K=1
        return np.array([np.hstack([states[i], states[i+1]]) for i in range(n)])


    def initialize(self, states: List[np.ndarray]):
        '''Initializes the GMMs\' parameters and the CS statistics.'''

        self.GMM_s.initialize(np.array(states[:self.k_s]))
        states_concatenated = self._concatenate_states(states, n=self.k_c)
        print('concatenated states of shape {}'.format(states_concatenated.shape))
        self.GMM_c.initialize(states_concatenated)


    def sequence_freshness_sheer(self, state_sequence: List[np.ndarray], tau: float = None):
        '''
        ``Sheer'' version of the function, that strictly follows the algorithm.
        As such, it is not optimized, mostly because the joint density probababilities can be computed twice.
        Precisely, to first compute the density, and then during the online update of the GMMs.
        '''
        first_state = state_sequence[0]
        density = self.GMM_s.gmm(first_state, add_offset=True)
        # indices are shifted compared to the definition
        for i in range(1, len(state_sequence)):
            density *= (self.GMM_s.gmm(state_sequence[i], add_offset=True) / self.GMM_c.gmm(np.hstack([state_sequence[i-1], state_sequence[i]]), add_offset=True))

        if (tau is not None) and (density < tau):
            self.dynamic_EM(state_sequence)


    def sequence_freshness(self, states: np.ndarray, states_cond: np.ndarray, tau: float = None):
        '''Computes the freshness of the state sequence and dynamically updates the models if the freshness is higher than tau.'''
        first_state = states[0]
        first_state_pdf = self.GMM_s.gmm(first_state, add_offset=True)
        density = np.sum(first_state_pdf)

        # states
        states_pdf = np.zeros((states.shape[0], self.k_s))
        for i in range(states.shape[0]):
           states_pdf[i] = self.GMM_s.gmm(states[i], add_offset=True)
        # concatenated states
        states_cond_pdf = np.zeros((states_cond.shape[0], self.k_c))
        for i in range(states_cond.shape[0]):
            states_cond_pdf[i] = self.GMM_c.gmm(states_cond[i], add_offset=True)
            tmp = (np.sum(states_cond_pdf[i]) / np.sum(states_pdf[i]))
            density *= tmp
            # avoids useless computations
            if density > MAXIMUM_DENSITY:
                break

        if tau is None or ((tau is not None) and (density < tau)):
            self.dynamic_EM(states, states_cond)
        return density


    def dynamic_EM(self, state_sequence: List[np.ndarray], states_concatenated = None):
        '''Active version of the dynamic EM by calling the online_EM function of the GMM models.'''
        if states_concatenated is None:
            states_concatenated = self._concatenate_states(state_sequence)

        self.GMM_s.online_EM(state_sequence)
        self.GMM_c.online_EM(states_concatenated)


    def sequence_freshness_passive(self, states: np.ndarray, states_cond: np.ndarray, tau: float = None):
        '''
        Computes the freshness of the state sequence and updates the models without recomputing the pdf values.
        As such, even though the models are updated online (i.e., with every state),
        The pdf values used correspond to the ones computed with the parameters before those updates.
        '''
        first_state = states[0]
        first_state_pdf = self.GMM_s.gmm(first_state, add_offset=True)
        density = np.sum(first_state_pdf)

        states_pdf = np.zeros((states.shape[0], self.k_s))
        for i in range(states.shape[0]):
           states_pdf[i] = self.GMM_s.gmm(states[i], add_offset=True)
        states_cond_pdf = np.zeros((states_cond.shape[0], self.k_c))
        for i in range(states_cond.shape[0]):
            states_cond_pdf[i] = self.GMM_c.gmm(states_cond[i], add_offset=True)
            density *= (np.sum(states_cond_pdf[i]) / np.sum(states_pdf[i]))

        if (tau is not None) and (density < tau):
            self.dynamic_EM_passive(states, states_pdf, states_cond_pdf, states_cond)

        return density


    def dynamic_EM_passive(self, state_seq: np.ndarray, pdf_s: np.ndarray, pdf_c: np.ndarray, state_seq_cond = None):
        '''Passive version of the dynamic EM which assumes pdf values have already been calculated.'''
        if state_seq_cond is None:
            state_seq_cond = self._concatenate_states(state_seq)

        # the inputs are the probability density values, which thus need to be normalized
        resp_s = pdf_s / np.sum(pdf_s, axis=1, keepdims=True)
        resp_c = pdf_c / np.sum(pdf_c, axis=1, keepdims=True)

        for (state_s, r_s, state_c, r_c) in zip(state_seq, resp_s, state_seq_cond, resp_c):
            self.GMM_s.update(state_s, r_s)
            self.GMM_c.update(state_c, r_c)


    def save(self, filepath: str):
        filepath = filepath.split('.json')[0]
        self.GMM_s.save(filepath + '_s')
        self.GMM_c.save(filepath + '_c')


    def load(self, filepath: str):
        self.GMM_s.load(filepath + '_s_config')
        self.GMM_c.load(filepath + '_c_config')


    def cover_state_sequence(self, n: int, state_sequence: np.ndarray, state_sequence_conc = None):
        if state_sequence_conc is None:
            state_sequence_conc = self._concatenate_states(state_sequence)

        coverages = [self.sequence_freshness(state_sequence, state_sequence_conc, tau=None)]
        for _ in tqdm.tqdm(range(n)):
            coverages.append(self.sequence_freshness(state_sequence, state_sequence_conc, tau=None))
        return coverages


def test_online_gmm(**kwargs):
    '''More or less the same as the unit tests.'''
    test_rng = np.random.default_rng(0) # type: np.random.Generator
    k = 2
    dim = 2
    num_iterations = kwargs.get('num_iterations', 10)
    batch_size = kwargs.get('batch_size', 50)
    gamma = kwargs.get('gamma', 0.05)
    cluster_means = [
        [1, 1],
        [4, 4],
    ]
    initial_data, fig, ax = generate_clustered_data(dim, k, cluster_means, num_points_per_cluster=1000, plot=True, spread_factor=0.05, rng=test_rng)
    shuffle_data = test_rng.permutation(initial_data)
    gmm = GMM(0, 2)

    init_means = kwargs.get('init_means', None)
    # random initialization
    if init_means is not None:
        gmm_init_data = copy.deepcopy(init_means)
    else:
        gmm_init_data = shuffle_data[:k]

    gmm.initialize(gmm_init_data)
    gmm.set_gamma(gamma)

    plot_gaussians(gmm.means, gmm.covariances, ax, cmap_values=[0.42, 0.69])
    ax.set_xlim((-3.0419119305947655, 5.62625913293577))
    ax.set_ylim((-2.961636908743682, 5.903138354379044))

    i = 0
    ax.set_title('Iteration: {}'.format(i))
    fig.tight_layout()

    ll = [gmm.log_likelihood(shuffle_data)]
    for _ in tqdm.tqdm(range(num_iterations)):
        samples = shuffle_data[test_rng.choice(len(shuffle_data), size=batch_size)]
        gmm.online_EM(samples)
        ll.append(gmm.log_likelihood(shuffle_data))

        remove_gaussian(ax)
        ax.set_title('Iteration: {}'.format(i))
        plot_gaussians(gmm.means, gmm.covariances, ax, cmap_values=[0.42, 0.69])
        fig.savefig('imgs/iteration_{:02d}.png'.format(i))
        i += 1

    create_gif('imgs', 'test_online_gmm.gif', duration=400, prefix='iteration')
    fig, ax = plt.subplots()
    color = 'blue'
    plot_points(ax, ll, color=color, step=5)
    ax.set_title('log likelihood')
    fig.savefig('test_online_gmm.png')
    gmm.save('test_online_gmm_save')
    final_ll = gmm.log_likelihood(shuffle_data)
    gmm2 = GMM(0, 2)
    gmm2.load('test_online_gmm_save_config')
    print('Does save/load work:', np.array_equal(final_ll, gmm2.log_likelihood(shuffle_data)))


def test_gmm_4D():
    test_rng = np.random.default_rng(0) # type: np.random.Generator
    k = 2
    dim = 2
    # fails with gamma = 0.01
    gamma = 0.05
    num_iterations = 75
    batch_size = 50
    cluster_means = [
        [1, 1],
        [4, 4],
    ]
    initial_data, fig, ax = generate_clustered_data(dim, k, cluster_means, num_points_per_cluster=1000, plot=True, spread_factor=0.05, rng=test_rng)
    shuffle_data = test_rng.permutation(initial_data)

    def concatenate_data(data: np.ndarray) -> np.ndarray:
        data_concat = []
        for i in range(len(data) - 1):
            data_concat.append(np.hstack([data[i], data[i+1]]))
        return np.array(data_concat)

    concat_data = concatenate_data(shuffle_data)

    centers = [[1, 1], [4, 4], [4, 1], [1, 4]]

    gmm = GMM(0, 4)
    gmm.set_gamma(gamma)
    gmm.initialize(concat_data[:4])
    ll = [gmm.log_likelihood(concat_data)]
    i = 0
    for _ in tqdm.tqdm(range(num_iterations)):
        samples = concat_data[test_rng.choice(len(concat_data), size=batch_size)]
        gmm.online_EM(samples)
        ll.append(gmm.log_likelihood(concat_data))
        i += 1

    fig, ax = plt.subplots()
    color = 'blue'
    plot_points(ax, ll, color=color, step=5, markersize=5)
    ax.set_title('Evolution of the log likelihood of the model for the training data')
    fig.savefig('test_gmm_4D.png')
    print('K means oracles found:')
    print(centers)
    print('Model found:')
    print(gmm.means)

if __name__ == '__main__':
    main_to_test_gmm = True

    if main_to_test_gmm:
        # difficult initial means
        init_means = np.array([
            [2.48980347, 2.65893762],
            [2.81867544, 2.60949529]
            ])
        # means when testing CoverageModel
        init_means = np.array([[0.35209585, 1.6631099 ], [1.71380809, 0.44130742]])
        test_online_gmm(init_means=init_means)
        # test_gmm_4D()
        exit(10)

    test_rng = np.random.default_rng(0) # type: np.random.Generator
    k = 2
    dim = 2
    gamma = 0.05
    num_iterations = 100
    batch_size = 40
    cluster_means = [
        [1, 1],
        [4, 4],
    ]
    initial_data, fig, ax = generate_clustered_data(dim, k, cluster_means, num_points_per_cluster=1000, plot=True, spread_factor=0.05, rng=test_rng)
    shuffle_data = test_rng.permutation(initial_data)
    def concatenate_data(data: np.ndarray) -> np.ndarray:
        data_concat = []
        for i in range(len(data) - 1):
            data_concat.append(np.hstack([data[i], data[i+1]]))
        return np.array(data_concat)

    concat_data = concatenate_data(shuffle_data)

    cov_model = CoverageModel(0, k, gamma)
    cov_model.initialize(shuffle_data)
    print('------ INIT MEANS ------')
    print(cov_model.GMM_s.means)
    print('--------------------')

    ll_s = [cov_model.GMM_s.log_likelihood(shuffle_data)]
    ll_c = [cov_model.GMM_c.log_likelihood(concat_data)]
    densities = []

    for _ in tqdm.tqdm(range(num_iterations)):
        indices = test_rng.choice(len(concat_data), size=batch_size)
        samples = shuffle_data[indices]
        samples_c = concat_data[indices]

        densities.append(cov_model.sequence_freshness(samples, samples_c, tau=10.0))
        ll_s.append(cov_model.GMM_s.log_likelihood(shuffle_data))
        ll_c.append(cov_model.GMM_c.log_likelihood(concat_data))


    # print('densities:', densities)

    color = 'blue'
    fig, ax = plt.subplots()
    plot_points(ax, ll_s, color=color, step=5, markersize=6)
    ax.set_title('LL of GMM_s')
    fig.savefig('test_gmm_s.png')

    fig, ax = plt.subplots()
    plot_points(ax, ll_c, color=color, step=5, markersize=6)
    ax.set_title('LL of GMM_c')
    fig.savefig('test_gmm_c.png')

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(densities)), densities)
    ax.set_title('Densities')
    fig.savefig('densities.png')

    print('----- Final Means -------')
    print(cov_model.GMM_s.means)
    print('-------------------')
    print(cov_model.GMM_c.means)

    cov_model.save('cov_model')
    # cov_model.load('cov_model')
    # eval_densities = []
    # n_evals = 1000
    # m_eval = 10
    # for _ in tqdm.tqdm(range(n_evals)):
    #     indices = test_rng.choice(len(concat_data), size=m_eval)
    #     samples = shuffle_data[indices]
    #     samples_c = concat_data[indices]
    #     eval_densities.append(cov_model.sequence_freshness(samples, samples_c, tau=None))
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(len(eval_densities)), eval_densities)
    # ax.set_title(f'Densities of state sequences of length {m_eval}')
    # fig.savefig(f'eval_densities_load_{m_eval}.png')