import numpy as np

from typing import List, Tuple, Union
from sklearn.neighbors import NearestNeighbors


class NoveltyArchive:
    def __init__(self, individuals: np.ndarray, k: int, threshold: float) -> None:
        # number of neighbors to compute novelty scores (i.e., average distance of the k-nearest neighbors)
        self.k = k
        # novelty threshold (to update the archive)
        self.t = threshold
        # archive as a list of list (of integers or floats)
        self.points: List[List] = [ind.tolist() for ind in individuals]
        self.updates = []


    def score(self, data: np.ndarray, pop_data: np.ndarray = None) -> Union[float, np.ndarray]:
        '''Scores @data w.r.t the archive no @pop_data is provided; otherwise the evaluation considers the joined set of @pop_data and the archive.'''
        # archive
        if pop_data is None:
            ref = np.array(self.points)
        else:
            ref = np.vstack([np.array(self.points), pop_data])

        # scores multiple points at once
        if len(data.shape) == 2:
            scores = []
            for x in data:
                dists = np.linalg.norm(x - ref, axis=1)
                novelty_score: float = np.mean(np.sort(dists)[:self.k])
                scores.append(novelty_score)
            return np.array(scores)
        # single score
        else:
            return np.mean(np.sort(np.linalg.norm(data - ref, axis=1))[:self.k])


    def update(self, candidates: np.ndarray) -> Tuple[bool, List[int]]:
        '''
        Sequentially attempts to add in the archive the candidates.
        It returns whether it has been updated and the list of the indices of the candidates added.
        '''
        updated = False
        indices = []
        for i, x in enumerate(candidates):
            dists = np.linalg.norm(x - np.array(self.points), axis=1)
            # should we check if the candidate is already in the archive (to consider [1:k+1] instead?)
            novelty_score: float = np.mean(np.sort(dists)[:self.k])
            self.updates.append(novelty_score)
            if novelty_score > self.t:
                self.points.append(x.tolist())
                indices.append(i)
                if not updated:
                    updated = True
        return updated, indices


    def update2(self, candidates: np.ndarray) -> Tuple[bool, List[int]]:
        '''
        Second version for updating the archive, by adding only the candidates that improve the density of the archive.
        The archive is improved if the mean of the k-nn distances increases.
        '''
        updated = False
        indices = []

        def preview_sparseness(points: List[np.ndarray]) -> float:
            data = np.unique(np.array(points), axis=1)
            neighbors = NearestNeighbors(n_neighbors=self.k).fit(data)
            distances, _ = neighbors.kneighbors()
            return np.mean(distances)

        for i, x in enumerate(candidates):
                current_sparseness = preview_sparseness(self.points)
                # print(current_density, preview_density(self.points + [x.tolist()]))
                if preview_sparseness(self.points + [x.tolist()]) > current_sparseness:
                    self.points.append(x.tolist())
                    indices.append(i)
                    if not updated:
                        updated = True
        return updated, indices


    def update3(self, candidates: np.ndarray) -> Tuple[bool, List[int]]:
        '''
        Third version for updating the archive, inspired by the QD Unified Framework paper.
        Here, the novelty threshold is used as the minimum distance it the nearest neighbor of the candidate with the archive.
        (In the paper, they further conditionally add the candidate if this distance is too low which is not been implemented here.)
        '''
        updated = False
        indices = []


        for i, x in enumerate(candidates):
            dists = np.linalg.norm(x - np.array(self.points), axis=1)
            nearest_neighbor_dist = np.min(dists)
            self.updates.append(nearest_neighbor_dist)
            if nearest_neighbor_dist > self.t:
                self.points.append(x.tolist())
                indices.append(i)
                if not updated:
                    updated = True
            # else:
            #     print(f'candidate is too closed to its nearest neighbor ({nearest_neighbor_dist:0.3f}.')
        return updated, indices


    def size(self) -> int:
        return len(self.points)


    def archive_sparseness(self) -> float:
        '''Returns the sparseness/density of the archive as the mean novelty score.'''
        data = np.unique(np.array(self.points), axis=1)
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(data)
        distances, _ = neighbors.kneighbors()
        return np.mean(distances)


    def update_attempts(self) -> np.ndarray:
        return np.array(self.updates)