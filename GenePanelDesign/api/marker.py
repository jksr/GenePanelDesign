

import numpy as np
import pandas as pd
from enum import Enum
from .FastMarkerCaller.FastMarkerCaller.group_obs_mean import group_obs_mean
from .FastMarkerCaller.FastMarkerCaller.markercaller import MarkerCaller
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mutual_info_score
from tqdm import tqdm



class RankMarkers:
    def __call__(self, markerdf:pd.DataFrame) -> pd.DataFrame:
        pass
    
class RankMarkersWelchT(RankMarkers):
    def __init__(self):
        super().__init__()
    def __call__(self, markerdf:pd.DataFrame) -> pd.DataFrame:
        markerdf['score'] = markerdf['welch-t']
        return markerdf.sort_values('score', ascending=False)
    
class RankMarkersFcDiff(RankMarkers):
    def __init__(self):
        super().__init__()
    def __call__(self, markerdf:pd.DataFrame) -> pd.DataFrame:
        markerdf['score'] = markerdf['fc']*markerdf['diff']
        return markerdf.sort_values('score', ascending=False)
        
class RankGenesMethod(str,Enum):
    RANK_WITH_WELCHT = 'welch-t'
    RANK_WITH_FCDIF = 'fc-diff'
    
class InitGenesMethod(str,Enum):
    INIT_WITH_MI = 'mi'
    INIT_WITH_OVR = 'ovr'

class ConditionMarkerSelector:
    def __init__(self, adata, cond_col, name, 
                 dist_sample_cells_per_group = 500,
                 dist_n_samples = 10,
                ):
        if isinstance(cond_col, pd.Series):
            self.adata = adata[adata.obs_names.isin(cond_col.index)].copy()
        elif len(cond_col)!=len(adata):
            raise ValueError
        else:
            self.adata = adata.copy()
            cond_col = pd.Series(cand_col, index=adata.obs_names)
                
        self.adata.obs[name] = cond_col
        self.name = name
        
        self._init_designer()
    
    
    def _init_designer(self):
        ad = self.adata[self.adata.obs[self.name].dropna().index]
        self.marker_caller = MarkerCaller(ad, self.name)
        self._marker_cache = {}

        self._group_obs_mean_cache = []
        full_dists, _ = self._calc_pdists()
        self.obs_dists = full_dists
        
    
    def _call_markers(self, group1, group2, **kwargs):
        if (group1,group2) not in self._marker_cache:
            self._marker_cache[group1, group2] = self.marker_caller.call_markers(group1, group2, auroc=False, **kwargs)
        return self._marker_cache[group1, group2]
        
    def _get_seed_features_ovr(self):
        marker_df = self.marker_caller.call_one_vs_rest(auroc=False, topn=1)
        marker_df['Reason'] = f'{self.name} :: init-ovr'
        return marker_df

    def _get_seed_features_mi(self):
        marker_df = []
        for ct in self.adata.obs[self.name].unique():
            is_ct = self.adata.obs[self.name]==ct
            mis = [mutual_info_score(is_ct, self.adata.obs_vector(var)>0) for var in self.adata.var_names]
            marker_df.append(self.adata.var_names[np.argmax(mis)], ct, '_REST_', f'{self.name} :: init-mi')
        return marker_df

#     def _get_seed_features(self):
#         return self._get_seed_features_ovr()


#     def _has_enough_markers(self, dists):
#         if len(self.panel.index.unique())>=self.max_marker or\
#                 dists.min()>=self.stop_dist:
#             return True
#         return False

#     def _calc_pdists0(self, var_names=None, sample_ncells=500, nsamples=10):
#         cells = self.adata.obs[[self.name]].dropna()
#         if var_names is None:
#             var_names = self.adata.var_names
            
#         ntodo = nsamples - len(self._group_obs_mean_cache)
#         ntodo = max(0,ntodo)

#         _group_obs_mean_new = []
#         for _ in range(ntodo):
#             sample_cells = cells.groupby(self.name, group_keys=False).apply(
#                                 lambda x: x.sample(sample_ncells) if len(x)>sample_ncells else x).index
#             _adata = self.adata[sample_cells]
#             _group_obs_mean_new.append(group_obs_mean(_adata, self.name).T)
#         self._group_obs_mean_cache.extend(_group_obs_mean_new)
        
#         dists = []
#         for i in range(nsamples):
#             means = self._group_obs_mean_cache[i][var_names]
#             dists.append(pdist(means))
#         dists = sum(dists)/nsamples
#         return dists, means.index.tolist()
    
    
    def _calc_pdists(self, var_names=None, sample_ncells=500, nsamples=10):
        cells = self.adata.obs[[self.name]].dropna()
        if var_names is None:
            var_names = self.adata.var_names
            
        ntodo = nsamples - len(self._group_obs_mean_cache)
        ntodo = max(0,ntodo)

        _group_obs_mean_new = []
        for _ in range(ntodo):
            sample_cells = cells.groupby(self.name, group_keys=False, sort=True).apply(
                                lambda x: x.sample(sample_ncells) if len(x)>sample_ncells else x).index
            _adata = self.adata[sample_cells]
            means = group_obs_mean(_adata, self.name).T
            single_dists = [pdist(means[[col]], metric='sqeuclidean') for col in means.columns]
            single_dists = pd.DataFrame(single_dists, index=means.columns).T
            _group_obs_mean_new.append(single_dists)
        self._group_obs_mean_cache.extend(_group_obs_mean_new)
        
        dists = []
        
        for i in range(nsamples):
            pdists = np.sqrt(self._group_obs_mean_cache[i][var_names].sum(1))
            dists.append(pdists)
        dists = sum(dists)/nsamples
        return dists, self.adata.obs[self.name].dropna().sort_values().unique()

    @staticmethod
    def _has_enough_markers(panel, cur_dists, obs_dists, dist_ratio=0.1, 
                            min_genes=500, max_genes=500, ):
        genes = panel.index.unique()
        if len(genes)>=max_genes:
            return True, '>= max-genes', 1
        
        progress_dist = min(cur_dists.min()/np.median(obs_dists)/dist_ratio, 1)
        progress_num = min(len(genes)/min_genes, 1)
        progress = progress_dist*progress_num
        
        if np.isclose(progress_dist, 1):
            if np.isclose(progress_num, 1):
                return True, 'has-enough', progress
            else:
                return False, '< min-genes', progress
        else:
            return False, '< dist-cutoff', progress

    @staticmethod
    def _n_pairs_to_add(progress, panel, min_genes=500):
        progress_cutoff = 0.85
        remaining_cutoff = 20
        big_step = 5
        small_step = 1
        
        if progress < progress_cutoff:
            if min_genes - len(panel.index.unique()) > remaining_cutoff:
                return big_step
            else:
                return small_step
        else:
            return small_step
        
    
    def make_panel(self, panel=None,
                   genes_to_include=None, 
                   min_genes=100,
                   max_genes=500,
                   dist_ratio=0.1,
                   fdr=0.05, 
                   init_with = InitGenesMethod.INIT_WITH_OVR,
                   rank_with = RankGenesMethod.RANK_WITH_WELCHT,
                  ):

        if panel is None:
            if genes_to_include is None:
                if init_with==InitGenesMethod.INIT_WITH_OVR:
                    panel = self._get_seed_features_ovr()
                elif init_with==InitGenesMethod.INIT_WITH_MI:
                    panel = self._get_seed_features_mi()
                else:
                    raise ValueError()
            else:
                panel = pd.DataFrame([['_Predefine_', 'User', 0]], 
                                     index = genes_to_include, 
                                     columns=['Name','Reason','PrevCount'])
                panel.index.name='gene'
            panel['PrevCount'] = 0
        else:
            if genes_to_include is not None:
                to_add = pd.DataFrame([['_Predefine_', 'User', len(panel.index.unique())]], 
                                     index = genes_to_include, 
                                     columns=['Name','Reason','PrevCount'])
                to_add.index.name='gene'
                panel = panel.append(to_add)
                
        if rank_with == RankGenesMethod.RANK_WITH_WELCHT:
            ranker = RankMarkersWelchT()
        elif rank_with == RankGenesMethod.RANK_WITH_FCDIF:
            ranker = RankMarkersFcDiff()
        else:
            raise ValueError(rank_with)
            

        with tqdm(total=1.0, desc=f'For the condition {self.name}') as pbar:
#             pbar.write(f'Analyze genes for the condition {self.name}')
            prev_progress = 0
            while True:
                dists, groups = self._calc_pdists(var_names=panel.index.unique())

                has_enough, reason, progress = self._has_enough_markers(panel, dists, self.obs_dists, 
                                                                        dist_ratio, min_genes, max_genes)
                pbar.update(progress-prev_progress)
                prev_progress = progress
                
                if has_enough:
                    break

                n_pairs = self._n_pairs_to_add(progress, panel, min_genes)

                dist_cands = pd.DataFrame(squareform(dists)).melt(ignore_index=False).reset_index()
                dist_cands = dist_cands[dist_cands['index']<dist_cands['variable']].sort_values('value')

                prev_count = len(panel.index.unique())
                marker_dfs = []
                for c,(_,(i,j,*_)) in enumerate(dist_cands[['index','variable']].iterrows()):
                    if c >= n_pairs:
                        break

                    tmp1 = ranker(self._call_markers(groups[i], groups[j], fdr=fdr))#sorting 
                    tmp1 = tmp1[~tmp1.index.isin(panel.index)].head(1).copy()
                    tmp2 = ranker(self._call_markers(groups[j], groups[i], fdr=fdr))
                    tmp2 = tmp2[~tmp2.index.isin(panel.index)].head(1).copy()
                    tmp1['Name'] = self.name
                    tmp2['Name'] = self.name
                    tmp1['Reason'] = reason
                    tmp2['Reason'] = reason
                    tmp1['PrevCount'] = prev_count
                    tmp2['PrevCount'] = prev_count
                    marker_dfs.extend([tmp1,tmp2])
                panel = pd.concat([panel]+marker_dfs)
        return panel    
