import multiprocessing
from collections.abc import Iterable
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
from enum import Enum
from .marker import ConditionMarkerSelector, InitGenesMethod, RankGenesMethod
from .summary import Expression, GenePanelSummerizer



class RunOrder(str,Enum):
    SIML = 'simultaneous'
    SEQL = 'sequential'
    

class GenePanelDesigner:
    def __init__(self, adata, design_df, 
                 design_cols=None,
                 downsample_per_group=None,
                 downsample_max=None,
                 is_count_matrix=False,
                 n_jobs = 1,
                ):
        
        
        self.n_jobs = n_jobs
        self.adata = adata[adata.obs_names.isin(design_df.index)].copy()
        self.adata.obs = design_df.loc[self.adata.obs_names]
        self.names = list(self.adata.obs.columns)
        
        print('processing the data ...')
        self._downsample(downsample_per_group, downsample_max)
        
        if is_count_matrix:
            print('\tInput is count matrix. Normalizing and selecting highly variable genes')
            self._preprocess()
        print('Done.')
    
        print('Initialize gene selectors ...', end=' ')
        self._init_marker_selectors()
        print('Done.')
        
    def _downsample(self, downsample_per_group, downsample_max):
        if isinstance(downsample_per_group, int):
            downsample_per_group = dict(zip(self.names, 
                                            [downsample_per_group]*len(self.names)))
        elif isinstance(downsample_per_group, dict):
            pass
        elif downsample_per_group is None:
            pass
        else:
            raise ValueError()

        cells = pd.Index([])
        for col,downn in downsample_per_group.items():
            downed = self.adata.obs.groupby(col, group_keys=False).apply(
                lambda x: x.sample(downn) if len(x)>downn else x)
            cells = cells.union(downed.index)
        
        if downsample_max is not None:
            cells = np.random.choice(cells, downsample_max)
        
        self.adata = self.adata[self.adata.obs_names.isin(cells)].copy()
        
    def _preprocess(self, min_cells_per_gene=10, min_genes_per_cell=10, 
                    n_top_highly_variable_genes=10000):
        import scanpy as sc
        sc.pp.filter_genes(self.adata, min_cells=min_cells_per_gene)
        sc.pp.filter_cells(self.adata, min_genes=min_genes_per_cell)
        sc.pp.normalize_total(self.adata, target_sum=1e6)
        sc.pp.log1p(self.adata)
        
        sc.pp.highly_variable_genes(self.adata, n_bins=50)
        sc.pp.highly_variable_genes(self.adata, max_mean=self.adata.var['means'].max(), 
                                    n_bins=50, n_top_genes=n_top_highly_variable_genes)
        self.adata = self.adata[:, self.adata.var['highly_variable']].copy()

    def _init_single_marker_selectors(self, col):
        return ConditionMarkerSelector(self.adata, self.adata.obs[col], col)
    
    def _init_marker_selectors(self):
        if self.n_jobs>1:
            with multiprocessing.Pool(self.n_jobs) as pool:
                selectors = pool.map(self._init_single_marker_selectors, self.names)
        else:
            selectors = list(map(self._init_single_marker_selectors, self.names))
        self.selectors = dict(zip(self.names, selectors))
        
    def make_panels(self, order=RunOrder.SIML, 
                    genes_to_include=None, 
                    min_genes=200, max_genes=200, 
                    dist_ratio=0.1, fdr=0.05,
                    init_panel_with=InitGenesMethod.INIT_WITH_OVR,
                    rank_genes_with=RankGenesMethod.RANK_WITH_WELCHT,
                   ):

        def process_param(names, param):
            if isinstance(param, dict):
                if len(set(names)-set(param.keys()))>0:
                    raise ValueError()
                return param
            elif isinstance(param, Iterable) and not isinstance(param, str):
                if len(param)<len(names):
                    raise ValueError()
                return dict(zip(names, param))
            else:
                return dict(zip(names, [param]*len(names)))

        def schedule_runs(names, order):
            if order==RunOrder.SIML:
                order = dict(zip(names, [0]*len(names)))
            elif order==RunOrder.SEQL:
                order = dict(zip(names, range(len(names))))
            elif isinstance(order, dict):
                pass
            elif isinstance(order, Iterable):
                order = dict(zip(names, order))
            else:
                raise ValueError()
            rorder = dict()
            for col,rank in order.items():
                if rank not in rorder:
                    rorder[rank] = []
                rorder[rank].append(col)
            schedule = [rorder[k] for k in sorted(rorder.keys())]

            return schedule
        
        def draw_schedule(schedule, genes_to_include, min_genes, max_genes=None, 
                          dist_ratio=None, fdr=None, nh=6, nv=2):
            
            print('Genes will be selected in the order of \n')
            print('V')
            if genes_to_include is not None:
                print('|\n'*nv, end='')
                print('+'+'-'*nh, f'user-specified: {len(genes_to_include)}' )
            for step in schedule:
                print('|\n'*nv, end='')
                for name in step:
                    print('+'+'-'*nh, f'{name} : {min_genes[name]}')
            print('|\n'*nv, end='')
            print('V\n')
            

            
        min_genes = process_param(self.names, min_genes)
        max_genes = process_param(self.names, max_genes)
        dist_ratio = process_param(self.names, dist_ratio)
        fdr = process_param(self.names, fdr)

        schedule = schedule_runs(self.names, order)
        
        draw_schedule(schedule, genes_to_include, min_genes)

        print('Analyzing...')
        
        panels = {}
        for i,step in enumerate(schedule):
            if i>0:
                genes_to_include = None
                tot_panel = pd.concat(panels.values())
            else:
                tot_panel = pd.DataFrame()
            for name in step:
                panels[name] = self.selectors[name].make_panel(tot_panel, genes_to_include, 
                                                               min_genes[name]+len(tot_panel), 
                                                               max_genes[name]+len(tot_panel), 
                                                               dist_ratio=dist_ratio[name],
                                                               fdr=fdr[name],
                                                               init_with=init_panel_with,
                                                               rank_with=rank_genes_with,
                                                              )
        panel = pd.concat(panels.values()).drop_duplicates()
        panels = {name:subpanel for name,subpanel in panel.groupby('Name')}
        return panels
    
    
    def finalize_panels(self, panels, final_genes=500, weights=None):
        if isinstance(weights, dict):
            extra = list(set(panels.keys())-set(weights.keys()))
            if len(extra)>1 or extra[0]!='_Predefine_':
                raise ValueError()
        elif isinstance(weights, Iterable):
            weights = dict(zip(self.names, weights))
        elif weights is None:
            weights = dict(zip(self.names, [1]*len(self.names)))
        else:
            raise ValueError()
        
        def _union_index(dfs):
            return  pd.concat(dfs).index.unique()
        def _sub_panel(dfs, last_pos):
            return [ df.iloc[:pos] for df,pos in zip(dfs, last_pos)]
        def get_dists(dist_cache, selector, pos_arr, genes):
            key = (tuple(pos_arr), selector.name)
            if key not in dist_cache:
                dist_cache[key] = selector._calc_pdists(var_names=genes)[0]
            return dist_cache[key]

        panels_ = {name:panel for name,panel in panels.items() if name in self.names}
        fixed_panels = [panel for name,panel in panels.items() if name not in self.names]

        names = list(panels_.keys())
        panels_= list(panels_.values())
        selectors = [self.selectors[name] for name in names]
        weights = [weights[name] for name in names]
        dist_cache = {}


        last_pos = np.array([len(panel) for panel in panels_])
        genes = _union_index(_sub_panel(panels_, last_pos)+fixed_panels)
        n_prev = len(genes)-final_genes
        print(f'{len(genes)} gene candidates -> {final_genes} final genes')
        with tqdm(total=n_prev, desc='Remove less informative genes ') as pbar:
            while True:
                genes = _union_index(_sub_panel(panels_, last_pos)+fixed_panels)
                if len(genes)<= final_genes:
                    break

                distdiffs = []
                for i in range(len(panels_)):
                    if last_pos[i]<=0:
                        distdiffs.append(np.inf)
                        continue

                    test_last_pos = np.array(last_pos)
                    test_last_pos[i] -= 1
                    test_genes = _union_index(_sub_panel(panels_, test_last_pos)+fixed_panels)

                    distdiff = []
                    for selector,w in zip(selectors, weights):
                        ori_dists = get_dists(dist_cache, selector, last_pos, genes)[0]
                        test_dists = get_dists(dist_cache, selector, test_last_pos, test_genes)[0]
                        distdiff.append((ori_dists-test_dists).max()*w)
                    distdiffs.append( sum(distdiff) )
                sel = np.argmin(distdiffs)

                last_pos[sel] -= 1

                genes = _union_index(_sub_panel(panels_, last_pos)+fixed_panels)

                n_curr = len(genes)-final_genes
                pbar.update(n_prev-n_curr)
                n_prev = n_curr

        final = [ panels[name][panels[name].index.isin(genes)] for name in panels ]
        return pd.concat(final)
    
    def summarize(self, panel, show_plot=False, to_html=False):
        #TODO
        summer = GenePanelSummerizer
        gxc, fig_sum = summer.summarize_gene_by_condition(panel, plot=True, show_plot=show_plot)
        genes_p = list(panel.index.unique())
        genes_n = list(set(self.adata.var.index)-set(genes_p))
        
        for name in gxc:
            for kind in Expression:
                df_p, fig_p = summer.summarize_expression(self.adata, self.adata.obs[name], 
                                                          genes_p, kind=kind, plot=True,
                                                          show_plot=show_plot)
                df_n, fig_n = summer.summarize_expression(self.adata, self.adata.obs[name], 
                                                          genes_n, kind=kind, plot=True,
                                                          show_plot=show_plot)
            fulldist, dist, ratio, fig_dist = summer.summarize_dist_matrix(self.selectors[name], 
                                                                           genes_p, plot=True, 
                                                                           show_plot=show_plot)
        summer.check_clusters(self.adata,  self.adata.obs, panel, plot=True, show_plot=show_plot)

