import matplotlib.pyplot as plt
import seaborn as sns
import upsetplot
import scanpy as sc
import pandas as pd
from enum import Enum
from scipy.spatial.distance import squareform
from .FastMarkerCaller.FastMarkerCaller.group_obs_mean import group_obs_mean

class Expression(str, Enum):
    Frac = 'frac'
    ZSCORE = 'z-score'
    EXPR = 'expr'

    
class ExpressionTransform:
    def __call__(self, kind, adata):
        if kind==Expression.Frac:
            return self.to_frac(adata)
        elif kind==Expression.ZSCORE:
            return self.to_z_score(adata)
        elif kind==Expression.EXPR:
            return self.to_expr(adata)
        else:
            raise ValueError()
        
    @staticmethod
    def to_frac(adata):
        adata.X = adata.X>0
        return adata
    @staticmethod
    def to_z_score(adata):
        sc.pp.scale(adata)
        return adata
    
    @staticmethod
    def to_expr(adata):
        return adata
    
    
EXPR_PLOT_PARAMS = {
    Expression.Frac:dict(vmin=0,vmax=1,cmap='Greens'),
    Expression.ZSCORE:dict(vmin=-2, vmax=2, cmap='bwr'),
    Expression.EXPR:dict(cmap='viridis'),
}

    
class GenePanelSummerizer:
    @staticmethod
    def summarize_gene_by_condition(panel, plot=True, show_plot=True):
        panel_ = panel.reset_index()
        panel_['fg>bg'] = panel_['fg']+' > '+panel_['bg']
        gxc = panel_.pivot_table(index='gene', columns='Name', values='fg>bg', aggfunc=sum)
        
        if plot:
            usp = upsetplot.from_contents({col:gxc[col].dropna().index for col in gxc.columns})
            axes = upsetplot.plot(usp)
            if show_plot:
                plt.show()
                return gxc
            else:
                return gxc, plt.gcf()
        return gxc

    @staticmethod
    def condition_dist_matrix(selector, genes=None):
        if genes is not None:
            genes = list(set(genes))
        dist, cols = selector._calc_pdists(genes)
        dist = pd.DataFrame(squareform(dist), index=cols, columns = cols)
        return dist

    @staticmethod
    def summarize_dist_matrix(selector, genes, plot=True, show_plot=True, ):
        panel = GenePanelSummerizer.condition_dist_matrix(selector, list(set(genes)))
        full = GenePanelSummerizer.condition_dist_matrix(selector)
        ratio = (panel/full)#.fillna(1)
        if plot:
            fig, axes = plt.subplots(1,3, figsize=(10,3))
            sns.heatmap(full, ax=axes[0], cmap='viridis')
            sns.heatmap(panel, ax=axes[1], cmap='viridis')
            sns.heatmap(ratio, ax=axes[2], cmap='inferno', vmin=0, vmax=1)
            axes[0].set_title('full')
            axes[1].set_title('panel')
            axes[2].set_title('panel/full')
            plt.tight_layout()
            plt.suptitle(selector.name)
            
            if show_plot:
                plt.show()
                return full, panel, ratio
            else:
                return full, panel, ratio, plt.gcf()
        return full, panel, ratio
    

    @staticmethod
    def summarize_expression(adata, cond, genes, kind=Expression.ZSCORE, plot=True, show_plot=True, ):
        ad = adata[:, list(set(genes))].copy()
        if isinstance(cond, str):
            ad.obs['_COND_'] = ad.obs[cond]
        elif isinstance(cond, Iterable):
            ad.obs['_COND_'] = cond
        else:
            raise ValueError()
            
        trans_func = ExpressionTransform()
        ad = trans_func(kind, ad)
        
        plot_params = EXPR_PLOT_PARAMS[kind]
        
            
        avg = group_obs_mean(ad, '_COND_')

        
        if plot:
            grid = sns.clustermap(avg, **plot_params)
            if show_plot:
                plt.show()
                return avg
            else:
                return avg, plt.gcf()
        else:
            return avg        
        
    @staticmethod
    def check_clusters(adata, design_df,  panel, names=None, max_cells=5000, plot=True, show_plot=True):
        ad = sc.pp.subsample(adata[:, list(panel.index.unique())], n_obs=max_cells, copy=True)
        if names is None:
            names = panel['Name'][panel['Name']!='_Predefine_'].unique()
        elif len(set(names)-set(panel['Name']))>0:
            raise ValueError()
        
        for name in names:
            ad.obs[name] = design_df[name]
            
        sc.tl.pca(ad)
        sc.pp.neighbors(ad)
        sc.tl.umap(ad)
        
        if plot:
            axes = sc.pl.umap(ad, color=names, show=show_plot)
            if show_plot:
                return ad
            else:
                return ad, plt.gcf()
        else:
            return ad
