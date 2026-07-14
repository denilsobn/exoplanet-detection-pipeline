import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

os.makedirs('reports/figures', exist_ok=True)

def plot_lightcurve(lc, filename, title="Curva de Luz"):
    """
    Plota e salva uma curva de luz simples
    """
    ax = lc.scatter(alpha=0.5)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f'reports/figures/{filename}')
    plt.close()


def plot_bls_periodogram(periodogram, best_period, filename='periodogram.png'):
    """
    Plota o periodograma Box Least Squares destacando o melhor período
    """
    ax = periodogram.plot()
    ax.axvline(best_period, color='red', linestyle='--', lw=2, alpha=0.8, label=f'Best Period: {best_period:.4f} d')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'reports/figures/{filename}')
    plt.close()


def plot_folded_transit(folded_lc, filename='folded_transit.png'):
    """
    Plota a curva de luz dobrada no período do planeta
    focando no transito central
    """
    ax = folded_lc.scatter(color='black', alpha=0.5, label='Dados Observados')
    ax.set_title("Transito Dobrado (Phase Folded)")
    plt.tight_layout()
    plt.savefig(f'reports/figures/{filename}')
    plt.close()


def plot_mcmc_posteriors(samples, labels=['t0', 'rp/Rs', 'a/Rs', 'inc'], filename='mcmc_pairplot.png'):
    """
    Cria um pairplot (Matriz de Correlação e Histogramas) usando Seaborn
    para avaliar as distribuicoes a posteriori do MCMC
    """
    df_samples = pd.DataFrame(samples, columns=labels)
    
    sns_plot = sns.pairplot(df_samples, kind='hist', diag_kind='kde', corner=True)
    sns_plot.fig.suptitle("Distribuicoes a Posteriori (MCMC)", y=1.02)
    
    plt.savefig(f'reports/figures/{filename}')
    plt.close()