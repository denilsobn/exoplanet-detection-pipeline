import argparse
import numpy as np
import pandas as pd
import os

from src.exoplanet_analysis import data_loader, processing, analysis, plotting

def run_target_pipeline(target_id, sector=None):
    """
    Executa a pipeline completa para um unico alvo.
    """
    print(f"\n --- Iniciando Análise para {target_id} --- ")
    
    # O Lightkurve vai procurar os dados, se for um arquivo local 
    # voce tambem pode usar data_loader.load_tpf_from_disk(target_id)
    tpf = data_loader.download_tess_tpf(target_id, sector=sector)
    
    if isinstance(tpf, list) or type(tpf).__name__ == 'TargetPixelFileCollection':
        tpf = tpf[0] 
        
    lc = processing.extract_lightcurve(tpf, mask_type='pipeline')
    clean_lc = processing.clean_lightcurve(lc, sigma=5.0)
    flat_lc = processing.normalize_and_flatten(clean_lc)
    
    print("Calculando Periodograma BLS...")
    best_period, t0, periodogram = analysis.calculate_bls(flat_lc)
    print(f"Período detectado: {best_period:.4f} dias.")
    
    folded_lc = flat_lc.fold(period=best_period, epoch_time=t0)
    
    print("Iniciando MCMC (Emcee + Batman)")
    r_init, a_init = analysis.estimate_initial_params(flat_lc.flux.value, best_period)
    
    samples = analysis.fit_transit(
        time=flat_lc.time.value, 
        flux=flat_lc.flux.value, 
        flux_err=flat_lc.flux_err.value,
        period=best_period, 
        t0=t0, 
        r_init=r_init, 
        a_init=a_init,
        n_steps=2000,
        burn=500
    )
    
    med = np.median(samples, axis=0)
    rp_rs, a_rs, inc = med[1], med[2], med[3]
    print(f"Resultados MCMC: Rp/Rs = {rp_rs:.4f}, a/Rs = {a_rs:.4f}, Inclinação = {inc:.2f} graus")
    
    print("Gerando e salvando gráficos em reports/figures/...")
    target_clean_name = target_id.replace(" ", "_")
    plotting.plot_lightcurve(flat_lc, f"{target_clean_name}_flat_lc.png", title=f"{target_id} - Flattened")
    plotting.plot_bls_periodogram(periodogram, best_period, f"{target_clean_name}_bls.png")
    plotting.plot_folded_transit(folded_lc, f"{target_clean_name}_folded.png")
    plotting.plot_mcmc_posteriors(samples, filename=f"{target_clean_name}_mcmc.png")
    
    return {
        'target': target_id,
        'period': best_period,
        't0': t0,
        'rp/Rs': rp_rs,
        'a/Rs': a_rs,
        'inc': inc
    }

def main():
    # Configuração de argumentos de terminal
    parser = argparse.ArgumentParser(description="Pipeline de Análise de Exoplanetas TESS")
    parser.add_argument("--target", type=str, required=True, help="Nome do alvo")
    parser.add_argument("--sector", type=int, default=None, help="Setor do TESS")
    args = parser.parse_args()

    result = run_target_pipeline(args.target, args.sector)
    
    catalog_file = "catalogo_exoplanetas.csv"
    df_new = pd.DataFrame([result])
    
    if os.path.exists(catalog_file):
        df_new.to_csv(catalog_file, mode='a', header=False, index=False)
        print(f"Resultado anexado a {catalog_file}")
    else:
        df_new.to_csv(catalog_file, index=False)
        print(f"Novo catálogo criado em {catalog_file}")

if __name__ == "__main__":
    main()