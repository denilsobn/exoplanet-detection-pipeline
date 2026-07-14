import numpy as np
from scipy.signal import savgol_filter

def extract_lightcurve(tpf, mask_type='pipeline', threshold=3):
    """
    Extrai a curva de luz de um Target Pixel File (TPF) do TESS ou Kepler
    
    Parametros:
    tpf : lightkurve.TessTargetPixelFile
        O arquivo de pixels do alvo
    mask_type : str
        Define como os pixels serao selecionados. 
        Opções: 'pipeline' (padrao da missão), 'threshold' (acima do ruido de fundo) ou 'all'
    threshold : int
        O valor limite em sigmas usado se mask_type='threshold'
        
    Retorno:
    --------
    lc : lightkurve.LightCurve
        Curva de luz bruta extraida
    """
    if mask_type == 'pipeline':
        aperture_mask = tpf.pipeline_mask
    elif mask_type == 'threshold':
        aperture_mask = tpf.create_threshold_mask(threshold=threshold)
    elif mask_type == 'all':
        aperture_mask = 'all'
    else:
        raise ValueError("mask_type deve ser 'pipeline', 'threshold' ou 'all'.")
    
    lc = tpf.to_lightcurve(aperture_mask=aperture_mask)
  
    lc = lc.remove_nans()
    
    return lc


def clean_lightcurve(lc, sigma=5.0, niters=3):
    """
    Remove outliers e raios cosmicos da curva de luz usando sigma-clipping
    
    Parametros:
    lc : lightkurve.LightCurve
        A curva de luz original
    sigma : float
        Quantos desvios padroes um ponto precisa estar fora da media para ser cortado
    niters : int
        Numero de iteracoes do corte
    
    Retorno:
    clean_lc : lightkurve.LightCurve
        Curva de luz sem spikes de ruido
    """
    clean_lc = lc.remove_outliers(sigma=sigma, sigma_upper=sigma, sigma_lower=sigma)
    return clean_lc


def normalize_and_flatten(lc, window_length=401):
    """
    Remove ruidos de longo prazo como rotação estelar e escalona o fluxo
    Gera uma curva de luz adimensional focada nos transitos rápidos
    
    Parametros:
    lc : lightkurve.LightCurve
        A curva limpa
    window_length : int
        Tamanho da janela usada no filtro de achatamento, numero impar
        
    Retorno:
    flat_lc : lightkurve.LightCurve
        Curva de luz achatada e normalizada em torno de 1.0
    """

    if window_length % 2 == 0:
        window_length += 1
        
    flat_lc = lc.flatten(window_length=window_length)
    return flat_lc


def smooth_flux(flux, window_length=51, polyorder=2):
    """
    Aplica um filtro Savitzky-Golay para suavizar variações de alta frequencia
    
    Parametros:
    flux : array-like
        O array de fluxo da curva de luz.
    window_length : int
        Tamanho da janela do filtro, deve ser impar
    polyorder : int
        Ordem do polinomio ajustado aos dados
        
    Retorno:
    --------
    flux_smooth : np.array
        O array de fluxo suavizado
    """
    if window_length % 2 == 0:
        window_length += 1
        
    if window_length <= polyorder:
        raise ValueError("window_length deve ser estritamente maior que polyorder.")
        
    flux_smooth = savgol_filter(flux, window_length=window_length, polyorder=polyorder)
    return flux_smooth