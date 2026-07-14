import numpy as np
import batman
import emcee

def calculate_bls(flat_lc, min_period=2.0, max_period=15.0, step=0.001):
    """
    Calcula o periodograma Box Least Squares (BLS) para detectar transitos
    
    Parametros:
    flat_lc : lightkurve.LightCurve
        Curva de luz processada e achatada
    min_period, max_period, step : float
        Definem a grade de busca de periodos em dias
        
    Retorno:
    best_period : float
        O periodo orbital mais provavel
    t0 : float
        O tempo central do transito
    periodogram : lightkurve.periodogram.Periodogram
        O objeto periodograma completo para posterior visualizacao
    """
    period_grid = np.arange(min_period, max_period, step)
    periodogram = flat_lc.to_periodogram(method='bls', period=period_grid)
    
    best_period = periodogram.period_at_max_power.value
    t0 = periodogram.transit_time_at_max_power.value
    
    return best_period, t0, periodogram


def estimate_initial_params(flux, best_period):
    """
    Estima os parametros iniciais de r_init e a_init de forma heuristica
    para acelerar a convergencia do MCMC
    """
    depth = 1 - np.min(flux) / np.median(flux)
    
    # Previne que ruidos causem depth negativo e quebrem a raiz quadrada
    depth = max(depth, 1e-5) 
    
    r_init = np.sqrt(depth)
    a_init = (best_period / 365.25)**(2/3) * 215
    
    return r_init, a_init


def fit_transit(time, flux, flux_err, period, t0, r_init, a_init, 
                inc_init=89.0, u_params=[0.3, 0.2], 
                n_walkers=32, n_steps=5000, burn=1000):
    """
    Ajusta modelo de transito estelar com batman e emcee
    
    Parametros:
    time, flux, flux_err : np.array
        Arrays de tempo, fluxo e erro do fluxo da estrela
    period, t0, r_init, a_init, inc_init : float
        Sugestoes iniciais para o ajuste
    u_params : list
        Parametros de escurecimento de limbo quadratico - Limb Darkening
    n_walkers, n_steps, burn : int
        Configurações da simulação MCMC - Cadeias de Markov
        
    Retorno:
    samples : np.ndarray
        Amostras do posterior -> pós-burn-in para [t0, rp, a, inc]
    """
    params = batman.TransitParams()
    params.t0        = t0
    params.per       = period
    params.rp        = r_init
    params.a         = a_init
    params.inc       = inc_init
    params.ecc       = 0.0
    params.w         = 90.0
    params.limb_dark = "quadratic"
    params.u         = u_params

    def log_prior(theta):
        t0_, rp_, a_, inc_ = theta
        # Trava para evitar que o MCMC explore areas da física impossiveis
        if 0 < rp_ < 0.5 and 1 < a_ < 100 and 80 < inc_ < 90:
            return 0.0
        return -np.inf

    def log_likelihood(theta, t, f, ferr):
        t0_, rp_, a_, inc_ = theta
        params.t0, params.rp, params.a, params.inc = t0_, rp_, a_, inc_
        
        model = batman.TransitModel(params, t).light_curve(params)
        
        ferr_safe = np.where(ferr == 0, 1e-5, ferr)
        
        return -0.5 * np.sum(((f - model) / ferr_safe)**2)

    def log_posterior(theta, t, f, ferr):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, t, f, ferr)

    pos = np.array([t0, r_init, a_init, inc_init])
    pos = pos + 1e-4 * np.random.randn(n_walkers, len(pos))

    sampler = emcee.EnsembleSampler(
        n_walkers, len(pos[0]), log_posterior,
        args=(time, flux, flux_err)
    )
    
    # Roda a simulacao Monte Carlo
    sampler.run_mcmc(pos, n_steps, progress=True)

    # Retorna os resultados achatados descartando a fase de queima -> burn-in
    return sampler.get_chain(discard=burn, flat=True)